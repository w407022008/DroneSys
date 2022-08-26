#ifndef LOCAL_PLANNING_H
#define LOCAL_PLANNING_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Time.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/transform_listener.h>
#include <tf/message_filter.h>

#include "drone_msgs/PositionReference.h"
#include "drone_msgs/Message.h"
#include "drone_msgs/DroneState.h"
#include "drone_msgs/RCInput.h"
#include "drone_msgs/ControlCommand.h"

#include "geo_guide_apf.h"
#include "histogram.h"
#include "tools.h"
#include "message_utils.h"

using namespace std;
#define NODE_NAME "Local_Planner [main]"

#define MIN_DIS 0.3

namespace Local_Planning
{

extern ros::Publisher message_pub;

class Local_Planner
{

private:

    ros::NodeHandle local_planner_nh;

    // 参数
    int algorithm_mode;
    int map_input;
    bool is_2D, is_rgbd, is_lidar;
    int yaw_tracking_mode, control_from_joy;
    double _max_goal_range_xy, min_goal_height, _max_goal_range_z, _max_manual_yaw_rate;
    double max_planning_vel;
    double sensor_max_range;
    double forbidden_range;
    double fly_height_2D;
    double safe_distance;

    // 订阅无人机状态、目标点、传感器数据（生成地图）
    ros::Subscriber planner_switch_sub;
    ros::Subscriber goal_sub;
    ros::Subscriber manual_control_sub;
    ros::Subscriber drone_state_sub;

    ros::Subscriber local_point_clound_sub;
    ros::Subscriber swith_sub;
    
    tf::TransformListener tfListener;

    // 发布控制指令
    ros::Publisher command_pub,rviz_guide_pub,point_cloud_pub;
    ros::Timer mainloop_timer,control_timer;

    // 局部避障算法 算子
    local_planning_alg::Ptr local_alg_ptr;

    drone_msgs::DroneState _DroneState;
    nav_msgs::Odometry Drone_odom;
    drone_msgs::ControlCommand Command_Now;  

    double distance_to_goal;

    // 规划器状态
    bool sim_mode;
    bool odom_ready;
    bool drone_ready;
    bool sensor_ready;
    bool goal_ready; 
    bool path_ok;
    bool planner_enable_default;
    bool planner_enable;
    bool vfh_guide_point;

    // 规划初始状态及终端状态
    double drone_yaw_init, user_yaw_init = 0.0;
    double user_yaw;
    Eigen::Vector3d start_pos, start_vel, start_acc, goal_pos, goal_vel;
    float rc_x, rc_y, rc_z, rc_r;

    int planner_state;
    Eigen::Vector3d desired_vel;
    float desired_yaw;

    geometry_msgs::PointStamped  guide_rviz;

    // 打印的提示消息
    string message;

    // 五种状态机
    enum EXEC_STATE
    {
        WAIT_GOAL,
        PLANNING,
        TRACKING,
        LANDING,
    };
    EXEC_STATE exec_state;

	// 点云获取
	bool flag_pcl_ground_removal, flag_pcl_downsampling;
	double max_ground_height, ceil_height, size_of_voxel_grid;
	int timeSteps_fusingSamples;
    sensor_msgs::PointCloud2ConstPtr  local_map_ptr_;
    pcl::PointCloud<pcl::PointXYZ> latest_local_pcl_, local_pcl_tm1, local_pcl_tm2, local_pcl_tm3, concatenate_PointCloud, local_point_cloud;

    void planner_switch_cb(const std_msgs::Bool::ConstPtr& msg);
    void goal_cb(const geometry_msgs::PoseStampedConstPtr& msg);
    void manual_control_cb(const drone_msgs::RCInputConstPtr &msg);
    void drone_state_cb(const drone_msgs::DroneStateConstPtr &msg);
    void localcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
    void Callback_2dlaserscan(const sensor_msgs::LaserScanConstPtr &msg);
    void Callback_3dpointcloud(const sensor_msgs::PointCloud2ConstPtr &msg);
    void mainloop_cb(const ros::TimerEvent& e);
    void control_cb(const ros::TimerEvent& e);

public:

    Local_Planner(void):
        local_planner_nh("~") {}~Local_Planner(){}

    double obs_distance;
    double att_distance;

    Eigen::Matrix3f R_Body_to_ENU;

    void init(ros::NodeHandle& nh);

    //旋转矩阵：机体系到惯性系 R = R_z(psi) * R_y(theta) * R_x(phi)
    Eigen::Matrix3f get_rotation_matrix(float phi, float theta, float psi)
    {
        Eigen::Matrix3f Rota_Mat;

        float r11 = cos(theta)*cos(psi);
        float r12 = - cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi);
        float r13 = sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi);
        float r21 = cos(theta)*sin(psi);
        float r22 = cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi);
        float r23 = - sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
        float r31 = - sin(theta);
        float r32 = sin(phi)*cos(theta);
        float r33 = cos(phi)*cos(theta); 
        Rota_Mat << r11,r12,r13,r21,r22,r23,r31,r32,r33;

        return Rota_Mat;
    }

    // q to (roll,pitch,yaw)  by a 3-2-1 intrinsic Tait-Bryan rotation sequence
    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    // q0 q1 q2 q3
    // w x y z
    Eigen::Vector3d quaternion_to_euler(const Eigen::Quaterniond &q)
    {
        float quat[4];
        quat[0] = q.w();
        quat[1] = q.x();
        quat[2] = q.y();
        quat[3] = q.z();

        Eigen::Vector3d ans;
        ans[0] = atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]));
        ans[1] = asin(2.0 * (quat[2] * quat[0] - quat[3] * quat[1]));
        ans[2] = atan2(2.0 * (quat[3] * quat[0] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]));
        return ans;
    }
    // euler to q  by a 3-2-1 intrinsic Tait-Bryan rotation sequence
    Eigen::Quaternionf quaternion_from_rpy(const Eigen::Vector3d &rpy)
    {
            // YPR - ZYX
            return Eigen::Quaternionf(
                            Eigen::AngleAxisf(rpy.z(), Eigen::Vector3f::UnitZ()) *
                            Eigen::AngleAxisf(rpy.y(), Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(rpy.x(), Eigen::Vector3f::UnitX())
                            );
    }
};



}
#endif 
