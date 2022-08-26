#ifndef HISTO_PLANNING_H
#define HISTO_PLANNING_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
// #include <std_msgs/Float32.h>
#include <std_msgs/Time.h>
#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Imu.h"
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/conversions.h>

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
#include "drone_msgs/Bspline.h"

#include "histogram.h"
#include "bspline_optimizer.h"
#include "sdf_map.h"
#include "planning_visualization.h"
#include "message_utils.h"

using namespace std;
#define NODE_NAME "Histo_Planner [main]"

namespace Histo_Planning
{

class Histo_Planner
{
    // task state machines
    string state_str[5] = { "WAIT_GOAL", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "LANDING" };
    enum EXEC_STATE
    {
        WAIT_GOAL,
        GEN_NEW_TRAJ,
        REPLAN_TRAJ,
        EXEC_TRAJ,
        LANDING,
    };
    EXEC_STATE exec_state;
    
private:
    // Fuction Initialisation
    ros::NodeHandle histo_planner_nh;
    // SDFMap::Ptr sdf_map_;
    histo_planning_alg::Ptr histo_planning_;  
    BsplineOptimizer::Ptr bspline_optimizer_;
    PlanningVisualization::Ptr visualization_;
	
    // [SUB]
    ros::Subscriber swith_sub;
    ros::Subscriber goal_sub;
    ros::Subscriber manual_control_sub;
    ros::Subscriber user_yaw_sub;
    ros::Subscriber drone_state_sub;
    ros::Subscriber local_point_clound_sub;
    
    // [PUB]
    ros::Publisher message_pub;
    ros::Publisher command_pub;//,a_pub,b_pub,c_pub,d_pub;
    ros::Publisher goal_pub;
    ros::Publisher rviz_guide_pub;
    ros::Publisher rviz_closest_pub;
    ros::Publisher rviz_joy_goal_pub;
    image_transport::Publisher obs_img_pub, his_img_pub;
    
    // [Timer loop]
    ros::Timer mission_loop;
    ros::Timer control_loop;
    ros::Timer joy_loop;

    // callback function
    void switchCallback(const std_msgs::Bool::ConstPtr &msg);
    void user_yaw_cb(const sensor_msgs::ImuConstPtr& msg);
    void goal_cb(const geometry_msgs::PoseStampedConstPtr& msg);
    void manual_control_cb(const drone_msgs::RCInputConstPtr &msg);
    void drone_state_cb(const drone_msgs::DroneStateConstPtr &msg);
    void localcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
    
    // loop function
    void joy_cb(const ros::TimerEvent& e);
    void mission_cb(const ros::TimerEvent& e);
    void control_cb(const ros::TimerEvent& e);

    /* calculating time using */
    double time_planning_ = 0.0;
    double time_optimize_ = 0.0;
	
    
    /* ---------- helper function ---------- */
    void changeExecState(EXEC_STATE new_state, string pos_call)
    {
        int pre_s = int(exec_state);
        exec_state = new_state;
        pub_message(message_pub, drone_msgs::Message::NORMAL,  NODE_NAME, "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)]);
    }
    
    double traj_time_after_(double v)
    {
        return t_start_+min(traj_duration_,max(0.0,(ros::Time::now()-time_traj_start).toSec())+v);
    }
	
    // Rotation Matrix: in Body frame to in ENU frame R = R_z(psi) * R_y(theta) * R_x(phi)
    Eigen::Matrix3f R_Body_to_ENU;
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
public:
    Histo_Planner(void):
        histo_planner_nh("~") {}~Histo_Planner(){}

	/* Parameter Setting */
	double geo_fence_x_min,geo_fence_x_max,geo_fence_y_min,geo_fence_y_max,geo_fence_z_min,geo_fence_z_max;
    double min_vel_default;
    
    double drone_yaw_init, user_yaw_init = 0.0;
    double user_yaw;
    bool sim_mode;
    bool tracking_controller_enable;
    bool planner_enable;
    bool CNNLogEnable;
    bool is_2D;
    double fly_height_2D;
    int control_from_joy;
    double _max_goal_range_xy, _max_goal_range_z;
    int yaw_tracking_mode;
    bool spinning_once_first;
    double yaw_rate, time_forward_facing_toward;
    double delta_yaw, yaw_tracking_err_max;
    
    int goal_regenerate_mode;
    double min_goal_height;
    double sensor_max_range, forbidden_range, safe_distance, forbidden_plus_safe_distance;
    double range_near_start, range_near_end;
    double time_interval, time_to_replan;
    //Input point cloud processing
    int map_input;
    double max_ground_height, ceil_height;
    
	
	/* Flight Status */
    // Planner Status
    bool odom_ready;
    bool drone_ready;
    bool map_ready;
    bool goal_ready;
    bool raw_goal_ready; 
    bool path_ok;
    bool is_generating;
	int planner_state;
    bool escape_mode;

    // Flight status
    ros::Time last_mission_exec_time;
    drone_msgs::DroneState _DroneState;
    nav_msgs::Odometry Drone_odom;
    drone_msgs::ControlCommand Command_Now; 
    float yaw_ref_comd;
    Eigen::Vector3d raw_goal_pos;
    geometry_msgs::PointStamped  joy_goal_rviz; 
    int goalPoints_seq_id;
    Eigen::Vector3d stop_pos, goal_pos, goal_vel, goal_acc, cur_pos, cur_vel, cur_acc, cur_pos_ref, cur_vel_ref, cur_acc_ref;
    double cur_roll, cur_pitch, cur_yaw, cur_yaw_rate;
    float rc_x, rc_y, rc_z, rc_r;
    int flag_tracking;
    
	
	/* trajectory status */
    Eigen::Vector3d guide_point;
    geometry_msgs::PointStamped  guide_rviz; 
    Eigen::Vector3d closest_obs;
    geometry_msgs::PointStamped  closest_rviz; 
    UniformBspline traj_init_;
    UniformBspline traj_pos_, traj_vel_, traj_acc_;
    double traj_duration_, t_start_, t_end_;
    ros::Time time_traj_start;
	

	/* ---------- main function ---------- */ 
    void init(ros::NodeHandle& nh);
    
    bool checkTrajCollision(double t_start, double t_end, double safe_range, double dt_min=0.02, double dt_max=0.2, bool force=false);
    
    int safetyCheck();
    
    bool generateTrajectory(Eigen::Vector3d start_pt_, Eigen::Vector3d start_vel_, Eigen::Vector3d start_acc_, Eigen::Vector3d end_pt_, Eigen::Vector3d end_vel_, Eigen::Vector3d end_acc_);  // front-end && back-end
		                  
    void csv_writer(Eigen::Vector3d start_pos_, Eigen::Vector3d start_vel_, Eigen::Vector3d start_acc_, Eigen::Vector3d end_pos_, Eigen::Vector3d end_vel_, Eigen::Vector3d end_acc_, Eigen::MatrixXd control_pts, int index);
    
    /* ---------- evaluation function ---------- */
    void getCalculatingTime(double& ts, double& to)
    {
        ts = time_planning_;
        to = time_optimize_;
    }
    void getCostCurve(vector<double>& cost, vector<double>& time)
    {
        bspline_optimizer_->getCostCurve(cost, time);
    }

};



}
#endif 
