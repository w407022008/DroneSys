#ifndef GLOBAL_PLANNER
#define GLOBAL_PLANNER

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>
#include <iostream>

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include "drone_msgs/PositionReference.h"
#include "drone_msgs/Message.h"
#include "drone_msgs/DroneState.h"
#include "drone_msgs/ControlCommand.h"

#include "A_star.h"
#include <kinodynamic_astar.h>
#include "occupy_map.h"
#include "tools.h"
#include "message_utils.h"

using namespace std;

#define NODE_NAME "Global_Planner [main]"

#define MIN_DIS 0.1

namespace Global_Planning
{
class Global_Planner
{
private:

    ros::NodeHandle global_planner_nh;

    int algorithm_mode;
    bool is_2D;
    bool yaw_tracking_mode;
    double fly_height_2D;
    double safe_distance;
    double time_per_path;
    int map_input;
    double replan_time, tracking_start_time;
    bool consider_neighbour;
    bool sim_mode;
    bool map_groundtruth;
    bool planner_enable_default;
    bool planner_enable;


    ros::Subscriber goal_sub;
    ros::Subscriber planner_switch_sub;
    ros::Subscriber drone_state_sub;
    
    ros::Subscriber Gpointcloud_sub;
    ros::Subscriber Lpointcloud_sub;
    ros::Subscriber laserscan_sub;


    ros::Publisher command_pub,path_cmd_pub;
    ros::Timer mainloop_timer, track_path_timer, safety_timer;

    global_planning_alg::Ptr global_alg_ptr;

    drone_msgs::DroneState _DroneState;
    nav_msgs::Odometry Drone_odom;

    nav_msgs::Path path_cmd;
    double distance_walked;
    drone_msgs::ControlCommand Command_Now;   

    double distance_to_goal;

    bool odom_ready;
    bool drone_ready;
    bool sensor_ready;
    bool goal_ready; 
    bool is_safety;
    bool is_new_path;
    bool path_ok;
    int start_point_index;
    int Num_total_wp;
    int cur_id;

    Eigen::Vector3d start_pos, start_vel, start_acc, goal_pos, goal_vel;

    float desired_yaw;

    ros::Time tra_start_time;
    float tra_running_time;
    
    string message;

    enum EXEC_STATE
    {
        WAIT_GOAL,
        PLANNING,
        TRACKING,
        LANDING,
    };
    EXEC_STATE exec_state;

    void planner_switch_cb(const std_msgs::Bool::ConstPtr& msg);
    void goal_cb(const geometry_msgs::PoseStampedConstPtr& msg);
    void drone_state_cb(const drone_msgs::DroneStateConstPtr &msg);
    void Gpointcloud_cb(const sensor_msgs::PointCloud2ConstPtr &msg);
    void Lpointcloud_cb(const sensor_msgs::PointCloud2ConstPtr &msg);
    void laser_cb(const sensor_msgs::LaserScanConstPtr &msg);

    void safety_cb(const ros::TimerEvent& e);
    void mainloop_cb(const ros::TimerEvent& e);
    void track_path_cb(const ros::TimerEvent& e);
   
    float get_time_in_sec(const ros::Time& begin_time);

    int get_start_point_id(void);
    
public:
    Global_Planner(void):
        global_planner_nh("~")
    {}~Global_Planner(){}

    void init(ros::NodeHandle& nh);
};

}

#endif
