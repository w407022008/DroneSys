//ros头文件
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include "message_utils.h"
#include "math.h"

//topic 头文件
#include <geometry_msgs/Point.h>
#include <drone_msgs/ControlCommand.h>
#include <drone_msgs/DroneState.h>
#include <drone_msgs/PositionReference.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
using namespace std;

#define MIN_DIS 0.1
# define NODE_NAME "planning_tracking"

drone_msgs::ControlCommand Command_Now; 
drone_msgs::DroneState _DroneState;   

geometry_msgs::PoseStamped goal;                  

drone_msgs::PositionReference fast_planner_cmd; 

bool sim_mode,enable;
bool yaw_tracking_mode;
int flag_get_cmd = 0;
int flag_get_goal = 0;
float desired_yaw = 0;  //[rad]
float distance_to_goal = 0;
ros::Time TimeNow;
Eigen::Vector3d stop_point;

void fast_planner_cmd_cb(const drone_msgs::PositionReference::ConstPtr& msg)
{
    flag_get_cmd = 1;
    fast_planner_cmd = *msg;
}

void drone_state_cb(const drone_msgs::DroneState::ConstPtr& msg)
{
    _DroneState = *msg;
    distance_to_goal = sqrt(  pow(_DroneState.position[0] - goal.pose.position.x, 2) 
                            + pow(_DroneState.position[1] - goal.pose.position.y, 2) );
}
void goal_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    goal = *msg;
    flag_get_goal = 1;
    cout << "Get a new goal!"<<endl;
    if (msg->pose.position.z < 1)  // the minimal goal height 
    {	
        goal.pose.position.z = 1;
    }else if (msg->pose.position.z > 2)  // the maximal goal height 
    {	
        goal.pose.position.z = 2;
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "fast_planning_tracking");
    ros::NodeHandle nh("~");

    nh.param<bool>("fast_planning_tracking/sim_mode", sim_mode, false); 
    nh.param<bool>("fast_planning_tracking/enable", enable, true); 
    nh.param<bool>("fast_planning_tracking/yaw_tracking_mode", yaw_tracking_mode, true);

    ros::Subscriber drone_state_sub = nh.subscribe<drone_msgs::DroneState>("/drone_msg/drone_state", 10, drone_state_cb);
    ros::Subscriber fast_planner_sub = nh.subscribe<drone_msgs::PositionReference>("/fast_planner/position_cmd", 50, fast_planner_cmd_cb);
    ros::Subscriber goal_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone_msg/planning/goal", 10,goal_cb);
    
    ros::Publisher command_pub = nh.advertise<drone_msgs::ControlCommand>("/drone_msg/control_command", 10);
    ros::Publisher message_pub = nh.advertise<drone_msgs::Message>("/drone_msg/message", 10);
   
    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Fast Planner <<<<<<<<<<<<<<<<<<<<<<<<<<< "<< endl;
    int start_flag = 0;
    if(sim_mode)
    {
        // Waiting for type in 
        while(start_flag == 0)
        {
            cout << "Please type in 1 for taking off:"<<endl;
            cin >> start_flag;
        }
        // Takeoff
        Command_Now.header.stamp = ros::Time::now();
        Command_Now.Mode  = drone_msgs::ControlCommand::Idle;
        Command_Now.Command_ID = 1;
        Command_Now.source = NODE_NAME;
        Command_Now.Reference_State.yaw_ref = 999;
        command_pub.publish(Command_Now);   
        ros::Duration(2.0).sleep();
        cout << "Switched to OFFBOARD and armed, drone will take off after 1.0s"<<endl;
        ros::Duration(1.0).sleep();

        Command_Now.header.stamp = ros::Time::now();
        Command_Now.Mode = drone_msgs::ControlCommand::Takeoff;
        Command_Now.Command_ID = Command_Now.Command_ID + 1;
        Command_Now.source = NODE_NAME;
        command_pub.publish(Command_Now);
        cout << "Takeoff"<<endl;
    }else
    {
        // Real flight situation: Manually switch to offboard mode and send a take-off command
        while(start_flag == 0)
        {
            cout << "Please manually switch to offboard mode and arm it and then type in 1 for takeoff:"<<endl;
            cin >> start_flag;
        }
        Command_Now.header.stamp = ros::Time::now();
        Command_Now.Mode = drone_msgs::ControlCommand::Takeoff;
        Command_Now.Command_ID = Command_Now.Command_ID + 1;
        Command_Now.source = NODE_NAME;
        command_pub.publish(Command_Now);
        cout << "Takeoff"<<endl;
    }

    while (fabs(_DroneState.velocity[2])>0.3 || fabs(_DroneState.position[2])<0.1){
        ros::spinOnce();
        ros::Duration(1.0).sleep();
    }
    stop_point[0] = _DroneState.position[0];
    stop_point[1] = _DroneState.position[1];
    stop_point[2] = _DroneState.position[2];
    
    while (ros::ok())
    {
        static int exec_num=0;
        exec_num++;

        // 若goal为99，则降落并退出任务
        if(goal.pose.position.x == 99)
        {
            // 抵达目标附近，则停止速度控制，改为位置控制
            Command_Now.header.stamp = ros::Time::now();
            Command_Now.Mode                                = drone_msgs::ControlCommand::Land;
            Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
            Command_Now.source = NODE_NAME;

            command_pub.publish(Command_Now);
            cout << "Quit... " << endl;

            return 0;
        }

        //回调
        ros::spinOnce();

        if(!enable) continue;

        if( flag_get_cmd == 0 || flag_get_goal == 0)
        {
            if (yaw_tracking_mode){
                desired_yaw = desired_yaw + 0.5;//M_PI*(ros::Time::now()-TimeNow).toSec();
                TimeNow = ros::Time::now();
                Command_Now.header.stamp = TimeNow;
                Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
                Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
                Command_Now.source = NODE_NAME;
                Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XYZ_POS;
                Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
                Command_Now.Reference_State.position_ref[0]     = stop_point[0];
                Command_Now.Reference_State.position_ref[1]     = stop_point[1];
                Command_Now.Reference_State.position_ref[2]     = stop_point[2];

                Command_Now.Reference_State.yaw_ref             = desired_yaw;
                command_pub.publish(Command_Now);
            }
            if(exec_num == 10)
            {
                if (flag_get_goal == 0)
                    cout << "Waiting for goal... " << endl;
		else if (flag_get_cmd == 0)
                    cout << "Waiting for trajectory..." << endl;
                exec_num=0;
            }
            ros::Duration(0.5).sleep();
        }
        else if (distance_to_goal < MIN_DIS)
        {
            cout << "Arrived the goal, waiting for a new goal... " << endl;
            cout << "drone_pos: " << _DroneState.position[0] << " [m] "<< _DroneState.position[1] << " [m] "<< _DroneState.position[2] << " [m] "<<endl;
            cout << "goal_pos: " << goal.pose.position.x << " [m] "<< goal.pose.position.y << " [m] "<< goal.pose.position.z << " [m] "<<endl;
            
            // 抵达目标附近，则停止速度控制，改为位置控制
            TimeNow = ros::Time::now();
            Command_Now.header.stamp = TimeNow;
            Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
            Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
            Command_Now.source = NODE_NAME;
            Command_Now.Reference_State.Move_mode           = drone_msgs::PositionReference::XYZ_POS;
            Command_Now.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
            Command_Now.Reference_State.position_ref[0]     = goal.pose.position.x;
            Command_Now.Reference_State.position_ref[1]     = goal.pose.position.y;
            Command_Now.Reference_State.position_ref[2]     = goal.pose.position.z;

            Command_Now.Reference_State.yaw_ref             = desired_yaw;
            command_pub.publish(Command_Now);

            flag_get_goal = 0;
            flag_get_cmd = 0;
            stop_point[0] = goal.pose.position.x;
			stop_point[1] = goal.pose.position.y;
			stop_point[2] = goal.pose.position.z;
        }
        else
        {
            if (yaw_tracking_mode)
            {
                if( sqrt( fast_planner_cmd.velocity_ref[1]* fast_planner_cmd.velocity_ref[1]
                        +  fast_planner_cmd.velocity_ref[0]* fast_planner_cmd.velocity_ref[0])  >  0.1  )
                {
                    auto sign=[](double v)->double
                    {
                        return v<0.0? -1.0:1.0;
                    };
                    Eigen::Vector3d ref_vel;
                    ref_vel[0] = fast_planner_cmd.velocity_ref[0];
                    ref_vel[1] = fast_planner_cmd.velocity_ref[1];
                    ref_vel[2] = 0.0;

                    float next_desired_yaw_vel      = sign(ref_vel[1]) * acos(ref_vel[0]/ref_vel.norm());
        //            float next_desired_yaw_pos      = sign((ref_pos - curr_pos)[1]) * acos((ref_pos - curr_pos)[0]/curr_pos.norm());

                    if (fabs(desired_yaw-next_desired_yaw_vel)<M_PI)
                        desired_yaw = (0.3*desired_yaw + 0.7*next_desired_yaw_vel);
                    else
                        desired_yaw = next_desired_yaw_vel + sign(next_desired_yaw_vel) * 0.3/(0.3+0.7)*(2*M_PI-fabs(desired_yaw-next_desired_yaw_vel));
                } else {
                    desired_yaw = desired_yaw + 0.5;//M_PI*(ros::Time::now()-TimeNow).toSec();
                }
                
                if(desired_yaw>M_PI)
                    desired_yaw -= 2*M_PI;
                else if (desired_yaw<-M_PI)
                    desired_yaw += 2*M_PI;
            }else
            {
                desired_yaw = 0.0;
            }

            TimeNow = ros::Time::now();
            Command_Now.header.stamp = TimeNow;
            Command_Now.Mode                                = drone_msgs::ControlCommand::Move;
            Command_Now.Command_ID                          = Command_Now.Command_ID + 1;
            Command_Now.source = NODE_NAME;
            Command_Now.Reference_State =  fast_planner_cmd;
            Command_Now.Reference_State.yaw_ref = desired_yaw;

            command_pub.publish(Command_Now);
            
            if(Eigen::Vector3d(fast_planner_cmd.velocity_ref[0],fast_planner_cmd.velocity_ref[1],fast_planner_cmd.velocity_ref[2]).norm() < 1e-6){
                stop_point[0] = fast_planner_cmd.position_ref[0];
                stop_point[1] = fast_planner_cmd.position_ref[1];
                stop_point[2] = fast_planner_cmd.position_ref[2];
                flag_get_cmd = 0;
            }
            ros::Duration(0.05).sleep();
        }
    }

    return 0;

}