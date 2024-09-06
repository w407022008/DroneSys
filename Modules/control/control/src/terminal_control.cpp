#include <ros/ros.h>
#include <iostream>

#include <drone_msgs/ControlCommand.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Bool.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <drone_msgs/Message.h>
#include <drone_msgs/DroneState.h>

#include "message_utils.h"
#include "trajectory_generation.h"
#include "KeyboardEvent.h"

#define VEL_XY_STEP_SIZE 0.1
#define VEL_Z_STEP_SIZE 0.1
#define YAW_STEP_SIZE 0.08
#define TRA_WINDOW 2000
#define NODE_NAME "terminal_control"

using namespace std;

drone_msgs::ControlCommand Command_to_pub;
drone_msgs::DroneState _DroneState;
std::vector<geometry_msgs::PoseStamped> posehistory_vector_;


float time_trajectory = 0.0;
float trajectory_total_time = 50.0;
float cur_pos[3];
float init_pos_xy[2];
bool armed;
bool custom_controller_available = false;
int Remote_Mode = -1;
int controller_switch = 0;
std::string mesg;
char key_now;
char key_last = U_KEY_NONE;
bool _pause_ = false;


ros::Subscriber state_sub;
ros::Publisher move_pub, command_pose_pub, command_trajPoint_pub, message_pub, ref_trajectory_pub, command_active_pub;

void state_cb(const drone_msgs::DroneStateConstPtr& msg)
{
    // _DroneState = *msg; // in ENU frame
    armed = msg->armed;
    if(!armed){
        init_pos_xy[0] = msg->position[0];
        init_pos_xy[1] = msg->position[1];
    }
    cur_pos[0] = msg->position[0];
    cur_pos[1] = msg->position[1];
    cur_pos[2] = msg->position[2];
}

void arm_disarm()
{
    if(armed){
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Disarm Mode.");
    
        Command_to_pub.Mode = drone_msgs::ControlCommand::Disarm;
        Command_to_pub.source = NODE_NAME;
    }else{
        pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Arming and Switch to OFFBOARD.");
    
        Command_to_pub.Mode = drone_msgs::ControlCommand::Idle;
        Command_to_pub.source = NODE_NAME;
        Command_to_pub.Reference_State.yaw_ref = 999; // set to armed
    }
    Command_to_pub.header.stamp = ros::Time::now();
    Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
    
    move_pub.publish(Command_to_pub);
}

void setpointIn();
void keyboardControl();
void generate_com(int Move_mode, float state_desired[4]);
void Draw_in_rviz(const drone_msgs::PositionReference& pos_ref, bool draw_trajectory);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "terminal_control");
    ros::NodeHandle nh("~");

    cout.setf(ios::fixed);
    cout<<setprecision(2);
    cout.setf(ios::left);
    cout.setf(ios::showpoint);
    
    string uav_name;
    nh.param<string>("uav_name", uav_name, "");
    if(uav_name!="") cout<<"[terminal_control]: uav_name: "<<uav_name<<endl;

    // [SUB] Drone state
    state_sub = nh.subscribe<drone_msgs::DroneState>(uav_name+"/drone_msg/drone_state", 10, state_cb);

    // [PUB] control command
    move_pub = nh.advertise<drone_msgs::ControlCommand>(uav_name+"/drone_msg/control_command", 10);
    command_pose_pub = nh.advertise<geometry_msgs::PoseStamped>(uav_name+"/command/pose", 10);
    command_trajPoint_pub = nh.advertise<quadrotor_msgs::TrajectoryPoint>(uav_name+"/command/reference_state", 10);
    command_active_pub = nh.advertise<std_msgs::Bool>(uav_name+"/command/active",10);

    // [PUB] Rviz trajectory
    ref_trajectory_pub = nh.advertise<nav_msgs::Path>(uav_name+"/drone_msg/reference_trajectory", 10);

    // [PUB] ground station messages
    message_pub = nh.advertise<drone_msgs::Message>(uav_name+"/drone_msg/message", 10);

    // Initialization: Idle mode default
    Command_to_pub.Mode                                = drone_msgs::ControlCommand::Idle;
    Command_to_pub.Command_ID                          = 0;
    Command_to_pub.source = NODE_NAME;
    Command_to_pub.Reference_State.Move_mode           = drone_msgs::PositionReference::XYZ_POS;
    Command_to_pub.Reference_State.Move_frame          = drone_msgs::PositionReference::ENU_FRAME;
    Command_to_pub.Reference_State.position_ref[0]     = 0;
    Command_to_pub.Reference_State.position_ref[1]     = 0;
    Command_to_pub.Reference_State.position_ref[2]     = 0;
    Command_to_pub.Reference_State.velocity_ref[0]     = 0;
    Command_to_pub.Reference_State.velocity_ref[1]     = 0;
    Command_to_pub.Reference_State.velocity_ref[2]     = 0;
    Command_to_pub.Reference_State.acceleration_ref[0] = 0;
    Command_to_pub.Reference_State.acceleration_ref[1] = 0;
    Command_to_pub.Reference_State.acceleration_ref[2] = 0;
    Command_to_pub.Reference_State.yaw_ref             = 0;


    // Remote mode
    while(ros::ok())
    {
        // system ("clear");
        if(Remote_Mode == 0)
        {
            setpointIn();
        }else if(Remote_Mode == 1)
        {
            custom_controller_available = false;
            keyboardControl();
        }else//(Remote_Mode == -1)
        {
            system ("clear");
            cout << ">>>>>>>>>>>>>>>> Terminal Control <<<<<<<<<<<<<<<<"<< endl;
            cout << "Please choose a controller: 0 for PX4 cascade PID controller, 1 for custom controller"<<endl;
            cin >> controller_switch;
            cout << "Please choose the Remote Mode: 0 for keyboard input one setpoint, 1 for keyboard input control"<<endl;
            cin >> Remote_Mode;

            if(Remote_Mode == 1){
                system ("clear");
                Traj_gen Traj_gen;
                Traj_gen.printf_param();
                cout << ">>>>>>>>>>>>>>>> Terminal Keyboard Input Control <<<<<<<<<<<<<<<<"<< endl;
                cout << "R:\tArm or Disarm\t T:\tTakeoff\n L:\tLand\t H:\tHold\n 8:\tfigure-of-8 trajectory\t 0:\tcircle trajectory" <<endl;
                cout << "Move (XYZ_VEL) in (BODY_FRAME): w/s for body_x, a/d for body_y, k/m for body_z, q/e for body_yaw" <<endl;
                cout << "SPACE to pause." <<endl;
                cout << "Z to roll back." <<endl;
            }else if(Remote_Mode == 0)
            {
                system ("clear");
                cout << "Type R for arming or disarm" <<endl;
                cout << "Type T for takeoff" <<endl;
                cout << "Type Z to roll back" <<endl;
                cout << "Type S to set a setpoint" <<endl;
            }
            sleep(0.1);
        }
        ros::spinOnce();
    }
}

void setpointIn()
{
    KeyboardEvent keyboardcontrol;
    key_now = keyboardcontrol.GetKeyOnce();
    if(key_now == U_KEY_NONE) return;

    int Move_mode = 0;
    int Move_frame = 0;
    float state_desired[4];
    switch(key_now){
        // If type in S: set setpoint
        case U_KEY_S:
            cout << "Please choose the Command.Reference_State.Move_mode: 0 for XYZ_POS, 1 for XY_POS_Z_VEL, 2 for XY_VEL_Z_POS, 3 for XYZ_VEL"<<endl;
            cin >> Move_mode;
            if(Move_mode == 999){Remote_Mode=-1;return;}
            cout << "Please choose the Command.Reference_State.Move_frame: 0 for ENU_FRAME, 1 for BODY_FRAME"<<endl;
            cin >> Move_frame; 
            if(Move_frame == 999){Remote_Mode=-1;return;}
            cout << "Please input the reference state [x y z yaw]: "<< endl;
            cout << "setpoint --- x [m or m/s] : ";
            cin >> state_desired[0];
            if(state_desired[0] == 999){Remote_Mode=-1;return;}
            cout << "\nsetpoint --- y [m or m/s] : ";
            cin >> state_desired[1];
            if(state_desired[1] == 999){Remote_Mode=-1;return;}
            cout << "\nsetpoint --- z [m or m/s] : ";
            cin >> state_desired[2];
            if(state_desired[2] == 999){Remote_Mode=-1;return;}
            cout << "\nsetpoint --- yaw [deg or deg/s] : ";
            cin >> state_desired[3];
            if(state_desired[3] == 999){Remote_Mode=-1;return;}

            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode  = Move_mode;
            Command_to_pub.Reference_State.Move_frame = Move_frame;
            Command_to_pub.Reference_State.time_from_start = 0.0;
            generate_com(Move_mode, state_desired);

            custom_controller_available = true;

            system ("clear");
            cout << "Type R for arming or disarm" <<endl;
            cout << "Type T for takeoff" <<endl;
            cout << "Type Z to roll back" <<endl;
            cout << "Type S to set a setpoint" <<endl;
            break;

        // If type in R: switch to OFFBOARD mode and Arming or Disarm
        case U_KEY_R:
            arm_disarm();
            custom_controller_available = false;

            return;

        // If type in T: Takeoff
        case U_KEY_T:
            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Takeoff Mode.");

            Command_to_pub.Mode = drone_msgs::ControlCommand::Takeoff;
            Command_to_pub.Reference_State.yaw_ref = 0.0;
            Command_to_pub.source = NODE_NAME;
            custom_controller_available = false;
            
            break;

        // If type in L: Landing
        case U_KEY_L:
            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Land Mode.");
        
            Command_to_pub.Mode = drone_msgs::ControlCommand::Land;
            Command_to_pub.source = NODE_NAME;
            custom_controller_available = false;
            
            break;

        // If type in Z: roll back
        case U_KEY_Z:
            Remote_Mode = -1;
            
        // If type in H: Hovering
        case U_KEY_H:
            custom_controller_available = false;
            if(key_last != key_now){
                pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Hold Mode.");

                Command_to_pub.Mode = drone_msgs::ControlCommand::Hold;
                Command_to_pub.source = NODE_NAME;
                Command_to_pub.Reference_State.position_ref[0]     = cur_pos[0];
                Command_to_pub.Reference_State.position_ref[1]     = cur_pos[1];
                Command_to_pub.Reference_State.position_ref[2]     = cur_pos[2];
                Command_to_pub.Reference_State.velocity_ref[0]     = 0;
                Command_to_pub.Reference_State.velocity_ref[1]     = 0;
                Command_to_pub.Reference_State.velocity_ref[2]     = 0;
                Command_to_pub.Reference_State.acceleration_ref[0] = 0;
                Command_to_pub.Reference_State.acceleration_ref[1] = 0;
                Command_to_pub.Reference_State.acceleration_ref[2] = 0;

            }
            
            break;

    }
    Command_to_pub.header.stamp = ros::Time::now();
    Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;

    if(controller_switch){
        std_msgs::Bool command_active;
        command_active.data = custom_controller_available;
        command_active_pub.publish(command_active);
    }
    if(controller_switch && custom_controller_available){
        // quadrotor_msgs::TrajectoryPoint pose_cmd;
        // pose_cmd.time_from_start = ros::Duration(Command_to_pub.Reference_State.time_from_start);
        // pose_cmd.pose.position.x = Command_to_pub.Reference_State.position_ref[0];
        // pose_cmd.pose.position.y = Command_to_pub.Reference_State.position_ref[1];
        // pose_cmd.pose.position.z = Command_to_pub.Reference_State.position_ref[2];
        // pose_cmd.velocity.linear.x = Command_to_pub.Reference_State.velocity_ref[0];
        // pose_cmd.velocity.linear.y = Command_to_pub.Reference_State.velocity_ref[1];
        // pose_cmd.velocity.linear.z = Command_to_pub.Reference_State.velocity_ref[2];
        // pose_cmd.heading = Command_to_pub.Reference_State.yaw_ref;

        // command_trajPoint_pub.publish(pose_cmd);
        geometry_msgs::PoseStamped pose_cmd;
        pose_cmd.pose.position.x = Command_to_pub.Reference_State.position_ref[0];
        pose_cmd.pose.position.y = Command_to_pub.Reference_State.position_ref[1];
        pose_cmd.pose.position.z = Command_to_pub.Reference_State.position_ref[2];
        Eigen::Quaterniond q_yaw = Eigen::Quaterniond(
                                Eigen::AngleAxisd(
                                    Command_to_pub.Reference_State.yaw_ref, 
                                    Eigen::Vector3d::UnitZ()));
        pose_cmd.pose.orientation.w = q_yaw.w();
        pose_cmd.pose.orientation.x = q_yaw.x();
        pose_cmd.pose.orientation.y = q_yaw.y();
        pose_cmd.pose.orientation.z = q_yaw.z();

        command_pose_pub.publish(pose_cmd);
    }else{
        move_pub.publish(Command_to_pub);
    }
    key_last = key_now;
}

void keyboardControl()
{
    Traj_gen Traj_gen;
    KeyboardEvent keyboardcontrol;

    key_now = keyboardcontrol.GetKeyOnce();
    if(key_now == U_KEY_NONE)
    {
        if(trajmove.find(key_last) != trajmove.end())
        {
            key_now = key_last;
        }else if(joymove.find(key_last) != joymove.end())
        {
            key_now = U_KEY_PASS;
        }
    }

    switch(key_now){
        // If type in W: flying forward
        case U_KEY_W:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.velocity_ref[0]     += VEL_XY_STEP_SIZE;
            
            mesg = "Current Velocity [X Y Z]: " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[0]) + " [m/s] " +
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[1]) + " [m/s] " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[2]) + " [m/s] " + "\r";
            std::cout << mesg << std::flush;

            sleep(0.1);
            
            break;
        
        // If type in S: flying backward
        case U_KEY_S:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.velocity_ref[0]     -= VEL_XY_STEP_SIZE;

            mesg = "Current Velocity [X Y Z]: " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[0]) + " [m/s] " +
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[1]) + " [m/s] " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[2]) + " [m/s] " + "\r";
            std::cout << mesg << std::flush;

            sleep(0.1);

            break;

        // If type in A: flying left
        case U_KEY_A:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.velocity_ref[1]     += VEL_XY_STEP_SIZE;
            
            mesg = "Current Velocity [X Y Z]: " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[0]) + " [m/s] " +
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[1]) + " [m/s] " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[2]) + " [m/s] " + "\r";
            std::cout << mesg << std::flush;
            
            sleep(0.1);

            break;

        // If type in D: flying right
        case U_KEY_D:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.velocity_ref[1]     -= VEL_XY_STEP_SIZE;

            mesg = "Current Velocity [X Y Z]: " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[0]) + " [m/s] " +
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[1]) + " [m/s] " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[2]) + " [m/s] " + "\r";
            std::cout << mesg << std::flush;
            
            sleep(0.1);

            break;

        // If type in K: flying upward
        case U_KEY_K:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.velocity_ref[2]     += VEL_Z_STEP_SIZE;

            mesg = "Current Velocity [X Y Z]: " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[0]) + " [m/s] " +
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[1]) + " [m/s] " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[2]) + " [m/s] " + "\r";
            std::cout << mesg << std::flush;
            
            sleep(0.1);

            break;

        // If type in M: flying downward
        case U_KEY_M:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.velocity_ref[2]     -= VEL_Z_STEP_SIZE;

            mesg = "Current Velocity [X Y Z]: " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[0]) + " [m/s] " +
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[1]) + " [m/s] " + 
                                    std::to_string(Command_to_pub.Reference_State.velocity_ref[2]) + " [m/s] " + "\r";
            std::cout << mesg << std::flush;
            
            sleep(0.1);
            
            break;

        // If type in Q: turn left 
        case U_KEY_Q:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.yaw_ref         += YAW_STEP_SIZE;
            
            mesg = "Increase the Yaw angle as: " + std::to_string(Command_to_pub.Reference_State.yaw_ref) + "\r";
            std::cout << mesg << std::flush;

            sleep(0.1);
            
            break;

        // If type in E: turn right
        case U_KEY_E:
            Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_POS;
            Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
            Command_to_pub.Reference_State.yaw_ref         -= YAW_STEP_SIZE;
            
            mesg = "Decrease the Yaw angle as: " + std::to_string(Command_to_pub.Reference_State.yaw_ref) + "\r";
            std::cout << mesg << std::flush;
            
            sleep(0.1);
            
            break;
    }

    switch(key_now){
        // If type in 0: tracking a circle traj 
        case U_KEY_0:
            if(key_last != key_now){
                _pause_ = false;
                time_trajectory = 0.0;
                cout << "Input the trajectory_total_time:"<<endl;
                cin >> trajectory_total_time;
                cout << "\r";
            }
            if(time_trajectory < trajectory_total_time)
            {
                Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
                Command_to_pub.source = NODE_NAME;

                Command_to_pub.Reference_State = Traj_gen.Circle_trajectory_generation(time_trajectory, init_pos_xy);

                float dis=0.0, dif[3];
                for(int i=0;i<3;i++)
                {
                    dif[i] = Command_to_pub.Reference_State.position_ref[i]-cur_pos[i];
                    dis += dif[i]*dif[i];
                }
                dis = sqrt(dis);

                if(!_pause_ && time_trajectory>0.01){
                    custom_controller_available = true;
                    time_trajectory = time_trajectory + 0.01;
                }else{
                    if(_pause_){
                        Command_to_pub.Reference_State.velocity_ref[0] = 0.0;
                        Command_to_pub.Reference_State.velocity_ref[1] = 0.0;
                        Command_to_pub.Reference_State.velocity_ref[2] = 0.0;
                    }else if(time_trajectory<0.01)
                    {
                        if(dis>0.3){
                            Command_to_pub.Reference_State.Move_mode  = drone_msgs::PositionReference::XYZ_VEL;
                            Command_to_pub.Reference_State.Move_frame = drone_msgs::PositionReference::ENU_FRAME;
                            Command_to_pub.Reference_State.time_from_start = 0.0;
                            Command_to_pub.Reference_State.velocity_ref[0] = dif[0]/dis;
                            Command_to_pub.Reference_State.velocity_ref[1] = dif[1]/dis;
                            Command_to_pub.Reference_State.velocity_ref[2] = dif[2]/dis;
                            Command_to_pub.Reference_State.yaw_ref = 0.0;
                        }else{
                            Command_to_pub.Reference_State.Move_mode  = drone_msgs::PositionReference::XYZ_POS;
                            Command_to_pub.Reference_State.Move_frame = drone_msgs::PositionReference::ENU_FRAME;
                            Command_to_pub.Reference_State.time_from_start = 0.0;
                            Command_to_pub.Reference_State.position_ref[0] = Command_to_pub.Reference_State.position_ref[0];
                            Command_to_pub.Reference_State.position_ref[1] = Command_to_pub.Reference_State.position_ref[1];
                            Command_to_pub.Reference_State.position_ref[2] = Command_to_pub.Reference_State.position_ref[2];
                            Command_to_pub.Reference_State.yaw_ref = 0.0;
                            time_trajectory = time_trajectory + 0.002;
                        }
                    }
                }
                mesg = "Trajectory tracking: " + 
                                        std::to_string(time_trajectory) + " / " +
                                        std::to_string(trajectory_total_time) + " [s] " + "\r";
                std::cout << mesg << std::flush;
                
                Draw_in_rviz(Command_to_pub.Reference_State, true);

                sleep(0.01);
            }else{
                key_now = U_KEY_H;
            }
            break;

        // If type in 8: tracking a figure_of_8 traj 
        case U_KEY_8:
            if(key_last != key_now){
                _pause_ = false;
                time_trajectory = 0.0;
                cout << "Input the trajectory_total_time:"<<endl;
                cin >> trajectory_total_time;
                cout << "\r";
            }
            if(time_trajectory < trajectory_total_time)
            {
                Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
                Command_to_pub.source = NODE_NAME;

                Command_to_pub.Reference_State = Traj_gen.Eight_trajectory_generation(time_trajectory, init_pos_xy);

                float dis=0.0, dif[3];
                for(int i=0;i<3;i++)
                {
                    dif[i] = Command_to_pub.Reference_State.position_ref[i]-cur_pos[i];
                    dis += dif[i]*dif[i];
                }
                dis = sqrt(dis);

                if(!_pause_ && time_trajectory>0.01){
                    custom_controller_available = true;
                    time_trajectory = time_trajectory + 0.01;
                }else{
                    if(_pause_){
                        Command_to_pub.Reference_State.velocity_ref[0] = 0.0;
                        Command_to_pub.Reference_State.velocity_ref[1] = 0.0;
                        Command_to_pub.Reference_State.velocity_ref[2] = 0.0;
                    }else if(time_trajectory<0.01)
                    {
                        if(dis>0.3){
                            Command_to_pub.Reference_State.Move_mode  = drone_msgs::PositionReference::XYZ_VEL;
                            Command_to_pub.Reference_State.Move_frame = drone_msgs::PositionReference::ENU_FRAME;
                            Command_to_pub.Reference_State.time_from_start = 0.0;
                            Command_to_pub.Reference_State.velocity_ref[0] = dif[0]/dis;
                            Command_to_pub.Reference_State.velocity_ref[1] = dif[1]/dis;
                            Command_to_pub.Reference_State.velocity_ref[2] = dif[2]/dis;
                            Command_to_pub.Reference_State.yaw_ref = 0.0;
                        }else{
                            Command_to_pub.Reference_State.Move_mode  = drone_msgs::PositionReference::XYZ_POS;
                            Command_to_pub.Reference_State.Move_frame = drone_msgs::PositionReference::ENU_FRAME;
                            Command_to_pub.Reference_State.time_from_start = 0.0;
                            Command_to_pub.Reference_State.position_ref[0] = Command_to_pub.Reference_State.position_ref[0];
                            Command_to_pub.Reference_State.position_ref[1] = Command_to_pub.Reference_State.position_ref[1];
                            Command_to_pub.Reference_State.position_ref[2] = Command_to_pub.Reference_State.position_ref[2];
                            Command_to_pub.Reference_State.yaw_ref = 0.0;
                            time_trajectory = time_trajectory + 0.002;
                        }
                    }
                }

                mesg = "Trajectory tracking: " + 
                                        std::to_string(time_trajectory) + " / " +
                                        std::to_string(trajectory_total_time) + " [s] " + "\r";
                std::cout << mesg << std::flush;
                
                Draw_in_rviz(Command_to_pub.Reference_State, true);

                sleep(0.01);
            }else{
                key_now = U_KEY_H;
            }
            break;

        // If type in 7: step 
        case U_KEY_7:
            if(key_last != key_now){
                _pause_ = false;
                time_trajectory = 0.0;
                cout << "Input the trajectory_total_time:"<<endl;
                cin >> trajectory_total_time;
                cout << "\r";
            }
            if(time_trajectory < trajectory_total_time)
            {
                custom_controller_available = true;
                Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
                Command_to_pub.source = NODE_NAME;

                Command_to_pub.Reference_State = Traj_gen.Step_trajectory_generation(time_trajectory, init_pos_xy);
                if(_pause_){
                    Command_to_pub.Reference_State.velocity_ref[0] = 0.0;
                    Command_to_pub.Reference_State.velocity_ref[1] = 0.0;
                    Command_to_pub.Reference_State.velocity_ref[2] = 0.0;
                }else{
                    custom_controller_available = true;
                    time_trajectory = time_trajectory + 0.01;
                }

                mesg = "Trajectory tracking: " + 
                                        std::to_string(time_trajectory) + " / " +
                                        std::to_string(trajectory_total_time) + " [s] " + "\r";
                std::cout << mesg << std::flush;
                
                Draw_in_rviz(Command_to_pub.Reference_State, true);

                sleep(0.01);
            }else{
                key_now = U_KEY_H;
            }
            break;

        // If type in 9: tracking a line 
        case U_KEY_9:
            if(key_last != key_now){
                _pause_ = false;
                time_trajectory = 0.0;
                cout << "Input the trajectory_total_time:"<<endl;
                cin >> trajectory_total_time;
                cout << "\r";
            }
            if(time_trajectory < trajectory_total_time)
            {
                custom_controller_available = true;
                Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
                Command_to_pub.source = NODE_NAME;

                Command_to_pub.Reference_State = Traj_gen.Line_trajectory_generation(time_trajectory, init_pos_xy);
                if(_pause_){
                    Command_to_pub.Reference_State.velocity_ref[0] = 0.0;
                    Command_to_pub.Reference_State.velocity_ref[1] = 0.0;
                    Command_to_pub.Reference_State.velocity_ref[2] = 0.0;
                }else{
                    custom_controller_available = true;
                    time_trajectory = time_trajectory + 0.01;
                }

                mesg = "Trajectory tracking: " + 
                                        std::to_string(time_trajectory) + " / " +
                                        std::to_string(trajectory_total_time) + " [s] " + "\r";
                std::cout << mesg << std::flush;
                
                Draw_in_rviz(Command_to_pub.Reference_State, true);

                sleep(0.01);
            }else{
                key_now = U_KEY_H;
            }
            break;
    }

    switch (key_now)
    {
        case U_KEY_NONE:
            return;

        // If type in R: switch to OFFBOARD mode and Arming or Disarm
        case U_KEY_R:
            arm_disarm();

            return;

        // If type in T: Takeoff
        case U_KEY_T:
            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Takeoff Mode.");

            Command_to_pub.Mode = drone_msgs::ControlCommand::Takeoff;
            Command_to_pub.Reference_State.yaw_ref = 0.0;
            Command_to_pub.source = NODE_NAME;
            
            break;

        // If type in L: Landing
        case U_KEY_L:
            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Land Mode.");
        
            Command_to_pub.Mode = drone_msgs::ControlCommand::Land;
            Command_to_pub.source = NODE_NAME;
            
            break;

        // If type in Z: roll back
        case U_KEY_Z:
            Remote_Mode = -1;
            
        // If type in H: Hovering
        case U_KEY_H:
            if(key_last != key_now){
                pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Hold Mode.");

                Command_to_pub.Mode = drone_msgs::ControlCommand::Hold;
                Command_to_pub.source = NODE_NAME;
                Command_to_pub.Reference_State.position_ref[0]     = cur_pos[0];
                Command_to_pub.Reference_State.position_ref[1]     = cur_pos[1];
                Command_to_pub.Reference_State.position_ref[2]     = cur_pos[2];
                Command_to_pub.Reference_State.velocity_ref[0]     = 0;
                Command_to_pub.Reference_State.velocity_ref[1]     = 0;
                Command_to_pub.Reference_State.velocity_ref[2]     = 0;
                Command_to_pub.Reference_State.acceleration_ref[0] = 0;
                Command_to_pub.Reference_State.acceleration_ref[1] = 0;
                Command_to_pub.Reference_State.acceleration_ref[2] = 0;

            }
            
            break;

        // If type in SPACE: pause
        case U_KEY_SPACE:
            if(trajmove.find(key_last) != trajmove.end())
            {
                _pause_ = !_pause_;
                key_now = key_last;
                break;
            }else if(joymove.find(key_last) != joymove.end()){
                pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switch to Hold Mode.");

                Command_to_pub.Mode = drone_msgs::ControlCommand::Hold;
                Command_to_pub.source = NODE_NAME;
                Command_to_pub.Reference_State.position_ref[0]     = cur_pos[0];
                Command_to_pub.Reference_State.position_ref[1]     = cur_pos[1];
                Command_to_pub.Reference_State.position_ref[2]     = cur_pos[2];
                Command_to_pub.Reference_State.velocity_ref[0]     = 0;
                Command_to_pub.Reference_State.velocity_ref[1]     = 0;
                Command_to_pub.Reference_State.velocity_ref[2]     = 0;
                Command_to_pub.Reference_State.acceleration_ref[0] = 0;
                Command_to_pub.Reference_State.acceleration_ref[1] = 0;
                Command_to_pub.Reference_State.acceleration_ref[2] = 0;
                
            }
            
            break;
    }

    if(key_now != U_KEY_PASS)
        key_last = key_now;
    Command_to_pub.header.stamp = ros::Time::now();
    Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
    
    if(controller_switch){
        std_msgs::Bool command_active;
        command_active.data = custom_controller_available;
        command_active_pub.publish(command_active);
    }
    if(controller_switch && custom_controller_available){
        quadrotor_msgs::TrajectoryPoint pose_cmd;
        pose_cmd.time_from_start = ros::Duration(Command_to_pub.Reference_State.time_from_start);
        pose_cmd.pose.position.x = Command_to_pub.Reference_State.position_ref[0];
        pose_cmd.pose.position.y = Command_to_pub.Reference_State.position_ref[1];
        pose_cmd.pose.position.z = Command_to_pub.Reference_State.position_ref[2];
        pose_cmd.velocity.linear.x = Command_to_pub.Reference_State.velocity_ref[0];
        pose_cmd.velocity.linear.y = Command_to_pub.Reference_State.velocity_ref[1];
        pose_cmd.velocity.linear.z = Command_to_pub.Reference_State.velocity_ref[2];
        pose_cmd.heading = Command_to_pub.Reference_State.yaw_ref;

        command_trajPoint_pub.publish(pose_cmd);
    }else
        move_pub.publish(Command_to_pub);
    
}


void generate_com(int Move_mode, float state_desired[4])
{
    //# Move_mode 2-bit value:
    //# 0 for position, 1 for vel, 1st for xy, 2nd for z.
    //#                   xy position     xy velocity
    //# z position       	0b00(0)       0b10(2)
    //# z velocity		0b01(1)       0b11(3)

    if(Move_mode == drone_msgs::PositionReference::XYZ_ACC)
    {
        cout << "ACC control not support yet." <<endl;
    }

    if((Move_mode & 0b10) == 0) //xy channel
    {
        Command_to_pub.Reference_State.position_ref[0] = state_desired[0];
        Command_to_pub.Reference_State.position_ref[1] = state_desired[1];
        Command_to_pub.Reference_State.velocity_ref[0] = 0;
        Command_to_pub.Reference_State.velocity_ref[1] = 0;
    }
    else
    {
        Command_to_pub.Reference_State.position_ref[0] = 0;
        Command_to_pub.Reference_State.position_ref[1] = 0;
        Command_to_pub.Reference_State.velocity_ref[0] = state_desired[0];
        Command_to_pub.Reference_State.velocity_ref[1] = state_desired[1];
    }

    if((Move_mode & 0b01) == 0) //z channel
    {
        Command_to_pub.Reference_State.position_ref[2] = state_desired[2];
        Command_to_pub.Reference_State.velocity_ref[2] = 0;
    }
    else
    {
        Command_to_pub.Reference_State.position_ref[2] = 0;
        Command_to_pub.Reference_State.velocity_ref[2] = state_desired[2];
    }

    Command_to_pub.Reference_State.acceleration_ref[0] = 0;
    Command_to_pub.Reference_State.acceleration_ref[1] = 0;
    Command_to_pub.Reference_State.acceleration_ref[2] = 0;


    Command_to_pub.Reference_State.yaw_ref = state_desired[3]/180.0*M_PI;
}

void Draw_in_rviz(const drone_msgs::PositionReference& pos_ref, bool draw_trajectory)
{
    geometry_msgs::PoseStamped reference_pose;

    reference_pose.header.stamp = ros::Time::now();
    reference_pose.header.frame_id = "world";

    reference_pose.pose.position.x = pos_ref.position_ref[0];
    reference_pose.pose.position.y = pos_ref.position_ref[1];
    reference_pose.pose.position.z = pos_ref.position_ref[2];

    if(draw_trajectory)
    {
        posehistory_vector_.insert(posehistory_vector_.begin(), reference_pose);
        if(posehistory_vector_.size() > TRA_WINDOW){
            posehistory_vector_.pop_back();
        }
        
        nav_msgs::Path reference_trajectory;
        reference_trajectory.header.stamp = ros::Time::now();
        reference_trajectory.header.frame_id = "world";
        reference_trajectory.poses = posehistory_vector_;
        ref_trajectory_pub.publish(reference_trajectory);
    }else
    {
        posehistory_vector_.clear();
        
        nav_msgs::Path reference_trajectory;
        reference_trajectory.header.stamp = ros::Time::now();
        reference_trajectory.header.frame_id = "world";
        reference_trajectory.poses = posehistory_vector_;
        ref_trajectory_pub.publish(reference_trajectory);
    }
}
