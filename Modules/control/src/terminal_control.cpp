#include <ros/ros.h>
#include <iostream>

#include <drone_msgs/ControlCommand.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <drone_msgs/DroneState.h>

#include "trajectory_generation.h"
#include "KeyboardEvent.h"

#define VEL_XY_STEP_SIZE 0.1
#define VEL_Z_STEP_SIZE 0.1
#define YAW_STEP_SIZE 0.08
#define TRA_WINDOW 2000
#define NODE_NAME "terminal_control"

using namespace std;

drone_msgs::ControlCommand Command_to_pub;

std::vector<geometry_msgs::PoseStamped> posehistory_vector_;


float time_trajectory = 0.0;
// 轨迹追踪总时长，键盘控制时固定时长，指令输入控制可调
float trajectory_total_time = 50.0;
float cur_pos[3];

ros::Subscriber velocity_sub;
ros::Publisher move_pub;
ros::Publisher ref_trajectory_pub;

void odom_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    cur_pos[0] = msg->pose.pose.position.x;
    cur_pos[1] = msg->pose.pose.position.y;
    cur_pos[2] = msg->pose.pose.position.z;
}

void mainloop1();
void mainloop2();
void generate_com(int Move_mode, float state_desired[4]);
void Draw_in_rviz(const drone_msgs::PositionReference& pos_ref, bool draw_trajectory);
void timerCallback(const ros::TimerEvent& e)
{
    cout << ">>>>>>>>>>>>>>>> Welcome to use Prometheus Terminal Control <<<<<<<<<<<<<<<<"<< endl;
    cout << "ENTER key to control the drone: " <<endl;
    cout << "1 for Arm, Space for Takeoff, L for Land, H for Hold, 0 for Disarm, 8/9 for Trajectory tracking" <<endl;
    cout << "Move mode is fixed (XYZ_VEL,BODY_FRAME): w/s for body_x, a/d for body_y, k/m for z, q/e for body_yaw" <<endl;
    cout << "CTRL-C to quit." <<endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "terminal_control");
    ros::NodeHandle nh;

    cout.setf(ios::fixed);
    cout<<setprecision(2);
    cout.setf(ios::left);
    cout.setf(ios::showpoint);

    velocity_sub=nh.subscribe<nav_msgs::Odometry>("/drone_msg/drone_odom", 100, odom_cb);

    // [PUB] control command
    move_pub = nh.advertise<drone_msgs::ControlCommand>("/drone_msg/control_command", 10);

    // [PUB] Rviz trajectory
    ref_trajectory_pub = nh.advertise<nav_msgs::Path>("/drone_msg/reference_trajectory", 10);

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
    int Remote_Mode;
    cout << ">>>>>>>>>>>>>>>> Welcome to use Prometheus Terminal Control <<<<<<<<<<<<<<<<"<< endl;
    cout << "Please choose the Remote Mode: 0 for command input control, 1 for keyboard input control"<<endl;
    cin >> Remote_Mode;

    if (Remote_Mode == 0)
    {
        cout << "Command input control mode"<<endl;
        mainloop1();
    }else if(Remote_Mode == 1)
    {
        ros::Timer timer = nh.createTimer(ros::Duration(30.0), timerCallback);
        cout << "Keyboard input control mode"<<endl;
        mainloop2();
    }
}

void mainloop1()
{
    Traj_gen Traj_gen;
    Traj_gen.printf_param();
    int Control_Mode = 0;
    int Move_mode = 0;
    int Move_frame = 0;
    int Trjectory_mode = 0;
    float state_desired[4];

    while(ros::ok())
    {
        // Waiting for input
        cout << ">>>>>>>>>>>>>>>> Welcome to use Prometheus Terminal Control <<<<<<<<<<<<<<<<"<< endl;
        cout << "Please choose the Command.Mode: 0 for Idle, 1 for Takeoff, 2 for Hold, 3 for Land, 4 for Move, 5 for Disarm, 6 for User_Mode1, 7 for User_Mode2"<<endl;
        cout << "Input 999 to switch to offboard mode and arm the drone (ONLY for simulation, please use RC in experiment!!!)"<<endl;
        cin  >> Control_Mode;

        if(Control_Mode == drone_msgs::ControlCommand::Move)
        {
            cout << "Please choose the Command.Reference_State.Move_mode: 0 for XYZ_POS, 1 for XY_POS_Z_VEL, 2 for XY_VEL_Z_POS, 3 for XYZ_VEL, 5 for TRAJECTORY"<<endl;
            cin >> Move_mode;

            if(Move_mode == drone_msgs::PositionReference::TRAJECTORY)
            {
                cout << "For safety, please move the drone near to the trajectory start point firstly!!!"<<endl;
                cout << "Please choose the trajectory type: 0 for Circle, 1 for Eight Shape, 2 for Step, 3 for Line"<<endl;
                cin >> Trjectory_mode;  
                cout << "Input the trajectory_total_time:"<<endl;
                cin >> trajectory_total_time;
            }else
            {
                cout << "Please choose the Command.Reference_State.Move_frame: 0 for ENU_FRAME, 1 for BODY_FRAME"<<endl;
                cin >> Move_frame; 
                cout << "Please input the reference state [x y z yaw]: "<< endl;
                cout << "setpoint_t[0] --- x [m] : "<< endl;
                cin >> state_desired[0];
                cout << "setpoint_t[1] --- y [m] : "<< endl;
                cin >> state_desired[1];
                cout << "setpoint_t[2] --- z [m] : "<< endl;
                cin >> state_desired[2];
                cout << "setpoint_t[3] --- yaw [du] : "<< endl;
                cin >> state_desired[3];
            }
        }else if(Control_Mode == 999)
        {
            Command_to_pub.header.stamp = ros::Time::now();
            Command_to_pub.Mode = drone_msgs::ControlCommand::Idle;
            Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
            Command_to_pub.source = NODE_NAME;
            Command_to_pub.Reference_State.yaw_ref = 999;
            move_pub.publish(Command_to_pub);
            Command_to_pub.Reference_State.yaw_ref = 0.0;
        }

        switch (Control_Mode)
        {
            case drone_msgs::ControlCommand::Idle:
                Command_to_pub.header.stamp = ros::Time::now();
                Command_to_pub.Mode = drone_msgs::ControlCommand::Idle;
                Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                Command_to_pub.source = NODE_NAME;
                move_pub.publish(Command_to_pub);
                break;

            case drone_msgs::ControlCommand::Takeoff:
                Command_to_pub.header.stamp = ros::Time::now();
                Command_to_pub.Mode = drone_msgs::ControlCommand::Takeoff;
                Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                Command_to_pub.source = NODE_NAME;
                move_pub.publish(Command_to_pub);
                break;

            case drone_msgs::ControlCommand::Land:
                Command_to_pub.header.stamp = ros::Time::now();
                Command_to_pub.Mode = drone_msgs::ControlCommand::Land;
                Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                Command_to_pub.source = NODE_NAME;
                move_pub.publish(Command_to_pub);
                break;

            case drone_msgs::ControlCommand::Move:
                if(Move_mode == drone_msgs::PositionReference::TRAJECTORY)
                {
                     time_trajectory = 0.0;

                    while(time_trajectory < trajectory_total_time)
                    {
                        Command_to_pub.header.stamp = ros::Time::now();
                        Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
                        Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                        Command_to_pub.source = NODE_NAME;

                        if(Trjectory_mode == 0)
                        {
                            Command_to_pub.Reference_State = Traj_gen.Circle_trajectory_generation(time_trajectory);
                        }else if(Trjectory_mode == 1)
                        {
                            Command_to_pub.Reference_State = Traj_gen.Eight_trajectory_generation(time_trajectory);
                        }else if(Trjectory_mode == 2)
                        {
                            Command_to_pub.Reference_State = Traj_gen.Step_trajectory_generation(time_trajectory);
                        }else if(Trjectory_mode == 3)
                        {
                            Command_to_pub.Reference_State = Traj_gen.Line_trajectory_generation(time_trajectory);
                        }

                        if(time_trajectory<0.05)
                        {
                            ros::spinOnce();
                            // std::cout<<cur_pos[0]<<' '<<cur_pos[1]<<' '<<cur_pos[2]<<std::endl;
                            float dis=0.0, dif[3];
                            for(int i=0;i<3;i++)
                            {
                                dif[i] = Command_to_pub.Reference_State.position_ref[i]-cur_pos[i];
                                dis += dif[i]*dif[i];
                            }
                            dis = sqrt(dis);
                            // std::cout<<dif[0]<<' '<<dif[1]<<' '<<dif[2]<<std::endl;
                            if(dis > 0.1)
                            {
                                Command_to_pub.Reference_State.Move_mode  = drone_msgs::PositionReference::XYZ_VEL;
                                Command_to_pub.Reference_State.Move_frame = drone_msgs::PositionReference::ENU_FRAME;
                                Command_to_pub.Reference_State.time_from_start = 0.0;
                                Command_to_pub.Reference_State.velocity_ref[0] = dif[0]/dis;
                                Command_to_pub.Reference_State.velocity_ref[1] = dif[1]/dis;
                                Command_to_pub.Reference_State.velocity_ref[2] = dif[2]/dis;
                                Command_to_pub.Reference_State.yaw_ref = 0.0;
                                move_pub.publish(Command_to_pub);
                                continue;
                            }
                            time_trajectory = time_trajectory + 0.01;
                        }

                        move_pub.publish(Command_to_pub);
                        time_trajectory = time_trajectory + 0.01;

                        cout << "Trajectory tracking: "<< time_trajectory << " / " << trajectory_total_time  << " [ s ]" <<endl;

                        Draw_in_rviz(Command_to_pub.Reference_State, true);

                        ros::Duration(0.01).sleep();
                    }
                    
                    Command_to_pub.header.stamp = ros::Time::now();
		            Command_to_pub.Mode = drone_msgs::ControlCommand::Hold;
		            Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
		            move_pub.publish(Command_to_pub);
		            
                }else
                {
                    Command_to_pub.header.stamp = ros::Time::now();
                    Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
                    Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                    Command_to_pub.source = NODE_NAME;
                    Command_to_pub.Reference_State.Move_mode  = Move_mode;
                    Command_to_pub.Reference_State.Move_frame = Move_frame;
                    Command_to_pub.Reference_State.time_from_start = 0.0;
                    generate_com(Move_mode, state_desired);

                    move_pub.publish(Command_to_pub);
                }
                break;
            
            case drone_msgs::ControlCommand::Hold:
                Command_to_pub.header.stamp = ros::Time::now();
                Command_to_pub.Mode = drone_msgs::ControlCommand::Hold;
                Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                Command_to_pub.source = NODE_NAME;
                move_pub.publish(Command_to_pub);
                break;
    
            case drone_msgs::ControlCommand::Disarm:
                Command_to_pub.header.stamp = ros::Time::now();
                Command_to_pub.Mode = drone_msgs::ControlCommand::Disarm;
                Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                Command_to_pub.source = NODE_NAME;
                move_pub.publish(Command_to_pub);
                break;

            case drone_msgs::ControlCommand::User_Mode1:
                Command_to_pub.header.stamp = ros::Time::now();
                Command_to_pub.Mode = drone_msgs::ControlCommand::User_Mode1;
                Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                Command_to_pub.source = NODE_NAME;
                move_pub.publish(Command_to_pub);
                break;
            
            case drone_msgs::ControlCommand::User_Mode2:
                Command_to_pub.header.stamp = ros::Time::now();
                Command_to_pub.Mode = drone_msgs::ControlCommand::User_Mode2;
                Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
                Command_to_pub.source = NODE_NAME;
                move_pub.publish(Command_to_pub);
                break;
        }
        
        cout << "....................................................." <<endl;
        
        sleep(1.0);
    }
}

void mainloop2()
{
    Traj_gen Traj_gen;
    Traj_gen.printf_param();
    KeyboardEvent keyboardcontrol;

    char key_now;
    char key_last;

    cout << ">>>>>>>>>>>>>>>> Welcome to use Prometheus Terminal Control <<<<<<<<<<<<<<<<"<< endl;
    cout << "ENTER key to control the drone: " <<endl;
    cout << "1 for Arm, Space for Takeoff, L for Land, H for Hold, 0 for Disarm, 8/9 for Trajectory tracking" <<endl;
    cout << "Move mode is fixed (XYZ_VEL,BODY_FRAME): w/s for body_x, a/d for body_y, k/m for z, q/e for body_yaw" <<endl;
    cout << "CTRL-C to quit." <<endl;

    while (ros::ok())
    {
        keyboardcontrol.RosWhileLoopRun();
        key_now = keyboardcontrol.GetPressedKey();
        switch (key_now)
        {
        case U_KEY_NONE:
          sleep(0.5);
          break;

        // If type in 1: switch to OFFBOARD mode and arming
        case U_KEY_1:
          cout << "Arming and Switch to OFFBOARD." <<endl;
      
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Idle;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.yaw_ref = 999;
          move_pub.publish(Command_to_pub);
          sleep(1.0);
          break;

        // If type in SPACE: Takeoff
        case U_KEY_SPACE:
          cout << "Switch to Takeoff Mode." <<endl;

          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Takeoff;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.Reference_State.yaw_ref = 0.0;
          Command_to_pub.source = NODE_NAME;
          move_pub.publish(Command_to_pub);

          sleep(1.0);

          break;

        // If type in L: Landing
        case U_KEY_L:
          cout << "Switch to Land Mode." <<endl;
      
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Land;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          move_pub.publish(Command_to_pub);

          break;

        // If type in 0: Disarm
        case U_KEY_0:
          cout << "Switch to Disarm Mode." <<endl;
      
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Disarm;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          move_pub.publish(Command_to_pub);

          break;

        // If type in H: Loiter
        case U_KEY_H:
          cout << "Switch to Hold Mode." <<endl;

          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Hold;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.position_ref[0]     = 0;
          Command_to_pub.Reference_State.position_ref[1]     = 0;
          Command_to_pub.Reference_State.position_ref[2]     = 0;
          Command_to_pub.Reference_State.velocity_ref[0]     = 0;
          Command_to_pub.Reference_State.velocity_ref[1]     = 0;
          Command_to_pub.Reference_State.velocity_ref[2]     = 0;
          Command_to_pub.Reference_State.acceleration_ref[0] = 0;
          Command_to_pub.Reference_State.acceleration_ref[1] = 0;
          Command_to_pub.Reference_State.acceleration_ref[2] = 0;
          move_pub.publish(Command_to_pub);

          sleep(1.0);
        
          break;

        // If type in W: flying forward
        case U_KEY_W:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.velocity_ref[0]     += VEL_XY_STEP_SIZE;
          move_pub.publish(Command_to_pub);
          
          cout << "Current Velocity [X Y Z]: " << Command_to_pub.Reference_State.velocity_ref[0] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[1] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[2] << " [m/s] "<<endl;

          sleep(0.1);
        
          break;
        
        // If type in S: flying backward
        case U_KEY_S:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.velocity_ref[0]     -= VEL_XY_STEP_SIZE;
          move_pub.publish(Command_to_pub);

          cout << "Current Velocity [X Y Z]: " << Command_to_pub.Reference_State.velocity_ref[0] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[1] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[2] << " [m/s] "<<endl;

          sleep(0.1);

          break;

        // If type in A: flying left
        case U_KEY_A:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.velocity_ref[1]     += VEL_XY_STEP_SIZE;
          move_pub.publish(Command_to_pub);
        
          cout << "Current Velocity [X Y Z]: " << Command_to_pub.Reference_State.velocity_ref[0] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[1] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[2] << " [m/s] "<<endl;

          sleep(0.1);

          break;

        // If type in D: flying right
        case U_KEY_D:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.velocity_ref[1]     -= VEL_XY_STEP_SIZE;
          move_pub.publish(Command_to_pub);

          cout << "Current Velocity [X Y Z]: " << Command_to_pub.Reference_State.velocity_ref[0] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[1] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[2] << " [m/s] "<<endl;

          sleep(0.1);

          break;

        // If type in K: flying upward
        case U_KEY_K:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.velocity_ref[2]     += VEL_Z_STEP_SIZE;
          move_pub.publish(Command_to_pub);

          cout << "Current Velocity [X Y Z]: " << Command_to_pub.Reference_State.velocity_ref[0] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[1] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[2] << " [m/s] "<<endl;

          sleep(0.1);

          break;

        // If type in M: flying downward
        case U_KEY_M:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.velocity_ref[2]     -= VEL_Z_STEP_SIZE;
          move_pub.publish(Command_to_pub);

          cout << "Current Velocity [X Y Z]: " << Command_to_pub.Reference_State.velocity_ref[0] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[1] << " [m/s] " << Command_to_pub.Reference_State.velocity_ref[2] << " [m/s] "<<endl;

          sleep(0.1);
        
          break;

        // If type in Q: turn left 
        case U_KEY_Q:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_VEL;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.yaw_ref             = YAW_STEP_SIZE;
          move_pub.publish(Command_to_pub);
          
          cout << "Increase the Yaw angle." <<endl;

          sleep(0.1);
        
          break;

        // If type in E: turn right
        case U_KEY_E:
          Command_to_pub.header.stamp = ros::Time::now();
          Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
          Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
          Command_to_pub.source = NODE_NAME;
          Command_to_pub.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_POS;
          Command_to_pub.Reference_State.Move_frame      = drone_msgs::PositionReference::BODY_FRAME;
          Command_to_pub.Reference_State.yaw_ref             = YAW_STEP_SIZE;
          move_pub.publish(Command_to_pub);
          
          cout << "Decrease the Yaw angle." <<endl;

          sleep(0.1);
        
          break;
        
        // If type in 0: tracking a circle traj 
        case U_KEY_9:
          time_trajectory = 0.0;
          trajectory_total_time = 50.0;
          // 需要设置
          while(time_trajectory < trajectory_total_time)
          {
              Command_to_pub.header.stamp = ros::Time::now();
              Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
              Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
              Command_to_pub.source = NODE_NAME;

              Command_to_pub.Reference_State = Traj_gen.Circle_trajectory_generation(time_trajectory);

              move_pub.publish(Command_to_pub);
              time_trajectory = time_trajectory + 0.01;

              cout << "Trajectory tracking: "<< time_trajectory << " / " << trajectory_total_time  << " [ s ]" <<endl;

              Draw_in_rviz(Command_to_pub.Reference_State, true);

              ros::Duration(0.01).sleep();
          }
          break;

        // If type in 8: tracking a figure_of_8 traj 
        case U_KEY_8:
          time_trajectory = 0.0;
          trajectory_total_time = 50.0;
          // 需要设置
          while(time_trajectory < trajectory_total_time)
          {
              Command_to_pub.header.stamp = ros::Time::now();
              Command_to_pub.Mode = drone_msgs::ControlCommand::Move;
              Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
              Command_to_pub.source = NODE_NAME;

              Command_to_pub.Reference_State = Traj_gen.Eight_trajectory_generation(time_trajectory);

              move_pub.publish(Command_to_pub);
              time_trajectory = time_trajectory + 0.01;

              cout << "Trajectory tracking: "<< time_trajectory << " / " << trajectory_total_time  << " [ s ]" <<endl;

              Draw_in_rviz(Command_to_pub.Reference_State, true);

              ros::Duration(0.01).sleep();
          }
          break;
        }

        key_last = key_now;
        ros::spinOnce();
        sleep(0.1);
    }
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
