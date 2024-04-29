#include <ros/ros.h>
#include <stdlib.h>
#include "message_utils.h"
#include "math_utils.h"

#include <drone_msgs/Message.h>
#include <drone_msgs/ControlCommand.h>
#include <drone_msgs/DroneState.h>
#include <drone_msgs/PositionReference.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/PositionTarget.h>
#include <geometry_msgs/Quaternion.h>

using namespace std;

float update_rate_Hz;
bool gimbal_enable;

drone_msgs::DroneState _DroneState;
drone_msgs::ControlCommand Command_Now;

Eigen::Quaterniond q_fcu_target;
Eigen::Vector3d euler_fcu_target;
Eigen::Vector3d bodyrate_fcu_target;
float Thrust_target;
//Target pos of the drone [from fcu]
Eigen::Vector3d pos_drone_fcu_target;
//Target vel of the drone [from fcu]
Eigen::Vector3d vel_drone_fcu_target;
//Target accel of the drone [from fcu]
Eigen::Vector3d accel_drone_fcu_target;

Eigen::Vector3d gimbal_att_deg;

void printf_info();
void printf_command_control(const drone_msgs::ControlCommand& Command_Now);
void prinft_drone_state(const drone_msgs::DroneState& _DroneState);

void att_target_cb(const mavros_msgs::AttitudeTarget::ConstPtr& msg)
{
    q_fcu_target = Eigen::Quaterniond(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

    //Transform the Quaternion to euler Angles
    euler_fcu_target = quaternion_to_euler(q_fcu_target);

    bodyrate_fcu_target = Eigen::Vector3d(msg->body_rate.x, msg->body_rate.y, msg->body_rate.z);

    Thrust_target = msg->thrust;
}
void pos_target_cb(const mavros_msgs::PositionTarget::ConstPtr& msg)
{
    pos_drone_fcu_target = Eigen::Vector3d(msg->position.x, msg->position.y, msg->position.z);

    vel_drone_fcu_target = Eigen::Vector3d(msg->velocity.x, msg->velocity.y, msg->velocity.z);

    accel_drone_fcu_target = Eigen::Vector3d(msg->acceleration_or_force.x, msg->acceleration_or_force.y, msg->acceleration_or_force.z);
}
void Command_cb(const drone_msgs::ControlCommand::ConstPtr& msg)
{
    Command_Now = *msg; // when stabilization: from dronestate, when movement: from controller
}
void drone_state_cb(const drone_msgs::DroneState::ConstPtr& msg)
{
    _DroneState = *msg;
}
void gimbal_att_cb(const geometry_msgs::Quaternion::ConstPtr& msg)
{
    Eigen::Quaterniond gimbal_att_quat;

    gimbal_att_quat = Eigen::Quaterniond(msg->w, msg->x, msg->y, msg->z);

    Eigen::Vector3d gimbal_att;
    //Transform the Quaternion to euler Angles
    gimbal_att = quaternion_to_euler(gimbal_att_quat);

    gimbal_att_deg = gimbal_att/M_PI*180;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ground_station");
    ros::NodeHandle nh("~");

    nh.param<float>("update_rate_Hz", update_rate_Hz, 1.0);
    nh.param<bool>("gimbal_enable", gimbal_enable, false);

    // [SUB] drone state
    ros::Subscriber drone_state_sub = nh.subscribe<drone_msgs::DroneState>("/drone_msg/drone_state", 10, drone_state_cb);
    ros::Subscriber Command_sub = nh.subscribe<drone_msgs::ControlCommand>("/drone_msg/control_command", 10, Command_cb);
    ros::Subscriber attitude_target_sub = nh.subscribe<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/target_attitude", 10, att_target_cb);
    ros::Subscriber position_target_sub = nh.subscribe<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/target_local", 10, pos_target_cb);

    ros::Subscriber gimbal_att_sub = nh.subscribe<geometry_msgs::Quaternion>("/mavros/mount_control/orientation", 10, gimbal_att_cb);

    ros::Rate rate(update_rate_Hz);

    while(ros::ok())
    {
        ros::spinOnce();
        system ("clear");
        printf_info();
        rate.sleep();
    }

    return 0;

}

void printf_info()
{
    cout <<"=======================================================================" <<endl;
    cout <<"===>>>>>>>>>>>>>>>>>>>>> Ground Station  <<<<<<<<<<<<<<<<<<<<<<<<<<<===" <<endl;
    cout <<"=======================================================================" <<endl;
    cout.setf(ios::fixed);
    cout<<setprecision(2);
    cout.setf(ios::left);
    cout.setf(ios::showpoint);
    cout.setf(ios::showpos);

    prinft_drone_state(_DroneState);

    printf_command_control(Command_Now);

    cout <<">>>>>>>>>>>>>>>>>>>>>>>> Setpoint from PX4 <<<<<<<<<<<<<<<<<<<<<<<<<<<<" <<endl;
    cout << "Pos_target [X Y Z] : [" << pos_drone_fcu_target[0] << ", "<< pos_drone_fcu_target[1]<<", "<<pos_drone_fcu_target[2]<<"] [ m ]"<<endl;
    cout << "Vel_target [X Y Z] : [" << vel_drone_fcu_target[0] << ", "<< vel_drone_fcu_target[1]<<", "<<vel_drone_fcu_target[2]<<"] "<<vel_drone_fcu_target.norm()<<"[m/s]"<<endl;
    cout << "Acc_target [X Y Z] : [" << accel_drone_fcu_target[0] << ", "<< accel_drone_fcu_target[1]<<", "<<accel_drone_fcu_target[2]<<"] "<<accel_drone_fcu_target.norm()<<"[m/s^2] "<<endl;
    cout << "Att_target [R P Y] : [" << euler_fcu_target[0] * 180/M_PI <<", "<<euler_fcu_target[1] * 180/M_PI << ", "<< euler_fcu_target[2] * 180/M_PI<<"] [deg]"<<endl;
    cout << "Bodyrate_target [R P Y] : [" << bodyrate_fcu_target[0] * 180/M_PI <<", "<<bodyrate_fcu_target[1] * 180/M_PI << ", "<< bodyrate_fcu_target[2] * 180/M_PI<<"] "<<bodyrate_fcu_target.norm()<<"[deg/s]"<<endl;
    cout << "Thr_target [ 0-1 ] : " << Thrust_target <<endl;

    if(Command_Now.Mode == drone_msgs::ControlCommand::Move)
    {
        //Only for TRAJECTORY tracking
        if(Command_Now.Reference_State.Move_mode == drone_msgs::PositionReference::XYZ_POS_VEL)
        {
            cout <<">>>>>>>>>>>>>>>>>>>>>>>> Tracking Error <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" <<endl;

            static Eigen::Vector3d tracking_error;
            tracking_error[0] = sqrt(pow(_DroneState.position[0] - Command_Now.Reference_State.position_ref[0],2)+
                        pow(_DroneState.position[1] - Command_Now.Reference_State.position_ref[1],2)+
                        pow(_DroneState.position[2] - Command_Now.Reference_State.position_ref[2],2));
            tracking_error[1] = sqrt(pow(_DroneState.velocity[0] - Command_Now.Reference_State.velocity_ref[0],2)+
                        pow(_DroneState.velocity[1] - Command_Now.Reference_State.velocity_ref[1],2)+
                        pow(_DroneState.velocity[2] - Command_Now.Reference_State.velocity_ref[2],2));
            tracking_error[2] = 0;

            cout << "Pos_error [m]:   " << tracking_error[0] <<endl;
            cout << "Vel_error [m/s]: " << tracking_error[1] <<endl;
        }
        
    }

    if(gimbal_enable)
    {
        cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>Gimbal State<<<<<<<<<<<<<<<<<<<<<<<<<<" <<endl;
        cout << "Gimbal_att    : [" << gimbal_att_deg[0] << ", "<< gimbal_att_deg[1] << ", "<< gimbal_att_deg[2] << "] [deg]"<<endl;
    }

}

void printf_command_control(const drone_msgs::ControlCommand& Command_Now)
{
    cout <<">>>>>>>>>>>>>>>>>>>>>>>> Control Command <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" <<endl;
    cout << "Source: [ "<< Command_Now.source << " ]  Command_ID: "<< Command_Now.Command_ID <<endl;
    switch(Command_Now.Mode){
        case drone_msgs::ControlCommand::Idle:
            if(Command_Now.Reference_State.yaw_ref == 999)
                cout << "Command: [ Idle + Arming + Switching to OFFBOARD mode ] " <<endl;
            else
                cout << "Command: [ Idle ] " <<endl;
            break;

        case drone_msgs::ControlCommand::Takeoff:
            cout << "Command: [ Takeoff ] " <<endl;
            break;

        case drone_msgs::ControlCommand::Hold:
            cout << "Command: [ Hold ] " <<endl;
            break;

        case drone_msgs::ControlCommand::Land:
            cout << "Command: [ Land ] " <<endl;
            break;

        case drone_msgs::ControlCommand::Move:
            switch(Command_Now.Reference_State.Move_mode){
                case drone_msgs::PositionReference::XYZ_POS:
                    cout << "Command: [ Move ] Move_mode: [ XYZ_POS ] " <<endl;
                    break;
                case drone_msgs::PositionReference::XY_POS_Z_VEL:  
                    cout << "Command: [ Move ] Move_mode: [ XY_POS_Z_VEL ] " <<endl;
                    break;
                case drone_msgs::PositionReference::XY_VEL_Z_POS:
                    cout << "Command: [ Move ] Move_mode: [ XY_VEL_Z_POS ] " <<endl;
                    break;
                case drone_msgs::PositionReference::XYZ_VEL:
                    cout << "Command: [ Move ] Move_mode: [ XYZ_VEL ] " <<endl;
                    break;
                case drone_msgs::PositionReference::XYZ_POS_VEL:
                    cout << "Command: [ Move ] Move_mode: [ XYZ_POS_VEL ] " <<endl;
                    break;
            }
            if(Command_Now.Reference_State.Move_frame == drone_msgs::PositionReference::ENU_FRAME)
                cout << "Move_frame: [ ENU_FRAME ] " <<endl;
            else if(Command_Now.Reference_State.Move_frame == drone_msgs::PositionReference::BODY_FRAME)
                cout << "Move_frame: [ BODY_FRAME ] " <<endl;

            cout << "Position [X Y Z] : [" << Command_Now.Reference_State.position_ref[0] << ", "<< Command_Now.Reference_State.position_ref[1]<<", "<< Command_Now.Reference_State.position_ref[2]<<"] [ m ]" <<endl;
            cout << "Velocity [X Y Z] : [" << Command_Now.Reference_State.velocity_ref[0] << ", "<< Command_Now.Reference_State.velocity_ref[1]<<", "<< Command_Now.Reference_State.velocity_ref[2]<<"] "<<Eigen::Vector3d(Command_Now.Reference_State.velocity_ref[0],Command_Now.Reference_State.velocity_ref[1],Command_Now.Reference_State.velocity_ref[2]).norm()<<"[m/s]"<<endl;
            cout << "Acceleration [X Y Z] [: " << Command_Now.Reference_State.acceleration_ref[0] << ", "<< Command_Now.Reference_State.acceleration_ref[1]<<", "<< Command_Now.Reference_State.acceleration_ref[2]<<"] "<<Eigen::Vector3d(Command_Now.Reference_State.acceleration_ref[0],Command_Now.Reference_State.acceleration_ref[1],Command_Now.Reference_State.acceleration_ref[2]).norm()<<"[m/s^2]"<<endl;
            cout << "Yaw : "  << Command_Now.Reference_State.yaw_ref* 180/M_PI << " [deg] " <<endl;

            break;

        case drone_msgs::ControlCommand::Disarm:
            cout << "Command: [ Disarm ] " <<endl;
            break;

        case drone_msgs::ControlCommand::User_Mode1:
            cout << "Command: [ User_Mode1 ] " <<endl;
            break;
        
        case drone_msgs::ControlCommand::User_Mode2:
            cout << "Command: [ User_Mode2 ] " <<endl;
            break;
    }
}

void prinft_drone_state(const drone_msgs::DroneState& _DroneState)
{
    cout <<">>>>>>>>>>>>>>>>>>>>>>>>   Drone State   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" <<endl;

    cout << "Time: " << _DroneState.time_from_start <<" [s] ";

    if (_DroneState.connected == true)
        cout << " [ Connected ]";
    else
        cout << " [ Unconnected ]";

    if (_DroneState.armed == true)
        cout << " [ Armed ]";
    else
        cout << " [ DisArmed ]";

    if (_DroneState.landed == true)
        cout << " [ Ground ] ";
    else
        cout << " [ Air ] ";

    cout << "[ " << _DroneState.mode<<" ] " <<endl;

    cout << "Position [X Y Z] : [" << _DroneState.position[0] << ", "<< _DroneState.position[1]<<", "<<_DroneState.position[2]<<"] [ m ]"<<endl;
    cout << "Velocity [X Y Z] : [" << _DroneState.velocity[0] << ", "<< _DroneState.velocity[1]<<", "<<_DroneState.velocity[2]<<"] "<<Eigen::Vector3d(_DroneState.velocity[0],_DroneState.velocity[1],_DroneState.velocity[2]).norm()<<"[m/s]"<<endl;
    cout << "Attitude [R P Y] : [" << _DroneState.attitude[0] * 180/M_PI <<", "<<_DroneState.attitude[1] * 180/M_PI << ", "<< _DroneState.attitude[2] * 180/M_PI<<"] [deg]"<<endl;
    cout << "Att_rate [R P Y] : [" << _DroneState.attitude_rate[0] * 180/M_PI <<", "<<_DroneState.attitude_rate[1] * 180/M_PI << ", "<< _DroneState.attitude_rate[2] * 180/M_PI<<"] "<<Eigen::Vector3d(_DroneState.attitude_rate[0],_DroneState.attitude_rate[1],_DroneState.attitude_rate[2]).norm() * 180/M_PI<<"[deg/s]"<<endl;
}

