#include <ros/ros.h>
#include <map>
#include <geometry_msgs/PoseStamped.h>
#include <drone_msgs/DroneState.h>
#include <drone_msgs/ControlCommand.h>
#include <drone_msgs/Message.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_common/geometry_eigen_conversions.h>
#include "command_to_mavros.h"
#include "message_utils.h"
#define NODE_NAME "px4_commander"

using namespace std;
map<string,int> command;
float cur_time;
float dt = 0;
float Takeoff_height;
Eigen::Vector4d Takeoff_position(0,0,0,0);
float Disarm_height;
float Land_speed;
float Command_rate;
bool _DroneState_updated;
// Geo Fence
Eigen::Vector2f geo_fence_x;
Eigen::Vector2f geo_fence_y;
Eigen::Vector2f geo_fence_z;

Eigen::Vector3d state_sp(0,0,0);              // state to be pub
Eigen::Vector3d state_vel_sp(0,0,0);
double yaw_rate_sp;
double yaw_sp;
Eigen::Vector3d ref_pos(0,0,0);               // GeoFence
Eigen::Vector3d hold_pos(0,0,0);              // loiter
double hold_yaw;

drone_msgs::DroneState _DroneState;
geometry_msgs::PoseStamped ref_pose_rviz;
drone_msgs::ControlCommand Command_Now;
drone_msgs::ControlCommand Command_Last;

ros::Publisher message_pub;

void printf_param(){
    cout <<">>>>>>>>>>>>>>>>>>>>>>>> px4_commander Parameter <<<<<<<<<<<<<<<<<<<<<<" <<endl;
    cout << "Takeoff_height   : "<< Takeoff_height<<" [m] "<<endl;
    cout << "Disarm_height    : "<< Disarm_height <<" [m] "<<endl;
    cout << "Land_speed       : "<< Land_speed <<" [m/s] "<<endl;
    cout << "geo_fence_x : "<< geo_fence_x[0] << " [m]  to  "<<geo_fence_x[1] << " [m]"<< endl;
    cout << "geo_fence_y : "<< geo_fence_y[0] << " [m]  to  "<<geo_fence_y[1] << " [m]"<< endl;
    cout << "geo_fence_z : "<< geo_fence_z[0] << " [m]  to  "<<geo_fence_z[1] << " [m]"<< endl;
}
int check_failsafe(){
    if (ref_pos[0] < geo_fence_x[0] || ref_pos[0] > geo_fence_x[1] ||
        ref_pos[1] < geo_fence_y[0] || ref_pos[1] > geo_fence_y[1] ||
        ref_pos[2] < geo_fence_z[0] || ref_pos[2] > geo_fence_z[1]){
        pub_message(message_pub, drone_msgs::Message::ERROR, NODE_NAME, "Out of the geo fence, the drone is holding...");
        return 1;
    }else{
		hold_pos[0] = max(geo_fence_x[0]+0.05f,min(geo_fence_x[1]-0.05f,_DroneState.position[0]));
		hold_pos[1] = max(geo_fence_y[0]+0.05f,min(geo_fence_y[1]-0.05f,_DroneState.position[1]));
		hold_pos[2] = max(geo_fence_z[0]+0.05f,min(geo_fence_z[1]-0.05f,_DroneState.position[2]));
		hold_yaw    = _DroneState.attitude[2];
        return 0;
    }
}
float get_time_in_sec(const ros::Time& begin_time){
    ros::Time time_now = ros::Time::now();
    float currTimeSec = time_now.sec - begin_time.sec;
    float currTimenSec = time_now.nsec / 1e9 - begin_time.nsec / 1e9;
    return (currTimeSec + currTimenSec);
}

geometry_msgs::PoseStamped get_rviz_ref_posistion(const drone_msgs::ControlCommand& cmd);

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Callback Function <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void Command_cb(const drone_msgs::ControlCommand::ConstPtr& msg)
{
    if(command.find(msg->source)==command.end()){
        command[msg->source] = msg->Command_ID;
        Command_Now = *msg;
    }else if(msg->Command_ID > command[Command_Now.source]){
        Command_Now = *msg;
    }else{
        pub_message(message_pub, drone_msgs::Message::WARN, msg->source, "Wrong Command ID.");
    }
    // if( msg->Command_ID  >  Command_Now.Command_ID )
    // {
    //     Command_Now = *msg;
    // }else
    // {
    //     pub_message(message_pub, drone_msgs::Message::WARN, msg->source, "Wrong Command ID.");
    // }
}
void station_command_cb(const drone_msgs::ControlCommand::ConstPtr& msg)
{
    Command_Now = *msg;
    pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Get a command from Ground Station.");
    
}
void drone_state_cb(const drone_msgs::DroneState::ConstPtr& msg)
{
    _DroneState = *msg;
    _DroneState_updated = true;
    if(!_DroneState.armed) Command_Now.Command_ID = 0;
}
void timerCallback(const ros::TimerEvent& e)
{
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "px4_commander");
    ros::NodeHandle nh("~");

    string uav_name;
    nh.param<string>("uav_name", uav_name, "");
    if(uav_name!="") cout<<"[perception]: uav_name: "<<uav_name<<endl;
    // [SUB] control commands from planner
    ros::Subscriber Command_sub = nh.subscribe<drone_msgs::ControlCommand>(uav_name+"/drone_msg/control_command", 10, Command_cb);

    // [SUB] control commands from ground station
    ros::Subscriber station_command_sub = nh.subscribe<drone_msgs::ControlCommand>(uav_name+"/drone_msg/control_command_station", 10, station_command_cb);
    
    // [SUB] drone state
    ros::Subscriber drone_state_sub = nh.subscribe<drone_msgs::DroneState>(uav_name+"/drone_msg/drone_state", 10, drone_state_cb);
    
    // [PUB] rviz position
    ros::Publisher rivz_ref_pose_pub = nh.advertise<geometry_msgs::PoseStamped>(uav_name+"/drone_msg/control/ref_pose_rviz", 10);

    // [PUB] ground station messages
    message_pub = nh.advertise<drone_msgs::Message>(uav_name+"/drone_msg/message", 10);

    // custome timer callback
    ros::Timer timer = nh.createTimer(ros::Duration(10.0), timerCallback);

    // Param
    nh.param<float>("Takeoff_height", Takeoff_height, 1.5);      // takeoff height default
    nh.param<float>("Disarm_height", Disarm_height, 0.15);       // landing height default
    nh.param<float>("Land_speed", Land_speed, 0.2);              // landing speed
    nh.param<float>("Command_rate", Command_rate, 20.0);         // pub rate
    Command_rate = max(10.0f,min(50.0f,Command_rate));
    // GeoFence
    nh.param<float>("geo_fence/x_min", geo_fence_x[0], -100.0);
    nh.param<float>("geo_fence/x_max", geo_fence_x[1], 100.0);
    nh.param<float>("geo_fence/y_min", geo_fence_y[0], -100.0);
    nh.param<float>("geo_fence/y_max", geo_fence_y[1], 100.0);
    nh.param<float>("geo_fence/z_min", geo_fence_z[0], -100.0);
    nh.param<float>("geo_fence/z_max", geo_fence_z[1], 100.0);


    ros::Rate rate(Command_rate); // 10hz ~ 50hz

    // publish commands via mavros to autopilot
    command_to_mavros _command_to_mavros(uav_name, nh);

    printf_param();
    
    // Initialisation
    Command_Now.Mode           = drone_msgs::ControlCommand::Idle;
    Command_Now.Command_ID     = 0;

    ros::Time begin_time = ros::Time::now();
    float last_time = get_time_in_sec(begin_time);

    while(ros::ok())
    {
        cur_time = get_time_in_sec(begin_time);
        dt = cur_time  - last_time;
        // dt = max(0.02f,min(0.1f,dt)); // 10hz ~ 50hz
        last_time = cur_time;

        ros::spinOnce();

        // Check for geo fence: If drone is out of the geo fence.
        if(check_failsafe() == 1)
        {
            Command_Now.Mode = drone_msgs::ControlCommand::Hold;	
        }


        switch (Command_Now.Mode){
            // ================================= Disarm =================================
            case drone_msgs::ControlCommand::Disarm:
                if(_DroneState.armed)
                {
                    _command_to_mavros.arm_cmd.request.value = false;
                    _command_to_mavros.arming_client.call(_command_to_mavros.arm_cmd);
                }else{
                    // _command_to_mavros.idle();
                    if(_DroneState.mode == "OFFBOARD")
                    {
                        _command_to_mavros.mode_cmd.request.custom_mode = "ALTCTL";
                        _command_to_mavros.set_mode_client.call(_command_to_mavros.mode_cmd);
                    }
                }
                break;

            // ================================ Arming =================================
            case drone_msgs::ControlCommand::Idle:
                
                _command_to_mavros.idle(); 
            
                // switch to offboard mode and arming, if yaw_ref = 999
                if(Command_Now.Reference_State.yaw_ref == 999)
                {            	
                    if(_DroneState.mode != "OFFBOARD")
                    {
                        _command_to_mavros.mode_cmd.request.custom_mode = "OFFBOARD";
                        if(_command_to_mavros.set_mode_client.call(_command_to_mavros.mode_cmd) &&
                            _command_to_mavros.mode_cmd.response.mode_sent){
                            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Switching to OFFBOARD Mode");
                        }
                        ros::Duration(0.1).sleep();
                    }else{
                        _command_to_mavros.arm_cmd.request.value = true;
                        if(!_DroneState.armed &&
                            _command_to_mavros.arming_client.call(_command_to_mavros.arm_cmd) &&
                            _command_to_mavros.arm_cmd.response.success)
                        {
                            pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Arming...");
                        }
                        ros::Duration(0.1).sleep();
                    }
                }

                if(_DroneState_updated)
                    Takeoff_position << _DroneState.position[0], _DroneState.position[1], _DroneState.position[2], _DroneState.attitude[2];
                break;

            // =============================== Takeoff ================================
            // keep position xy and yaw   
            case drone_msgs::ControlCommand::Takeoff:
                if (Command_Last.Mode != drone_msgs::ControlCommand::Takeoff)
                {
                    pub_message(message_pub, drone_msgs::Message::NORMAL, NODE_NAME, "Takeoff at the set height.");
                    Command_Now.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_POS;
                    Command_Now.Reference_State.Move_frame      = drone_msgs::PositionReference::ENU_FRAME;
                    Command_Now.Reference_State.position_ref[0] = Takeoff_position[0];
                    Command_Now.Reference_State.position_ref[1] = Takeoff_position[1];
                    Command_Now.Reference_State.position_ref[2] = Takeoff_position[2] + Takeoff_height;
                    Command_Now.Reference_State.yaw_ref         = Takeoff_position[3];
                
                    state_sp = Eigen::Vector3d(Takeoff_position[0],Takeoff_position[1],Takeoff_position[2] + Takeoff_height);
                    yaw_sp = Takeoff_position[3]; //rad
                    ref_pos = state_sp;

                    if(fabs(_DroneState.position[2] - Takeoff_height)<0.1)
                    {
                        Command_Now.Mode = drone_msgs::ControlCommand::Hold;
                    }
                }
                _command_to_mavros.send_pos_setpoint(state_sp, yaw_sp);

                break;

            // ================================= Hold =================================
            case drone_msgs::ControlCommand::Hold:
                if (Command_Last.Mode != drone_msgs::ControlCommand::Hold && !_DroneState.landed)
                {
                    Command_Now.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_POS;
                    Command_Now.Reference_State.Move_frame      = drone_msgs::PositionReference::ENU_FRAME;
                    Command_Now.Reference_State.position_ref[0] = hold_pos[0];
                    Command_Now.Reference_State.position_ref[1] = hold_pos[1];
                    Command_Now.Reference_State.position_ref[2] = hold_pos[2];
                    Command_Now.Reference_State.yaw_ref         = hold_yaw; //rad

                    state_sp = Eigen::Vector3d(hold_pos[0],hold_pos[1],hold_pos[2]);
                    yaw_sp = hold_yaw; //rad
                    ref_pos = hold_pos;
                }
                _command_to_mavros.send_pos_setpoint(state_sp, yaw_sp);

                break;

            // ================================= Land =================================
            case drone_msgs::ControlCommand::Land:
                if(!_DroneState.landed)
                {
                    if (Command_Last.Mode != drone_msgs::ControlCommand::Land)
                    {
                        Command_Now.Reference_State.Move_mode       = drone_msgs::PositionReference::XY_POS_Z_VEL;
                        Command_Now.Reference_State.Move_frame      = drone_msgs::PositionReference::ENU_FRAME;
                        Command_Now.Reference_State.position_ref[0] = _DroneState.position[0];
                        Command_Now.Reference_State.position_ref[1] = _DroneState.position[1];
                        Command_Now.Reference_State.yaw_ref         = _DroneState.attitude[2]; //rad
                    }
                    if(_DroneState.position[2] > Disarm_height)
                    {
                        Command_Now.Reference_State.position_ref[2] = _DroneState.position[2] - Land_speed * dt ;
                        Command_Now.Reference_State.velocity_ref[0] = 0.0;
                        Command_Now.Reference_State.velocity_ref[1] =  0.0;
                        Command_Now.Reference_State.velocity_ref[2] = - Land_speed; //Land_speed

                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.position_ref[0],Command_Now.Reference_State.position_ref[1], Command_Now.Reference_State.position_ref[2] );
                        state_vel_sp = Eigen::Vector3d(0.0, 0.0 , Command_Now.Reference_State.velocity_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                        _command_to_mavros.send_pos_vel_xyz_setpoint(state_sp, state_vel_sp, yaw_sp);
                        
                    }else
                    {
                        Command_Now.Mode = drone_msgs::ControlCommand::Idle;
                    }

                    Takeoff_position << state_sp[0], state_sp[1], Takeoff_position[2], yaw_sp;
                }
                ref_pos = state_sp;
                break;

            // ================================= Move =================================
            case drone_msgs::ControlCommand::Move:
                if(Command_Now.Reference_State.Move_frame  == drone_msgs::PositionReference::ENU_FRAME)
                {
                    if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_POS )
                    {
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.position_ref[0],Command_Now.Reference_State.position_ref[1],Command_Now.Reference_State.position_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_VEL )
                    {
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.velocity_ref[0],Command_Now.Reference_State.velocity_ref[1],Command_Now.Reference_State.velocity_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_VEL_Z_POS )
                    {
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.velocity_ref[0],Command_Now.Reference_State.velocity_ref[1],Command_Now.Reference_State.position_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_POS_Z_VEL )
                    {
                        Command_Now.Reference_State.position_ref[2] = _DroneState.position[2] + Command_Now.Reference_State.velocity_ref[2] * dt;
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.position_ref[0],Command_Now.Reference_State.position_ref[1],Command_Now.Reference_State.position_ref[2]);
                        state_vel_sp = Eigen::Vector3d(0.0, 0.0 ,Command_Now.Reference_State.velocity_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_ACC )
                    {
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.acceleration_ref[0],Command_Now.Reference_State.acceleration_ref[1],Command_Now.Reference_State.acceleration_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }
                }
                else
                {
                    // convert to ENU frame
                    if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_POS )
                    {
                        float d_pos_body[2] = {Command_Now.Reference_State.position_ref[0], Command_Now.Reference_State.position_ref[1]};         //the desired xy position in Body Frame
                        float d_pos_enu[2];                       //the desired xy position in enu Frame (The origin point is the drone)
                        d_pos_enu[0] = d_pos_body[0] * cos(_DroneState.attitude[2]) - d_pos_body[1] * sin(_DroneState.attitude[2]);
                        d_pos_enu[1] = d_pos_body[0] * sin(_DroneState.attitude[2]) + d_pos_body[1] * cos(_DroneState.attitude[2]);

                        Command_Now.Reference_State.position_ref[0] = _DroneState.position[0] + d_pos_enu[0];
                        Command_Now.Reference_State.position_ref[1] = _DroneState.position[1] + d_pos_enu[1];
                        Command_Now.Reference_State.position_ref[2] = _DroneState.position[2] + Command_Now.Reference_State.position_ref[2];
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.position_ref[0],Command_Now.Reference_State.position_ref[1],Command_Now.Reference_State.position_ref[2]);
                        yaw_sp = _DroneState.attitude[2] + Command_Now.Reference_State.yaw_ref;
                    }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_VEL )
                    {
                        //xy velocity mode
                        float d_vel_body[2] = {Command_Now.Reference_State.velocity_ref[0], Command_Now.Reference_State.velocity_ref[1]};         //the desired xy velocity in Body Frame
                        float d_vel_enu[2];                 //the desired xy velocity in NED Frame

                        //根据无人机当前偏航角进行坐标系转换
                        d_vel_enu[0] = d_vel_body[0] * cos(_DroneState.attitude[2]) - d_vel_body[1] * sin(_DroneState.attitude[2]);
                        d_vel_enu[1] = d_vel_body[0] * sin(_DroneState.attitude[2]) + d_vel_body[1] * cos(_DroneState.attitude[2]);
                        Command_Now.Reference_State.velocity_ref[0] = d_vel_enu[0];
                        Command_Now.Reference_State.velocity_ref[1] = d_vel_enu[1];
                        Command_Now.Reference_State.velocity_ref[2] = Command_Now.Reference_State.velocity_ref[2];
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.velocity_ref[0],Command_Now.Reference_State.velocity_ref[1],Command_Now.Reference_State.velocity_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_VEL_Z_POS )
                    {
                        //xy velocity mode
                        float d_vel_body[2] = {Command_Now.Reference_State.velocity_ref[0], Command_Now.Reference_State.velocity_ref[1]};         //the desired xy velocity in Body Frame
                        float d_vel_enu[2];                   //the desired xy velocity in NED Frame

                        //根据无人机当前偏航角进行坐标系转换
                        d_vel_enu[0] = d_vel_body[0] * cos(_DroneState.attitude[2]) - d_vel_body[1] * sin(_DroneState.attitude[2]);
                        d_vel_enu[1] = d_vel_body[0] * sin(_DroneState.attitude[2]) + d_vel_body[1] * cos(_DroneState.attitude[2]);
                        Command_Now.Reference_State.position_ref[0] = 0;
                        Command_Now.Reference_State.position_ref[1] = 0;
                        Command_Now.Reference_State.velocity_ref[0] = d_vel_enu[0];
                        Command_Now.Reference_State.velocity_ref[1] = d_vel_enu[1];
                        Command_Now.Reference_State.velocity_ref[2] = 0.0;

                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.velocity_ref[0],Command_Now.Reference_State.velocity_ref[1],Command_Now.Reference_State.position_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_POS_Z_VEL )
                    {
                        float d_pos_body[2] = {Command_Now.Reference_State.position_ref[0], Command_Now.Reference_State.position_ref[1]};         //the desired xy position in Body Frame
                        float d_pos_enu[2];                     //the desired xy position in enu Frame (The origin point is the drone)
                        d_pos_enu[0] = d_pos_body[0] * cos(_DroneState.attitude[2]) - d_pos_body[1] * sin(_DroneState.attitude[2]);
                        d_pos_enu[1] = d_pos_body[0] * sin(_DroneState.attitude[2]) + d_pos_body[1] * cos(_DroneState.attitude[2]);

                        Command_Now.Reference_State.position_ref[0] = _DroneState.position[0] + d_pos_enu[0];
                        Command_Now.Reference_State.position_ref[1] = _DroneState.position[1] + d_pos_enu[1];
                        Command_Now.Reference_State.position_ref[2] = _DroneState.position[2] + Command_Now.Reference_State.velocity_ref[2] * dt;
                        state_sp = Eigen::Vector3d(Command_Now.Reference_State.position_ref[0],Command_Now.Reference_State.position_ref[1],Command_Now.Reference_State.position_ref[2]);
                        state_vel_sp = Eigen::Vector3d(0.0, 0.0 ,Command_Now.Reference_State.velocity_ref[2]);
                        yaw_sp = Command_Now.Reference_State.yaw_ref;
                    }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_ACC )
                    {
                    pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, "XYZ_ACC not Defined. Change to XYZ_ACC in ENU frame");
                    }
                }

                if(Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_POS )
                {
                    _command_to_mavros.send_pos_setpoint(state_sp, yaw_sp);
                    ref_pos = state_sp;
                }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_VEL )
                {
                    if(Command_Now.Reference_State.Yaw_Rate_Mode)
                    {
                        yaw_rate_sp = Command_Now.Reference_State.yaw_rate_ref;
                        _command_to_mavros.send_vel_setpoint_yaw_rate(state_sp, yaw_rate_sp);
                    }else
                    {
                        _command_to_mavros.send_vel_setpoint(state_sp, yaw_sp);
                    }
                    ref_pos = dt*state_sp + Eigen::Vector3d(_DroneState.position[0],_DroneState.position[1],_DroneState.position[2]);
                }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_VEL_Z_POS )
                {
                    if(Command_Now.Reference_State.Yaw_Rate_Mode)
                    {
                        yaw_rate_sp = Command_Now.Reference_State.yaw_rate_ref;
                        _command_to_mavros.send_vel_xy_pos_z_setpoint_yawrate(state_sp, yaw_rate_sp);
                    }else
                    {
                        _command_to_mavros.send_vel_xy_pos_z_setpoint(state_sp, yaw_sp);
                    }
                    ref_pos = dt*Eigen::Vector3d(state_sp[0],state_sp[1],0.0) + Eigen::Vector3d(_DroneState.position[0],_DroneState.position[1],state_sp[2]);
                }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_POS_Z_VEL )
                {
                    _command_to_mavros.send_pos_vel_xyz_setpoint(state_sp, state_vel_sp, yaw_sp);
                    ref_pos = dt*Eigen::Vector3d(0.0,0.0,state_sp[2]) + Eigen::Vector3d(state_sp[0],state_sp[1],0.0);
                }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_ACC )
                {
                    _command_to_mavros.send_acc_xyz_setpoint(state_sp, yaw_sp);
                    ref_pos = 0.5*dt*dt*state_sp + Eigen::Vector3d(_DroneState.position[0],_DroneState.position[1],_DroneState.position[2]);
                }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_POS_VEL )
                {
                    state_sp = Eigen::Vector3d(Command_Now.Reference_State.position_ref[0],Command_Now.Reference_State.position_ref[1],Command_Now.Reference_State.position_ref[2]);
                    state_vel_sp = Eigen::Vector3d(Command_Now.Reference_State.velocity_ref[0], Command_Now.Reference_State.velocity_ref[1] ,Command_Now.Reference_State.velocity_ref[2]);
                    yaw_sp = Command_Now.Reference_State.yaw_ref;
                    _command_to_mavros.send_pos_vel_xyz_setpoint(state_sp, state_vel_sp,yaw_sp);
                    ref_pos = state_sp;
                }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::TRAJECTORY )
                {
                    // ToDo with Command_Now.Reference_State.bspline
                }else
                {
                    pub_message(message_pub, drone_msgs::Message::WARN, NODE_NAME, "Move_mode not Defined. Hold there");
                    
                    Command_Now.Reference_State.Move_mode       = drone_msgs::PositionReference::XYZ_POS;
                    Command_Now.Reference_State.Move_frame      = drone_msgs::PositionReference::ENU_FRAME;
                    Command_Now.Reference_State.position_ref[0] = hold_pos[0];
                    Command_Now.Reference_State.position_ref[1] = hold_pos[1];
                    Command_Now.Reference_State.position_ref[2] = hold_pos[2];
                    Command_Now.Reference_State.yaw_ref         = hold_yaw; //rad

                    state_sp = Eigen::Vector3d(hold_pos[0],hold_pos[1],hold_pos[2]);
                    yaw_sp = hold_yaw; //rad
                    ref_pos = hold_pos;
                
                    _command_to_mavros.send_pos_setpoint(state_sp, yaw_sp);
                }
                break;

            // ================================= Attitude =================================
            case drone_msgs::ControlCommand::Attitude:
                _command_to_mavros.send_attitude_setpoint(
                    quadrotor_common::geometryToEigen(Command_Now.Attitude_sp.desired_att_q),
                    Command_Now.Attitude_sp.collective_accel);
                break;

            // ================================= AttitudeRate =================================
            case drone_msgs::ControlCommand::AttitudeRate:
                _command_to_mavros.send_attitude_rate_setpoint(
                    quadrotor_common::geometryToEigen(Command_Now.Attitude_sp.desired_att_q),
                    quadrotor_common::geometryToEigen(Command_Now.Attitude_sp.body_rate),
                    Command_Now.Attitude_sp.collective_accel);
                break;

            // ================================= Rate =================================
            case drone_msgs::ControlCommand::Rate:
                _command_to_mavros.send_rate_setpoint(
                    quadrotor_common::geometryToEigen(Command_Now.Attitude_sp.body_rate),
                    Command_Now.Attitude_sp.collective_accel);
                break;
        }

        hold_pos = state_sp;
        hold_yaw = yaw_sp;

        // Rviz state reference pose
        ref_pose_rviz = get_rviz_ref_posistion(Command_Now);   
        rivz_ref_pose_pub.publish(ref_pose_rviz);

        Command_Last = Command_Now;
        rate.sleep();
    }

    return 0;

}

geometry_msgs::PoseStamped get_rviz_ref_posistion(const drone_msgs::ControlCommand& cmd)
{
    geometry_msgs::PoseStamped ref_pose;

    ref_pose.header.stamp = ros::Time::now();
    ref_pose.header.frame_id = "world";

    if(cmd.Mode == drone_msgs::ControlCommand::Idle)
    {
        ref_pose.pose.position.x = _DroneState.position[0];
        ref_pose.pose.position.y = _DroneState.position[1];
        ref_pose.pose.position.z = _DroneState.position[2];
        ref_pose.pose.orientation = _DroneState.attitude_q;
    }else if(cmd.Mode == drone_msgs::ControlCommand::Takeoff || cmd.Mode == drone_msgs::ControlCommand::Hold)
    {
        ref_pose.pose.position.x = cmd.Reference_State.position_ref[0];
        ref_pose.pose.position.y = cmd.Reference_State.position_ref[1];
        ref_pose.pose.position.z = cmd.Reference_State.position_ref[2];
        ref_pose.pose.orientation = _DroneState.attitude_q;
    }else if( cmd.Mode == drone_msgs::ControlCommand::Land )
    {
        ref_pose.pose.position.x = cmd.Reference_State.position_ref[0];
        ref_pose.pose.position.y = cmd.Reference_State.position_ref[1];
        ref_pose.pose.position.z = Disarm_height;
        ref_pose.pose.orientation = _DroneState.attitude_q;
    }
    else if(cmd.Mode == drone_msgs::ControlCommand::Move)
    {
        if( cmd.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_POS )
        {
            ref_pose.pose.position.x = cmd.Reference_State.position_ref[0];
            ref_pose.pose.position.y = cmd.Reference_State.position_ref[1];
            ref_pose.pose.position.z = cmd.Reference_State.position_ref[2];
        }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_VEL )
        {
            ref_pose.pose.position.x = _DroneState.position[0] + cmd.Reference_State.velocity_ref[0] * dt;
            ref_pose.pose.position.y = _DroneState.position[1] + cmd.Reference_State.velocity_ref[1] * dt;
            ref_pose.pose.position.z = _DroneState.position[2] + cmd.Reference_State.velocity_ref[2] * dt;
        }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_VEL_Z_POS )
        {
            ref_pose.pose.position.x = _DroneState.position[0] + cmd.Reference_State.velocity_ref[0] * dt;
            ref_pose.pose.position.y = _DroneState.position[1] + cmd.Reference_State.velocity_ref[1] * dt;
            ref_pose.pose.position.z = cmd.Reference_State.position_ref[2];
        }else if( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XY_POS_Z_VEL )
        {
            ref_pose.pose.position.x = cmd.Reference_State.position_ref[0];
            ref_pose.pose.position.y = cmd.Reference_State.position_ref[1];
            ref_pose.pose.position.z = _DroneState.position[2] + cmd.Reference_State.velocity_ref[2] * dt;
        }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_ACC )
        {
            ref_pose.pose.position.x = _DroneState.position[0] + 0.5 * cmd.Reference_State.acceleration_ref[0] * dt * dt;
            ref_pose.pose.position.y = _DroneState.position[1] + 0.5 * cmd.Reference_State.acceleration_ref[1] * dt * dt;
            ref_pose.pose.position.z = _DroneState.position[2] + 0.5 * cmd.Reference_State.acceleration_ref[2] * dt * dt;
        }else if ( Command_Now.Reference_State.Move_mode  == drone_msgs::PositionReference::XYZ_POS_VEL )
        {
            ref_pose.pose.position.x = cmd.Reference_State.position_ref[0];
            ref_pose.pose.position.y = cmd.Reference_State.position_ref[1];
            ref_pose.pose.position.z = cmd.Reference_State.position_ref[2];
        }

        ref_pose.pose.orientation = _DroneState.attitude_q;
    }
    else if(cmd.Mode == drone_msgs::ControlCommand::Disarm)
    {
        ref_pose.pose.position.x = _DroneState.position[0];
        ref_pose.pose.position.y = _DroneState.position[1];
        ref_pose.pose.position.z = Disarm_height;
        ref_pose.pose.orientation = _DroneState.attitude_q;
    }
    else if(cmd.Mode >= drone_msgs::ControlCommand::Attitude)
    {
        ref_pose.pose.position.x = cmd.Reference_State.position_ref[0];
        ref_pose.pose.position.y = cmd.Reference_State.position_ref[1];
        ref_pose.pose.position.z = cmd.Reference_State.position_ref[2];
        ref_pose.pose.orientation = _DroneState.attitude_q;
    }
    else
    {
        ref_pose.pose.position.x = 0.0;
        ref_pose.pose.position.y = 0.0;
        ref_pose.pose.position.z = 0.0;
        ref_pose.pose.orientation = _DroneState.attitude_q;
    }

    return ref_pose;
}
