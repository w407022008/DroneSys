#ifndef STATE_FROM_MAVROS_H
#define STATE_FROM_MAVROS_H

#include <ros/ros.h>
#include <bitset>
#include <math_utils.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/ExtendedState.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/Altitude.h>
#include <mavros_msgs/PositionTarget.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/ActuatorControl.h>
#include <mavros_msgs/ManualControl.h>
#include <sensor_msgs/Imu.h>
#include <drone_msgs/DroneState.h>
#include <drone_msgs/DroneTarget.h>
#include <drone_msgs/RCInput.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float64.h>
#include <tf2_msgs/TFMessage.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_listener.h>

using namespace std;

class state_from_mavros
{
private:

    ros::NodeHandle state_nh;

    ros::Subscriber state_sub;
    ros::Subscriber extended_state_sub;
    ros::Subscriber odometry_sub;
    ros::Subscriber position_sub;
    ros::Subscriber velocity_sub;
    ros::Subscriber attitude_sub;
    ros::Subscriber alt_sub;
    ros::Subscriber position_target_sub;
    ros::Subscriber attitude_target_sub;
    ros::Subscriber actuator_target_sub;
    ros::Subscriber manual_control_sub;


    void pos_target_cb(const mavros_msgs::PositionTarget::ConstPtr& msg)
    {
        _DroneTraget.position_target[0] = msg->position.x;
        _DroneTraget.position_target[1] = msg->position.y;
        _DroneTraget.position_target[2] = msg->position.z;
        
        _DroneTraget.velocity_target[0] = msg->velocity.x;
        _DroneTraget.velocity_target[1] = msg->velocity.y;
        _DroneTraget.velocity_target[2] = msg->velocity.z;

        _DroneTraget.acceleration_target[0] = msg->acceleration_or_force.x;
        _DroneTraget.acceleration_target[1] = msg->acceleration_or_force.y;
        _DroneTraget.acceleration_target[2] = msg->acceleration_or_force.z;
    }

    void att_target_cb(const mavros_msgs::AttitudeTarget::ConstPtr& msg)
    {
        _DroneTraget.q_target.w = msg->orientation.w;
        _DroneTraget.q_target.x = msg->orientation.x;
        _DroneTraget.q_target.y = msg->orientation.y;
        _DroneTraget.q_target.z = msg->orientation.z;
        
        //Transform the Quaternion to euler Angles
        Eigen::Vector3d euler_target = quaternion_to_euler(msg->orientation.w,
                                                            msg->orientation.x,
                                                            msg->orientation.y,
                                                            msg->orientation.z);
        _DroneTraget.euler_target[0] = euler_target[0];
        _DroneTraget.euler_target[1] = euler_target[1];
        _DroneTraget.euler_target[2] = euler_target[2];

        _DroneTraget.rate_target[0] = msg->body_rate.x;
        _DroneTraget.rate_target[1] = msg->body_rate.y;
        _DroneTraget.rate_target[2] = msg->body_rate.z;

        _DroneTraget.thrust_target = msg->thrust;
    }

    void actuator_target_cb(const mavros_msgs::ActuatorControl::ConstPtr& msg)
    {
        _DroneTraget.actuator_target = *msg;
    }

    void state_cb(const mavros_msgs::State::ConstPtr &msg)
    {
        _DroneState.connected = msg->connected;
        _DroneState.armed = msg->armed;
        _DroneState.mode = msg->mode;
    }

    void extended_state_cb(const mavros_msgs::ExtendedState::ConstPtr &msg)
    {
        if(msg->landed_state == msg->LANDED_STATE_ON_GROUND)
        {
            _DroneState.landed = true;
        }else
        {
            _DroneState.landed = false;
        }
    }

    void odom_cb(const nav_msgs::Odometry::ConstPtr &msg)
    {
        _DroneState.header.stamp = msg->header.stamp;

        _DroneState.position[0] = msg->pose.pose.position.x;
        _DroneState.position[1] = msg->pose.pose.position.y;
        _DroneState.position[2] = msg->pose.pose.position.z;

        _DroneState.velocity[0] = msg->twist.twist.linear.x;
        _DroneState.velocity[1] = msg->twist.twist.linear.y;
        _DroneState.velocity[2] = msg->twist.twist.linear.z;

        Eigen::Quaterniond q_fcu = Eigen::Quaterniond(
            msg->pose.pose.orientation.w, 
            msg->pose.pose.orientation.x, 
            msg->pose.pose.orientation.y, 
            msg->pose.pose.orientation.z);

        Eigen::Vector3d euler_fcu = quaternion_to_euler(q_fcu);
        _DroneState.attitude[0] = euler_fcu[0];
        _DroneState.attitude[1] = euler_fcu[1];
        _DroneState.attitude[2] = euler_fcu[2];
        
        Eigen::Quaterniond q_ = quaternion_from_rpy(euler_fcu);
        _DroneState.attitude_q.w = q_.w();
        _DroneState.attitude_q.x = q_.x();
        _DroneState.attitude_q.y = q_.y();
        _DroneState.attitude_q.z = q_.z();

        _DroneState.attitude_rate[0] = msg->twist.twist.angular.x;
        _DroneState.attitude_rate[1] = msg->twist.twist.angular.y;
        _DroneState.attitude_rate[2] = msg->twist.twist.angular.z;
    }

    void pos_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        _DroneState.position[0] = msg->pose.position.x;
        _DroneState.position[1] = msg->pose.position.y;
        _DroneState.position[2] = msg->pose.position.z;
    }

    void vel_cb(const geometry_msgs::TwistStamped::ConstPtr &msg)
    {
        _DroneState.velocity[0] = msg->twist.linear.x;
        _DroneState.velocity[1] = msg->twist.linear.y;
        _DroneState.velocity[2] = msg->twist.linear.z;
    }

    void att_cb(const sensor_msgs::Imu::ConstPtr& msg)
    {
        _DroneState.header.stamp = msg->header.stamp;
        
        Eigen::Quaterniond q_fcu = Eigen::Quaterniond(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

        Eigen::Vector3d euler_fcu = quaternion_to_euler(q_fcu);
        _DroneState.attitude[0] = euler_fcu[0];
        _DroneState.attitude[1] = euler_fcu[1];
        _DroneState.attitude[2] = euler_fcu[2];
        
        Eigen::Quaterniond q_ = quaternion_from_rpy(euler_fcu);
        _DroneState.attitude_q.w = q_.w();
        _DroneState.attitude_q.x = q_.x();
        _DroneState.attitude_q.y = q_.y();
        _DroneState.attitude_q.z = q_.z();

        _DroneState.attitude_rate[0] = msg->angular_velocity.x;
        _DroneState.attitude_rate[1] = msg->angular_velocity.y;
        _DroneState.attitude_rate[2] = msg->angular_velocity.z;
    }

    void alt_cb(const mavros_msgs::Altitude::ConstPtr &msg)
    {
        _DroneState.rel_alt = msg->relative;
    }

    void manual_control_cb(const mavros_msgs::ManualControl::ConstPtr &msg)
    {
        // NED frame map to ENU, or FRD map to FLU
        _RCInput.rc_x = msg->x;
        _RCInput.rc_y = -msg->y;
        double z = msg->z;
        if (z<=0.5)
            _RCInput.rc_z = (z-0.5)/0.5;
        else
            _RCInput.rc_z = (z-0.5)/0.5;
        _RCInput.rc_r = -msg->r;
        _RCInput.buttons = msg->buttons;
        _RCInput.data_source = drone_msgs::RCInput::MAVROS_MANUAL_CONTROL;
    }

public:

    drone_msgs::DroneState _DroneState;
    drone_msgs::DroneTarget _DroneTraget;
    drone_msgs::RCInput _RCInput;
    string uav_name;
    
    //constructed function
    state_from_mavros(void):state_nh("~")
    {
        state_nh.param<string>("uav_name", uav_name, "/uav0");

        if (uav_name == "/uav0")
        {
            uav_name = "";
        }

        // =========================== [SUB] ===========================
        // Drone status
        // mavros/src/plugins/sys_status.cpp: Mavlink message (MAVLINK_MSG_ID_SYS_STATUS (#1)) <- uORB message (vehicle_status.msg)
        state_sub = state_nh.subscribe<mavros_msgs::State>(uav_name + "/mavros/state", 10, &state_from_mavros::state_cb,this);

        // Drone extended status
        // mavros/src/plugins/sys_status.cpp: Mavlink message (MAVLINK_MSG_ID_EXTENDED_SYS_STATE (#245)) <- uORB message (vehicle_land_detected.msg)
        extended_state_sub = state_nh.subscribe<mavros_msgs::ExtendedState>(uav_name + "/mavros/extended_state", 10, &state_from_mavros::extended_state_cb,this);

        // Drone state in ENU_ROS frame, NED frame used in PX4
        // mavros/src/plugins/local_position.cpp: Mavlink message (LOCAL_POSITION_NED (#32) & ATTITUDE (#30)) <- uORB message (vehicle_local_position.msg & vehicle_attitude.msg)
        odometry_sub = state_nh.subscribe<nav_msgs::Odometry>(uav_name + "/mavros/local_position/odom", 10, &state_from_mavros::odom_cb,this);

        // Drone state in ENU_ROS frame, defined in NED_PX4 frame in PX4
        // mavros/src/plugins/local_position.cpp: Mavlink message (LOCAL_POSITION_NED (#32) & ATTITUDE (#30)) <- uORB message (vehicle_local_position.msg & vehicle_attitude.msg)
        // position_sub = state_nh.subscribe<geometry_msgs::PoseStamped>(uav_name + "/mavros/local_position/pose", 10, &state_from_mavros::pos_cb,this);

        // Drone velocity
        // mavros/src/plugins/local_position.cpp: Mavlink message (LOCAL_POSITION_NED (#32) & ATTITUDE (#30)) <- uORB message (vehicle_local_position.msg & vehicle_angular_velocity.msg)
        // velocity_sub = state_nh.subscribe<geometry_msgs::TwistStamped>(uav_name + "/mavros/local_position/velocity_local", 10, &state_from_mavros::vel_cb,this);

        // Drone attitude and Imu
        // mavros/src/plugins/imu.cpp: Mavlink message (ATTITUDE (#30) & MAVLINK_MSG_ID_HIGHRES_IMU (#105)) <- uORB message (vehicle_attitude.msg & vehicle_imu.msg)
        // attitude_sub = state_nh.subscribe<sensor_msgs::Imu>(uav_name + "/mavros/imu/data", 10, &state_from_mavros::att_cb,this); 

        // Drone altitude
        // mavros/src/plugins/altitude.cpp: Mavlink message (MAVLINK_MSG_ID_ALTITUDE (#141)) <- uORB message (vehicle_local_position.msg & home_position.msg)
        alt_sub = state_nh.subscribe<mavros_msgs::Altitude>(uav_name + "/mavros/altitude", 10, &state_from_mavros::alt_cb,this);
        
        // Pos / Vel / Acc [Local Fixed Frame ENU_ROS]
        // mavros/src/plugins/setpoint_raw.cpp: uORB message (vehicle_local_position_setpoint.msg) -> Mavlink message (POSITION_TARGET_LOCAL_NED (#85))
        position_target_sub = state_nh.subscribe<mavros_msgs::PositionTarget>(uav_name + "/mavros/setpoint_raw/target_local", 10, &state_from_mavros::pos_target_cb,this);

        // Attitude / Rate [Local Fixed Frame ENU_ROS]
        // mavros/src/plugins/setpoint_raw.cpp: uORB message (vehicle_attitude_setpoint.msg) -> Mavlink message (ATTITUDE_TARGET (#83))
        attitude_target_sub = state_nh.subscribe<mavros_msgs::AttitudeTarget>(uav_name + "/mavros/setpoint_raw/target_attitude", 10, &state_from_mavros::att_target_cb,this);

        // Actuator contorl, throttle for each single rotation direction motor
        // mavros/src/plugins/actuator_control.cpp: uORB message (actuator_controls.msg) -> Mavlink message (ACTUATOR_CONTROL_TARGET (#140))
        actuator_target_sub = state_nh.subscribe<mavros_msgs::ActuatorControl>(uav_name + "/mavros/target_actuator_control", 10, &state_from_mavros::actuator_target_cb,this);

        // Drone RC in
        // mavros/src/plugins/manual_control.cpp: Mavlink message (MANUAL_CONTROL (#69)) <- uORB message (manual_control_setpoint.msg)
        manual_control_sub = state_nh.subscribe<mavros_msgs::ManualControl>(uav_name + "/mavros/manual_control/control", 10, &state_from_mavros::manual_control_cb,this); 
    }

};

    
#endif
