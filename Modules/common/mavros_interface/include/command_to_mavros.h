#ifndef COMMAND_TO_MAVROS_H
#define COMMAND_TO_MAVROS_H

#include <ros/ros.h>
#include <math_utils.h>
#include <bitset>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/ActuatorControl.h>
#include <mavros_msgs/MountControl.h>
#include <mavros_msgs/Thrust.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Imu.h>
#include <drone_msgs/DroneState.h>
#include <drone_msgs/AttitudeReference.h>
#include <drone_msgs/DroneState.h>
using namespace std;

class command_to_mavros
{
private:
    ros::NodeHandle command_nh;

    ros::Publisher local_pos_pub,attitude_pub,rate_pub,thrust_pub;
    ros::Publisher setpoint_raw_local_pub;
    ros::Publisher setpoint_raw_attitude_pub;
    ros::Publisher actuator_setpoint_pub;
    ros::Publisher mount_control_pub;

public:
    string uav_name;

    command_to_mavros(void): command_nh("")
    {
        command_nh.param<string>("uav_name", uav_name, "/uav0");

        if (uav_name == "/uav0")
        {
            uav_name = "";
        }

        
        bool use_quaternion = false;
        command_nh.getParam("/mavros/setpoint_attitude/use_quaternion", use_quaternion);

        // =========================== [PUB] ===========================   
        // Pos / Vel / Acc / Yaw / Yaw_rate [Local Fixed Frame ENU_ROS]
        // mavros/src/plugins/setpoint_raw.cpp: Mavlink message (SET_POSITION_TARGET_LOCAL_NED (#84)) -> uORB message (trajectory_setpoint.msg)
        // [Prioritized]
        setpoint_raw_local_pub = command_nh.advertise<mavros_msgs::PositionTarget>(uav_name + "/mavros/setpoint_raw/local", 10);

        // Thrust + Attitude / Rate [Local Fixed Frame ENU_ROS]
        // mavros/src/plugins/setpoint_raw.cpp: Mavlink message (SET_ATTITUDE_TARGET (#82)) -> uORB message (vehicle_attitude_setpoint.msg) + (vehicle_rates_setpoint.msg)
        // [Prioritized]
        setpoint_raw_attitude_pub = command_nh.advertise<mavros_msgs::AttitudeTarget>(uav_name + "/mavros/setpoint_raw/attitude", 10);

		// Pos / Attitude [Local Fixed Frame ENU_ROS]
        // mavros/src/plugins/setpoint_position.cpp: Mavlink message (SET_POSITION_TARGET_LOCAL_NED (#84)) -> uORB message (trajectory_setpoint.msg)
        // Ref to test_mavros/tests/offboard_control.h
		local_pos_pub = command_nh.advertise<geometry_msgs::PoseStamped>(uav_name + "/mavros/setpoint_position/local", 10);
         
		// Thrust axis body_up (sign self-inverted in px4)
        // mavros/src/plugins/setpoint_attitude.cpp: Mavlink message (SET_ATTITUDE_TARGET (#82)) -> uORB message (vehicle_rates_setpoint.msg)
		thrust_pub = command_nh.advertise<mavros_msgs::Thrust>(uav_name + "/mavros/setpoint_attitude/thrust", 10);
         
		// Attitude [Local Fixed Frame ENU_ROS]
        // mavros/src/plugins/setpoint_attitude.cpp: Mavlink message (SET_ATTITUDE_TARGET (#82)) -> uORB message (vehicle_attitude_setpoint.msg)
		if(use_quaternion) attitude_pub = command_nh.advertise<geometry_msgs::PoseStamped>(uav_name + "/mavros/setpoint_attitude/attitude", 10);

		// Rate [Body Frame FRD]
        // mavros/src/plugins/setpoint_attitude.cpp: Mavlink message (SET_ATTITUDE_TARGET (#82)) -> uORB message (vehicle_rates_setpoint.msg)
		else rate_pub = command_nh.advertise<geometry_msgs::TwistStamped>(uav_name + "/mavros/setpoint_attitude/cmd_vel", 10);
         
        // Actuator contorl, throttle for each single rotation direction motor
        // mavros/src/plugins/actuator_control.cpp : Mavlink message (SET_ACTUATOR_CONTROL_TARGET) -> nothing
        // [none sub]
        actuator_setpoint_pub = command_nh.advertise<mavros_msgs::ActuatorControl>(uav_name + "/mavros/actuator_control", 10);

        // Mount control
        // mavros_extra/src/plugins/mount_control.cpp: Mavlink message (MAV_CMD_DO_MOUNT_CONTROL (#205)) -> nothing
        // uorb message (gimbal_manager_set_attitude.msg) need Mavlink message (GIMBAL_MANAGER_SET_ATTITUDE (#282)) but none in mavros
        // [none sub]
        // Alternative Mavlink message (COMMAND_LONG (#76)) but need to add api in mavlink_receiver.cpp
        mount_control_pub = command_nh.advertise<mavros_msgs::MountControl>(uav_name + "/mavros/mount_control/command", 1);

        // =========================== [SERVICE] ===========================
        // Arm / Disarm
        // mavros/src/plugins/command.cpp: Mavlink message (COMMAND_LONG (#76)) -> uorb message (vehicle_command.msg in Commander.cpp)
        arming_client = command_nh.serviceClient<mavros_msgs::CommandBool>(uav_name + "/mavros/cmd/arming");

        // Mode Switcher
        // mavros/src/plugins/sys_status.cpp: Mavlink message (SET_MODE (#11)) -> uorb message (vehicle_command.msg in Commander.cpp)
        set_mode_client = command_nh.serviceClient<mavros_msgs::SetMode>(uav_name + "/mavros/set_mode");
    }

    // [Service]
    ros::ServiceClient arming_client;
    mavros_msgs::CommandBool arm_cmd;
    ros::ServiceClient set_mode_client;
    mavros_msgs::SetMode mode_cmd;

    void idle();
    void takeoff();
    void loiter();
    void land();

    void send_pos_setpoint(const Eigen::Vector3d& pos_sp, float yaw_sp);
    void send_vel_setpoint(const Eigen::Vector3d& vel_sp, float yaw_sp);
    void send_vel_xy_pos_z_setpoint(const Eigen::Vector3d& state_sp, float yaw_sp);
    void send_vel_xy_pos_z_setpoint_yawrate(const Eigen::Vector3d& state_sp, float yaw_rate_sp);
    void send_vel_setpoint_body(const Eigen::Vector3d& vel_sp, float yaw_sp);
    void send_vel_setpoint_yaw_rate(const Eigen::Vector3d& vel_sp, float yaw_rate_sp);
    void send_pos_vel_xyz_setpoint(const Eigen::Vector3d& pos_sp, const Eigen::Vector3d& vel_sp, float yaw_sp);
    void send_acc_xyz_setpoint(const Eigen::Vector3d& accel_sp, float yaw_sp);


    void send_attitude_setpoint(const drone_msgs::AttitudeReference& _AttitudeReference);
    void send_attitude_rate_setpoint(const Eigen::Vector3d& attitude_rate_sp, float throttle_sp);
    void send_attitude_setpoint_yawrate(const drone_msgs::AttitudeReference& _AttitudeReference, float yaw_rate_sp);
    void send_actuator_setpoint(const Eigen::Vector4d& actuator_sp);


    void send_mount_control_command(const Eigen::Vector3d& gimbal_att_sp);
    
};

void command_to_mavros::send_mount_control_command(const Eigen::Vector3d& gimbal_att_sp)
{
  mavros_msgs::MountControl mount_setpoint;
  //
  mount_setpoint.mode = 2;
  mount_setpoint.pitch = gimbal_att_sp[0]; // Gimbal Pitch
  mount_setpoint.roll = gimbal_att_sp[1]; // Gimbal  Yaw
  mount_setpoint.yaw = gimbal_att_sp[2]; // Gimbal  Yaw

  mount_control_pub.publish(mount_setpoint);

}

void command_to_mavros::takeoff()
{
}

void command_to_mavros::land()
{
}

void command_to_mavros::loiter()
{
}

void command_to_mavros::idle()
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.type_mask = 0b010111000111;
    pos_setpoint.velocity.x = 0.0;
    pos_setpoint.velocity.y = 0.0;
    pos_setpoint.velocity.z = 0.0;
    pos_setpoint.yaw_rate = 0.0;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// px + py + pz + body_yaw [Local Frame ENU_ROS]
void command_to_mavros::send_pos_setpoint(const Eigen::Vector3d& pos_sp, float yaw_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b100111111000;  // 100 111 111 000  xyz + yaw

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.position.x = pos_sp[0];
    pos_setpoint.position.y = pos_sp[1];
    pos_setpoint.position.z = pos_sp[2];

    pos_setpoint.yaw = yaw_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// vx + vy + vz + body_yaw [Local Frame ENU_ROS]
void command_to_mavros::send_vel_setpoint(const Eigen::Vector3d& vel_sp, float yaw_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b100111000111;

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.velocity.x = vel_sp[0];
    pos_setpoint.velocity.y = vel_sp[1];
    pos_setpoint.velocity.z = vel_sp[2];

    pos_setpoint.yaw = yaw_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// vx + vy + vz + body_yaw_rate [Local Frame ENU_ROS]
void command_to_mavros::send_vel_setpoint_yaw_rate(const Eigen::Vector3d& vel_sp, float yaw_rate_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b010111000111;

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.velocity.x = vel_sp[0];
    pos_setpoint.velocity.y = vel_sp[1];
    pos_setpoint.velocity.z = vel_sp[2];

    pos_setpoint.yaw_rate = yaw_rate_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// vx + vy + vz + body_yaw  [Body Frame]
void command_to_mavros::send_vel_setpoint_body(const Eigen::Vector3d& vel_sp, float yaw_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b100111000111;

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 8;

    pos_setpoint.position.x = vel_sp[0];
    pos_setpoint.position.y = vel_sp[1];
    pos_setpoint.position.z = vel_sp[2];

    pos_setpoint.yaw = yaw_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// vx + vy + pz + body_yaw [Local Frame ENU_ROS]
void command_to_mavros::send_vel_xy_pos_z_setpoint(const Eigen::Vector3d& state_sp, float yaw_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b100111100011;   // 100 111 000 011  vx vy vz z + yaw

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.velocity.x = state_sp[0];
    pos_setpoint.velocity.y = state_sp[1];
    pos_setpoint.position.z = state_sp[2];

    pos_setpoint.yaw = yaw_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// vx + vy + pz + body_yaw_rate [Local Frame ENU_ROS]
void command_to_mavros::send_vel_xy_pos_z_setpoint_yawrate(const Eigen::Vector3d& state_sp, float yaw_rate_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b010111100011;   // 100 111 000 011  vx vy vz z + yawrate

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.velocity.x = state_sp[0];
    pos_setpoint.velocity.y = state_sp[1];
    pos_setpoint.position.z = state_sp[2];

    pos_setpoint.yaw_rate = yaw_rate_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// px + py + pz + vx + vy + vz + body_yaw [Local Frame ENU_ROS]
void command_to_mavros::send_pos_vel_xyz_setpoint(const Eigen::Vector3d& pos_sp, const Eigen::Vector3d& vel_sp, float yaw_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b100111000000;   // 100 111 000 000  vx vy　vz x y z+ yaw

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.position.x = pos_sp[0];
    pos_setpoint.position.y = pos_sp[1];
    pos_setpoint.position.z = pos_sp[2];
    pos_setpoint.velocity.x = vel_sp[0];
    pos_setpoint.velocity.y = vel_sp[1];
    pos_setpoint.velocity.z = vel_sp[2];

    pos_setpoint.yaw = yaw_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);
}

// ax + ay + az + body_yaw [Local Frame ENU_ROS]
void command_to_mavros::send_acc_xyz_setpoint(const Eigen::Vector3d& accel_sp, float yaw_sp)
{
    mavros_msgs::PositionTarget pos_setpoint;
    //Bitmask toindicate which dimensions should be ignored (1 means ignoring, 0 means selection; Bit 10 must be set to 0)
    //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
    //Bit 10 should be set as 0, which means it's not force sp
    pos_setpoint.type_mask = 0b100000111111;

    //uint8 FRAME_LOCAL_NED = 1
    //uint8 FRAME_BODY_NED = 8
    pos_setpoint.coordinate_frame = 1;

    pos_setpoint.acceleration_or_force.x = accel_sp[0];
    pos_setpoint.acceleration_or_force.y = accel_sp[1];
    pos_setpoint.acceleration_or_force.z = accel_sp[2];

    pos_setpoint.yaw = yaw_sp;

    setpoint_raw_local_pub.publish(pos_setpoint);

}

// quaternion attitude + throttle
void command_to_mavros::send_attitude_setpoint(const drone_msgs::AttitudeReference& _AttitudeReference)
{
    mavros_msgs::AttitudeTarget att_setpoint;

    //Mappings: If any of these bits are set, the corresponding input should be ignored:
    //bit 1: body roll rate, bit 2: body pitch rate, bit 3: body yaw rate. bit 4-bit 5: reserved, bit 6: 3D body thrust sp instead of throttle, bit 7: throttle, bit 8: attitude

    att_setpoint.type_mask = 0b00111111;

    att_setpoint.orientation.x = _AttitudeReference.desired_att_q.x;
    att_setpoint.orientation.y = _AttitudeReference.desired_att_q.y;
    att_setpoint.orientation.z = _AttitudeReference.desired_att_q.z;
    att_setpoint.orientation.w = _AttitudeReference.desired_att_q.w;

    att_setpoint.thrust = _AttitudeReference.desired_throttle; // throttle [0,1] rather att_setpoint.thrust_body[]

    setpoint_raw_attitude_pub.publish(att_setpoint);
}


// quaternion attitude + throttle + body_yaw_rate
void command_to_mavros::send_attitude_setpoint_yawrate(const drone_msgs::AttitudeReference& _AttitudeReference, float yaw_rate_sp)
{
    mavros_msgs::AttitudeTarget att_setpoint;

    //Mappings: If any of these bits are set, the corresponding input should be ignored:
    //bit 1: body roll rate, bit 2: body pitch rate, bit 3: body yaw rate. bit 4-bit 5: reserved, bit 6: 3D body thrust sp instead of throttle, bit 7: throttle, bit 8: attitude

    att_setpoint.type_mask = 0b00111011;

    att_setpoint.orientation.x = _AttitudeReference.desired_att_q.x;
    att_setpoint.orientation.y = _AttitudeReference.desired_att_q.y;
    att_setpoint.orientation.z = _AttitudeReference.desired_att_q.z;
    att_setpoint.orientation.w = _AttitudeReference.desired_att_q.w;

    att_setpoint.thrust = _AttitudeReference.desired_throttle; // throttle [0,1] rather att_setpoint.thrust_body[]

    att_setpoint.body_rate.x = 0.0;
    att_setpoint.body_rate.y = 0.0;
    att_setpoint.body_rate.z = yaw_rate_sp;

    setpoint_raw_attitude_pub.publish(att_setpoint);
}

// body_rate + throttle
void command_to_mavros::send_attitude_rate_setpoint(const Eigen::Vector3d& attitude_rate_sp, float throttle_sp)
{
    mavros_msgs::AttitudeTarget att_setpoint;

    //Mappings: If any of these bits are set, the corresponding input should be ignored:
    //bit 1: body roll rate, bit 2: body pitch rate, bit 3: body yaw rate. bit 4-bit 5: reserved, bit 6: 3D body thrust sp instead of throttle, bit 7: throttle, bit 8: attitude

    att_setpoint.type_mask = 0b10111000;

    att_setpoint.body_rate.x = attitude_rate_sp[0];
    att_setpoint.body_rate.y = attitude_rate_sp[1];
    att_setpoint.body_rate.z = attitude_rate_sp[2];

    att_setpoint.thrust = throttle_sp; // throttle [0,1] rather att_setpoint.thrust_body[]

    setpoint_raw_attitude_pub.publish(att_setpoint);
}

// actuator control setpoint [PWM]
void command_to_mavros::send_actuator_setpoint(const Eigen::Vector4d& actuator_sp)
{
    mavros_msgs::ActuatorControl actuator_setpoint;

    actuator_setpoint.group_mix = 0;
    actuator_setpoint.controls[0] = actuator_sp(0);
    actuator_setpoint.controls[1] = actuator_sp(1);
    actuator_setpoint.controls[2] = actuator_sp(2);
    actuator_setpoint.controls[3] = actuator_sp(3);
    actuator_setpoint.controls[4] = 0.0;
    actuator_setpoint.controls[5] = 0.0;
    actuator_setpoint.controls[6] = 0.0;
    actuator_setpoint.controls[7] = 0.0;

    actuator_setpoint_pub.publish(actuator_setpoint);
}


#endif


