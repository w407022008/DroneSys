#ifndef GIMBAL_CONTROL_H
#define GIMBAL_CONTROL_H

// PX4云台控制类
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <math.h>
#include <mavros_msgs/MountControl.h>
#include <geometry_msgs/Quaternion.h>
#include <drone_msgs/ControlCommand.h>
#include <drone_msgs/DroneState.h>
#include <drone_msgs/Message.h>
#include <mavros_msgs/ActuatorControl.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>

using namespace std;

#define DIS_THRES 0.1
#define VISION_THRES 10

//相机安装OFFSET
#define FRONT_CAMERA_OFFSET_X 0.2
#define FRONT_CAMERA_OFFSET_Y 0.0
#define FRONT_CAMERA_OFFSET_Z -0.05

#define DOWN_CAMERA_OFFSET_X 0.0
#define DOWN_CAMERA_OFFSET_Y 0.0
#define DOWN_CAMERA_OFFSET_Z -0.1

float cal_distance(const Eigen::Vector3f& pos_drone,const Eigen::Vector3f& pos_target)
{
    Eigen::Vector3f relative;
    relative =  pos_target - pos_drone; 
    return relative.norm(); 
}

float cal_distance_tracking(const Eigen::Vector3f& pos_drone,const Eigen::Vector3f& pos_target,const Eigen::Vector3f& delta)
{
    Eigen::Vector3f relative;
    relative =  pos_target - pos_drone - delta; 
    return relative.norm(); 
}

//constrain_function
float constrain_function(float data, float Max)
{
    if(abs(data)>Max)
    {
        return (data > 0) ? Max : -Max;
    }
    else
    {
        return data;
    }
}

//constrain_function2
float constrain_function2(float data, float Min,float Max)
{
    if(data > Max)
    {
        return Max;
    }
    else if (data < Min)
    {
        return Min;
    }else
    {
        return data;
    }
}

//sign_function
float sign_function(float data)
{
    if(data>0)
    {
        return 1.0;
    }
    else if(data<0)
    {
        return -1.0;
    }
    else if(data == 0)
    {
        return 0.0;
    }
}

// min function
float min(float data1,float data2)
{
    if(data1>=data2)
    {
        return data2;
    }
    else
    {
        return data1;
    }
}

//旋转矩阵：机体系到惯性系
Eigen::Matrix3f get_rotation_matrix(float phi, float theta, float psi)
{
    Eigen::Matrix3f R_Body_to_ENU;

    float r11 = cos(theta)*cos(psi);
    float r12 = - cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi);
    float r13 = sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi);
    float r21 = cos(theta)*sin(psi);
    float r22 = cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi);
    float r23 = - sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
    float r31 = - sin(theta);
    float r32 = sin(phi)*cos(theta);
    float r33 = cos(phi)*cos(theta); 
    R_Body_to_ENU << r11,r12,r13,r21,r22,r23,r31,r32,r33;

    return R_Body_to_ENU;
}

class gimbal_control
{
    public:
    gimbal_control(void):
        nh("~")
    {
        // 订阅云台当前角度
        gimbal_att_sub = nh.subscribe<geometry_msgs::Quaternion>("/mavros/mount_control/orientation", 10, &gimbal_control::gimbal_att_cb,this);

        // 云台控制：本话题要发送至飞控(通过Mavros_extra功能包 /plugins/mount_control.cpp发送)
        mount_control_pub = nh.advertise<mavros_msgs::MountControl>( "/mavros/mount_control/command", 1);

        // 云台角度初始化
        gimbal_att        = Eigen::Vector3d(0.0,0.0,0.0);
        gimbal_att_last   = Eigen::Vector3d(0.0,0.0,0.0);

        begin_time = ros::Time::now();

        dt_time = 0.0;

        last_time = get_time_in_sec(begin_time);
    }

    // 云台角度
    Eigen::Vector3d gimbal_att;
    // 上一时刻云台角度
    Eigen::Vector3d gimbal_att_last;
    // 估算的云台角速度
    Eigen::Vector3d gimbal_att_rate;

    // 估算云台角速度
    ros::Time begin_time;
    float last_time;
    float dt_time;

    //发送云台控制指令API函数
    void send_mount_control_command(const Eigen::Vector3d& gimbal_att_sp);

    Eigen::Vector3d get_gimbal_att();

    Eigen::Vector3d get_gimbal_att_rate();
    
    private:

    ros::NodeHandle nh;

    ros::Subscriber gimbal_att_sub;
    ros::Publisher mount_control_pub;

    Eigen::Vector3d quaternion_to_euler(const Eigen::Quaterniond &q);

    float get_time_in_sec(const ros::Time& begin_time);

    void gimbal_att_cb(const geometry_msgs::Quaternion::ConstPtr& msg)
    {
        Eigen::Quaterniond gimbal_att_quat;

        gimbal_att_quat = Eigen::Quaterniond(msg->w, msg->x, msg->y, msg->z);

        //Transform the Quaternion to euler Angles
        gimbal_att = quaternion_to_euler(gimbal_att_quat);
        
        float cur_time = get_time_in_sec(begin_time);
        dt_time = cur_time  - last_time;
        dt_time = constrain_function2(dt_time, 0.01, 0.03);
        last_time = cur_time;

        gimbal_att_rate = (gimbal_att - gimbal_att_last)/dt_time;

        gimbal_att_last = gimbal_att;
    }
};

void gimbal_control::send_mount_control_command(const Eigen::Vector3d& gimbal_att_sp)
{
  mavros_msgs::MountControl mount_setpoint;
  //
  mount_setpoint.header.stamp = ros::Time::now();
  mount_setpoint.header.frame_id = "map";
  mount_setpoint.mode = 2;
  mount_setpoint.roll = gimbal_att_sp[0]; // Gimbal Roll [deg]
  mount_setpoint.pitch = gimbal_att_sp[1]; // Gimbal   Pitch[deg]
  mount_setpoint.yaw = gimbal_att_sp[2]; // Gimbal  Yaw [deg]

  mount_control_pub.publish(mount_setpoint);

}

Eigen::Vector3d gimbal_control::get_gimbal_att_rate()
{
    return gimbal_att_rate;
}

Eigen::Vector3d gimbal_control::get_gimbal_att()
{
    return gimbal_att;
}

Eigen::Vector3d gimbal_control::quaternion_to_euler(const Eigen::Quaterniond &q)
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

float gimbal_control::get_time_in_sec(const ros::Time& begin_time)
{
    ros::Time time_now = ros::Time::now();
    float currTimeSec = time_now.sec - begin_time.sec;
    float currTimenSec = time_now.nsec / 1e9 - begin_time.nsec / 1e9;
    return (currTimeSec + currTimenSec);
}
    

#endif


