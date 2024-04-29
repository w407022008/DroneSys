#pragma once
#include "ros/ros.h"
#include "std_msgs/Empty.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseStamped.h"
//#include "UtilityFunctions.h"
#include <eigen3/Eigen/Eigen>
//maximum window size
#ifndef max_windowsize
#define max_windowsize 10
#endif
using namespace Eigen;

struct optitrack_pose{
    ros::Time Stamp;
    Vector3d Position;
    double q0;
    double q1;
    double q2;
    double q3;
    double t;
    Matrix3d R_IB; //
    Matrix3d R_BI; //
};

struct rigidbody_state{
    ros::Time Stamp;
    Vector4d Quaternion;
    Vector3d Position;// inertial position
    Vector3d Velocity; // inertial velocity
    Matrix3d Omega_Cross; // angular velocity skew
    Vector3d Omega_B;// angular velocity in Frame B
    Matrix3d R_IB; // rotation matrix
    Matrix3d R_BI; //
    Vector3d Euler;// euler angle
    double time_stamp;
};

class OptiTrackFeedBackRigidBody{

    //-------Optitrack Related-----///
    geometry_msgs::PoseStamped OptiTrackdata;
    unsigned int OptiTrackFlag; // OptiTrackState 0: no data feed,: 1 data feed present
    void OptiTrackCallback(const geometry_msgs::PoseStamped& msg);   
    unsigned int FeedbackState;// 0 no feedback, 1 has feedback
    ros::Subscriber subOptiTrack;// OptiTrack Data
    //--------Filter Parameters-------//
    unsigned int linear_velocity_window; // window size
    unsigned int angular_velocity_window; // window size
    //--------Filter Buffer-----------//
    // raw velocity buffer from numerical differentiation
    Vector3d  velocity_raw[max_windowsize];
    Vector3d  angular_velocity_raw[max_windowsize];
    Vector3d  velocity_filtered;        // filtered velocity
    Vector3d  angular_velocity_filtered;// filtered angular velocity
    optitrack_pose  pose[2];/*pose info from optitrack pose[1] should be the latest mesaured value, 
    pose[0] is value of the last measurment (in world frame by default, if other frames
    are used , please changle the frame selectioin in the launch file */ 
    //--------Filter Methods-----------//
    void CalculateVelocityFromPose();// calculate velocity info from pose update measurements
    void MovingWindowAveraging();// a filter using moving window
    void PushRawVelocity(Vector3d& new_linear_velocity, Vector3d& new_angular_velocity);// push newly measured velocity into raw velocity buffer
    void PushPose();//push newly measured pose into dronepose buffer
    void Reset();
    //--------Update Rigid-body State ------//
    rigidbody_state state;
public:
    OptiTrackFeedBackRigidBody(const char* name,ros::NodeHandle& n, unsigned int linear_window, unsigned int angular_window);
    ~OptiTrackFeedBackRigidBody();
    int GetOptiTrackState();
    void GetState(rigidbody_state& state);
    void GetRaWVelocity(Vector3d& linear_velocity,Vector3d& angular_velocity);
    void RosWhileLoopRun();// This function should be put into ros while loop
    void GetEulerAngleFromQuaterion_NormalConvention(double (&eulerangle)[3]);
    void GetEulerAngleFromQuaterion_OptiTrackYUpConvention(double (&eulerangle)[3]);
    void Veemap(Matrix3d& cross_matrix, Vector3d& vector);
    void Hatmap(Vector3d& vector, Matrix3d& cross_matrix);
};
