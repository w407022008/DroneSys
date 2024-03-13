#include "ros/ros.h"
#include "OptiTrackFeedBackRigidBody.h"
#include <eigen3/Eigen/Eigen>
using namespace std;
using namespace Eigen;
int angular_window, linear_window, Control_Rate;
int main(int argc, char **argv)
{
  // Initialize ros node
  ros::init(argc, argv, "OptiTrackTest");
  ros::NodeHandle n;
  n.param<int>("Control_Rate", Control_Rate, 100);
  n.param<int>("linear_window", linear_window, 1);
  n.param<int>("angular_window", angular_window, 1);
  // Initialize OptiTrack System
  OptiTrackFeedBackRigidBody UAV("/vrpn_client_node/UAV/pose",n,linear_window,angular_window);
  ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("/vrpn_client_node/UAV/odometry", 10);
  rigidbody_state State;
  nav_msgs::Odometry odom;
  ros::Rate loop_rate(Control_Rate);
  while (ros::ok())
  {
    UAV.RosWhileLoopRun();
    // UAV.GetOptiTrackState();
    UAV.GetState(State);

    odom.pose.pose.position.x = State.Position[0];
    odom.pose.pose.position.y = State.Position[1];
    odom.pose.pose.position.z = State.Position[2];

    odom.twist.twist.linear.x = State.Velocity[0];
    odom.twist.twist.linear.y = State.Velocity[1];
    odom.twist.twist.linear.z = State.Velocity[2];

    odom.pose.pose.orientation.w = State.Quaterion[0];
    odom.pose.pose.orientation.x = State.Quaterion[1];
    odom.pose.pose.orientation.y = State.Quaterion[2];
    odom.pose.pose.orientation.z = State.Quaterion[3];

    odom.twist.twist.angular.x = State.Omega_B[0];
    odom.twist.twist.angular.y = State.Omega_B[1];
    odom.twist.twist.angular.z = State.Omega_B[2];

    odom.header.stamp = State.Stamp;
    odom.header.frame_id = "odom_ned";
    odom.child_frame_id = "base_link";
    odom_pub.publish(odom);

    cout<< "Position: " << State.Position.transpose() << endl;
    cout<< "Velocity: " << State.Velocity.transpose() << endl;
    cout<< "AngularVelocity: " << State.Omega_B.transpose() << endl;
    cout<< "R_IB: \n" << State.R_IB << endl;
    cout<< "R_BI: \n" << State.R_BI << endl;
    cout<< "Euler: " <<State.Euler.transpose()*57.3<<endl;
    cout<< "Quaterion: " <<State.Quaterion.transpose()<<endl;

    ros::spinOnce();// do the loop once
    loop_rate.sleep();

  }
  return 0;
}
