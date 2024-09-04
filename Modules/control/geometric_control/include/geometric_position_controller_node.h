/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GEOMETRIC_POSITION_CONTROLLER_NODE_H
#define GEOMETRIC_POSITION_CONTROLLER_NODE_H

#include <boost/bind.hpp>
#include <Eigen/Eigen>
#include <stdio.h>

#include <geometry_msgs/PoseStamped.h>
#include <mav_msgs/RollPitchYawrateThrust.h>
#include <mav_msgs/Actuators.h>
#include <mav_msgs/AttitudeThrust.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <quadrotor_common/trajectory_point.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include "common.h"
#include "geometric_position_controller.h"

namespace geometric_position_controller {

class GeometricPositionControllerNode {
 public:
  GeometricPositionControllerNode();
  ~GeometricPositionControllerNode();

  void InitializeParams();
  void Publish();

 private: 
  GeometricPositionController geometric_position_controller_;

  std::string namespace_;
  bool cmd_active_;
  float control_frequency_;
  bool rate_control_;
  drone_msgs::ControlCommand Command_to_pub;

  // subscribers
  ros::Subscriber cmd_active_sub_;
  ros::Subscriber cmd_trajectory_sub_;
  ros::Subscriber cmd_multi_dof_joint_trajectory_sub_;
  ros::Subscriber cmd_pose_sub_;
  ros::Subscriber cmd_trajectory_point_sub_;
  ros::Subscriber cmd_roll_pitch_yawrate_thrust_sub_;
  ros::Subscriber odometry_sub_;

  ros::Publisher rotors_motor_velocity_reference_pub_;
  ros::Publisher mavros_setpoint_raw_attitude_pub;
  ros::Publisher drone_msg_pub;

  std::deque<quadrotor_common::TrajectoryPoint> commands_;
  std::deque<ros::Duration> command_waiting_times_;
  ros::Timer command_timer_, odometry_timer_;

  void CommandActiveCallback(const std_msgs::Bool& active);
  void RollPitchYawrateThrustCallback(
      const mav_msgs::RollPitchYawrateThrustConstPtr& roll_pitch_yawrate_thrust_reference_msg);
  
  void TrajecotryPointCallback(
      const quadrotor_msgs::TrajectoryPointConstPtr& msg);
  void CommandPoseCallback(
      const geometry_msgs::PoseStampedConstPtr& pose_msg);
    
  void TrajectoryCallback(
      const quadrotor_msgs::TrajectoryConstPtr& msg);
  void MultiDofJointTrajectoryCallback(
      const trajectory_msgs::MultiDOFJointTrajectoryConstPtr& msg);
  void TimedCommandCallback(const ros::TimerEvent& e);

  std_msgs::Header time_pub_header_now;
  void OdometryCallback(const nav_msgs::OdometryConstPtr& odometry_msg);
  void TimedPublish(const ros::TimerEvent& e);
};
}

#endif // GEOMETRIC_POSITION_CONTROLLER_NODE_H
