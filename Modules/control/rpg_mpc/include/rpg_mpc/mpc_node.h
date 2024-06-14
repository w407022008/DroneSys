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

#ifndef MPC_CONTROLLER_NODE_H
#define MPC_CONTROLLER_NODE_H

#include <boost/bind.hpp>
#include <Eigen/Eigen>
#include <stdio.h>
#include <atomic>

#include <geometry_msgs/PoseStamped.h>
#include <mav_msgs/RollPitchYawrateThrust.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/Trajectory.h>
#include <quadrotor_msgs/TrajectoryPoint.h>
#include <quadrotor_common/trajectory_point.h>
#include <quadrotor_common/quad_state_estimate.h>
#include <quadrotor_msgs/ControlCommand.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>

#include <rpg_mpc/mpc_controller.h>
#include <rpg_mpc/mpc_params.h>

namespace mpc_controller {

class MPCControllerNode {
 public:
  MPCControllerNode(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  ~MPCControllerNode();

  bool InitializeParams();
  void Publish();

 private: 
  rpg_mpc::MpcController<float> mpc_controller_;
  rpg_mpc::MpcParams<float> mpc_controller_params;

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  std::atomic_bool destructor_invoked_;
  float control_frequency_;
  bool poly_interpolation_;
  bool rate_control_;

  // subscribers
  ros::Subscriber cmd_trajectory_sub_;
  ros::Subscriber cmd_pose_sub_;
  ros::Subscriber cmd_trajectory_point_sub_;
  ros::Subscriber cmd_roll_pitch_yawrate_thrust_sub_;
  ros::Subscriber odometry_sub_;

  ros::Publisher control_command_pub_;
  ros::Publisher mavros_setpoint_raw_attitude_pub;

  quadrotor_common::Trajectory trajectory_;
  quadrotor_common::TrajectoryPoint odometry_state_;
  quadrotor_common::QuadStateEstimate odometry_;
  quadrotor_common::Trajectory reference_trajectory_;

  ros::Timer odometry_timer_;
  ros::Time time_start_trajectory_execution_;

  void RollPitchYawrateThrustCallback(
      const mav_msgs::RollPitchYawrateThrustConstPtr& roll_pitch_yawrate_thrust_reference_msg);
  
  void TrajecotryPointCallback(
      const quadrotor_msgs::TrajectoryPointConstPtr& trajectory_point_msg);
  void CommandPoseCallback(
      const geometry_msgs::PoseStampedConstPtr& pose_msg);
    
  void TrajectoryCallback(
      const quadrotor_msgs::TrajectoryConstPtr& trajectory_msg);

  void OdometryCallback(const nav_msgs::OdometryConstPtr& odometry_msg);

  void TimedPublishCommand(const ros::TimerEvent& e);
};
}

#endif // MPC_CONTROLLER_NODE_H
