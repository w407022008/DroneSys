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

#include <thread>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <mav_msgs/default_topics.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <drone_msgs/ControlCommand.h>
#include <quadrotor_common/geometry_eigen_conversions.h> 
#include <std_srvs/Empty.h>

#include "geometric_position_controller_node.h"

#include "vehicle_parameters_ros.h"

namespace geometric_position_controller {

GeometricPositionControllerNode::GeometricPositionControllerNode(){

  InitializeParams();
  
  ros::NodeHandle nh;

  cmd_active_sub_ = nh.subscribe(
      "command/active", 1,
      &GeometricPositionControllerNode::CommandActiveCallback, this);
  cmd_pose_sub_ = nh.subscribe(
      "command/pose", 1,
      &GeometricPositionControllerNode::CommandPoseCallback, this);
  cmd_trajectory_point_sub_ = nh.subscribe(
      "command/reference_state", 1,
      &GeometricPositionControllerNode::TrajecotryPointCallback, this);
  cmd_trajectory_sub_ = nh.subscribe(
      "command/trajectory", 1,
      &GeometricPositionControllerNode::TrajectoryCallback, this);
  cmd_multi_dof_joint_trajectory_sub_ = nh.subscribe(
      "command/MDJtrajectory", 1,
      &GeometricPositionControllerNode::MultiDofJointTrajectoryCallback, this);
  command_timer_ = nh.createTimer(ros::Duration(0), &GeometricPositionControllerNode::TimedCommandCallback, this,
                                  true, false);

  cmd_roll_pitch_yawrate_thrust_sub_ = nh.subscribe("command/roll_pitch_yawrate_thrust", 1,
                                     &GeometricPositionControllerNode::RollPitchYawrateThrustCallback, this);

  odometry_sub_ = nh.subscribe("odometry", 1,
                               &GeometricPositionControllerNode::OdometryCallback, this);


  rotors_motor_velocity_reference_pub_ = nh.advertise<mav_msgs::Actuators>(
      "command/motor_speed", 1);
  mavros_setpoint_raw_attitude_pub = nh.advertise<mavros_msgs::AttitudeTarget>(
      "/mavros/setpoint_raw/attitude", 1);
  drone_msg_pub = nh.advertise<drone_msgs::ControlCommand>("/drone_msg/control_command", 1);
  Command_to_pub.source = "geometry_controller";
  Command_to_pub.Command_ID = 0;
  if(control_frequency_>0.0)
    odometry_timer_ = nh.createTimer(ros::Duration(1.0/control_frequency_), &GeometricPositionControllerNode::TimedPublish, this);
  cmd_active_ = false;
}

GeometricPositionControllerNode::~GeometricPositionControllerNode() { }

void GeometricPositionControllerNode::InitializeParams() {

  ros::NodeHandle private_nh("~");
  // Read parameters from rosparam.
  GetRosParameter<float>(private_nh, "control_frequency", 200.0, &control_frequency_);
  GetRosParameter<bool>(private_nh, "rate_control", false, &rate_control_);
  GetRosParameter(private_nh, "position_gain/x",
                  geometric_position_controller_.controller_parameters_.position_gain_.x(),
                  &geometric_position_controller_.controller_parameters_.position_gain_.x());
  GetRosParameter(private_nh, "position_gain/y",
                  geometric_position_controller_.controller_parameters_.position_gain_.y(),
                  &geometric_position_controller_.controller_parameters_.position_gain_.y());
  GetRosParameter(private_nh, "position_gain/z",
                  geometric_position_controller_.controller_parameters_.position_gain_.z(),
                  &geometric_position_controller_.controller_parameters_.position_gain_.z());
  GetRosParameter(private_nh, "velocity_gain/x",
                  geometric_position_controller_.controller_parameters_.velocity_gain_.x(),
                  &geometric_position_controller_.controller_parameters_.velocity_gain_.x());
  GetRosParameter(private_nh, "velocity_gain/y",
                  geometric_position_controller_.controller_parameters_.velocity_gain_.y(),
                  &geometric_position_controller_.controller_parameters_.velocity_gain_.y());
  GetRosParameter(private_nh, "velocity_gain/z",
                  geometric_position_controller_.controller_parameters_.velocity_gain_.z(),
                  &geometric_position_controller_.controller_parameters_.velocity_gain_.z());
  GetRosParameter(private_nh, "attitude_gain/x",
                  geometric_position_controller_.controller_parameters_.attitude_gain_.x(),
                  &geometric_position_controller_.controller_parameters_.attitude_gain_.x());
  GetRosParameter(private_nh, "attitude_gain/y",
                  geometric_position_controller_.controller_parameters_.attitude_gain_.y(),
                  &geometric_position_controller_.controller_parameters_.attitude_gain_.y());
  GetRosParameter(private_nh, "attitude_gain/z",
                  geometric_position_controller_.controller_parameters_.attitude_gain_.z(),
                  &geometric_position_controller_.controller_parameters_.attitude_gain_.z());
  GetRosParameter(private_nh, "angular_rate_gain/x",
                  geometric_position_controller_.controller_parameters_.angular_rate_gain_.x(),
                  &geometric_position_controller_.controller_parameters_.angular_rate_gain_.x());
  GetRosParameter(private_nh, "angular_rate_gain/y",
                  geometric_position_controller_.controller_parameters_.angular_rate_gain_.y(),
                  &geometric_position_controller_.controller_parameters_.angular_rate_gain_.y());
  GetRosParameter(private_nh, "angular_rate_gain/z",
                  geometric_position_controller_.controller_parameters_.angular_rate_gain_.z(),
                  &geometric_position_controller_.controller_parameters_.angular_rate_gain_.z());

  GetVehicleParameters(private_nh, &geometric_position_controller_.vehicle_parameters_);

  geometric_position_controller_.InitializeParameters();

  commands_.clear();
}

void GeometricPositionControllerNode::CommandActiveCallback(const std_msgs::Bool& active){
  if(active.data) {cmd_active_ = true;}
  else {cmd_active_ = false;}
}

void GeometricPositionControllerNode::TimedPublish(const ros::TimerEvent& e) {
  if(!cmd_active_) return;
  // publish rotors speed by calculating inner loop control
  if(rotors_motor_velocity_reference_pub_.getNumSubscribers() > 0 ){
    Eigen::VectorXd ref_rotor_velocities;
    geometric_position_controller_.CalculateRotorVelocities(&ref_rotor_velocities);

    // Todo(ffurrer): Do this in the conversions header.
    mav_msgs::ActuatorsPtr actuator_msg(new mav_msgs::Actuators);

    actuator_msg->angular_velocities.clear();
    for (int i = 0; i < ref_rotor_velocities.size(); i++)
      actuator_msg->angular_velocities.push_back(ref_rotor_velocities[i]);
    actuator_msg->header.stamp = time_pub_header_now.stamp;

    rotors_motor_velocity_reference_pub_.publish(actuator_msg);
  }

  if(drone_msg_pub.getNumSubscribers() > 0){
    quadrotor_common::ControlCommand control_cmd = geometric_position_controller_.CalculateCommand();
    Command_to_pub.header.stamp = ros::Time::now();
    Command_to_pub.Command_ID = Command_to_pub.Command_ID + 1;
    if(rate_control_){
      Command_to_pub.Mode = drone_msgs::ControlCommand::Rate;
    }else{
      Command_to_pub.Mode = drone_msgs::ControlCommand::AttitudeRate;
      Command_to_pub.Attitude_sp.desired_att_q = quadrotor_common::eigenToGeometry(control_cmd.orientation);
    }
    Command_to_pub.Attitude_sp.body_rate = quadrotor_common::eigenToGeometry(control_cmd.bodyrates);

    Command_to_pub.Attitude_sp.desired_throttle = control_cmd.collective_thrust; // throttle [0,1] rather att_setpoint.thrust_body[]
    drone_msg_pub.publish(Command_to_pub);
  }
  // publish mavros attitude setpoint
  else if(mavros_setpoint_raw_attitude_pub.getNumSubscribers() > 0 ){
    quadrotor_common::ControlCommand control_cmd = geometric_position_controller_.CalculateCommand();
    mavros_msgs::AttitudeTarget att_setpoint;

    // Mappings: If any of these bits are set, the corresponding input should be ignored:
    // bit 1: body roll rate, bit 2: body pitch rate, bit 3: body yaw rate. 
    // bit 4: use hover thrust estimation, bit 5: reserved
    // bit 6: 3D body thrust sp instead of throttle, bit 7: throttle, bit 8: attitude
    if(rate_control_){
      att_setpoint.type_mask = 0b10010000; // only bodyrates setpoint
    }else{
      att_setpoint.type_mask = 0b00010000; // att_rate setpoint
      att_setpoint.orientation = quadrotor_common::eigenToGeometry(control_cmd.orientation);
    }
    // att_setpoint.type_mask = 0b00011111; // only attitude setpoint
    att_setpoint.body_rate = quadrotor_common::eigenToGeometry(control_cmd.bodyrates);

    att_setpoint.thrust = control_cmd.collective_thrust; // throttle [0,1] rather att_setpoint.thrust_body[]

    mavros_setpoint_raw_attitude_pub.publish(att_setpoint);
  }
}

void GeometricPositionControllerNode::RollPitchYawrateThrustCallback(
    const mav_msgs::RollPitchYawrateThrustConstPtr& roll_pitch_yawrate_thrust_reference_msg) {
  mav_msgs::EigenRollPitchYawrateThrust roll_pitch_yawrate_thrust;
  mav_msgs::eigenRollPitchYawrateThrustFromMsg(*roll_pitch_yawrate_thrust_reference_msg, &roll_pitch_yawrate_thrust);
  geometric_position_controller_.SetRollPitchYawrateThrust(roll_pitch_yawrate_thrust);
}

void GeometricPositionControllerNode::TrajecotryPointCallback(
    const quadrotor_msgs::TrajectoryPointConstPtr& msg) {

  quadrotor_common::TrajectoryPoint desired_state(*msg);

  Command_to_pub.Reference_State.position_ref[0] = desired_state.position.x();
  Command_to_pub.Reference_State.position_ref[1] = desired_state.position.y();
  Command_to_pub.Reference_State.position_ref[2] = desired_state.position.z();

  geometric_position_controller_.SetTrajectoryPoint(desired_state);
}

void GeometricPositionControllerNode::CommandPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& pose_msg) {

  quadrotor_common::TrajectoryPoint desired_state;
  desired_state.position = mav_msgs::vector3FromPointMsg(pose_msg->pose.position);
  desired_state.orientation = mav_msgs::quaternionFromMsg(pose_msg->pose.orientation);
  desired_state.velocity.setZero();
  desired_state.bodyrates.setZero();
  desired_state.acceleration.setZero();
  desired_state.heading = mav_msgs::yawFromQuaternion(desired_state.orientation);
  desired_state.heading_rate = (desired_state.orientation * desired_state.bodyrates).z();
  
  Command_to_pub.Reference_State.position_ref[0] = desired_state.position.x();
  Command_to_pub.Reference_State.position_ref[1] = desired_state.position.y();
  Command_to_pub.Reference_State.position_ref[2] = desired_state.position.z();
  
  geometric_position_controller_.SetTrajectoryPoint(desired_state);
}

void GeometricPositionControllerNode::TrajectoryCallback(
    const quadrotor_msgs::TrajectoryConstPtr& msg) {
  // Clear all pending commands.
  command_timer_.stop();
  commands_.clear();
  command_waiting_times_.clear();

  const size_t n_commands = msg->points.size();

  if(msg->type == msg->UNDEFINED || n_commands < 1){
    ROS_WARN_STREAM("Got MultiDOFJointTrajectory message, but message has no points.");
    return;
  }

  quadrotor_common::TrajectoryPoint desired_state(msg->points.front());
  commands_.push_front(desired_state);

  for (size_t i = 1; i < n_commands; ++i) {
    const quadrotor_msgs::TrajectoryPoint& reference_before = msg->points[i-1];
    const quadrotor_msgs::TrajectoryPoint& current_reference = msg->points[i];

    quadrotor_common::TrajectoryPoint desired_state(current_reference);

    commands_.push_back(desired_state);
    command_waiting_times_.push_back(current_reference.time_from_start - reference_before.time_from_start);
  }

  // We can trigger the first command immediately.
  geometric_position_controller_.SetTrajectoryPoint(commands_.front());
  commands_.pop_front();

  if (n_commands > 1) {
    command_timer_.setPeriod(command_waiting_times_.front());
    command_waiting_times_.pop_front();
    command_timer_.start();
  }
}

void GeometricPositionControllerNode::MultiDofJointTrajectoryCallback(
    const trajectory_msgs::MultiDOFJointTrajectoryConstPtr& msg) {
  // Clear all pending commands.
  command_timer_.stop();
  commands_.clear();
  command_waiting_times_.clear();

  const size_t n_commands = msg->points.size();

  if(n_commands < 1){
    ROS_WARN_STREAM("Got MultiDOFJointTrajectory message, but message has no points.");
    return;
  }

  mav_msgs::EigenTrajectoryPoint eigen_reference;
  mav_msgs::eigenTrajectoryPointFromMsg(msg->points.front(), &eigen_reference);
  quadrotor_common::TrajectoryPoint desired_state;
  desired_state.position = eigen_reference.position_W;
  desired_state.velocity = eigen_reference.velocity_W;
  desired_state.acceleration = eigen_reference.acceleration_W;
  desired_state.heading = eigen_reference.getYaw();
  desired_state.heading_rate = eigen_reference.getYawRate();
  commands_.push_front(desired_state);

  for (size_t i = 1; i < n_commands; ++i) {
    const trajectory_msgs::MultiDOFJointTrajectoryPoint& reference_before = msg->points[i-1];
    const trajectory_msgs::MultiDOFJointTrajectoryPoint& current_reference = msg->points[i];

    mav_msgs::eigenTrajectoryPointFromMsg(current_reference, &eigen_reference);
    quadrotor_common::TrajectoryPoint desired_state;
    desired_state.position = eigen_reference.position_W;
    desired_state.velocity = eigen_reference.velocity_W;
    desired_state.acceleration = eigen_reference.acceleration_W;
    desired_state.heading = eigen_reference.getYaw();
    desired_state.heading_rate = eigen_reference.getYawRate();

    commands_.push_back(desired_state);
    command_waiting_times_.push_back(current_reference.time_from_start - reference_before.time_from_start);
  }

  // We can trigger the first command immediately.
  geometric_position_controller_.SetTrajectoryPoint(commands_.front());
  commands_.pop_front();

  if (n_commands > 1) {
    command_timer_.setPeriod(command_waiting_times_.front());
    command_waiting_times_.pop_front();
    command_timer_.start();
  }
}

void GeometricPositionControllerNode::TimedCommandCallback(const ros::TimerEvent& e) {

  if(commands_.empty()){
    ROS_WARN("Commands empty, this should not happen here");
    return;
  }

  const quadrotor_common::TrajectoryPoint desired_state = commands_.front();
  geometric_position_controller_.SetTrajectoryPoint(desired_state);
  commands_.pop_front();
  command_timer_.stop();
  if(!command_waiting_times_.empty()){
    command_timer_.setPeriod(command_waiting_times_.front());
    command_waiting_times_.pop_front();
    command_timer_.start();
  }
}

void GeometricPositionControllerNode::OdometryCallback(const nav_msgs::OdometryConstPtr& odometry_msg) {

  ROS_INFO_ONCE("GeometricPositionController got first odometry message.");

  time_pub_header_now = odometry_msg->header;
  EigenOdometry odometry_cur_;
  eigenOdometryFromMsg(odometry_msg, &odometry_cur_);
  geometric_position_controller_.SetOdometry(odometry_cur_);
}

}

int main(int argc, char** argv) {
  ros::init(argc, argv, "geometric_position_controller_node");

  // Make Gazebo run correctly
  std_srvs::Empty srv;
  bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
  unsigned int i = 0;

  // Trying to unpause Gazebo for 10 seconds.
  while (i <= 10 && !unpaused) {
    ROS_INFO("Wait for 1 second before trying to unpause Gazebo again.");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    unpaused = ros::service::call("/gazebo/unpause_physics", srv);
    ++i;
  }

  if (!unpaused) {
    ROS_FATAL("Could not wake up Gazebo.");
    return -1;
  } else {
    ROS_INFO("Unpaused the Gazebo simulation.");
  }

  // Wait for 5 seconds to let the Gazebo GUI show up.
  ros::Duration(5.0).sleep();

  //ros::NodeHandle nh;
  geometric_position_controller::GeometricPositionControllerNode geometric_position_controller_node;

  ros::spin();

  return 0;
}
