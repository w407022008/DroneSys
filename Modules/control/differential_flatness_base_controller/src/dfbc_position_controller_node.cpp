#include <thread>
#include <chrono>

#include <ros/ros.h>
#include <mav_msgs/default_topics.h>
#include <quadrotor_common/math_common.h>
#include <quadrotor_common/parameter_helper.h>
#include <quadrotor_common/geometry_eigen_conversions.h>
#include <trajectory_generation_helper/polynomial_trajectory_helper.h>

#include <dfbc_position_controller_node.h>

namespace dfbc_position_controller {

DFBCPositionControllerNode::DFBCPositionControllerNode(
                                           const ros::NodeHandle& nh,
                                           const ros::NodeHandle& pnh)
  : nh_(nh),
    pnh_(pnh),
    destructor_invoked_(false)
{
  if (!InitializeParams()) {
    ROS_ERROR("[%s] Could not load parameters.", pnh_.getNamespace().c_str());
    ros::shutdown();
    return;
  }

  cmd_pose_sub_ = nh_.subscribe(
      "command/pose", 1,
      &DFBCPositionControllerNode::CommandPoseCallback, this);
  cmd_trajectory_point_sub_ = nh_.subscribe(
      "autopilot/reference_state", 1,
      &DFBCPositionControllerNode::TrajecotryPointCallback, this);
  cmd_trajectory_sub_ = nh_.subscribe(
      "autopilot/trajectory", 1,
      &DFBCPositionControllerNode::TrajectoryCallback, this);

  cmd_roll_pitch_yawrate_thrust_sub_ = nh_.subscribe(
      "command/roll_pitch_yawrate_thrust", 1,
      &DFBCPositionControllerNode::RollPitchYawrateThrustCallback, this);

  odometry_sub_ = nh_.subscribe(
      "autopilot/state_estimate", 1,
      &DFBCPositionControllerNode::OdometryCallback,
      this, ros::TransportHints().tcpNoDelay());

  control_command_pub_ = nh_.advertise<quadrotor_msgs::ControlCommand>("control_command", 1);

  if(control_frequency_>0.0){
    odometry_timer_ = nh_.createTimer(ros::Duration(1.0/control_frequency_), &DFBCPositionControllerNode::TimedPublishCommand, this);
  }
}

DFBCPositionControllerNode::~DFBCPositionControllerNode() { 
  destructor_invoked_ = true;

  quadrotor_common::ControlCommand control_cmd;
  control_cmd.zero();
  quadrotor_msgs::ControlCommand control_cmd_msg;
  control_cmd_msg = control_cmd.toRosMessage();
  control_command_pub_.publish(control_cmd_msg);
}

bool DFBCPositionControllerNode::InitializeParams() {

  if (!quadrotor_common::getParam("control_frequency", control_frequency_, 200.0f, pnh_)){
    return false;
  }

  if (!position_controller_params.loadParameters(pnh_)) {
    return false;
  }

  return true;

}

void DFBCPositionControllerNode::RollPitchYawrateThrustCallback(
    const mav_msgs::RollPitchYawrateThrustConstPtr& msg) {
  if (destructor_invoked_) {
    return;
  }
  Eigen::Vector3d ypr(0.0, msg->pitch, msg->roll);
  // double roll = msg->roll;
  // double pitch = msg->pitch;
  // double yawrate = msg->yaw_rate;
  // Eigen::Vector3d thrust = quadrotor_common::geometryToEigen(msg->thrust);

  reference_trajectory_.points.clear();
  quadrotor_common::TrajectoryPoint desired_state;
  desired_state.position.setZero();
  desired_state.orientation = quadrotor_common::eulerAnglesZYXToQuaternion(ypr);
  desired_state.velocity.setZero();
  desired_state.bodyrates.z() = msg->yaw_rate;
  desired_state.acceleration.setZero();
  desired_state.heading = quadrotor_common::quaternionToEulerAnglesZYX(odometry_.orientation).z();
  desired_state.heading_rate = msg->yaw_rate;
  desired_state.thrust = msg->thrust.z;

  reference_trajectory_ = quadrotor_common::Trajectory(desired_state);
}

void DFBCPositionControllerNode::TrajecotryPointCallback(
    const quadrotor_msgs::TrajectoryPointConstPtr& trajectory_point_msg) {
  if (destructor_invoked_) {
    return;
  }
  reference_trajectory_.points.clear();
  reference_trajectory_ = quadrotor_common::Trajectory(
                          quadrotor_common::TrajectoryPoint(*trajectory_point_msg));
  reference_trajectory_.points.front().time_from_start = ros::Duration(0.0);
}

void DFBCPositionControllerNode::CommandPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& pose_msg) {
  if (destructor_invoked_) {
    return;
  }
        
  // reference_trajectory_.points.clear();

  quadrotor_common::TrajectoryPoint desired_state;
  desired_state.position = quadrotor_common::geometryToEigen(
                          pose_msg->pose.position);
  desired_state.orientation = quadrotor_common::geometryToEigen(
                          pose_msg->pose.orientation);
  desired_state.velocity.setZero();
  desired_state.bodyrates.setZero();
  desired_state.acceleration.setZero();
  desired_state.heading = odometry_state_.heading;
  desired_state.heading_rate = (desired_state.orientation * desired_state.bodyrates).z();
  
  quadrotor_common::Trajectory go_to_pose_traj =
            trajectory_generation_helper::polynomials::
                computeTimeOptimalTrajectory(
                    odometry_state_, desired_state,
                    5, //kGoToPosePolynomialOrderOfContinuity_
                    1.5, //go_to_pose_max_velocity_
                    12, //go_to_pose_max_normalized_thrust_
                    0.5, //go_to_pose_max_roll_pitch_rate_
                    50.0);//kGoToPoseTrajectorySamplingFrequency_

  // trajectory_generation_helper::heading::addConstantHeadingRate(
  //     odometry_state_.heading, desired_state.heading, &go_to_pose_traj);

  reference_trajectory_ = go_to_pose_traj;

  time_start_trajectory_execution_ = ros::Time::now();    
}

void DFBCPositionControllerNode::TrajectoryCallback(
    const quadrotor_msgs::TrajectoryConstPtr& trajectory_msg) {
  if (destructor_invoked_) {
    return;
  }

  time_start_trajectory_execution_ = ros::Time::now();

  const size_t n_commands = trajectory_msg->points.size();

  if(trajectory_msg->type == trajectory_msg->UNDEFINED || n_commands < 1){
    ROS_WARN_STREAM("Got MultiDOFJointTrajectory message, but message has no points.");
    return;
  }

  quadrotor_common::Trajectory go_to_pose_traj =
            trajectory_generation_helper::polynomials::
                computeTimeOptimalTrajectory(
                    odometry_state_, trajectory_msg->points.back(),
                    5, //kGoToPosePolynomialOrderOfContinuity_
                    1.5, //go_to_pose_max_velocity_
                    12, //go_to_pose_max_normalized_thrust_
                    0.5, //go_to_pose_max_roll_pitch_rate_
                    50.0);//kGoToPoseTrajectorySamplingFrequency_

  // trajectory_generation_helper::heading::addConstantHeadingRate(
  //     odometry_state_.heading, desired_state.heading, &go_to_pose_traj);

  reference_trajectory_ = go_to_pose_traj;
}

void DFBCPositionControllerNode::OdometryCallback(
    const nav_msgs::OdometryConstPtr& odometry_msg) {
  if (destructor_invoked_) {
    return;
  }

  ROS_INFO_ONCE("DFBCPositionController got first odometry message.");

  odometry_ = quadrotor_common::QuadStateEstimate(*odometry_msg);
  
  odometry_state_.position = odometry_.position;
  odometry_state_.orientation = odometry_.orientation;
  odometry_state_.velocity = odometry_.velocity;
  odometry_state_.bodyrates = odometry_.bodyrates;
  odometry_state_.heading = quadrotor_common::quaternionToEulerAnglesZYX(odometry_.orientation).z();
}

void DFBCPositionControllerNode::TimedPublishCommand(const ros::TimerEvent& e){
  if (destructor_invoked_) {
    return;
  }

  ros::Time wall_time_now = ros::Time::now();
  const size_t n_commands = reference_trajectory_.points.size();

  // One point trajectory used to compute control command
  if(n_commands == 0){
    // trajectory_ = quadrotor_common::Trajectory(odometry_state_);
  }else if(n_commands == 1){
    trajectory_ = reference_trajectory_;
    if((reference_trajectory_.points.front().position - odometry_state_.position).norm() < 0.1){
      reference_trajectory_.points.clear();
      // trajectory_ = quadrotor_common::Trajectory(odometry_state_);
    }
  }else{
    // tracking a trajectory whose points with TimeStamp
    const ros::Duration dt = ros::Time::now() - time_start_trajectory_execution_;
    static ros::Time last_time_start_trajectory_execution_ = time_start_trajectory_execution_;

    // Check wether we reached our lookahead.
    // Use boolen flag to also break the outer loop.
    size_t i = n_commands;
    static quadrotor_common::TrajectoryPoint left = reference_trajectory_.points.front();
    static quadrotor_common::TrajectoryPoint right = reference_trajectory_.points.front();
    if(time_start_trajectory_execution_!=last_time_start_trajectory_execution_){
      last_time_start_trajectory_execution_ = time_start_trajectory_execution_;
      right.time_from_start = ros::Duration(0.0);
    }
    while(right.time_from_start.toSec() < dt.toSec()){
      left = right;
      reference_trajectory_.points.pop_front();
      i--;
      if(i == 0){
        break;
      }
      right = reference_trajectory_.points.front();
    }
    // Add a point if the time corresponds to a sample on the lookahead.
    trajectory_ = quadrotor_common::Trajectory(right);
// std::cout<<right.time_from_start.toSec()<<" "<< dt.toSec()<<" ";
  }
// std::cout
//         <<reference_trajectory_.points.size()<<" "
//         <<trajectory_.points.front().position.transpose()<<std::endl;
  quadrotor_common::ControlCommand control_cmd = position_controller_.run(
      odometry_, trajectory_, position_controller_params);
// std::cout<<control_cmd.armed<<" "<<control_cmd.collective_thrust<<std::endl;
  double control_command_delay_ = 0.001; // TODO set as a param
  control_cmd.timestamp = wall_time_now;
  control_cmd.expected_execution_time = wall_time_now + ros::Duration(control_command_delay_);
  
  quadrotor_msgs::ControlCommand control_cmd_msg;
  control_cmd_msg = control_cmd.toRosMessage();
  control_command_pub_.publish(control_cmd_msg);
}

}

int main(int argc, char **argv) {
  ros::init(argc, argv, "dfbc_position_controller_node");

  dfbc_position_controller::DFBCPositionControllerNode controller_node_(ros::NodeHandle(), ros::NodeHandle("~"));

  ros::spin();

  return 0;
}
