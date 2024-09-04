#include <thread>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <mav_msgs/default_topics.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <drone_msgs/ControlCommand.h>
#include <quadrotor_common/math_common.h>
#include <quadrotor_common/parameter_helper.h>
#include <quadrotor_common/geometry_eigen_conversions.h>
#include <trajectory_generation_helper/polynomial_trajectory_helper.h>

#include <rpg_mpc/mpc_node.h>

namespace mpc_controller {

MPCControllerNode::MPCControllerNode(
                      const ros::NodeHandle& nh,
                      const ros::NodeHandle& pnh)
  : nh_(nh),
    pnh_(pnh),
    destructor_invoked_(true)
{
  if (!InitializeParams()) {
    ROS_ERROR("[%s] Could not load parameters.", pnh_.getNamespace().c_str());
    ros::shutdown();
    return;
  }

  cmd_active_sub_ = nh_.subscribe(
      "command/active", 1,
      &MPCControllerNode::CommandActiveCallback, this);
  cmd_pose_sub_ = nh_.subscribe(
      "command/pose", 1,
      &MPCControllerNode::CommandPoseCallback, this);
  cmd_trajectory_point_sub_ = nh_.subscribe(
      "command/reference_state", 1,
      &MPCControllerNode::TrajecotryPointCallback, this);
  cmd_trajectory_sub_ = nh_.subscribe(
      "command/trajectory", 1,
      &MPCControllerNode::TrajectoryCallback, this);

  cmd_roll_pitch_yawrate_thrust_sub_ = nh_.subscribe(
      "command/roll_pitch_yawrate_thrust", 1,
      &MPCControllerNode::RollPitchYawrateThrustCallback, this);

  odometry_sub_ = nh_.subscribe(
      "command/state_estimate", 1,
      &MPCControllerNode::OdometryCallback,
      this, ros::TransportHints().tcpNoDelay());

  control_command_pub_ = nh_.advertise<quadrotor_msgs::ControlCommand>("control_command", 1);
  drone_msg_pub = nh_.advertise<drone_msgs::ControlCommand>("/drone_msg/control_command", 1);
  Command_to_pub.source = "mpc";
  Command_to_pub.Command_ID = 0;
  mavros_setpoint_raw_attitude_pub = nh_.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 1);
  if(control_frequency_>0.0){
    odometry_timer_ = nh_.createTimer(ros::Duration(1.0/control_frequency_), &MPCControllerNode::TimedPublishCommand, this);
  }
}

MPCControllerNode::~MPCControllerNode() { 
  destructor_invoked_ = true;

  quadrotor_common::ControlCommand control_cmd;
  control_cmd.zero();
  quadrotor_msgs::ControlCommand control_cmd_msg;
  control_cmd_msg = control_cmd.toRosMessage();
  control_command_pub_.publish(control_cmd_msg);
}

bool MPCControllerNode::InitializeParams() {

  if (!quadrotor_common::getParam("control_frequency", control_frequency_, 200.0f, pnh_)){
    return false;
  }
  if (!quadrotor_common::getParam("poly_interpolation", poly_interpolation_, true, pnh_)){
    return false;
  }
  if (!quadrotor_common::getParam("rate_control", rate_control_, true, pnh_)){
    return false;
  }

  if (!mpc_controller_params.loadParameters(pnh_)) {
    return false;
  }

  return true;

}

void MPCControllerNode::CommandActiveCallback(const std_msgs::Bool& active){
  if(active.data) {destructor_invoked_ = false;}
  else {destructor_invoked_ = true;}
}

void MPCControllerNode::RollPitchYawrateThrustCallback(
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

void MPCControllerNode::TrajecotryPointCallback(
    const quadrotor_msgs::TrajectoryPointConstPtr& trajectory_point_msg) {
  if (destructor_invoked_) {
    return;
  }
  quadrotor_common::TrajectoryPoint desired_state(*trajectory_point_msg);
  // reference_trajectory_.points.clear();
  reference_trajectory_ = quadrotor_common::Trajectory(desired_state);
  // reference_trajectory_.points.front().time_from_start = ros::Duration(0.0);

  Command_to_pub.Reference_State.position_ref[0] = desired_state.position.x();
  Command_to_pub.Reference_State.position_ref[1] = desired_state.position.y();
  Command_to_pub.Reference_State.position_ref[2] = desired_state.position.z();
}

void MPCControllerNode::CommandPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& pose_msg) {
  if (destructor_invoked_) {
    return;
  }
        
  // reference_trajectory_.points.clear();

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

  if(poly_interpolation_){
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
  }else{
    reference_trajectory_ = quadrotor_common::Trajectory(desired_state);
  }
  time_start_trajectory_execution_ = ros::Time::now();    
}

void MPCControllerNode::TrajectoryCallback(
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

  reference_trajectory_ = *trajectory_msg;
}

void MPCControllerNode::OdometryCallback(
    const nav_msgs::OdometryConstPtr& odometry_msg) {
  if (destructor_invoked_) {
    return;
  }

  ROS_INFO_ONCE("MPCController got first odometry message.");

  odometry_ = quadrotor_common::QuadStateEstimate(*odometry_msg);
  
  odometry_state_.position = odometry_.position;
  odometry_state_.orientation = odometry_.orientation;
  odometry_state_.velocity = odometry_.velocity;
  odometry_state_.bodyrates = odometry_.bodyrates;
  odometry_state_.heading = quadrotor_common::quaternionToEulerAnglesZYX(odometry_.orientation).z();
}

void MPCControllerNode::TimedPublishCommand(const ros::TimerEvent& e){
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
    // if((reference_trajectory_.points.front().position - odometry_state_.position).norm() < 0.1){
    //   reference_trajectory_.points.clear();
    //   trajectory_ = quadrotor_common::Trajectory(odometry_state_);
    // }
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
    
  }
// std::cout<<trajectory_.points.front().position.transpose()<<" ";
  quadrotor_common::ControlCommand control_cmd = mpc_controller_.run(
      odometry_, trajectory_, mpc_controller_params);
// std::cout<<odometry_.position.transpose()<<std::endl;
  // publish control command to inner loop control
  if(control_command_pub_.getNumSubscribers() > 0 ){
    double control_command_delay_ = 0.001; // TODO set as a param
    control_cmd.timestamp = wall_time_now;
    control_cmd.expected_execution_time = wall_time_now + ros::Duration(control_command_delay_);
    
    quadrotor_msgs::ControlCommand control_cmd_msg;
    control_cmd_msg = control_cmd.toRosMessage();
    control_command_pub_.publish(control_cmd_msg);
  }

  if(drone_msg_pub.getNumSubscribers() > 0){
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
    mavros_msgs::AttitudeTarget att_setpoint;

    //Mappings: If any of these bits are set, the corresponding input should be ignored:
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

}

int main(int argc, char **argv) {
  ros::init(argc, argv, "mpc_controller_node");

  mpc_controller::MPCControllerNode controller_node_(ros::NodeHandle(), ros::NodeHandle("~"));

  ros::spin();

  return 0;
}
