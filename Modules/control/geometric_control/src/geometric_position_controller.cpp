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

#include "geometric_position_controller.h"

namespace geometric_position_controller {

GeometricPositionController::GeometricPositionController()
    : initialized_params_(false),
      controller_active_(false) {
  InitializeParameters();
}

GeometricPositionController::~GeometricPositionController() {}

void GeometricPositionController::InitializeParameters() {
  calculateAllocationMatrix(vehicle_parameters_.rotor_configuration_, &(controller_parameters_.allocation_matrix_));
  // To make the tuning independent of the inertia matrix we divide here.
  normalized_attitude_gain_ = controller_parameters_.attitude_gain_.transpose()
      * vehicle_parameters_.inertia_.inverse();
  // To make the tuning independent of the inertia matrix we divide here.
  normalized_angular_rate_gain_ = controller_parameters_.angular_rate_gain_.transpose()
      * vehicle_parameters_.inertia_.inverse();

  Eigen::Matrix4d I;
  I.setZero();
  I.block<3, 3>(0, 0) = vehicle_parameters_.inertia_;
  I(3, 3) = 1;
  angular_acc_to_rotor_velocities_.resize(vehicle_parameters_.rotor_configuration_.rotors.size(), 4);
  // Calculate the pseude-inverse A^{ \dagger} and then multiply by the inertia matrix I.
  // Pseudo-inverse A^{ \dagger} = A^T*(A*A^T)^{-1}
  angular_acc_to_rotor_velocities_ = controller_parameters_.allocation_matrix_.transpose()
      * (controller_parameters_.allocation_matrix_
      * controller_parameters_.allocation_matrix_.transpose()).inverse() * I;
  initialized_params_ = true;
}

void GeometricPositionController::SetOdometry(const EigenOdometry& odometry) {
  odometry_ = odometry;
}

void GeometricPositionController::SetRollPitchYawrateThrust(
    const mav_msgs::EigenRollPitchYawrateThrust& roll_pitch_yawrate_thrust) {
  roll_pitch_yawrate_thrust_ = roll_pitch_yawrate_thrust;
  controller_active_ = true;
  controller_mode_ = CommandModes::EigenRollPitchYawrateThrust;
}

void GeometricPositionController::SetTrajectoryPoint(
    const quadrotor_common::TrajectoryPoint& command_trajectory_point) {
  command_trajectory_point_ = command_trajectory_point;

  controller_active_ = true;
  controller_mode_ = CommandModes::TrajectoryPoint;
}

quadrotor_common::ControlCommand GeometricPositionController::CalculateCommand(){
  assert(initialized_params_);
  quadrotor_common::ControlCommand command;
  command.armed = true;
  
  // Return 0 thrust, until the first command is received.
  if (!controller_active_) {
    command.collective_thrust = 0.0;
    return command;
  }

  Eigen::Vector3d acceleration_des;
  Eigen::Matrix3d R_des;
  Eigen::Vector3d bodyrates_des;
  Eigen::Vector3d angular_acceleration_des;
  switch (controller_mode_)
  {
    case CommandModes::EigenRollPitchYawrateThrust:{
        command.collective_thrust = roll_pitch_yawrate_thrust_.thrust.z();

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        Eigen::Quaterniond quaternion_(R_des);
        command.orientation = quaternion_;

        ComputeDesiredBodyrates(R_des, &bodyrates_des);
        command.bodyrates = bodyrates_des;

        ComputeDesiredAngularAcc(bodyrates_des, R_des, &angular_acceleration_des);
        command.angular_accelerations = angular_acceleration_des;

      break;
    }
    case CommandModes::TrajectoryPoint:{
        ComputeDesiredAcceleration(&acceleration_des);
        // Project thrust onto body z axis.
        command.collective_thrust = acceleration_des.dot(odometry_.orientation.toRotationMatrix().col(2));

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        Eigen::Quaterniond quaternion_(R_des);
        command.orientation = quaternion_;

        ComputeDesiredBodyrates(R_des, &bodyrates_des);
        command.bodyrates = bodyrates_des;

        ComputeDesiredAngularAcc(bodyrates_des, R_des, &angular_acceleration_des);
        command.angular_accelerations = angular_acceleration_des;

      break;
    }
    default:
      break;
  }
  return command;
}

void GeometricPositionController::CalculateQuaternionThrust(Eigen::Quaterniond* quaternion, double* thrust) const {
  assert(quaternion);
  assert(thrust);
  assert(initialized_params_);

  // Return 0 thrust, until the first command is received.
  if (!controller_active_) {
    *thrust = 0.0;
    return;
  }

  Eigen::Vector3d acceleration_des;
  Eigen::Matrix3d R_des;
  switch (controller_mode_)
  {
    case CommandModes::EigenRollPitchYawrateThrust:{
        *thrust = roll_pitch_yawrate_thrust_.thrust.z();

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        Eigen::Quaterniond quaternion_(R_des);
        *quaternion = quaternion_;

      break;
    }
    case CommandModes::TrajectoryPoint:{
        ComputeDesiredAcceleration(&acceleration_des);
        // Project thrust onto body z axis.
        *thrust = acceleration_des.dot(odometry_.orientation.toRotationMatrix().col(2));

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        Eigen::Quaterniond quaternion_(R_des);
        *quaternion = quaternion_;

      break;
    }
    default:
      break;
  }
}

void GeometricPositionController::CalculateBodyrateThrust(Eigen::Vector3d* bodyrates, double* thrust) const {
  assert(bodyrates);
  assert(thrust);
  assert(initialized_params_);

  // Return 0 thrust, until the first command is received.
  if (!controller_active_) {
    *thrust = 0.0;
    return;
  }

  Eigen::Vector3d acceleration_des;
  Eigen::Matrix3d R_des;
  switch (controller_mode_)
  {
    case CommandModes::EigenRollPitchYawrateThrust:{
        *thrust = roll_pitch_yawrate_thrust_.thrust.z();

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        ComputeDesiredBodyrates(R_des, bodyrates);

      break;
    }
    case CommandModes::TrajectoryPoint:{
        ComputeDesiredAcceleration(&acceleration_des);
        // Project thrust onto body z axis.
        *thrust = acceleration_des.dot(odometry_.orientation.toRotationMatrix().col(2));

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        ComputeDesiredBodyrates(R_des, bodyrates);

      break;
    }
    default:
      break;
  }
}

void GeometricPositionController::CalculateRateAccThrust(Eigen::Vector3d* angular_acceleration, double* thrust) const {
  assert(angular_acceleration);
  assert(thrust);
  assert(initialized_params_);

  // Return 0 thrust, until the first command is received.
  if (!controller_active_) {
    *thrust = 0.0;
    return;
  }

  Eigen::Vector3d acceleration_des;
  Eigen::Vector3d bodyrates_des;
  Eigen::Matrix3d R_des;
  switch (controller_mode_)
  {
    case CommandModes::EigenRollPitchYawrateThrust:{
        *thrust = roll_pitch_yawrate_thrust_.thrust.z();

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        ComputeDesiredBodyrates(R_des, &bodyrates_des);
        ComputeDesiredAngularAcc(bodyrates_des, R_des, angular_acceleration);

      break;
    }
    case CommandModes::TrajectoryPoint:{
        ComputeDesiredAcceleration(&acceleration_des);
        // Project thrust onto body z axis.
        *thrust = acceleration_des.dot(odometry_.orientation.toRotationMatrix().col(2));

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        ComputeDesiredBodyrates(R_des, &bodyrates_des);
        ComputeDesiredAngularAcc(bodyrates_des, R_des, angular_acceleration);

      break;
    }
    default:
      break;
  }
}

void GeometricPositionController::CalculateRotorVelocities(Eigen::VectorXd* rotor_velocities) const {
  assert(rotor_velocities);
  assert(initialized_params_);

  rotor_velocities->resize(vehicle_parameters_.rotor_configuration_.rotors.size());
  // Return 0 velocities on all rotors, until the first command is received.
  if (!controller_active_) {
    *rotor_velocities = Eigen::VectorXd::Zero(rotor_velocities->rows());
    return;
  }

  Eigen::Vector3d acceleration_des;
  Eigen::Vector4d angular_acceleration_thrust;
  Eigen::Vector3d angular_acceleration_des, bodyrates_des;
  Eigen::Matrix3d R_des;
  switch (controller_mode_)
  {
    case CommandModes::EigenRollPitchYawrateThrust:{
        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        ComputeDesiredBodyrates(R_des, &bodyrates_des);
        ComputeDesiredAngularAcc(bodyrates_des, R_des, &angular_acceleration_des);

        angular_acceleration_thrust.block<3, 1>(0, 0) = angular_acceleration_des;
        angular_acceleration_thrust(3) = roll_pitch_yawrate_thrust_.thrust.z();

      break;
    }
    case CommandModes::TrajectoryPoint:{
        ComputeDesiredAcceleration(&acceleration_des);

        ComputeDesiredRotationMatrix(acceleration_des, &R_des);
        ComputeDesiredBodyrates(R_des, &bodyrates_des);
// static int cnt = 0;
// if(bodyrates_des.norm() > 0.01){
//   std::cout << bodyrates_des << std::endl << std::endl;
//   cnt++;
// }else if(cnt>0){
//   std::cout<<cnt<<std::endl;
//   cnt = 0;
// }
        ComputeDesiredAngularAcc(bodyrates_des, R_des, &angular_acceleration_des);

        // Project thrust onto body z axis.
        double thrust = acceleration_des.dot(odometry_.orientation.toRotationMatrix().col(2));

        angular_acceleration_thrust.block<3, 1>(0, 0) = angular_acceleration_des;
        angular_acceleration_thrust(3) = thrust;

      break;
    }
    default:
      break;
  }

  angular_acceleration_thrust(3) = vehicle_parameters_.mass_ * angular_acceleration_thrust(3);
  // [/tau, T].trans() = G * u^2
  *rotor_velocities = angular_acc_to_rotor_velocities_ * angular_acceleration_thrust;
  *rotor_velocities = rotor_velocities->cwiseMax(Eigen::VectorXd::Zero(rotor_velocities->rows()));
  *rotor_velocities = rotor_velocities->cwiseSqrt();
}

void GeometricPositionController::ComputeDesiredAcceleration(Eigen::Vector3d* acceleration) const {
  assert(acceleration);

  Eigen::Vector3d position_error;
  position_error = command_trajectory_point_.position - odometry_.position;

  // Transform velocity to world frame.
  const Eigen::Matrix3d R_W_I = odometry_.orientation.toRotationMatrix();
  Eigen::Vector3d velocity_W =  R_W_I * odometry_.velocity;
  Eigen::Vector3d velocity_error;
  velocity_error = command_trajectory_point_.velocity - velocity_W;

  Eigen::Vector3d e_3(Eigen::Vector3d::UnitZ());

  Eigen::Vector3d drag_accelerations_in_bodyframe = Eigen::Vector3d::Zero();
  // TODO calculate air drag
  *acceleration = position_error.cwiseProduct(controller_parameters_.position_gain_)
      + velocity_error.cwiseProduct(controller_parameters_.velocity_gain_)
      + vehicle_parameters_.gravity_ * e_3 + command_trajectory_point_.acceleration
      - drag_accelerations_in_bodyframe;
}

// Implementation from the T. Lee et al. paper
// Control of complex maneuvers for a quadrotor UAV using geometric methods on SE(3)
void GeometricPositionController::ComputeDesiredRotationMatrix(const Eigen::Vector3d& acceleration,
                                                            Eigen::Matrix3d* R_des) const {
  assert(R_des);
  Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();
  switch (controller_mode_)
  {
    case CommandModes::EigenRollPitchYawrateThrust:{
      double yaw = atan2(R(1, 0), R(0, 0));

      // Get the desired rotation matrix.
      *R_des = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())  // yaw
            * Eigen::AngleAxisd(roll_pitch_yawrate_thrust_.roll, Eigen::Vector3d::UnitX())  // roll
            * Eigen::AngleAxisd(roll_pitch_yawrate_thrust_.pitch, Eigen::Vector3d::UnitY());  // pitch

      break;
    }
    case CommandModes::TrajectoryPoint:{
      // Get the desired rotation matrix.
      Eigen::Vector3d b1_des;
      double yaw = command_trajectory_point_.heading;
      b1_des << cos(yaw), sin(yaw), 0;

      Eigen::Vector3d b3_des;
      double almostZero = 0.1;
      if (acceleration.norm() < almostZero) {
        // In case of free fall we keep the thrust direction to be the current one
        // This only works assuming that we are in this condition for a very short
        // time (otherwise attitude drifts)
        b3_des = R * Eigen::Vector3d::UnitZ();
      } else {
        b3_des = acceleration.normalized();
      }

      Eigen::Vector3d b2_des;
      b2_des = b3_des.cross(b1_des);
      if (b2_des.norm() < almostZero) {
        // if cross(b3, b1) == 0, they are collinear =>
        // every b2_des lies automatically in the b2 - b3 plane

        // Project estimated body y-axis into the b2 - b3 plane
        const Eigen::Vector3d b2_estimated = R * Eigen::Vector3d::UnitY();
        const Eigen::Vector3d b2_projected =
            b2_estimated - (b2_estimated.dot(b1_des)) * b1_des;
        if (b2_projected.norm() < almostZero) {
          // Not too much intelligent stuff we can do in this case but it should
          // basically never occur
          b2_des = b1_des;
        } else {
          b2_des = b2_projected.normalized();
        }
      } else {
        b2_des.normalize();
      }

      R_des->col(0) = b2_des.cross(b3_des);
      R_des->col(1) = b2_des;
      R_des->col(2) = b3_des;

      break;
    }
    default:
      break;
  }
}

void GeometricPositionController::ComputeDesiredBodyrates(const Eigen::Matrix3d& desired_R, 
                                                      Eigen::Vector3d* desired_bodyrates) const {
  assert(desired_bodyrates);

  Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();
  // Angle error according to lee et al.
  Eigen::Matrix3d angle_error_matrix = 0.5 * (desired_R.transpose() * R - R.transpose() * desired_R);
  vectorFromSkewMatrix(angle_error_matrix, desired_bodyrates);
  // *desired_bodyrates = (*desired_bodyrates).cwiseProduct(controller_parameters_.attitude_gain_);
}

void GeometricPositionController::ComputeDesiredAngularAcc(const Eigen::Vector3d& attitude_error_vector,
                                                     const Eigen::Matrix3d& R_des,
                                                     Eigen::Vector3d* desired_angular_acceleration) const {
  assert(desired_angular_acceleration);

  // TODO(burrimi) include angular rate references at some point.
  Eigen::Vector3d body_rate_des;
  switch (controller_mode_)
  {
    case CommandModes::EigenRollPitchYawrateThrust:{
      body_rate_des[2] = roll_pitch_yawrate_thrust_.yaw_rate;

      break;
    }
    case CommandModes::TrajectoryPoint:{
      body_rate_des = command_trajectory_point_.bodyrates;
      body_rate_des[2] = command_trajectory_point_.heading_rate;

      break;
    }
    default:
      break;
  }

  Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix().transpose() * R_des;
  Eigen::Vector3d bodyrates_des_proj = R * body_rate_des;
  Eigen::Vector3d bodyrate_error = odometry_.angular_velocity - bodyrates_des_proj;

  // Eigen::Matrix3d inertia_inv_ = vehicle_parameters_.inertia_.inverse();
  // *desired_angular_acceleration = -1 * inertia_inv_ * attitude_error_vector.cwiseProduct(controller_parameters_.attitude_gain_)
  //                          - inertia_inv_ * bodyrate_error.cwiseProduct(controller_parameters_.angular_rate_gain_)
  //                          + odometry_.angular_velocity.cross(odometry_.angular_velocity); // we don't need the inertia matrix here
  *desired_angular_acceleration = -1 * attitude_error_vector.cwiseProduct(normalized_attitude_gain_)
                           - bodyrate_error.cwiseProduct(normalized_angular_rate_gain_)
                           + odometry_.angular_velocity.cross(odometry_.angular_velocity); // we don't need the inertia matrix here
}
}
