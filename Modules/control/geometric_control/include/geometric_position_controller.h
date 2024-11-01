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

#ifndef ROTORS_CONTROL_GEOMETRIC_POSITION_CONTROLLER_H
#define ROTORS_CONTROL_GEOMETRIC_POSITION_CONTROLLER_H

#include <mav_msgs/conversions.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <quadrotor_common/trajectory_point.h>
#include <quadrotor_common/control_command.h>

#include "common.h"
#include "vehicle_parameters.h"

namespace geometric_position_controller {

// Default values for the geometric position controller and the Asctec Firefly.
static const float kDefaultControlFrequency = 0.0;
static const Eigen::Vector3d kDefaultPositionGain = Eigen::Vector3d(6, 6, 6);
static const Eigen::Vector3d kDefaultVelocityGain = Eigen::Vector3d(4.7, 4.7, 4.7);
static const Eigen::Vector3d kDefaultAttitudeGain = Eigen::Vector3d(3, 3, 0.035);
static const Eigen::Vector3d kDefaultAngularRateGain = Eigen::Vector3d(0.52, 0.52, 0.025);

class GeometricPositionControllerParameters {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GeometricPositionControllerParameters()
      : position_gain_(kDefaultPositionGain),
        velocity_gain_(kDefaultVelocityGain),
        attitude_gain_(kDefaultAttitudeGain),
        angular_rate_gain_(kDefaultAngularRateGain),
        control_frequency_(kDefaultControlFrequency) {
    calculateAllocationMatrix(rotor_configuration_, &allocation_matrix_);
  }

  float control_frequency_;
  Eigen::Matrix4Xd allocation_matrix_;
  Eigen::Vector3d position_gain_;
  Eigen::Vector3d velocity_gain_;
  Eigen::Vector3d attitude_gain_;
  Eigen::Vector3d angular_rate_gain_;
  RotorConfiguration rotor_configuration_;
};

class GeometricPositionController {
 public:
  GeometricPositionController();
  ~GeometricPositionController();
  void InitializeParameters();
  quadrotor_common::ControlCommand CalculateCommand();
  void CalculateQuaternionThrust(Eigen::Quaterniond* quaternion, double* thrust) const;
  void CalculateBodyrateThrust(Eigen::Vector3d* bodyrates, double* thrust) const;
  void CalculateRateAccThrust(Eigen::Vector3d* angular_acceleration, double* thrust) const;
  void CalculateRotorVelocities(Eigen::VectorXd* rotor_velocities) const;

  void SetOdometry(const EigenOdometry& odometry);
  void SetTrajectoryPoint(
    const quadrotor_common::TrajectoryPoint& command_trajectory_point);

  void SetRollPitchYawrateThrust(
      const mav_msgs::EigenRollPitchYawrateThrust& roll_pitch_yawrate_thrust);

  GeometricPositionControllerParameters controller_parameters_;
  VehicleParameters vehicle_parameters_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  enum class CommandModes {
    TrajectoryPoint,
    EigenRollPitchYawrateThrust
  } controller_mode_;

  bool initialized_params_;
  bool controller_active_;

  Eigen::Vector3d normalized_attitude_gain_;
  Eigen::Vector3d normalized_angular_rate_gain_;
  Eigen::MatrixX4d angular_acc_to_rotor_velocities_;

  quadrotor_common::TrajectoryPoint command_trajectory_point_;
  mav_msgs::EigenRollPitchYawrateThrust roll_pitch_yawrate_thrust_;
  EigenOdometry odometry_;

  void ComputeDesiredAcceleration(Eigen::Vector3d* acceleration) const;
  void ComputeDesiredRotationMatrix(const Eigen::Vector3d& acceleration,
                                Eigen::Matrix3d* R_des) const;
  void ComputeDesiredBodyrates(const Eigen::Matrix3d& desired_R,
                                Eigen::Vector3d* bodyrates) const;
  void ComputeDesiredAngularAcc(const Eigen::Vector3d& attitude_error_vector,
                                const Eigen::Matrix3d& R_des,
                                Eigen::Vector3d* angular_acceleration) const;
};
}

#endif // ROTORS_CONTROL_GEOMETRIC_POSITION_CONTROLLER_H
