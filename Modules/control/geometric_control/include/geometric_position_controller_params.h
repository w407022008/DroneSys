#pragma once

#include <quadrotor_common/parameter_helper.h>
#include <mav_msgs/eigen_mav_msgs.h>

#include "parameters.h"

namespace geometric_position_controller {

// Default values for the geometric position controller and the Asctec Firefly.
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
        angular_rate_gain_(kDefaultAngularRateGain) {
    calculateAllocationMatrix(rotor_configuration_, &allocation_matrix_);
  }

  Eigen::Matrix4Xd allocation_matrix_;
  Eigen::Vector3d position_gain_;
  Eigen::Vector3d velocity_gain_;
  Eigen::Vector3d attitude_gain_;
  Eigen::Vector3d angular_rate_gain_;
  RotorConfiguration rotor_configuration_;
};

class GeometricPositionControllerParams {
 public:
  GeometricPositionControllerParams()
      : use_att_mode(true),
        position_gain_(kDefaultPositionGain),
        velocity_gain_(kDefaultVelocityGain),
        attitude_gain_(kDefaultAttitudeGain),
        angular_rate_gain_(kDefaultAngularRateGain) {}

  ~GeometricPositionControllerParams() {}

  bool loadParameters(const ros::NodeHandle& pnh) {
    const std::string path_rel_to_node = "position_controller";

    if (!quadrotor_common::getParam(path_rel_to_node + "/use_att_mode",
                                    use_att_mode, pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam(path_rel_to_node + "/position_gain/x", controller_parameters_.position_gain_.x(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/position_gain/y", controller_parameters_.position_gain_.y(), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam(path_rel_to_node + "/position_gain/z", controller_parameters_.position_gain_.z(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/velocity_gain/x", controller_parameters_.velocity_gain_.x(), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam(path_rel_to_node + "/velocity_gain/y", controller_parameters_.velocity_gain_.y(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/velocity_gain/z", controller_parameters_.velocity_gain_.z(), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam(path_rel_to_node + "/attitude_gain/x", controller_parameters_.attitude_gain_.x(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/attitude_gain/y", controller_parameters_.attitude_gain_.y(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/attitude_gain/z", controller_parameters_.attitude_gain_.z(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/angular_rate_gain/x", controller_parameters_.angular_rate_gain_.x(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/angular_rate_gain/y", controller_parameters_.angular_rate_gain_.y(), pnh)) {
      return false;
    }
    if (!quadrotor_common::getParam(path_rel_to_node + "/angular_rate_gain/z", controller_parameters_.angular_rate_gain_.z(), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam("mass", vehicle_parameters_.mass, pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam("inertia/xx", inertia_(0, 0), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam("inertia/xy", inertia_(0, 1), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam("inertia/xz", inertia_(0, 2), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam("inertia/yy", inertia_(1, 1), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam("inertia/yz", inertia_(1, 2), pnh)) {
      return false;
    }

    if (!quadrotor_common::getParam("inertia/zz", inertia_(2, 2), pnh)) {
      return false;
    }

    return GetRotorConfiguration(pnh, &vehicle_parameters_.rotor_configuration_);
    return true;
  }

  inline bool GetRotorConfiguration(const ros::NodeHandle& nh,
                                  RotorConfiguration* rotor_configuration) {
    std::map<std::string, double> single_rotor;
    std::string rotor_configuration_string = "rotor_configuration/";
    unsigned int i = 0;
    while (nh.getParam(rotor_configuration_string + std::to_string(i), single_rotor)) {
      if (i == 0) {
        rotor_configuration->rotors.clear();
      }
      Rotor rotor;
      if (!quadrotor_common::getParam(rotor_configuration_string + std::to_string(i) + "/angle", rotor.angle, pnh)) {
        return false;
      }
      if (!quadrotor_common::getParam(rotor_configuration_string + std::to_string(i) + "/arm_length", rotor.arm_length, pnh)) {
        return false;
      }
      if (!quadrotor_common::getParam(rotor_configuration_string + std::to_string(i) + "/rotor_force_constant", rotor.rotor_force_constant, pnh)) {
        return false;
      }
      if (!quadrotor_common::getParam(rotor_configuration_string + std::to_string(i) + "/rotor_moment_constant", rotor.rotor_moment_constant, pnh)) {
        return false;
      }
      if (!quadrotor_common::getParam(rotor_configuration_string + std::to_string(i) + "/direction", rotor.direction, pnh)) {
        return false;
      }
      rotor_configuration->rotors.push_back(rotor);
      ++i;
    }
    return true;
  }

  // Send bodyrate commands if true, attitude commands otherwise
  bool use_att_mode;

  GeometricPositionControllerParameters controller_parameters_;
  VehicleParameters vehicle_parameters_;


  double pxy_error_max;  // [m]
  double vxy_error_max;  // [m/s]
  double pz_error_max;   // [m]
  double vz_error_max;   // [m/s]
  double yaw_error_max;  // [rad]

  // Whether or not to compensate for aerodynamic effects
  bool perform_aerodynamics_compensation;
  double k_drag_x;  // x-direction rotor drag coefficient
  double k_drag_y;  // y-direction rotor drag coefficient
  double k_drag_z;  // z-direction rotor drag coefficient
  // thrust correction coefficient due to body horizontal velocity
  double k_thrust_horz;
};

}  // namespace position_controller
