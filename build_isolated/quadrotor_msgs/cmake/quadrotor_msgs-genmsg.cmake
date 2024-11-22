# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "quadrotor_msgs: 6 messages, 0 services")

set(MSG_I_FLAGS "-Iquadrotor_msgs:/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Inav_msgs:/opt/ros/noetic/share/nav_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(quadrotor_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg" NAME_WE)
add_custom_target(_quadrotor_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "quadrotor_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg" "std_msgs/Header:geometry_msgs/Point:geometry_msgs/Twist:quadrotor_msgs/LowLevelFeedback:geometry_msgs/Vector3:geometry_msgs/Pose:geometry_msgs/TwistWithCovariance:quadrotor_msgs/TrajectoryPoint:geometry_msgs/PoseWithCovariance:nav_msgs/Odometry:geometry_msgs/Quaternion"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg" NAME_WE)
add_custom_target(_quadrotor_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "quadrotor_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg" "std_msgs/Header:geometry_msgs/Quaternion:geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg" NAME_WE)
add_custom_target(_quadrotor_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "quadrotor_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg" NAME_WE)
add_custom_target(_quadrotor_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "quadrotor_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg" "std_msgs/Header:geometry_msgs/Point:geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg" NAME_WE)
add_custom_target(_quadrotor_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "quadrotor_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg" "std_msgs/Header:geometry_msgs/Point:geometry_msgs/Twist:geometry_msgs/Vector3:geometry_msgs/Pose:quadrotor_msgs/TrajectoryPoint:geometry_msgs/Quaternion"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg" NAME_WE)
add_custom_target(_quadrotor_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "quadrotor_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg" "geometry_msgs/Point:geometry_msgs/Twist:geometry_msgs/Vector3:geometry_msgs/Pose:geometry_msgs/Quaternion"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TwistWithCovariance.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseWithCovariance.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/Odometry.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_cpp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_cpp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_cpp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_cpp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_cpp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
)

### Generating Services

### Generating Module File
_generate_module_cpp(quadrotor_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(quadrotor_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(quadrotor_msgs_generate_messages quadrotor_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_cpp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_cpp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_cpp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_cpp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_cpp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_cpp _quadrotor_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quadrotor_msgs_gencpp)
add_dependencies(quadrotor_msgs_gencpp quadrotor_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quadrotor_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TwistWithCovariance.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseWithCovariance.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/Odometry.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_eus(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_eus(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_eus(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_eus(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_eus(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
)

### Generating Services

### Generating Module File
_generate_module_eus(quadrotor_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(quadrotor_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(quadrotor_msgs_generate_messages quadrotor_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_eus _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_eus _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_eus _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_eus _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_eus _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_eus _quadrotor_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quadrotor_msgs_geneus)
add_dependencies(quadrotor_msgs_geneus quadrotor_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quadrotor_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TwistWithCovariance.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseWithCovariance.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/Odometry.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_lisp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_lisp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_lisp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_lisp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_lisp(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
)

### Generating Services

### Generating Module File
_generate_module_lisp(quadrotor_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(quadrotor_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(quadrotor_msgs_generate_messages quadrotor_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_lisp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_lisp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_lisp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_lisp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_lisp _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_lisp _quadrotor_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quadrotor_msgs_genlisp)
add_dependencies(quadrotor_msgs_genlisp quadrotor_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quadrotor_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TwistWithCovariance.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseWithCovariance.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/Odometry.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_nodejs(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_nodejs(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_nodejs(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_nodejs(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_nodejs(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
)

### Generating Services

### Generating Module File
_generate_module_nodejs(quadrotor_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(quadrotor_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(quadrotor_msgs_generate_messages quadrotor_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_nodejs _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_nodejs _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_nodejs _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_nodejs _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_nodejs _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_nodejs _quadrotor_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quadrotor_msgs_gennodejs)
add_dependencies(quadrotor_msgs_gennodejs quadrotor_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quadrotor_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TwistWithCovariance.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/PoseWithCovariance.msg;/opt/ros/noetic/share/nav_msgs/cmake/../msg/Odometry.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_py(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_py(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_py(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_py(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
)
_generate_msg_py(quadrotor_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
)

### Generating Services

### Generating Module File
_generate_module_py(quadrotor_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(quadrotor_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(quadrotor_msgs_generate_messages quadrotor_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/AutopilotFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_py _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_py _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/LowLevelFeedback.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_py _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/PositionCommand.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_py _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/Trajectory.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_py _quadrotor_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/quadrotor_msgs/msg/TrajectoryPoint.msg" NAME_WE)
add_dependencies(quadrotor_msgs_generate_messages_py _quadrotor_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quadrotor_msgs_genpy)
add_dependencies(quadrotor_msgs_genpy quadrotor_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quadrotor_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quadrotor_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(quadrotor_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET nav_msgs_generate_messages_cpp)
  add_dependencies(quadrotor_msgs_generate_messages_cpp nav_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(quadrotor_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quadrotor_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(quadrotor_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET nav_msgs_generate_messages_eus)
  add_dependencies(quadrotor_msgs_generate_messages_eus nav_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(quadrotor_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quadrotor_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(quadrotor_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET nav_msgs_generate_messages_lisp)
  add_dependencies(quadrotor_msgs_generate_messages_lisp nav_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(quadrotor_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quadrotor_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(quadrotor_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET nav_msgs_generate_messages_nodejs)
  add_dependencies(quadrotor_msgs_generate_messages_nodejs nav_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(quadrotor_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quadrotor_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(quadrotor_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET nav_msgs_generate_messages_py)
  add_dependencies(quadrotor_msgs_generate_messages_py nav_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(quadrotor_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
