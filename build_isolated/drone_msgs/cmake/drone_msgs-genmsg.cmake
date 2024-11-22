# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "drone_msgs: 10 messages, 0 services")

set(MSG_I_FLAGS "-Idrone_msgs:/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Imavros_msgs:/home/sique/src/DroneSys_sim/install_isolated/share/mavros_msgs/cmake/../msg;-Igeographic_msgs:/opt/ros/noetic/share/geographic_msgs/cmake/../msg;-Iuuid_msgs:/opt/ros/noetic/share/uuid_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(drone_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg" "geometry_msgs/Quaternion:geometry_msgs/Vector3:std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg" "geometry_msgs/Point"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg" "std_msgs/Header:drone_msgs/PositionReference:drone_msgs/AttitudeReference:drone_msgs/Bspline:geometry_msgs/Vector3:geometry_msgs/Quaternion:geometry_msgs/Point"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg" "geometry_msgs/Point:drone_msgs/Bspline:std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg" "geometry_msgs/Quaternion:geometry_msgs/Vector3:std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg" "geometry_msgs/Quaternion:std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg" "geometry_msgs/Quaternion:mavros_msgs/ActuatorControl:std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg" NAME_WE)
add_custom_target(_drone_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "drone_msgs" "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg" "std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/sique/src/DroneSys_sim/install_isolated/share/mavros_msgs/cmake/../msg/ActuatorControl.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)
_generate_msg_cpp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
)

### Generating Services

### Generating Module File
_generate_module_cpp(drone_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(drone_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(drone_msgs_generate_messages drone_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_cpp _drone_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(drone_msgs_gencpp)
add_dependencies(drone_msgs_gencpp drone_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS drone_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/sique/src/DroneSys_sim/install_isolated/share/mavros_msgs/cmake/../msg/ActuatorControl.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)
_generate_msg_eus(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
)

### Generating Services

### Generating Module File
_generate_module_eus(drone_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(drone_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(drone_msgs_generate_messages drone_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_eus _drone_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(drone_msgs_geneus)
add_dependencies(drone_msgs_geneus drone_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS drone_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/sique/src/DroneSys_sim/install_isolated/share/mavros_msgs/cmake/../msg/ActuatorControl.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)
_generate_msg_lisp(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
)

### Generating Services

### Generating Module File
_generate_module_lisp(drone_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(drone_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(drone_msgs_generate_messages drone_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_lisp _drone_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(drone_msgs_genlisp)
add_dependencies(drone_msgs_genlisp drone_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS drone_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/sique/src/DroneSys_sim/install_isolated/share/mavros_msgs/cmake/../msg/ActuatorControl.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)
_generate_msg_nodejs(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
)

### Generating Services

### Generating Module File
_generate_module_nodejs(drone_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(drone_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(drone_msgs_generate_messages drone_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_nodejs _drone_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(drone_msgs_gennodejs)
add_dependencies(drone_msgs_gennodejs drone_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS drone_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/sique/src/DroneSys_sim/install_isolated/share/mavros_msgs/cmake/../msg/ActuatorControl.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)
_generate_msg_py(drone_msgs
  "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
)

### Generating Services

### Generating Module File
_generate_module_py(drone_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(drone_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(drone_msgs_generate_messages drone_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Arduino.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Bspline.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlCommand.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/PositionReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/AttitudeReference.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneState.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/DroneTarget.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/ControlOutput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/Message.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/sique/src/DroneSys_sim/Modules/msgs/drone_msgs/msg/RCInput.msg" NAME_WE)
add_dependencies(drone_msgs_generate_messages_py _drone_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(drone_msgs_genpy)
add_dependencies(drone_msgs_genpy drone_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS drone_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/drone_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_cpp)
  add_dependencies(drone_msgs_generate_messages_cpp actionlib_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(drone_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(drone_msgs_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(drone_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET mavros_msgs_generate_messages_cpp)
  add_dependencies(drone_msgs_generate_messages_cpp mavros_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/drone_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_eus)
  add_dependencies(drone_msgs_generate_messages_eus actionlib_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(drone_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(drone_msgs_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(drone_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET mavros_msgs_generate_messages_eus)
  add_dependencies(drone_msgs_generate_messages_eus mavros_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/drone_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_lisp)
  add_dependencies(drone_msgs_generate_messages_lisp actionlib_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(drone_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(drone_msgs_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(drone_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET mavros_msgs_generate_messages_lisp)
  add_dependencies(drone_msgs_generate_messages_lisp mavros_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/drone_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_nodejs)
  add_dependencies(drone_msgs_generate_messages_nodejs actionlib_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(drone_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(drone_msgs_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(drone_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET mavros_msgs_generate_messages_nodejs)
  add_dependencies(drone_msgs_generate_messages_nodejs mavros_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/drone_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET actionlib_msgs_generate_messages_py)
  add_dependencies(drone_msgs_generate_messages_py actionlib_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(drone_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(drone_msgs_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(drone_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET mavros_msgs_generate_messages_py)
  add_dependencies(drone_msgs_generate_messages_py mavros_msgs_generate_messages_py)
endif()
