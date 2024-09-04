cmake_minimum_required(VERSION 2.8.3)
project(geometric_control)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mav_msgs
  nav_msgs
  quadrotor_msgs
  quadrotor_common
  roscpp
  sensor_msgs
  drone_msgs
  cmake_modules
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include ${Eigen3_INCLUDE_DIRS}
  LIBRARIES geometric_position_controller roll_pitch_yawrate_thrust_controller
  CATKIN_DEPENDS geometry_msgs mav_msgs nav_msgs roscpp sensor_msgs quadrotor_msgs quadrotor_common
  DEPENDS Eigen3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

add_library(geometric_position_controller
  src/geometric_position_controller.cpp
)
target_link_libraries(geometric_position_controller ${catkin_LIBRARIES})
add_dependencies(geometric_position_controller ${catkin_EXPORTED_TARGETS})

add_executable(geometric_position_controller_node src/geometric_position_controller_node.cpp)
add_dependencies(geometric_position_controller_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(geometric_position_controller_node
  geometric_position_controller ${catkin_LIBRARIES})

install(TARGETS geometric_position_controller
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(TARGETS geometric_position_controller_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
