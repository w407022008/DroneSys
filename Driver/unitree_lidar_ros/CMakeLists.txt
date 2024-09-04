cmake_minimum_required(VERSION 2.8.3)
project(unitree_lidar_ros)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -llz4")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -DNDEBUG -llz4") 

find_package(PCL REQUIRED QUIET)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  rosparam
  pcl_conversions
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
  roslib
)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL

  CATKIN_DEPENDS
  rosparam
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  message_generation
  visualization_msgs
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
)

link_directories(
  lib
  ${PCL_LIBRARY_DIRS}
  ./lib/${CMAKE_SYSTEM_PROCESSOR}
)

set(EXTRA_LIBS 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  libunitree_lidar_sdk.a
)

add_executable(unitree_lidar_ros_node src/unitree_lidar_ros_node.cpp)
target_link_libraries(unitree_lidar_ros_node ${EXTRA_LIBS})


# Install nodelet library
install(TARGETS ${PROJECT_NAME}_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

# Install header files
install(DIRECTORY include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    )

# Install launch files
install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    )

# Install rviz files
install(DIRECTORY rviz/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
    )

# Install xml files
#install(FILES nodelet_plugins.xml
#    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#    )

