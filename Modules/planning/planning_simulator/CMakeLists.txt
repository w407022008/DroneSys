cmake_minimum_required(VERSION 2.8.3)
project(planning_simulator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  pcl_ros
  pcl_conversions
  sensor_msgs
  nav_msgs
  visualization_msgs
  tf
  drone_msgs
)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(map_generation src/map_generation.cpp)
target_link_libraries(map_generation
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES})
  
add_executable (random_forest src/random_forest_sensing.cpp )
target_link_libraries(random_forest
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES})

add_executable(uav_sim src/uav_sim.cpp)
target_link_libraries(uav_sim 
   ${catkin_LIBRARIES}
)
