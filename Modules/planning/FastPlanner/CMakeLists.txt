cmake_minimum_required(VERSION 2.8.3)
project(fast_planner)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

add_subdirectory(${PROJECT_SOURCE_DIR}/ThirdParty/arc_utilities)
add_subdirectory(${PROJECT_SOURCE_DIR}/ThirdParty/sdf_tools)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(octomap REQUIRED)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	rospy
	std_msgs
	tf
	octomap_msgs
	geometry_msgs
	sdf_tools
	sensor_msgs
	visualization_msgs
)

# Generate messages in the 'msg' folder

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES plan_manage
 CATKIN_DEPENDS message_runtime pcl_conversions pcl_ros tf
#  DEPENDS system_lib
)

include_directories(${OCTOMAP_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/sdf_tools/include)
include_directories(${PROJECT_SOURCE_DIR}/ThirdParty/arc_utilities/include)

include_directories(
  include
  SYSTEM
  ${PROJECT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  /usr/local/include # add include path for nlopt
  ${PROJECT_SOURCE_DIR}/../../common/common/include/common
)

         
add_executable(dyn_planner_node
    src/bspline_optimizer.cpp 
    src/non_uniform_bspline.cpp
    src/kinodynamic_astar.cpp 
    src/astar.cpp
    src/sdf_map.cpp 
    src/edt_environment.cpp
    src/global_point_sdf.cpp
    src/planning_visualization.cpp 
    src/dyn_planner_node.cpp 
    src/planning_fsm.cpp
    src/dyn_planner_manager.cpp
    src/tools
    )
target_link_libraries(dyn_planner_node 
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OCTOMAP_LIBRARIES}
    /usr/local/lib/libnlopt.so
    )


add_executable(traj_server 
    src/non_uniform_bspline.cpp
    src/traj_server.cpp)
target_link_libraries(traj_server ${catkin_LIBRARIES})


add_executable(planning_tracking 
    src/planning_tracking.cpp
    )
target_link_libraries(planning_tracking 
    ${catkin_LIBRARIES}
    )


add_executable(read_pub_ot
    src/read_pub_ot.cpp)
target_link_libraries(read_pub_ot
    ${OCTOMAP_LIBRARIES}
    ${catkin_LIBRARIES}
    )


add_executable(test_planning_static 
    src/test_planning_static.cpp)
target_link_libraries(test_planning_static 
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    )

