cmake_minimum_required(VERSION 2.8.3)
project(histo_planner)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED COMPONENT common io)

find_package(catkin REQUIRED COMPONENTS
		roscpp
		rospy
		message_generation
#		std_msgs
#		sensor_msgs
#		geometry_msgs
		pcl_ros
		pcl_conversions
		nav_msgs
		visualization_msgs
		drone_msgs
		mavros_msgs
		tf
		image_transport
		cv_bridge
	)

## Generate messages in the 'msg' folder

## Generate added messages and services with any dependencies listed here

catkin_package(
       INCLUDE_DIRS include
       LIBRARIES Histo_Planning
       #  DEPENDS system_lib
	)

# SET(USER_KEY, "Hello World")
MESSAGE( STATUS "this lcal trajectory planner path:  ******** = ${PROJECT_SOURCE_DIR}.")
### FOR PROFILING (need to set after catkin_package)
add_compile_options(-pg)
set(catkin_LIBRARIES ${catkin_LIBRARIES} -pg)
###

include_directories(
       SYSTEM
       include
       ${PROJECT_SOURCE_DIR}/include
       ${catkin_INCLUDE_DIRS}
       ${Eigen3_INCLUDE_DIRS}
       ${PCL_INCLUDE_DIRS}
       ${PROJECT_SOURCE_DIR}/../../common/include/common
     )
link_directories(${PCL_LIBRARY_DIRS})

add_definitions( ${PCL_DEFINITIONS} )

add_executable(histo_planner_main 
		src/histo_planner_node.cpp
		src/histo_planning.cpp
		src/histogram.cpp
		src/bspline_optimizer.cpp 
		src/bspline.cpp
		src/planning_visualization.cpp 
	)

target_link_libraries(histo_planner_main
       ${catkin_LIBRARIES}
	   ${PCL_LIBRARIES}
    	/usr/local/lib/libnlopt.so
		dw
	)
