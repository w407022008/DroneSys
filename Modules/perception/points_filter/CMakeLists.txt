cmake_minimum_required(VERSION 2.8.3)
project(points_filter)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
		roscpp
		rospy
		message_generation
		pcl_ros
		pcl_conversions
		tf
		cv_bridge
	)

catkin_package(
       INCLUDE_DIRS include
       LIBRARIES Points_Filter
	)

# SET(USER_KEY, "Hello World")
MESSAGE( STATUS "this lcal trajectory planner path:  ******** = ${PROJECT_SOURCE_DIR}.")


include_directories(
       SYSTEM
       include
       ${PROJECT_SOURCE_DIR}/include
       ${catkin_INCLUDE_DIRS}
       ${Eigen3_INCLUDE_DIRS}
       ${PCL_INCLUDE_DIRS}
     )
link_directories(${PCL_LIBRARY_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -Wall")


add_executable(points_filter_main
	src/points_filter.cpp 
	)

target_link_libraries(points_filter_main
       ${catkin_LIBRARIES}
	   ${PCL_LIBRARIES}
	)
