cmake_minimum_required(VERSION 2.8.3)
project(joy_remote)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
		roscpp
		rospy
		drone_msgs
		tf
	)

catkin_package(
       LIBRARIES Joy_Remote
	)


include_directories(
       SYSTEM
       include
       ${catkin_INCLUDE_DIRS}
       ${Eigen3_INCLUDE_DIRS}
	   ../../common/include/common
     )



add_executable(joy_remote_main
	src/joy_remote.cpp 
	)

target_link_libraries(joy_remote_main
       ${catkin_LIBRARIES}
	)
