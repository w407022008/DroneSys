cmake_minimum_required(VERSION 2.8.3)
project(opencv_sgbm)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
		roscpp
		std_msgs
		sensor_msgs
		cv_bridge
		image_transport
	)

find_package(OpenCV REQUIRED)

catkin_package(
	LIBRARIES opencv_sgbm
	)


include_directories(
       ${catkin_INCLUDE_DIRS}
       ${OpenCV_INCLUDE_DIRS}
     )

add_executable(opencv_sgbm
	src/opencv_sgbm.cpp 
	)

target_link_libraries(opencv_sgbm
       ${catkin_LIBRARIES}
       ${OpenCV_LIBRARIES}
	)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
 install(TARGETS opencv_sgbm
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )


## Mark other files for installation (e.g. launch and bag files, etc.)
 install(DIRECTORY launch/
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
 )

