cmake_minimum_required(VERSION 2.8.3)
project(points_worker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
		roscpp
		rospy
		message_generation
		pcl_ros
		pcl_conversions
		tf
		cv_bridge
		message_filters
	)

catkin_package(
	INCLUDE_DIRS include
	LIBRARIES points_worker
	)

find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(PCL REQUIRED COMPONENT common io )

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


add_executable(points_octomap_node 
    src/points2octomap.cpp
    )
target_link_libraries(points_octomap_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OCTOMAP_LIBRARIES} 
)

add_executable(points_saver_node 
	src/points_saver.cpp
	)
target_link_libraries(points_saver_node
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
	)

add_executable(points_publisher_node 
	src/points_puber.cpp
	)
	target_link_libraries(points_publisher_node
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
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
 install(TARGETS points_filter_main
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## Mark cpp header files for installation
 install(DIRECTORY include/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
 )

## Mark other files for installation (e.g. launch and bag files, etc.)
 install(DIRECTORY launch/
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
 )

