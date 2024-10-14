cmake_minimum_required(VERSION 2.8.3)
project(bspline_opt)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  plan_env
  active_perception
  cv_bridge

)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(NLopt REQUIRED)
set(NLopt_INCLUDE_DIRS ${NLOPT_INCLUDE_DIR})

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES bspline_opt
 CATKIN_DEPENDS plan_env active_perception
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${NLOPT_INCLUDE_DIR}
)

add_library( bspline_opt 
    src/bspline_optimizer.cpp 
    )
target_link_libraries( bspline_opt
    ${catkin_LIBRARIES} 
    ${NLOPT_LIBRARIES}
    # /usr/local/lib/libnlopt.so
    )  
    

install(DIRECTORY script DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})    
install(DIRECTORY include/bspline_opt
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/..
  FILES_MATCHING PATTERN "*.h"
)
  
install(TARGETS bspline_opt
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
 
