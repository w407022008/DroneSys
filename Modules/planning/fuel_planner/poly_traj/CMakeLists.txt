cmake_minimum_required(VERSION 2.8.3)
project(poly_traj)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
roscpp
std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES poly_traj
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
)

add_library( poly_traj 
    src/polynomial_traj.cpp 
    )
target_link_libraries( poly_traj
    ${catkin_LIBRARIES} 
    )  

install(DIRECTORY include/poly_traj
DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/..
FILES_MATCHING PATTERN "*.h"
)
    
install(TARGETS poly_traj
ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
    
