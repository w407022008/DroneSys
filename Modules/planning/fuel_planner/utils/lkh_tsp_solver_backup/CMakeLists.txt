cmake_minimum_required(VERSION 2.8.3)
project(lkh_tsp_solver)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES lkh_tsp_solver
 CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
)

file( GLOB SRCS src/*.c )

# message("${SRCS}")

add_library( lkh_tsp_solver 
  ${SRCS}
  src/lkh_interface.cpp
)
target_link_libraries(lkh_tsp_solver -lm)

add_executable( lkh_tsp 
  ${SRCS}
)
target_link_libraries(lkh_tsp ${catkin_LIBRARIES} -lm)


install(DIRECTORY resource DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
  
install(TARGETS lkh_tsp_solver
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
 
