cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++14)
project(semi_global_matching)

find_package(catkin REQUIRED
    message_generation
    message_runtime
    roscpp
    rospy
    geometry_msgs
    sensor_msgs
    std_msgs
    cv_bridge
    tf
)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

catkin_package()

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)



add_executable(${PROJECT_NAME}_node
   src/sgm_node.cpp
   src/SemiGlobalMatching.cpp
   src/sgm_util.cpp
)
target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
)


