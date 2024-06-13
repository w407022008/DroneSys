cmake_minimum_required(VERSION 2.8.3)
add_compile_options(-std=c++14)
project(stereo_matching)

find_package(PCL 1.7 REQUIRED)
find_package(octomap REQUIRED)
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
    ${OCTOMAP_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

include_directories("/usr/include/eigen3/")


find_package( PCL REQUIRED COMPONENT common io )
add_definitions( ${PCL_DEFINITIONS} )
link_directories(${PCL_LIBRARY_DIRS})

add_executable(${PROJECT_NAME}_node 
   src/elas_stereo_node.cpp
   src/lib/descriptor.cpp
   src/lib/elas.cpp
   src/lib/filter.cpp
   src/lib/matrix.cpp
   src/lib/stereomatch.cpp
   src/lib/pointcloud.cpp
   src/lib/triangle.cpp
)
target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${PCL_LIBRARIES}
   ${OCTOMAP_LIBRARIES} 
)


