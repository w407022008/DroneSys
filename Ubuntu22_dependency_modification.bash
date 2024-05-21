#!/bin/bash

## Driver
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Driver/realsense-ros-2.3.2/realsense2_camera/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./Driver/joystick_drivers/joy/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./Driver/joystick_drivers/spacenav_node/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./Driver/joystick_drivers/wiimote/CMakeLists.txt

## Tools
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/tools/imu_cali/CMakeLists.txt

## mavros_interface
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/mavros_interface/CMakeLists.txt

## control
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/control/CMakeLists.txt

## perception
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/perception/joy_remote/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/perception/OptiTrack/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/perception/sensors/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/perception/points_filter/CMakeLists.txt

## Slam
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/loop_fusion/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/global_fusion/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/slam/orb_slam3_ros_wrapper/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/vins_estimator/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/msckf_vio/CMakeLists.txt

## planning
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/planning_simulator/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/globle_planner/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/local_planner/CMakeLists.txt
sed -i '16,19d' ./Modules/planning/local_planner/src/geo_guide_apf.cpp
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/FastPlanner/CMakeLists.txt
sed -i '65i add_compile_options(-std=c++17)' ./Modules/planning/FastPlanner/ThirdParty/sdf_tools/CMakeLists.txt
sed -i -e s/"-std=c++0x"/"-std=c++17"/g ./Modules/planning/FastPlanner/ThirdParty/sdf_tools/CMakeLists.txt
sed -i '5i #include <stdlib.h>' ./Modules/planning/FastPlanner/ThirdParty/arc_utilities/include/arc_utilities/first_order_deformation.h

sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/ego-planner/CMakeLists.txt
sed -i -e s/"mat_jerk(i, j)"/"mat_jerk((int)i, (int)j)"/g ./Modules/planning/ego-planner/include/polynomial_traj.h
sed -i '11i find_package(cv_bridge REQUIRED)'  ./Modules/planning/ego-planner/CMakeLists.txt
sed -i '11i find_package(visualization_msgs REQUIRED)'  ./Modules/planning/ego-planner/CMakeLists.txt
sed -i '11i find_package(drone_msgs REQUIRED)'  ./Modules/planning/ego-planner/CMakeLists.txt
sed -i '11i find_package(message_filters REQUIRED)'  ./Modules/planning/ego-planner/CMakeLists.txt
sed -i '11i find_package(roscpp REQUIRED)'  ./Modules/planning/ego-planner/CMakeLists.txt

sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(cv_bridge REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(image_transport REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(tf REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(mavros_msgs REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(drone_msgs REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(visualization_msgs REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(nav_msgs REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(pcl_conversions REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(pcl_ros REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(rospy REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
sed -i '10i find_package(roscpp REQUIRED)'  ./Modules/planning/histo-planner/CMakeLists.txt
