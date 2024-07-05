#!/bin/bash

## Driver
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Driver/realsense-ros-2.3.2/realsense2_camera/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./Driver/joystick_drivers/joy/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./Driver/joystick_drivers/spacenav_node/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./Driver/joystick_drivers/wiimote/CMakeLists.txt

## Experiment
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Experiment/utils/joy_remote/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Experiment/sensors/CMakeLists.txt

## Tools
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/tools/imu_cali/CMakeLists.txt

## common
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/common/mavros_interface/CMakeLists.txt
#sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./Modules/common/mavros/libmavconn/CMakeLists.txt
sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./Modules/common/mavros/mavros/CMakeLists.txt
sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./Modules/common/mavros/mavros_extras/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/common/quadrotor_common/CMakeLists.txt

## control
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/control/control/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/control/geometric_control/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/control/autopilot/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/control/inner_loop_controller/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/control/differential_flatness_base_controller/CMakeLists.txt
sed -i '30,35d' ./Modules/control/rpg_mpc/CMakeLists.txt
sed -i '30i add_compile_options(-std=c++11)' ./Modules/control/rpg_mpc/CMakeLists.txt

## perception
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/perception/OptiTrack/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/perception/points_filter/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/perception/state_predictor/CMakeLists.txt

## Slam
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/loop_fusion/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/global_fusion/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/slam/orb_slam3/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/vins_estimator/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/msckf_vio/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 14"/"CMAKE_CXX_STANDARD 17"/g ./Modules/slam/open_vins/ov_core/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 14"/"CMAKE_CXX_STANDARD 17"/g ./Modules/slam/open_vins/ov_init/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 14"/"CMAKE_CXX_STANDARD 17"/g ./Modules/slam/open_vins/ov_eval/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 14"/"CMAKE_CXX_STANDARD 17"/g ./Modules/slam/open_vins/ov_msckf/CMakeLists.txt

## planning
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/globle_planner/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/local_planner/CMakeLists.txt
sed -i '16,19d' ./Modules/planning/local_planner/src/geo_guide_apf.cpp
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/FastPlanner/CMakeLists.txt
sed -i '65i add_compile_options(-std=c++17)' ./Modules/planning/FastPlanner/ThirdParty/sdf_tools/CMakeLists.txt
sed -i -e s/"-std=c++0x"/"-std=c++17"/g ./Modules/planning/FastPlanner/ThirdParty/sdf_tools/CMakeLists.txt
sed -i '5i #include <stdlib.h>' ./Modules/planning/FastPlanner/ThirdParty/arc_utilities/include/arc_utilities/first_order_deformation.h

sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/ego-planner/CMakeLists.txt
sed -i -e s/"mat_jerk(i, j)"/"mat_jerk((int)i, (int)j)"/g ./Modules/planning/ego-planner/include/polynomial_traj.h

sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/histo-planner/CMakeLists.txt

sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/planning/polynomial_planning/polynomial_trajectories/CMakeLists.txt

## Simulator
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Simulator/planning_simulator/CMakeLists.txt
