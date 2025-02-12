#!/bin/bash

## Driver
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Driver/realsense-ros-2.3.2/realsense2_camera/CMakeLists.txt

## utils
#sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./Modules/utils/mavros/libmavconn/CMakeLists.txt
sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./Modules/utils/mavros/mavros/CMakeLists.txt
sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./Modules/utils/mavros/mavros_extras/CMakeLists.txt

## common
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/common/mavros_interface/CMakeLists.txt
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/common/quadrotor_common/CMakeLists.txt

## control
sed -i -e s/"-std=c++11"/"-std=c++17"/g ./Modules/control/control/CMakeLists.txt

## perception
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/perception/points_worker/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/perception/opencv_sgbm/CMakeLists.txt

## Slam
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/loop_fusion/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/global_fusion/CMakeLists.txt
sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/slam/VINS-Fusion/vins_estimator/CMakeLists.txt

## planning
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/plan_env/CMakeLists.txt
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/bspline/CMakeLists.txt
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/path_searching/CMakeLists.txt
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/active_perception/CMakeLists.txt
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/bspline_opt/CMakeLists.txt
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/traj_utils/CMakeLists.txt
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/plan_manage/CMakeLists.txt
sed -i -e s/"set(CMAKE_CXX_STANDARD 14)"/"set(CMAKE_CXX_STANDARD 17)"/g ./Modules/planning/fuel_planner/exploration_manager/CMakeLists.txt

sed -i -e s/"-std=c++14"/"-std=c++17"/g ./Modules/planning/ego-planner/CMakeLists.txt
#sed -i -e s/"mat_jerk(i, j)"/"mat_jerk((int)i, (int)j)"/g ./Modules/planning/ego-planner/include/polynomial_traj.h

