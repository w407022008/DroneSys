#!/bin/sh
echo "build SVO: cpu_num="
read cpu_num
catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
catkin_make --source Modules/utils/ceres_catkin --build build/utils/ceres_catkin
catkin_make --source Modules/utils/cmake_external_project_catkin --build build/utils/cmake_external_project_catkin
catkin_make --source Modules/utils/dbow2_catkin --build build/utils/dbow2_catkin
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin
catkin_make --source Modules/utils/eigen_checks --build build/utils/eigen_checks
catkin_make --source Modules/utils/fast_neon --build build/utils/fast_neon
catkin_make --source Modules/utils/gflags_catkin --build build/utils/gflags_catkin
catkin_make --source Modules/utils/glog_catkin --build build/utils/glog_catkin
catkin_make --source Modules/utils/minkindr --build build/utils/minkindr
catkin_make --source Modules/utils/minkindr_ros --build build/utils/minkindr_ros
catkin_make --source Modules/utils/opengv --build build/utils/opengv

catkin_make --source Modules/slam/rpg_svo_pro_open --build build/slam/rpg_svo_pro_open -j$cpu_num
