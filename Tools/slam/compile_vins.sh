#!/bin/sh
echo "build VINS: cpu_num="
read cpu_num
catkin_make --source Modules/utils/camera_models --build build/utils/camera_models  -j$cpu_num
catkin_make --source Modules/slam/VINS-Fusion --build build/slam/VINS-Fusion -j$cpu_num
