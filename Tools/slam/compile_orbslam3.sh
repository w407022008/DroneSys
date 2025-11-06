#!/bin/sh
cd Modules/slam/orb_slam3/library
bash build.sh
cd ../../../..
echo "build ORBSlam wrapper: cpu_num="
read cpu_num
catkin_make --source Modules/slam/orb_slam3 --build build/slam/orb_slam3 -j$cpu_num
