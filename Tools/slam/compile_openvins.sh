#!/bin/sh
sudo apt-get install libeigen3-dev libboost-all-dev libceres-dev
echo "build VINS: cpu_num="
read cpu_num
catkin_make --source Modules/slam/openvins --build build/slam/openvins -j$cpu_num -DDISABLE_MATPLOTLIB=OFF
