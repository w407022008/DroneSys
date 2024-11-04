#!/bin/sh
sudo apt-get install libeigen3-dev libboost-all-dev libceres-dev
echo "build OpenVINS: cpu_num="
read cpu_num
catkin_make --source Modules/slam/open_vins --build build/slam/open_vins -j$cpu_num -DDISABLE_MATPLOTLIB=OFF
