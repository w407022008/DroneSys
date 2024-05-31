#!/bin/sh
sudo apt-get install libdw-dev
pip install numpy==1.21
catkin_make --source Modules/tools/imu_cali --build build/tools/imu_cali
