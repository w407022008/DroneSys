sudo apt-get install libsuitesparse-dev
#!/bin/sh
echo "build msckf_vio: cpu_num="
read cpu_num
catkin_make --source Modules/slam/msckf_vio/ --build build/slam/msckf_vio  -j$cpu_num --cmake-args -DCMAKE_BUILD_TYPE=Release
