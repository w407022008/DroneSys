#!/bin/sh
echo "build planning: cpu_num="
read cpu_num
bash Tools/compile_base.sh
catkin_make --source Modules/planning --build build/planning -j$cpu_num
