catkin_make --source Modules/utils/fast --build build/utils/fast
#!/bin/sh
echo "build msckf_mono: cpu_num="
read cpu_num
catkin_make --source Modules/slam/msckf_mono --build build/slam/msckf_mono -j$cpu_num
