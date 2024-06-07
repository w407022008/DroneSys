#!/bin/sh
echo "build ORBSlam: cpu_num="
read cpu_num
catkin_make --source Modules/slam/orb_slam3_ros_wrapper --build build/slam/orb_slam3_ros_wrapper -j$cpu_num
bash Tools/compile_base.sh
bash Tools/slam/compile_openvins.sh
bash Tools/slam/compile_msckf_vio.sh
bash Tools/slam/compile_vins.sh
bash Tools/slam/compile_svo_pro.sh
#bash Tools/slam/compile_msckf_mono.sh
