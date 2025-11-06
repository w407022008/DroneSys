#!/bin/sh
#bash Tools/slam/compile_orbslam3.sh
bash Tools/compile_base.sh
bash Tools/slam/compile_openvins.sh
bash Tools/slam/compile_msckf_vio.sh
bash Tools/slam/compile_vins.sh
#bash Tools/slam/compile_svo_pro.sh
#bash Tools/slam/compile_msckf_mono.sh
bash Tools/slam/compile_point_lio.sh
