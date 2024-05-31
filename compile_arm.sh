
bash Tools/compile_base.sh

## Driver
catkin_make --source Driver/realsense-ros-2.3.2 --build build/Driver/realsense-ros-2.3.2
#catkin_make --source Driver/realsense-ros-2.3.2 --build build/Driver/realsense-ros -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release

## Controller
bash Tools/compile_module_control.sh

## IO & Perception
catkin_make --source Modules/perception --build build/perception

## SLAM
bash Tools/slam/compile_openvins.sh
bash Tools/slam/compile_vins.sh
bash ./Tool/slam/compile_msckf_vio.sh
#bash ./Tool/compile_svo_pro.sh

## Planning
#catkin_make --source Modules/planning/FastPlanner --build build/planning/FastPlanner -j2
catkin_make --source Modules/planning/ego-planner --build build/planning/ego-planner -j2
catkin_make --source Modules/planning/histo-planner --build build/planning/histo-planner -j2
catkin_make --source Modules/planning/polynomial_planning --build build/planning/polynomial_planning
catkin_make --source Modules/planning/planning_simulator --build build/planning/planning_simulator -j2
