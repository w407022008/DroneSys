catkin_make --source Driver/realsense-ros-2.3.2 --build build/Driver/realsense-ros-2.3.2
#catkin_make --source Driver/realsense-ros-2.3.2 --build build/Driver/realsense-ros -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
catkin_make --source Modules/msgs --build build/msgs
catkin_make --source Modules/common --build build/common_util
catkin_make --source Experiment --build build/drone_experiment
catkin_make --source Modules/control --build build/control
catkin_make --source Modules/perception --build build/perception
catkin_make --source Modules/tools/camera_models --build build/tools/camera_models -j3
catkin_make --source Modules/slam/VINS-Fusion --build build/slam/VINS-Fusion -j3
catkin_make --source Modules/slam/msckf_vio/ --build build/slam/msckf_vio --cmake-args -DCMAKE_BUILD_TYPE=Release
#catkin_make --source Modules/tools/fast --build build/tools/fast
#catkin_make --source Modules/slam/msckf_mono --build build/slam/msckf_mono
catkin_make --source Modules/slam/orb_slam3_ros_wrapper --build build/slam/orb_slam3_ros_wrapper -j2
#catkin_make --source Modules/planning/FastPlanner --build build/planning/FastPlanner -j2
catkin_make --source Modules/planning/ego-planner --build build/planning/ego-planner -j2
catkin_make --source Modules/planning/histo-planner --build build/planning/histo-planner -j2
catkin_make --source Modules/planning/planning_simulator --build build/planning/planning_simulator -j2
