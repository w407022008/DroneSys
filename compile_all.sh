catkin_make --source Driver --build build/Driver
#catkin_make --source Driver/realsense-ros-2.3.2 --build build/Driver/realsense-ros -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
catkin_make --source Modules/msgs --build build/msgs
catkin_make --source Modules/common --build build/common_util
catkin_make --source Modules/tools --build build/tools
catkin_make --source Experiment --build build/drone_experiment
catkin_make --source Modules/control --build build/control
catkin_make --source Modules/perception --build build/perception
catkin_make --source Modules/slam --build build/slam
catkin_make --source Modules/planning --build build/planning -j10
catkin_make --source Simulator --build build/simulator
