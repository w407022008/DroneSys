
bash Tools/compile_base.sh

## Driver
catkin_make --source Driver/realsense-ros-2.3.2 --build build/Driver/realsense-ros-2.3.2
#catkin_make --source Driver/realsense-ros-2.3.2 --build build/Driver/realsense-ros -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release

## Experiment PX4
catkin_make --source Experiment/drone_experiment --build build/Experiment/drone_experiment
catkin_make --source Experiment/utils/joy_remote --build build/Experiment/utils/joy_remote
## Experiment rpg
#catkin_make --source Experiment/bridges/sbus_bridge --build build/Experiment/bridges/sbus_bridge
#catkin_make --source Experiment/utils/manual_flight_assistant --build build/Experiment/utils/manual_flight_assistant
#catkin_make --source Experiment/utils/vbat_thrust_calibration --build build/Experiment/utils/vbat_thrust_calibration
## Experiment Sensor
#catkin_make --source Experiment/companion_computer_io --build build/Experiment/companion_computer_io
catkin_make --source Experiment/sensors --build build/Experiment/sensors

## Controller
bash Tools/control/compile_control.sh
bash Tools/control/compile_autopilot.sh
bash Tools/control/compile_inner_loop_controller.sh
bash Tools/control/compile_dfbc.sh
bash Tools/control/compile_rpg_mpc.sh
bash Tools/control/compile_geo_control.sh
#bash Tools/control/compile_mpc_rw.sh

## IO & Perception
catkin_make --source Modules/perception/points_filter --build build/perception/points_filter
catkin_make --source Modules/perception/OptiTrack --build build/perception/OptiTrack
catkin_make --source Modules/perception/semi_global_matching --build build/perception/semi_global_matching
catkin_make --source Modules/perception/elas_stereo_matching --build build/perception/elas_stereo_matching
#catkin_make --source Modules/perception/state_predictor --build build/perception/state_predictor

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

## Simulator
catkin_make --source Simulator/planning_simulator --build build/Simulator/planning_simulator
