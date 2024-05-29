catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin
catkin_make --source Modules/msgs/quadrotor_msgs --build build/msgs/quadrotor_msgs
catkin_make --source Modules/common/quadrotor_common --build build/common/quadrotor_common
catkin_make --source Modules/perception/state_predictor --build build/perception/state_predictor
catkin_make --source Modules/planning/polynomial_planning --build build/planning/polynomial_planning
catkin_make --source Modules/control/differential_flatness_base_controller --build build/control/differential_flatness_base_controller

