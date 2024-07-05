bash Tools/compile_base.sh
catkin_make --source Modules/perception/state_predictor --build build/perception/state_predictor
catkin_make --source Modules/planning/polynomial_planning --build build/planning/polynomial_planning
catkin_make --source Modules/control/autopilot --build build/control/autopilot

