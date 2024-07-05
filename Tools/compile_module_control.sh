## Mavlink control
bash Tools/control/compile_control.sh

## Autopilot
bash Tools/control/compile_control_common.sh
catkin_make --source Modules/perception/state_predictor --build build/perception/state_predictor
catkin_make --source Modules/planning/polynomial_planning --build build/planning/polynomial_planning
catkin_make --source Modules/control/autopilot --build build/control/autopilot
catkin_make --source Modules/msgs/mav_comm --build build/msgs/mav_comm
# inner_loop controller
catkin_make --source Modules/control/inner_loop_controller --build build/control/inner_loop_controller
# dfbc controller
catkin_make --source Modules/control/differential_flatness_base_controller --build build/control/differential_flatness_base_controller
# mpc controller
catkin_make --source Modules/control/rpg_mpc --build build/control/rpg_mpc

## Geometric controller
catkin_make --source Modules/control/geometric_control --build build/control/geometric_contol

## Others
#bash Tools/control/compile_mpc_rw.sh
