## basic
bash Tools/compile_base.sh
catkin_make --source Modules/common/rqt_quad_gui --build build/common_util/rqt_quad_gui

## controller
bash Tools/control/compile_autopilot.sh
bash Tools/control/compile_inner_loop_controller.sh
bash Tools/control/compile_dfbc.sh
bash Tools/control/compile_rpg_mpc.sh
bash Tools/control/compile_geo_control.sh
## simulator
bash Tools/simulator/compile_rotors.sh
catkin_make --source Simulator/rpg_simulator --build build/Simulator/rpg_simulator
