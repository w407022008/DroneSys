catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
source devel/setup.bash
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin
catkin_make --source Modules/msgs/quadrotor_msgs --build build/msgs/quadrotor_msgs
catkin_make --source Modules/common/quadrotor_common --build build/common_util/quadrotor_common
catkin_make --source Modules/common/rqt_quad_gui --build build/common_util/rqt_quad_gui

## controller
bash Tools/control/compile_inner_loop_controller.sh
bash Tools/control/compile_dfbc.sh
bash Tools/control/compile_rpg_mpc.sh
## simulator
bash Tools/simulator/compile_rotors.sh
catkin_make --source Simulator/rpg_simulator --build build/Simulator/rpg_simulator
