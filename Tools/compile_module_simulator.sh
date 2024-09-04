#!/bin/sh
bash Tools/simulator/compile_gazebo.sh
bash Tools/simulator/compile_rotors.sh
catkin_make --source Modules/common/rqt_quad_gui --build build/common_util/rqt_quad_gui
catkin_make --source Simulator/rpg_simulator --build build/Simulator/rpg_simulator
catkin_make --source Simulator/planning_simulator --build build/Simulator/planning_simulator
