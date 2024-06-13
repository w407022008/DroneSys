#!/bin/sh
bash Tools/simulator/compile_gazebo.sh
bash Tools/simulator/compile_rotors.sh
bash Tools/simulator/compile_rpg.sh
catkin_make --source Simulator/planning_simulator --build build/Simulator/planning_simulator
