## Driver
catkin_make --source Driver --build build/Driver

## Basic common msgs
bash Tools/compile_base.sh

## Controller
bash Tools/compile_module_control.sh

## Perception
catkin_make --source Modules/perception --build build/perception

## SLAM
bash Tools/compile_module_slam.sh

## Planning
catkin_make --source Modules/planning --build build/planning -j

## Simulator
bash Tools/compile_module_simulator.sh
#bash Tools/simulator/compile_gazebo.sh
#bash Tools/simulator/compile_rotors.sh
#bash Tools/simulator/compile_rpg.sh
#catkin_make --source Simulator/planning_simulator --build build/Simulator/planning_simulator

## Tools
catkin_make --source Modules/tools --build build/tools
