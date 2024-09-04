## Driver
catkin_make --source Driver --build build/Driver

## Basic common msgs
bash Tools/compile_base.sh

## Experiment 
catkin_make --source Experiment --build build/Experiment

## Controller
bash Tools/compile_module_control.sh

## Perception
catkin_make --source Modules/perception --build build/perception

## SLAM
bash Tools/compile_module_slam.sh

## Planning
catkin_make --source Modules/planning --build build/planning

## Simulator
bash Tools/compile_module_simulator.sh

## Tools
catkin_make --source Modules/tools --build build/tools
