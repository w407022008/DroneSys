## Driver
catkin_make --source Driver --build build/Driver

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
catkin_make --source Simulator --build build/simulator -j
#catkin_make --source Simulator/gazebo_simulator --build build/simulator/gazebo_simulator
#catkin_make --source Simulator/rotors_simulator --build build/simulator/rotors_simulator
#catkin_make --source Simulator/rpg_quadrotor_control --build build/simulator/rpg_quadrotor_control

## Tools
catkin_make --source Modules/tools --build build/tools
