catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin
catkin_make --source Modules/msgs/mav_comm --build build/msgs/mav_comm
# careful! clean dynamic_reconfigure cfg cache or just temporarily remove those lines
catkin_make --source Modules/control/mav_control_rw --build build/control/mav_control_rw
