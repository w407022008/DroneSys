catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
source devel/setup.bash
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin
catkin_make --source Modules/msgs/quadrotor_msgs --build build/msgs/quadrotor_msgs
catkin_make --source Modules/common/quadrotor_common --build build/common/quadrotor_common
catkin_make --source Modules/msgs/mav_comm --build build/msgs/mav_comm
catkin_make --source Modules/control/inner_loop_controller --build build/control/inner_loop_controller

