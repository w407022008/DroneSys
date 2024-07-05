catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
source devel/setup.sh
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin

catkin_make --source Modules/common/mavros --build build/common_util/mavros
catkin_make --source Modules/msgs/drone_msgs --build build/msgs/drone_msgs
catkin_make --source Modules/msgs/mav_comm --build build/msgs/mav_comm
catkin_make --source Modules/msgs/quadrotor_msgs --build build/msgs/quadrotor_msgs
catkin_make --source Modules/common/common --build build/common_util/common
catkin_make --source Modules/common/ground_station --build build/common_util/ground_station
catkin_make --source Modules/common/mavros_interface --build build/common_util/mavros_interface
catkin_make --source Modules/common/quadrotor_common --build build/common_util/quadrotor_common
#catkin_make --source Modules/common/rqt_quad_gui --build build/common_util/rqt_quad_gui
