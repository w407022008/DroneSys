catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
source devel/setup.sh
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin

catkin_make --source Modules/utils/mavros --build build/utils/mavros
catkin_make --source Modules/msgs/ --build build/msgs/
catkin_make --source Modules/common/common --build build/common/common
catkin_make --source Modules/common/quadrotor_common --build build/common/quadrotor_common
catkin_make --source Modules/common/mavros_interface --build build/common/mavros_interface
catkin_make --source Modules/common/ground_station --build build/common/ground_station
#catkin_make --source Modules/common/rqt_quad_gui --build build/common/rqt_quad_gui
