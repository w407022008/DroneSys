catkin_make --source Modules/utils/catkin_simple --build build/utils/catkin_simple
source devel/setup.sh
catkin_make --source Modules/utils/eigen_catkin --build build/utils/eigen_catkin

catkin_make --source Modules/msgs --build build/msgs
catkin_make --source Modules/common --build build/common_util
