source /opt/ros/noetic/setup.bash
source $(pwd)/install_isolated/setup.bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)/install_isolated/share/simulation_gazebo/drone_models
source $(pwd)/px4/Tools/simulation/gazebo-classic/setup_gazebo.bash $(pwd)/px4 $(pwd)/px4/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/px4
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/px4/Tools/simulation/gazebo-classic/sitl_gazebo-classic

