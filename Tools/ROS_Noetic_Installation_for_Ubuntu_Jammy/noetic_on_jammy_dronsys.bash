#!/bin/bash

rm -r ./src

ROS_DISTRO=noetic

sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

sudo apt install python3-rosdep python3-rosinstall-generator python3-vcstools python3-vcstool build-essential

# if dpkg: error processing archive /var/cache/apt/archives/python3-rospkg-modules_1.5.0-1_all.deb (--unpack):
# sudo dpkg -i --force-overwrite /var/cache/apt/archives//var/cache/apt/archives/python3-rospkg-modules_1.5.0-1_all.deb
# sudo apt update

sudo rosdep init
rosdep update

mkdir ~/noetic && cd noetic
mkdir ./src
sudo apt install libfltk1.3-dev

## download packages
rosinstall_generator ros_base tf2_ros tf2_eigen tf_conversions random_numbers mavros_msgs mavros mavros_extras tf cv_bridge pcl_ros octomap octomap_msgs image_transport image_transport_plugins ddynamic_reconfigure vrpn_client_ros roslint --rosdistro noetic --deps --tar > noetic-dronesys.rosinstall

vcs import --input noetic-dronesys.rosinstall ./src

rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y

## modification to adapt to jammy
"bash ./Ubuntu22_dependency_modification.bash"

## install noetic
#sudo apt install cmake python3-empy libboost-all-dev libconsole-bridge-dev python3-future libtinyxml-dev libtinyxml2-dev libgtest-dev liblz4-dev

## temporarily skip mavlink & vrpn
mkdir tmp
mv src/mavlink tmp/mavlink
mv src/mavros tmp/mavros
mv src/vrpn tmp/vrpn
mv src/vrpn_client_ros tmp/vrpn_client_ros

./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3

## install mavlink
#sudo pip install future
mv tmp/mavlink src/mavlink
#mv tmp/mavros src/mavros
./src/catkin/bin/catkin_make_isolated --source src/diagnostics/diagnostic_updater --install -DCMAKE_BUILD_TYPE=Release
./src/catkin/bin/catkin_make_isolated --source src/unique_identifier/uuid_msgs --install -DCMAKE_BUILD_TYPE=Release
./src/catkin/bin/catkin_make_isolated --source src/geographic_info/geographic_msgs --install -DCMAKE_BUILD_TYPE=Release
./src/catkin/bin/catkin_make_isolated --source src/geometry/eigen_conversions --install -DCMAKE_BUILD_TYPE=Release
./src/catkin/bin/catkin_make_isolated --source src/urdf --install -DCMAKE_BUILD_TYPE=Release
./src/catkin/bin/catkin_make_isolated --source src/mavlink --install -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
#./src/catkin/bin/catkin_make_isolated --source src/mavros --install -DCMAKE_BUILD_TYPE=Release ## use custom build

## install vrpn
source ~/noetic/install_isolated/setup.bash
mv tmp/vrpn src/vrpn
mv tmp/vrpn_client_ros src/vrpn_client_ros
./src/catkin/bin/catkin_make_isolated --source src/vrpn --install -DCMAKE_BUILD_TYPE=Release
./src/catkin/bin/catkin_make_isolated --source src/vrpn_client_ros --install -DCMAKE_BUILD_TYPE=Release


sudo mkdir /opt/ros
sudo cp -r /install_isolated /opt/ros/noetic

rm -r tmp
