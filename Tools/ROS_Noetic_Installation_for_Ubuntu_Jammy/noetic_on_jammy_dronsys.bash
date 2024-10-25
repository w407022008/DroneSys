#!/bin/bash

rm -r ./src

ROS_DISTRO=noetic

## 1\ Add repository
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

## 2\ install rosdep and dependency
sudo apt install python3-rosdep python3-rosinstall-generator python3-vcstools python3-vcstool build-essential libfltk1.3-dev

# if dpkg: error processing archive /var/cache/apt/archives/python3-rospkg-modules_1.5.0-1_all.deb (--unpack):
# sudo dpkg -i --force-overwrite /var/cache/apt/archives//var/cache/apt/archives/python3-rospkg-modules_1.5.0-1_all.deb
# sudo apt update

sudo rosdep init
rosdep update

sudo apt install libarmadillo-dev libdw-dev libusb-dev libspnav-dev libbluetooth-dev libgtk2.0-dev flex bison byacc python2.7-dev

## 3\ download ros-noetic packages

	# Option 1 - from github (ofcourse it has been cloned)
	git clone https://github.com/w407022008/Noetic_on_Ubuntu22.git noetic
	sudo -H apt install -y python3-numpy libboost-all-dev libopencv-dev python3-opencv libboost-date-time-dev libboost-filesystem-dev libboost-program-options-dev libboost-regex-dev libboost-thread-dev python3-pycryptodome python3-gnupg python3-rospkg sbcl libboost-dev libboost-thread1.74.0 libgtest-dev libeigen3-dev libgeographic-dev geographiclib-tools libboost-system-dev libconsole-bridge-dev libpoco-dev liblz4-dev liburdfdom-headers-dev liburdfdom-dev libtinyxml-dev libtinyxml2-dev graphviz python3-empy python3-paramiko liborocos-kdl-dev liborocos-kdl1.5 python3-pykdl cmake python3-mock python3-nose python3-catkin-pkg google-mock libpcl-dev libbz2-dev libgpgme-dev libboost-chrono-dev python3-defusedxml python3-coverage python3-lxml python3-future libapr1-dev libaprutil1-dev liblog4cxx-dev libogg-dev libtheora-dev libturbojpeg libturbojpeg0-dev python3-netifaces
	## make sure python as python3
	#sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100
	#sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150

	# Option 2 - from rosdep
	mkdir ~/noetic && cd noetic
	mkdir ./src
	rosinstall_generator ros_base tf2_ros tf2_eigen tf2_geometry_msgs tf_conversions random_numbers mavros_msgs mavros mavros_extras tf cv_bridge pcl_ros octomap octomap_msgs image_transport image_transport_plugins ddynamic_reconfigure vrpn_client_ros roslint --rosdistro noetic --deps --tar > noetic-dronesys.rosinstall
	vcs import --input noetic-dronesys.rosinstall ./src
	rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y
	# modification to adapt to jammy
	curl https://raw.githubusercontent.com/w407022008/Noetic_on_Ubuntu22/master/Ubuntu22_dependency_modification.bash | bash
	
## 4\ build & install noetic packages
./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release 
echo "source ~/noetic/install_isolated/setup.bash" >> ~/.bashrc
source ~/.bashrc

## 5\ Debug
# if error occurred: temporarily skip mavlink & vrpn
mkdir tmp
mv src/mavlink tmp/mavlink
mv src/mavros tmp/mavros
mv src/vrpn tmp/vrpn
mv src/vrpn_client_ros tmp/vrpn_client_ros

./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3

# install mavlink (modified mavros is included in DroneSys)
mv tmp/mavlink src/mavlink
#mv tmp/mavros src/mavros
./src/catkin/bin/catkin_make_isolated --source src/mavlink --install -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
#./src/catkin/bin/catkin_make_isolated --source src/mavros --install -DCMAKE_BUILD_TYPE=Release ## use custom build

## install vrpn
mv tmp/vrpn src/vrpn
mv tmp/vrpn_client_ros src/vrpn_client_ros
./src/catkin/bin/catkin_make_isolated --source src/vrpn --install -DCMAKE_BUILD_TYPE=Release
./src/catkin/bin/catkin_make_isolated --source src/vrpn_client_ros --install -DCMAKE_BUILD_TYPE=Release

## 6\ Option
sudo mkdir /opt/ros
sudo cp -r /install_isolated /opt/ros/noetic

rm -r tmp