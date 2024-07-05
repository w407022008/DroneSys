# DroneSys
__DroneSys__ is a [ROS 1](http://wiki.ros.org/Documentation) based system for MAV navigation, sensing and control, compatible with the [PX4 Autopilot](https://docs.px4.io/main/en/) system via [MavLink](https://mavlink.io/en/) protocol. It consists of several modules: __Driver__, __Control__, __Planning__, __Perception__, __SLAM__ and __Mavros interface__. 

**Video Links:** 
 - More demonstrations in the gazebo physical simulation environment: [2dLidar](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/Gazebo_2dLidar.mp4), [3dLidar](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/Gazebo_3dLidar.mp4), [Camera](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/Gazebo_Camera.mp4)
 - More comparisons: [comparison_1](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/comparison1.mp4), [comparison_2](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/comparison2.mp4), [comparison_3](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/comparison3.mp4)

Please kindly give us a star :star:, if you find this work useful or interesting. We take great efforts to develope and maintain it, thanks!:grinning:

## Table of Contents

* [Setup and Config](#3-Setup-and-Config)
* [Run Simulations](#4-Run-Simulations)

## 1. Setup and Config
### Prerequisites

1. Our software is developed and tested in Ubuntu 16.04(ROS Kinetic), 18.04(ROS Melodic) and 20.04(ROS Noetic). Follow the documents to install [Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu), [Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) or [Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) according to your Ubuntu version. 
```
	ros-${ROS_VERSION_NAME}-ros-base
	ros-${ROS_VERSION_NAME}-gazebo-ros
	ros-${ROS_VERSION_NAME}-gazebo-plugins
	ros-${ROS_VERSION_NAME}-camera-info-manager
	ros-${ROS_VERSION_NAME}-velodyne-gazebo-plugins
	ros-${ROS_VERSION_NAME}-xacro
	ros-${ROS_VERSION_NAME}-tf2-ros
	ros-${ROS_VERSION_NAME}-tf2-eigen
	ros-${ROS_VERSION_NAME}-mavros-msgs
	ros-${ROS_VERSION_NAME}-mavros
	ros-${ROS_VERSION_NAME}-mavros-extras
	ros-${ROS_VERSION_NAME}-tf
	ros-${ROS_VERSION_NAME}-cv-bridge
	libarmadillo-dev
	ros-${ROS_VERSION_NAME}-rviz
	ros-${ROS_VERSION_NAME}-octomap-rviz-plugins
	ros-${ROS_VERSION_NAME}-pcl-ros
	ros-${ROS_VERSION_NAME}-hector-trajectory-server
	ros-${ROS_VERSION_NAME}-octomap
	ros-${ROS_VERSION_NAME}-octomap-msgs
	ros-${ROS_VERSION_NAME}-image-transport
	ros-${ROS_VERSION_NAME}-image-transport-plugins
	ros-${ROS_VERSION_NAME}-ddynamic-reconfigure
	ros-${ROS_VERSION_NAME}-vrpn-client-ros
	ros-${ROS_VERSION_NAME}-velodyne-gazebo-plugins
	ros-${ROS_VERSION_NAME}-roslint
	libdw-dev
```

2. We use [**NLopt**](https://nlopt.readthedocs.io/en/latest/NLopt_Installation) as the optimization solver in Histo-planner to solve the non-linear optimization problem, which can be installed by the following commands.
```
  git clone https://github.com/stevengj/nlopt.git
  cd nlopt && mkdir build && cd build  
  cmake ..  
  make  
  sudo make install 
```

1. The DroneSys has been tested to communicate with PX4 via the mavlink protocol. In order to test the complete DroneSys, we need to install the PX4 system. 
```
    git clone https://github.com/SyRoCo-ISIR/Firmware_PX4_v1.14.2.git --depth 1 PX4_v1.14.2 --recursive
    cd PX4_v1.14.2
    git tag v1.14.2
    sudo bash Tools/setup/ubuntu.sh
    # Restart the computer on completion. And then
    cd PX4_v1.14.2 && sudo make px4_sitl_default gazebo
```

1. Installing RealSense SDK:
```
  sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at
  ## LibRealSense v2.50.0 for example, which matchs with realsense-ros v2.3.2:
  wget https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.50.0.zip
  unzip v2.50.0.zip && rm v2.50.0.zip
  cd librealsense-2.50.0
  ./scripts/setup_udev_rules.sh
  mkdir build && cd build
  
  sudo apt install xorg-dev libglu1-mesa-dev ## if missing RandR
  sudo apt install libusb-1.0-0-dev ## if missing config.h

  cmake ../ -DFORCE_RSUSB_BACKEND=true -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true
  sudo make uninstall && make clean && make && sudo make install
  # test it, if you want
  realsense_viwer
```

1. Installing [ORB-SLAM3](Modules/slam/orb_slam3/README.md):

2. Installing Ceres for VINS:
```
  sudo apt-get install libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev 
  wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0.zip
  unzip ceres-solver-2.1.0.zip && rm ceres-solver-2.1.0.zip
  cd ceres-solver-2.1.0
  mkdir build && cd build
  cmake ..
  make
  sudo make install
```


### Build the DroneSys on ROS

After the prerequisites are satisfied, you can clone this repository to any expected path. 

```
  git clone https://github.com/SyRoCo-ISIR/DroneSys.git
  cd DroneSys
  wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
  sudo bash install_geographiclib_datasets.sh
  ./compile_all.sh # compile all packages to the current path
```
Add the following into ```~/.bashrc```:
```
export PATH="/usr/lib/ccache:$PATH"
source ~/src/DroneSys/devel/setup.bash
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/src/DroneSys/devel/lib
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/src/DroneSys/Simulator/gazebo_simulator/models
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/src/DroneSys/Simulator/gazebo_simulator/drone_models
source ~/src/PX4_v1.14.2/Tools/setup_gazebo.bash ~/src/PX4_v1.14.2 ~/src/PX4_v1.14.2/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4_v1.14.2
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4_v1.14.2/Tools/sitl_gazebo
```

If you encounter problems in this step, please first refer to existing __issues__, __pull requests__ and __Google__ before raising a new issue.



### Known compilation issue
**Joystick** is needed. You may encounter the following problems when you compile the Driver/joystick_drivers. Please make sure the following packages are installed.
```
  libusb-dev # if missing libusb
  libspnav-dev # if missing spnav.h
  libbluetooth-dev # if missing bluetooth.h
  
  # if missing cwiid.h:
  git clone https://github.com/abstrakraft/cwiid.git
  cd cwiid
  aclocal
  autoconf
  ./configure
  ## if missing pakcage gtk2.0:
    sudo apt install libgtk2.0-dev
  ## if missing flex:
    sudo apt install flex
  ## if missing bison:
    sudo apt install bison byacc
  ## if missing python.h:
    sudo apt install python2.7-dev
  make
  ## if there is an error about wmdemo
    # ignore it and go ahead
  sudo make install
  
  # if missing libdw.h
  sudo apt install libdw-dev
```
## 2. Run Simulations

Run our planner and the Gazebo simulation platform with ```roslaunch```:

```
  source ~/.bashrc
  roslaunch simulation_gazebo sitl_histo_planner.launch
```


Normally, you will find the randomly generated map and the drone model in ```Rviz```. At this time, you can trigger the planner using the ```3D Nav Goal``` tool. When a point is clicked in ```Rviz```, a new trajectory will be generated immediately and executed by the drone. 


## Acknowledgements
  - The [DroneSys](https://github.com/SyRoCo-ISIR/DroneSys) is built with reference to [Prometheus](https://github.com/amov-lab/Prometheus).

## Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.


## Disclaimer
This is research code, it is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of merchantability or fitness for a particular purpose.
