# Histo-Planner

Histo-Planner : A Real-time Local Planner for MAVs Teleoperation based on Histogram of Obstacle Distribution

**Histo-Planner** is a histogram-based local planner without relying on the global 3D occupancy grid, which is designed to work on MAVs with limited computational power for tele-operation. It has a significantly lower total planning time compared to state-of-the-art methods ([Ego-planner](https://github.com/ZJU-FAST-Lab/ego-planner) and [Fast-Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner)). The map update time will remain around 0.3 ms.

<p align = "center">
<img src="https://github.com/w407022008/histo-planner/blob/main/documentation/figures/feature.jpg?raw=true" width = "968" height = "544" border="1" />
</p>

**Video Links:** 
 - The submission [video](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/video_audio.mp4) shows the algorithm framework.
 - More demonstrations in the gazebo physical simulation environment: [2dLidar](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/Gazebo_2dLidar.mp4), [3dLidar](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/Gazebo_3dLidar.mp4), [Camera](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/Gazebo_Camera.mp4)
 - More comparisons: [comparison_1](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/comparison1.mp4), [comparison_2](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/comparison2.mp4), [comparison_3](https://raw.githubusercontent.com/w407022008/histo-planner/main/documentation/videos/comparison3.mp4)

Please kindly give us a star :star:, if you find this work useful or interesting. We take great efforts to develope and maintain it, thanks!:grinning:

## Table of Contents

* [Quick Start](#1-Quick-Start)
* [Algorithms and Papers](#2-Algorithms-and-Papers)
* [Setup and Config](#3-Setup-and-Config)
* [Run Simulations](#4-Run-Simulations)



## 1. Quick Start

Compiling tests passed on ubuntu **18.04 and 20.04** with ros installed. Take Ubuntu 20.04 as an example. If you want only to install the Histo-planner, you can just execute the following commands one by one. If you want to try the complete [DroneSys](https://github.com/SyRoCo-ISIR/DroneSys.git), you may check the detailed [instruction](#3-Setup-and-Config) to setup it.
```
sudo apt-get install ros_${ROS_VERSION_NAME}_nlopt
git clone https://github.com/SyRoCo-ISIR/histo-planner
catkin_make
roslaunch histo_planner histo-planner.launch
```

## 2. Algorithms and Papers

The algorithm provides a robust and computationally efficient local obstacle avoidance trajectory planning for tele-operated MAVs.

The method is detailed in our paper below.
- **Histo-Planner : A Real-time Local Planner for MAVs Teleoperation based on Histogram of Obstacle Distribution**, Ze Wang, Zhenyu Gao, Jingang Qu, Pascal Morin, IEEE International Conference on Robotics and Automation (__ICRA__), 2023. (Under review)

The main programs in the algorithm are implemented in __histo_planner__ (To be Open):

- __histo_planning__: Planning manager that schedule and call the mapping and planning programs. The whole algorithm starts with receiving the message and ends with publishing the track reference control, and contains a total of four threads except for listening to the message: Histogram update, goal update for teleoperation, trajectory generation and safety monitoring, and reference trajectory tracking control. The last three are done in the program.
- __histogram__: Map service and guidance point generation. This program contains a separate thread to update the local spatial obstacle distribution and generate histograms in real time. According to the histogram, when called, the program provides local spatial gradient fields and obstacle avoidance optimal guidance point.
- __bspline__: A implementation of the B-spline-based trajectory representation. The cubic Hamiton curve is reparameterized as a B-spline.
- __bspline_optimizer__: The gradient-based trajectory optimization using B-spline trajectory.


Besides the folder __histo_planner__, a lightweight __planning_simulator__ is used for testing.


## 3. Setup and Config

### Prerequisites

1. Our software is developed and tested in Ubuntu 16.04(ROS Kinetic), 18.04(ROS Melodic) and 20.04(ROS Noetic). Follow the documents to install [Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu), [Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) or [Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) according to your Ubuntu version. 
Histo-planner does not rely on the following packages, but if you are willing to try the complete [DroneSys](https://github.com/SyRoCo-ISIR/DroneSys.git), please make sure the following packages are installed.
```
	ros-${ROS_VERSION_NAME}-ros-base
	ros-${ROS_VERSION_NAME}-gazebo-ros
	ros-${ROS_VERSION_NAME}-camera-info-manager
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
```

2. We use [**NLopt**](https://nlopt.readthedocs.io/en/latest/NLopt_Installation) as the optimization solver in Histo-planner to solve the non-linear optimization problem, which can be installed by the following commands.
```
  git clone https://github.com/stevengj/nlopt.git
  cd nlopt && mkdir build && cd build  
  cmake ..  
  make  
  sudo make install 
```

3. The DroneSys has been tested to communicate with PX4 via the mavlink protocol. In order to test the complete DroneSys, we need to install the PX4 system, but it is not necessary for the Histo-planner. The following installation is for the DroneSys only:
```
  git clone https://github.com/w407022008/Firmware_PX4_v1.12.3.git --depth 1 PX4_v1.12.3 --recursive
  cd PX4_v1.12.3
	git tag v1.12.3
	sudo bash Tools/setup/ubuntu.sh
  # Restart the computer on completion. And then
	cd PX4_v1.12.3 && sudo make px4_sitl_default
```

4. Installing RealSense SDK (DroneSys dependent but Histo-planner):
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

5. Installing [ORB-SLAM3](https://github.com/w407022008/ORB_SLAM3) (DroneSys dependent but Histo-planner):

6. Installing Ceres for VINS (DroneSys dependent but Histo-planner):
```
  sudo apt-get install libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev 
  git clone https://ceres-solver.googlesource.com/ceres-solver
  cd ceres-solver
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
source ~/src/PX4_v1.12.3/Tools/setup_gazebo.bash ~/src/PX4_v1.12.3 ~/src/PX4_v1.12.3/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4_v1.12.3
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4_v1.12.3/Tools/sitl_gazebo
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
```
## 4. Run Simulations

Run our planner and the Gazebo simulation platform with ```roslaunch```:

```
  source ~/.bashrc
  roslaunch simulation_gazebo sitl_histo_planner.launch
```


Normally, you will find the randomly generated map and the drone model in ```Rviz```. At this time, you can trigger the planner using the ```3D Nav Goal``` tool. When a point is clicked in ```Rviz```, a new trajectory will be generated immediately and executed by the drone. 


## Acknowledgements
  - We use **NLopt** for non-linear optimization.
  - The framework of this repository is based on [Fast-Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner).
  - The complete drone system is [DroneSys](https://github.com/SyRoCo-ISIR/DroneSys), which was built with reference to [Prometheus](https://github.com/amov-lab/Prometheus).

## Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.


## Disclaimer
This is research code, it is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of merchantability or fitness for a particular purpose.
