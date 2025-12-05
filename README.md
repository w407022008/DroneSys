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

1. Our software is developed and tested in Ubuntu 16.04(ROS Kinetic), 18.04(ROS Melodic) and 20.04(ROS Noetic). Follow the documents to install [Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu), [Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) or [Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) according to your Ubuntu version. (ATTENTION! If you want also to install Noetic onto Ubuntu 22.04(Jammy), you could also refer to [Jammy-Noetic](https://github.com/w407022008/Noetic_on_Ubuntu22/blob/master/noetic_on_jammy_dronsys.bash).)
```
  sudo apt install ros-noetic-ros-base ros-noetic-gazebo-ros ros-noetic-mavlink geographiclib-tools libgeographic-dev libgeographic19 ros-noetic-geographic-msgs ros-noetic-eigen-conversions ros-noetic-gazebo-plugins ros-noetic-camera-info-manager ros-noetic-velodyne-gazebo-plugins ros-noetic-octomap-rviz-plugins ros-noetic-xacro ros-noetic-tf2-ros ros-noetic-tf2-eigen ros-noetic-tf ros-noetic-cv-bridge libarmadillo-dev ros-noetic-rviz ros-noetic-pcl-ros ros-noetic-hector-trajectory-server ros-noetic-octomap ros-noetic-octomap-msgs ros-noetic-image-transport ros-noetic-image-transport-plugins ros-noetic-ddynamic-reconfigure ros-noetic-vrpn-client-ros ros-noetic-roslint cmake libdw-dev libusb-dev libspnav-dev libbluetooth-dev libgtk2.0-dev flex bison byacc python2.7-dev python ros-noetic-random-numbers ros-noetic-tf-conversions libsuitesparse-dev ros-noetic-rviz-imu-plugin libv4l-dev v4l-utils
```

```
	sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
	sudo rosdep init
	rosdep update
```

```
	echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
	source ~/.bashrc
```

2. We use [**NLopt**](https://nlopt.readthedocs.io/en/latest/NLopt_Installation) as the optimization solver in Histo-planner to solve the non-linear optimization problem, which can be installed by the following commands.
```
  git clone https://github.com/stevengj/nlopt.git
  cd nlopt && mkdir build && cd build  
  cmake ..  
  make & sudo make install 
```

3. [**Ceres**](http://ceres-solver.org/) is depended by VINS to solve the non-linear optimization problem, which can be installed by the following commands.
```
  sudo apt-get install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev 
	wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0.zip
	unzip 2.1.0.zip && rm 2.1.0.zip
	cd ceres-solver-2.1.0
	mkdir build && cd build
	cmake ..
	make
	sudo make install
```

4. The DroneSys has been tested to communicate with [PX4](https://github.com/PX4) via the mavlink protocol. In order to test the complete DroneSys, we need to install the PX4 system. 
```
    git clone https://github.com/w407022008/Firmware_PX4_v1.15.2.git --depth 1 PX4_v1.15.2 --recursive
    cd PX4_v1.15.2
    git tag v1.15.2
    sudo bash Tools/setup/ubuntu.sh
    # Restart the computer on completion. And then
    cd PX4_v1.15.2 && sudo make px4_sitl_default gazebo
```
Any error, please refer to [PX4 official installation](https://docs.px4.io/v1.14/en/dev_setup/dev_env_linux_ubuntu.html#ros-gazebo-classic).

5. Installing RealSense SDK:
Here we offer two modified SDK which allocates images with or without emitters for depth and visual localization, and fixes the default transfer rate of 30hz instead of 15hz when using a USB 2.0 interface. i.e. [librealsense_v2.50.0](https://github.com/w407022008/librealsense-2.50.0), [librealsense_v2.55.1](https://github.com/w407022008/librealsense-2.50.0)
```
  sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at
  ## A modified LibRealSense v2.50.0 for example, which matchs with realsense-ros v2.3.2:
  git clone https://github.com/w407022008/librealsense-2.50.0.git librealsense-2.50.0
  
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

6. Installing [ORB-SLAM3](Modules/slam/orb_slam3/README.md) and its dependencies: ORB-SLAM3 without Pangolin is already supported in DroneSys, so the installation is not needed.


### Build the DroneSys on ROS

After the prerequisites are satisfied (including: ROS, NLOpt, Ceres, PX4, RealSense SDK), you can now clone this repository to any expected path. 

```
  git clone https://github.com/w407022008/DroneSys.git
  curl https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh | sudo bash
```
Build it!
(ATTENTION: If you install Noetic onto Ubuntu 22.04, run firstly ```bash Ubuntu22_dependency_modification.bash```.)
```
  ./compile_try.sh # compile all packages to the current path
```
Add the following into ```~/.bashrc```:
```
export PATH="/usr/lib/ccache:$PATH"
source ~/src/DroneSys/devel/setup.bash
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/src/DroneSys/devel/lib
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/src/DroneSys/Simulator/gazebo_simulator/models
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/src/DroneSys/Simulator/gazebo_simulator/drone_models
source ~/src/PX4_v1.15.2/Tools/simulation/gazebo-classic/setup_gazebo.bash ~/src/PX4_v1.15.2 ~/src/PX4_v1.15.2/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4_v1.15.2
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4_v1.15.2/Tools/simulation/gazebo-classic/sitl_gazebo-classic
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
