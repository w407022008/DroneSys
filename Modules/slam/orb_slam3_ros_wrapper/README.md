**Ongoing development.**

# ROS wrapper for ORB-SLAM3

A ROS wrapper for [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (currently in [V1.0, December 22th, 2021](https://github.com/w407022008/ORB_SLAM3)). The main idea is to use the ORB-SLAM3 as a standalone library and interface with it instead of putting everything in one package.

## 1. ORB-SLAM3 (original or other variants)

### 1.1  Prerequisites - Pangolin-0.6
We have tested the library in **Ubuntu 16.04** and **18.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.
- Install Dependencies: 
```
  libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev
	libc++-dev libglew-dev libeigen3-dev cmake
	libjpeg-dev libpng-dev
	libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev
	libdc1394-22-dev libraw1394-dev libopenni-dev python3.7-dev python3-distutils
  ```
- Build Sources
```
  wget https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.6.zip
	mkdir build; cd build
	cmake ..
	cmake --build .
  ```

### 1.2  Prerequisites - OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 3.0. Tested with OpenCV 3.2.0**.
Check your opencv version:  
```
/usr/bin/python3 -c "import cv2;print(cv2.__version__)"
```

### 1.3  Prerequisites - Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

### 1.4  Prerequisites - DBoW2, g2o and Sophus (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations, [Sophus](https://github.com/strasdat/Sophus) library to perform Lie groups commonly used for 2d and 3d geometric problems. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

### 1.5  Prerequisites - Python
Required to calculate the alignment of the trajectory with the ground truth. **Required Numpy module**.

* (win) http://www.python.org/downloads/windows
* (deb) `sudo apt install libpython2.7-dev`
* (mac) preinstalled with osx

### 1.6  Install ROS

We provide the ros wrapper to process input of a monocular, monocular-inertial, stereo, stereo-inertial or RGB-D camera using ROS. Building these examples is optional. These have been tested with [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) under Ubuntu 18.04.

### 1.7 Install ORB_SLAM3

- Build and install ORB-SLAM3. Any location is fine (default directory that I use later on is the home folder `~`):
```
cd ~
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

- Make sure that **`libORB_SLAM3.so`** is created in the *ORB_SLAM3/lib* folder. If not, check the issue list from the [original repo](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues) and rebuild the package.

## 2. orb_slam3_ros_wrapper (this package)

- Clone the package. Note that it should be a `catkin build` or `catkin_make` workspace.
```
cd ~/catkin_ws/src/ # Or the name of your workspace
git clone https://github.com/w407022008/orb_slam3_ros_wrapper.git
```

- Open `CMakeLists.txt` and change the directory that leads to ORB-SLAM3 library at the beginning of the file (default is home folder)
```
cd ~/catkin_ws/src/orb_slam3_ros_wrapper/
nano CMakeLists.txt

# Change this to your installation of ORB-SLAM3. Default is ~/
set(ORB_SLAM3_DIR
   $ENV{HOME}/src/ORB_SLAM3
)
```

- Build the package normally.
```
cd ~/catkin_ws/
catkin build
```

- Unzip the `ORBvoc.txt` file in the `config` folder in this package. Alternatively, you can change the `voc_file` param in the launch file to point to the right folder.
```
cd ~/catkin_ws/src/orb_slam3_ros_wrapper/config
tar -xf ORBvoc.txt.tar.gz
```
We also provide `ORBvoc.bin` files for improving the computational efficiency, which has become optional in the algorithm. You just need to make sure that PATH_TO_VOCABULARY points to this file at runtime.

- Install `hector-trajectory-server` to visualize the trajectory.
```
sudo apt install ros-[kinetic/melodic]-hector-trajectory-server
```

- If everything works fine, you can now try the different launch files in the `launch` folder. The EuRoC dataset is recommended for testing. 

- If you are using a realsense sensor, you can complete a calibration following [Calibration_Tutorial.pdf](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Calibration_Tutorial.pdf).

## 3. How to run

Example with EuRoC dataset:
```
# In one terminal
roslaunch orb_slam3_ros_wrapper orb_slam3_mono_inertial_euroc.launch
# In another terminal
rosbag play MH_01_easy.bag
```
Similarly for other sensor types.

# Topics
The following topics are published by each node:
- `/orb_slam3_ros/map_points` ([`PointCloud2`](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html)) containing all keypoints being tracked.
- `/orb_slam3_ros/camera` ([`PoseStamped`](http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/PoseStamped.html)): current pose of the camera in the world frame, as returned by ORB-SLAM3 with the world coordinate transformed to conform the ROS standard. To improve data stability, we provide the option of interpolation, which smoothes the raw data by different polynomials to obtain a more stable and high-speed output.
- `tf`: transformation from the camera fraame to the world frame.