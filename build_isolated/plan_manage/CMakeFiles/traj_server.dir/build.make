# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sique/src/DroneSys_sim/Modules/planning/fuel_planner/plan_manage

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sique/src/DroneSys_sim/build_isolated/plan_manage

# Include any dependencies generated for this target.
include CMakeFiles/traj_server.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/traj_server.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/traj_server.dir/flags.make

CMakeFiles/traj_server.dir/src/traj_server.cpp.o: CMakeFiles/traj_server.dir/flags.make
CMakeFiles/traj_server.dir/src/traj_server.cpp.o: /home/sique/src/DroneSys_sim/Modules/planning/fuel_planner/plan_manage/src/traj_server.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sique/src/DroneSys_sim/build_isolated/plan_manage/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/traj_server.dir/src/traj_server.cpp.o"
	/usr/lib/ccache/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/traj_server.dir/src/traj_server.cpp.o -c /home/sique/src/DroneSys_sim/Modules/planning/fuel_planner/plan_manage/src/traj_server.cpp

CMakeFiles/traj_server.dir/src/traj_server.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/traj_server.dir/src/traj_server.cpp.i"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sique/src/DroneSys_sim/Modules/planning/fuel_planner/plan_manage/src/traj_server.cpp > CMakeFiles/traj_server.dir/src/traj_server.cpp.i

CMakeFiles/traj_server.dir/src/traj_server.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/traj_server.dir/src/traj_server.cpp.s"
	/usr/lib/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sique/src/DroneSys_sim/Modules/planning/fuel_planner/plan_manage/src/traj_server.cpp -o CMakeFiles/traj_server.dir/src/traj_server.cpp.s

# Object files for target traj_server
traj_server_OBJECTS = \
"CMakeFiles/traj_server.dir/src/traj_server.cpp.o"

# External object files for target traj_server
traj_server_EXTERNAL_OBJECTS =

/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: CMakeFiles/traj_server.dir/src/traj_server.cpp.o
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: CMakeFiles/traj_server.dir/build.make
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /home/sique/src/DroneSys_sim/install_isolated/lib/libtraj_utils.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /home/sique/src/DroneSys_sim/install_isolated/lib/libbspline_opt.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /home/sique/src/DroneSys_sim/install_isolated/lib/libpoly_traj.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /home/sique/src/DroneSys_sim/install_isolated/lib/libactive_perception.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /home/sique/src/DroneSys_sim/install_isolated/lib/libbspline.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /home/sique/src/DroneSys_sim/install_isolated/lib/libpath_searching.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /home/sique/src/DroneSys_sim/install_isolated/lib/libplan_env.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/libroscpp.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/libxmlrpcpp.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/libcv_bridge.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/librosconsole.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/librosconsole_log4cxx.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/librosconsole_backend_interface.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/libroscpp_serialization.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/librostime.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /opt/ros/noetic/lib/libcpp_common.so
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server: CMakeFiles/traj_server.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sique/src/DroneSys_sim/build_isolated/plan_manage/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/traj_server.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/traj_server.dir/build: /home/sique/src/DroneSys_sim/devel_isolated/plan_manage/lib/plan_manage/traj_server

.PHONY : CMakeFiles/traj_server.dir/build

CMakeFiles/traj_server.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/traj_server.dir/cmake_clean.cmake
.PHONY : CMakeFiles/traj_server.dir/clean

CMakeFiles/traj_server.dir/depend:
	cd /home/sique/src/DroneSys_sim/build_isolated/plan_manage && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sique/src/DroneSys_sim/Modules/planning/fuel_planner/plan_manage /home/sique/src/DroneSys_sim/Modules/planning/fuel_planner/plan_manage /home/sique/src/DroneSys_sim/build_isolated/plan_manage /home/sique/src/DroneSys_sim/build_isolated/plan_manage /home/sique/src/DroneSys_sim/build_isolated/plan_manage/CMakeFiles/traj_server.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/traj_server.dir/depend

