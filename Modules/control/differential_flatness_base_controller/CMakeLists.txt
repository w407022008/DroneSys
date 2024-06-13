cmake_minimum_required(VERSION 2.8.3)
project(differential_flatness_base_controller)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
add_compile_options(-O3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

if(catkin_LIBRARIES MATCHES "autopilot")

  message("Building with MPC Autopilot")
  cs_add_executable(autopilot_dfbc_instance
    src/autopilot_dfbc_instance.cpp
    src/position_controller.cpp
  )

endif(catkin_LIBRARIES MATCHES "autopilot")

cs_add_executable(dfbc_position_controller_node
  src/dfbc_position_controller_node.cpp
  src/position_controller.cpp
)

cs_install()
cs_export()
