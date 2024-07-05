bash Tools/compile_base.sh
# careful! clean dynamic_reconfigure cfg cache or just temporarily remove those lines
catkin_make --source Modules/control/mav_control_rw --build build/control/mav_control_rw
