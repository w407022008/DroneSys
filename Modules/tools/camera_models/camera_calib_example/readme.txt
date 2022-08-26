# help for checking input parameters.
rosrun camera_models Calibrations --help

# example pinhole model.
rosrun camera_models Calibrations -w 12 -h 8 -s 80 -i DATA/left --camera-model pinhole

# example mei model.
rosrun camera_models Calibrations -w 12 -h 8 -s 80 -i DATA/left --camera-model mei

# save imgae
rosrun image_view image_saver image:=camera/infra1/image_rect_raw
