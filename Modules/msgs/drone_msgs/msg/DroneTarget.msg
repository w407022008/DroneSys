std_msgs/Header header

float32[3] position_target          ## [m]
float32[3] velocity_target          ## [m/s]
float32[3] acceleration_target      ## [m/s/s]
geometry_msgs/Quaternion q_target   ## quat
float32[3] euler_target             ## [rad]
float32[3] rate_target              ## [rad/s]
float32 thrust_target
mavros_msgs/ActuatorControl actuator_target
