std_msgs/Header header

## Setpoint Attitude + T
float32[3] throttle_sp                 ## [0-1] Throttle setpoint
float32 desired_thrust               ## [0~1] Axis Body_Z Thrust septoint
float32[3] desired_attitude            ## [rad] Eurler angle setpoint
geometry_msgs/Quaternion desired_att_q ## quat setpoint
