std_msgs/Header header

## Setpoint Attitude + T
float32[3] thrust_sp                   ## Single Rotor Thrust setpoint
float32 desired_throttle               ## [0~1] Axis Body_Z Throttle septoint
float32[3] desired_attitude            ## [rad] Eurler angle setpoint
geometry_msgs/Quaternion desired_att_q ## quat setpoint
