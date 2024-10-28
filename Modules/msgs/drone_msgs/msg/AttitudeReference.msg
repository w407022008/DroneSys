std_msgs/Header header

## Setpoint Attitude + T
float32[3] thrust_sp                   ## Single Rotor Thrust setpoint
float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint
float32[3] desired_attitude            ## [rad] Eurler angle setpoint
geometry_msgs/Quaternion desired_att_q ## quat setpoint
geometry_msgs/Vector3 body_rate  ## [rad/s]