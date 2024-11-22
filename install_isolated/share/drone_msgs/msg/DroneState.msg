std_msgs/Header header

## is connected with PX4 flight controller
bool connected
bool armed
## landed / in-air
bool landed
## Control mode in PX4 
string mode

float32 time_from_start             ## [s]

## Kinmatic states
float32[3] position                 ## [m]
float32  rel_alt                    ## [m] only for outdoor
float32[3] velocity                 ## [m/s]
float32[3] attitude                 ## [rad]
geometry_msgs/Quaternion attitude_q ## quat
float32[3] attitude_rate            ## [rad/s]
