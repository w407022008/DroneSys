std_msgs/Header header

## Setpoint position reference for PX4 Control

## Setpoint Mode
uint8 Move_mode

uint8 XYZ_POS      = 0  ##0b00
uint8 XY_POS_Z_VEL = 1  ##0b01
uint8 XY_VEL_Z_POS = 2  ##0b10
uint8 XYZ_VEL = 3       ##0b11
uint8 XYZ_ACC = 4
uint8 XYZ_POS_VEL   = 5  
uint8 TRAJECTORY   = 6

## Reference Frame
uint8 Move_frame

uint8 ENU_FRAME  = 0
uint8 BODY_FRAME = 1



## Tracking life
float32 time_from_start          ## [s]

float32[3] position_ref          ## [m]
float32[3] velocity_ref          ## [m/s]
float32[3] acceleration_ref      ## [m/s^2]

bool Yaw_Rate_Mode                      ## True 代表控制偏航角速率
float32 yaw_ref                  ## [rad]
float32 yaw_rate_ref             ## [rad/s] 

Bspline bspline