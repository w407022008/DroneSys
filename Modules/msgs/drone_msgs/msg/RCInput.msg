## Radio Control Input
std_msgs/Header header

# Data Source
uint8 DISABLE = 0
uint8 MAVROS_MANUAL_CONTROL = 1
uint8 DRIVER_JOYSTICK = 2

float32 rc_x             # stick position in x direction -1..1
                         # in general corresponds to forward/back motion or pitch of vehicle,
                         # in general a positive value means forward or negative pitch and
                         # a negative value means backward or positive pitch
float32 rc_y             # stick position in y direction -1..1
                         # in general corresponds to right/left motion or roll of vehicle,
                         # in general a positive value means right or positive roll and
                         # a negative value means left or negative roll
float32 rc_z             # throttle stick position 0..1
                         # in general corresponds to up/down motion or thrust of vehicle,
                         # in general the value corresponds to the demanded throttle by the user,
                         # if the input is used for setting the setpoint of a vertical position
                         # controller any value > 0.5 means up and any value < 0.5 means down
float32 rc_r             # yaw stick/twist position, -1..1
                         # in general corresponds to the righthand rotation around the vertical
                         # (downwards) axis of the vehicle
uint32 buttons           # Binary

uint8 goal_enable        # push down(1):enable; release(0):disable

int32 data_source # determin the data source
