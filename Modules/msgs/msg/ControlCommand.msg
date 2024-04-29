std_msgs/Header header

## ID should increased self
uint32 Command_ID

string source

uint8 Mode
# enum
uint8 Idle=0
uint8 Takeoff=1
uint8 Hold=2
uint8 Land=3
uint8 Move=4
uint8 Disarm=5
uint8 User_Mode1=6
uint8 User_Mode2=7

## Setpoint Reference
PositionReference Reference_State
