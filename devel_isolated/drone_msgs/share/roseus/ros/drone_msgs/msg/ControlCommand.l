;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::ControlCommand)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'ControlCommand (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::CONTROLCOMMAND")
  (make-package "DRONE_MSGS::CONTROLCOMMAND"))

(in-package "ROS")
;;//! \htmlinclude ControlCommand.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*IDLE*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*IDLE* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*IDLE* 0)
(intern "*TAKEOFF*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*TAKEOFF* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*TAKEOFF* 1)
(intern "*HOLD*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*HOLD* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*HOLD* 2)
(intern "*LAND*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*LAND* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*LAND* 3)
(intern "*MOVE*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*MOVE* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*MOVE* 4)
(intern "*DISARM*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*DISARM* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*DISARM* 5)
(intern "*ATTITUDE*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*ATTITUDE* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*ATTITUDE* 6)
(intern "*ATTITUDERATE*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*ATTITUDERATE* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*ATTITUDERATE* 7)
(intern "*RATE*" (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(shadow '*RATE* (find-package "DRONE_MSGS::CONTROLCOMMAND"))
(defconstant drone_msgs::ControlCommand::*RATE* 8)

(defun drone_msgs::ControlCommand-to-symbol (const)
  (cond
        ((= const 0) 'drone_msgs::ControlCommand::*Idle*)
        ((= const 1) 'drone_msgs::ControlCommand::*Takeoff*)
        ((= const 2) 'drone_msgs::ControlCommand::*Hold*)
        ((= const 3) 'drone_msgs::ControlCommand::*Land*)
        ((= const 4) 'drone_msgs::ControlCommand::*Move*)
        ((= const 5) 'drone_msgs::ControlCommand::*Disarm*)
        ((= const 6) 'drone_msgs::ControlCommand::*Attitude*)
        ((= const 7) 'drone_msgs::ControlCommand::*AttitudeRate*)
        ((= const 8) 'drone_msgs::ControlCommand::*Rate*)
        (t nil)))

(defclass drone_msgs::ControlCommand
  :super ros::object
  :slots (_header _Command_ID _source _Mode _Reference_State _Attitude_sp ))

(defmethod drone_msgs::ControlCommand
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:Command_ID __Command_ID) 0)
    ((:source __source) "")
    ((:Mode __Mode) 0)
    ((:Reference_State __Reference_State) (instance drone_msgs::PositionReference :init))
    ((:Attitude_sp __Attitude_sp) (instance drone_msgs::AttitudeReference :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _Command_ID (round __Command_ID))
   (setq _source (string __source))
   (setq _Mode (round __Mode))
   (setq _Reference_State __Reference_State)
   (setq _Attitude_sp __Attitude_sp)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:Command_ID
   (&optional __Command_ID)
   (if __Command_ID (setq _Command_ID __Command_ID)) _Command_ID)
  (:source
   (&optional __source)
   (if __source (setq _source __source)) _source)
  (:Mode
   (&optional __Mode)
   (if __Mode (setq _Mode __Mode)) _Mode)
  (:Reference_State
   (&rest __Reference_State)
   (if (keywordp (car __Reference_State))
       (send* _Reference_State __Reference_State)
     (progn
       (if __Reference_State (setq _Reference_State (car __Reference_State)))
       _Reference_State)))
  (:Attitude_sp
   (&rest __Attitude_sp)
   (if (keywordp (car __Attitude_sp))
       (send* _Attitude_sp __Attitude_sp)
     (progn
       (if __Attitude_sp (setq _Attitude_sp (car __Attitude_sp)))
       _Attitude_sp)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint32 _Command_ID
    4
    ;; string _source
    4 (length _source)
    ;; uint8 _Mode
    1
    ;; drone_msgs/PositionReference _Reference_State
    (send _Reference_State :serialization-length)
    ;; drone_msgs/AttitudeReference _Attitude_sp
    (send _Attitude_sp :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint32 _Command_ID
       (write-long _Command_ID s)
     ;; string _source
       (write-long (length _source) s) (princ _source s)
     ;; uint8 _Mode
       (write-byte _Mode s)
     ;; drone_msgs/PositionReference _Reference_State
       (send _Reference_State :serialize s)
     ;; drone_msgs/AttitudeReference _Attitude_sp
       (send _Attitude_sp :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint32 _Command_ID
     (setq _Command_ID (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; string _source
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _source (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; uint8 _Mode
     (setq _Mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; drone_msgs/PositionReference _Reference_State
     (send _Reference_State :deserialize buf ptr-) (incf ptr- (send _Reference_State :serialization-length))
   ;; drone_msgs/AttitudeReference _Attitude_sp
     (send _Attitude_sp :deserialize buf ptr-) (incf ptr- (send _Attitude_sp :serialization-length))
   ;;
   self)
  )

(setf (get drone_msgs::ControlCommand :md5sum-) "969640b304f3a446799efdd5c334e9b7")
(setf (get drone_msgs::ControlCommand :datatype-) "drone_msgs/ControlCommand")
(setf (get drone_msgs::ControlCommand :definition-)
      "std_msgs/Header header

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
uint8 Attitude=6
uint8 AttitudeRate=7
uint8 Rate=8

## Setpoint Reference
PositionReference Reference_State
AttitudeReference Attitude_sp

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: drone_msgs/PositionReference
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
================================================================================
MSG: drone_msgs/Bspline
int32 order                 ## 
int64 traj_id               ## id of trajecotry
float64[] knots             ## knots list
geometry_msgs/Point[] pts   ## control points list
time start_time             ## time stamp


================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: drone_msgs/AttitudeReference
std_msgs/Header header

## Setpoint Attitude + T
float32[3] thrust_sp                   ## Single Rotor Thrust setpoint
float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint
float32[3] desired_attitude            ## [rad] Eurler angle setpoint
geometry_msgs/Quaternion desired_att_q ## quat setpoint
geometry_msgs/Vector3 body_rate  ## [rad/s]

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :drone_msgs/ControlCommand "969640b304f3a446799efdd5c334e9b7")


