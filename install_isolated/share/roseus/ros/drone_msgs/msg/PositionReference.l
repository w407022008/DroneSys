;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::PositionReference)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'PositionReference (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::POSITIONREFERENCE")
  (make-package "DRONE_MSGS::POSITIONREFERENCE"))

(in-package "ROS")
;;//! \htmlinclude PositionReference.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*XYZ_POS*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*XYZ_POS* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*XYZ_POS* 0)
(intern "*XY_POS_Z_VEL*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*XY_POS_Z_VEL* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*XY_POS_Z_VEL* 1)
(intern "*XY_VEL_Z_POS*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*XY_VEL_Z_POS* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*XY_VEL_Z_POS* 2)
(intern "*XYZ_VEL*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*XYZ_VEL* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*XYZ_VEL* 3)
(intern "*XYZ_ACC*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*XYZ_ACC* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*XYZ_ACC* 4)
(intern "*XYZ_POS_VEL*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*XYZ_POS_VEL* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*XYZ_POS_VEL* 5)
(intern "*TRAJECTORY*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*TRAJECTORY* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*TRAJECTORY* 6)
(intern "*ENU_FRAME*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*ENU_FRAME* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*ENU_FRAME* 0)
(intern "*BODY_FRAME*" (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(shadow '*BODY_FRAME* (find-package "DRONE_MSGS::POSITIONREFERENCE"))
(defconstant drone_msgs::PositionReference::*BODY_FRAME* 1)

(defun drone_msgs::PositionReference-to-symbol (const)
  (cond
        ((= const 0) 'drone_msgs::PositionReference::*XYZ_POS*)
        ((= const 1) 'drone_msgs::PositionReference::*XY_POS_Z_VEL*)
        ((= const 2) 'drone_msgs::PositionReference::*XY_VEL_Z_POS*)
        ((= const 3) 'drone_msgs::PositionReference::*XYZ_VEL*)
        ((= const 4) 'drone_msgs::PositionReference::*XYZ_ACC*)
        ((= const 5) 'drone_msgs::PositionReference::*XYZ_POS_VEL*)
        ((= const 6) 'drone_msgs::PositionReference::*TRAJECTORY*)
        ((= const 0) 'drone_msgs::PositionReference::*ENU_FRAME*)
        ((= const 1) 'drone_msgs::PositionReference::*BODY_FRAME*)
        (t nil)))

(defclass drone_msgs::PositionReference
  :super ros::object
  :slots (_header _Move_mode _Move_frame _time_from_start _position_ref _velocity_ref _acceleration_ref _Yaw_Rate_Mode _yaw_ref _yaw_rate_ref _bspline ))

(defmethod drone_msgs::PositionReference
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:Move_mode __Move_mode) 0)
    ((:Move_frame __Move_frame) 0)
    ((:time_from_start __time_from_start) 0.0)
    ((:position_ref __position_ref) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:velocity_ref __velocity_ref) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:acceleration_ref __acceleration_ref) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:Yaw_Rate_Mode __Yaw_Rate_Mode) nil)
    ((:yaw_ref __yaw_ref) 0.0)
    ((:yaw_rate_ref __yaw_rate_ref) 0.0)
    ((:bspline __bspline) (instance drone_msgs::Bspline :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _Move_mode (round __Move_mode))
   (setq _Move_frame (round __Move_frame))
   (setq _time_from_start (float __time_from_start))
   (setq _position_ref __position_ref)
   (setq _velocity_ref __velocity_ref)
   (setq _acceleration_ref __acceleration_ref)
   (setq _Yaw_Rate_Mode __Yaw_Rate_Mode)
   (setq _yaw_ref (float __yaw_ref))
   (setq _yaw_rate_ref (float __yaw_rate_ref))
   (setq _bspline __bspline)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:Move_mode
   (&optional __Move_mode)
   (if __Move_mode (setq _Move_mode __Move_mode)) _Move_mode)
  (:Move_frame
   (&optional __Move_frame)
   (if __Move_frame (setq _Move_frame __Move_frame)) _Move_frame)
  (:time_from_start
   (&optional __time_from_start)
   (if __time_from_start (setq _time_from_start __time_from_start)) _time_from_start)
  (:position_ref
   (&optional __position_ref)
   (if __position_ref (setq _position_ref __position_ref)) _position_ref)
  (:velocity_ref
   (&optional __velocity_ref)
   (if __velocity_ref (setq _velocity_ref __velocity_ref)) _velocity_ref)
  (:acceleration_ref
   (&optional __acceleration_ref)
   (if __acceleration_ref (setq _acceleration_ref __acceleration_ref)) _acceleration_ref)
  (:Yaw_Rate_Mode
   (&optional (__Yaw_Rate_Mode :null))
   (if (not (eq __Yaw_Rate_Mode :null)) (setq _Yaw_Rate_Mode __Yaw_Rate_Mode)) _Yaw_Rate_Mode)
  (:yaw_ref
   (&optional __yaw_ref)
   (if __yaw_ref (setq _yaw_ref __yaw_ref)) _yaw_ref)
  (:yaw_rate_ref
   (&optional __yaw_rate_ref)
   (if __yaw_rate_ref (setq _yaw_rate_ref __yaw_rate_ref)) _yaw_rate_ref)
  (:bspline
   (&rest __bspline)
   (if (keywordp (car __bspline))
       (send* _bspline __bspline)
     (progn
       (if __bspline (setq _bspline (car __bspline)))
       _bspline)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8 _Move_mode
    1
    ;; uint8 _Move_frame
    1
    ;; float32 _time_from_start
    4
    ;; float32[3] _position_ref
    (* 4    3)
    ;; float32[3] _velocity_ref
    (* 4    3)
    ;; float32[3] _acceleration_ref
    (* 4    3)
    ;; bool _Yaw_Rate_Mode
    1
    ;; float32 _yaw_ref
    4
    ;; float32 _yaw_rate_ref
    4
    ;; drone_msgs/Bspline _bspline
    (send _bspline :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8 _Move_mode
       (write-byte _Move_mode s)
     ;; uint8 _Move_frame
       (write-byte _Move_frame s)
     ;; float32 _time_from_start
       (sys::poke _time_from_start (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32[3] _position_ref
     (dotimes (i 3)
       (sys::poke (elt _position_ref i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[3] _velocity_ref
     (dotimes (i 3)
       (sys::poke (elt _velocity_ref i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[3] _acceleration_ref
     (dotimes (i 3)
       (sys::poke (elt _acceleration_ref i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; bool _Yaw_Rate_Mode
       (if _Yaw_Rate_Mode (write-byte -1 s) (write-byte 0 s))
     ;; float32 _yaw_ref
       (sys::poke _yaw_ref (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _yaw_rate_ref
       (sys::poke _yaw_rate_ref (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; drone_msgs/Bspline _bspline
       (send _bspline :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8 _Move_mode
     (setq _Move_mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _Move_frame
     (setq _Move_frame (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32 _time_from_start
     (setq _time_from_start (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32[3] _position_ref
   (dotimes (i (length _position_ref))
     (setf (elt _position_ref i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[3] _velocity_ref
   (dotimes (i (length _velocity_ref))
     (setf (elt _velocity_ref i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[3] _acceleration_ref
   (dotimes (i (length _acceleration_ref))
     (setf (elt _acceleration_ref i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; bool _Yaw_Rate_Mode
     (setq _Yaw_Rate_Mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32 _yaw_ref
     (setq _yaw_ref (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _yaw_rate_ref
     (setq _yaw_rate_ref (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; drone_msgs/Bspline _bspline
     (send _bspline :deserialize buf ptr-) (incf ptr- (send _bspline :serialization-length))
   ;;
   self)
  )

(setf (get drone_msgs::PositionReference :md5sum-) "d029a5a9568f27a7cd91ef81b8f15a11")
(setf (get drone_msgs::PositionReference :datatype-) "drone_msgs/PositionReference")
(setf (get drone_msgs::PositionReference :definition-)
      "std_msgs/Header header

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

")



(provide :drone_msgs/PositionReference "d029a5a9568f27a7cd91ef81b8f15a11")


