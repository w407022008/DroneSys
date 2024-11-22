;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::RCInput)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'RCInput (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::RCINPUT")
  (make-package "DRONE_MSGS::RCINPUT"))

(in-package "ROS")
;;//! \htmlinclude RCInput.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*DISABLE*" (find-package "DRONE_MSGS::RCINPUT"))
(shadow '*DISABLE* (find-package "DRONE_MSGS::RCINPUT"))
(defconstant drone_msgs::RCInput::*DISABLE* 0)
(intern "*MAVROS_MANUAL_CONTROL*" (find-package "DRONE_MSGS::RCINPUT"))
(shadow '*MAVROS_MANUAL_CONTROL* (find-package "DRONE_MSGS::RCINPUT"))
(defconstant drone_msgs::RCInput::*MAVROS_MANUAL_CONTROL* 1)
(intern "*DRIVER_JOYSTICK*" (find-package "DRONE_MSGS::RCINPUT"))
(shadow '*DRIVER_JOYSTICK* (find-package "DRONE_MSGS::RCINPUT"))
(defconstant drone_msgs::RCInput::*DRIVER_JOYSTICK* 2)

(defun drone_msgs::RCInput-to-symbol (const)
  (cond
        ((= const 0) 'drone_msgs::RCInput::*DISABLE*)
        ((= const 1) 'drone_msgs::RCInput::*MAVROS_MANUAL_CONTROL*)
        ((= const 2) 'drone_msgs::RCInput::*DRIVER_JOYSTICK*)
        (t nil)))

(defclass drone_msgs::RCInput
  :super ros::object
  :slots (_header _rc_x _rc_y _rc_z _rc_r _buttons _goal_enable _data_source ))

(defmethod drone_msgs::RCInput
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:rc_x __rc_x) 0.0)
    ((:rc_y __rc_y) 0.0)
    ((:rc_z __rc_z) 0.0)
    ((:rc_r __rc_r) 0.0)
    ((:buttons __buttons) 0)
    ((:goal_enable __goal_enable) 0)
    ((:data_source __data_source) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _rc_x (float __rc_x))
   (setq _rc_y (float __rc_y))
   (setq _rc_z (float __rc_z))
   (setq _rc_r (float __rc_r))
   (setq _buttons (round __buttons))
   (setq _goal_enable (round __goal_enable))
   (setq _data_source (round __data_source))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:rc_x
   (&optional __rc_x)
   (if __rc_x (setq _rc_x __rc_x)) _rc_x)
  (:rc_y
   (&optional __rc_y)
   (if __rc_y (setq _rc_y __rc_y)) _rc_y)
  (:rc_z
   (&optional __rc_z)
   (if __rc_z (setq _rc_z __rc_z)) _rc_z)
  (:rc_r
   (&optional __rc_r)
   (if __rc_r (setq _rc_r __rc_r)) _rc_r)
  (:buttons
   (&optional __buttons)
   (if __buttons (setq _buttons __buttons)) _buttons)
  (:goal_enable
   (&optional __goal_enable)
   (if __goal_enable (setq _goal_enable __goal_enable)) _goal_enable)
  (:data_source
   (&optional __data_source)
   (if __data_source (setq _data_source __data_source)) _data_source)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32 _rc_x
    4
    ;; float32 _rc_y
    4
    ;; float32 _rc_z
    4
    ;; float32 _rc_r
    4
    ;; uint32 _buttons
    4
    ;; uint8 _goal_enable
    1
    ;; int32 _data_source
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32 _rc_x
       (sys::poke _rc_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rc_y
       (sys::poke _rc_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rc_z
       (sys::poke _rc_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _rc_r
       (sys::poke _rc_r (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint32 _buttons
       (write-long _buttons s)
     ;; uint8 _goal_enable
       (write-byte _goal_enable s)
     ;; int32 _data_source
       (write-long _data_source s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32 _rc_x
     (setq _rc_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rc_y
     (setq _rc_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rc_z
     (setq _rc_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _rc_r
     (setq _rc_r (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint32 _buttons
     (setq _buttons (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint8 _goal_enable
     (setq _goal_enable (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; int32 _data_source
     (setq _data_source (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get drone_msgs::RCInput :md5sum-) "13d10a65cefb07444f918f9ce0babb28")
(setf (get drone_msgs::RCInput :datatype-) "drone_msgs/RCInput")
(setf (get drone_msgs::RCInput :definition-)
      "## Radio Control Input
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

")



(provide :drone_msgs/RCInput "13d10a65cefb07444f918f9ce0babb28")


