;; Auto-generated. Do not edit!


(when (boundp 'quadrotor_msgs::LowLevelFeedback)
  (if (not (find-package "QUADROTOR_MSGS"))
    (make-package "QUADROTOR_MSGS"))
  (shadow 'LowLevelFeedback (find-package "QUADROTOR_MSGS")))
(unless (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK")
  (make-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))

(in-package "ROS")
;;//! \htmlinclude LowLevelFeedback.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*BAT_INVALID*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*BAT_INVALID* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*BAT_INVALID* 0)
(intern "*BAT_GOOD*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*BAT_GOOD* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*BAT_GOOD* 1)
(intern "*BAT_LOW*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*BAT_LOW* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*BAT_LOW* 2)
(intern "*BAT_CRITICAL*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*BAT_CRITICAL* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*BAT_CRITICAL* 3)
(intern "*NONE*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*NONE* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*NONE* 0)
(intern "*ATTITUDE*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*ATTITUDE* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*ATTITUDE* 1)
(intern "*BODY_RATES*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*BODY_RATES* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*BODY_RATES* 2)
(intern "*ANGULAR_ACCELERATION*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*ANGULAR_ACCELERATION* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*ANGULAR_ACCELERATION* 3)
(intern "*ROTOR_THRUSTS*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*ROTOR_THRUSTS* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*ROTOR_THRUSTS* 4)
(intern "*RC_MANUAL*" (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(shadow '*RC_MANUAL* (find-package "QUADROTOR_MSGS::LOWLEVELFEEDBACK"))
(defconstant quadrotor_msgs::LowLevelFeedback::*RC_MANUAL* 10)

(defun quadrotor_msgs::LowLevelFeedback-to-symbol (const)
  (cond
        ((= const 0) 'quadrotor_msgs::LowLevelFeedback::*BAT_INVALID*)
        ((= const 1) 'quadrotor_msgs::LowLevelFeedback::*BAT_GOOD*)
        ((= const 2) 'quadrotor_msgs::LowLevelFeedback::*BAT_LOW*)
        ((= const 3) 'quadrotor_msgs::LowLevelFeedback::*BAT_CRITICAL*)
        ((= const 0) 'quadrotor_msgs::LowLevelFeedback::*NONE*)
        ((= const 1) 'quadrotor_msgs::LowLevelFeedback::*ATTITUDE*)
        ((= const 2) 'quadrotor_msgs::LowLevelFeedback::*BODY_RATES*)
        ((= const 3) 'quadrotor_msgs::LowLevelFeedback::*ANGULAR_ACCELERATION*)
        ((= const 4) 'quadrotor_msgs::LowLevelFeedback::*ROTOR_THRUSTS*)
        ((= const 10) 'quadrotor_msgs::LowLevelFeedback::*RC_MANUAL*)
        (t nil)))

(defclass quadrotor_msgs::LowLevelFeedback
  :super ros::object
  :slots (_header _battery_voltage _battery_state _control_mode _motor_speeds _thrust_mapping_coeffs ))

(defmethod quadrotor_msgs::LowLevelFeedback
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:battery_voltage __battery_voltage) 0.0)
    ((:battery_state __battery_state) 0)
    ((:control_mode __control_mode) 0)
    ((:motor_speeds __motor_speeds) (make-array 0 :initial-element 0 :element-type :integer))
    ((:thrust_mapping_coeffs __thrust_mapping_coeffs) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _battery_voltage (float __battery_voltage))
   (setq _battery_state (round __battery_state))
   (setq _control_mode (round __control_mode))
   (setq _motor_speeds __motor_speeds)
   (setq _thrust_mapping_coeffs __thrust_mapping_coeffs)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:battery_voltage
   (&optional __battery_voltage)
   (if __battery_voltage (setq _battery_voltage __battery_voltage)) _battery_voltage)
  (:battery_state
   (&optional __battery_state)
   (if __battery_state (setq _battery_state __battery_state)) _battery_state)
  (:control_mode
   (&optional __control_mode)
   (if __control_mode (setq _control_mode __control_mode)) _control_mode)
  (:motor_speeds
   (&optional __motor_speeds)
   (if __motor_speeds (setq _motor_speeds __motor_speeds)) _motor_speeds)
  (:thrust_mapping_coeffs
   (&optional __thrust_mapping_coeffs)
   (if __thrust_mapping_coeffs (setq _thrust_mapping_coeffs __thrust_mapping_coeffs)) _thrust_mapping_coeffs)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32 _battery_voltage
    4
    ;; uint8 _battery_state
    1
    ;; uint8 _control_mode
    1
    ;; int16[] _motor_speeds
    (* 2    (length _motor_speeds)) 4
    ;; float64[] _thrust_mapping_coeffs
    (* 8    (length _thrust_mapping_coeffs)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32 _battery_voltage
       (sys::poke _battery_voltage (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint8 _battery_state
       (write-byte _battery_state s)
     ;; uint8 _control_mode
       (write-byte _control_mode s)
     ;; int16[] _motor_speeds
     (write-long (length _motor_speeds) s)
     (dotimes (i (length _motor_speeds))
       (write-word (elt _motor_speeds i) s)
       )
     ;; float64[] _thrust_mapping_coeffs
     (write-long (length _thrust_mapping_coeffs) s)
     (dotimes (i (length _thrust_mapping_coeffs))
       (sys::poke (elt _thrust_mapping_coeffs i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32 _battery_voltage
     (setq _battery_voltage (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint8 _battery_state
     (setq _battery_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _control_mode
     (setq _control_mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; int16[] _motor_speeds
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _motor_speeds (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _motor_speeds i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     ))
   ;; float64[] _thrust_mapping_coeffs
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _thrust_mapping_coeffs (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _thrust_mapping_coeffs i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get quadrotor_msgs::LowLevelFeedback :md5sum-) "e3cfad3ba98dfdc505bcf1fe91833d87")
(setf (get quadrotor_msgs::LowLevelFeedback :datatype-) "quadrotor_msgs/LowLevelFeedback")
(setf (get quadrotor_msgs::LowLevelFeedback :definition-)
      "# battery state enums
uint8 BAT_INVALID=0
uint8 BAT_GOOD=1
uint8 BAT_LOW=2
uint8 BAT_CRITICAL=3

# control mode enums as defined in ControlCommand.msg
uint8 NONE=0
uint8 ATTITUDE=1
uint8 BODY_RATES=2
uint8 ANGULAR_ACCELERATION=3
uint8 ROTOR_THRUSTS=4
# Additionally to the control command we want to know whether an RC has taken
# over from the low level feedback
uint8 RC_MANUAL=10

Header header

# Battery information
float32 battery_voltage
uint8 battery_state

# Control mode as defined above
uint8 control_mode

# Motor speed feedback [rpm]
int16[] motor_speeds

# Thrust mapping coefficients
# thrust = thrust_mapping_coeffs[2] * u^2 + thrust_mapping_coeffs[1] * u +
#     thrust_mapping_coeffs[0]
float64[] thrust_mapping_coeffs

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



(provide :quadrotor_msgs/LowLevelFeedback "e3cfad3ba98dfdc505bcf1fe91833d87")


