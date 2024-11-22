;; Auto-generated. Do not edit!


(when (boundp 'quadrotor_msgs::ControlCommand)
  (if (not (find-package "QUADROTOR_MSGS"))
    (make-package "QUADROTOR_MSGS"))
  (shadow 'ControlCommand (find-package "QUADROTOR_MSGS")))
(unless (find-package "QUADROTOR_MSGS::CONTROLCOMMAND")
  (make-package "QUADROTOR_MSGS::CONTROLCOMMAND"))

(in-package "ROS")
;;//! \htmlinclude ControlCommand.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*NONE*" (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(shadow '*NONE* (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(defconstant quadrotor_msgs::ControlCommand::*NONE* 0)
(intern "*ATTITUDE*" (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(shadow '*ATTITUDE* (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(defconstant quadrotor_msgs::ControlCommand::*ATTITUDE* 1)
(intern "*BODY_RATES*" (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(shadow '*BODY_RATES* (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(defconstant quadrotor_msgs::ControlCommand::*BODY_RATES* 2)
(intern "*ANGULAR_ACCELERATIONS*" (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(shadow '*ANGULAR_ACCELERATIONS* (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(defconstant quadrotor_msgs::ControlCommand::*ANGULAR_ACCELERATIONS* 3)
(intern "*ROTOR_THRUSTS*" (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(shadow '*ROTOR_THRUSTS* (find-package "QUADROTOR_MSGS::CONTROLCOMMAND"))
(defconstant quadrotor_msgs::ControlCommand::*ROTOR_THRUSTS* 4)

(defun quadrotor_msgs::ControlCommand-to-symbol (const)
  (cond
        ((= const 0) 'quadrotor_msgs::ControlCommand::*NONE*)
        ((= const 1) 'quadrotor_msgs::ControlCommand::*ATTITUDE*)
        ((= const 2) 'quadrotor_msgs::ControlCommand::*BODY_RATES*)
        ((= const 3) 'quadrotor_msgs::ControlCommand::*ANGULAR_ACCELERATIONS*)
        ((= const 4) 'quadrotor_msgs::ControlCommand::*ROTOR_THRUSTS*)
        (t nil)))

(defclass quadrotor_msgs::ControlCommand
  :super ros::object
  :slots (_header _control_mode _armed _expected_execution_time _orientation _bodyrates _angular_accelerations _collective_thrust _rotor_thrusts ))

(defmethod quadrotor_msgs::ControlCommand
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:control_mode __control_mode) 0)
    ((:armed __armed) nil)
    ((:expected_execution_time __expected_execution_time) (instance ros::time :init))
    ((:orientation __orientation) (instance geometry_msgs::Quaternion :init))
    ((:bodyrates __bodyrates) (instance geometry_msgs::Vector3 :init))
    ((:angular_accelerations __angular_accelerations) (instance geometry_msgs::Vector3 :init))
    ((:collective_thrust __collective_thrust) 0.0)
    ((:rotor_thrusts __rotor_thrusts) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _control_mode (round __control_mode))
   (setq _armed __armed)
   (setq _expected_execution_time __expected_execution_time)
   (setq _orientation __orientation)
   (setq _bodyrates __bodyrates)
   (setq _angular_accelerations __angular_accelerations)
   (setq _collective_thrust (float __collective_thrust))
   (setq _rotor_thrusts __rotor_thrusts)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:control_mode
   (&optional __control_mode)
   (if __control_mode (setq _control_mode __control_mode)) _control_mode)
  (:armed
   (&optional (__armed :null))
   (if (not (eq __armed :null)) (setq _armed __armed)) _armed)
  (:expected_execution_time
   (&optional __expected_execution_time)
   (if __expected_execution_time (setq _expected_execution_time __expected_execution_time)) _expected_execution_time)
  (:orientation
   (&rest __orientation)
   (if (keywordp (car __orientation))
       (send* _orientation __orientation)
     (progn
       (if __orientation (setq _orientation (car __orientation)))
       _orientation)))
  (:bodyrates
   (&rest __bodyrates)
   (if (keywordp (car __bodyrates))
       (send* _bodyrates __bodyrates)
     (progn
       (if __bodyrates (setq _bodyrates (car __bodyrates)))
       _bodyrates)))
  (:angular_accelerations
   (&rest __angular_accelerations)
   (if (keywordp (car __angular_accelerations))
       (send* _angular_accelerations __angular_accelerations)
     (progn
       (if __angular_accelerations (setq _angular_accelerations (car __angular_accelerations)))
       _angular_accelerations)))
  (:collective_thrust
   (&optional __collective_thrust)
   (if __collective_thrust (setq _collective_thrust __collective_thrust)) _collective_thrust)
  (:rotor_thrusts
   (&optional __rotor_thrusts)
   (if __rotor_thrusts (setq _rotor_thrusts __rotor_thrusts)) _rotor_thrusts)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8 _control_mode
    1
    ;; bool _armed
    1
    ;; time _expected_execution_time
    8
    ;; geometry_msgs/Quaternion _orientation
    (send _orientation :serialization-length)
    ;; geometry_msgs/Vector3 _bodyrates
    (send _bodyrates :serialization-length)
    ;; geometry_msgs/Vector3 _angular_accelerations
    (send _angular_accelerations :serialization-length)
    ;; float64 _collective_thrust
    8
    ;; float64[] _rotor_thrusts
    (* 8    (length _rotor_thrusts)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8 _control_mode
       (write-byte _control_mode s)
     ;; bool _armed
       (if _armed (write-byte -1 s) (write-byte 0 s))
     ;; time _expected_execution_time
       (write-long (send _expected_execution_time :sec) s) (write-long (send _expected_execution_time :nsec) s)
     ;; geometry_msgs/Quaternion _orientation
       (send _orientation :serialize s)
     ;; geometry_msgs/Vector3 _bodyrates
       (send _bodyrates :serialize s)
     ;; geometry_msgs/Vector3 _angular_accelerations
       (send _angular_accelerations :serialize s)
     ;; float64 _collective_thrust
       (sys::poke _collective_thrust (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64[] _rotor_thrusts
     (write-long (length _rotor_thrusts) s)
     (dotimes (i (length _rotor_thrusts))
       (sys::poke (elt _rotor_thrusts i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8 _control_mode
     (setq _control_mode (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; bool _armed
     (setq _armed (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; time _expected_execution_time
     (send _expected_execution_time :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _expected_execution_time :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; geometry_msgs/Quaternion _orientation
     (send _orientation :deserialize buf ptr-) (incf ptr- (send _orientation :serialization-length))
   ;; geometry_msgs/Vector3 _bodyrates
     (send _bodyrates :deserialize buf ptr-) (incf ptr- (send _bodyrates :serialization-length))
   ;; geometry_msgs/Vector3 _angular_accelerations
     (send _angular_accelerations :deserialize buf ptr-) (incf ptr- (send _angular_accelerations :serialization-length))
   ;; float64 _collective_thrust
     (setq _collective_thrust (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64[] _rotor_thrusts
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _rotor_thrusts (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _rotor_thrusts i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get quadrotor_msgs::ControlCommand :md5sum-) "a1918a34164f6647c898e4d55efbfcef")
(setf (get quadrotor_msgs::ControlCommand :datatype-) "quadrotor_msgs/ControlCommand")
(setf (get quadrotor_msgs::ControlCommand :definition-)
      "# Quadrotor control command

# control mode enums
uint8 NONE=0
uint8 ATTITUDE=1
uint8 BODY_RATES=2
uint8 ANGULAR_ACCELERATIONS=3
uint8 ROTOR_THRUSTS=4

Header header

# Control mode as defined above
uint8 control_mode

# Flag whether controller is allowed to arm
bool armed

# Time at which this command should be executed
time expected_execution_time

# Orientation of the body frame with respect to the world frame [-]
geometry_msgs/Quaternion orientation

# Body rates in body frame [rad/s]
# Note that in ATTITUDE mode the x-y-bodyrates are only feed forward terms 
# computed from a reference trajectory
# Also in ATTITUDE mode, the z-bodyrate has to be from feedback control
geometry_msgs/Vector3 bodyrates

# Angular accelerations in body frame [rad/s^2]
geometry_msgs/Vector3 angular_accelerations

# Collective mass normalized thrust [m/s^2]
float64 collective_thrust

# Single rotor thrusts [N]
# These are only considered in the ROTOR_THRUSTS control mode
float64[] rotor_thrusts

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



(provide :quadrotor_msgs/ControlCommand "a1918a34164f6647c898e4d55efbfcef")


