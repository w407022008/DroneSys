;; Auto-generated. Do not edit!


(when (boundp 'quadrotor_msgs::AutopilotFeedback)
  (if (not (find-package "QUADROTOR_MSGS"))
    (make-package "QUADROTOR_MSGS"))
  (shadow 'AutopilotFeedback (find-package "QUADROTOR_MSGS")))
(unless (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK")
  (make-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))

(in-package "ROS")
;;//! \htmlinclude AutopilotFeedback.msg.html
(if (not (find-package "NAV_MSGS"))
  (ros::roseus-add-msgs "nav_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*OFF*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*OFF* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*OFF* 0)
(intern "*START*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*START* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*START* 1)
(intern "*HOVER*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*HOVER* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*HOVER* 2)
(intern "*LAND*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*LAND* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*LAND* 3)
(intern "*EMERGENCY_LAND*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*EMERGENCY_LAND* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*EMERGENCY_LAND* 4)
(intern "*BREAKING*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*BREAKING* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*BREAKING* 5)
(intern "*GO_TO_POSE*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*GO_TO_POSE* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*GO_TO_POSE* 6)
(intern "*VELOCITY_CONTROL*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*VELOCITY_CONTROL* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*VELOCITY_CONTROL* 7)
(intern "*REFERENCE_CONTROL*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*REFERENCE_CONTROL* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*REFERENCE_CONTROL* 8)
(intern "*TRAJECTORY_CONTROL*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*TRAJECTORY_CONTROL* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*TRAJECTORY_CONTROL* 9)
(intern "*COMMAND_FEEDTHROUGH*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*COMMAND_FEEDTHROUGH* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*COMMAND_FEEDTHROUGH* 10)
(intern "*RC_MANUAL*" (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(shadow '*RC_MANUAL* (find-package "QUADROTOR_MSGS::AUTOPILOTFEEDBACK"))
(defconstant quadrotor_msgs::AutopilotFeedback::*RC_MANUAL* 11)

(defun quadrotor_msgs::AutopilotFeedback-to-symbol (const)
  (cond
        ((= const 0) 'quadrotor_msgs::AutopilotFeedback::*OFF*)
        ((= const 1) 'quadrotor_msgs::AutopilotFeedback::*START*)
        ((= const 2) 'quadrotor_msgs::AutopilotFeedback::*HOVER*)
        ((= const 3) 'quadrotor_msgs::AutopilotFeedback::*LAND*)
        ((= const 4) 'quadrotor_msgs::AutopilotFeedback::*EMERGENCY_LAND*)
        ((= const 5) 'quadrotor_msgs::AutopilotFeedback::*BREAKING*)
        ((= const 6) 'quadrotor_msgs::AutopilotFeedback::*GO_TO_POSE*)
        ((= const 7) 'quadrotor_msgs::AutopilotFeedback::*VELOCITY_CONTROL*)
        ((= const 8) 'quadrotor_msgs::AutopilotFeedback::*REFERENCE_CONTROL*)
        ((= const 9) 'quadrotor_msgs::AutopilotFeedback::*TRAJECTORY_CONTROL*)
        ((= const 10) 'quadrotor_msgs::AutopilotFeedback::*COMMAND_FEEDTHROUGH*)
        ((= const 11) 'quadrotor_msgs::AutopilotFeedback::*RC_MANUAL*)
        (t nil)))

(defclass quadrotor_msgs::AutopilotFeedback
  :super ros::object
  :slots (_header _autopilot_state _control_command_delay _control_computation_time _trajectory_execution_left_duration _trajectories_left_in_queue _low_level_feedback _reference_state _state_estimate ))

(defmethod quadrotor_msgs::AutopilotFeedback
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:autopilot_state __autopilot_state) 0)
    ((:control_command_delay __control_command_delay) (instance ros::time :init))
    ((:control_computation_time __control_computation_time) (instance ros::time :init))
    ((:trajectory_execution_left_duration __trajectory_execution_left_duration) (instance ros::time :init))
    ((:trajectories_left_in_queue __trajectories_left_in_queue) 0)
    ((:low_level_feedback __low_level_feedback) (instance quadrotor_msgs::LowLevelFeedback :init))
    ((:reference_state __reference_state) (instance quadrotor_msgs::TrajectoryPoint :init))
    ((:state_estimate __state_estimate) (instance nav_msgs::Odometry :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _autopilot_state (round __autopilot_state))
   (setq _control_command_delay __control_command_delay)
   (setq _control_computation_time __control_computation_time)
   (setq _trajectory_execution_left_duration __trajectory_execution_left_duration)
   (setq _trajectories_left_in_queue (round __trajectories_left_in_queue))
   (setq _low_level_feedback __low_level_feedback)
   (setq _reference_state __reference_state)
   (setq _state_estimate __state_estimate)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:autopilot_state
   (&optional __autopilot_state)
   (if __autopilot_state (setq _autopilot_state __autopilot_state)) _autopilot_state)
  (:control_command_delay
   (&optional __control_command_delay)
   (if __control_command_delay (setq _control_command_delay __control_command_delay)) _control_command_delay)
  (:control_computation_time
   (&optional __control_computation_time)
   (if __control_computation_time (setq _control_computation_time __control_computation_time)) _control_computation_time)
  (:trajectory_execution_left_duration
   (&optional __trajectory_execution_left_duration)
   (if __trajectory_execution_left_duration (setq _trajectory_execution_left_duration __trajectory_execution_left_duration)) _trajectory_execution_left_duration)
  (:trajectories_left_in_queue
   (&optional __trajectories_left_in_queue)
   (if __trajectories_left_in_queue (setq _trajectories_left_in_queue __trajectories_left_in_queue)) _trajectories_left_in_queue)
  (:low_level_feedback
   (&rest __low_level_feedback)
   (if (keywordp (car __low_level_feedback))
       (send* _low_level_feedback __low_level_feedback)
     (progn
       (if __low_level_feedback (setq _low_level_feedback (car __low_level_feedback)))
       _low_level_feedback)))
  (:reference_state
   (&rest __reference_state)
   (if (keywordp (car __reference_state))
       (send* _reference_state __reference_state)
     (progn
       (if __reference_state (setq _reference_state (car __reference_state)))
       _reference_state)))
  (:state_estimate
   (&rest __state_estimate)
   (if (keywordp (car __state_estimate))
       (send* _state_estimate __state_estimate)
     (progn
       (if __state_estimate (setq _state_estimate (car __state_estimate)))
       _state_estimate)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8 _autopilot_state
    1
    ;; duration _control_command_delay
    8
    ;; duration _control_computation_time
    8
    ;; duration _trajectory_execution_left_duration
    8
    ;; uint8 _trajectories_left_in_queue
    1
    ;; quadrotor_msgs/LowLevelFeedback _low_level_feedback
    (send _low_level_feedback :serialization-length)
    ;; quadrotor_msgs/TrajectoryPoint _reference_state
    (send _reference_state :serialization-length)
    ;; nav_msgs/Odometry _state_estimate
    (send _state_estimate :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8 _autopilot_state
       (write-byte _autopilot_state s)
     ;; duration _control_command_delay
       (write-long (send _control_command_delay :sec) s) (write-long (send _control_command_delay :nsec) s)
     ;; duration _control_computation_time
       (write-long (send _control_computation_time :sec) s) (write-long (send _control_computation_time :nsec) s)
     ;; duration _trajectory_execution_left_duration
       (write-long (send _trajectory_execution_left_duration :sec) s) (write-long (send _trajectory_execution_left_duration :nsec) s)
     ;; uint8 _trajectories_left_in_queue
       (write-byte _trajectories_left_in_queue s)
     ;; quadrotor_msgs/LowLevelFeedback _low_level_feedback
       (send _low_level_feedback :serialize s)
     ;; quadrotor_msgs/TrajectoryPoint _reference_state
       (send _reference_state :serialize s)
     ;; nav_msgs/Odometry _state_estimate
       (send _state_estimate :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8 _autopilot_state
     (setq _autopilot_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; duration _control_command_delay
     (send _control_command_delay :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _control_command_delay :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; duration _control_computation_time
     (send _control_computation_time :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _control_computation_time :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; duration _trajectory_execution_left_duration
     (send _trajectory_execution_left_duration :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _trajectory_execution_left_duration :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint8 _trajectories_left_in_queue
     (setq _trajectories_left_in_queue (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; quadrotor_msgs/LowLevelFeedback _low_level_feedback
     (send _low_level_feedback :deserialize buf ptr-) (incf ptr- (send _low_level_feedback :serialization-length))
   ;; quadrotor_msgs/TrajectoryPoint _reference_state
     (send _reference_state :deserialize buf ptr-) (incf ptr- (send _reference_state :serialization-length))
   ;; nav_msgs/Odometry _state_estimate
     (send _state_estimate :deserialize buf ptr-) (incf ptr- (send _state_estimate :serialization-length))
   ;;
   self)
  )

(setf (get quadrotor_msgs::AutopilotFeedback :md5sum-) "8c8e08f7c3465bc93596097f7c8ecc39")
(setf (get quadrotor_msgs::AutopilotFeedback :datatype-) "quadrotor_msgs/AutopilotFeedback")
(setf (get quadrotor_msgs::AutopilotFeedback :definition-)
      "# Autopilot state enums
uint8 OFF=0
uint8 START=1
uint8 HOVER=2
uint8 LAND=3
uint8 EMERGENCY_LAND=4
uint8 BREAKING=5
uint8 GO_TO_POSE=6
uint8 VELOCITY_CONTROL=7
uint8 REFERENCE_CONTROL=8
uint8 TRAJECTORY_CONTROL=9
uint8 COMMAND_FEEDTHROUGH=10
uint8 RC_MANUAL=11


Header header

# Autopilot state as defined above. This reflects what is implemented in
# autopilot/include/autopilot/autopilot.h
uint8 autopilot_state

# Control command delay
duration control_command_delay

# Controller computation time [s]
duration control_computation_time

# Duration left of the trajectories in the queue
# Only valid in TRAJECTORY_CONTROL mode
duration trajectory_execution_left_duration

# Number of trajectories that were sent to the autopilot and are stored in its
# queue. Only valid in TRAJECTORY_CONTROL mode
uint8 trajectories_left_in_queue

# Low level feedback
quadrotor_msgs/LowLevelFeedback low_level_feedback

# Desired state used to compute the control command
quadrotor_msgs/TrajectoryPoint reference_state

# State estimate used to compute the control command
nav_msgs/Odometry state_estimate

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
MSG: quadrotor_msgs/LowLevelFeedback
# battery state enums
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
MSG: quadrotor_msgs/TrajectoryPoint
duration time_from_start

geometry_msgs/Pose pose

geometry_msgs/Twist velocity

geometry_msgs/Twist acceleration

geometry_msgs/Twist jerk

geometry_msgs/Twist snap

# Heading angle with respect to world frame [rad]
float64 heading

# First derivative of the heading angle [rad/s]
float64 heading_rate

# Second derivative of the heading angle [rad/s^2]
float64 heading_acceleration

# Collective thrust [m/s^2]
float64 thrust
================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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
================================================================================
MSG: nav_msgs/Odometry
# This represents an estimate of a position and velocity in free space.  
# The pose in this message should be specified in the coordinate frame given by header.frame_id.
# The twist in this message should be specified in the coordinate frame given by the child_frame_id
Header header
string child_frame_id
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist

================================================================================
MSG: geometry_msgs/PoseWithCovariance
# This represents a pose in free space with uncertainty.

Pose pose

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance

================================================================================
MSG: geometry_msgs/TwistWithCovariance
# This expresses velocity in free space with uncertainty.

Twist twist

# Row-major representation of the 6x6 covariance matrix
# The orientation parameters use a fixed-axis representation.
# In order, the parameters are:
# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
float64[36] covariance

")



(provide :quadrotor_msgs/AutopilotFeedback "8c8e08f7c3465bc93596097f7c8ecc39")


