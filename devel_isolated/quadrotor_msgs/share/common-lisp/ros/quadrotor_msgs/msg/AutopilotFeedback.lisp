; Auto-generated. Do not edit!


(cl:in-package quadrotor_msgs-msg)


;//! \htmlinclude AutopilotFeedback.msg.html

(cl:defclass <AutopilotFeedback> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (autopilot_state
    :reader autopilot_state
    :initarg :autopilot_state
    :type cl:fixnum
    :initform 0)
   (control_command_delay
    :reader control_command_delay
    :initarg :control_command_delay
    :type cl:real
    :initform 0)
   (control_computation_time
    :reader control_computation_time
    :initarg :control_computation_time
    :type cl:real
    :initform 0)
   (trajectory_execution_left_duration
    :reader trajectory_execution_left_duration
    :initarg :trajectory_execution_left_duration
    :type cl:real
    :initform 0)
   (trajectories_left_in_queue
    :reader trajectories_left_in_queue
    :initarg :trajectories_left_in_queue
    :type cl:fixnum
    :initform 0)
   (low_level_feedback
    :reader low_level_feedback
    :initarg :low_level_feedback
    :type quadrotor_msgs-msg:LowLevelFeedback
    :initform (cl:make-instance 'quadrotor_msgs-msg:LowLevelFeedback))
   (reference_state
    :reader reference_state
    :initarg :reference_state
    :type quadrotor_msgs-msg:TrajectoryPoint
    :initform (cl:make-instance 'quadrotor_msgs-msg:TrajectoryPoint))
   (state_estimate
    :reader state_estimate
    :initarg :state_estimate
    :type nav_msgs-msg:Odometry
    :initform (cl:make-instance 'nav_msgs-msg:Odometry)))
)

(cl:defclass AutopilotFeedback (<AutopilotFeedback>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AutopilotFeedback>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AutopilotFeedback)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name quadrotor_msgs-msg:<AutopilotFeedback> is deprecated: use quadrotor_msgs-msg:AutopilotFeedback instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:header-val is deprecated.  Use quadrotor_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'autopilot_state-val :lambda-list '(m))
(cl:defmethod autopilot_state-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:autopilot_state-val is deprecated.  Use quadrotor_msgs-msg:autopilot_state instead.")
  (autopilot_state m))

(cl:ensure-generic-function 'control_command_delay-val :lambda-list '(m))
(cl:defmethod control_command_delay-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:control_command_delay-val is deprecated.  Use quadrotor_msgs-msg:control_command_delay instead.")
  (control_command_delay m))

(cl:ensure-generic-function 'control_computation_time-val :lambda-list '(m))
(cl:defmethod control_computation_time-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:control_computation_time-val is deprecated.  Use quadrotor_msgs-msg:control_computation_time instead.")
  (control_computation_time m))

(cl:ensure-generic-function 'trajectory_execution_left_duration-val :lambda-list '(m))
(cl:defmethod trajectory_execution_left_duration-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:trajectory_execution_left_duration-val is deprecated.  Use quadrotor_msgs-msg:trajectory_execution_left_duration instead.")
  (trajectory_execution_left_duration m))

(cl:ensure-generic-function 'trajectories_left_in_queue-val :lambda-list '(m))
(cl:defmethod trajectories_left_in_queue-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:trajectories_left_in_queue-val is deprecated.  Use quadrotor_msgs-msg:trajectories_left_in_queue instead.")
  (trajectories_left_in_queue m))

(cl:ensure-generic-function 'low_level_feedback-val :lambda-list '(m))
(cl:defmethod low_level_feedback-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:low_level_feedback-val is deprecated.  Use quadrotor_msgs-msg:low_level_feedback instead.")
  (low_level_feedback m))

(cl:ensure-generic-function 'reference_state-val :lambda-list '(m))
(cl:defmethod reference_state-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:reference_state-val is deprecated.  Use quadrotor_msgs-msg:reference_state instead.")
  (reference_state m))

(cl:ensure-generic-function 'state_estimate-val :lambda-list '(m))
(cl:defmethod state_estimate-val ((m <AutopilotFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:state_estimate-val is deprecated.  Use quadrotor_msgs-msg:state_estimate instead.")
  (state_estimate m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<AutopilotFeedback>)))
    "Constants for message type '<AutopilotFeedback>"
  '((:OFF . 0)
    (:START . 1)
    (:HOVER . 2)
    (:LAND . 3)
    (:EMERGENCY_LAND . 4)
    (:BREAKING . 5)
    (:GO_TO_POSE . 6)
    (:VELOCITY_CONTROL . 7)
    (:REFERENCE_CONTROL . 8)
    (:TRAJECTORY_CONTROL . 9)
    (:COMMAND_FEEDTHROUGH . 10)
    (:RC_MANUAL . 11))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'AutopilotFeedback)))
    "Constants for message type 'AutopilotFeedback"
  '((:OFF . 0)
    (:START . 1)
    (:HOVER . 2)
    (:LAND . 3)
    (:EMERGENCY_LAND . 4)
    (:BREAKING . 5)
    (:GO_TO_POSE . 6)
    (:VELOCITY_CONTROL . 7)
    (:REFERENCE_CONTROL . 8)
    (:TRAJECTORY_CONTROL . 9)
    (:COMMAND_FEEDTHROUGH . 10)
    (:RC_MANUAL . 11))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AutopilotFeedback>) ostream)
  "Serializes a message object of type '<AutopilotFeedback>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'autopilot_state)) ostream)
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'control_command_delay)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'control_command_delay) (cl:floor (cl:slot-value msg 'control_command_delay)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'control_computation_time)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'control_computation_time) (cl:floor (cl:slot-value msg 'control_computation_time)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'trajectory_execution_left_duration)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'trajectory_execution_left_duration) (cl:floor (cl:slot-value msg 'trajectory_execution_left_duration)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'trajectories_left_in_queue)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'low_level_feedback) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'reference_state) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'state_estimate) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AutopilotFeedback>) istream)
  "Deserializes a message object of type '<AutopilotFeedback>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'autopilot_state)) (cl:read-byte istream))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'control_command_delay) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'control_computation_time) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'trajectory_execution_left_duration) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'trajectories_left_in_queue)) (cl:read-byte istream))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'low_level_feedback) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'reference_state) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'state_estimate) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AutopilotFeedback>)))
  "Returns string type for a message object of type '<AutopilotFeedback>"
  "quadrotor_msgs/AutopilotFeedback")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AutopilotFeedback)))
  "Returns string type for a message object of type 'AutopilotFeedback"
  "quadrotor_msgs/AutopilotFeedback")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AutopilotFeedback>)))
  "Returns md5sum for a message object of type '<AutopilotFeedback>"
  "8c8e08f7c3465bc93596097f7c8ecc39")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AutopilotFeedback)))
  "Returns md5sum for a message object of type 'AutopilotFeedback"
  "8c8e08f7c3465bc93596097f7c8ecc39")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AutopilotFeedback>)))
  "Returns full string definition for message of type '<AutopilotFeedback>"
  (cl:format cl:nil "# Autopilot state enums~%uint8 OFF=0~%uint8 START=1~%uint8 HOVER=2~%uint8 LAND=3~%uint8 EMERGENCY_LAND=4~%uint8 BREAKING=5~%uint8 GO_TO_POSE=6~%uint8 VELOCITY_CONTROL=7~%uint8 REFERENCE_CONTROL=8~%uint8 TRAJECTORY_CONTROL=9~%uint8 COMMAND_FEEDTHROUGH=10~%uint8 RC_MANUAL=11~%~%~%Header header~%~%# Autopilot state as defined above. This reflects what is implemented in~%# autopilot/include/autopilot/autopilot.h~%uint8 autopilot_state~%~%# Control command delay~%duration control_command_delay~%~%# Controller computation time [s]~%duration control_computation_time~%~%# Duration left of the trajectories in the queue~%# Only valid in TRAJECTORY_CONTROL mode~%duration trajectory_execution_left_duration~%~%# Number of trajectories that were sent to the autopilot and are stored in its~%# queue. Only valid in TRAJECTORY_CONTROL mode~%uint8 trajectories_left_in_queue~%~%# Low level feedback~%quadrotor_msgs/LowLevelFeedback low_level_feedback~%~%# Desired state used to compute the control command~%quadrotor_msgs/TrajectoryPoint reference_state~%~%# State estimate used to compute the control command~%nav_msgs/Odometry state_estimate~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: quadrotor_msgs/LowLevelFeedback~%# battery state enums~%uint8 BAT_INVALID=0~%uint8 BAT_GOOD=1~%uint8 BAT_LOW=2~%uint8 BAT_CRITICAL=3~%~%# control mode enums as defined in ControlCommand.msg~%uint8 NONE=0~%uint8 ATTITUDE=1~%uint8 BODY_RATES=2~%uint8 ANGULAR_ACCELERATION=3~%uint8 ROTOR_THRUSTS=4~%# Additionally to the control command we want to know whether an RC has taken~%# over from the low level feedback~%uint8 RC_MANUAL=10~%~%Header header~%~%# Battery information~%float32 battery_voltage~%uint8 battery_state~%~%# Control mode as defined above~%uint8 control_mode~%~%# Motor speed feedback [rpm]~%int16[] motor_speeds~%~%# Thrust mapping coefficients~%# thrust = thrust_mapping_coeffs[2] * u^2 + thrust_mapping_coeffs[1] * u +~%#     thrust_mapping_coeffs[0]~%float64[] thrust_mapping_coeffs~%~%================================================================================~%MSG: quadrotor_msgs/TrajectoryPoint~%duration time_from_start~%~%geometry_msgs/Pose pose~%~%geometry_msgs/Twist velocity~%~%geometry_msgs/Twist acceleration~%~%geometry_msgs/Twist jerk~%~%geometry_msgs/Twist snap~%~%# Heading angle with respect to world frame [rad]~%float64 heading~%~%# First derivative of the heading angle [rad/s]~%float64 heading_rate~%~%# Second derivative of the heading angle [rad/s^2]~%float64 heading_acceleration~%~%# Collective thrust [m/s^2]~%float64 thrust~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: nav_msgs/Odometry~%# This represents an estimate of a position and velocity in free space.  ~%# The pose in this message should be specified in the coordinate frame given by header.frame_id.~%# The twist in this message should be specified in the coordinate frame given by the child_frame_id~%Header header~%string child_frame_id~%geometry_msgs/PoseWithCovariance pose~%geometry_msgs/TwistWithCovariance twist~%~%================================================================================~%MSG: geometry_msgs/PoseWithCovariance~%# This represents a pose in free space with uncertainty.~%~%Pose pose~%~%# Row-major representation of the 6x6 covariance matrix~%# The orientation parameters use a fixed-axis representation.~%# In order, the parameters are:~%# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)~%float64[36] covariance~%~%================================================================================~%MSG: geometry_msgs/TwistWithCovariance~%# This expresses velocity in free space with uncertainty.~%~%Twist twist~%~%# Row-major representation of the 6x6 covariance matrix~%# The orientation parameters use a fixed-axis representation.~%# In order, the parameters are:~%# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)~%float64[36] covariance~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AutopilotFeedback)))
  "Returns full string definition for message of type 'AutopilotFeedback"
  (cl:format cl:nil "# Autopilot state enums~%uint8 OFF=0~%uint8 START=1~%uint8 HOVER=2~%uint8 LAND=3~%uint8 EMERGENCY_LAND=4~%uint8 BREAKING=5~%uint8 GO_TO_POSE=6~%uint8 VELOCITY_CONTROL=7~%uint8 REFERENCE_CONTROL=8~%uint8 TRAJECTORY_CONTROL=9~%uint8 COMMAND_FEEDTHROUGH=10~%uint8 RC_MANUAL=11~%~%~%Header header~%~%# Autopilot state as defined above. This reflects what is implemented in~%# autopilot/include/autopilot/autopilot.h~%uint8 autopilot_state~%~%# Control command delay~%duration control_command_delay~%~%# Controller computation time [s]~%duration control_computation_time~%~%# Duration left of the trajectories in the queue~%# Only valid in TRAJECTORY_CONTROL mode~%duration trajectory_execution_left_duration~%~%# Number of trajectories that were sent to the autopilot and are stored in its~%# queue. Only valid in TRAJECTORY_CONTROL mode~%uint8 trajectories_left_in_queue~%~%# Low level feedback~%quadrotor_msgs/LowLevelFeedback low_level_feedback~%~%# Desired state used to compute the control command~%quadrotor_msgs/TrajectoryPoint reference_state~%~%# State estimate used to compute the control command~%nav_msgs/Odometry state_estimate~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: quadrotor_msgs/LowLevelFeedback~%# battery state enums~%uint8 BAT_INVALID=0~%uint8 BAT_GOOD=1~%uint8 BAT_LOW=2~%uint8 BAT_CRITICAL=3~%~%# control mode enums as defined in ControlCommand.msg~%uint8 NONE=0~%uint8 ATTITUDE=1~%uint8 BODY_RATES=2~%uint8 ANGULAR_ACCELERATION=3~%uint8 ROTOR_THRUSTS=4~%# Additionally to the control command we want to know whether an RC has taken~%# over from the low level feedback~%uint8 RC_MANUAL=10~%~%Header header~%~%# Battery information~%float32 battery_voltage~%uint8 battery_state~%~%# Control mode as defined above~%uint8 control_mode~%~%# Motor speed feedback [rpm]~%int16[] motor_speeds~%~%# Thrust mapping coefficients~%# thrust = thrust_mapping_coeffs[2] * u^2 + thrust_mapping_coeffs[1] * u +~%#     thrust_mapping_coeffs[0]~%float64[] thrust_mapping_coeffs~%~%================================================================================~%MSG: quadrotor_msgs/TrajectoryPoint~%duration time_from_start~%~%geometry_msgs/Pose pose~%~%geometry_msgs/Twist velocity~%~%geometry_msgs/Twist acceleration~%~%geometry_msgs/Twist jerk~%~%geometry_msgs/Twist snap~%~%# Heading angle with respect to world frame [rad]~%float64 heading~%~%# First derivative of the heading angle [rad/s]~%float64 heading_rate~%~%# Second derivative of the heading angle [rad/s^2]~%float64 heading_acceleration~%~%# Collective thrust [m/s^2]~%float64 thrust~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: nav_msgs/Odometry~%# This represents an estimate of a position and velocity in free space.  ~%# The pose in this message should be specified in the coordinate frame given by header.frame_id.~%# The twist in this message should be specified in the coordinate frame given by the child_frame_id~%Header header~%string child_frame_id~%geometry_msgs/PoseWithCovariance pose~%geometry_msgs/TwistWithCovariance twist~%~%================================================================================~%MSG: geometry_msgs/PoseWithCovariance~%# This represents a pose in free space with uncertainty.~%~%Pose pose~%~%# Row-major representation of the 6x6 covariance matrix~%# The orientation parameters use a fixed-axis representation.~%# In order, the parameters are:~%# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)~%float64[36] covariance~%~%================================================================================~%MSG: geometry_msgs/TwistWithCovariance~%# This expresses velocity in free space with uncertainty.~%~%Twist twist~%~%# Row-major representation of the 6x6 covariance matrix~%# The orientation parameters use a fixed-axis representation.~%# In order, the parameters are:~%# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)~%float64[36] covariance~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AutopilotFeedback>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     8
     8
     8
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'low_level_feedback))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'reference_state))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'state_estimate))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AutopilotFeedback>))
  "Converts a ROS message object to a list"
  (cl:list 'AutopilotFeedback
    (cl:cons ':header (header msg))
    (cl:cons ':autopilot_state (autopilot_state msg))
    (cl:cons ':control_command_delay (control_command_delay msg))
    (cl:cons ':control_computation_time (control_computation_time msg))
    (cl:cons ':trajectory_execution_left_duration (trajectory_execution_left_duration msg))
    (cl:cons ':trajectories_left_in_queue (trajectories_left_in_queue msg))
    (cl:cons ':low_level_feedback (low_level_feedback msg))
    (cl:cons ':reference_state (reference_state msg))
    (cl:cons ':state_estimate (state_estimate msg))
))
