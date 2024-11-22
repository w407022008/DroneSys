; Auto-generated. Do not edit!


(cl:in-package drone_msgs-msg)


;//! \htmlinclude ControlCommand.msg.html

(cl:defclass <ControlCommand> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (Command_ID
    :reader Command_ID
    :initarg :Command_ID
    :type cl:integer
    :initform 0)
   (source
    :reader source
    :initarg :source
    :type cl:string
    :initform "")
   (Mode
    :reader Mode
    :initarg :Mode
    :type cl:fixnum
    :initform 0)
   (Reference_State
    :reader Reference_State
    :initarg :Reference_State
    :type drone_msgs-msg:PositionReference
    :initform (cl:make-instance 'drone_msgs-msg:PositionReference))
   (Attitude_sp
    :reader Attitude_sp
    :initarg :Attitude_sp
    :type drone_msgs-msg:AttitudeReference
    :initform (cl:make-instance 'drone_msgs-msg:AttitudeReference)))
)

(cl:defclass ControlCommand (<ControlCommand>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ControlCommand>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ControlCommand)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name drone_msgs-msg:<ControlCommand> is deprecated: use drone_msgs-msg:ControlCommand instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <ControlCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:header-val is deprecated.  Use drone_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'Command_ID-val :lambda-list '(m))
(cl:defmethod Command_ID-val ((m <ControlCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:Command_ID-val is deprecated.  Use drone_msgs-msg:Command_ID instead.")
  (Command_ID m))

(cl:ensure-generic-function 'source-val :lambda-list '(m))
(cl:defmethod source-val ((m <ControlCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:source-val is deprecated.  Use drone_msgs-msg:source instead.")
  (source m))

(cl:ensure-generic-function 'Mode-val :lambda-list '(m))
(cl:defmethod Mode-val ((m <ControlCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:Mode-val is deprecated.  Use drone_msgs-msg:Mode instead.")
  (Mode m))

(cl:ensure-generic-function 'Reference_State-val :lambda-list '(m))
(cl:defmethod Reference_State-val ((m <ControlCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:Reference_State-val is deprecated.  Use drone_msgs-msg:Reference_State instead.")
  (Reference_State m))

(cl:ensure-generic-function 'Attitude_sp-val :lambda-list '(m))
(cl:defmethod Attitude_sp-val ((m <ControlCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:Attitude_sp-val is deprecated.  Use drone_msgs-msg:Attitude_sp instead.")
  (Attitude_sp m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<ControlCommand>)))
    "Constants for message type '<ControlCommand>"
  '((:IDLE . 0)
    (:TAKEOFF . 1)
    (:HOLD . 2)
    (:LAND . 3)
    (:MOVE . 4)
    (:DISARM . 5)
    (:ATTITUDE . 6)
    (:ATTITUDERATE . 7)
    (:RATE . 8))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'ControlCommand)))
    "Constants for message type 'ControlCommand"
  '((:IDLE . 0)
    (:TAKEOFF . 1)
    (:HOLD . 2)
    (:LAND . 3)
    (:MOVE . 4)
    (:DISARM . 5)
    (:ATTITUDE . 6)
    (:ATTITUDERATE . 7)
    (:RATE . 8))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ControlCommand>) ostream)
  "Serializes a message object of type '<ControlCommand>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Command_ID)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'Command_ID)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'Command_ID)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'Command_ID)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'source))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'source))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Mode)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'Reference_State) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'Attitude_sp) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ControlCommand>) istream)
  "Deserializes a message object of type '<ControlCommand>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Command_ID)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'Command_ID)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'Command_ID)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'Command_ID)) (cl:read-byte istream))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'source) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'source) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Mode)) (cl:read-byte istream))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'Reference_State) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'Attitude_sp) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ControlCommand>)))
  "Returns string type for a message object of type '<ControlCommand>"
  "drone_msgs/ControlCommand")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ControlCommand)))
  "Returns string type for a message object of type 'ControlCommand"
  "drone_msgs/ControlCommand")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ControlCommand>)))
  "Returns md5sum for a message object of type '<ControlCommand>"
  "969640b304f3a446799efdd5c334e9b7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ControlCommand)))
  "Returns md5sum for a message object of type 'ControlCommand"
  "969640b304f3a446799efdd5c334e9b7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ControlCommand>)))
  "Returns full string definition for message of type '<ControlCommand>"
  (cl:format cl:nil "std_msgs/Header header~%~%## ID should increased self~%uint32 Command_ID~%~%string source~%~%uint8 Mode~%# enum~%uint8 Idle=0~%uint8 Takeoff=1~%uint8 Hold=2~%uint8 Land=3~%uint8 Move=4~%uint8 Disarm=5~%uint8 Attitude=6~%uint8 AttitudeRate=7~%uint8 Rate=8~%~%## Setpoint Reference~%PositionReference Reference_State~%AttitudeReference Attitude_sp~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: drone_msgs/PositionReference~%std_msgs/Header header~%~%## Setpoint position reference for PX4 Control~%~%## Setpoint Mode~%uint8 Move_mode~%~%uint8 XYZ_POS      = 0  ##0b00~%uint8 XY_POS_Z_VEL = 1  ##0b01~%uint8 XY_VEL_Z_POS = 2  ##0b10~%uint8 XYZ_VEL = 3       ##0b11~%uint8 XYZ_ACC = 4~%uint8 XYZ_POS_VEL   = 5  ~%uint8 TRAJECTORY   = 6~%~%## Reference Frame~%uint8 Move_frame~%~%uint8 ENU_FRAME  = 0~%uint8 BODY_FRAME = 1~%~%~%~%## Tracking life~%float32 time_from_start          ## [s]~%~%float32[3] position_ref          ## [m]~%float32[3] velocity_ref          ## [m/s]~%float32[3] acceleration_ref      ## [m/s^2]~%~%bool Yaw_Rate_Mode                      ## True 代表控制偏航角速率~%float32 yaw_ref                  ## [rad]~%float32 yaw_rate_ref             ## [rad/s] ~%~%Bspline bspline~%================================================================================~%MSG: drone_msgs/Bspline~%int32 order                 ## ~%int64 traj_id               ## id of trajecotry~%float64[] knots             ## knots list~%geometry_msgs/Point[] pts   ## control points list~%time start_time             ## time stamp~%~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: drone_msgs/AttitudeReference~%std_msgs/Header header~%~%## Setpoint Attitude + T~%float32[3] thrust_sp                   ## Single Rotor Thrust setpoint~%float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint~%float32[3] desired_attitude            ## [rad] Eurler angle setpoint~%geometry_msgs/Quaternion desired_att_q ## quat setpoint~%geometry_msgs/Vector3 body_rate  ## [rad/s]~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ControlCommand)))
  "Returns full string definition for message of type 'ControlCommand"
  (cl:format cl:nil "std_msgs/Header header~%~%## ID should increased self~%uint32 Command_ID~%~%string source~%~%uint8 Mode~%# enum~%uint8 Idle=0~%uint8 Takeoff=1~%uint8 Hold=2~%uint8 Land=3~%uint8 Move=4~%uint8 Disarm=5~%uint8 Attitude=6~%uint8 AttitudeRate=7~%uint8 Rate=8~%~%## Setpoint Reference~%PositionReference Reference_State~%AttitudeReference Attitude_sp~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: drone_msgs/PositionReference~%std_msgs/Header header~%~%## Setpoint position reference for PX4 Control~%~%## Setpoint Mode~%uint8 Move_mode~%~%uint8 XYZ_POS      = 0  ##0b00~%uint8 XY_POS_Z_VEL = 1  ##0b01~%uint8 XY_VEL_Z_POS = 2  ##0b10~%uint8 XYZ_VEL = 3       ##0b11~%uint8 XYZ_ACC = 4~%uint8 XYZ_POS_VEL   = 5  ~%uint8 TRAJECTORY   = 6~%~%## Reference Frame~%uint8 Move_frame~%~%uint8 ENU_FRAME  = 0~%uint8 BODY_FRAME = 1~%~%~%~%## Tracking life~%float32 time_from_start          ## [s]~%~%float32[3] position_ref          ## [m]~%float32[3] velocity_ref          ## [m/s]~%float32[3] acceleration_ref      ## [m/s^2]~%~%bool Yaw_Rate_Mode                      ## True 代表控制偏航角速率~%float32 yaw_ref                  ## [rad]~%float32 yaw_rate_ref             ## [rad/s] ~%~%Bspline bspline~%================================================================================~%MSG: drone_msgs/Bspline~%int32 order                 ## ~%int64 traj_id               ## id of trajecotry~%float64[] knots             ## knots list~%geometry_msgs/Point[] pts   ## control points list~%time start_time             ## time stamp~%~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: drone_msgs/AttitudeReference~%std_msgs/Header header~%~%## Setpoint Attitude + T~%float32[3] thrust_sp                   ## Single Rotor Thrust setpoint~%float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint~%float32[3] desired_attitude            ## [rad] Eurler angle setpoint~%geometry_msgs/Quaternion desired_att_q ## quat setpoint~%geometry_msgs/Vector3 body_rate  ## [rad/s]~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ControlCommand>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4 (cl:length (cl:slot-value msg 'source))
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'Reference_State))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'Attitude_sp))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ControlCommand>))
  "Converts a ROS message object to a list"
  (cl:list 'ControlCommand
    (cl:cons ':header (header msg))
    (cl:cons ':Command_ID (Command_ID msg))
    (cl:cons ':source (source msg))
    (cl:cons ':Mode (Mode msg))
    (cl:cons ':Reference_State (Reference_State msg))
    (cl:cons ':Attitude_sp (Attitude_sp msg))
))
