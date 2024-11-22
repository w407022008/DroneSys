; Auto-generated. Do not edit!


(cl:in-package drone_msgs-msg)


;//! \htmlinclude RCInput.msg.html

(cl:defclass <RCInput> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (rc_x
    :reader rc_x
    :initarg :rc_x
    :type cl:float
    :initform 0.0)
   (rc_y
    :reader rc_y
    :initarg :rc_y
    :type cl:float
    :initform 0.0)
   (rc_z
    :reader rc_z
    :initarg :rc_z
    :type cl:float
    :initform 0.0)
   (rc_r
    :reader rc_r
    :initarg :rc_r
    :type cl:float
    :initform 0.0)
   (buttons
    :reader buttons
    :initarg :buttons
    :type cl:integer
    :initform 0)
   (goal_enable
    :reader goal_enable
    :initarg :goal_enable
    :type cl:fixnum
    :initform 0)
   (data_source
    :reader data_source
    :initarg :data_source
    :type cl:integer
    :initform 0))
)

(cl:defclass RCInput (<RCInput>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RCInput>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RCInput)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name drone_msgs-msg:<RCInput> is deprecated: use drone_msgs-msg:RCInput instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:header-val is deprecated.  Use drone_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'rc_x-val :lambda-list '(m))
(cl:defmethod rc_x-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:rc_x-val is deprecated.  Use drone_msgs-msg:rc_x instead.")
  (rc_x m))

(cl:ensure-generic-function 'rc_y-val :lambda-list '(m))
(cl:defmethod rc_y-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:rc_y-val is deprecated.  Use drone_msgs-msg:rc_y instead.")
  (rc_y m))

(cl:ensure-generic-function 'rc_z-val :lambda-list '(m))
(cl:defmethod rc_z-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:rc_z-val is deprecated.  Use drone_msgs-msg:rc_z instead.")
  (rc_z m))

(cl:ensure-generic-function 'rc_r-val :lambda-list '(m))
(cl:defmethod rc_r-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:rc_r-val is deprecated.  Use drone_msgs-msg:rc_r instead.")
  (rc_r m))

(cl:ensure-generic-function 'buttons-val :lambda-list '(m))
(cl:defmethod buttons-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:buttons-val is deprecated.  Use drone_msgs-msg:buttons instead.")
  (buttons m))

(cl:ensure-generic-function 'goal_enable-val :lambda-list '(m))
(cl:defmethod goal_enable-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:goal_enable-val is deprecated.  Use drone_msgs-msg:goal_enable instead.")
  (goal_enable m))

(cl:ensure-generic-function 'data_source-val :lambda-list '(m))
(cl:defmethod data_source-val ((m <RCInput>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:data_source-val is deprecated.  Use drone_msgs-msg:data_source instead.")
  (data_source m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<RCInput>)))
    "Constants for message type '<RCInput>"
  '((:DISABLE . 0)
    (:MAVROS_MANUAL_CONTROL . 1)
    (:DRIVER_JOYSTICK . 2))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'RCInput)))
    "Constants for message type 'RCInput"
  '((:DISABLE . 0)
    (:MAVROS_MANUAL_CONTROL . 1)
    (:DRIVER_JOYSTICK . 2))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RCInput>) ostream)
  "Serializes a message object of type '<RCInput>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rc_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rc_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rc_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'rc_r))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'buttons)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'buttons)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'buttons)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'buttons)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'goal_enable)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'data_source)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RCInput>) istream)
  "Deserializes a message object of type '<RCInput>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rc_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rc_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rc_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'rc_r) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'buttons)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'buttons)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'buttons)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'buttons)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'goal_enable)) (cl:read-byte istream))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'data_source) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RCInput>)))
  "Returns string type for a message object of type '<RCInput>"
  "drone_msgs/RCInput")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RCInput)))
  "Returns string type for a message object of type 'RCInput"
  "drone_msgs/RCInput")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RCInput>)))
  "Returns md5sum for a message object of type '<RCInput>"
  "13d10a65cefb07444f918f9ce0babb28")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RCInput)))
  "Returns md5sum for a message object of type 'RCInput"
  "13d10a65cefb07444f918f9ce0babb28")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RCInput>)))
  "Returns full string definition for message of type '<RCInput>"
  (cl:format cl:nil "## Radio Control Input~%std_msgs/Header header~%~%# Data Source~%uint8 DISABLE = 0~%uint8 MAVROS_MANUAL_CONTROL = 1~%uint8 DRIVER_JOYSTICK = 2~%~%float32 rc_x             # stick position in x direction -1..1~%                         # in general corresponds to forward/back motion or pitch of vehicle,~%                         # in general a positive value means forward or negative pitch and~%                         # a negative value means backward or positive pitch~%float32 rc_y             # stick position in y direction -1..1~%                         # in general corresponds to right/left motion or roll of vehicle,~%                         # in general a positive value means right or positive roll and~%                         # a negative value means left or negative roll~%float32 rc_z             # throttle stick position 0..1~%                         # in general corresponds to up/down motion or thrust of vehicle,~%                         # in general the value corresponds to the demanded throttle by the user,~%                         # if the input is used for setting the setpoint of a vertical position~%                         # controller any value > 0.5 means up and any value < 0.5 means down~%float32 rc_r             # yaw stick/twist position, -1..1~%                         # in general corresponds to the righthand rotation around the vertical~%                         # (downwards) axis of the vehicle~%uint32 buttons           # Binary~%~%uint8 goal_enable        # push down(1):enable; release(0):disable~%~%int32 data_source # determin the data source~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RCInput)))
  "Returns full string definition for message of type 'RCInput"
  (cl:format cl:nil "## Radio Control Input~%std_msgs/Header header~%~%# Data Source~%uint8 DISABLE = 0~%uint8 MAVROS_MANUAL_CONTROL = 1~%uint8 DRIVER_JOYSTICK = 2~%~%float32 rc_x             # stick position in x direction -1..1~%                         # in general corresponds to forward/back motion or pitch of vehicle,~%                         # in general a positive value means forward or negative pitch and~%                         # a negative value means backward or positive pitch~%float32 rc_y             # stick position in y direction -1..1~%                         # in general corresponds to right/left motion or roll of vehicle,~%                         # in general a positive value means right or positive roll and~%                         # a negative value means left or negative roll~%float32 rc_z             # throttle stick position 0..1~%                         # in general corresponds to up/down motion or thrust of vehicle,~%                         # in general the value corresponds to the demanded throttle by the user,~%                         # if the input is used for setting the setpoint of a vertical position~%                         # controller any value > 0.5 means up and any value < 0.5 means down~%float32 rc_r             # yaw stick/twist position, -1..1~%                         # in general corresponds to the righthand rotation around the vertical~%                         # (downwards) axis of the vehicle~%uint32 buttons           # Binary~%~%uint8 goal_enable        # push down(1):enable; release(0):disable~%~%int32 data_source # determin the data source~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RCInput>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
     4
     4
     4
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RCInput>))
  "Converts a ROS message object to a list"
  (cl:list 'RCInput
    (cl:cons ':header (header msg))
    (cl:cons ':rc_x (rc_x msg))
    (cl:cons ':rc_y (rc_y msg))
    (cl:cons ':rc_z (rc_z msg))
    (cl:cons ':rc_r (rc_r msg))
    (cl:cons ':buttons (buttons msg))
    (cl:cons ':goal_enable (goal_enable msg))
    (cl:cons ':data_source (data_source msg))
))
