; Auto-generated. Do not edit!


(cl:in-package drone_msgs-msg)


;//! \htmlinclude DroneTarget.msg.html

(cl:defclass <DroneTarget> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (position_target
    :reader position_target
    :initarg :position_target
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (velocity_target
    :reader velocity_target
    :initarg :velocity_target
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (acceleration_target
    :reader acceleration_target
    :initarg :acceleration_target
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (q_target
    :reader q_target
    :initarg :q_target
    :type geometry_msgs-msg:Quaternion
    :initform (cl:make-instance 'geometry_msgs-msg:Quaternion))
   (euler_target
    :reader euler_target
    :initarg :euler_target
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (rate_target
    :reader rate_target
    :initarg :rate_target
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (thrust_target
    :reader thrust_target
    :initarg :thrust_target
    :type cl:float
    :initform 0.0)
   (actuator_target
    :reader actuator_target
    :initarg :actuator_target
    :type mavros_msgs-msg:ActuatorControl
    :initform (cl:make-instance 'mavros_msgs-msg:ActuatorControl)))
)

(cl:defclass DroneTarget (<DroneTarget>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DroneTarget>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DroneTarget)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name drone_msgs-msg:<DroneTarget> is deprecated: use drone_msgs-msg:DroneTarget instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:header-val is deprecated.  Use drone_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'position_target-val :lambda-list '(m))
(cl:defmethod position_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:position_target-val is deprecated.  Use drone_msgs-msg:position_target instead.")
  (position_target m))

(cl:ensure-generic-function 'velocity_target-val :lambda-list '(m))
(cl:defmethod velocity_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:velocity_target-val is deprecated.  Use drone_msgs-msg:velocity_target instead.")
  (velocity_target m))

(cl:ensure-generic-function 'acceleration_target-val :lambda-list '(m))
(cl:defmethod acceleration_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:acceleration_target-val is deprecated.  Use drone_msgs-msg:acceleration_target instead.")
  (acceleration_target m))

(cl:ensure-generic-function 'q_target-val :lambda-list '(m))
(cl:defmethod q_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:q_target-val is deprecated.  Use drone_msgs-msg:q_target instead.")
  (q_target m))

(cl:ensure-generic-function 'euler_target-val :lambda-list '(m))
(cl:defmethod euler_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:euler_target-val is deprecated.  Use drone_msgs-msg:euler_target instead.")
  (euler_target m))

(cl:ensure-generic-function 'rate_target-val :lambda-list '(m))
(cl:defmethod rate_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:rate_target-val is deprecated.  Use drone_msgs-msg:rate_target instead.")
  (rate_target m))

(cl:ensure-generic-function 'thrust_target-val :lambda-list '(m))
(cl:defmethod thrust_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:thrust_target-val is deprecated.  Use drone_msgs-msg:thrust_target instead.")
  (thrust_target m))

(cl:ensure-generic-function 'actuator_target-val :lambda-list '(m))
(cl:defmethod actuator_target-val ((m <DroneTarget>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:actuator_target-val is deprecated.  Use drone_msgs-msg:actuator_target instead.")
  (actuator_target m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DroneTarget>) ostream)
  "Serializes a message object of type '<DroneTarget>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'position_target))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'velocity_target))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'acceleration_target))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'q_target) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'euler_target))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'rate_target))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'thrust_target))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'actuator_target) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DroneTarget>) istream)
  "Deserializes a message object of type '<DroneTarget>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:setf (cl:slot-value msg 'position_target) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'position_target)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'velocity_target) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'velocity_target)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'acceleration_target) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'acceleration_target)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'q_target) istream)
  (cl:setf (cl:slot-value msg 'euler_target) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'euler_target)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'rate_target) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'rate_target)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'thrust_target) (roslisp-utils:decode-single-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'actuator_target) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DroneTarget>)))
  "Returns string type for a message object of type '<DroneTarget>"
  "drone_msgs/DroneTarget")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DroneTarget)))
  "Returns string type for a message object of type 'DroneTarget"
  "drone_msgs/DroneTarget")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DroneTarget>)))
  "Returns md5sum for a message object of type '<DroneTarget>"
  "b13c4477f8a36524e314a3b537e64de4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DroneTarget)))
  "Returns md5sum for a message object of type 'DroneTarget"
  "b13c4477f8a36524e314a3b537e64de4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DroneTarget>)))
  "Returns full string definition for message of type '<DroneTarget>"
  (cl:format cl:nil "std_msgs/Header header~%~%float32[3] position_target          ## [m]~%float32[3] velocity_target          ## [m/s]~%float32[3] acceleration_target      ## [m/s/s]~%geometry_msgs/Quaternion q_target   ## quat~%float32[3] euler_target             ## [rad]~%float32[3] rate_target              ## [rad/s]~%float32 thrust_target~%mavros_msgs/ActuatorControl actuator_target~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: mavros_msgs/ActuatorControl~%# raw servo values for direct actuator controls~%#~%# about groups, mixing and channels:~%# https://pixhawk.org/dev/mixing~%~%# constant for mixer group~%uint8 PX4_MIX_FLIGHT_CONTROL = 0~%uint8 PX4_MIX_FLIGHT_CONTROL_VTOL_ALT = 1~%uint8 PX4_MIX_PAYLOAD = 2~%uint8 PX4_MIX_MANUAL_PASSTHROUGH = 3~%#uint8 PX4_MIX_FC_MC_VIRT = 4~%#uint8 PX4_MIX_FC_FW_VIRT = 5~%~%std_msgs/Header header~%uint8 group_mix~%float32[8] controls~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DroneTarget)))
  "Returns full string definition for message of type 'DroneTarget"
  (cl:format cl:nil "std_msgs/Header header~%~%float32[3] position_target          ## [m]~%float32[3] velocity_target          ## [m/s]~%float32[3] acceleration_target      ## [m/s/s]~%geometry_msgs/Quaternion q_target   ## quat~%float32[3] euler_target             ## [rad]~%float32[3] rate_target              ## [rad/s]~%float32 thrust_target~%mavros_msgs/ActuatorControl actuator_target~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: mavros_msgs/ActuatorControl~%# raw servo values for direct actuator controls~%#~%# about groups, mixing and channels:~%# https://pixhawk.org/dev/mixing~%~%# constant for mixer group~%uint8 PX4_MIX_FLIGHT_CONTROL = 0~%uint8 PX4_MIX_FLIGHT_CONTROL_VTOL_ALT = 1~%uint8 PX4_MIX_PAYLOAD = 2~%uint8 PX4_MIX_MANUAL_PASSTHROUGH = 3~%#uint8 PX4_MIX_FC_MC_VIRT = 4~%#uint8 PX4_MIX_FC_FW_VIRT = 5~%~%std_msgs/Header header~%uint8 group_mix~%float32[8] controls~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DroneTarget>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'position_target) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'velocity_target) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'acceleration_target) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'q_target))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'euler_target) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'rate_target) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'actuator_target))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DroneTarget>))
  "Converts a ROS message object to a list"
  (cl:list 'DroneTarget
    (cl:cons ':header (header msg))
    (cl:cons ':position_target (position_target msg))
    (cl:cons ':velocity_target (velocity_target msg))
    (cl:cons ':acceleration_target (acceleration_target msg))
    (cl:cons ':q_target (q_target msg))
    (cl:cons ':euler_target (euler_target msg))
    (cl:cons ':rate_target (rate_target msg))
    (cl:cons ':thrust_target (thrust_target msg))
    (cl:cons ':actuator_target (actuator_target msg))
))
