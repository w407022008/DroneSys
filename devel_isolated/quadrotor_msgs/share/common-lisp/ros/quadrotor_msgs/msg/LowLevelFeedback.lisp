; Auto-generated. Do not edit!


(cl:in-package quadrotor_msgs-msg)


;//! \htmlinclude LowLevelFeedback.msg.html

(cl:defclass <LowLevelFeedback> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (battery_voltage
    :reader battery_voltage
    :initarg :battery_voltage
    :type cl:float
    :initform 0.0)
   (battery_state
    :reader battery_state
    :initarg :battery_state
    :type cl:fixnum
    :initform 0)
   (control_mode
    :reader control_mode
    :initarg :control_mode
    :type cl:fixnum
    :initform 0)
   (motor_speeds
    :reader motor_speeds
    :initarg :motor_speeds
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0))
   (thrust_mapping_coeffs
    :reader thrust_mapping_coeffs
    :initarg :thrust_mapping_coeffs
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass LowLevelFeedback (<LowLevelFeedback>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <LowLevelFeedback>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'LowLevelFeedback)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name quadrotor_msgs-msg:<LowLevelFeedback> is deprecated: use quadrotor_msgs-msg:LowLevelFeedback instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <LowLevelFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:header-val is deprecated.  Use quadrotor_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'battery_voltage-val :lambda-list '(m))
(cl:defmethod battery_voltage-val ((m <LowLevelFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:battery_voltage-val is deprecated.  Use quadrotor_msgs-msg:battery_voltage instead.")
  (battery_voltage m))

(cl:ensure-generic-function 'battery_state-val :lambda-list '(m))
(cl:defmethod battery_state-val ((m <LowLevelFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:battery_state-val is deprecated.  Use quadrotor_msgs-msg:battery_state instead.")
  (battery_state m))

(cl:ensure-generic-function 'control_mode-val :lambda-list '(m))
(cl:defmethod control_mode-val ((m <LowLevelFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:control_mode-val is deprecated.  Use quadrotor_msgs-msg:control_mode instead.")
  (control_mode m))

(cl:ensure-generic-function 'motor_speeds-val :lambda-list '(m))
(cl:defmethod motor_speeds-val ((m <LowLevelFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:motor_speeds-val is deprecated.  Use quadrotor_msgs-msg:motor_speeds instead.")
  (motor_speeds m))

(cl:ensure-generic-function 'thrust_mapping_coeffs-val :lambda-list '(m))
(cl:defmethod thrust_mapping_coeffs-val ((m <LowLevelFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader quadrotor_msgs-msg:thrust_mapping_coeffs-val is deprecated.  Use quadrotor_msgs-msg:thrust_mapping_coeffs instead.")
  (thrust_mapping_coeffs m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<LowLevelFeedback>)))
    "Constants for message type '<LowLevelFeedback>"
  '((:BAT_INVALID . 0)
    (:BAT_GOOD . 1)
    (:BAT_LOW . 2)
    (:BAT_CRITICAL . 3)
    (:NONE . 0)
    (:ATTITUDE . 1)
    (:BODY_RATES . 2)
    (:ANGULAR_ACCELERATION . 3)
    (:ROTOR_THRUSTS . 4)
    (:RC_MANUAL . 10))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'LowLevelFeedback)))
    "Constants for message type 'LowLevelFeedback"
  '((:BAT_INVALID . 0)
    (:BAT_GOOD . 1)
    (:BAT_LOW . 2)
    (:BAT_CRITICAL . 3)
    (:NONE . 0)
    (:ATTITUDE . 1)
    (:BODY_RATES . 2)
    (:ANGULAR_ACCELERATION . 3)
    (:ROTOR_THRUSTS . 4)
    (:RC_MANUAL . 10))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <LowLevelFeedback>) ostream)
  "Serializes a message object of type '<LowLevelFeedback>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'battery_voltage))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'battery_state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'control_mode)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'motor_speeds))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'motor_speeds))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'thrust_mapping_coeffs))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'thrust_mapping_coeffs))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <LowLevelFeedback>) istream)
  "Deserializes a message object of type '<LowLevelFeedback>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'battery_voltage) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'battery_state)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'control_mode)) (cl:read-byte istream))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'motor_speeds) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'motor_speeds)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536)))))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'thrust_mapping_coeffs) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'thrust_mapping_coeffs)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<LowLevelFeedback>)))
  "Returns string type for a message object of type '<LowLevelFeedback>"
  "quadrotor_msgs/LowLevelFeedback")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'LowLevelFeedback)))
  "Returns string type for a message object of type 'LowLevelFeedback"
  "quadrotor_msgs/LowLevelFeedback")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<LowLevelFeedback>)))
  "Returns md5sum for a message object of type '<LowLevelFeedback>"
  "e3cfad3ba98dfdc505bcf1fe91833d87")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'LowLevelFeedback)))
  "Returns md5sum for a message object of type 'LowLevelFeedback"
  "e3cfad3ba98dfdc505bcf1fe91833d87")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<LowLevelFeedback>)))
  "Returns full string definition for message of type '<LowLevelFeedback>"
  (cl:format cl:nil "# battery state enums~%uint8 BAT_INVALID=0~%uint8 BAT_GOOD=1~%uint8 BAT_LOW=2~%uint8 BAT_CRITICAL=3~%~%# control mode enums as defined in ControlCommand.msg~%uint8 NONE=0~%uint8 ATTITUDE=1~%uint8 BODY_RATES=2~%uint8 ANGULAR_ACCELERATION=3~%uint8 ROTOR_THRUSTS=4~%# Additionally to the control command we want to know whether an RC has taken~%# over from the low level feedback~%uint8 RC_MANUAL=10~%~%Header header~%~%# Battery information~%float32 battery_voltage~%uint8 battery_state~%~%# Control mode as defined above~%uint8 control_mode~%~%# Motor speed feedback [rpm]~%int16[] motor_speeds~%~%# Thrust mapping coefficients~%# thrust = thrust_mapping_coeffs[2] * u^2 + thrust_mapping_coeffs[1] * u +~%#     thrust_mapping_coeffs[0]~%float64[] thrust_mapping_coeffs~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'LowLevelFeedback)))
  "Returns full string definition for message of type 'LowLevelFeedback"
  (cl:format cl:nil "# battery state enums~%uint8 BAT_INVALID=0~%uint8 BAT_GOOD=1~%uint8 BAT_LOW=2~%uint8 BAT_CRITICAL=3~%~%# control mode enums as defined in ControlCommand.msg~%uint8 NONE=0~%uint8 ATTITUDE=1~%uint8 BODY_RATES=2~%uint8 ANGULAR_ACCELERATION=3~%uint8 ROTOR_THRUSTS=4~%# Additionally to the control command we want to know whether an RC has taken~%# over from the low level feedback~%uint8 RC_MANUAL=10~%~%Header header~%~%# Battery information~%float32 battery_voltage~%uint8 battery_state~%~%# Control mode as defined above~%uint8 control_mode~%~%# Motor speed feedback [rpm]~%int16[] motor_speeds~%~%# Thrust mapping coefficients~%# thrust = thrust_mapping_coeffs[2] * u^2 + thrust_mapping_coeffs[1] * u +~%#     thrust_mapping_coeffs[0]~%float64[] thrust_mapping_coeffs~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <LowLevelFeedback>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     1
     1
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'motor_speeds) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'thrust_mapping_coeffs) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <LowLevelFeedback>))
  "Converts a ROS message object to a list"
  (cl:list 'LowLevelFeedback
    (cl:cons ':header (header msg))
    (cl:cons ':battery_voltage (battery_voltage msg))
    (cl:cons ':battery_state (battery_state msg))
    (cl:cons ':control_mode (control_mode msg))
    (cl:cons ':motor_speeds (motor_speeds msg))
    (cl:cons ':thrust_mapping_coeffs (thrust_mapping_coeffs msg))
))
