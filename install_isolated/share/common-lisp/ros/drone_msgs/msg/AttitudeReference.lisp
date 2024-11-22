; Auto-generated. Do not edit!


(cl:in-package drone_msgs-msg)


;//! \htmlinclude AttitudeReference.msg.html

(cl:defclass <AttitudeReference> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (thrust_sp
    :reader thrust_sp
    :initarg :thrust_sp
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (collective_accel
    :reader collective_accel
    :initarg :collective_accel
    :type cl:float
    :initform 0.0)
   (desired_attitude
    :reader desired_attitude
    :initarg :desired_attitude
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (desired_att_q
    :reader desired_att_q
    :initarg :desired_att_q
    :type geometry_msgs-msg:Quaternion
    :initform (cl:make-instance 'geometry_msgs-msg:Quaternion))
   (body_rate
    :reader body_rate
    :initarg :body_rate
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass AttitudeReference (<AttitudeReference>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AttitudeReference>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AttitudeReference)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name drone_msgs-msg:<AttitudeReference> is deprecated: use drone_msgs-msg:AttitudeReference instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <AttitudeReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:header-val is deprecated.  Use drone_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'thrust_sp-val :lambda-list '(m))
(cl:defmethod thrust_sp-val ((m <AttitudeReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:thrust_sp-val is deprecated.  Use drone_msgs-msg:thrust_sp instead.")
  (thrust_sp m))

(cl:ensure-generic-function 'collective_accel-val :lambda-list '(m))
(cl:defmethod collective_accel-val ((m <AttitudeReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:collective_accel-val is deprecated.  Use drone_msgs-msg:collective_accel instead.")
  (collective_accel m))

(cl:ensure-generic-function 'desired_attitude-val :lambda-list '(m))
(cl:defmethod desired_attitude-val ((m <AttitudeReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:desired_attitude-val is deprecated.  Use drone_msgs-msg:desired_attitude instead.")
  (desired_attitude m))

(cl:ensure-generic-function 'desired_att_q-val :lambda-list '(m))
(cl:defmethod desired_att_q-val ((m <AttitudeReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:desired_att_q-val is deprecated.  Use drone_msgs-msg:desired_att_q instead.")
  (desired_att_q m))

(cl:ensure-generic-function 'body_rate-val :lambda-list '(m))
(cl:defmethod body_rate-val ((m <AttitudeReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:body_rate-val is deprecated.  Use drone_msgs-msg:body_rate instead.")
  (body_rate m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AttitudeReference>) ostream)
  "Serializes a message object of type '<AttitudeReference>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'thrust_sp))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'collective_accel))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'desired_attitude))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'desired_att_q) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'body_rate) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AttitudeReference>) istream)
  "Deserializes a message object of type '<AttitudeReference>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:setf (cl:slot-value msg 'thrust_sp) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'thrust_sp)))
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
    (cl:setf (cl:slot-value msg 'collective_accel) (roslisp-utils:decode-single-float-bits bits)))
  (cl:setf (cl:slot-value msg 'desired_attitude) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'desired_attitude)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'desired_att_q) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'body_rate) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AttitudeReference>)))
  "Returns string type for a message object of type '<AttitudeReference>"
  "drone_msgs/AttitudeReference")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AttitudeReference)))
  "Returns string type for a message object of type 'AttitudeReference"
  "drone_msgs/AttitudeReference")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AttitudeReference>)))
  "Returns md5sum for a message object of type '<AttitudeReference>"
  "ad65c8727b64e262c550df8ad8b37905")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AttitudeReference)))
  "Returns md5sum for a message object of type 'AttitudeReference"
  "ad65c8727b64e262c550df8ad8b37905")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AttitudeReference>)))
  "Returns full string definition for message of type '<AttitudeReference>"
  (cl:format cl:nil "std_msgs/Header header~%~%## Setpoint Attitude + T~%float32[3] thrust_sp                   ## Single Rotor Thrust setpoint~%float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint~%float32[3] desired_attitude            ## [rad] Eurler angle setpoint~%geometry_msgs/Quaternion desired_att_q ## quat setpoint~%geometry_msgs/Vector3 body_rate  ## [rad/s]~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AttitudeReference)))
  "Returns full string definition for message of type 'AttitudeReference"
  (cl:format cl:nil "std_msgs/Header header~%~%## Setpoint Attitude + T~%float32[3] thrust_sp                   ## Single Rotor Thrust setpoint~%float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint~%float32[3] desired_attitude            ## [rad] Eurler angle setpoint~%geometry_msgs/Quaternion desired_att_q ## quat setpoint~%geometry_msgs/Vector3 body_rate  ## [rad/s]~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AttitudeReference>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'thrust_sp) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'desired_attitude) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'desired_att_q))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'body_rate))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AttitudeReference>))
  "Converts a ROS message object to a list"
  (cl:list 'AttitudeReference
    (cl:cons ':header (header msg))
    (cl:cons ':thrust_sp (thrust_sp msg))
    (cl:cons ':collective_accel (collective_accel msg))
    (cl:cons ':desired_attitude (desired_attitude msg))
    (cl:cons ':desired_att_q (desired_att_q msg))
    (cl:cons ':body_rate (body_rate msg))
))
