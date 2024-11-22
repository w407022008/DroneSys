; Auto-generated. Do not edit!


(cl:in-package mavros_msgs-msg)


;//! \htmlinclude SysStatus.msg.html

(cl:defclass <SysStatus> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (sensors_present
    :reader sensors_present
    :initarg :sensors_present
    :type cl:integer
    :initform 0)
   (sensors_enabled
    :reader sensors_enabled
    :initarg :sensors_enabled
    :type cl:integer
    :initform 0)
   (sensors_health
    :reader sensors_health
    :initarg :sensors_health
    :type cl:integer
    :initform 0)
   (load
    :reader load
    :initarg :load
    :type cl:fixnum
    :initform 0)
   (voltage_battery
    :reader voltage_battery
    :initarg :voltage_battery
    :type cl:fixnum
    :initform 0)
   (current_battery
    :reader current_battery
    :initarg :current_battery
    :type cl:fixnum
    :initform 0)
   (battery_remaining
    :reader battery_remaining
    :initarg :battery_remaining
    :type cl:fixnum
    :initform 0)
   (drop_rate_comm
    :reader drop_rate_comm
    :initarg :drop_rate_comm
    :type cl:fixnum
    :initform 0)
   (errors_comm
    :reader errors_comm
    :initarg :errors_comm
    :type cl:fixnum
    :initform 0)
   (errors_count1
    :reader errors_count1
    :initarg :errors_count1
    :type cl:fixnum
    :initform 0)
   (errors_count2
    :reader errors_count2
    :initarg :errors_count2
    :type cl:fixnum
    :initform 0)
   (errors_count3
    :reader errors_count3
    :initarg :errors_count3
    :type cl:fixnum
    :initform 0)
   (errors_count4
    :reader errors_count4
    :initarg :errors_count4
    :type cl:fixnum
    :initform 0))
)

(cl:defclass SysStatus (<SysStatus>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SysStatus>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SysStatus)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name mavros_msgs-msg:<SysStatus> is deprecated: use mavros_msgs-msg:SysStatus instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:header-val is deprecated.  Use mavros_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'sensors_present-val :lambda-list '(m))
(cl:defmethod sensors_present-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:sensors_present-val is deprecated.  Use mavros_msgs-msg:sensors_present instead.")
  (sensors_present m))

(cl:ensure-generic-function 'sensors_enabled-val :lambda-list '(m))
(cl:defmethod sensors_enabled-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:sensors_enabled-val is deprecated.  Use mavros_msgs-msg:sensors_enabled instead.")
  (sensors_enabled m))

(cl:ensure-generic-function 'sensors_health-val :lambda-list '(m))
(cl:defmethod sensors_health-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:sensors_health-val is deprecated.  Use mavros_msgs-msg:sensors_health instead.")
  (sensors_health m))

(cl:ensure-generic-function 'load-val :lambda-list '(m))
(cl:defmethod load-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:load-val is deprecated.  Use mavros_msgs-msg:load instead.")
  (load m))

(cl:ensure-generic-function 'voltage_battery-val :lambda-list '(m))
(cl:defmethod voltage_battery-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:voltage_battery-val is deprecated.  Use mavros_msgs-msg:voltage_battery instead.")
  (voltage_battery m))

(cl:ensure-generic-function 'current_battery-val :lambda-list '(m))
(cl:defmethod current_battery-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:current_battery-val is deprecated.  Use mavros_msgs-msg:current_battery instead.")
  (current_battery m))

(cl:ensure-generic-function 'battery_remaining-val :lambda-list '(m))
(cl:defmethod battery_remaining-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:battery_remaining-val is deprecated.  Use mavros_msgs-msg:battery_remaining instead.")
  (battery_remaining m))

(cl:ensure-generic-function 'drop_rate_comm-val :lambda-list '(m))
(cl:defmethod drop_rate_comm-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:drop_rate_comm-val is deprecated.  Use mavros_msgs-msg:drop_rate_comm instead.")
  (drop_rate_comm m))

(cl:ensure-generic-function 'errors_comm-val :lambda-list '(m))
(cl:defmethod errors_comm-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:errors_comm-val is deprecated.  Use mavros_msgs-msg:errors_comm instead.")
  (errors_comm m))

(cl:ensure-generic-function 'errors_count1-val :lambda-list '(m))
(cl:defmethod errors_count1-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:errors_count1-val is deprecated.  Use mavros_msgs-msg:errors_count1 instead.")
  (errors_count1 m))

(cl:ensure-generic-function 'errors_count2-val :lambda-list '(m))
(cl:defmethod errors_count2-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:errors_count2-val is deprecated.  Use mavros_msgs-msg:errors_count2 instead.")
  (errors_count2 m))

(cl:ensure-generic-function 'errors_count3-val :lambda-list '(m))
(cl:defmethod errors_count3-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:errors_count3-val is deprecated.  Use mavros_msgs-msg:errors_count3 instead.")
  (errors_count3 m))

(cl:ensure-generic-function 'errors_count4-val :lambda-list '(m))
(cl:defmethod errors_count4-val ((m <SysStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:errors_count4-val is deprecated.  Use mavros_msgs-msg:errors_count4 instead.")
  (errors_count4 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SysStatus>) ostream)
  "Serializes a message object of type '<SysStatus>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sensors_present)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'sensors_present)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'sensors_present)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'sensors_present)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sensors_enabled)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'sensors_enabled)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'sensors_enabled)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'sensors_enabled)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sensors_health)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'sensors_health)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'sensors_health)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'sensors_health)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'load)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'load)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'voltage_battery)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'voltage_battery)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'current_battery)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'battery_remaining)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'drop_rate_comm)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'drop_rate_comm)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_comm)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_comm)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count1)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count1)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count2)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count2)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count3)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count3)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count4)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count4)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SysStatus>) istream)
  "Deserializes a message object of type '<SysStatus>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sensors_present)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'sensors_present)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'sensors_present)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'sensors_present)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sensors_enabled)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'sensors_enabled)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'sensors_enabled)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'sensors_enabled)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sensors_health)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'sensors_health)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'sensors_health)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'sensors_health)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'load)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'load)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'voltage_battery)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'voltage_battery)) (cl:read-byte istream))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'current_battery) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'battery_remaining) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'drop_rate_comm)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'drop_rate_comm)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_comm)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_comm)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count1)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count1)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count2)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count2)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count3)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count3)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'errors_count4)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'errors_count4)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SysStatus>)))
  "Returns string type for a message object of type '<SysStatus>"
  "mavros_msgs/SysStatus")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SysStatus)))
  "Returns string type for a message object of type 'SysStatus"
  "mavros_msgs/SysStatus")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SysStatus>)))
  "Returns md5sum for a message object of type '<SysStatus>"
  "4039be26d76b32d20c569c754da6e25c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SysStatus)))
  "Returns md5sum for a message object of type 'SysStatus"
  "4039be26d76b32d20c569c754da6e25c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SysStatus>)))
  "Returns full string definition for message of type '<SysStatus>"
  (cl:format cl:nil "std_msgs/Header header~%~%uint32 sensors_present~%uint32 sensors_enabled~%uint32 sensors_health~%uint16 load~%uint16 voltage_battery~%int16 current_battery~%int8 battery_remaining~%uint16 drop_rate_comm~%uint16 errors_comm~%uint16 errors_count1~%uint16 errors_count2~%uint16 errors_count3~%uint16 errors_count4~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SysStatus)))
  "Returns full string definition for message of type 'SysStatus"
  (cl:format cl:nil "std_msgs/Header header~%~%uint32 sensors_present~%uint32 sensors_enabled~%uint32 sensors_health~%uint16 load~%uint16 voltage_battery~%int16 current_battery~%int8 battery_remaining~%uint16 drop_rate_comm~%uint16 errors_comm~%uint16 errors_count1~%uint16 errors_count2~%uint16 errors_count3~%uint16 errors_count4~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SysStatus>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
     4
     2
     2
     2
     1
     2
     2
     2
     2
     2
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SysStatus>))
  "Converts a ROS message object to a list"
  (cl:list 'SysStatus
    (cl:cons ':header (header msg))
    (cl:cons ':sensors_present (sensors_present msg))
    (cl:cons ':sensors_enabled (sensors_enabled msg))
    (cl:cons ':sensors_health (sensors_health msg))
    (cl:cons ':load (load msg))
    (cl:cons ':voltage_battery (voltage_battery msg))
    (cl:cons ':current_battery (current_battery msg))
    (cl:cons ':battery_remaining (battery_remaining msg))
    (cl:cons ':drop_rate_comm (drop_rate_comm msg))
    (cl:cons ':errors_comm (errors_comm msg))
    (cl:cons ':errors_count1 (errors_count1 msg))
    (cl:cons ':errors_count2 (errors_count2 msg))
    (cl:cons ':errors_count3 (errors_count3 msg))
    (cl:cons ':errors_count4 (errors_count4 msg))
))
