; Auto-generated. Do not edit!


(cl:in-package mavros_msgs-msg)


;//! \htmlinclude CellularStatus.msg.html

(cl:defclass <CellularStatus> (roslisp-msg-protocol:ros-message)
  ((status
    :reader status
    :initarg :status
    :type cl:fixnum
    :initform 0)
   (failure_reason
    :reader failure_reason
    :initarg :failure_reason
    :type cl:fixnum
    :initform 0)
   (type
    :reader type
    :initarg :type
    :type cl:fixnum
    :initform 0)
   (quality
    :reader quality
    :initarg :quality
    :type cl:fixnum
    :initform 0)
   (mcc
    :reader mcc
    :initarg :mcc
    :type cl:fixnum
    :initform 0)
   (mnc
    :reader mnc
    :initarg :mnc
    :type cl:fixnum
    :initform 0)
   (lac
    :reader lac
    :initarg :lac
    :type cl:fixnum
    :initform 0))
)

(cl:defclass CellularStatus (<CellularStatus>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CellularStatus>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CellularStatus)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name mavros_msgs-msg:<CellularStatus> is deprecated: use mavros_msgs-msg:CellularStatus instead.")))

(cl:ensure-generic-function 'status-val :lambda-list '(m))
(cl:defmethod status-val ((m <CellularStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:status-val is deprecated.  Use mavros_msgs-msg:status instead.")
  (status m))

(cl:ensure-generic-function 'failure_reason-val :lambda-list '(m))
(cl:defmethod failure_reason-val ((m <CellularStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:failure_reason-val is deprecated.  Use mavros_msgs-msg:failure_reason instead.")
  (failure_reason m))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <CellularStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:type-val is deprecated.  Use mavros_msgs-msg:type instead.")
  (type m))

(cl:ensure-generic-function 'quality-val :lambda-list '(m))
(cl:defmethod quality-val ((m <CellularStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:quality-val is deprecated.  Use mavros_msgs-msg:quality instead.")
  (quality m))

(cl:ensure-generic-function 'mcc-val :lambda-list '(m))
(cl:defmethod mcc-val ((m <CellularStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:mcc-val is deprecated.  Use mavros_msgs-msg:mcc instead.")
  (mcc m))

(cl:ensure-generic-function 'mnc-val :lambda-list '(m))
(cl:defmethod mnc-val ((m <CellularStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:mnc-val is deprecated.  Use mavros_msgs-msg:mnc instead.")
  (mnc m))

(cl:ensure-generic-function 'lac-val :lambda-list '(m))
(cl:defmethod lac-val ((m <CellularStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mavros_msgs-msg:lac-val is deprecated.  Use mavros_msgs-msg:lac instead.")
  (lac m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CellularStatus>) ostream)
  "Serializes a message object of type '<CellularStatus>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'status)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'failure_reason)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'type)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'quality)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mcc)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'mcc)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mnc)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'mnc)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'lac)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'lac)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CellularStatus>) istream)
  "Deserializes a message object of type '<CellularStatus>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'status)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'failure_reason)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'type)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'quality)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mcc)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'mcc)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'mnc)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'mnc)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'lac)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'lac)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CellularStatus>)))
  "Returns string type for a message object of type '<CellularStatus>"
  "mavros_msgs/CellularStatus")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CellularStatus)))
  "Returns string type for a message object of type 'CellularStatus"
  "mavros_msgs/CellularStatus")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CellularStatus>)))
  "Returns md5sum for a message object of type '<CellularStatus>"
  "a474bdb9df111b4e16fab4c29f7a6956")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CellularStatus)))
  "Returns md5sum for a message object of type 'CellularStatus"
  "a474bdb9df111b4e16fab4c29f7a6956")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CellularStatus>)))
  "Returns full string definition for message of type '<CellularStatus>"
  (cl:format cl:nil "#Follows https://mavlink.io/en/messages/common.html#CELLULAR_STATUS specification~%~%uint8 status~%uint8 failure_reason~%uint8 type~%uint8 quality~%uint16 mcc~%uint16 mnc~%uint16 lac~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CellularStatus)))
  "Returns full string definition for message of type 'CellularStatus"
  (cl:format cl:nil "#Follows https://mavlink.io/en/messages/common.html#CELLULAR_STATUS specification~%~%uint8 status~%uint8 failure_reason~%uint8 type~%uint8 quality~%uint16 mcc~%uint16 mnc~%uint16 lac~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CellularStatus>))
  (cl:+ 0
     1
     1
     1
     1
     2
     2
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CellularStatus>))
  "Converts a ROS message object to a list"
  (cl:list 'CellularStatus
    (cl:cons ':status (status msg))
    (cl:cons ':failure_reason (failure_reason msg))
    (cl:cons ':type (type msg))
    (cl:cons ':quality (quality msg))
    (cl:cons ':mcc (mcc msg))
    (cl:cons ':mnc (mnc msg))
    (cl:cons ':lac (lac msg))
))
