; Auto-generated. Do not edit!


(cl:in-package drone_msgs-msg)


;//! \htmlinclude PositionReference.msg.html

(cl:defclass <PositionReference> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (Move_mode
    :reader Move_mode
    :initarg :Move_mode
    :type cl:fixnum
    :initform 0)
   (Move_frame
    :reader Move_frame
    :initarg :Move_frame
    :type cl:fixnum
    :initform 0)
   (time_from_start
    :reader time_from_start
    :initarg :time_from_start
    :type cl:float
    :initform 0.0)
   (position_ref
    :reader position_ref
    :initarg :position_ref
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (velocity_ref
    :reader velocity_ref
    :initarg :velocity_ref
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (acceleration_ref
    :reader acceleration_ref
    :initarg :acceleration_ref
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (Yaw_Rate_Mode
    :reader Yaw_Rate_Mode
    :initarg :Yaw_Rate_Mode
    :type cl:boolean
    :initform cl:nil)
   (yaw_ref
    :reader yaw_ref
    :initarg :yaw_ref
    :type cl:float
    :initform 0.0)
   (yaw_rate_ref
    :reader yaw_rate_ref
    :initarg :yaw_rate_ref
    :type cl:float
    :initform 0.0)
   (bspline
    :reader bspline
    :initarg :bspline
    :type drone_msgs-msg:Bspline
    :initform (cl:make-instance 'drone_msgs-msg:Bspline)))
)

(cl:defclass PositionReference (<PositionReference>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PositionReference>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PositionReference)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name drone_msgs-msg:<PositionReference> is deprecated: use drone_msgs-msg:PositionReference instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:header-val is deprecated.  Use drone_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'Move_mode-val :lambda-list '(m))
(cl:defmethod Move_mode-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:Move_mode-val is deprecated.  Use drone_msgs-msg:Move_mode instead.")
  (Move_mode m))

(cl:ensure-generic-function 'Move_frame-val :lambda-list '(m))
(cl:defmethod Move_frame-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:Move_frame-val is deprecated.  Use drone_msgs-msg:Move_frame instead.")
  (Move_frame m))

(cl:ensure-generic-function 'time_from_start-val :lambda-list '(m))
(cl:defmethod time_from_start-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:time_from_start-val is deprecated.  Use drone_msgs-msg:time_from_start instead.")
  (time_from_start m))

(cl:ensure-generic-function 'position_ref-val :lambda-list '(m))
(cl:defmethod position_ref-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:position_ref-val is deprecated.  Use drone_msgs-msg:position_ref instead.")
  (position_ref m))

(cl:ensure-generic-function 'velocity_ref-val :lambda-list '(m))
(cl:defmethod velocity_ref-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:velocity_ref-val is deprecated.  Use drone_msgs-msg:velocity_ref instead.")
  (velocity_ref m))

(cl:ensure-generic-function 'acceleration_ref-val :lambda-list '(m))
(cl:defmethod acceleration_ref-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:acceleration_ref-val is deprecated.  Use drone_msgs-msg:acceleration_ref instead.")
  (acceleration_ref m))

(cl:ensure-generic-function 'Yaw_Rate_Mode-val :lambda-list '(m))
(cl:defmethod Yaw_Rate_Mode-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:Yaw_Rate_Mode-val is deprecated.  Use drone_msgs-msg:Yaw_Rate_Mode instead.")
  (Yaw_Rate_Mode m))

(cl:ensure-generic-function 'yaw_ref-val :lambda-list '(m))
(cl:defmethod yaw_ref-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:yaw_ref-val is deprecated.  Use drone_msgs-msg:yaw_ref instead.")
  (yaw_ref m))

(cl:ensure-generic-function 'yaw_rate_ref-val :lambda-list '(m))
(cl:defmethod yaw_rate_ref-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:yaw_rate_ref-val is deprecated.  Use drone_msgs-msg:yaw_rate_ref instead.")
  (yaw_rate_ref m))

(cl:ensure-generic-function 'bspline-val :lambda-list '(m))
(cl:defmethod bspline-val ((m <PositionReference>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:bspline-val is deprecated.  Use drone_msgs-msg:bspline instead.")
  (bspline m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<PositionReference>)))
    "Constants for message type '<PositionReference>"
  '((:XYZ_POS . 0)
    (:XY_POS_Z_VEL . 1)
    (:XY_VEL_Z_POS . 2)
    (:XYZ_VEL . 3)
    (:XYZ_ACC . 4)
    (:XYZ_POS_VEL . 5)
    (:TRAJECTORY . 6)
    (:ENU_FRAME . 0)
    (:BODY_FRAME . 1))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'PositionReference)))
    "Constants for message type 'PositionReference"
  '((:XYZ_POS . 0)
    (:XY_POS_Z_VEL . 1)
    (:XY_VEL_Z_POS . 2)
    (:XYZ_VEL . 3)
    (:XYZ_ACC . 4)
    (:XYZ_POS_VEL . 5)
    (:TRAJECTORY . 6)
    (:ENU_FRAME . 0)
    (:BODY_FRAME . 1))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PositionReference>) ostream)
  "Serializes a message object of type '<PositionReference>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Move_mode)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Move_frame)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'time_from_start))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'position_ref))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'velocity_ref))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'acceleration_ref))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'Yaw_Rate_Mode) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'yaw_ref))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'yaw_rate_ref))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'bspline) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PositionReference>) istream)
  "Deserializes a message object of type '<PositionReference>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Move_mode)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'Move_frame)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'time_from_start) (roslisp-utils:decode-single-float-bits bits)))
  (cl:setf (cl:slot-value msg 'position_ref) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'position_ref)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'velocity_ref) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'velocity_ref)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'acceleration_ref) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'acceleration_ref)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
    (cl:setf (cl:slot-value msg 'Yaw_Rate_Mode) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'yaw_ref) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'yaw_rate_ref) (roslisp-utils:decode-single-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'bspline) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PositionReference>)))
  "Returns string type for a message object of type '<PositionReference>"
  "drone_msgs/PositionReference")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PositionReference)))
  "Returns string type for a message object of type 'PositionReference"
  "drone_msgs/PositionReference")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PositionReference>)))
  "Returns md5sum for a message object of type '<PositionReference>"
  "d029a5a9568f27a7cd91ef81b8f15a11")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PositionReference)))
  "Returns md5sum for a message object of type 'PositionReference"
  "d029a5a9568f27a7cd91ef81b8f15a11")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PositionReference>)))
  "Returns full string definition for message of type '<PositionReference>"
  (cl:format cl:nil "std_msgs/Header header~%~%## Setpoint position reference for PX4 Control~%~%## Setpoint Mode~%uint8 Move_mode~%~%uint8 XYZ_POS      = 0  ##0b00~%uint8 XY_POS_Z_VEL = 1  ##0b01~%uint8 XY_VEL_Z_POS = 2  ##0b10~%uint8 XYZ_VEL = 3       ##0b11~%uint8 XYZ_ACC = 4~%uint8 XYZ_POS_VEL   = 5  ~%uint8 TRAJECTORY   = 6~%~%## Reference Frame~%uint8 Move_frame~%~%uint8 ENU_FRAME  = 0~%uint8 BODY_FRAME = 1~%~%~%~%## Tracking life~%float32 time_from_start          ## [s]~%~%float32[3] position_ref          ## [m]~%float32[3] velocity_ref          ## [m/s]~%float32[3] acceleration_ref      ## [m/s^2]~%~%bool Yaw_Rate_Mode                      ## True 代表控制偏航角速率~%float32 yaw_ref                  ## [rad]~%float32 yaw_rate_ref             ## [rad/s] ~%~%Bspline bspline~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: drone_msgs/Bspline~%int32 order                 ## ~%int64 traj_id               ## id of trajecotry~%float64[] knots             ## knots list~%geometry_msgs/Point[] pts   ## control points list~%time start_time             ## time stamp~%~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PositionReference)))
  "Returns full string definition for message of type 'PositionReference"
  (cl:format cl:nil "std_msgs/Header header~%~%## Setpoint position reference for PX4 Control~%~%## Setpoint Mode~%uint8 Move_mode~%~%uint8 XYZ_POS      = 0  ##0b00~%uint8 XY_POS_Z_VEL = 1  ##0b01~%uint8 XY_VEL_Z_POS = 2  ##0b10~%uint8 XYZ_VEL = 3       ##0b11~%uint8 XYZ_ACC = 4~%uint8 XYZ_POS_VEL   = 5  ~%uint8 TRAJECTORY   = 6~%~%## Reference Frame~%uint8 Move_frame~%~%uint8 ENU_FRAME  = 0~%uint8 BODY_FRAME = 1~%~%~%~%## Tracking life~%float32 time_from_start          ## [s]~%~%float32[3] position_ref          ## [m]~%float32[3] velocity_ref          ## [m/s]~%float32[3] acceleration_ref      ## [m/s^2]~%~%bool Yaw_Rate_Mode                      ## True 代表控制偏航角速率~%float32 yaw_ref                  ## [rad]~%float32 yaw_rate_ref             ## [rad/s] ~%~%Bspline bspline~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: drone_msgs/Bspline~%int32 order                 ## ~%int64 traj_id               ## id of trajecotry~%float64[] knots             ## knots list~%geometry_msgs/Point[] pts   ## control points list~%time start_time             ## time stamp~%~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PositionReference>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     1
     4
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'position_ref) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'velocity_ref) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'acceleration_ref) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     1
     4
     4
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'bspline))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PositionReference>))
  "Converts a ROS message object to a list"
  (cl:list 'PositionReference
    (cl:cons ':header (header msg))
    (cl:cons ':Move_mode (Move_mode msg))
    (cl:cons ':Move_frame (Move_frame msg))
    (cl:cons ':time_from_start (time_from_start msg))
    (cl:cons ':position_ref (position_ref msg))
    (cl:cons ':velocity_ref (velocity_ref msg))
    (cl:cons ':acceleration_ref (acceleration_ref msg))
    (cl:cons ':Yaw_Rate_Mode (Yaw_Rate_Mode msg))
    (cl:cons ':yaw_ref (yaw_ref msg))
    (cl:cons ':yaw_rate_ref (yaw_rate_ref msg))
    (cl:cons ':bspline (bspline msg))
))
