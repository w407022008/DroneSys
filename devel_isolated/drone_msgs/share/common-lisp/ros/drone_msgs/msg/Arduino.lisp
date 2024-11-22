; Auto-generated. Do not edit!


(cl:in-package drone_msgs-msg)


;//! \htmlinclude Arduino.msg.html

(cl:defclass <Arduino> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (message_type
    :reader message_type
    :initarg :message_type
    :type cl:fixnum
    :initform 0)
   (airflow_sensor_num
    :reader airflow_sensor_num
    :initarg :airflow_sensor_num
    :type cl:fixnum
    :initform 0)
   (current
    :reader current
    :initarg :current
    :type (cl:vector cl:float)
   :initform (cl:make-array 16 :element-type 'cl:float :initial-element 0.0))
   (voltage
    :reader voltage
    :initarg :voltage
    :type (cl:vector cl:float)
   :initform (cl:make-array 16 :element-type 'cl:float :initial-element 0.0))
   (power
    :reader power
    :initarg :power
    :type (cl:vector cl:float)
   :initform (cl:make-array 16 :element-type 'cl:float :initial-element 0.0))
   (pow_diff
    :reader pow_diff
    :initarg :pow_diff
    :type (cl:vector cl:float)
   :initform (cl:make-array 8 :element-type 'cl:float :initial-element 0.0))
   (diff_volt
    :reader diff_volt
    :initarg :diff_volt
    :type (cl:vector cl:float)
   :initform (cl:make-array 4 :element-type 'cl:float :initial-element 0.0))
   (quaternion
    :reader quaternion
    :initarg :quaternion
    :type geometry_msgs-msg:Quaternion
    :initform (cl:make-instance 'geometry_msgs-msg:Quaternion))
   (eular_angle
    :reader eular_angle
    :initarg :eular_angle
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (acc
    :reader acc
    :initarg :acc
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (mag
    :reader mag
    :initarg :mag
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (gyro
    :reader gyro
    :initarg :gyro
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (baro
    :reader baro
    :initarg :baro
    :type cl:integer
    :initform 0)
   (temp
    :reader temp
    :initarg :temp
    :type cl:float
    :initform 0.0))
)

(cl:defclass Arduino (<Arduino>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Arduino>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Arduino)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name drone_msgs-msg:<Arduino> is deprecated: use drone_msgs-msg:Arduino instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:header-val is deprecated.  Use drone_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'message_type-val :lambda-list '(m))
(cl:defmethod message_type-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:message_type-val is deprecated.  Use drone_msgs-msg:message_type instead.")
  (message_type m))

(cl:ensure-generic-function 'airflow_sensor_num-val :lambda-list '(m))
(cl:defmethod airflow_sensor_num-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:airflow_sensor_num-val is deprecated.  Use drone_msgs-msg:airflow_sensor_num instead.")
  (airflow_sensor_num m))

(cl:ensure-generic-function 'current-val :lambda-list '(m))
(cl:defmethod current-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:current-val is deprecated.  Use drone_msgs-msg:current instead.")
  (current m))

(cl:ensure-generic-function 'voltage-val :lambda-list '(m))
(cl:defmethod voltage-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:voltage-val is deprecated.  Use drone_msgs-msg:voltage instead.")
  (voltage m))

(cl:ensure-generic-function 'power-val :lambda-list '(m))
(cl:defmethod power-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:power-val is deprecated.  Use drone_msgs-msg:power instead.")
  (power m))

(cl:ensure-generic-function 'pow_diff-val :lambda-list '(m))
(cl:defmethod pow_diff-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:pow_diff-val is deprecated.  Use drone_msgs-msg:pow_diff instead.")
  (pow_diff m))

(cl:ensure-generic-function 'diff_volt-val :lambda-list '(m))
(cl:defmethod diff_volt-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:diff_volt-val is deprecated.  Use drone_msgs-msg:diff_volt instead.")
  (diff_volt m))

(cl:ensure-generic-function 'quaternion-val :lambda-list '(m))
(cl:defmethod quaternion-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:quaternion-val is deprecated.  Use drone_msgs-msg:quaternion instead.")
  (quaternion m))

(cl:ensure-generic-function 'eular_angle-val :lambda-list '(m))
(cl:defmethod eular_angle-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:eular_angle-val is deprecated.  Use drone_msgs-msg:eular_angle instead.")
  (eular_angle m))

(cl:ensure-generic-function 'acc-val :lambda-list '(m))
(cl:defmethod acc-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:acc-val is deprecated.  Use drone_msgs-msg:acc instead.")
  (acc m))

(cl:ensure-generic-function 'mag-val :lambda-list '(m))
(cl:defmethod mag-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:mag-val is deprecated.  Use drone_msgs-msg:mag instead.")
  (mag m))

(cl:ensure-generic-function 'gyro-val :lambda-list '(m))
(cl:defmethod gyro-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:gyro-val is deprecated.  Use drone_msgs-msg:gyro instead.")
  (gyro m))

(cl:ensure-generic-function 'baro-val :lambda-list '(m))
(cl:defmethod baro-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:baro-val is deprecated.  Use drone_msgs-msg:baro instead.")
  (baro m))

(cl:ensure-generic-function 'temp-val :lambda-list '(m))
(cl:defmethod temp-val ((m <Arduino>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader drone_msgs-msg:temp-val is deprecated.  Use drone_msgs-msg:temp instead.")
  (temp m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<Arduino>)))
    "Constants for message type '<Arduino>"
  '((:AIRFLOW . 0)
    (:FORCE . 1)
    (:IMU . 2))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'Arduino)))
    "Constants for message type 'Arduino"
  '((:AIRFLOW . 0)
    (:FORCE . 1)
    (:IMU . 2))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Arduino>) ostream)
  "Serializes a message object of type '<Arduino>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'message_type)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'airflow_sensor_num)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'current))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'voltage))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'power))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'pow_diff))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'diff_volt))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'quaternion) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'eular_angle) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'acc) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'mag) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'gyro) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'baro)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'baro)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'baro)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'baro)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'temp))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Arduino>) istream)
  "Deserializes a message object of type '<Arduino>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'message_type)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'airflow_sensor_num)) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'current) (cl:make-array 16))
  (cl:let ((vals (cl:slot-value msg 'current)))
    (cl:dotimes (i 16)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'voltage) (cl:make-array 16))
  (cl:let ((vals (cl:slot-value msg 'voltage)))
    (cl:dotimes (i 16)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'power) (cl:make-array 16))
  (cl:let ((vals (cl:slot-value msg 'power)))
    (cl:dotimes (i 16)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'pow_diff) (cl:make-array 8))
  (cl:let ((vals (cl:slot-value msg 'pow_diff)))
    (cl:dotimes (i 8)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'diff_volt) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'diff_volt)))
    (cl:dotimes (i 4)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'quaternion) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'eular_angle) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'acc) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'mag) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'gyro) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'baro)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'baro)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'baro)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'baro)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'temp) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Arduino>)))
  "Returns string type for a message object of type '<Arduino>"
  "drone_msgs/Arduino")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Arduino)))
  "Returns string type for a message object of type 'Arduino"
  "drone_msgs/Arduino")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Arduino>)))
  "Returns md5sum for a message object of type '<Arduino>"
  "75d87b27eabead7e8d84149e18bb1bd0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Arduino)))
  "Returns md5sum for a message object of type 'Arduino"
  "75d87b27eabead7e8d84149e18bb1bd0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Arduino>)))
  "Returns full string definition for message of type '<Arduino>"
  (cl:format cl:nil "std_msgs/Header header~%~%uint8 message_type~%# enum message_type~%uint8 AIRFLOW = 0  ~%uint8 FORCE   = 1  ~%uint8 IMU  = 2  ~%~%## Airflow Measurement~%uint8 airflow_sensor_num        ## the number of airflow sensor~%float32[16] current             ## airflow sensor current measurement [mA]~%float32[16] voltage             ## airflow sensor voltage measurement [mV]~%float32[16] power               ## airflow sensor power measurement [mW]~%float32[8] pow_diff             ## airflow sensor power measurement difference [mW]~%~%## Force Measurement~%float32[4] diff_volt            ## Bridge voltage difference of force sensor [uV]~%~%## IMU Measurement~%geometry_msgs/Quaternion quaternion			## Quaternion rotation from XYZ body frame to ENU earth frame.~%geometry_msgs/Vector3 eular_angle			## Eular angle rotation from XYZ body frame to ENU earth frame.~%geometry_msgs/Vector3 acc                  ## in XYZ body frame[m/s^2]~%geometry_msgs/Vector3 mag                  ## [m/s^2]~%geometry_msgs/Vector3 gyro                 ## [m/s^2]~%uint32 baro                      ## [pascal]~%float32 temp                     ## [degree]~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Arduino)))
  "Returns full string definition for message of type 'Arduino"
  (cl:format cl:nil "std_msgs/Header header~%~%uint8 message_type~%# enum message_type~%uint8 AIRFLOW = 0  ~%uint8 FORCE   = 1  ~%uint8 IMU  = 2  ~%~%## Airflow Measurement~%uint8 airflow_sensor_num        ## the number of airflow sensor~%float32[16] current             ## airflow sensor current measurement [mA]~%float32[16] voltage             ## airflow sensor voltage measurement [mV]~%float32[16] power               ## airflow sensor power measurement [mW]~%float32[8] pow_diff             ## airflow sensor power measurement difference [mW]~%~%## Force Measurement~%float32[4] diff_volt            ## Bridge voltage difference of force sensor [uV]~%~%## IMU Measurement~%geometry_msgs/Quaternion quaternion			## Quaternion rotation from XYZ body frame to ENU earth frame.~%geometry_msgs/Vector3 eular_angle			## Eular angle rotation from XYZ body frame to ENU earth frame.~%geometry_msgs/Vector3 acc                  ## in XYZ body frame[m/s^2]~%geometry_msgs/Vector3 mag                  ## [m/s^2]~%geometry_msgs/Vector3 gyro                 ## [m/s^2]~%uint32 baro                      ## [pascal]~%float32 temp                     ## [degree]~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Arduino>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     1
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'current) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'voltage) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'power) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'pow_diff) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'diff_volt) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'quaternion))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'eular_angle))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'acc))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'mag))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'gyro))
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Arduino>))
  "Converts a ROS message object to a list"
  (cl:list 'Arduino
    (cl:cons ':header (header msg))
    (cl:cons ':message_type (message_type msg))
    (cl:cons ':airflow_sensor_num (airflow_sensor_num msg))
    (cl:cons ':current (current msg))
    (cl:cons ':voltage (voltage msg))
    (cl:cons ':power (power msg))
    (cl:cons ':pow_diff (pow_diff msg))
    (cl:cons ':diff_volt (diff_volt msg))
    (cl:cons ':quaternion (quaternion msg))
    (cl:cons ':eular_angle (eular_angle msg))
    (cl:cons ':acc (acc msg))
    (cl:cons ':mag (mag msg))
    (cl:cons ':gyro (gyro msg))
    (cl:cons ':baro (baro msg))
    (cl:cons ':temp (temp msg))
))
