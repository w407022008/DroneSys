;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::Arduino)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'Arduino (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::ARDUINO")
  (make-package "DRONE_MSGS::ARDUINO"))

(in-package "ROS")
;;//! \htmlinclude Arduino.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*AIRFLOW*" (find-package "DRONE_MSGS::ARDUINO"))
(shadow '*AIRFLOW* (find-package "DRONE_MSGS::ARDUINO"))
(defconstant drone_msgs::Arduino::*AIRFLOW* 0)
(intern "*FORCE*" (find-package "DRONE_MSGS::ARDUINO"))
(shadow '*FORCE* (find-package "DRONE_MSGS::ARDUINO"))
(defconstant drone_msgs::Arduino::*FORCE* 1)
(intern "*IMU*" (find-package "DRONE_MSGS::ARDUINO"))
(shadow '*IMU* (find-package "DRONE_MSGS::ARDUINO"))
(defconstant drone_msgs::Arduino::*IMU* 2)

(defun drone_msgs::Arduino-to-symbol (const)
  (cond
        ((= const 0) 'drone_msgs::Arduino::*AIRFLOW*)
        ((= const 1) 'drone_msgs::Arduino::*FORCE*)
        ((= const 2) 'drone_msgs::Arduino::*IMU*)
        (t nil)))

(defclass drone_msgs::Arduino
  :super ros::object
  :slots (_header _message_type _airflow_sensor_num _current _voltage _power _pow_diff _diff_volt _quaternion _eular_angle _acc _mag _gyro _baro _temp ))

(defmethod drone_msgs::Arduino
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:message_type __message_type) 0)
    ((:airflow_sensor_num __airflow_sensor_num) 0)
    ((:current __current) (make-array 16 :initial-element 0.0 :element-type :float))
    ((:voltage __voltage) (make-array 16 :initial-element 0.0 :element-type :float))
    ((:power __power) (make-array 16 :initial-element 0.0 :element-type :float))
    ((:pow_diff __pow_diff) (make-array 8 :initial-element 0.0 :element-type :float))
    ((:diff_volt __diff_volt) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:quaternion __quaternion) (instance geometry_msgs::Quaternion :init))
    ((:eular_angle __eular_angle) (instance geometry_msgs::Vector3 :init))
    ((:acc __acc) (instance geometry_msgs::Vector3 :init))
    ((:mag __mag) (instance geometry_msgs::Vector3 :init))
    ((:gyro __gyro) (instance geometry_msgs::Vector3 :init))
    ((:baro __baro) 0)
    ((:temp __temp) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _message_type (round __message_type))
   (setq _airflow_sensor_num (round __airflow_sensor_num))
   (setq _current __current)
   (setq _voltage __voltage)
   (setq _power __power)
   (setq _pow_diff __pow_diff)
   (setq _diff_volt __diff_volt)
   (setq _quaternion __quaternion)
   (setq _eular_angle __eular_angle)
   (setq _acc __acc)
   (setq _mag __mag)
   (setq _gyro __gyro)
   (setq _baro (round __baro))
   (setq _temp (float __temp))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:message_type
   (&optional __message_type)
   (if __message_type (setq _message_type __message_type)) _message_type)
  (:airflow_sensor_num
   (&optional __airflow_sensor_num)
   (if __airflow_sensor_num (setq _airflow_sensor_num __airflow_sensor_num)) _airflow_sensor_num)
  (:current
   (&optional __current)
   (if __current (setq _current __current)) _current)
  (:voltage
   (&optional __voltage)
   (if __voltage (setq _voltage __voltage)) _voltage)
  (:power
   (&optional __power)
   (if __power (setq _power __power)) _power)
  (:pow_diff
   (&optional __pow_diff)
   (if __pow_diff (setq _pow_diff __pow_diff)) _pow_diff)
  (:diff_volt
   (&optional __diff_volt)
   (if __diff_volt (setq _diff_volt __diff_volt)) _diff_volt)
  (:quaternion
   (&rest __quaternion)
   (if (keywordp (car __quaternion))
       (send* _quaternion __quaternion)
     (progn
       (if __quaternion (setq _quaternion (car __quaternion)))
       _quaternion)))
  (:eular_angle
   (&rest __eular_angle)
   (if (keywordp (car __eular_angle))
       (send* _eular_angle __eular_angle)
     (progn
       (if __eular_angle (setq _eular_angle (car __eular_angle)))
       _eular_angle)))
  (:acc
   (&rest __acc)
   (if (keywordp (car __acc))
       (send* _acc __acc)
     (progn
       (if __acc (setq _acc (car __acc)))
       _acc)))
  (:mag
   (&rest __mag)
   (if (keywordp (car __mag))
       (send* _mag __mag)
     (progn
       (if __mag (setq _mag (car __mag)))
       _mag)))
  (:gyro
   (&rest __gyro)
   (if (keywordp (car __gyro))
       (send* _gyro __gyro)
     (progn
       (if __gyro (setq _gyro (car __gyro)))
       _gyro)))
  (:baro
   (&optional __baro)
   (if __baro (setq _baro __baro)) _baro)
  (:temp
   (&optional __temp)
   (if __temp (setq _temp __temp)) _temp)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8 _message_type
    1
    ;; uint8 _airflow_sensor_num
    1
    ;; float32[16] _current
    (* 4    16)
    ;; float32[16] _voltage
    (* 4    16)
    ;; float32[16] _power
    (* 4    16)
    ;; float32[8] _pow_diff
    (* 4    8)
    ;; float32[4] _diff_volt
    (* 4    4)
    ;; geometry_msgs/Quaternion _quaternion
    (send _quaternion :serialization-length)
    ;; geometry_msgs/Vector3 _eular_angle
    (send _eular_angle :serialization-length)
    ;; geometry_msgs/Vector3 _acc
    (send _acc :serialization-length)
    ;; geometry_msgs/Vector3 _mag
    (send _mag :serialization-length)
    ;; geometry_msgs/Vector3 _gyro
    (send _gyro :serialization-length)
    ;; uint32 _baro
    4
    ;; float32 _temp
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8 _message_type
       (write-byte _message_type s)
     ;; uint8 _airflow_sensor_num
       (write-byte _airflow_sensor_num s)
     ;; float32[16] _current
     (dotimes (i 16)
       (sys::poke (elt _current i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[16] _voltage
     (dotimes (i 16)
       (sys::poke (elt _voltage i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[16] _power
     (dotimes (i 16)
       (sys::poke (elt _power i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[8] _pow_diff
     (dotimes (i 8)
       (sys::poke (elt _pow_diff i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[4] _diff_volt
     (dotimes (i 4)
       (sys::poke (elt _diff_volt i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; geometry_msgs/Quaternion _quaternion
       (send _quaternion :serialize s)
     ;; geometry_msgs/Vector3 _eular_angle
       (send _eular_angle :serialize s)
     ;; geometry_msgs/Vector3 _acc
       (send _acc :serialize s)
     ;; geometry_msgs/Vector3 _mag
       (send _mag :serialize s)
     ;; geometry_msgs/Vector3 _gyro
       (send _gyro :serialize s)
     ;; uint32 _baro
       (write-long _baro s)
     ;; float32 _temp
       (sys::poke _temp (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8 _message_type
     (setq _message_type (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _airflow_sensor_num
     (setq _airflow_sensor_num (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32[16] _current
   (dotimes (i (length _current))
     (setf (elt _current i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[16] _voltage
   (dotimes (i (length _voltage))
     (setf (elt _voltage i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[16] _power
   (dotimes (i (length _power))
     (setf (elt _power i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[8] _pow_diff
   (dotimes (i (length _pow_diff))
     (setf (elt _pow_diff i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[4] _diff_volt
   (dotimes (i (length _diff_volt))
     (setf (elt _diff_volt i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; geometry_msgs/Quaternion _quaternion
     (send _quaternion :deserialize buf ptr-) (incf ptr- (send _quaternion :serialization-length))
   ;; geometry_msgs/Vector3 _eular_angle
     (send _eular_angle :deserialize buf ptr-) (incf ptr- (send _eular_angle :serialization-length))
   ;; geometry_msgs/Vector3 _acc
     (send _acc :deserialize buf ptr-) (incf ptr- (send _acc :serialization-length))
   ;; geometry_msgs/Vector3 _mag
     (send _mag :deserialize buf ptr-) (incf ptr- (send _mag :serialization-length))
   ;; geometry_msgs/Vector3 _gyro
     (send _gyro :deserialize buf ptr-) (incf ptr- (send _gyro :serialization-length))
   ;; uint32 _baro
     (setq _baro (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _temp
     (setq _temp (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get drone_msgs::Arduino :md5sum-) "75d87b27eabead7e8d84149e18bb1bd0")
(setf (get drone_msgs::Arduino :datatype-) "drone_msgs/Arduino")
(setf (get drone_msgs::Arduino :definition-)
      "std_msgs/Header header

uint8 message_type
# enum message_type
uint8 AIRFLOW = 0  
uint8 FORCE   = 1  
uint8 IMU  = 2  

## Airflow Measurement
uint8 airflow_sensor_num        ## the number of airflow sensor
float32[16] current             ## airflow sensor current measurement [mA]
float32[16] voltage             ## airflow sensor voltage measurement [mV]
float32[16] power               ## airflow sensor power measurement [mW]
float32[8] pow_diff             ## airflow sensor power measurement difference [mW]

## Force Measurement
float32[4] diff_volt            ## Bridge voltage difference of force sensor [uV]

## IMU Measurement
geometry_msgs/Quaternion quaternion			## Quaternion rotation from XYZ body frame to ENU earth frame.
geometry_msgs/Vector3 eular_angle			## Eular angle rotation from XYZ body frame to ENU earth frame.
geometry_msgs/Vector3 acc                  ## in XYZ body frame[m/s^2]
geometry_msgs/Vector3 mag                  ## [m/s^2]
geometry_msgs/Vector3 gyro                 ## [m/s^2]
uint32 baro                      ## [pascal]
float32 temp                     ## [degree]

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :drone_msgs/Arduino "75d87b27eabead7e8d84149e18bb1bd0")


