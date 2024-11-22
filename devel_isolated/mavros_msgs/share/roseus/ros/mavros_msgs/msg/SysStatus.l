;; Auto-generated. Do not edit!


(when (boundp 'mavros_msgs::SysStatus)
  (if (not (find-package "MAVROS_MSGS"))
    (make-package "MAVROS_MSGS"))
  (shadow 'SysStatus (find-package "MAVROS_MSGS")))
(unless (find-package "MAVROS_MSGS::SYSSTATUS")
  (make-package "MAVROS_MSGS::SYSSTATUS"))

(in-package "ROS")
;;//! \htmlinclude SysStatus.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass mavros_msgs::SysStatus
  :super ros::object
  :slots (_header _sensors_present _sensors_enabled _sensors_health _load _voltage_battery _current_battery _battery_remaining _drop_rate_comm _errors_comm _errors_count1 _errors_count2 _errors_count3 _errors_count4 ))

(defmethod mavros_msgs::SysStatus
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:sensors_present __sensors_present) 0)
    ((:sensors_enabled __sensors_enabled) 0)
    ((:sensors_health __sensors_health) 0)
    ((:load __load) 0)
    ((:voltage_battery __voltage_battery) 0)
    ((:current_battery __current_battery) 0)
    ((:battery_remaining __battery_remaining) 0)
    ((:drop_rate_comm __drop_rate_comm) 0)
    ((:errors_comm __errors_comm) 0)
    ((:errors_count1 __errors_count1) 0)
    ((:errors_count2 __errors_count2) 0)
    ((:errors_count3 __errors_count3) 0)
    ((:errors_count4 __errors_count4) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _sensors_present (round __sensors_present))
   (setq _sensors_enabled (round __sensors_enabled))
   (setq _sensors_health (round __sensors_health))
   (setq _load (round __load))
   (setq _voltage_battery (round __voltage_battery))
   (setq _current_battery (round __current_battery))
   (setq _battery_remaining (round __battery_remaining))
   (setq _drop_rate_comm (round __drop_rate_comm))
   (setq _errors_comm (round __errors_comm))
   (setq _errors_count1 (round __errors_count1))
   (setq _errors_count2 (round __errors_count2))
   (setq _errors_count3 (round __errors_count3))
   (setq _errors_count4 (round __errors_count4))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:sensors_present
   (&optional __sensors_present)
   (if __sensors_present (setq _sensors_present __sensors_present)) _sensors_present)
  (:sensors_enabled
   (&optional __sensors_enabled)
   (if __sensors_enabled (setq _sensors_enabled __sensors_enabled)) _sensors_enabled)
  (:sensors_health
   (&optional __sensors_health)
   (if __sensors_health (setq _sensors_health __sensors_health)) _sensors_health)
  (:load
   (&optional __load)
   (if __load (setq _load __load)) _load)
  (:voltage_battery
   (&optional __voltage_battery)
   (if __voltage_battery (setq _voltage_battery __voltage_battery)) _voltage_battery)
  (:current_battery
   (&optional __current_battery)
   (if __current_battery (setq _current_battery __current_battery)) _current_battery)
  (:battery_remaining
   (&optional __battery_remaining)
   (if __battery_remaining (setq _battery_remaining __battery_remaining)) _battery_remaining)
  (:drop_rate_comm
   (&optional __drop_rate_comm)
   (if __drop_rate_comm (setq _drop_rate_comm __drop_rate_comm)) _drop_rate_comm)
  (:errors_comm
   (&optional __errors_comm)
   (if __errors_comm (setq _errors_comm __errors_comm)) _errors_comm)
  (:errors_count1
   (&optional __errors_count1)
   (if __errors_count1 (setq _errors_count1 __errors_count1)) _errors_count1)
  (:errors_count2
   (&optional __errors_count2)
   (if __errors_count2 (setq _errors_count2 __errors_count2)) _errors_count2)
  (:errors_count3
   (&optional __errors_count3)
   (if __errors_count3 (setq _errors_count3 __errors_count3)) _errors_count3)
  (:errors_count4
   (&optional __errors_count4)
   (if __errors_count4 (setq _errors_count4 __errors_count4)) _errors_count4)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint32 _sensors_present
    4
    ;; uint32 _sensors_enabled
    4
    ;; uint32 _sensors_health
    4
    ;; uint16 _load
    2
    ;; uint16 _voltage_battery
    2
    ;; int16 _current_battery
    2
    ;; int8 _battery_remaining
    1
    ;; uint16 _drop_rate_comm
    2
    ;; uint16 _errors_comm
    2
    ;; uint16 _errors_count1
    2
    ;; uint16 _errors_count2
    2
    ;; uint16 _errors_count3
    2
    ;; uint16 _errors_count4
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint32 _sensors_present
       (write-long _sensors_present s)
     ;; uint32 _sensors_enabled
       (write-long _sensors_enabled s)
     ;; uint32 _sensors_health
       (write-long _sensors_health s)
     ;; uint16 _load
       (write-word _load s)
     ;; uint16 _voltage_battery
       (write-word _voltage_battery s)
     ;; int16 _current_battery
       (write-word _current_battery s)
     ;; int8 _battery_remaining
       (write-byte _battery_remaining s)
     ;; uint16 _drop_rate_comm
       (write-word _drop_rate_comm s)
     ;; uint16 _errors_comm
       (write-word _errors_comm s)
     ;; uint16 _errors_count1
       (write-word _errors_count1 s)
     ;; uint16 _errors_count2
       (write-word _errors_count2 s)
     ;; uint16 _errors_count3
       (write-word _errors_count3 s)
     ;; uint16 _errors_count4
       (write-word _errors_count4 s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint32 _sensors_present
     (setq _sensors_present (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _sensors_enabled
     (setq _sensors_enabled (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _sensors_health
     (setq _sensors_health (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint16 _load
     (setq _load (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _voltage_battery
     (setq _voltage_battery (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _current_battery
     (setq _current_battery (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int8 _battery_remaining
     (setq _battery_remaining (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _battery_remaining 127) (setq _battery_remaining (- _battery_remaining 256)))
   ;; uint16 _drop_rate_comm
     (setq _drop_rate_comm (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _errors_comm
     (setq _errors_comm (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _errors_count1
     (setq _errors_count1 (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _errors_count2
     (setq _errors_count2 (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _errors_count3
     (setq _errors_count3 (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _errors_count4
     (setq _errors_count4 (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get mavros_msgs::SysStatus :md5sum-) "4039be26d76b32d20c569c754da6e25c")
(setf (get mavros_msgs::SysStatus :datatype-) "mavros_msgs/SysStatus")
(setf (get mavros_msgs::SysStatus :definition-)
      "std_msgs/Header header

uint32 sensors_present
uint32 sensors_enabled
uint32 sensors_health
uint16 load
uint16 voltage_battery
int16 current_battery
int8 battery_remaining
uint16 drop_rate_comm
uint16 errors_comm
uint16 errors_count1
uint16 errors_count2
uint16 errors_count3
uint16 errors_count4
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

")



(provide :mavros_msgs/SysStatus "4039be26d76b32d20c569c754da6e25c")


