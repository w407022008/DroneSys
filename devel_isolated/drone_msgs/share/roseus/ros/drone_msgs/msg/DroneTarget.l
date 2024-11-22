;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::DroneTarget)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'DroneTarget (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::DRONETARGET")
  (make-package "DRONE_MSGS::DRONETARGET"))

(in-package "ROS")
;;//! \htmlinclude DroneTarget.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "MAVROS_MSGS"))
  (ros::roseus-add-msgs "mavros_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass drone_msgs::DroneTarget
  :super ros::object
  :slots (_header _position_target _velocity_target _acceleration_target _q_target _euler_target _rate_target _thrust_target _actuator_target ))

(defmethod drone_msgs::DroneTarget
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:position_target __position_target) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:velocity_target __velocity_target) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:acceleration_target __acceleration_target) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:q_target __q_target) (instance geometry_msgs::Quaternion :init))
    ((:euler_target __euler_target) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:rate_target __rate_target) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:thrust_target __thrust_target) 0.0)
    ((:actuator_target __actuator_target) (instance mavros_msgs::ActuatorControl :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _position_target __position_target)
   (setq _velocity_target __velocity_target)
   (setq _acceleration_target __acceleration_target)
   (setq _q_target __q_target)
   (setq _euler_target __euler_target)
   (setq _rate_target __rate_target)
   (setq _thrust_target (float __thrust_target))
   (setq _actuator_target __actuator_target)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:position_target
   (&optional __position_target)
   (if __position_target (setq _position_target __position_target)) _position_target)
  (:velocity_target
   (&optional __velocity_target)
   (if __velocity_target (setq _velocity_target __velocity_target)) _velocity_target)
  (:acceleration_target
   (&optional __acceleration_target)
   (if __acceleration_target (setq _acceleration_target __acceleration_target)) _acceleration_target)
  (:q_target
   (&rest __q_target)
   (if (keywordp (car __q_target))
       (send* _q_target __q_target)
     (progn
       (if __q_target (setq _q_target (car __q_target)))
       _q_target)))
  (:euler_target
   (&optional __euler_target)
   (if __euler_target (setq _euler_target __euler_target)) _euler_target)
  (:rate_target
   (&optional __rate_target)
   (if __rate_target (setq _rate_target __rate_target)) _rate_target)
  (:thrust_target
   (&optional __thrust_target)
   (if __thrust_target (setq _thrust_target __thrust_target)) _thrust_target)
  (:actuator_target
   (&rest __actuator_target)
   (if (keywordp (car __actuator_target))
       (send* _actuator_target __actuator_target)
     (progn
       (if __actuator_target (setq _actuator_target (car __actuator_target)))
       _actuator_target)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32[3] _position_target
    (* 4    3)
    ;; float32[3] _velocity_target
    (* 4    3)
    ;; float32[3] _acceleration_target
    (* 4    3)
    ;; geometry_msgs/Quaternion _q_target
    (send _q_target :serialization-length)
    ;; float32[3] _euler_target
    (* 4    3)
    ;; float32[3] _rate_target
    (* 4    3)
    ;; float32 _thrust_target
    4
    ;; mavros_msgs/ActuatorControl _actuator_target
    (send _actuator_target :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32[3] _position_target
     (dotimes (i 3)
       (sys::poke (elt _position_target i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[3] _velocity_target
     (dotimes (i 3)
       (sys::poke (elt _velocity_target i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[3] _acceleration_target
     (dotimes (i 3)
       (sys::poke (elt _acceleration_target i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; geometry_msgs/Quaternion _q_target
       (send _q_target :serialize s)
     ;; float32[3] _euler_target
     (dotimes (i 3)
       (sys::poke (elt _euler_target i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[3] _rate_target
     (dotimes (i 3)
       (sys::poke (elt _rate_target i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32 _thrust_target
       (sys::poke _thrust_target (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; mavros_msgs/ActuatorControl _actuator_target
       (send _actuator_target :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32[3] _position_target
   (dotimes (i (length _position_target))
     (setf (elt _position_target i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[3] _velocity_target
   (dotimes (i (length _velocity_target))
     (setf (elt _velocity_target i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[3] _acceleration_target
   (dotimes (i (length _acceleration_target))
     (setf (elt _acceleration_target i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; geometry_msgs/Quaternion _q_target
     (send _q_target :deserialize buf ptr-) (incf ptr- (send _q_target :serialization-length))
   ;; float32[3] _euler_target
   (dotimes (i (length _euler_target))
     (setf (elt _euler_target i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[3] _rate_target
   (dotimes (i (length _rate_target))
     (setf (elt _rate_target i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32 _thrust_target
     (setq _thrust_target (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; mavros_msgs/ActuatorControl _actuator_target
     (send _actuator_target :deserialize buf ptr-) (incf ptr- (send _actuator_target :serialization-length))
   ;;
   self)
  )

(setf (get drone_msgs::DroneTarget :md5sum-) "b13c4477f8a36524e314a3b537e64de4")
(setf (get drone_msgs::DroneTarget :datatype-) "drone_msgs/DroneTarget")
(setf (get drone_msgs::DroneTarget :definition-)
      "std_msgs/Header header

float32[3] position_target          ## [m]
float32[3] velocity_target          ## [m/s]
float32[3] acceleration_target      ## [m/s/s]
geometry_msgs/Quaternion q_target   ## quat
float32[3] euler_target             ## [rad]
float32[3] rate_target              ## [rad/s]
float32 thrust_target
mavros_msgs/ActuatorControl actuator_target

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
MSG: mavros_msgs/ActuatorControl
# raw servo values for direct actuator controls
#
# about groups, mixing and channels:
# https://pixhawk.org/dev/mixing

# constant for mixer group
uint8 PX4_MIX_FLIGHT_CONTROL = 0
uint8 PX4_MIX_FLIGHT_CONTROL_VTOL_ALT = 1
uint8 PX4_MIX_PAYLOAD = 2
uint8 PX4_MIX_MANUAL_PASSTHROUGH = 3
#uint8 PX4_MIX_FC_MC_VIRT = 4
#uint8 PX4_MIX_FC_FW_VIRT = 5

std_msgs/Header header
uint8 group_mix
float32[8] controls

")



(provide :drone_msgs/DroneTarget "b13c4477f8a36524e314a3b537e64de4")


