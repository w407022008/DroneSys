;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::AttitudeReference)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'AttitudeReference (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::ATTITUDEREFERENCE")
  (make-package "DRONE_MSGS::ATTITUDEREFERENCE"))

(in-package "ROS")
;;//! \htmlinclude AttitudeReference.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass drone_msgs::AttitudeReference
  :super ros::object
  :slots (_header _thrust_sp _collective_accel _desired_attitude _desired_att_q _body_rate ))

(defmethod drone_msgs::AttitudeReference
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:thrust_sp __thrust_sp) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:collective_accel __collective_accel) 0.0)
    ((:desired_attitude __desired_attitude) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:desired_att_q __desired_att_q) (instance geometry_msgs::Quaternion :init))
    ((:body_rate __body_rate) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _thrust_sp __thrust_sp)
   (setq _collective_accel (float __collective_accel))
   (setq _desired_attitude __desired_attitude)
   (setq _desired_att_q __desired_att_q)
   (setq _body_rate __body_rate)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:thrust_sp
   (&optional __thrust_sp)
   (if __thrust_sp (setq _thrust_sp __thrust_sp)) _thrust_sp)
  (:collective_accel
   (&optional __collective_accel)
   (if __collective_accel (setq _collective_accel __collective_accel)) _collective_accel)
  (:desired_attitude
   (&optional __desired_attitude)
   (if __desired_attitude (setq _desired_attitude __desired_attitude)) _desired_attitude)
  (:desired_att_q
   (&rest __desired_att_q)
   (if (keywordp (car __desired_att_q))
       (send* _desired_att_q __desired_att_q)
     (progn
       (if __desired_att_q (setq _desired_att_q (car __desired_att_q)))
       _desired_att_q)))
  (:body_rate
   (&rest __body_rate)
   (if (keywordp (car __body_rate))
       (send* _body_rate __body_rate)
     (progn
       (if __body_rate (setq _body_rate (car __body_rate)))
       _body_rate)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32[3] _thrust_sp
    (* 4    3)
    ;; float32 _collective_accel
    4
    ;; float32[3] _desired_attitude
    (* 4    3)
    ;; geometry_msgs/Quaternion _desired_att_q
    (send _desired_att_q :serialization-length)
    ;; geometry_msgs/Vector3 _body_rate
    (send _body_rate :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32[3] _thrust_sp
     (dotimes (i 3)
       (sys::poke (elt _thrust_sp i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32 _collective_accel
       (sys::poke _collective_accel (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32[3] _desired_attitude
     (dotimes (i 3)
       (sys::poke (elt _desired_attitude i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; geometry_msgs/Quaternion _desired_att_q
       (send _desired_att_q :serialize s)
     ;; geometry_msgs/Vector3 _body_rate
       (send _body_rate :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32[3] _thrust_sp
   (dotimes (i (length _thrust_sp))
     (setf (elt _thrust_sp i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32 _collective_accel
     (setq _collective_accel (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32[3] _desired_attitude
   (dotimes (i (length _desired_attitude))
     (setf (elt _desired_attitude i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; geometry_msgs/Quaternion _desired_att_q
     (send _desired_att_q :deserialize buf ptr-) (incf ptr- (send _desired_att_q :serialization-length))
   ;; geometry_msgs/Vector3 _body_rate
     (send _body_rate :deserialize buf ptr-) (incf ptr- (send _body_rate :serialization-length))
   ;;
   self)
  )

(setf (get drone_msgs::AttitudeReference :md5sum-) "ad65c8727b64e262c550df8ad8b37905")
(setf (get drone_msgs::AttitudeReference :datatype-) "drone_msgs/AttitudeReference")
(setf (get drone_msgs::AttitudeReference :definition-)
      "std_msgs/Header header

## Setpoint Attitude + T
float32[3] thrust_sp                   ## Single Rotor Thrust setpoint
float32 collective_accel               ## [m/s^2] Axis Body_Z Collective accel septoint
float32[3] desired_attitude            ## [rad] Eurler angle setpoint
geometry_msgs/Quaternion desired_att_q ## quat setpoint
geometry_msgs/Vector3 body_rate  ## [rad/s]

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



(provide :drone_msgs/AttitudeReference "ad65c8727b64e262c550df8ad8b37905")


