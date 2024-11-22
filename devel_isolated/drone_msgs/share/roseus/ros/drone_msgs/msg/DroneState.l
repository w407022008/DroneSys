;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::DroneState)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'DroneState (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::DRONESTATE")
  (make-package "DRONE_MSGS::DRONESTATE"))

(in-package "ROS")
;;//! \htmlinclude DroneState.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass drone_msgs::DroneState
  :super ros::object
  :slots (_header _connected _armed _landed _mode _time_from_start _position _rel_alt _velocity _attitude _attitude_q _attitude_rate ))

(defmethod drone_msgs::DroneState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:connected __connected) nil)
    ((:armed __armed) nil)
    ((:landed __landed) nil)
    ((:mode __mode) "")
    ((:time_from_start __time_from_start) 0.0)
    ((:position __position) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:rel_alt __rel_alt) 0.0)
    ((:velocity __velocity) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:attitude __attitude) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:attitude_q __attitude_q) (instance geometry_msgs::Quaternion :init))
    ((:attitude_rate __attitude_rate) (make-array 3 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _connected __connected)
   (setq _armed __armed)
   (setq _landed __landed)
   (setq _mode (string __mode))
   (setq _time_from_start (float __time_from_start))
   (setq _position __position)
   (setq _rel_alt (float __rel_alt))
   (setq _velocity __velocity)
   (setq _attitude __attitude)
   (setq _attitude_q __attitude_q)
   (setq _attitude_rate __attitude_rate)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:connected
   (&optional (__connected :null))
   (if (not (eq __connected :null)) (setq _connected __connected)) _connected)
  (:armed
   (&optional (__armed :null))
   (if (not (eq __armed :null)) (setq _armed __armed)) _armed)
  (:landed
   (&optional (__landed :null))
   (if (not (eq __landed :null)) (setq _landed __landed)) _landed)
  (:mode
   (&optional __mode)
   (if __mode (setq _mode __mode)) _mode)
  (:time_from_start
   (&optional __time_from_start)
   (if __time_from_start (setq _time_from_start __time_from_start)) _time_from_start)
  (:position
   (&optional __position)
   (if __position (setq _position __position)) _position)
  (:rel_alt
   (&optional __rel_alt)
   (if __rel_alt (setq _rel_alt __rel_alt)) _rel_alt)
  (:velocity
   (&optional __velocity)
   (if __velocity (setq _velocity __velocity)) _velocity)
  (:attitude
   (&optional __attitude)
   (if __attitude (setq _attitude __attitude)) _attitude)
  (:attitude_q
   (&rest __attitude_q)
   (if (keywordp (car __attitude_q))
       (send* _attitude_q __attitude_q)
     (progn
       (if __attitude_q (setq _attitude_q (car __attitude_q)))
       _attitude_q)))
  (:attitude_rate
   (&optional __attitude_rate)
   (if __attitude_rate (setq _attitude_rate __attitude_rate)) _attitude_rate)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; bool _connected
    1
    ;; bool _armed
    1
    ;; bool _landed
    1
    ;; string _mode
    4 (length _mode)
    ;; float32 _time_from_start
    4
    ;; float32[3] _position
    (* 4    3)
    ;; float32 _rel_alt
    4
    ;; float32[3] _velocity
    (* 4    3)
    ;; float32[3] _attitude
    (* 4    3)
    ;; geometry_msgs/Quaternion _attitude_q
    (send _attitude_q :serialization-length)
    ;; float32[3] _attitude_rate
    (* 4    3)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; bool _connected
       (if _connected (write-byte -1 s) (write-byte 0 s))
     ;; bool _armed
       (if _armed (write-byte -1 s) (write-byte 0 s))
     ;; bool _landed
       (if _landed (write-byte -1 s) (write-byte 0 s))
     ;; string _mode
       (write-long (length _mode) s) (princ _mode s)
     ;; float32 _time_from_start
       (sys::poke _time_from_start (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32[3] _position
     (dotimes (i 3)
       (sys::poke (elt _position i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32 _rel_alt
       (sys::poke _rel_alt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32[3] _velocity
     (dotimes (i 3)
       (sys::poke (elt _velocity i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[3] _attitude
     (dotimes (i 3)
       (sys::poke (elt _attitude i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; geometry_msgs/Quaternion _attitude_q
       (send _attitude_q :serialize s)
     ;; float32[3] _attitude_rate
     (dotimes (i 3)
       (sys::poke (elt _attitude_rate i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; bool _connected
     (setq _connected (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _armed
     (setq _armed (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _landed
     (setq _landed (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _mode
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _mode (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _time_from_start
     (setq _time_from_start (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32[3] _position
   (dotimes (i (length _position))
     (setf (elt _position i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32 _rel_alt
     (setq _rel_alt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32[3] _velocity
   (dotimes (i (length _velocity))
     (setf (elt _velocity i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[3] _attitude
   (dotimes (i (length _attitude))
     (setf (elt _attitude i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; geometry_msgs/Quaternion _attitude_q
     (send _attitude_q :deserialize buf ptr-) (incf ptr- (send _attitude_q :serialization-length))
   ;; float32[3] _attitude_rate
   (dotimes (i (length _attitude_rate))
     (setf (elt _attitude_rate i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;;
   self)
  )

(setf (get drone_msgs::DroneState :md5sum-) "42ddaae3b4854fce0ad7e5cb7bf4370a")
(setf (get drone_msgs::DroneState :datatype-) "drone_msgs/DroneState")
(setf (get drone_msgs::DroneState :definition-)
      "std_msgs/Header header

## is connected with PX4 flight controller
bool connected
bool armed
## landed / in-air
bool landed
## Control mode in PX4 
string mode

float32 time_from_start             ## [s]

## Kinmatic states
float32[3] position                 ## [m]
float32  rel_alt                    ## [m] only for outdoor
float32[3] velocity                 ## [m/s]
float32[3] attitude                 ## [rad]
geometry_msgs/Quaternion attitude_q ## quat
float32[3] attitude_rate            ## [rad/s]

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

")



(provide :drone_msgs/DroneState "42ddaae3b4854fce0ad7e5cb7bf4370a")


