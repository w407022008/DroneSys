;; Auto-generated. Do not edit!


(when (boundp 'quadrotor_msgs::TrajectoryPoint)
  (if (not (find-package "QUADROTOR_MSGS"))
    (make-package "QUADROTOR_MSGS"))
  (shadow 'TrajectoryPoint (find-package "QUADROTOR_MSGS")))
(unless (find-package "QUADROTOR_MSGS::TRAJECTORYPOINT")
  (make-package "QUADROTOR_MSGS::TRAJECTORYPOINT"))

(in-package "ROS")
;;//! \htmlinclude TrajectoryPoint.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass quadrotor_msgs::TrajectoryPoint
  :super ros::object
  :slots (_time_from_start _pose _velocity _acceleration _jerk _snap _heading _heading_rate _heading_acceleration _thrust ))

(defmethod quadrotor_msgs::TrajectoryPoint
  (:init
   (&key
    ((:time_from_start __time_from_start) (instance ros::time :init))
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    ((:velocity __velocity) (instance geometry_msgs::Twist :init))
    ((:acceleration __acceleration) (instance geometry_msgs::Twist :init))
    ((:jerk __jerk) (instance geometry_msgs::Twist :init))
    ((:snap __snap) (instance geometry_msgs::Twist :init))
    ((:heading __heading) 0.0)
    ((:heading_rate __heading_rate) 0.0)
    ((:heading_acceleration __heading_acceleration) 0.0)
    ((:thrust __thrust) 0.0)
    )
   (send-super :init)
   (setq _time_from_start __time_from_start)
   (setq _pose __pose)
   (setq _velocity __velocity)
   (setq _acceleration __acceleration)
   (setq _jerk __jerk)
   (setq _snap __snap)
   (setq _heading (float __heading))
   (setq _heading_rate (float __heading_rate))
   (setq _heading_acceleration (float __heading_acceleration))
   (setq _thrust (float __thrust))
   self)
  (:time_from_start
   (&optional __time_from_start)
   (if __time_from_start (setq _time_from_start __time_from_start)) _time_from_start)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:velocity
   (&rest __velocity)
   (if (keywordp (car __velocity))
       (send* _velocity __velocity)
     (progn
       (if __velocity (setq _velocity (car __velocity)))
       _velocity)))
  (:acceleration
   (&rest __acceleration)
   (if (keywordp (car __acceleration))
       (send* _acceleration __acceleration)
     (progn
       (if __acceleration (setq _acceleration (car __acceleration)))
       _acceleration)))
  (:jerk
   (&rest __jerk)
   (if (keywordp (car __jerk))
       (send* _jerk __jerk)
     (progn
       (if __jerk (setq _jerk (car __jerk)))
       _jerk)))
  (:snap
   (&rest __snap)
   (if (keywordp (car __snap))
       (send* _snap __snap)
     (progn
       (if __snap (setq _snap (car __snap)))
       _snap)))
  (:heading
   (&optional __heading)
   (if __heading (setq _heading __heading)) _heading)
  (:heading_rate
   (&optional __heading_rate)
   (if __heading_rate (setq _heading_rate __heading_rate)) _heading_rate)
  (:heading_acceleration
   (&optional __heading_acceleration)
   (if __heading_acceleration (setq _heading_acceleration __heading_acceleration)) _heading_acceleration)
  (:thrust
   (&optional __thrust)
   (if __thrust (setq _thrust __thrust)) _thrust)
  (:serialization-length
   ()
   (+
    ;; duration _time_from_start
    8
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ;; geometry_msgs/Twist _velocity
    (send _velocity :serialization-length)
    ;; geometry_msgs/Twist _acceleration
    (send _acceleration :serialization-length)
    ;; geometry_msgs/Twist _jerk
    (send _jerk :serialization-length)
    ;; geometry_msgs/Twist _snap
    (send _snap :serialization-length)
    ;; float64 _heading
    8
    ;; float64 _heading_rate
    8
    ;; float64 _heading_acceleration
    8
    ;; float64 _thrust
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; duration _time_from_start
       (write-long (send _time_from_start :sec) s) (write-long (send _time_from_start :nsec) s)
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;; geometry_msgs/Twist _velocity
       (send _velocity :serialize s)
     ;; geometry_msgs/Twist _acceleration
       (send _acceleration :serialize s)
     ;; geometry_msgs/Twist _jerk
       (send _jerk :serialize s)
     ;; geometry_msgs/Twist _snap
       (send _snap :serialize s)
     ;; float64 _heading
       (sys::poke _heading (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _heading_rate
       (sys::poke _heading_rate (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _heading_acceleration
       (sys::poke _heading_acceleration (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _thrust
       (sys::poke _thrust (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; duration _time_from_start
     (send _time_from_start :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _time_from_start :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;; geometry_msgs/Twist _velocity
     (send _velocity :deserialize buf ptr-) (incf ptr- (send _velocity :serialization-length))
   ;; geometry_msgs/Twist _acceleration
     (send _acceleration :deserialize buf ptr-) (incf ptr- (send _acceleration :serialization-length))
   ;; geometry_msgs/Twist _jerk
     (send _jerk :deserialize buf ptr-) (incf ptr- (send _jerk :serialization-length))
   ;; geometry_msgs/Twist _snap
     (send _snap :deserialize buf ptr-) (incf ptr- (send _snap :serialization-length))
   ;; float64 _heading
     (setq _heading (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _heading_rate
     (setq _heading_rate (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _heading_acceleration
     (setq _heading_acceleration (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _thrust
     (setq _thrust (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get quadrotor_msgs::TrajectoryPoint :md5sum-) "1839a691c60e7ab9d8c3da0ab668b51b")
(setf (get quadrotor_msgs::TrajectoryPoint :datatype-) "quadrotor_msgs/TrajectoryPoint")
(setf (get quadrotor_msgs::TrajectoryPoint :definition-)
      "duration time_from_start

geometry_msgs/Pose pose

geometry_msgs/Twist velocity

geometry_msgs/Twist acceleration

geometry_msgs/Twist jerk

geometry_msgs/Twist snap

# Heading angle with respect to world frame [rad]
float64 heading

# First derivative of the heading angle [rad/s]
float64 heading_rate

# Second derivative of the heading angle [rad/s^2]
float64 heading_acceleration

# Collective thrust [m/s^2]
float64 thrust
================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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



(provide :quadrotor_msgs/TrajectoryPoint "1839a691c60e7ab9d8c3da0ab668b51b")


