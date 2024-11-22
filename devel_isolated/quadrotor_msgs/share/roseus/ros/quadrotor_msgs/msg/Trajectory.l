;; Auto-generated. Do not edit!


(when (boundp 'quadrotor_msgs::Trajectory)
  (if (not (find-package "QUADROTOR_MSGS"))
    (make-package "QUADROTOR_MSGS"))
  (shadow 'Trajectory (find-package "QUADROTOR_MSGS")))
(unless (find-package "QUADROTOR_MSGS::TRAJECTORY")
  (make-package "QUADROTOR_MSGS::TRAJECTORY"))

(in-package "ROS")
;;//! \htmlinclude Trajectory.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*UNDEFINED*" (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(shadow '*UNDEFINED* (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(defconstant quadrotor_msgs::Trajectory::*UNDEFINED* 0)
(intern "*GENERAL*" (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(shadow '*GENERAL* (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(defconstant quadrotor_msgs::Trajectory::*GENERAL* 1)
(intern "*ACCELERATION*" (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(shadow '*ACCELERATION* (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(defconstant quadrotor_msgs::Trajectory::*ACCELERATION* 2)
(intern "*JERK*" (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(shadow '*JERK* (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(defconstant quadrotor_msgs::Trajectory::*JERK* 3)
(intern "*SNAP*" (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(shadow '*SNAP* (find-package "QUADROTOR_MSGS::TRAJECTORY"))
(defconstant quadrotor_msgs::Trajectory::*SNAP* 4)

(defun quadrotor_msgs::Trajectory-to-symbol (const)
  (cond
        ((= const 0) 'quadrotor_msgs::Trajectory::*UNDEFINED*)
        ((= const 1) 'quadrotor_msgs::Trajectory::*GENERAL*)
        ((= const 2) 'quadrotor_msgs::Trajectory::*ACCELERATION*)
        ((= const 3) 'quadrotor_msgs::Trajectory::*JERK*)
        ((= const 4) 'quadrotor_msgs::Trajectory::*SNAP*)
        (t nil)))

(defclass quadrotor_msgs::Trajectory
  :super ros::object
  :slots (_header _type _points ))

(defmethod quadrotor_msgs::Trajectory
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:type __type) 0)
    ((:points __points) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _type (round __type))
   (setq _points __points)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:type
   (&optional __type)
   (if __type (setq _type __type)) _type)
  (:points
   (&rest __points)
   (if (keywordp (car __points))
       (send* _points __points)
     (progn
       (if __points (setq _points (car __points)))
       _points)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8 _type
    1
    ;; quadrotor_msgs/TrajectoryPoint[] _points
    (apply #'+ (send-all _points :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8 _type
       (write-byte _type s)
     ;; quadrotor_msgs/TrajectoryPoint[] _points
     (write-long (length _points) s)
     (dolist (elem _points)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8 _type
     (setq _type (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; quadrotor_msgs/TrajectoryPoint[] _points
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _points (let (r) (dotimes (i n) (push (instance quadrotor_msgs::TrajectoryPoint :init) r)) r))
     (dolist (elem- _points)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get quadrotor_msgs::Trajectory :md5sum-) "18a34f2514fbc4cc1b109ed1c473a1d8")
(setf (get quadrotor_msgs::Trajectory :datatype-) "quadrotor_msgs/Trajectory")
(setf (get quadrotor_msgs::Trajectory :definition-)
      "# Trajectory type enums

# Undefined trajectory type
uint8 UNDEFINED=0

# General trajectory type that considers orientation from the pose and
# neglects heading values
uint8 GENERAL=1

# Trajectory types that compute orientation from acceleration and heading
# values and consider derivatives up to what is indicated by the name
uint8 ACCELERATION=2
uint8 JERK=3
uint8 SNAP=4

Header header

# Trajectory type as defined above
uint8 type

quadrotor_msgs/TrajectoryPoint[] points

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
MSG: quadrotor_msgs/TrajectoryPoint
duration time_from_start

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



(provide :quadrotor_msgs/Trajectory "18a34f2514fbc4cc1b109ed1c473a1d8")


