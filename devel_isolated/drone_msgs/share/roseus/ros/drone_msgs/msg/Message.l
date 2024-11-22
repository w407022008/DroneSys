;; Auto-generated. Do not edit!


(when (boundp 'drone_msgs::Message)
  (if (not (find-package "DRONE_MSGS"))
    (make-package "DRONE_MSGS"))
  (shadow 'Message (find-package "DRONE_MSGS")))
(unless (find-package "DRONE_MSGS::MESSAGE")
  (make-package "DRONE_MSGS::MESSAGE"))

(in-package "ROS")
;;//! \htmlinclude Message.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(intern "*NORMAL*" (find-package "DRONE_MSGS::MESSAGE"))
(shadow '*NORMAL* (find-package "DRONE_MSGS::MESSAGE"))
(defconstant drone_msgs::Message::*NORMAL* 0)
(intern "*WARN*" (find-package "DRONE_MSGS::MESSAGE"))
(shadow '*WARN* (find-package "DRONE_MSGS::MESSAGE"))
(defconstant drone_msgs::Message::*WARN* 1)
(intern "*ERROR*" (find-package "DRONE_MSGS::MESSAGE"))
(shadow '*ERROR* (find-package "DRONE_MSGS::MESSAGE"))
(defconstant drone_msgs::Message::*ERROR* 2)

(defun drone_msgs::Message-to-symbol (const)
  (cond
        ((= const 0) 'drone_msgs::Message::*NORMAL*)
        ((= const 1) 'drone_msgs::Message::*WARN*)
        ((= const 2) 'drone_msgs::Message::*ERROR*)
        (t nil)))

(defclass drone_msgs::Message
  :super ros::object
  :slots (_header _message_type _source_node _content ))

(defmethod drone_msgs::Message
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:message_type __message_type) 0)
    ((:source_node __source_node) "")
    ((:content __content) "")
    )
   (send-super :init)
   (setq _header __header)
   (setq _message_type (round __message_type))
   (setq _source_node (string __source_node))
   (setq _content (string __content))
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
  (:source_node
   (&optional __source_node)
   (if __source_node (setq _source_node __source_node)) _source_node)
  (:content
   (&optional __content)
   (if __content (setq _content __content)) _content)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint8 _message_type
    1
    ;; string _source_node
    4 (length _source_node)
    ;; string _content
    4 (length _content)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint8 _message_type
       (write-byte _message_type s)
     ;; string _source_node
       (write-long (length _source_node) s) (princ _source_node s)
     ;; string _content
       (write-long (length _content) s) (princ _content s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint8 _message_type
     (setq _message_type (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; string _source_node
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _source_node (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _content
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _content (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get drone_msgs::Message :md5sum-) "298ffdf82be3ca999f3a78d890347d59")
(setf (get drone_msgs::Message :datatype-) "drone_msgs/Message")
(setf (get drone_msgs::Message :definition-)
      "std_msgs/Header header

## message_type
uint8 message_type
# enum 
uint8 NORMAL = 0  
uint8 WARN   = 1  
uint8 ERROR  = 2  
 
string source_node

string content
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



(provide :drone_msgs/Message "298ffdf82be3ca999f3a78d890347d59")


