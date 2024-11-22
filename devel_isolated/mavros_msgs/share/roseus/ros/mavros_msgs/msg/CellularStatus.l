;; Auto-generated. Do not edit!


(when (boundp 'mavros_msgs::CellularStatus)
  (if (not (find-package "MAVROS_MSGS"))
    (make-package "MAVROS_MSGS"))
  (shadow 'CellularStatus (find-package "MAVROS_MSGS")))
(unless (find-package "MAVROS_MSGS::CELLULARSTATUS")
  (make-package "MAVROS_MSGS::CELLULARSTATUS"))

(in-package "ROS")
;;//! \htmlinclude CellularStatus.msg.html


(defclass mavros_msgs::CellularStatus
  :super ros::object
  :slots (_status _failure_reason _type _quality _mcc _mnc _lac ))

(defmethod mavros_msgs::CellularStatus
  (:init
   (&key
    ((:status __status) 0)
    ((:failure_reason __failure_reason) 0)
    ((:type __type) 0)
    ((:quality __quality) 0)
    ((:mcc __mcc) 0)
    ((:mnc __mnc) 0)
    ((:lac __lac) 0)
    )
   (send-super :init)
   (setq _status (round __status))
   (setq _failure_reason (round __failure_reason))
   (setq _type (round __type))
   (setq _quality (round __quality))
   (setq _mcc (round __mcc))
   (setq _mnc (round __mnc))
   (setq _lac (round __lac))
   self)
  (:status
   (&optional __status)
   (if __status (setq _status __status)) _status)
  (:failure_reason
   (&optional __failure_reason)
   (if __failure_reason (setq _failure_reason __failure_reason)) _failure_reason)
  (:type
   (&optional __type)
   (if __type (setq _type __type)) _type)
  (:quality
   (&optional __quality)
   (if __quality (setq _quality __quality)) _quality)
  (:mcc
   (&optional __mcc)
   (if __mcc (setq _mcc __mcc)) _mcc)
  (:mnc
   (&optional __mnc)
   (if __mnc (setq _mnc __mnc)) _mnc)
  (:lac
   (&optional __lac)
   (if __lac (setq _lac __lac)) _lac)
  (:serialization-length
   ()
   (+
    ;; uint8 _status
    1
    ;; uint8 _failure_reason
    1
    ;; uint8 _type
    1
    ;; uint8 _quality
    1
    ;; uint16 _mcc
    2
    ;; uint16 _mnc
    2
    ;; uint16 _lac
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _status
       (write-byte _status s)
     ;; uint8 _failure_reason
       (write-byte _failure_reason s)
     ;; uint8 _type
       (write-byte _type s)
     ;; uint8 _quality
       (write-byte _quality s)
     ;; uint16 _mcc
       (write-word _mcc s)
     ;; uint16 _mnc
       (write-word _mnc s)
     ;; uint16 _lac
       (write-word _lac s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _status
     (setq _status (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _failure_reason
     (setq _failure_reason (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _type
     (setq _type (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _quality
     (setq _quality (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint16 _mcc
     (setq _mcc (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _mnc
     (setq _mnc (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _lac
     (setq _lac (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get mavros_msgs::CellularStatus :md5sum-) "a474bdb9df111b4e16fab4c29f7a6956")
(setf (get mavros_msgs::CellularStatus :datatype-) "mavros_msgs/CellularStatus")
(setf (get mavros_msgs::CellularStatus :definition-)
      "#Follows https://mavlink.io/en/messages/common.html#CELLULAR_STATUS specification

uint8 status
uint8 failure_reason
uint8 type
uint8 quality
uint16 mcc
uint16 mnc
uint16 lac
")



(provide :mavros_msgs/CellularStatus "a474bdb9df111b4e16fab4c29f7a6956")


