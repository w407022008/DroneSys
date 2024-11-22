
(cl:in-package :asdf)

(defsystem "drone_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :mavros_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "Arduino" :depends-on ("_package_Arduino"))
    (:file "_package_Arduino" :depends-on ("_package"))
    (:file "AttitudeReference" :depends-on ("_package_AttitudeReference"))
    (:file "_package_AttitudeReference" :depends-on ("_package"))
    (:file "Bspline" :depends-on ("_package_Bspline"))
    (:file "_package_Bspline" :depends-on ("_package"))
    (:file "ControlCommand" :depends-on ("_package_ControlCommand"))
    (:file "_package_ControlCommand" :depends-on ("_package"))
    (:file "ControlOutput" :depends-on ("_package_ControlOutput"))
    (:file "_package_ControlOutput" :depends-on ("_package"))
    (:file "DroneState" :depends-on ("_package_DroneState"))
    (:file "_package_DroneState" :depends-on ("_package"))
    (:file "DroneTarget" :depends-on ("_package_DroneTarget"))
    (:file "_package_DroneTarget" :depends-on ("_package"))
    (:file "Message" :depends-on ("_package_Message"))
    (:file "_package_Message" :depends-on ("_package"))
    (:file "PositionReference" :depends-on ("_package_PositionReference"))
    (:file "_package_PositionReference" :depends-on ("_package"))
    (:file "RCInput" :depends-on ("_package_RCInput"))
    (:file "_package_RCInput" :depends-on ("_package"))
  ))