#!/usr/bin/env python3

import rospy
from quadrotor_msgs.msg import PositionCommand
from drone_msgs.msg import ControlCommand

class CommandConverter:
    def __init__(self):
        rospy.init_node('command_converter', anonymous=True)
        self.id = 0
        self.sub = rospy.Subscriber('/planning/pos_cmd', PositionCommand, self.callback)
        self.pub = rospy.Publisher('/drone_msg/control_command', ControlCommand, queue_size=10)

    def callback(self, pos_cmd_msg):
        control_command_msg = ControlCommand()

        control_command_msg.header = pos_cmd_msg.header
        control_command_msg.Command_ID = self.id
        control_command_msg.Mode = control_command_msg.Move
        control_command_msg.Reference_State.Move_mode = control_command_msg.Reference_State.XYZ_POS_VEL
        control_command_msg.Reference_State.Move_frame = control_command_msg.Reference_State.ENU_FRAME

        control_command_msg.Reference_State.position_ref[0] = pos_cmd_msg.position.x
        control_command_msg.Reference_State.position_ref[1] = pos_cmd_msg.position.y
        control_command_msg.Reference_State.position_ref[2] = pos_cmd_msg.position.z
        control_command_msg.Reference_State.velocity_ref[0] = pos_cmd_msg.velocity.x
        control_command_msg.Reference_State.velocity_ref[1] = pos_cmd_msg.velocity.y
        control_command_msg.Reference_State.velocity_ref[2] = pos_cmd_msg.velocity.z
        control_command_msg.Reference_State.acceleration_ref[0] = pos_cmd_msg.acceleration.x
        control_command_msg.Reference_State.acceleration_ref[1] = pos_cmd_msg.acceleration.y
        control_command_msg.Reference_State.acceleration_ref[2] = pos_cmd_msg.acceleration.z
        control_command_msg.Reference_State.yaw_ref = pos_cmd_msg.yaw
        control_command_msg.Reference_State.yaw_rate_ref = pos_cmd_msg.yaw_dot

        print("published")
        self.pub.publish(control_command_msg)
        self.id = self.id + 1

if __name__ == '__main__':
    try:
        converter = CommandConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

