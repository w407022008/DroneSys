#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import sys
import select
import tty
import termios

def get_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(sys.stdin.fileno())
        input_char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN, old_settings)
    return input_char

def keyboard_publisher():
    rospy.init_node('keyboard_input_node',anonymous=True)
    pub = rospy.Publisher('/keyboard/control',String, queue_size=10)

    rospy.loginfo("Type in any to pub. 'q' to exit")

    while not rospy.is_shutdown():
        char = get_char()
        if char == 'q':
            break
        msg = String(data=char)
        pub.publish(msg)
        rospy.loginfo(f"publish:{char}")

if __name__ == '__main__':
    try:
        keyboard_publisher()
    except rospy.ROSInterruptException:
        pass