"""
 * File: offb_node.py
 * Stack and tested in Gazebo Classic 9 SITL
"""

#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

# 监听 YOPO 控制指令
from control_msg import PositionCommand

current_state = State()


def state_cb(msg):
    global current_state
    current_state = msg


def yopo_cmd_cb(msg):
    """
    一旦收到 /yopo/pos_cmd（YOPO 开始发布控制指令），
    立即结束 takeoff.py 节点，不再发送悬停 setpoint。
    """
    rospy.loginfo("Detected /yopo/pos_cmd. Shutting down takeoff node.")
    rospy.signal_shutdown("YOPO control started, stop takeoff node.")


if __name__ == "__main__":
    rospy.init_node("offb_node_py")

    # MAVROS 状态订阅
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    # 悬停 setpoint 发布到本地位置控制
    local_pos_pub = rospy.Publisher(
        "mavros/setpoint_position/local", PoseStamped, queue_size=10
    )

    # 监听 YOPO 的 PositionCommand，一旦有消息就关掉本节点
    yopo_sub = rospy.Subscriber(
        "/yopo/pos_cmd", PositionCommand, yopo_cmd_cb, queue_size=1
    )

    # arming / 模式切换服务
    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    rospy.loginfo("Waiting for FCU connection...")
    while (not rospy.is_shutdown()) and (not current_state.connected):
        rate.sleep()
    rospy.loginfo("FCU connected.")

    pose = PoseStamped()
    pose.pose.position.x = 0.0
    pose.pose.position.y = 0.0
    pose.pose.position.z = 2.0  # 起飞高度（你可以改成 2.0）

    # Send a few setpoints before starting
    rospy.loginfo("Sending initial setpoints...")
    for i in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = "OFFBOARD"

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    rospy.loginfo("Attempting to switch to OFFBOARD and arm...")
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        # 先切 OFFBOARD
        if (
            current_state.mode != "OFFBOARD"
            and (now - last_req) > rospy.Duration(1.0)
        ):
            if set_mode_client.call(offb_set_mode).mode_sent:
                rospy.loginfo("OFFBOARD enabled")
            last_req = now
        # 再 arm
        elif (
            not current_state.armed
            and (now - last_req) > rospy.Duration(1.0)
        ):
            if arming_client.call(arm_cmd).success:
                rospy.loginfo("Vehicle armed")
            last_req = now

        # 持续发布悬停位置（直到：
        # 1）你手动 Ctrl+C，或者
        # 2）收到 /yopo/pos_cmd -> yopo_cmd_cb -> signal_shutdown）
        local_pos_pub.publish(pose)
        rate.sleep()
