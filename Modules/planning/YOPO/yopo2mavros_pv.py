#!/usr/bin/env python3
# yopo_to_mavros_bridge.py (Position Control with Velocity Feedforward)

import rospy
from mavros_msgs.msg import PositionTarget # 导入 PositionTarget
from tf.transformations import quaternion_from_euler # 可能需要用于 yaw
import math

# 导入 YOPO 使用的自定义消息类型
from control_msg import PositionCommand # 这个保持不变

class YopoBridge:
    def __init__(self):
        rospy.init_node('yopo_to_mavros_bridge_pos_vel_ff', anonymous=True)

        self.target_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.yopo_sub = rospy.Subscriber('/yopo/pos_cmd', PositionCommand, self.yopo_cmd_callback) # 或 /so3_control/pos_cmd

        # 定义 type_mask
        # !! 不忽略 位置 (PX, PY, PZ) !!
        # !! 不忽略 速度 (VX, VY, VZ) 作为前馈 !!
        # !! 不忽略 偏航角 (Yaw) !!
        # 忽略 加速度/力 (AFX, AFY, AFZ) - 也可以不忽略，如有加速度
        # 忽略 偏航角速度 (Yaw Rate) - 因为我们用 Yaw
        self.type_mask = (
            # PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | # 不忽略位置
            # PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ | # 不忽略速度
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | # 忽略加速度
            # PositionTarget.IGNORE_YAW | # 不忽略 Yaw
            PositionTarget.IGNORE_YAW_RATE # 忽略 Yaw Rate
        )
        # 你也可以尝试不忽略加速度，如果你想把 PositionCommand.acceleration 也用上
        # self.type_mask &= ~(PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ)

        rospy.loginfo("YOPO to MAVROS pos control w/ vel FF bridge is running.")
        rospy.loginfo(f"Using type_mask: {self.type_mask}")

    def yopo_cmd_callback(self, msg):
        """
        将 PositionCommand 转换为 PositionTarget (位置控制+速度前馈) 并发布。
        """
        target_msg = PositionTarget()
        target_msg.header.stamp = rospy.Time.now()
        # 假设 odom 是 ENU (x东 y北 z上)，并且 MAVROS 会处理转换到 NED
        target_msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        target_msg.type_mask = self.type_mask

        # --- 填充所有要使用的字段 ---
        # 位置
        target_msg.position.x = msg.position.x
        target_msg.position.y = msg.position.y
        target_msg.position.z = 2

        # 速度 (作为前馈)
        target_msg.velocity.x = msg.velocity.x
        target_msg.velocity.y = msg.velocity.y
        target_msg.velocity.z = 0

        # 加速度 (如果 type_mask 没有忽略)
        # target_msg.acceleration_or_force.x = msg.acceleration.x
        # target_msg.acceleration_or_force.y = msg.acceleration.y
        # target_msg.acceleration_or_force.z = 0

        # 偏航角
        target_msg.yaw = msg.yaw
        # 确保 Yaw 在 -pi 到 pi 范围内 (如果需要)
        # target_msg.yaw = math.atan2(math.sin(msg.yaw), math.cos(msg.yaw))

        # 偏航角速度 (这个字段会被忽略)
        # target_msg.yaw_rate = msg.yaw_dot

        # --- (日志记录) ---
        rospy.loginfo_throttle(1.0,"----------------- CONVERTED TO -----------------")
        rospy.loginfo_throttle(1.0,"SENDING PositionTarget (Pos + Vel FF) to /mavros/setpoint_raw/local:")
        rospy.loginfo_throttle(1.0, f"Pos: [{target_msg.position.x:.2f}, {target_msg.position.y:.2f}, {target_msg.position.z:.2f}], Vel: [{target_msg.velocity.x:.2f}, {target_msg.velocity.y:.2f}, {target_msg.velocity.z:.2f}], Yaw: {target_msg.yaw:.2f}")


        self.target_pub.publish(target_msg)

if __name__ == '__main__':
    try:
        bridge = YopoBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
