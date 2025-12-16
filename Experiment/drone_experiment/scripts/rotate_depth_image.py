#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthImageRotator:
    def __init__(self):
        rospy.init_node('depth_image_rotator', anonymous=True)
        
        # 获取参数
        self.input_topic = rospy.get_param("~input_topic", "/mav_isir/camera/depth/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "/mav_isir/camera/depth/image_raw_rotated")
        
        # 创建订阅者和发布者
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(self.input_topic, Image, self.image_callback)
        self.publisher = rospy.Publisher(self.output_topic, Image, queue_size=10)
        
        rospy.loginfo("DepthImageRotator initialized:")
        rospy.loginfo("  Input topic: %s", self.input_topic)
        rospy.loginfo("  Output topic: %s", self.output_topic)
        rospy.loginfo("  Rotation: 90 degrees counterclockwise")

    def image_callback(self, data):
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            # 将图像逆时针旋转90度
            rotated_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # 将处理后的图像转换回ROS消息
            rotated_msg = self.bridge.cv2_to_imgmsg(rotated_image, encoding=data.encoding)
            rotated_msg.header = data.header
            
            # 发布旋转后的图像
            self.publisher.publish(rotated_msg)
            
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        rotator = DepthImageRotator()
        rotator.run()
    except rospy.ROSInterruptException:
        pass