#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque

class StereoSGBMNode:
    def __init__(self):
        rospy.init_node('stereo_sgbm_node', anonymous=True)
        
        self.disparity_pub = rospy.Publisher('/disparity', Image, queue_size=10)

        # 设置图像订阅者
        self.left_image_sub = rospy.Subscriber('/camera/infra1/image_rect_raw', Image, self.left_image_callback)
        self.right_image_sub = rospy.Subscriber('/camera/infra2/image_rect_raw', Image, self.right_image_callback)

        self.bridge = CvBridge()
        self.left_queue = deque(maxlen=10)
        self.right_queue = deque(maxlen=10)

        # 创建 StereoBM 对象
        nDispFactor=2
        self.stereoBM = cv2.StereoBM_create(numDisparities=16*nDispFactor, blockSize=21)

        # 创建 StereoSGBM 对象
        window_size = 11
        num_image_channel = 1
        min_disp = 0
        num_disp = 16*nDispFactor-min_disp
        self.stereoSGBM = cv2.StereoSGBM_create(
            minDisparity=min_disp,		# minimum possible disparity value
            numDisparities=num_disp,		# maximun disparity - minimum. >0 %16==0
            blockSize=window_size,		# odd [3~11]
            P1=8 * num_image_channel * window_size ** 2,	# control disparity smoothness. 
            P2=32 * num_image_channel * window_size ** 2,
            disp12MaxDiff=0,			# maximum allowned difference in the left-right disparity check, >0 unless disable
            uniquenessRatio=10,		# margin in % by which the best(min) cost value win the second best. 5~15
            speckleWindowSize=200,		# maximum size of smooth dispatiry regions as noise speckles and invalidate. 50~200 or 0 to disable
            speckleRange=16*2,		# maximum disparity variation within each connected component. 1/2
            preFilterCap=63,			# truncation value for the prefiltered image pixels by computing x-derivative
            mode=cv2.STEREO_SGBM_MODE_SGBM
        )
        
    def left_image_callback(self, msg):
        left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.left_queue.append((left_image,msg.header.stamp))
        if self.right_queue:
            self.process_images()

    def right_image_callback(self, msg):
        right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.right_queue.append((right_image,msg.header.stamp))

    def process_images(self):
        left_image, left_stamp = self.left_queue[0]
        right_image, right_stamp = self.right_queue[0]
        if abs((left_stamp - right_stamp).to_nsec())<1e-3:
            # 计算视差图
            #disparity = self.stereoBM.compute(left_image, right_image)
            disparity = self.stereoSGBM.compute(left_image, right_image)

            # 归一化视差图以便显示
            #disparity_norm = (255*(disparity + 16) / (496+16)).astype(np.int8)
            disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # 显示结果
            #cv2.imshow('Disparity Map', disparity_norm)
            #cv2.waitKey(1)  # 等待1毫秒
            
            # 发布ROS消息
            disparity_msg = self.bridge.cv2_to_imgmsg(disparity_norm, encoding="mono8")
            disparity_msg.header.stamp = left_stamp
            self.disparity_pub.publish(disparity_msg)
            
            # 从队列中移除
            self.left_queue.popleft()
            self.right_queue.popleft()

if __name__ == '__main__':
    try:
        node = StereoSGBMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
