#!/usr/bin/env python
# license removed for brevity
import rospy      
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def image_callback(data):
	bridge = CvBridge()
	image = bridge.imgmsg_to_cv2(data,desired_encoding="passthrough")
	gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
	gray = np.uint8(gray)
	
	gray_msg = bridge.cv2_to_imgmsg(gray,encoding="mono8")
	depth_im_pub.publish(gray_msg)
	
if __name__ == '__main__':
	rospy.init_node('republisher', anonymous=True)
	rospy.Subscriber("/camera/depth/image_rect_raw", Image, image_callback)
	depth_im_pub = rospy.Publisher("/camera/depth", Image, queue_size=10)
	rate = rospy.Rate(100) # 100hz
	while not rospy.is_shutdown():
		rate.sleep()
