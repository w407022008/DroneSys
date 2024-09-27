import cv2
import numpy as np
import rospy
from collections import deque

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge

bridge = CvBridge()

def depth_callback(data):
	print((rospy.Time.now()-data.header.stamp))
	image = bridge.imgmsg_to_cv2(data)
	image_blur = cv2.GaussianBlur(image,(7,7),0)
	sobel_x = cv2.Sobel(image_blur,cv2.CV_64F,1,0,ksize=5)
	sobel_y = cv2.Sobel(image_blur,cv2.CV_64F,0,1,ksize=5)
	gradient_magnitude = np.sqrt(sobel_x**2,sobel_y**2)
	
	upper_bound = 6000
	lower_bound = 2000
	
	boundary_mask = np.zeros_like(image_blur,dtype=np.uint8)
	boundary_mask[gradient_magnitude >= upper_bound] = 255
	
	to_check = deque()
	for y in range(gradient_magnitude.shape[0]):
		for x in range(gradient_magnitude.shape[1]):
			if boundary_mask[y,x] == 255:
				to_check.append((y,x))
				
	neighbors = [(-1,-1), (-1,0), (-1,1),
								(0,-1),					(0,1),
								(1,-1), (1,0), (1,1)]
	while to_check:
		y, x = to_check.popleft()
		
		for dy,dx in neighbors:
			ny, nx = y+dy, x+dx
			if 0<= ny < gradient_magnitude.shape[0] and 0<= nx < gradient_magnitude.shape[1]:
				if (lower_bound <= gradient_magnitude[ny,nx] < upper_bound) and (boundary_mask[ny,nx]==0):
					boundary_mask[ny,nx]=255
					to_check.append((ny,nx))
	
	#filtered_image = image.copy()
	mask_inv = cv2.bitwise_not(boundary_mask)
	filtered_image = cv2.bitwise_and(image,image,mask=mask_inv)
	
	frame = cv2.Canny(cv2.GaussianBlur(boundary_mask,(9,9),0),300,400)
	
	for y in range(frame.shape[0]):
		for x in range(frame.shape[1]):
			if frame[y,x] == 255:
				to_check.append((y,x))
				
	while True and to_check:
		y, x = to_check.popleft()
		
		for dy,dx in neighbors:
			ny, nx = y+dy, x+dx
			if 0<= ny < gradient_magnitude.shape[0] and 0<= nx < gradient_magnitude.shape[1]:
				if (boundary_mask[ny,nx]==255):
					filtered_image[ny,nx] = filtered_image[y,x]
					boundary_mask[ny,nx]=0
					to_check.append((ny,nx))
	
	image_msg = bridge.cv2_to_imgmsg(filtered_image, encoding="mono16")
	image_msg.header.stamp = data.header.stamp
	depth_pub.publish(image_msg)
   
if __name__=='__main__':
	rospy.init_node('post_processing', anonymous=True)
	rospy.Subscriber("/sipeed_tof/depth", Image, depth_callback)
	depth_pub = rospy.Publisher('/sipeed_tof/depth_rect2', Image, queue_size=10)
	rate = rospy.Rate(100) # 100hz
	while not rospy.is_shutdown():
		rate.sleep()
