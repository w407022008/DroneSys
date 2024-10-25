#!/usr/bin/env python
# license removed for brevity
import rospy    
import tf   

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

def odom_callback(data):
	print((rospy.Time.now()-data.header.stamp).to_sec())
	
def image_callback(data):
	print((rospy.Time.now()-data.header.stamp).to_sec())
	
if __name__ == '__main__':
	tf_broadcaster = tf.TransformBroadcaster()
	rospy.init_node('republisher', anonymous=True)
	#rospy.Subscriber("/vins_estimator/odometry", Odometry, odom_callback)
	rospy.Subscriber("/disparity", Image, image_callback)
	rate = rospy.Rate(100) # 100hz
	while not rospy.is_shutdown():
		rate.sleep()
