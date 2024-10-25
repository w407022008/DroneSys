#!/usr/bin/env python
# license removed for brevity
import rospy    
import tf   

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

def odom_callback(data):
	transform_msg = TransformStamped()
	transform_msg.header.stamp = data.header.stamp
	transform_msg.header.frame_id = 'world'
	transform_msg.child_frame_id = 'base_link'
	transform_msg.transform.translation.x = data.pose.pose.position.x
	transform_msg.transform.translation.y = data.pose.pose.position.y
	transform_msg.transform.translation.z = data.pose.pose.position.z
	transform_msg.transform.rotation.x = data.pose.pose.orientation.x
	transform_msg.transform.rotation.y = data.pose.pose.orientation.y
	transform_msg.transform.rotation.z = data.pose.pose.orientation.z
	transform_msg.transform.rotation.w = data.pose.pose.orientation.w
	tf_broadcaster.sendTransformMessage(transform_msg)
	print("transform")
	
if __name__ == '__main__':
	tf_broadcaster = tf.TransformBroadcaster()
	rospy.init_node('republisher', anonymous=True)
	rospy.Subscriber("/odometry", Odometry, odom_callback)
	rate = rospy.Rate(100) # 100hz
	while not rospy.is_shutdown():
		rate.sleep()
