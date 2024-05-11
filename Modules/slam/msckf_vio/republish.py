#!/usr/bin/env python3

# Import pyserial modules
import rospy
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

i=True
def point_callback(data):
	data.header.frame_id="world"
	data.header.stamp = rospy.get_rostime()
	global i,a,b,c
	if i:
		a=data.point.x
		b=data.point.y
		c=data.point.z
		i=False
	data.point.x-=a
	data.point.y-=b
	data.point.z-=c
	data.point.x*=-1
	data.point.y*=-1
	point_pub.publish(data)
    
def Transform_callback(data):
	pose_ = PoseStamped()
	pose_.header.frame_id="world"
	pose_.header.stamp = rospy.get_rostime()
	pose_.pose.position.x = data.transform.translation.x
	pose_.pose.position.y = data.transform.translation.y
	pose_.pose.position.z = data.transform.translation.z
	pose_.pose.orientation = data.transform.rotation
	pose_pub.publish(pose_)
    
if __name__ == "__main__":

	rospy.init_node("nh") # initialize node
	rospy.Subscriber("/leica/position", PointStamped, point_callback)
	rospy.Subscriber("/vicon/firefly_sbx/firefly_sbx", TransformStamped, Transform_callback)

	pose_pub = rospy.Publisher("/gt_odom", PoseStamped, queue_size=10)
	point_pub = rospy.Publisher("/gt_point", PointStamped, queue_size=10)

	while not rospy.is_shutdown():
		pass
        
