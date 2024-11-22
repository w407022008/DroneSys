#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import numpy as np

class DepthPub:
  def __init__(self):
    rospy.init_node('pcl_depth_pub', anonymous=True)
    self.depth_sub = rospy.Subscriber('/ddk/rgbd/depth/image_raw', Image, self.callback)
    self.pcl_depth_pub = rospy.Publisher('/pcl_render_node/depth', Image, queue_size=1000)
    self.odom_sub = rospy.Subscriber('/ddk/ground_truth/odom', Odometry, self.odomcallback)
    #self.sensor_odom_pub = rospy.Publisher('pcl_render_node/sensor_pose', PoseStamped, queue_size = 1000)
  def callback(self, data):
    depth_msg = data
    depth_msg.header.frame_id = "SQ01s/camera"
    self.pcl_depth_pub.publish(depth_msg)

  def odomcallback(self, odom_data):
    pose_receive = np.identity(4)
    # request_position = np.array()
    pose_stamped_msg = PoseStamped()
    pose_stamped_msg.header = odom_data.header
    pose_stamped_msg.pose = odom_data.pose.pose

    #self.sensor_odom_pub.publish(pose_stamped_msg)


if __name__ == '__main__':
  cam2body = np.array([[0.0, 0.0, 1.0, 0.0], 
                      [-1.0, 0.0, 0.0, 0.0], 
                      [0.0, -1.0, 0.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0]])
  
  
  try:
    pcl_depth =  DepthPub() 
    rospy.spin()
  except rospy.ROSInterruptException:
    pass      
