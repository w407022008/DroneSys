#!/usr/bin/env python
# license removed for brevity
import rospy 
import numpy as np
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped

_pos=np.array([0.0, 0.0, 0.0])
_obs=np.array([0.0, 0.0, 0.0])
p0=np.array([0.0, 0.0])
p1=np.array([0.0, 1.3])
p2=np.array([0.0,-1.3])
p3=np.array([-1.3,0.0])
p4=np.array([1.3, 0.0])
p5=np.array([1.2-1.2])
p6=np.array([-1.2,-1.2])
p7=np.array([1.2, 1.2])
p8=np.array([-1.2, 1.2])

def obs_callback(data):
    _obs[0] = data.point.x
    _obs[1] = data.point.y
    _obs[2] = data.point.z
    if(np.linalg.norm(_pos)!=0):
        dist = np.linalg.norm(_pos-_obs)
        print("cur_pos: ",_pos)
        print("osb_pos: ",_obs)
        print("minimum distance: ", dist)

def pose_callback(data):
    _pos[0] = data.pose.position.x
    _pos[1] = data.pose.position.y
    _pos[2] = data.pose.position.z
    
    
if __name__ == '__main__':
    rospy.init_node('safty_check', anonymous=True)
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_callback)
    rospy.Subscriber("/histo_planner/closest", PointStamped, obs_callback)
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        # d = 10.0
        # d0 = np.linalg.norm(_pos-p0)
        # if(d > d0):
        #     d = d0
        # d1 = np.linalg.norm(_pos-p1)
        # if(d > d1):
        #     d = d1
        # d2 = np.linalg.norm(_pos-p2)
        # if(d > d2):
        #     d = d2
        # d3 = np.linalg.norm(_pos-p3)
        # if(d > d3):
        #     d = d3
        # d4 = np.linalg.norm(_pos-p4)
        # if(d > d4):
        #     d = d4
        # d5 = np.linalg.norm(_pos-p5)
        # if(d > d5):
        #     d = d5
        # d6 = np.linalg.norm(_pos-p6)
        # if(d > d6):
        #     d = d6
        # d7 = np.linalg.norm(_pos-p7)
        # if(d > d7):
        #     d = d7
        # d8 = np.linalg.norm(_pos-p8)
        # if(d > d8):
        #     d = d8
        # if(d<0.3):
        #     print("========= coli =========== ", d, " < 0.3m")
        # else:
        #     print("minimum distance: ", d, " > 0.3m")

        rate.sleep()

