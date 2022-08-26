#!/usr/bin/env python
# license removed for brevity
import rospy       
import serial
import time
import csv
from drone_msgs.msg import Arduino
from nav_msgs.msg import Odometry
#from sensor_msgs.msg import Imu

#_imu  = Imu()
_odom = Odometry()
_data = Arduino()

def decoder(data):
    val = 0
    for idx in range(4):
        val *= 16
        if data[idx]>='0' and data[idx]<='9':
            val += ord(data[idx])-48
        elif data[idx]>='a' and data[idx]<='f':
            val += ord(data[idx]) - 97 + 10
        elif data[idx]>='A' and data[idx]<='F':
            val += ord(data[idx]) -65 + 10
        else:
            print("it's not an ascii code!")
            return 0
    return val

def force_sensor_pub(data):
    "publish IMU measurement"
    length = len(data)
    if length == 4:
        _data.diff_volt[0] = decoder(data[0:4]) / 13.0 - 1000.0
    elif length == 16:
        _data.diff_volt[0] = decoder(data[0:4]) / 13.0 - 1000.0
        _data.diff_volt[1] = decoder(data[4:8]) / 13.0 - 1000.0
        _data.diff_volt[2] = decoder(data[8:12]) / 13.0 - 1000.0
        _data.diff_volt[3] = decoder(data[12:16]) / 13.0 - 1000.0

#def imu_callback(data):
#    global _imu
#    _imu = data
    
def odom_callback(data):
    global _odom
    _odom = data
    
def listener():
    buff = serial.readline().decode()
    valid = False
    if buff:
        flag = buff[0]
        if flag == 's':
            force_sensor_pub(buff[1:-1])
            valid = True
        else:
            valid = False
            #print(buff)
    if valid:
        _data.header.stamp = rospy.Time.now()
        #pub.publish(_data)
    return valid
    
if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    serial = serial.Serial("/dev/ttyACM0",2000000,timeout=0.001)
    f = open('4300hz.csv', 'w')
    writer = csv.writer(f)
    rospy.Subscriber("/mavros/local_position/odom", Odometry, odom_callback)
    #rospy.Subscriber("/mavros/imu/data", Imu, imu_callback)
    rate = rospy.Rate(5000) # 100hz
    count = 0
    tic = time.time()
    while not rospy.is_shutdown():
        valid = listener()
        if valid:
            count = count+1
            writer.writerows([[time.time_ns(),
                               _data.diff_volt[0],_data.diff_volt[1],_data.diff_volt[2],_data.diff_volt[3],
                               _odom.pose.pose.position.x,_odom.pose.pose.position.y,_odom.pose.pose.position.z,
                               _odom.pose.pose.orientation.w,_odom.pose.pose.orientation.x,_odom.pose.pose.orientation.y,_odom.pose.pose.orientation.z,
                               _odom.twist.twist.linear.x,_odom.twist.twist.linear.y,_odom.twist.twist.linear.z,
                               _odom.twist.twist.angular.x,_odom.twist.twist.angular.x,_odom.twist.twist.angular.x]])
        if time.time() - tic > 1.0:
            print(count)
            count = 0
            tic = time.time()
        rate.sleep()
    serial.close()
