#!/usr/bin/env python

import rospy       
import serial
from drone_msgs.msg import Arduino

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

def ina226_pub(data):
    "publish airflow measurement"
    length = len(data)
    num_sensor = length/8
    _data.airflow_sensor_num = num_sensor
    for index in range(num_sensor):
        _data.current[index] = decoder(data[0+4*index:4+4*index]) / 3276.0
        _data.voltage[index] = decoder(data[4+4*index:8+4*index]) / 80.0

def imu_pub(data):
    "publish IMU measurement"
    length = len(data)
    if length != 44:
        print(data)
        print("[WARN] imu: The number of characters collected is not enough!")
        return
    else:
        _data.acc.x = (decoder(data[0:4]) - 32768) / 32768.0 * 16.0
        _data.acc.y = (decoder(data[4:8]) - 32768) / 32768.0 * 16.0
        _data.acc.z = (decoder(data[8:12]) - 32768) / 32768.0 * 16.0
        _data.gyro.x = (decoder(data[12:16]) - 32768) / 32768.0 * 2000.0
        _data.gyro.y = (decoder(data[16:20]) - 32768) / 32768.0 * 2000.0
        _data.gyro.z = (decoder(data[20:24]) - 32768) / 32768.0 * 2000.0
#        _data.mag.x = (decoder(data[24:28]) - 32768)
#        _data.mag.y = (decoder(data[28:32]) - 32768)
#        _data.mag.z = (decoder(data[32:36]) - 32768)
#        _data.eular_angle.x = (decoder(data[36:40]) - 32768) / 32768.0 * 180.0
#        _data.eular_angle.y = (decoder(data[40:44]) - 32768) / 32768.0 * 180.0
#        _data.eular_angle.z = (decoder(data[44:48]) - 32768) / 32768.0 * 180.0
        _data.quaternion.w = (decoder(data[24:28]) - 32768) / -32768.0
        _data.quaternion.x = (decoder(data[28:32]) - 32768) / -32768.0
        _data.quaternion.y = (decoder(data[32:36]) - 32768) / -32768.0
        _data.quaternion.z = (decoder(data[36:40]) - 32768) / -32768.0
        _data.baro = (decoder(data[40:44]) + 100000)

def force_sensor_pub(data):
    "publish IMU measurement"
    for i in range(int(len(data) / 4)):
        _data.diff_volt[i] = decoder(data[4*i:4*i+4]) / 13.0 - 1000.0
        print(_data.diff_volt[i])

def collect():
    while not rospy.is_shutdown():
        buff = serial.readline().decode()
        
        if buff:
            valid = True
            flag = buff[0]
            data = buff[1:-1]
            length = len(data)
            if length % 4 != 0:
                #print("[WARN] The number of data cannot be divided by 4! which is %d", len(data))
                #print(buff)
                valid = False
            if valid:
                if flag == 'n':
                    ina226_pub(data)
                elif flag == 'm':
                    imu_pub(data)
                elif flag == 's' and length == 4:
                    force_sensor_pub(data)
                else:
                    #print(buff)
                    #print("Unknown marker! n:airflow, m:imu, s:force, but what i got is:%s",flag)
                    pass
                _data.header.stamp = rospy.Time.now()
                pub.publish(_data)
            buff = []
            
        rate.sleep()
        
    serial.close()

if __name__=='__main__':
    pub = rospy.Publisher("/arduino", Arduino, queue_size=10)
    rospy.init_node('arduino_data_collect', anonymous=True)
    rate = rospy.Rate(4800) # 100hz
    serial = serial.Serial("/dev/ttyACM0",115200,timeout=0.1)
    _data = Arduino()
    try:
        collect()
    except rospy.ROSInterruptException:
        pass


