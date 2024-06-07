#!/usr/bin/env python3

# Import pyserial modules
import serial
import serial.tools.list_ports
import struct
import math
import platform
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from tf.transformations import quaternion_from_euler

# Find ttyUSB* devices
def find_ttyUSB():
    posts = [port.device for port in serial.tools.list_ports.comports() if 'USB' in port.device]
    print('There are {} {} serial devices currently connected to the computer: {}'.format(len(posts), 'USB', posts))

# Find ttyUSB* devices
def printList():
    # Fetch/Detect all available COM ports
    ports_list = list(serial.tools.list_ports.comports())
    # List all the COM ports
    if len(ports_list) <= 0:
        print("There are currently no serial devices available.")
    else:
        print("The available serial devices are :")
        for comport in ports_list:
            print(list(comport)[0], list(comport)[1]) 

# Check whether the data structure is compliant, if it is compliant, keep it, otherwise discard it
def checkSum(list_data, check_data):
    return sum(list_data) & 0xff == check_data

# Convert hexadecimal integer to floating point number in short form
def hex_to_short(raw_data):
    return list(struct.unpack("hhhh", bytearray(raw_data)))

sign = lambda x: math.copysign(1,x)

# Data processing
def handleSerialData(raw_data):
    
    # declare global variables
    global buff, key, angle_degree, magnetometer, acceleration, angularVelocity, pub_flag, quaternion
    
    angle_flag=False
    
    buff[key] = raw_data # Fill data into dictionary
    key += 1
    
    # All data must start with 0x55, if not reset the dictionary
    if buff[0] != 0x55:
        key = 0
        return
    
    # Until the dictionary is filled with 11 data, the next step is processed
    if key < 11:  
        return
    else:
        data_buff = list(buff.values())
        
        # 51 52 53 54 59 correspond to different data
        if buff[1] == 0x51 :
            if checkSum(data_buff[0:10], data_buff[10]):
                acceleration = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 16 * 9.8 for i in range(0, 3)]
            else:
                print('0x51 Verification failed')
                
        elif buff[1] == 0x52:
            if checkSum(data_buff[0:10], data_buff[10]):
                angularVelocity = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 2000 * math.pi / 180 for i in range(0, 3)]

            else:
                print('0x52 Verification failed')    
                
        elif buff[1] == 0x53:
            if checkSum(data_buff[0:10], data_buff[10]):
                angle_degree = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 180 for i in range(0, 3)]
                angle_flag = True
            else:
                print('0x53 Verification failed')
        
        elif buff[1] == 0x54:
            if checkSum(data_buff[0:10], data_buff[10]):
                magnetometer = hex_to_short(data_buff[2:10])
            else:
                print('0x54 Verification failed')
                
        elif buff[1] ==0x59:
            if checkSum(data_buff[0:10], data_buff[10]):
                quaternion = [hex_to_short(data_buff[2:10])[i] / 32768.0 for i in range(0, 4)]
                angle_flag = True
            else:
                print('0x59 Verification failed')  
            
        else:
            print("The data processing class does not provide parsing of the " + str(buff[1]) )
            print("or data error")
            buff = {}
            key = 0
        
        buff = {}
        key = 0
        
        if angle_flag:
            stamp = rospy.get_rostime()

            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = "joystickIMU"
            
            mag_msg.header.stamp = stamp
            mag_msg.header.frame_id = "joystickIMU"
            
            angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
            qua = quaternion_from_euler(angle_radian[0], angle_radian[1], angle_radian[2])
            
            imu_msg.orientation.x = qua[0]
            imu_msg.orientation.y = qua[1]
            imu_msg.orientation.z = qua[2]
            imu_msg.orientation.w = qua[3]
            
#            imu_msg.orientation.x =  quaternion[1]
#            imu_msg.orientation.y =  quaternion[2]
#            imu_msg.orientation.z =  quaternion[3]
#            imu_msg.orientation.w =  quaternion[0]
            
            
            imu_msg.angular_velocity.x = angularVelocity[0]
            imu_msg.angular_velocity.y = angularVelocity[1]
            imu_msg.angular_velocity.z = angularVelocity[2]

            imu_msg.linear_acceleration.x = acceleration[0]
            imu_msg.linear_acceleration.y = acceleration[1]
            imu_msg.linear_acceleration.z = acceleration[2]
            
            imu_pub.publish(imu_msg)
            
            mag_msg.magnetic_field.x = magnetometer[0]
            mag_msg.magnetic_field.y = magnetometer[1]
            mag_msg.magnetic_field.z = magnetometer[2]
            
            mag_pub.publish(mag_msg)

key = 0  
flag = 0 
buff = {}
angularVelocity = [0, 0, 0]
acceleration = [0, 0, 0]
magnetometer = [0, 0, 0]
angle_degree = [0, 0, 0] 
quaternion = [0,0,0,0]
nombreIteration = 0

if __name__ == "__main__":

    printList()
    
    rospy.init_node("imu") # initialize node
    
    port = '/dev/rfcomm0' # port name
    baudrate = 921600       # baud rate
    
    print("IMU Type: Normal Port:%s baud:%d" %(port,baudrate)) 
    imu_msg = Imu()
    mag_msg = MagneticField()
    
    try:
        
        wt_imu = serial.Serial(port=port, baudrate=baudrate, timeout=0.5)
        if wt_imu.isOpen(): # Check if the port is open
            rospy.loginfo("\033[32mSerial port opened successfully...\033[0m")
        else:
            wt_imu.open() # If the port is not open then open the port
            rospy.loginfo("\033[32mSerial port opened successfully...\033[0m")
    except Exception as e:
        print(e) 
        rospy.loginfo("\033[31mSerial port open failed\033[0m")
        exit(0)
    else:
        imu_pub = rospy.Publisher("wit/imu", Imu, queue_size=10)
        mag_pub = rospy.Publisher("wit/mag", MagneticField, queue_size=10)
        
        while not rospy.is_shutdown():
            try:
                buff_count = wt_imu.inWaiting() # Metering the number of bytes in the cache
            except Exception as e: 
                print("exception:" + str(e))
                print("imu lost connection, poor contact, or broken wire")
                exit(0)
            else:
                if buff_count > 0: # If there is data in the cache
                    buff_data = wt_imu.read(buff_count) # read data
                    for i in range(0, buff_count): # Data processing
                        handleSerialData(buff_data[i])
        wt_imu.close()  # close port
        