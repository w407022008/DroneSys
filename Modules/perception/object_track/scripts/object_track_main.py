#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultralytics YOLO 🚀, AGPL-3.0 license

通过PID控制器计算无人机在x、y、z方向的速度，
并通过MAVROS的topic将速度指令发布给PX4飞控系统

控制目标：
    1. "目标像素框的中心点的横坐标"  在图像的中心位置
    2. "目标像素框的高度"  是特定值（例如10000像素方）

控制量：
    针对控制目标1： yaw方向加速度enu_wx
    针对控制目标2： 水平面的速度enu_vx, enu_vy（光轴方向的速度投影到水平面内的分量）
"""

import cv2
import torch
import os
import yaml
from ultralytics import YOLO
import numpy as np
import sys
import json
import threading
import time
import math

# 导入鼠标目标选择器
from mouse_target_selector import MouseTargetSelector
from tracker import SingleObjectBotSortTracker, SingleObjectByteTrackTracker
from control_system import ControlSystem, apply_filter

# 尝试导入ROS模块
ROS_AVAILABLE = False
try:
    import rospy
    from std_msgs.msg import String
    from geometry_msgs.msg import TwistStamped, PoseStamped
    from sensor_msgs.msg import Image, Imu
    from mavros_msgs.msg import PositionTarget, AttitudeTarget
    import std_msgs.msg
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
    print("ROS模块导入成功")
except ImportError as e:
    print(f"ROS模块导入失败: {e}")
    print("此脚本需要在ROS环境中运行")


# ========================
# 全局变量
# ========================
custom_tracker = None
control_system = ControlSystem()  # 延迟初始化，在获取到相机分辨率后初始化
bridge = None
model = None
target_info_pub = None
velocity_pub = None
attitude_pub = None
image_sub = None
attitude_sub = None
running = True
model_loaded = False

# USB相机配置
use_usb_camera = True
usb_camera_device = "/dev/video2"
usb_camera = None

# 目标类别过滤配置
target_classes = None
config_file = "detection_config.yaml"

# 图像参数（将在相机初始化时更新）
image_width = 480
image_height = 640

# 创建鼠标目标选择器实例
mouse_selector = MouseTargetSelector()

# 无人机姿态参数
current_roll = 0.0
current_pitch = 0.0
current_yaw = 0.0


# COCO数据集80个类别名称，用于配置文件中的类别名到索引的映射
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 四元数乘法函数
def quaternion_multiply(q1, q2):
    """
    计算两个四元数的乘积
    
    Args:
        q1 (tuple): 第一个四元数 (x, y, z, w)
        q2 (tuple): 第二个四元数 (x, y, z, w)
        
    Returns:
        tuple: 乘积四元数 (x, y, z, w)
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return (x, y, z, w)

def publish_velocity_command(velocity_x, velocity_y, velocity_z, angular_x=0.0, angular_y=0.0, angular_z=0.0):
    """
    通过ROS发布速度和角速度控制指令给PX4飞控系统
    
    Args:
        velocity_x (float): X方向速度（东向）
        velocity_y (float): Y方向速度（北向）
        velocity_z (float): Z方向速度（天向）
        angular_x (float): 绕X轴角速度（默认0.0）
        angular_y (float): 绕Y轴角速度（默认0.0）
        angular_z (float): 绕Z轴角速度（默认0.0）
    """
    # 如果ROS不可用或没有发布者，直接返回
    if not ROS_AVAILABLE or velocity_pub is None:
        return
        
    try:
        # 创建PositionTarget消息用于setpoint_raw/local
        position_msg = PositionTarget()
        position_msg.header.stamp = rospy.Time.now()
        position_msg.header.frame_id = "map"  # 使用map坐标系
        
        # 设置坐标系和类型掩码
        # http://docs.ros.org/api/mavros_msgs/html/msg/PositionTarget.html
        position_msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        # 忽略位置，只控制速度和偏航角速率
        position_msg.type_mask = (PositionTarget.IGNORE_PX | 
                                 PositionTarget.IGNORE_PY | 
                                 PositionTarget.IGNORE_PZ |
                                 PositionTarget.IGNORE_AFX |
                                 PositionTarget.IGNORE_AFY |
                                 PositionTarget.IGNORE_AFZ |
                                 PositionTarget.IGNORE_YAW)
        
        # PX4飞控使用的是ENU坐标系（东-北-天），但在MAVLink中使用NED，需要转换
        position_msg.velocity.x = velocity_x  # 东向速度
        position_msg.velocity.y = velocity_y  # 北向速度
        position_msg.velocity.z = velocity_z  # 天向速度
        
        # 设置偏航角速率
        position_msg.yaw_rate = angular_z
        
        # 发布位置控制指令
        velocity_pub.publish(position_msg)
        print(f"已发布位置控制指令: VX={velocity_x:.2f}, VY={velocity_y:.2f}, VZ={velocity_z:.2f}m/s, WZ={angular_z:.2f} rad/s")
    except Exception as e:
        print(f"发布位置控制指令时出错: {e}")

def publish_attitude_command(roll, pitch, yaw, thrust):
    """
    通过ROS发布姿态控制指令给PX4飞控系统
    
    Args:
        roll (float): 滚转角（弧度）
        pitch (float): 俯仰角（弧度）
        yaw (float): 偏航角（弧度）
        thrust (float): 推力（0-1）
    """
    # 如果ROS不可用或没有发布者，直接返回
    if not ROS_AVAILABLE or attitude_pub is None:
        return
        
    try:
        # 创建AttitudeTarget消息用于setpoint_raw/attitude
        attitude_msg = AttitudeTarget()
        attitude_msg.header.stamp = rospy.Time.now()
        attitude_msg.header.frame_id = "map"
        
        # 设置类型掩码，控制姿态和推力
        # http://docs.ros.org/api/mavros_msgs/html/msg/AttitudeTarget.html
        attitude_msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                                AttitudeTarget.IGNORE_PITCH_RATE | \
                                AttitudeTarget.IGNORE_YAW_RATE
        
        # 尝试使用tf库进行四元数转换
        try:
            import tf.transformations
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        except ImportError:
            # 如果tf库不可用，手动实现四元数转换
            # 参考 https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)
            
            quaternion = [
                sr * cp * cy - cr * sp * sy,  # x
                cr * sp * cy + sr * cp * sy,  # y
                cr * cp * sy - sr * sp * cy,  # z
                cr * cp * cy + sr * sp * sy   # w
            ]
        
        attitude_msg.orientation.x = quaternion[0]
        attitude_msg.orientation.y = quaternion[1]
        attitude_msg.orientation.z = quaternion[2]
        attitude_msg.orientation.w = quaternion[3]
        
        # 设置推力
        attitude_msg.thrust = thrust
        
        # 发布姿态控制指令
        attitude_pub.publish(attitude_msg)
        print(f"已发布姿态控制指令: Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}, Thrust={thrust:.2f}")
    except Exception as e:
        print(f"发布姿态控制指令时出错: {e}")

def extract_tracking_info(result):
    """
    从YOLOv8追踪结果中提取关键信息
    
    Args:
        result: YOLOv8追踪结果对象
        
    Returns:
        list: 包含边界框、ID、置信度、中心点位置和高度的字典列表
    """
    tracking_data = []
    
    # 检查是否有检测到的目标
    if result.boxes is not None:
        boxes = result.boxes
        
        # 获取边界框坐标 (xywh格式: 中心点x, 中心点y, 宽度, 高度)
        # 处理可能是numpy数组或PyTorch张量的情况
        if hasattr(boxes.xywh, 'cpu'):
            bounding_boxes = boxes.xywh.cpu().numpy()
        else:
            bounding_boxes = boxes.xywh.numpy() if hasattr(boxes.xywh, 'numpy') else boxes.xywh
        
        # 获取目标ID（用于追踪）- 如果是predict结果，则没有id属性
        object_ids = []
        if hasattr(boxes, 'id') and boxes.id is not None:
            # 处理可能是numpy数组或PyTorch张量的情况
            if hasattr(boxes.id, 'cpu'):
                object_ids = boxes.id.int().cpu().tolist()
            else:
                object_ids = boxes.id.tolist() if hasattr(boxes.id, 'tolist') else boxes.id
        else:
            # 如果没有追踪ID，则为每一帧生成临时ID（从0开始）
            object_ids = list(range(len(bounding_boxes)))
        
        # 获取置信度分数
        # 处理可能是numpy数组或PyTorch张量的情况
        if hasattr(boxes.conf, 'cpu'):
            confidence_scores = boxes.conf.cpu().numpy()
        else:
            confidence_scores = boxes.conf.numpy() if hasattr(boxes.conf, 'numpy') else boxes.conf
        
        # 计算目标中心位置、宽度和高度
        # xyxy格式: 左上角x, 左上角y, 右下角x, 右下角y
        # 处理可能是numpy数组或PyTorch张量的情况
        if hasattr(boxes.xyxy, 'cpu'):
            xyxy = boxes.xyxy.cpu().numpy()
        else:
            xyxy = boxes.xyxy.numpy() if hasattr(boxes.xyxy, 'numpy') else boxes.xyxy
        
        # 计算中心点坐标 (左上角和右下角坐标的平均值)
        center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
        center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
        
        # 计算底部中心点坐标 (x保持不变，y设为底部)
        bottom_center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
        bottom_center_y = xyxy[:, 3]  # 底部y坐标即为右下角y坐标
        
        # 计算边界框的宽度和高度
        width = xyxy[:, 2] - xyxy[:, 0]  # 右下角x - 左上角x
        height = xyxy[:, 3] - xyxy[:, 1]  # 右下角y - 左上角y
        
        # 计算边界框高度
        area = width * height
        
        # 将底部中心点坐标、尺寸和高度组合成数组
        target_positions = np.column_stack((bottom_center_x, bottom_center_y))
        target_sizes = np.column_stack((width, height))
        target_areas = area
        
        # 处理类别信息
        if boxes.cls is not None:
            if hasattr(boxes.cls, 'cpu'):
                class_ids = boxes.cls.cpu().numpy()
            else:
                class_ids = boxes.cls.numpy() if hasattr(boxes.cls, 'numpy') else boxes.cls
        else:
            class_ids = np.zeros(len(bounding_boxes))
        
        # 组合所有信息
        for i in range(len(bounding_boxes)):
            info = {
                'bbox': bounding_boxes[i],  # [中心点x, 中心点y, 宽度, 高度]
                'id': object_ids[i] if object_ids else -1,  # 目标ID
                'confidence': confidence_scores[i],  # 置信度
                'center': target_positions[i],  # [底部中心点x, 底部中心点y]
                'size': target_sizes[i],  # [宽度, 高度]
                'area': target_areas[i],  # 边界框面积（像素）
                'xyxy': xyxy[i],  # 添加边界框坐标 [x1, y1, x2, y2]
                'class': class_ids[i]  # 添加类别信息
            }
            tracking_data.append(info)
    
    return tracking_data

def publish_target_info(tracking_info):
    """
    通过ROS发布目标信息（用于调试）
    
    Args:
        tracking_info (list): 追踪信息列表
    """
    # 如果ROS不可用或没有发布者，直接返回
    if not ROS_AVAILABLE or target_info_pub is None:
        return
        
    try:
        # 将目标信息格式化为JSON字符串并发布
        target_data = []
        for i, info in enumerate(tracking_info):
            target_data.append({
                'id': int(info['id']),
                'class': int(info['class']),  # 添加类别ID
                'class_name': COCO_CLASSES[int(info['class'])] if int(info['class']) < len(COCO_CLASSES) else 'unknown',  # 添加类别名称
                'center_x': float(info['center'][0]),
                'center_y': float(info['center'][1]),
                'area': float(info['area']),
                'confidence': float(info['confidence'])
            })
        
        # 发布JSON格式的数据
        data = {
            'timestamp': rospy.Time.now().to_sec(),
            'target_count': len(tracking_info),
            'targets': target_data
        }
        
        json_str = json.dumps(data)
        target_info_pub.publish(json_str)
    except Exception as e:
        print(f"发布目标信息时出错: {e}")

# 控制指令滤波
last_velocity_x, last_velocity_y, last_velocity_z = 0.0, 0.0, 0.0
last_angular_z = 0.0
def process_frame(cv_image):
    """
    处理一帧图像，执行目标检测和跟踪
    
    Args:
        cv_image: OpenCV格式的图像
    """
    global running, model, model_loaded, image_width, image_height
    global mouse_selector, target_classes
    global last_velocity_x, last_velocity_y, last_velocity_z, last_angular_z
    global custom_tracker, control_system
    try:
        # 获取图像的实际尺寸
        image_shape = cv_image.shape
        image_height, image_width = image_shape[:2]
        control_system.set_camera_resolution(image_width, image_height)
        
        # 如果是灰度图，转换为三通道图像以兼容YOLO模型
        if len(image_shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        
        # 检查模型是否已加载完成
        if not model_loaded:
            print("警告：模型未加载完成，跳过图像处理")
            # 发布零速度指令
            publish_velocity_command(0.0, 0.0, 0.0)
            return
            
        # 运行目标追踪，应用类别过滤
        try:
            if target_classes is not None:
                # 使用指定的目标类别进行检测
                results = model.predict(cv_image, classes=target_classes)
                print(f"使用类别过滤进行检测 (classes={target_classes})")
            else:
                # 检测所有类别
                results = model.predict(cv_image)
            
            # 使用自定义的botsort追踪器进行单目标追踪
            global custom_tracker
            if custom_tracker is None:
                custom_tracker = SingleObjectBotSortTracker()
                # custom_tracker = SingleObjectByteTrackTracker()
            
            # 更新追踪器状态
            results = [custom_tracker.update(results[0], cv_image)]
        except Exception as e:
            print(f"目标追踪过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 提取追踪信息
        tracking_info = extract_tracking_info(results[0])
        mouse_selector.update_tracking_info(tracking_info)  # 更新跟踪信息缓存
        
        # 检查选中的目标是否还存在
        mouse_selector.check_target_exists()

        # 在图像上绘制追踪结果
        annotated_frame = results[0].plot()  # 先使用默认绘制
        annotated_frame = mouse_selector.draw_tracking_results(annotated_frame, tracking_info)  # 再添加自定义绘制
        
        # 添加提示信息到图像上
        annotated_frame = mouse_selector.draw_selection_message(annotated_frame)

        # 创建可调节大小的窗口并显示图像
        cv2.namedWindow('YOLOv8 Object Tracking and PID Control', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLOv8 Object Tracking and PID Control', annotated_frame)
        cv2.setMouseCallback('YOLOv8 Object Tracking and PID Control', mouse_selector.mouse_callback)

        # 发布目标信息（用于调试）
        publish_target_info(tracking_info)
        
        # 确保控制系统已初始化
        if control_system is not None:
            # 计算控制指令
            # velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z = control_system.angle_control(tracking_info,mouse_selector) # 矢量控制-高空-给定角度
            # velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z = control_system.height_control(tracking_info,mouse_selector) # 像素控制-平飞-给定高度
            velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z = control_system.hybrid_control(tracking_info, mouse_selector)
            
            # 应用低通滤波器平滑控制指令
            filtered_vx = apply_filter(velocity_x, last_velocity_x, alpha=1)
            filtered_vy = apply_filter(velocity_y, last_velocity_y, alpha=1)
            filtered_vz = apply_filter(velocity_z, last_velocity_z, alpha=1)
            filtered_wz = apply_filter(angular_z, last_angular_z, alpha=1)
            
            # 更新上一次滤波后的值
            last_velocity_x, last_velocity_y, last_velocity_z = filtered_vx, filtered_vy, filtered_vz
            last_angular_z = filtered_wz

            # 发布控制指令
            publish_velocity_command(filtered_vx, filtered_vy, filtered_vz, angular_x, angular_y, filtered_wz)
        else:
            print("警告: 控制系统未初始化")
            
    except Exception as e:
        print(f"图像处理时出错: {e}")
        import traceback
        traceback.print_exc()

def image_callback(msg):
    """
    图像回调函数，处理从/camera/rgb/image_raw接收的图像数据
    
    Args:
        msg: 图像消息
    """
    global running
    
    try:
        # 将ROS图像消息转换为OpenCV格式
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 如果图像是彩色的，将其转换为灰度图
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 处理图像
        process_frame(cv_image)
        
        # 按'q'键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
    except Exception as e:
        print(f"图像处理时出错: {e}")
        import traceback
        traceback.print_exc()

def fallback_attitude_processing(msg):
    """
    备用姿态处理方法（当tf库不可用时使用）
    
    Args:
        msg: IMU消息，包含姿态信息
    """
    global current_roll, current_pitch, current_yaw
    
    try:
        # 简单的四元数到欧拉角转换
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        
        # 直接将四元数传递给坐标变换器
        control_system.set_uav_attitude(qx, qy, qz, qw)
        
        # 为了保持兼容性，仍将四元数转换为欧拉角用于其他用途
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        current_roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            current_pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            current_pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
            
    except Exception as e:
        print(f"备用姿态处理方法也失败: {e}")

def attitude_callback(msg):
    """
    处理无人机姿态信息
    
    Args:
        msg: IMU消息，包含姿态信息
    """
    global current_roll, current_pitch, current_yaw
    
    try:
        # 直接获取四元数并存储
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        
        # 直接将四元数传递给坐标变换器
        control_system.set_uav_attitude(qx, qy, qz, qw)
        
        # 为了保持兼容性，仍将四元数转换为欧拉角用于其他用途
        try:
            import tf.transformations
            quaternion = (qx, qy, qz, qw)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            current_roll = euler[0]
            current_pitch = euler[1]
            current_yaw = euler[2]
        except Exception as e:
            print(f"使用备用方法转换四元数到欧拉角: {e}")
            # 如果tf库不可用，使用备用方法
            fallback_attitude_processing(msg)
    except Exception as e:
        print(f"处理姿态信息时出错: {e}")

def altitude_callback(msg):
    """
    处理无人机相对高度信息
    
    Args:
        msg: geometry_msgs/PoseStamped消息，包含位置信息
    """
    
    try:
        # 从PoseStamped消息中获取z坐标作为相对高度
        current_altitude = msg.pose.position.z
        control_system.set_current_altitude(current_altitude)
    except Exception as e:
        print(f"处理高度信息时出错: {e}")

def cleanup():
    """
    清理资源
    """
    global running, usb_camera
    running = False
    
    # 释放USB相机资源
    if use_usb_camera and usb_camera is not None:
        usb_camera.release()
        
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    print("资源已释放")

def run():
    """
    主循环：运行YOLOv8追踪并发布控制指令
    """
    global running, usb_camera
    
    # 根据配置选择不同的运行模式
    if use_usb_camera:
        print("按 'q' 键退出程序")
        # 使用USB相机直接读取图像
        while running and (not ROS_AVAILABLE or not rospy.is_shutdown()):
            # 读取一帧
            ret, frame = usb_camera.read()
            
            if not ret:
                print("无法从USB相机获取图像")
                time.sleep(0.1)
                continue
                
            # 处理图像
            process_frame(frame)
            
            # 处理按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
    else:
        # 使用ROS订阅获取图像
        if not ROS_AVAILABLE:
            print("错误：ROS不可用，无法使用ROS订阅方式")
            return
            
        while running and not rospy.is_shutdown():
            rospy.spin()
            
    
    # 释放资源
    cleanup()
    print("程序已退出")

def create_default_config(config_path):
    """
    创建默认的YAML配置文件

    Args:
        config_path (str): 要创建的配置文件路径
    """
    try:
        # 默认配置 - 检测人和车辆
        default_config = {
            'target_classes': ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'],
            'confidence_threshold': 0.25,
            'comments': '# 这是检测配置文件。\n# 你可以在target_classes列表中指定要检测的类别\n# 有效的类别名称包括: ' + ', '.join(
                COCO_CLASSES)
        }

        # 写入YAML文件
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, sort_keys=False)

        print(f"已创建默认配置文件: {config_path}")
    except Exception as e:
        print(f"创建默认配置文件出错: {e}")

def load_detection_config(config_path):
    """
    从YAML文件加载目标检测配置

    Args:
        config_path (str): YAML配置文件路径

    Returns:
        list: 目标类别索引列表，如果加载失败则返回None
    """
    global target_classes

    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}，将使用默认配置")
            # 创建一个默认配置文件
            create_default_config(config_path)

        # 读取YAML配置文件
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # 获取要检测的类别名称列表
        class_names = config.get('target_classes', [])

        if not class_names:
            print("警告: 未指定目标类别，将检测所有类别")
            return None

        # 将类别名称转换为索引
        class_indices = []
        for class_name in class_names:
            if class_name in COCO_CLASSES:
                class_indices.append(COCO_CLASSES.index(class_name))
            else:
                print(f"警告: 类别名称 '{class_name}' 不在COCO数据集中，已忽略")

        if not class_indices:
            print("警告: 没有有效的目标类别，将检测所有类别")
            return None

        print(f"已加载目标检测配置，将检测以下类别: {', '.join([COCO_CLASSES[idx] for idx in class_indices])}")
        return class_indices

    except Exception as e:
        print(f"加载检测配置出错: {e}")
        return None

def init_model():
    """初始化YOLOv8模型"""
    global model, model_loaded
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        print(f"当前工作目录: {current_dir}")

        # 检查模型文件是否存在
        model_path = os.path.join(current_dir, 'yolo11n.pt')
        print(f"检查模型文件路径: {model_path}")

        if os.path.exists(model_path):
            print(f"模型文件存在: {model_path}")
        else:
            print(f"警告：模型文件不存在: {model_path}")
            # 尝试在其他可能的位置查找
            possible_paths = [
                './yolov8n.pt',
                '../yolov8n.pt'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"在 {path} 找到模型文件")
                    break
            else:
                print("错误：在任何预期位置都未找到模型文件")
                return False

        # 加载YOLOv8模型（使用预训练的yolov8n模型）
        print("正在加载YOLOv8模型...")
        model = YOLO(model_path)
        
        # 验证模型是否加载成功
        if model is None:
            print("错误：模型加载失败")
            return False

        print("YOLOv8模型加载成功")
        print(f"模型类型: {type(model)}")
        model_loaded = True  # 设置模型加载完成标志
        return True
    except Exception as e:
        print(f"模型加载时出错: {e}")
        import traceback
        traceback.print_exc()
        model = None
        model_loaded = False  # 确保标志为False
        return False

def init_usb_camera():
    """初始化USB相机"""
    global usb_camera, image_width, image_height
    
    try:
        # 创建VideoCapture对象
        usb_camera = cv2.VideoCapture(usb_camera_device, cv2.CAP_V4L2)
        
        # 检查摄像头是否成功打开
        if not usb_camera.isOpened():
            print(f"无法打开USB相机设备 {usb_camera_device}")
            return False
        
        # 设置摄像头参数（根据相机支持的分辨率调整）
        usb_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
        usb_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        usb_camera.set(cv2.CAP_PROP_FPS, 30)
        
        # 设置视频格式为MJPEG
        usb_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # 检查是否成功设置MJPEG格式
        actual_fourcc = int(usb_camera.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"实际使用的视频格式: {fourcc_str}")
        
        # 获取实际分辨率
        image_width = int(usb_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_height = int(usb_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        control_system.set_camera_resolution(image_width, image_height)
        print(f"USB相机已初始化，分辨率: {image_width}x{image_height}")
        
        return True
    except Exception as e:
        print(f"初始化USB相机时出错: {e}")
        return False

def init_ros_components():
    """初始化ROS相关组件"""
    global bridge, target_info_pub, velocity_pub, attitude_pub, image_sub, attitude_sub
    
    # 创建CvBridge对象用于图像格式转换
    bridge = CvBridge()
    
    # 创建发布者
    # 发布目标信息（用于调试）
    target_info_pub = rospy.Publisher('/yolo/target_info', String, queue_size=10)
    # 发布位置控制指令给PX4飞控
    velocity_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
    # 新增：发布姿态控制指令给PX4飞控
    attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
    
    # 创建图像订阅者
    image_sub = rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    
    # 订阅无人机当前姿态信息
    attitude_sub = rospy.Subscriber('/mavros/imu/data', Imu, attitude_callback)
    
    # 新增：订阅无人机相对高度信息
    altitude_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, altitude_callback)

def init_controllers():
    """初始化PID控制器"""
    global control_system
    # 检查控制系统是否已初始化
    if control_system is not None:
        # 重置PID控制器状态
        control_system.pid_height_controller.reset()
        control_system.pid_yaw_controller.reset()
        control_system.pid_vertical_controller.reset()
        control_system.pid_altitude_controller.reset()
        control_system.pid_angle_controller.reset()
    else:
        print("警告: 控制系统未初始化，跳过控制器重置")

def init_coordinate_transformer():
    """初始化坐标变换器"""
    global control_system
    # 检查控制系统是否已初始化
    if control_system is not None:
        # 相机安装角度参数（弧度）
        # roll: 绕X轴旋转角度, pitch: 绕Y轴旋转角度, yaw: 绕Z轴旋转角度
        camera_roll = 0.0  # 相机绕机体X轴旋转角度（左右倾斜），根据描述设置为0.785弧度
        camera_pitch = math.radians(0)   # 相机绕机体Y轴旋转角度（俯仰角）
        camera_yaw = 0.0     # 相机绕机体Z轴旋转角度（偏航角）
        
        # 设置相机安装角度
        control_system.set_camera_orientation(camera_roll, camera_pitch, camera_yaw)
    else:
        print("警告: 控制系统未初始化，跳过坐标变换器设置")

def init_system():
    """初始化系统"""
    global image_width, image_height, running, target_classes, control_system

    # 初始化控制标志
    running = True

    # 加载目标检测配置
    target_classes = load_detection_config(config_file)

    # 初始化模型（在初始化ROS组件之前加载模型，避免在模型加载完成前处理图像）
    model_initialized = init_model()
    if not model_initialized:
        print("警告：模型初始化失败，将继续运行但不会进行目标追踪")

    # 根据配置选择初始化ROS组件或USB相机
    if use_usb_camera:
        print("使用USB相机作为图像输入")
        camera_initialized = init_usb_camera()
        if not camera_initialized:
            print("错误: USB相机初始化失败")
            return False
        # 如果ROS可用，仍然初始化ROS节点（用于发布控制命令）
        if ROS_AVAILABLE:
            rospy.init_node('yolo_tracker_pid', anonymous=True)
            # 创建发布者（只用于发布控制命令）
            target_info_pub = rospy.Publisher('/yolo/target_info', String, queue_size=10)
            velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
    else:
        # 使用ROS订阅方式
        if not ROS_AVAILABLE:
            print("错误：ROS不可用，无法使用ROS订阅方式")
            return False
        rospy.init_node('yolo_tracker_pid', anonymous=True)

    
    # 初始化ROS组件
    init_ros_components()

    # 初始化控制器
    init_controllers()

    # 初始化坐标变换器
    init_coordinate_transformer()

    print("YOLOv8 目标追踪节点已初始化")
    return True

def main():
    """
    主函数：运行YOLOv8追踪节点
    """
    try:
        # 初始化系统
        if not init_system():
            print("系统初始化失败")
            return
            
        # 运行主循环
        run()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源被释放
        cleanup()

# 程序入口点
if __name__ == "__main__":
    # 参数解析
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8目标追踪程序')
    parser.add_argument('--use-usb-camera', action='store_true', 
                        help='使用USB相机直接读取图像，而不是通过ROS订阅')
    parser.add_argument('--device', type=str, default='/dev/video0',
                        help='USB相机设备路径 (默认: /dev/video2)')
    parser.add_argument('--config', type=str, default='detection_config.yaml',
                        help='目标检测配置文件路径 (默认: detection_config.yaml)')
    
    args = parser.parse_args()
    
    # 设置全局配置
    use_usb_camera = args.use_usb_camera
    usb_camera_device = args.device
    config_file = "detection_config.yaml"
    
    # 启动主程序
    main()
