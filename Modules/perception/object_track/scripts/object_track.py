#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 目标追踪ROS节点
此脚本将YOLOv8目标追踪功能封装为ROS节点，
通过PID控制器计算无人机在x、y、z方向的速度，
并通过MAVROS的topic将速度指令发布给PX4飞控系统
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

# 尝试导入ROS模块
ROS_AVAILABLE = False
try:
    import rospy
    from std_msgs.msg import String
    from geometry_msgs.msg import TwistStamped
    from sensor_msgs.msg import Image, Imu
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
    print("ROS模块导入成功")
except ImportError as e:
    print(f"ROS模块导入失败: {e}")
    print("此脚本需要在ROS环境中运行")

# 导入鼠标目标选择器
from mouse_target_selector import MouseTargetSelector

# ========================
# 全局变量
# ========================
bridge = None
model = None
target_info_pub = None
velocity_pub = None
image_sub = None
attitude_sub = None
running = True
model_loaded = False  # 新增：模型加载状态标志

# 新增：USB相机配置
use_usb_camera = False  # 默认不使用USB相机
usb_camera_device = "/dev/video2"  # USB相机设备路径，可根据实际情况修改
usb_camera = None  # USB相机对象

# 新增：目标类别过滤配置
target_classes = None  # 用于存储要检测的目标类别列表
config_file = "detection_config.yaml"  # YAML配置文件路径

# 图像参数
# 修改为从相机获取实际尺寸，初始值设为默认值
image_width = 1920
image_height = 1080

# 第一帧图像处理标志
first_image_processed = False

# 创建鼠标目标选择器实例
mouse_selector = MouseTargetSelector()

# 目标参数
desired_area = 0

# 最大速度限制（m/s）
MAX_VELOCITY = 5.0

# 无人机姿态参数
current_roll = 0.0
current_pitch = 0.0
current_yaw = 0.0
#如何使跟踪目标快且稳定地出现在画面中心，需要调整PID参数和速度增益系数
# # 定义速度增益系数（将PID输出转换为相机坐标系下的实际速度）
VELOCITY_GAIN_X = 0.0001*10 # m/s 10~20
VELOCITY_GAIN_Y = 0.0001*10  # m/s yaw方向速度较小，因为y方向可以偏航转动
VELOCITY_GAIN_Z = 0.0001*100 # m/s 100~200
ANGULAR_GAIN = 0.001*4 # rad/s yaw方向的偏航角速度  4~6之间
# VELOCITY_GAIN_X = 0.0 # m/s 
# VELOCITY_GAIN_Y = 0.0  # m/s yaw方向速度较小，因为y方向可以偏航转动
# VELOCITY_GAIN_Z = 0.0  # m/s 
# ANGULAR_GAIN = 0.0 # rad/s yaw方向的偏航角速度  4~10之间
# ========================
# PID控制器模块（函数实现）
# ========================
# PID控制器状态变量
# 控制y向速度（相机系）- 对应水平位置误差
pid_x_state = {'kp': 0.1, 'ki': 0.000, 'kd': 0.000, 'previous_error': 0, 'integral': 0}
# 控制z向速度（相机系）- 对应垂直位置误差
pid_y_state = {'kp': 0.1, 'ki': 0.000, 'kd': 0.0, 'previous_error': 0, 'integral': 0}
# 控制x向速度（相机系）- 对应面积误差
pid_z_state = {'kp': 0.01, 'ki': 0.0000, 'kd': 0.000, 'previous_error': 0, 'integral': 0}

# 新增：COCO数据集80个类别名称，用于配置文件中的类别名到索引的映射
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

def pid_update(pid_state, error, dt=1.0):
    """
    更新PID控制器，计算控制输出
    
    Args:
        pid_state (dict): PID控制器状态
        error (float): 当前误差值
        dt (float): 时间间隔，默认为1.0
            
    Returns:
        float: PID控制器的输出值
    """
    # 累积误差（积分项）
    pid_state['integral'] += error * dt
    
    # 计算误差变化率（微分项）
    derivative = (error - pid_state['previous_error']) / dt
    
    # PID公式：输出 = Kp*误差 + Ki*积分项 + Kd*微分项
    output = pid_state['kp'] * error + pid_state['ki'] * pid_state['integral'] + pid_state['kd'] * derivative
    
    # 更新上一次的误差值
    pid_state['previous_error'] = error
    
    return output

def pid_reset(pid_state):
    """
    重置PID控制器状态
    将积分项和上一次误差清零
    
    Args:
        pid_state (dict): PID控制器状态
    """
    pid_state['previous_error'] = 0
    pid_state['integral'] = 0

# ========================
# 坐标变换器模块（函数实现）
# ========================

# 坐标变换器状态变量
coordinate_transformer_state = {
    'camera_roll': 0.0,
    'camera_pitch': 0.0,
    'camera_yaw': 0.0,
    'current_roll': 0.0,
    'current_pitch': 0.0,
    'current_yaw': 0.0
}

def set_camera_orientation(roll, pitch, yaw):
    """
    设置相机相对于机身的安装角度
    
    Args:
        roll (float): 相机绕机体X轴旋转角度
        pitch (float): 相机绕机体Y轴旋转角度
        yaw (float): 相机绕机体Z轴旋转角度
    """
    coordinate_transformer_state['camera_roll'] = roll
    coordinate_transformer_state['camera_pitch'] = pitch
    coordinate_transformer_state['camera_yaw'] = yaw

def set_uav_attitude(roll, pitch, yaw):
    """
    设置无人机当前姿态角
    
    Args:
        roll (float): 无人机绕X轴旋转角度
        pitch (float): 无人机绕Y轴旋转角度
        yaw (float): 无人机绕Z轴旋转角度
    """
    coordinate_transformer_state['current_roll'] = roll
    coordinate_transformer_state['current_pitch'] = pitch
    coordinate_transformer_state['current_yaw'] = yaw

def transform_camera_to_body(cam_x, cam_y, cam_z):
    """
    将相机坐标系下的速度转换为机身坐标系下的速度
    
    Args:
        cam_x (float): 相机坐标系X轴速度
        cam_y (float): 相机坐标系Y轴速度
        cam_z (float): 相机坐标系Z轴速度
            
    Returns:
        tuple: 机身坐标系下的速度分量 (body_x, body_y, body_z)
    """
    # 如果相机安装角度都为0，则直接返回
    if (coordinate_transformer_state['camera_roll'] == 0.0 and 
        coordinate_transformer_state['camera_pitch'] == 0.0 and 
        coordinate_transformer_state['camera_yaw'] == 0.0):
        return cam_x, cam_y, cam_z
    
    # 绕Z轴旋转（偏航）
    cos_yaw = math.cos(coordinate_transformer_state['camera_yaw'])
    sin_yaw = math.sin(coordinate_transformer_state['camera_yaw'])
    temp_x = cam_x * cos_yaw - cam_y * sin_yaw
    temp_y = cam_x * sin_yaw + cam_y * cos_yaw
    temp_z = cam_z
    
    # 绕Y轴旋转（俯仰）
    cos_pitch = math.cos(coordinate_transformer_state['camera_pitch'])
    sin_pitch = math.sin(coordinate_transformer_state['camera_pitch'])
    body_x = temp_x * cos_pitch + temp_z * sin_pitch
    body_y = temp_y
    body_z = -temp_x * sin_pitch + temp_z * cos_pitch
    
    # 绕X轴旋转（滚转）
    cos_roll = math.cos(coordinate_transformer_state['camera_roll'])
    sin_roll = math.sin(coordinate_transformer_state['camera_roll'])
    final_x = body_x
    final_y = body_y * cos_roll - body_z * sin_roll
    final_z = body_y * sin_roll + body_z * cos_roll
    
    return final_x, final_y, final_z

def transform_body_to_enu(body_x, body_y, body_z):
    """
    将机身坐标系下的速度转换为ENU坐标系下的速度
    
    Args:
        body_x (float): 机身坐标系X轴速度
        body_y (float): 机身坐标系Y轴速度
        body_z (float): 机身坐标系Z轴速度
            
    Returns:
        tuple: ENU坐标系下的速度分量 (enu_x, enu_y, enu_z)
    """
    # 如果无人机姿态角都为0，则直接返回
    if (coordinate_transformer_state['current_roll'] == 0.0 and 
        coordinate_transformer_state['current_pitch'] == 0.0 and 
        coordinate_transformer_state['current_yaw'] == 0.0):
        return body_x, body_y, body_z
    
    # 使用无人机当前姿态角进行坐标变换
    cos_yaw = math.cos(coordinate_transformer_state['current_yaw'])
    sin_yaw = math.sin(coordinate_transformer_state['current_yaw'])
    cos_pitch = math.cos(coordinate_transformer_state['current_pitch'])
    sin_pitch = math.sin(coordinate_transformer_state['current_pitch'])
    cos_roll = math.cos(coordinate_transformer_state['current_roll'])
    sin_roll = math.sin(coordinate_transformer_state['current_roll'])
    
    # 绕Z轴旋转（偏航）
    temp_x = body_x * cos_yaw - body_y * sin_yaw
    temp_y = body_x * sin_yaw + body_y * cos_yaw
    temp_z = body_z
    
    # 绕Y轴旋转（俯仰）
    enu_x = temp_x * cos_pitch + temp_z * sin_pitch
    enu_y = temp_y
    enu_z = -temp_x * sin_pitch + temp_z * cos_pitch
    
    # 绕X轴旋转（滚转）
    final_x = enu_x
    final_y = enu_y * cos_roll - enu_z * sin_roll
    final_z = enu_y * sin_roll + enu_z * cos_roll
    
    return final_x, final_y, final_z

def transform_camera_to_enu(cam_x, cam_y, cam_z):
    """
    直接将相机坐标系下的速度转换为ENU坐标系下的速度
    
    Args:
        cam_x (float): 相机坐标系X轴速度
        cam_y (float): 相机坐标系Y轴速度
        cam_z (float): 相机坐标系Z轴速度
            
    Returns:
        tuple: ENU坐标系下的速度分量 (enu_x, enu_y, enu_z)
    """
    # 先从相机坐标系转换到机身坐标系
    body_x, body_y, body_z = transform_camera_to_body(cam_x, cam_y, cam_z)
    
    # 再从机身坐标系转换到ENU坐标系
    enu_x, enu_y, enu_z = transform_body_to_enu(body_x, body_y, body_z)
    
    return enu_x, enu_y, enu_z

# 新增: 加载检测配置函数
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

# 新增: 创建默认配置文件
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
            'comments': '# 这是检测配置文件。\n# 你可以在target_classes列表中指定要检测的类别\n# 有效的类别名称包括: ' + ', '.join(COCO_CLASSES)
        }
        
        # 写入YAML文件
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, sort_keys=False)
            
        print(f"已创建默认配置文件: {config_path}")
    except Exception as e:
        print(f"创建默认配置文件出错: {e}")

# ========================
# 函数定义
# ========================
def init_ros_components():
    """初始化ROS相关组件"""
    global bridge, target_info_pub, velocity_pub, image_sub, attitude_sub
    
    # 创建CvBridge对象用于图像格式转换
    bridge = CvBridge()
    
    # 创建发布者
    # 发布目标信息（用于调试）
    target_info_pub = rospy.Publisher('/yolo/target_info', String, queue_size=10)
    # 发布速度控制指令给PX4飞控
    velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
    
    # 创建图像订阅者
    image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    
    # 订阅无人机当前姿态信息
    attitude_sub = rospy.Subscriber('/mavros/imu/data', Imu, attitude_callback)

# 新增: 初始化USB相机
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
        print(f"USB相机已初始化，分辨率: {image_width}x{image_height}")
        
        return True
    except Exception as e:
        print(f"初始化USB相机时出错: {e}")
        return False

def init_model():
    """初始化YOLOv8模型"""
    global model, model_loaded
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        print(f"当前工作目录: {current_dir}")
        
        # 检查模型文件是否存在
        model_path = os.path.join(current_dir, 'yolov8n.pt')
        print(f"检查模型文件路径: {model_path}")
        
        if os.path.exists(model_path):
            print(f"模型文件存在: {model_path}")
        else:
            print(f"警告：模型文件不存在: {model_path}")
            # 尝试在其他可能的位置查找
            possible_paths = [
                '/home/gzy/ultralytics/yolov8n.pt',
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

def init_controllers():
    """初始化PID控制器"""
    global pid_x_state, pid_y_state, pid_z_state
    # 重置PID控制器状态
    pid_reset(pid_x_state)
    pid_reset(pid_y_state)
    pid_reset(pid_z_state)

def init_coordinate_transformer():
    """初始化坐标变换器"""
    # 相机安装角度参数（弧度）
    # roll: 绕X轴旋转角度, pitch: 绕Y轴旋转角度, yaw: 绕Z轴旋转角度
    camera_roll = 0.0  # 相机绕机体X轴旋转角度（左右倾斜），根据描述设置为0.785弧度
    camera_pitch = math.radians(30)   # 相机绕机体Y轴旋转角度（俯仰角）
    camera_yaw = 0.0     # 相机绕机体Z轴旋转角度（偏航角）
    
    # 设置相机安装角度
    set_camera_orientation(camera_roll, camera_pitch, camera_yaw)

def init_target_parameters():
    """初始化目标相关参数"""
    global desired_area
    # 设置期望的目标面积（可根据实际情况调整）
    # 使用默认图像尺寸计算初始期望面积
    desired_area = (640 * 480) // 100  # 总像素的1/100 = 30720

def init_system():
    """初始化系统"""
    global image_width, image_height, running, target_classes
    
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
    
    # 初始化目标参数
    init_target_parameters()
    
    print("YOLOv8 目标追踪节点已初始化")
    print("按 'q' 键退出程序")
    return True

def extract_tracking_info(result):
    """
    从YOLOv8追踪结果中提取关键信息
    
    Args:
        result: YOLOv8追踪结果对象
        
    Returns:
        list: 包含边界框、ID、置信度、中心点位置和面积的字典列表
    """
    tracking_data = []
    
    # 检查是否有检测到的目标
    if result.boxes is not None:
        boxes = result.boxes
        
        # 获取边界框坐标 (xywh格式: 中心点x, 中心点y, 宽度, 高度)
        bounding_boxes = boxes.xywh.cpu().numpy()
        
        # 获取目标ID（用于追踪）
        object_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
        
        # 获取置信度分数
        confidence_scores = boxes.conf.cpu().numpy()
        
        # 计算目标中心位置、宽度和高度
        # xyxy格式: 左上角x, 左上角y, 右下角x, 右下角y
        xyxy = boxes.xyxy.cpu().numpy()
        
        # 计算中心点坐标 (左上角和右下角坐标的平均值)
        center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
        center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
        
        # 计算边界框的宽度和高度
        width = xyxy[:, 2] - xyxy[:, 0]  # 右下角x - 左上角x
        height = xyxy[:, 3] - xyxy[:, 1]  # 右下角y - 左上角y
        
        # 计算边界框面积
        area = width * height
        
        # 将中心点坐标、尺寸和面积组合成数组
        target_positions = np.column_stack((center_x, center_y))
        target_sizes = np.column_stack((width, height))
        target_areas = area
        
        # 组合所有信息
        for i in range(len(bounding_boxes)):
            info = {
                'bbox': bounding_boxes[i],  # [中心点x, 中心点y, 宽度, 高度]
                'id': object_ids[i] if object_ids else -1,  # 目标ID
                'confidence': confidence_scores[i],  # 置信度
                'center': target_positions[i],  # [中心点x, 中心点y]
                'size': target_sizes[i],  # [宽度, 高度]
                'area': target_areas[i],  # 边界框面积（像素）
                'xyxy': xyxy[i],  # 添加边界框坐标 [x1, y1, x2, y2]
                'class': boxes.cls[i].item() if boxes.cls is not None else 0  # 添加类别信息
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

def calculate_camera_velocity(tracking_info):
    """
    订阅YOLO输出信息，计算相机坐标系下的速度和角速度

    Args:
        tracking_info (list): 追踪信息列表
        
    Returns:
        tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) 相机坐标系下的速度和角速度（m/s, rad/s）
    """
    # 获取选中的目标ID
    selected_target_id = mouse_selector.get_selected_target_id()
    
    # 默认速度和角速度为0（无目标时保持静止）
    velocity_x, velocity_y, velocity_z = 0.0, 0.0, 0.0
    angular_x, angular_y, angular_z = 0.0, 0.0, 0.0  # 添加角速度初始化

    # 如果检测到目标且有选中的目标
    if len(tracking_info) > 0 and selected_target_id is not None:
        # 查找选中的目标
        target = None
        for t in tracking_info:
            if t['id'] == selected_target_id:
                target = t
                break
        
        # 如果找到选中的目标
        if target is not None:
            center_x, center_y = target['center']  # 目标中心点坐标
            area = target['area']  # 目标边界框面积
            
            # 计算图像中心点坐标
            center_image_x = image_width / 2
            center_image_y = image_height / 2
            
            # 计算误差
            error_x = center_image_x - center_x  # 期望的中心x - 实际中心x
            error_y = center_image_y - center_y  # 期望的中心y - 实际中心y
            error_z = desired_area - area        # 期望面积 - 实际面积
            
            # 新增逻辑：检查是否在允许的误差范围内（中心点误差不超过1%，面积误差不超过1%）
            # 计算图像中心点的10%误差范围
            center_tolerance_x = image_width * 0.1
            center_tolerance_y = image_height * 0.1
            # 计算面积的20%误差范围
            area_tolerance = desired_area * 0.2
            
            # 如果中心点误差和面积误差都在1%以内，则将所有速度和角速度指令设置为0
            if (abs(error_x) <= center_tolerance_x and 
                abs(error_y) <= center_tolerance_y and 
                abs(error_z) <= area_tolerance):
                control_signal_x = 0.0
                control_signal_y = 0.0
                control_signal_z = 0.0
                print(f"目标在允许误差范围内，所有控制指令设置为0")
                print(f"中心误差: X={error_x:.2f}, Y={error_y:.2f}, 面积误差: {error_z:.2f}")
                # 保持默认值0，不需要额外设置
            else:
                # 使用PID控制器计算各方向控制信号

                # 逻辑：如果目标检测框的任何顶点在屏幕边缘，则x方向速度指令设置为零
                # 原因：边缘目标可能未完全显示，无法准确判断距离远近
                edge_threshold = 0.005  # 边缘阈值，距离图像边缘10%范围内认为是边缘

                # 获取检测框的四个顶点坐标
                x1, y1, x2, y2 = target['xyxy']
                target_in_edge = (x1 <= image_width * edge_threshold or 
                                x2 >= image_width * (1 - edge_threshold) or 
                                y1 <= image_height * edge_threshold or 
                                y2 >= image_height * (1 - edge_threshold))

                if target_in_edge:
                    control_signal_x = 0.0  # 禁止x方向速度指令
                    print(f"目标检测框在屏幕边缘，禁止基于面积的距离控制")
                else:
                    control_signal_x = pid_update(pid_z_state, error_z)
                # 相机坐标系下，y轴的控制信号（对应水平位置误差）
                control_signal_y = pid_update(pid_x_state, error_x)  
                # 相机坐标系下，z轴的控制信号（对应垂直位置误差）
                control_signal_z = pid_update(pid_y_state, error_y)  
                
            # 将控制信号转换为实际速度和角速度（m/s）
            velocity_x = control_signal_x * VELOCITY_GAIN_X
            velocity_y = control_signal_y * VELOCITY_GAIN_Y
            velocity_z = control_signal_z * VELOCITY_GAIN_Z
            angular_x = 0.0
            angular_y = 0.0
            angular_z = control_signal_y * ANGULAR_GAIN
            
            # 打印控制信息（调试用）
            print(f"跟踪目标ID: {selected_target_id}")
            print(f"控制误差 - X: {error_x:.2f}, Y: {error_y:.2f}, Z: {error_z:.2f}")
            print(f"控制信号 - X: {control_signal_x:.2f}, Y: {control_signal_y:.2f}, Z: {control_signal_z:.2f}")
            print(f"相机坐标系速度 - VX: {velocity_x:.3f} m/s, VY: {velocity_y:.3f} m/s, VZ: {velocity_z:.3f} m/s")
            print(f"相机坐标系角速度 - WX: {angular_x:.3f} rad/s, WY: {angular_y:.3f} rad/s, WZ: {angular_z:.3f} rad/s")
    # 注意：这里不再处理目标丢失的情况，因为MouseTargetSelector已经处理了
    
    # 返回线速度和角速度
    return velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z

def calculate_velocity_control(tracking_info):
    """
    计算速度和角速度控制指令
    
    Args:
        tracking_info (list): 追踪信息列表
        
    Returns:
        tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) ENU坐标系下的速度和角速度
    """
    # 计算相机坐标系下的速度和角速度
    cam_vx, cam_vy, cam_vz, cam_wx, cam_wy, cam_wz = calculate_camera_velocity(tracking_info)
    
    # 使用坐标变换器进行坐标系转换
    enu_vx, enu_vy, enu_vz = transform_camera_to_enu(cam_vx, cam_vy, cam_vz)
    # 注意：角速度的变换可能与线速度不同，需要根据具体情况调整
    # 这里假设角速度也使用相同的方式变换（这可能不完全正确，需根据物理意义调整）
    enu_wx, enu_wy, enu_wz = transform_camera_to_enu(cam_wx, cam_wy, cam_wz)
        
    return enu_vx, enu_vy, enu_vz, enu_wx, enu_wy, enu_wz

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
        # 创建TwistStamped消息
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.header.frame_id = "base_link"
        
        # PX4飞控使用的是ENU坐标系（东-北-天）
        twist_msg.twist.linear.x = velocity_x  # 东向速度
        twist_msg.twist.linear.y = velocity_y  # 北向速度
        twist_msg.twist.linear.z = velocity_z  # 天向速度
        
        # 设置角速度（只发布绕Z轴旋转的角速度，其他轴角速度为0）
        twist_msg.twist.angular.x = 0.0  # 绕东轴旋转角速度设为0
        twist_msg.twist.angular.y = 0.0  # 绕北轴旋转角速度设为0
        twist_msg.twist.angular.z = angular_z  # 只发布绕天轴旋转角速度
        
        # 发布速度和角速度指令
        velocity_pub.publish(twist_msg)
        print(f"已发布速度指令: VX={velocity_x:.2f}, VY={velocity_y:.2f}, VZ={velocity_z:.2f}")
        if angular_z != 0.0:
            print(f"已发布角速度指令: WZ={angular_z:.2f} rad/s")
    except Exception as e:
        print(f"发布速度和角速度指令时出错: {e}")

def process_frame(cv_image):
    """
    处理一帧图像，执行目标检测和跟踪
    
    Args:
        cv_image: OpenCV格式的图像
    """
    global running, model, model_loaded, image_width, image_height, first_image_processed, desired_area
    global mouse_selector, target_classes
    
    try:
        # 获取图像的实际尺寸
        image_height, image_width = cv_image.shape[:2]
        
        # 如果是第一次处理图像，更新依赖于图像尺寸的参数
        if not first_image_processed:
            # 重新计算期望的目标面积
            desired_area = (image_width * image_height) // 50  # 总像素的1/50
            print(f"实际图像尺寸: {image_width}x{image_height}, 更新期望目标面积: {desired_area}")
            first_image_processed = True
        
        # 检查模型是否已加载完成
        if not model_loaded:
            print("警告：模型未加载完成，跳过图像处理")
            # 发布零速度指令
            publish_velocity_command(0.0, 0.0, 0.0)
            return
            
        # 运行目标追踪，应用类别过滤
        if target_classes is not None:
            # 使用指定的目标类别进行检测和跟踪
            results = model.track(cv_image, persist=True, classes=target_classes)
            print(f"使用类别过滤进行检测 (classes={target_classes})")
        else:
            # 检测所有类别
            results = model.track(cv_image, persist=True)
        
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
        
        # 显示当前检测的目标类别
        if target_classes is not None:
            class_names = [COCO_CLASSES[idx] for idx in target_classes]
            class_text = f"检测类别: {', '.join(class_names)}"
            cv2.putText(annotated_frame, class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
        
        # 显示窗口并设置鼠标回调
        cv2.imshow('YOLOv8 Object Tracking and PID Control', annotated_frame)
        cv2.setMouseCallback('YOLOv8 Object Tracking and PID Control', mouse_selector.mouse_callback)
        
        # 发布目标信息（用于调试）
        publish_target_info(tracking_info)
        
        # 计算速度和角速度控制指令
        velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z = calculate_velocity_control(tracking_info)
        
        # 发布速度和角速度控制指令给PX4飞控
        publish_velocity_command(velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z)
            
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
        
        # 更新坐标变换器中的姿态信息
        set_uav_attitude(current_roll, current_pitch, current_yaw)
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
        # 从四元数转换为欧拉角
        import tf.transformations
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        current_roll = euler[0]
        current_pitch = euler[1]
        current_yaw = euler[2]
        
        # 更新坐标变换器中的姿态信息
        set_uav_attitude(current_roll, current_pitch, current_yaw)
    except Exception as e:
        print(f"处理姿态信息时出错: {e}")
        # 如果tf库不可用，尝试使用备用方法
        fallback_attitude_processing(msg)

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
    
    print("YOLOv8 目标追踪已启动")
    print("按 'q' 键退出程序")
    
    # 根据配置选择不同的运行模式
    if use_usb_camera:
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
    parser.add_argument('--device', type=str, default='/dev/video6',
                        help='USB相机设备路径 (默认: /dev/video6)')
    parser.add_argument('--config', type=str, default='detection_config.yaml',
                        help='目标检测配置文件路径 (默认: detection_config.yaml)')
    
    args = parser.parse_args()
    
    # 设置全局配置
    use_usb_camera = args.use_usb_camera
    usb_camera_device = args.device
    config_file = args.config
    
    # 启动主程序
    main()