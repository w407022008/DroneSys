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

# 导入鼠标目标选择器
from mouse_target_selector import MouseTargetSelector

# 导入追踪相关模块
from ultralytics.trackers.bot_sort import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace

# 尝试导入ROS模块
ROS_AVAILABLE = False
try:
    import rospy
    from std_msgs.msg import String
    from geometry_msgs.msg import TwistStamped, PoseStamped
    from sensor_msgs.msg import Image, Imu
    from mavros_msgs.msg import PositionTarget, AttitudeTarget  # 添加这两个消息类型
    import std_msgs.msg  # 新增导入
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
    print("ROS模块导入成功")
except ImportError as e:
    print(f"ROS模块导入失败: {e}")
    print("此脚本需要在ROS环境中运行")

# ========================
# 自定义单目标追踪器类
# ========================

class SingleObjectBotSortTracker:
    """
    单目标追踪器类，基于BOTSORT算法实现
    """

    def __init__(self):
        """初始化追踪器"""
        # 创建类似botsort.yaml的配置
        args = IterableSimpleNamespace(
            tracker_type='botsort',
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method='sparseOptFlow',
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model='auto'
        )

        # 初始化BOTSORT追踪器
        self.tracker = BOTSORT(args=args, frame_rate=30)
        self.selected_track_id = None
        self.has_detected = False

    def update(self, detection_result, image):
        """
        更新追踪器状态

        Args:
            detection_result: YOLO检测结果
            image: 输入图像

        Returns:
            更新后的检测结果
        """
        # 检查是否有检测框
        if detection_result.boxes is not None and len(detection_result.boxes) > 0:
            boxes = detection_result.boxes

            # 如果还没有检测到目标，则标记为已检测
            if not self.has_detected:
                self.has_detected = True

            # 使用BOTSORT追踪器更新
            tracks = self.tracker.update(boxes.cpu().numpy(), image)

            # 如果有追踪结果，更新结果中的ID信息
            if len(tracks) > 0:
                # 创建包含追踪ID的新boxes数据
                # 原始boxes数据格式: [x1, y1, x2, y2, conf, cls]
                # 需要转换为: [x1, y1, x2, y2, track_id, conf, cls] (7列)
                original_boxes_data = boxes.data.cpu().numpy()
                new_boxes_data = np.zeros((tracks.shape[0], 7))

                # 复制原始box坐标、置信度和类别
                new_boxes_data[:, [0, 1, 2, 3, 5, 6]] = tracks[:, [0, 1, 2, 3, 5, 6]]
                # 添加追踪ID (第4列)
                new_boxes_data[:, 4] = tracks[:, 4]

                # 创建新的Boxes对象，包含追踪ID
                detection_result.boxes = detection_result.boxes.__class__(
                    new_boxes_data,
                    detection_result.boxes.orig_shape
                )

                # 如果之前没有选择目标且有追踪结果，则选择置信度最高的目标
                if self.selected_track_id is None and len(tracks) > 0:
                    # 找到置信度最高的追踪目标
                    max_conf_idx = np.argmax(tracks[:, 5])  # confidence是第6列
                    self.selected_track_id = int(tracks[max_conf_idx, 4])  # track id是第5列
                elif self.selected_track_id is not None:
                    # 检查之前选择的目标是否还在追踪中
                    matched_tracks = tracks[tracks[:, 4] == self.selected_track_id]
                    if len(matched_tracks) == 0 and len(tracks) > 0:
                        # 之前选择的目标丢失了，选择新的目标（置信度最高的）
                        max_conf_idx = np.argmax(tracks[:, 5])
                        self.selected_track_id = int(tracks[max_conf_idx, 4])
            else:
                # 没有追踪到任何目标，清除选择
                self.selected_track_id = None
                # 确保boxes不包含追踪ID
                if hasattr(boxes, 'is_track') and boxes.is_track:
                    # 如果当前boxes包含追踪ID，创建不包含追踪ID的新boxes
                    original_data = boxes.data.cpu().numpy()
                    # 只保留[x1, y1, x2, y2, conf, cls]
                    stripped_data = original_data[:, [0, 1, 2, 3, 5, 6]]
                    detection_result.boxes = boxes.__class__(
                        torch.from_numpy(stripped_data),
                        boxes.orig_shape
                    )
        else:
            # 没有检测到任何框
            if self.has_detected:
                # 如果之前检测到过目标，但现在没有检测到，重置追踪器
                self.tracker.reset()
                self.selected_track_id = None
                self.has_detected = False

        return detection_result

    def select_object(self, track_id):
        """
        选择特定的追踪目标

        Args:
            track_id (int): 要追踪的目标ID
        """
        self.selected_track_id = track_id

    def reset(self):
        """重置追踪器"""
        self.tracker.reset()
        self.selected_track_id = None
        self.has_detected = False

class SingleObjectByteTrackTracker:
    """
    单目标追踪器类，基于BYTETRACK算法实现
    """
    
    def __init__(self):
        """初始化追踪器"""
        # 创建类似bytetrack.yaml的配置
        args = IterableSimpleNamespace(
            tracker_type='bytetrack',
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True
        )
        
        # 初始化BYTETracker追踪器
        self.tracker = BYTETracker(args=args, frame_rate=30)
        self.selected_track_id = None
        self.has_detected = False
    
    def update(self, detection_result, image):
        """
        更新追踪器状态
        
        Args:
            detection_result: YOLO检测结果
            image: 输入图像
            
        Returns:
            更新后的检测结果
        """
        # 检查是否有检测框
        if detection_result.boxes is not None and len(detection_result.boxes) > 0:
            boxes = detection_result.boxes
            
            # 如果还没有检测到目标，则标记为已检测
            if not self.has_detected:
                self.has_detected = True
                
            # 使用BYTETracker追踪器更新
            # 确保将boxes数据从CUDA转移到CPU再转换为numpy数组
            tracks = self.tracker.update(boxes.cpu(), image)
            
            # 如果有追踪结果，更新结果中的ID信息
            if len(tracks) > 0:
                # 创建包含追踪ID的新boxes数据
                # 原始boxes数据格式: [x1, y1, x2, y2, conf, cls]
                # 需要转换为: [x1, y1, x2, y2, track_id, conf, cls] (7列)
                original_boxes_data = boxes.data.cpu().numpy()
                new_boxes_data = np.zeros((tracks.shape[0], 7))
                
                # 复制原始box坐标、置信度和类别
                new_boxes_data[:, [0, 1, 2, 3, 5, 6]] = tracks[:, [0, 1, 2, 3, 5, 6]]
                # 添加追踪ID (第4列)
                new_boxes_data[:, 4] = tracks[:, 4]
                
                # 创建新的Boxes对象，包含追踪ID
                detection_result.boxes = detection_result.boxes.__class__(
                    new_boxes_data, 
                    detection_result.boxes.orig_shape
                )
                
                # 如果之前没有选择目标且有追踪结果，则选择置信度最高的目标
                if self.selected_track_id is None and len(tracks) > 0:
                    # 找到置信度最高的追踪目标
                    max_conf_idx = np.argmax(tracks[:, 5])  # confidence是第6列
                    self.selected_track_id = int(tracks[max_conf_idx, 4])  # track id是第5列
                elif self.selected_track_id is not None:
                    # 检查之前选择的目标是否还在追踪中
                    matched_tracks = tracks[tracks[:, 4] == self.selected_track_id]
                    if len(matched_tracks) == 0 and len(tracks) > 0:
                        # 之前选择的目标丢失了，选择新的目标（置信度最高的）
                        max_conf_idx = np.argmax(tracks[:, 5])
                        self.selected_track_id = int(tracks[max_conf_idx, 4])
            else:
                # 没有追踪到任何目标，清除选择
                self.selected_track_id = None
                # 确保boxes不包含追踪ID
                if hasattr(boxes, 'is_track') and boxes.is_track:
                    # 如果当前boxes包含追踪ID，创建不包含追踪ID的新boxes
                    original_data = boxes.data.cpu().numpy()
                    # 只保留[x1, y1, x2, y2, conf, cls]
                    stripped_data = original_data[:, [0, 1, 2, 3, 5, 6]]
                    detection_result.boxes = boxes.__class__(
                        torch.from_numpy(stripped_data),
                        boxes.orig_shape
                    )
        else:
            # 没有检测到任何框
            if self.has_detected:
                # 如果之前检测到过目标，但现在没有检测到，重置追踪器
                self.tracker.reset()
                self.selected_track_id = None
                self.has_detected = False
            
        return detection_result
    
    def select_object(self, track_id):
        """
        选择特定的追踪目标
        
        Args:
            track_id (int): 要追踪的目标ID
        """
        self.selected_track_id = track_id
    
    def reset(self):
        """重置追踪器"""
        self.tracker.reset()
        self.selected_track_id = None
        self.has_detected = False

# 全局追踪器实例
custom_tracker = None

# ========================
# PID 控制器类
# ========================

class PIDController:
    """PID控制器类，用于封装PID控制逻辑"""
    
    def __init__(self, kp, ki, kd, max_integral=None, min_output=-float('inf'), max_output=float('inf')):
        """
        初始化PID控制器
        
        Args:
            kp (float): 比例系数
            ki (float): 积分系数
            kd (float): 微分系数
            max_integral (float): 积分项最大值，防止积分饱和，如果为None则不限制
            min_output (float): 输出最小值
            max_output (float): 输出最大值
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.min_output = min_output
        self.max_output = max_output
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update(self, error, dt=None, max_integral=None):
        """
        更新PID控制器，计算控制输出
        
        Args:
            error (float): 当前误差值
            dt (float): 时间间隔，如果为None则尝试自动计算
            max_integral (float): 积分项最大值，防止积分饱和，如果为None则使用初始化时的值
            
        Returns:
            float: PID控制器的输出值
        """
        current_time = time.time()
        
        if dt is None:
            if self.last_time is None:
                dt = 1.0/20  # 默认20Hz
            else:
                dt = current_time - self.last_time
        
        # 累积误差（积分项）
        self.integral += error * dt
        
        # 限制积分项范围，防止积分饱和
        integral_limit = max_integral if max_integral is not None else self.max_integral
        if integral_limit is not None:
            self.integral = max(-integral_limit, min(integral_limit, self.integral))
        
        # 计算误差变化率（微分项）
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        
        # PID公式：输出 = Kp*误差 + Ki*积分项 + Kd*微分项
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 限制输出范围
        output = max(self.min_output, min(self.max_output, output))
        
        # 更新上一次的误差值和时间
        self.previous_error = error
        self.last_time = current_time
        
        return output

    def reset(self):
        """重置PID控制器状态"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def set_params(self, kp=None, ki=None, kd=None):
        """设置PID参数"""
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

# ========================
# 全局变量
# ========================
bridge = None
model = None
target_info_pub = None
velocity_pub = None
attitude_pub = None  # 新增：用于发布姿态控制指令
image_sub = None
attitude_sub = None
running = True
model_loaded = False  # 新增：模型加载状态标志
change_target = True

# 新增：USB相机配置
use_usb_camera = True  # 默认不使用USB相机
usb_camera_device = "/dev/video2"  # USB相机设备路径，可根据实际情况修改
usb_camera = None  # USB相机对象

# 新增：目标类别过滤配置
target_classes = None  # 用于存储要检测的目标类别列表
config_file = "detection_config.yaml"  # YAML配置文件路径

# 图像参数
# 修改为从相机获取实际尺寸，初始值设为默认值
image_width = 480
image_height = 640
FOV_H = 58
FOV_V = 87

# 定义速度增益系数（将PID输出转换为相机坐标系下的实际速度）
desired_target_angle = 25*math.pi/180
default_target_altitude = 3.0  # 目标高度为1米
MINIMUM_ALTITUDE = 1.0  # 默认最低飞行高度为1米
MAXIMUM_ALTITUDE = 3.0  # 默认最大飞行高度为5米
current_altitude = 0.0  # 当前相对高度

# 创建鼠标目标选择器实例
mouse_selector = MouseTargetSelector()

# 无人机姿态参数
current_roll = 0.0
current_pitch = 0.0
current_yaw = 0.0

last_tracked_target_id = None  # 用于存储上一次跟踪的目标ID
initial_height = None       # 存储初始检测到的目标高度
desired_height_temp = None
start_transition_time = None  # 存储开始过渡的时间戳
TRANSITION_DURATION = 3.0  # 过渡持续时间（秒），可根据需要调整


# PID控制器实例
# 最大速度限制（m/s）
MAX_VELOCITY = 5.0
# 控制x向速度（相机系）- 对应水平位置误差
# 高度控制
desired_height_k = 2 # 期望目标高度的比例系数
desired_height = (image_height) // desired_height_k 
VELOCITY_GAIN_X = 5  # m/s 光轴方向
pid_height_controller = PIDController(
    kp=1/((image_height)/desired_height_k), 
    ki=0.1/((image_height)/desired_height_k), 
    kd=0.01/((image_height)/desired_height_k)
)
# 角度控制
VELOCITY_GAIN_XY = 5  # 水平面内速度增益系数
pid_angle_controller = PIDController(
    kp=3/(math.pi), 
    ki=1/((math.pi)), 
    kd=0.1/((math.pi))
)
# 控制y向速度（相机系）- 对应水平位置误差
ANGULAR_GAIN = 90/180*math.pi  # rad/s 偏航角速度
pid_yaw_controller = PIDController(
    kp=2/(image_width), 
    ki=0/(image_width), 
    kd=0/(image_width)
)
# 控制z向速度（相机系）- 对应垂直位置误差
pid_vertical_controller = PIDController(
    kp=1/(image_height), 
    ki=0.000, 
    kd=0.000
)
pid_altitude_controller = PIDController(
    kp=5, 
    ki=0.000, 
    kd=0.000
)


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

# ========================
# 坐标变换器模块（函数实现）
# ========================

# 坐标变换器状态变量
coordinate_transformer_state = {
    'camera_roll': 0.0,
    'camera_pitch': 0.0,
    'camera_yaw': 0.0,
    'current_orientation': {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'w': 1.0
    }  # 使用四元数存储无人机当前姿态
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

def set_uav_attitude(qx, qy, qz, qw):
    """
    设置无人机当前姿态（使用四元数）
    
    Args:
        qx (float): 四元数x分量
        qy (float): 四元数y分量
        qz (float): 四元数z分量
        qw (float): 四元数w分量
    """
    coordinate_transformer_state['current_orientation']['x'] = qx
    coordinate_transformer_state['current_orientation']['y'] = qy
    coordinate_transformer_state['current_orientation']['z'] = qz
    coordinate_transformer_state['current_orientation']['w'] = qw

def apply_filter(current_value, last_value, alpha=0.3):
    """
    应用一阶低通滤波器平滑数值变化
    
    Args:
        current_value (float): 当前计算的值
        last_value (float): 上一次滤波后的值
        alpha (float): 滤波系数，值在0到1之间，越小越平滑但响应越慢
        
    Returns:
        float: 滤波后的值
    """
    return alpha * current_value + (1 - alpha) * last_value

# ========================
# 高度追踪
# ========================
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
    将机身坐标系下的速度转换为ENU坐标系下的速度（使用四元数）
    
    Args:
        body_x (float): 机身坐标系X轴速度
        body_y (float): 机身坐标系Y轴速度
        body_z (float): 机身坐标系Z轴速度
            
    Returns:
        tuple: ENU坐标系下的速度分量 (enu_x, enu_y, enu_z)
    """
    # 获取当前姿态四元数
    qx = coordinate_transformer_state['current_orientation']['x']
    qy = coordinate_transformer_state['current_orientation']['y']
    qz = coordinate_transformer_state['current_orientation']['z']
    qw = coordinate_transformer_state['current_orientation']['w']
    
    # 如果是单位四元数(0,0,0,1)，则直接返回
    if qx == 0.0 and qy == 0.0 and qz == 0.0 and qw == 1.0:
        return body_x, body_y, body_z
    
    # 使用四元数构造旋转矩阵
    # 参考: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    
    # 旋转矩阵 - 从机身坐标系到ENU坐标系
    # 注意: 这里假设四元数表示的是从ENU到机身的旋转，所以我们要用它的转置(共轭)
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    
    # 应用旋转矩阵
    enu_x = r00 * body_x + r01 * body_y + r02 * body_z
    enu_y = r10 * body_x + r11 * body_y + r12 * body_z
    enu_z = r20 * body_x + r21 * body_y + r22 * body_z
    
    return enu_x, enu_y, enu_z

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

filtered_height = 0.0
last_control_signal_x = 0.0
last_enu_wz = 0.0
def height_control(tracking_info):
    """
    订阅YOLO输出信息，计算相机坐标系下的速度和角速度

    Args:
        tracking_info (list): 追踪信息列表
        
    Returns:
        tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) 相机坐标系下的速度和角速度（m/s, rad/s）
    """
    global last_control_signal_x, desired_height, change_target, last_enu_wz
    global last_tracked_target_id, filtered_height, initial_height, desired_height_temp, start_transition_time
    # 获取选中的目标ID
    selected_target_id = mouse_selector.get_selected_target_id()
    
    # 默认速度和角速度为0（无目标时保持静止）
    velocity_x, enu_wz = 0.0, 0.0
    
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
            width, height = target['size']  # 目标宽度和高度

            if change_target:
                filtered_height = height
                initial_height = height  # 更新初始高度为新目标的当前高度
                desired_height_temp = initial_height 
                last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
                print(f"切换跟踪目标至 ID: {selected_target_id}, 新初始高度: {initial_height}")
                
            filtered_height = apply_filter(height, filtered_height, alpha=1.0)
            desired_height_temp = apply_filter(desired_height, desired_height_temp, alpha=0.5)

            # =========================================================

            # 计算图像中心点坐标
            center_image_x = image_width / 2
            center_image_y = image_height / 2
            
            # 计算误差
            error_x = center_image_x - center_x  # 期望的中心x - 实际中心x
            error_y = center_image_y - center_y  # 期望的中心y - 实际中心y
            error_z = desired_height_temp - filtered_height        # 期望高度 - 实际高度
            
            # 根据目标高度动态调整
            # 当目标更近时（检测框更高），使用更大的增益
            # 当目标更远时（检测框更低），使用较小的增益
            normalized_height = height / image_height  # 归一化的高度值(0-1)
            # 使用平方反比关系调整增益，确保近距离时增益更大
            scale_factor = (4*normalized_height ** 2 + 0.75)  # 添加偏移量避免增益过小

            # 获取检测框的四个顶点坐标
            x1, y1, x2, y2 = target['xyxy']

            # edge_threshold = 0.01 # 边缘阈值，距离图像边缘1%范围内认为是边缘
            # if y2 >= image_height * (1- edge_threshold):
            #     scale_factor = 2.0
            #     print(f"目标检测框在屏幕边缘，control_signal_x逐渐增大: {control_signal_x:.4f}")
            #用于控制相机光轴速度
            control_signal_x = pid_height_controller.update(error_z, max_integral=100) * scale_factor
            last_control_signal_x = control_signal_x
            
            # 用于控制yaw的角速度
            # 检查目标是否贴住图像左右边缘
            edge_threshold = 0.05  # 边缘阈值，距离图像边缘5%范围内认为是贴住边缘
            control_signal_y = 0.0  # 此时不使用PID控制
            
            if x1 <= image_width * edge_threshold or x2 >= image_width * (1 - edge_threshold):
                # 当检测框贴住左右边缘时，使用最大的角速度使无人机快速转向，直到目标回到视野中心                
                # 根据目标在哪一侧决定旋转方向，使目标快速远离边缘
                if x1 <= image_width * edge_threshold:
                    # 目标在左侧边缘，需要快速向右转（顺时针，负角速度）使目标移向画面中央
                    enu_wz = min(math.pi,last_enu_wz+0.2)
                    print(f"目标在左侧边缘，快速向右转")
                elif x2 >= image_width * (1 - edge_threshold):
                    # 目标在右侧边缘，需要快速向左转（逆时针，正角速度）使目标移向画面中央
                    enu_wz = max(-math.pi,last_enu_wz-0.2)
                    print(f"目标在右侧边缘，快速向左转")

                print(f"目标贴住边缘，使用最大角速度快速调整: {enu_wz:.4f} rad/s")
            else:
                control_signal_y = pid_yaw_controller.update(error_x, max_integral=100/180*math.pi)
                enu_wz = control_signal_y * scale_factor * ANGULAR_GAIN
            last_enu_wz = enu_wz

            # 将控制信号转换为实际速度和角速度（m/s）
            velocity_x = control_signal_x * VELOCITY_GAIN_X
            
            # 打印控制信息（调试用）
            print(f"跟踪目标ID: {selected_target_id}")
            print(f"当前高度: {filtered_height:.2f}, 临时目标高度: {desired_height_temp:.2f}, 最终目标高度: {desired_height:.2f}")
            print(f"控制误差 - X: {error_x:.2f}, Y: {error_y:.2f}, Z: {error_z:.2f}")
            print(f"控制信号 - X: {control_signal_x:.2f}, Y: {control_signal_y:.2f}, Z: ")
            print(f"相机坐标系速度 - VX: {velocity_x:.3f} m/s")
            print(f"相机坐标系角速度 - WZ: {enu_wz:.3f} rad/s")

        change_target = False
    else:
        change_target = True

    # 仅使用相机坐标系下的velocity_x计算ENU速度
    enu_vx, enu_vy, _ = transform_camera_to_enu(velocity_x, 0, 0)

    # 获取当前高度
    target_altitude = default_target_altitude  # 目标高度为1米
    #计算定高飞行所需要的enu_z轴速度
    error_enu_z = target_altitude - current_altitude
    enu_vz = pid_altitude_controller.update(error_enu_z)

    # 添加最大速度限制
    speed_magnitude = math.sqrt(enu_vx**2 + enu_vy**2 + enu_vz**2)
    if speed_magnitude > MAX_VELOCITY:
        scale_factor = MAX_VELOCITY / speed_magnitude
        enu_vx *= scale_factor
        enu_vy *= scale_factor
        enu_vz *= scale_factor
        # print(f"速度超过限制，已缩放至{MAX_VELOCITY} m/s以内")
        
    return enu_vx, enu_vy, enu_vz, _, _, enu_wz

# ========================
# 角度追踪
# ========================
def pixel_to_angle(pixel_x, pixel_y, image_width, image_height, fov_h=FOV_H, fov_v=FOV_V):
    """
    将像素坐标差转换为相机坐标系下的角度
    
    Args:
        pixel_x (float): 像素水平差值（相对于图像中心）
        pixel_y (float): 像素垂直差值（相对于图像中心）
        image_width (int): 图像宽度
        image_height (int): 图像高度
        fov_h (float): 相机水平视场角（度）
        fov_v (float): 相机垂直视场角（度）
        
    Returns:
        tuple: (yaw_angle, pitch_angle) 相机坐标系下的偏航角和俯仰角（弧度）
    """
    # 计算每个像素对应的角度
    pixel_to_rad_h = math.radians(fov_h) / image_width
    pixel_to_rad_v = math.radians(fov_v) / image_height
    
    # 像素差转换为角度（弧度）
    # 在相机坐标系中：X前，Y左，Z上
    # 正的pixel_x（目标在右侧）对应负的yaw_angle（需要向右转）
    # 正的pixel_y（目标在下方）对应正的pitch_angle（需要向下转）
    yaw_angle = -pixel_x * pixel_to_rad_h
    pitch_angle = pixel_y * pixel_to_rad_v
    
    return yaw_angle, pitch_angle

def angle_to_vector(yaw_angle, pitch_angle):
    """
    将偏航角和俯仰角转换为单位向量
    
    Args:
        yaw_angle (float): 偏航角（弧度）
        pitch_angle (float): 俯仰角（弧度）
        
    Returns:
        tuple: (x, y, z) 单位向量坐标
    """
    # 球坐标系转换为笛卡尔坐标系
    # 相机坐标系：X前，Y左，Z上
    x = math.cos(yaw_angle) * math.cos(pitch_angle)
    y = math.sin(yaw_angle) * math.cos(pitch_angle)
    z = -math.sin(pitch_angle)
    
    # 归一化为单位向量
    magnitude = math.sqrt(x*x + y*y + z*z)
    if magnitude > 0:
        x /= magnitude
        y /= magnitude
        z /= magnitude
    
    return x, y, z

def transform_vector_camera_to_body(cam_vector):
    """
    将相机坐标系下的向量转换到机体坐标系
    
    Args:
        cam_vector (tuple): 相机坐标系下的向量 (x, y, z)
        
    Returns:
        tuple: 机体坐标系下的向量 (x, y, z)
    """
    # 获取相机安装角度
    camera_roll = coordinate_transformer_state['camera_roll']
    camera_pitch = coordinate_transformer_state['camera_pitch']
    camera_yaw = coordinate_transformer_state['camera_yaw']
    
    # 如果相机安装角度都为0，则直接返回
    if camera_roll == 0.0 and camera_pitch == 0.0 and camera_yaw == 0.0:
        return cam_vector
    
    # 预计算三角函数值以提高效率
    cr = math.cos(camera_roll)
    sr = math.sin(camera_roll)
    cp = math.cos(camera_pitch)
    sp = math.sin(camera_pitch)
    cy = math.cos(camera_yaw)
    sy = math.sin(camera_yaw)
    
    # 创建相机到机身的旋转矩阵（根据相机安装角度）
    # 直接计算矩阵元素而不是使用numpy数组
    r00 = cp * cy
    r01 = cp * sy
    r02 = -sp
    r10 = sr * sp * cy - cr * sy
    r11 = sr * sp * sy + cr * cy
    r12 = sr * cp
    r20 = cr * sp * cy + sr * sy
    r21 = cr * sp * sy - sr * cy
    r22 = cr * cp
    
    # 应用旋转矩阵
    body_x = r00 * cam_vector[0] + r01 * cam_vector[1] + r02 * cam_vector[2]
    body_y = r10 * cam_vector[0] + r11 * cam_vector[1] + r12 * cam_vector[2]
    body_z = r20 * cam_vector[0] + r21 * cam_vector[1] + r22 * cam_vector[2]
    
    return (body_x, body_y, body_z)
    
def transform_vector_body_to_enu(body_vector):
    """
    将机体坐标系下的向量转换到ENU世界坐标系（使用四元数）
    
    Args:
        body_vector (tuple): 机体坐标系下的向量 (x, y, z)
        
    Returns:
        tuple: ENU坐标系下的向量 (x, y, z)
    """
    # 获取当前姿态四元数
    qx = coordinate_transformer_state['current_orientation']['x']
    qy = coordinate_transformer_state['current_orientation']['y']
    qz = coordinate_transformer_state['current_orientation']['z']
    qw = coordinate_transformer_state['current_orientation']['w']
    
    # 如果是单位四元数(0,0,0,1)，则直接返回
    if qx == 0.0 and qy == 0.0 and qz == 0.0 and qw == 1.0:
        return body_vector
    
    # 使用四元数构造旋转矩阵
    # 参考: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    
    # 旋转矩阵 - 从机身坐标系到ENU坐标系
    # 注意: 这里假设四元数表示的是从ENU到机身的旋转，所以我们要用它的转置(共轭)
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    
    # 应用旋转矩阵
    enu_x = r00 * body_vector[0] + r01 * body_vector[1] + r02 * body_vector[2]
    enu_y = r10 * body_vector[0] + r11 * body_vector[1] + r12 * body_vector[2]
    enu_z = r20 * body_vector[0] + r21 * body_vector[1] + r22 * body_vector[2]
    
    return (enu_x, enu_y, enu_z)

    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    
    # 旋转矩阵 - 从机身坐标系到ENU坐标系
    # 注意: 这里假设四元数表示的是从ENU到机身的旋转，所以我们要用它的转置(共轭)
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    
    # 应用旋转矩阵
    enu_x = r00 * body_vector[0] + r01 * body_vector[1] + r02 * body_vector[2]
    enu_y = r10 * body_vector[0] + r11 * body_vector[1] + r12 * body_vector[2]
    enu_z = r20 * body_vector[0] + r21 * body_vector[1] + r22 * body_vector[2]
    
    return (enu_x, enu_y, enu_z)

def apply_vector_filter(current_vector, last_vector, alpha=0.3):
    """
    应用向量低通滤波器平滑向量变化
    
    Args:
        current_vector (numpy.ndarray): 当前计算的向量
        last_vector (numpy.ndarray): 上一次滤波后的向量
        alpha (float): 滤波系数，值在0到1之间，越小越平滑但响应越慢
        
    Returns:
        numpy.ndarray: 滤波后的向量
    """
    filtered_vector = alpha * np.array(current_vector) + (1 - alpha) * np.array(last_vector)
    
    # 归一化为单位向量
    magnitude = np.linalg.norm(filtered_vector)
    if magnitude > 0:
        filtered_vector = filtered_vector / magnitude
        
    return filtered_vector

last_angle_diff = math.radians(FOV_V)
last_enu_target_vector = np.array([0.0, 0.0, 0.0])
def angle_control(tracking_info):
    """
    计算速度和角速度控制指令
    
    Args:
        tracking_info (list): 追踪信息列表
        
    Returns:
        tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) ENU坐标系下的速度和角速度
    """
    global last_enu_target_vector, last_tracked_target_id
    global last_angle_diff, change_target, last_enu_wz
        # 获取选中的目标ID
    selected_target_id = mouse_selector.get_selected_target_id()
    
    # 默认速度和角速度为0（无目标时保持静止）
    enu_vx, enu_vy, enu_vz, enu_wx, enu_wy, enu_wz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 

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
            # print(f"目标中心像素坐标: （{center_x}，{center_y}）")
            width, height = target['size']  # 目标宽度和高度
            x1, y1, x2, y2 = target['xyxy'] # 获取检测框的四个顶点坐标
            

            # 计算图像中心点坐标
            center_image_x = image_width / 2
            center_image_y = image_height / 2
            # 计算像素误差（从图像中心到目标）
            pixel_error_x = center_x - center_image_x  # 正值表示目标在图像中心右侧
            pixel_error_y = center_y - center_image_y  # 正值表示目标在图像中心下方
            # print(f"目标距离中心像素差: （{pixel_error_x}，{pixel_error_y}）")
            error_x = center_image_x - center_x
            # 计算角度误差（相机坐标系）
            yaw_error, pitch_error = pixel_to_angle(pixel_error_x, pixel_error_y, 
                                                    image_width, image_height)
            # print(f"目标距离中心角度: （{yaw_error/math.pi*100}，{pitch_error/math.pi*100}）")
            # 将角度误差转换为相机坐标系下的单位向量
            cam_target_vector = angle_to_vector(yaw_error, pitch_error)

            # 将相机坐标系下的目标方向向量转换到ENU世界坐标系
                # 获取无人机当前姿态
            uav_roll = coordinate_transformer_state['current_roll']
            uav_pitch = coordinate_transformer_state['current_pitch']
            uav_yaw = coordinate_transformer_state['current_yaw']    
            # 将相机坐标系下的目标方向向量转换到ENU世界坐标系
            # 获取无人机当前姿态
            body_target_vector = transform_vector_camera_to_body(cam_target_vector)
            enu_vector = transform_vector_body_to_enu(body_target_vector)
            # 对向量应用滤波器
            if change_target:
                last_enu_target_vector = enu_vector 
                last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
            last_enu_target_vector = apply_vector_filter(enu_vector, last_enu_target_vector, alpha=1.0)
            # print(f"滤波前的ENU向量: {enu_vector}，滤波后的ENU向量: {last_enu_target_vector}")

            enu_vector = last_enu_target_vector

            # 根据目标高度动态调整
            # 当目标更近时（检测框更高），使用更大的增益
            # 当目标更远时（检测框更低），使用较小的增益
            normalized_height = height / image_height  # 归一化的高度值(0-1)
            # 使用平方反比关系调整增益，确保近距离时增益更大
            scale_factor = (4*normalized_height ** 2 + 0.75)  # 添加偏移量避免增益过小

            # 计算水平面内的速度，速度大小由enu_vector与水平方向的夹角与目标夹角的差值决定
            # 计算enu_vector与水平方向的夹角
            # 添加数值稳定性检查
            enu_magnitude = np.linalg.norm(enu_vector)
            horizontal_magnitude = np.linalg.norm(enu_vector[:2])
            
            if enu_magnitude > 0 and horizontal_magnitude > 0:
                # 使用clip确保点积结果在[-1, 1]范围内，避免arccos计算错误
                # 计算向量与Z轴的夹角（90度减去与水平面的夹角）
                # 向量与Z轴夹角的余弦值 = |z分量| / 向量模长
                cos_angle_with_z = abs(enu_vector[2]) / enu_magnitude
                # 限制在[-1, 1]范围内，防止计算误差
                cos_angle_with_z = np.clip(cos_angle_with_z, -1.0, 1.0)
                # 计算与Z轴的夹角
                angle_with_z = np.arccos(cos_angle_with_z)
                # 与水平面的夹角 = 90度 - 与Z轴的夹角
                angle_diff = abs(math.pi/2 - angle_with_z)
                
                # 如果检测框下边界接近图像下边界，则保留上次的angle_diff值
                edge_threshold = 0.01 # 边缘阈值，距离图像边缘1%范围内认为是边缘
                if y2 >= image_height * (1- edge_threshold):
                    scale_factor = 2*scale_factor
                
                # 初始化临时目标角度（如果尚未定义）
                if 'temporary_target_angle' not in globals():
                    global temporary_target_angle
                    temporary_target_angle = angle_diff
                elif change_target:
                    # 当切换目标时，更新临时目标角度为当前角度
                    temporary_target_angle = angle_diff
                    
                # 使用简单的低通滤波器使临时目标角度缓慢追踪期望目标角度
                temporary_target_angle = apply_filter(desired_target_angle, temporary_target_angle, alpha=0.01)
                
                # 速度大小由临时目标角度与angle_diff的差值决定
                error_angle = temporary_target_angle - angle_diff
                print(f"绝对水平面角度差: {math.degrees(error_angle):.4f} 期望目标角度: {math.degrees(desired_target_angle):.4f} 临时目标角度: {math.degrees(temporary_target_angle):.4f} 当前角度: {math.degrees(angle_diff):.4f}")


                velocity_xy = pid_angle_controller.update(error_angle, max_integral=10/180*math.pi) * VELOCITY_GAIN_XY * scale_factor

                # enu坐标系下xy平面内的水平速度，大小等于velocity_xy，方向和enu_vector在水平面内的投影方向一致
                # 修复负号问题并添加数值稳定性检查
                normalized_horizontal_x = enu_vector[0] / horizontal_magnitude
                normalized_horizontal_y = enu_vector[1] / horizontal_magnitude
                
                enu_vx = velocity_xy * normalized_horizontal_x
                enu_vy = velocity_xy * normalized_horizontal_y
                print(f"水平速度大小: {velocity_xy:.4f}")
            else:
                # 如果向量为零向量或没有水平分量，则不产生水平运动
                enu_vx, enu_vy = 0.0, 0.0
                print("警告：目标向量为零向量或没有水平分量，无法计算水平速度")

            # 用于控制yaw的角速度
            # 检查目标是否贴住图像左右边缘
            edge_threshold = 0.05  # 边缘阈值，距离图像边缘5%范围内认为是贴住边缘
            control_signal_y = 0.0  # 此时不使用PID控制
            
            if x1 <= image_width * edge_threshold or x2 >= image_width * (1 - edge_threshold):
                # 当检测框贴住左右边缘时，使用最大的角速度使无人机快速转向，直到目标回到视野中心                
                # 根据目标在哪一侧决定旋转方向，使目标快速远离边缘
                if x1 <= image_width * edge_threshold:
                    # 目标在左侧边缘，需要快速向右转（顺时针，负角速度）使目标移向画面中央
                    enu_wz = min(math.pi,last_enu_wz+0.2)
                    print(f"目标在左侧边缘，快速向右转")
                elif x2 >= image_width * (1 - edge_threshold):
                    # 目标在右侧边缘，需要快速向左转（逆时针，正角速度）使目标移向画面中央
                    enu_wz = max(-math.pi,last_enu_wz-0.2)
                    print(f"目标在右侧边缘，快速向左转")

                print(f"目标贴住边缘，使用最大角速度快速调整: {enu_wz:.4f} rad/s")
            else:
                control_signal_y = pid_yaw_controller.update(error_x, max_integral=100/180*math.pi)
                enu_wz = control_signal_y * scale_factor * ANGULAR_GAIN
            last_enu_wz = enu_wz
                
        change_target = False
    else:
        change_target = True
    
    # 获取当前高度
    if default_target_altitude < 1.0:
        target_altitude = 1.0
    else:
        target_altitude = default_target_altitude  # 目标高度
    #计算定高飞行所需要的enu_z轴速度
    error_enu_z = target_altitude - current_altitude
    enu_vz = pid_altitude_controller.update(error_enu_z)
    print(f"当前相对高度: {current_altitude:.2f} 米")

    # 添加最大速度限制
    speed_magnitude = math.sqrt(enu_vx**2 + enu_vy**2 + enu_vz**2)
    if speed_magnitude > MAX_VELOCITY:
        scale_factor = MAX_VELOCITY / speed_magnitude
        enu_vx *= scale_factor
        enu_vy *= scale_factor
        enu_vz *= scale_factor
        print(f"速度超过限制，已缩放至{MAX_VELOCITY} m/s以内")
        
    return enu_vx, enu_vy, enu_vz, enu_wx, enu_wy, enu_wz

# ========================
# 混合追踪
# ========================
def hybrid_control(tracking_info):
    """
    结合angle_control和height_control优点的混合控制方法
    
    使用pid_height_controller控制水平方向的速度，采用pid_angle_controller控制enu_vz的速度，
    角度过大则下降高度，角度过小则抬升高度，当然高度不小于默认高度
    
    Args:
        tracking_info (list): 追踪信息列表
        
    Returns:
        tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) ENU坐标系下的速度和角速度
    """
    global last_enu_target_vector, last_tracked_target_id
    global last_angle_diff, change_target, last_enu_wz, filtered_height, initial_height, desired_height_temp
    global temporary_target_angle
        # 获取选中的目标ID
    selected_target_id = mouse_selector.get_selected_target_id()
    
    # 默认速度和角速度为0（无目标时保持静止）
    enu_vx, enu_vy, enu_vz, enu_wx, enu_wy, enu_wz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 

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
            width, height = target['size']  # 目标宽度和高度
            x1, y1, x2, y2 = target['xyxy'] # 获取检测框的四个顶点坐标

            # =========================================================  
            # 根据目标高度动态调整       
            # 当目标更近时（检测框更高），使用更大的增益
            # 当目标更远时（检测框更低），使用较小的增益
            normalized_height = height / image_height  # 归一化的高度值(0-1)
            # 使用平方反比关系调整增益，确保近距离时增益更大
            scale_factor = (4*normalized_height ** 2 + 0.75)  # 添加偏移量避免增益过小

            # 使用pid_height_controller控制水平方向的速度
            if change_target:
                filtered_height = height
                initial_height = height  # 更新初始高度为新目标的当前高度
                desired_height_temp = initial_height 
                last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
                print(f"切换跟踪目标至 ID: {selected_target_id}, 新初始高度: {initial_height}")
                
            filtered_height = apply_filter(height, filtered_height, alpha=1.0)
            desired_height_temp = apply_filter(desired_height, desired_height_temp, alpha=0.5)

            # 计算误差
            error_z = desired_height_temp - filtered_height        # 期望高度 - 实际高度
            control_signal_x = pid_height_controller.update(error_z, max_integral=100) * scale_factor
            velocity_x = control_signal_x * VELOCITY_GAIN_X
            
            # 仅使用相机坐标系下的velocity_x计算ENU速度
            enu_vx, enu_vy, _ = transform_camera_to_enu(velocity_x, 0, 0)






            # 使用pid_angle_controller控制z轴速度（高度调整）
            # 计算图像中心点坐标
            center_image_x = image_width / 2
            center_image_y = image_height / 2
            # 计算像素误差（从图像中心到目标）
            pixel_error_x = center_x - center_image_x  # 正值表示目标在图像中心右侧
            pixel_error_y = center_y - center_image_y  # 正值表示目标在图像中心下方
            
            # 计算角度误差（相机坐标系）
            yaw_error, pitch_error = pixel_to_angle(pixel_error_x, pixel_error_y, 
                                                    image_width, image_height)
            
            # 将角度误差转换为相机坐标系下的单位向量
            cam_target_vector = angle_to_vector(yaw_error, pitch_error)

            # 将相机坐标系下的目标方向向量转换到ENU世界坐标系
            # 获取无人机当前姿态
            body_target_vector = transform_vector_camera_to_body(cam_target_vector)
            enu_vector = transform_vector_body_to_enu(body_target_vector)
            
            # 对向量应用滤波器
            if change_target:
                last_enu_target_vector = enu_vector 
                last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
            last_enu_target_vector = apply_vector_filter(enu_vector, last_enu_target_vector, alpha=1.0)
            enu_vector = last_enu_target_vector
            
            # 计算目标角度与水平面的夹角
            enu_magnitude = np.linalg.norm(enu_vector)
            if enu_magnitude > 0:
                # 计算向量与Z轴的夹角（90度减去与水平面的夹角）
                # 向量与Z轴夹角的余弦值 = |z分量| / 向量模长
                cos_angle_with_z = abs(enu_vector[2]) / enu_magnitude
                # 限制在[-1, 1]范围内，防止计算误差
                cos_angle_with_z = np.clip(cos_angle_with_z, -1.0, 1.0)
                # 计算与Z轴的夹角
                angle_with_z = np.arccos(cos_angle_with_z)
                # 与水平面的夹角 = 90度 - 与Z轴的夹角
                angle_diff = abs(math.pi/2 - angle_with_z)
                
                # 如果检测框下边界接近图像下边界，则增加scale_factor
                edge_threshold = 0.01 # 边缘阈值，距离图像边缘1%范围内认为是边缘
                if y2 >= image_height * (1- edge_threshold):
                    scale_factor = 2*scale_factor
                
                # 初始化临时目标角度（如果尚未定义）
                if 'temporary_target_angle' not in globals():
                    global temporary_target_angle
                    temporary_target_angle = angle_diff
                elif change_target:
                    # 当切换目标时，更新临时目标角度为当前角度
                    temporary_target_angle = angle_diff
                    
                # 使用简单的低通滤波器使临时目标角度缓慢追踪期望目标角度
                temporary_target_angle = apply_filter(desired_target_angle, temporary_target_angle, alpha=0.01)
                
                # 计算角度误差
                error_angle = temporary_target_angle - angle_diff
                print(f"绝对水平面角度差: {math.degrees(error_angle):.4f} 期望目标角度: {math.degrees(desired_target_angle):.4f} 临时目标角度: {math.degrees(temporary_target_angle):.4f} 当前角度: {math.degrees(angle_diff):.4f}")

                # 角度过大则下降高度，角度过小则抬升高度
                enu_vz = pid_angle_controller.update(error_angle, max_integral=10/180*math.pi) * VELOCITY_GAIN_XY * scale_factor

                print(f"水平速度大小: {velocity_x:.4f} 竖直速度大小：{enu_vz:.4f}")
            else:
                # 如果向量为零向量，则不产生竖直运动
                enu_vz = 0.0
                print("警告：目标向量为零向量或没有水平分量，无法计算竖直速度")



            # 用于控制yaw的角速度
            error_x = center_image_x - center_x
            # 检查目标是否贴住图像左右边缘
            edge_threshold = 0.05  # 边缘阈值，距离图像边缘5%范围内认为是贴住边缘
            control_signal_y = 0.0  # 此时不使用PID控制
            
            if x1 <= image_width * edge_threshold or x2 >= image_width * (1 - edge_threshold):
                # 当检测框贴住左右边缘时，使用最大的角速度使无人机快速转向，直到目标回到视野中心                
                # 根据目标在哪一侧决定旋转方向，使目标快速远离边缘
                if x1 <= image_width * edge_threshold:
                    # 目标在左侧边缘，需要快速向右转（顺时针，负角速度）使目标移向画面中央
                    enu_wz = min(math.pi,last_enu_wz+0.2)
                    print(f"目标在左侧边缘，快速向右转")
                elif x2 >= image_width * (1 - edge_threshold):
                    # 目标在右侧边缘，需要快速向左转（逆时针，正角速度）使目标移向画面中央
                    enu_wz = max(-math.pi,last_enu_wz-0.2)
                    print(f"目标在右侧边缘，快速向左转")

                print(f"目标贴住边缘，使用最大角速度快速调整: {enu_wz:.4f} rad/s")
            else:
                control_signal_y = pid_yaw_controller.update(error_x, max_integral=100/180*math.pi)
                enu_wz = control_signal_y * scale_factor * ANGULAR_GAIN
            last_enu_wz = enu_wz
                
        change_target = False
    else:
        change_target = True
    


    # 确保飞行高度不低于默认高度且不超过最大高度
    error_enu_z = MINIMUM_ALTITUDE - current_altitude
    if error_enu_z > 0:
        enu_vz = enu_vz + pid_altitude_controller.update(error_enu_z)
    
    # 添加最大高度限制
    error_max_altitude = current_altitude - MAXIMUM_ALTITUDE
    if error_max_altitude > 0:
        enu_vz = enu_vz - pid_altitude_controller.update(error_max_altitude)
    
    print(f"当前相对高度: {current_altitude:.2f} 米")



    # 添加最大速度限制
    speed_magnitude = math.sqrt(enu_vx**2 + enu_vy**2 + enu_vz**2)
    if speed_magnitude > MAX_VELOCITY:
        scale_factor = MAX_VELOCITY / speed_magnitude
        enu_vx *= scale_factor
        enu_vy *= scale_factor
        enu_vz *= scale_factor
        print(f"速度超过限制，已缩放至{MAX_VELOCITY} m/s以内")
        
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
    try:
        # 获取图像的实际尺寸
        image_shape = cv_image.shape
        image_height, image_width = image_shape[:2]
        
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
        
        # 计算速度和角速度控制指令
        # velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z = angle_control(tracking_info) # 矢量控制-高空-给定角度
        # velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z = height_control(tracking_info) # 像素控制-平飞-给定高度
        velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z = hybrid_control(tracking_info)
        
        # 应用低通滤波器平滑控制指令
        filtered_vx = apply_filter(velocity_x, last_velocity_x, alpha=0.8)
        filtered_vy = apply_filter(velocity_y, last_velocity_y, alpha=0.8)
        filtered_vz = apply_filter(velocity_z, last_velocity_z, alpha=0.8)
        filtered_wz = apply_filter(angular_z, last_angular_z, alpha=1)
        # 更新上一次滤波后的值
        last_velocity_x, last_velocity_y, last_velocity_z = filtered_vx, filtered_vy, filtered_vz
        last_angular_z = filtered_wz

        # 发布速度和角速度控制指令给PX4飞控
        # publish_velocity_command(velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z)
        publish_velocity_command(filtered_vx, filtered_vy, filtered_vz, angular_x, angular_y, filtered_wz)
            
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
        set_uav_attitude(qx, qy, qz, qw)
        
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
        set_uav_attitude(qx, qy, qz, qw)
        
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
    global current_altitude
    
    try:
        # 从PoseStamped消息中获取z坐标作为相对高度
        current_altitude = msg.pose.position.z
        # print(f"当前相对高度: {current_altitude:.2f} 米")
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
    global pid_height_controller, pid_yaw_controller, pid_vertical_controller, pid_altitude_controller, pid_angle_controller
    # 重置PID控制器状态
    pid_height_controller.reset()
    pid_yaw_controller.reset()
    pid_vertical_controller.reset()
    pid_altitude_controller.reset()
    pid_angle_controller.reset()

def init_coordinate_transformer():
    """初始化坐标变换器"""
    # 相机安装角度参数（弧度）
    # roll: 绕X轴旋转角度, pitch: 绕Y轴旋转角度, yaw: 绕Z轴旋转角度
    camera_roll = 0.0  # 相机绕机体X轴旋转角度（左右倾斜），根据描述设置为0.785弧度
    camera_pitch = math.radians(0)   # 相机绕机体Y轴旋转角度（俯仰角）
    camera_yaw = 0.0     # 相机绕机体Z轴旋转角度（偏航角）
    
    # 设置相机安装角度
    set_camera_orientation(camera_roll, camera_pitch, camera_yaw)

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
