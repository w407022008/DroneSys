#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
控制系统模块
用于封装控制系统的相关功能
"""

import math
import numpy as np
from pid_controller import PIDController


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

class ControlSystem:
    """
    控制系统类
    封装无人机控制系统的相关功能
    """

    def __init__(self, image_width=480, image_height=640):
        """初始化控制系统"""
        self.image_width = image_width
        self.image_height = image_height
        
        self.camera_roll = 0.0
        self.camera_pitch = 0.0
        self.camera_yaw = 0.0
        self.current_orientation = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'w': 1.0
        }  # 使用四元数存储无人机当前姿态
        
        # PID控制器
        self.pid_height_controller = PIDController(
            kp=1/((self.image_height)/2),
            ki=0.1/((self.image_height)/2),
            kd=0.01/((self.image_height)/2)
        )
        
        self.pid_angle_controller = PIDController(
            kp=3/(math.pi),
            ki=1/((math.pi)),
            kd=0.1/((math.pi))
        )
        
        self.pid_yaw_controller = PIDController(
            kp=2/(self.image_width),
            ki=0.5/(self.image_width),
            kd=0/(self.image_width)
        )
        
        self.pid_vertical_controller = PIDController(
            kp=1/(self.image_height),
            ki=0.000,
            kd=0.000
        )
        
        self.pid_altitude_controller = PIDController(
            kp=5,
            ki=0.000,
            kd=0.000
        )
        
        # 控制参数
        self.desired_target_angle = 25 * math.pi / 180
        self.default_target_altitude = 3.0
        self.MINIMUM_ALTITUDE = 1.0
        self.MAXIMUM_ALTITUDE = 3.0
        self.MAX_VELOCITY = 5.0
        self.VELOCITY_GAIN_X = 5
        self.VELOCITY_GAIN_XY = 5
        self.ANGULAR_GAIN = 90/180*math.pi
        
        # 状态变量
        self.current_altitude = 0.0
        self.change_target = True
        self.last_enu_wz = 0.0
        self.last_control_signal_x = 0.0
        self.last_tracked_target_id = None
        self.filtered_height = 0.0
        self.initial_height = None
        self.desired_height_temp = None
        self.last_angle_diff = math.radians(87)  # FOV_V
        self.last_enu_target_vector = np.array([0.0, 0.0, 0.0])

    def set_camera_resolution(self, image_width, image_height):
        """
        设置相机的分辨率

        Args:
            image_width (int): 相机的宽度
            image_height (int): 相机的高度
        """
        self.image_width = image_width
        self.image_height = image_height

    def set_current_altitude(self, altitude):
        """
        设置当前相对高度

        Args:
            altitude (float): 当前相对高度
        """
        self.current_altitude = altitude

    def set_camera_orientation(self, roll, pitch, yaw):
        """
        设置相机相对于机身的安装角度

        Args:
            roll (float): 相机绕机体X轴旋转角度
            pitch (float): 相机绕机体Y轴旋转角度
            yaw (float): 相机绕机体Z轴旋转角度
        """
        self.camera_roll = roll
        self.camera_pitch = pitch
        self.camera_yaw = yaw

    def set_uav_attitude(self, qx, qy, qz, qw):
        """
        设置无人机当前姿态（使用四元数）

        Args:
            qx (float): 四元数x分量
            qy (float): 四元数y分量
            qz (float): 四元数z分量
            qw (float): 四元数w分量
        """
        self.current_orientation['x'] = qx
        self.current_orientation['y'] = qy
        self.current_orientation['z'] = qz
        self.current_orientation['w'] = qw

    def transform_camera_to_body(self, cam_x, cam_y, cam_z):
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
        if (self.camera_roll == 0.0 and 
            self.camera_pitch == 0.0 and 
            self.camera_yaw == 0.0):
            return cam_x, cam_y, cam_z

        # 绕Z轴旋转（偏航）
        cos_yaw = math.cos(self.camera_yaw)
        sin_yaw = math.sin(self.camera_yaw)
        temp_x = cam_x * cos_yaw - cam_y * sin_yaw
        temp_y = cam_x * sin_yaw + cam_y * cos_yaw
        temp_z = cam_z

        # 绕Y轴旋转（俯仰）
        cos_pitch = math.cos(self.camera_pitch)
        sin_pitch = math.sin(self.camera_pitch)
        body_x = temp_x * cos_pitch + temp_z * sin_pitch
        body_y = temp_y
        body_z = -temp_x * sin_pitch + temp_z * cos_pitch

        # 绕X轴旋转（滚转）
        cos_roll = math.cos(self.camera_roll)
        sin_roll = math.sin(self.camera_roll)
        final_x = body_x
        final_y = body_y * cos_roll - body_z * sin_roll
        final_z = body_y * sin_roll + body_z * cos_roll

        return final_x, final_y, final_z

    def transform_body_to_enu(self, body_x, body_y, body_z):
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
        qx = self.current_orientation['x']
        qy = self.current_orientation['y']
        qz = self.current_orientation['z']
        qw = self.current_orientation['w']

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

    def transform_camera_to_enu(self, cam_x, cam_y, cam_z):
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
        body_x, body_y, body_z = self.transform_camera_to_body(cam_x, cam_y, cam_z)

        # 再从机身坐标系转换到ENU坐标系
        enu_x, enu_y, enu_z = self.transform_body_to_enu(body_x, body_y, body_z)
        return enu_x, enu_y, enu_z

    def height_control(self, tracking_info, mouse_selector):
        """
        订阅YOLO输出信息，计算相机坐标系下的速度和角速度

        Args:
            tracking_info (list): 追踪信息列表
            mouse_selector: 鼠标选择器实例

        Returns:
            tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) 相机坐标系下的速度和角速度（m/s, rad/s）
        """
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

                if self.change_target:
                    self.filtered_height = height
                    self.initial_height = height  # 更新初始高度为新目标的当前高度
                    self.desired_height_temp = self.initial_height
                    self.last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
                    print(f"切换跟踪目标至 ID: {selected_target_id}, 新初始高度: {self.initial_height}")

                self.filtered_height = apply_filter(height, self.filtered_height, alpha=1.0)
                self.desired_height_temp = apply_filter(
                    (self.image_height) // 2, self.desired_height_temp, alpha=0.5)

                # =========================================================

                # 计算图像中心点坐标
                center_image_x = self.image_width / 2
                center_image_y = self.image_height / 2

                # 计算误差
                error_x = center_image_x - center_x  # 期望的中心x - 实际中心x
                error_y = center_image_y - center_y  # 期望的中心y - 实际中心y
                error_z = self.desired_height_temp - self.filtered_height  # 期望高度 - 实际高度

                # 根据目标高度动态调整
                # 当目标更近时（检测框更高），使用更大的增益
                # 当目标更远时（检测框更低），使用较小的增益
                normalized_height = height / self.image_height  # 归一化的高度值(0-1)
                # 使用平方反比关系调整增益，确保近距离时增益更大
                scale_factor = (4*normalized_height ** 2 + 0.75)  # 添加偏移量避免增益过小

                # 获取检测框的四个顶点坐标
                x1, y1, x2, y2 = target['xyxy']

                # 用于控制相机光轴速度
                control_signal_x = self.pid_height_controller.update(error_z, max_integral=100) * scale_factor
                self.last_control_signal_x = control_signal_x

                # 用于控制yaw的角速度
                # 检查目标是否贴住图像左右边缘
                edge_threshold = 0.05  # 边缘阈值，距离图像边缘5%范围内认为是贴住边缘
                control_signal_y = 0.0  # 此时不使用PID控制

                if x1 <= self.image_width * edge_threshold or x2 >= self.image_width * (1 - edge_threshold):
                    # 当检测框贴住左右边缘时，使用最大的角速度使无人机快速转向，直到目标回到视野中心
                    # 根据目标在哪一侧决定旋转方向，使目标快速远离边缘
                    if x1 <= self.image_width * edge_threshold:
                        # 目标在左侧边缘，需要快速向右转（顺时针，负角速度）使目标移向画面中央
                        enu_wz = min(math.pi, self.last_enu_wz+0.1)
                        print(f"目标在左侧边缘，快速向右转")
                    elif x2 >= self.image_width * (1 - edge_threshold):
                        # 目标在右侧边缘，需要快速向左转（逆时针，正角速度）使目标移向画面中央
                        enu_wz = max(-math.pi, self.last_enu_wz-0.1)
                        print(f"目标在右侧边缘，快速向左转")

                    print(f"目标贴住边缘，使用最大角速度快速调整: {enu_wz:.4f} rad/s")
                else:
                    control_signal_y = self.pid_yaw_controller.update(error_x, max_integral=100/180*math.pi)
                    enu_wz = control_signal_y * scale_factor * self.ANGULAR_GAIN
                self.last_enu_wz = enu_wz

                # 将控制信号转换为实际速度和角速度（m/s）
                velocity_x = control_signal_x * self.VELOCITY_GAIN_X

                # 打印控制信息（调试用）
                print(f"跟踪目标ID: {selected_target_id}")
                print(f"当前高度: {self.filtered_height:.2f}, 临时目标高度: {self.desired_height_temp:.2f}, 最终目标高度: {(self.image_height) // 2:.2f}")
                print(f"控制误差 - X: {error_x:.2f}, Y: {error_y:.2f}, Z: {error_z:.2f}")
                print(f"控制信号 - X: {control_signal_x:.2f}, Y: {control_signal_y:.2f}, Z: ")
                print(f"相机坐标系速度 - VX: {velocity_x:.3f} m/s")
                print(f"相机坐标系角速度 - WZ: {enu_wz:.3f} rad/s")

            self.change_target = False
        else:
            self.change_target = True

        # 仅使用相机坐标系下的velocity_x计算ENU速度
        enu_vx, enu_vy, _ = self.transform_camera_to_enu(velocity_x, 0, 0)

        # 获取当前高度
        target_altitude = self.default_target_altitude  # 目标高度为1米
        # 计算定高飞行所需要的enu_z轴速度
        error_enu_z = target_altitude - self.current_altitude
        enu_vz = self.pid_altitude_controller.update(error_enu_z)

        # 添加最大速度限制
        speed_magnitude = math.sqrt(enu_vx**2 + enu_vy**2 + enu_vz**2)
        if speed_magnitude > self.MAX_VELOCITY:
            scale_factor = self.MAX_VELOCITY / speed_magnitude
            enu_vx *= scale_factor
            enu_vy *= scale_factor
            enu_vz *= scale_factor
            # print(f"速度超过限制，已缩放至{self.MAX_VELOCITY} m/s以内")

        return enu_vx, enu_vy, enu_vz, 0, 0, enu_wz

    def pixel_to_angle(self, pixel_x, pixel_y, image_width, image_height, fov_h=58, fov_v=87):
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

    def angle_to_vector(self, yaw_angle, pitch_angle):
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

    def transform_vector_camera_to_body(self, cam_vector):
        """
        将相机坐标系下的向量转换到机体坐标系

        Args:
            cam_vector (tuple): 相机坐标系下的向量 (x, y, z)

        Returns:
            tuple: 机体坐标系下的向量 (x, y, z)
        """
        # 获取相机安装角度
        camera_roll = self.camera_roll
        camera_pitch = self.camera_pitch
        camera_yaw = self.camera_yaw

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

    def transform_vector_body_to_enu(self, body_vector):
        """
        将机体坐标系下的向量转换到ENU世界坐标系（使用四元数）

        Args:
            body_vector (tuple): 机体坐标系下的向量 (x, y, z)

        Returns:
            tuple: ENU坐标系下的向量 (x, y, z)
        """
        # 获取当前姿态四元数
        qx = self.current_orientation['x']
        qy = self.current_orientation['y']
        qz = self.current_orientation['z']
        qw = self.current_orientation['w']

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

    def angle_control(self, tracking_info, mouse_selector):
        """
        计算速度和角速度控制指令

        Args:
            tracking_info (list): 追踪信息列表
            mouse_selector: 鼠标选择器实例

        Returns:
            tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) ENU坐标系下的速度和角速度
        """
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
                x1, y1, x2, y2 = target['xyxy']  # 获取检测框的四个顶点坐标

                # 计算图像中心点坐标
                center_image_x = self.image_width / 2
                center_image_y = self.image_height / 2
                # 计算像素误差（从图像中心到目标）
                pixel_error_x = center_x - center_image_x  # 正值表示目标在图像中心右侧
                pixel_error_y = center_y - center_image_y  # 正值表示目标在图像中心下方
                # print(f"目标距离中心像素差: （{pixel_error_x}，{pixel_error_y}）")
                error_x = center_image_x - center_x
                # 计算角度误差（相机坐标系）
                yaw_error, pitch_error = self.pixel_to_angle(
                    pixel_error_x, pixel_error_y, self.image_width, self.image_height)

                # print(f"目标距离中心角度: （{yaw_error/math.pi*100}，{pitch_error/math.pi*100}）")
                # 将角度误差转换为相机坐标系下的单位向量
                cam_target_vector = self.angle_to_vector(yaw_error, pitch_error)

                # 将相机坐标系下的目标方向向量转换到ENU世界坐标系
                # 获取无人机当前姿态
                body_target_vector = self.transform_vector_camera_to_body(cam_target_vector)
                enu_vector = self.transform_vector_body_to_enu(body_target_vector)
                # 对向量应用滤波器
                if self.change_target:
                    self.last_enu_target_vector = enu_vector
                    self.last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
                self.last_enu_target_vector = apply_vector_filter(enu_vector, self.last_enu_target_vector, alpha=1.0)
                # print(f"滤波前的ENU向量: {enu_vector}，滤波后的ENU向量: {self.last_enu_target_vector}")

                enu_vector = self.last_enu_target_vector

                # 根据目标高度动态调整
                # 当目标更近时（检测框更高），使用更大的增益
                # 当目标更远时（检测框更低），使用较小的增益
                normalized_height = height / self.image_height  # 归一化的高度值(0-1)
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
                    edge_threshold = 0.01  # 边缘阈值，距离图像边缘1%范围内认为是边缘
                    if y2 >= self.image_height * (1 - edge_threshold):
                        scale_factor = 2*scale_factor

                    # 初始化临时目标角度（如果尚未定义）
                    if not hasattr(self, 'temporary_target_angle'):
                        self.temporary_target_angle = angle_diff
                    elif self.change_target:
                        # 当切换目标时，更新临时目标角度为当前角度
                        self.temporary_target_angle = angle_diff

                    # 使用简单的低通滤波器使临时目标角度缓慢追踪期望目标角度
                    self.temporary_target_angle = apply_filter(
                        self.desired_target_angle, self.temporary_target_angle, alpha=0.01)

                    # 速度大小由临时目标角度与angle_diff的差值决定
                    error_angle = self.temporary_target_angle - angle_diff
                    print(f"绝对水平面角度差: {math.degrees(error_angle):.4f} 期望目标角度: {math.degrees(self.desired_target_angle):.4f} 临时目标角度: {math.degrees(self.temporary_target_angle):.4f} 当前角度: {math.degrees(angle_diff):.4f}")

                    velocity_xy = self.pid_angle_controller.update(
                        error_angle, max_integral=10/180*math.pi) * self.VELOCITY_GAIN_XY * scale_factor

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

                if x1 <= self.image_width * edge_threshold or x2 >= self.image_width * (1 - edge_threshold):
                    # 当检测框贴住左右边缘时，使用最大的角速度使无人机快速转向，直到目标回到视野中心
                    # 根据目标在哪一侧决定旋转方向，使目标快速远离边缘
                    if x1 <= self.image_width * edge_threshold:
                        # 目标在左侧边缘，需要快速向右转（顺时针，负角速度）使目标移向画面中央
                        enu_wz = min(math.pi, self.last_enu_wz+0.1)
                        print(f"目标在左侧边缘，快速向右转")
                    elif x2 >= self.image_width * (1 - edge_threshold):
                        # 目标在右侧边缘，需要快速向左转（逆时针，正角速度）使目标移向画面中央
                        enu_wz = max(-math.pi, self.last_enu_wz-0.1)
                        print(f"目标在右侧边缘，快速向左转")

                    print(f"目标贴住边缘，使用最大角速度快速调整: {enu_wz:.4f} rad/s")
                else:
                    control_signal_y = self.pid_yaw_controller.update(error_x, max_integral=100/180*math.pi)
                    enu_wz = control_signal_y * scale_factor * self.ANGULAR_GAIN
                self.last_enu_wz = enu_wz

            self.change_target = False
        else:
            self.change_target = True

        # 获取当前高度
        if self.default_target_altitude < 1.0:
            target_altitude = 1.0
        else:
            target_altitude = self.default_target_altitude  # 目标高度
        # 计算定高飞行所需要的enu_z轴速度
        error_enu_z = target_altitude - self.current_altitude
        enu_vz = self.pid_altitude_controller.update(error_enu_z)
        print(f"当前相对高度: {self.current_altitude:.2f} 米")

        # 添加最大速度限制
        speed_magnitude = math.sqrt(enu_vx**2 + enu_vy**2 + enu_vz**2)
        if speed_magnitude > self.MAX_VELOCITY:
            scale_factor = self.MAX_VELOCITY / speed_magnitude
            enu_vx *= scale_factor
            enu_vy *= scale_factor
            enu_vz *= scale_factor
            print(f"速度超过限制，已缩放至{self.MAX_VELOCITY} m/s以内")

        return enu_vx, enu_vy, enu_vz, enu_wx, enu_wy, enu_wz

    def hybrid_control(self, tracking_info, mouse_selector):
        """
        结合angle_control和height_control优点的混合控制方法

        使用pid_height_controller控制水平方向的速度，采用pid_angle_controller控制enu_vz的速度，
        角度过大则下降高度，角度过小则抬升高度，当然高度不小于默认高度

        Args:
            tracking_info (list): 追踪信息列表
            mouse_selector: 鼠标选择器实例

        Returns:
            tuple: (velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z) ENU坐标系下的速度和角速度
        """
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
                x1, y1, x2, y2 = target['xyxy']  # 获取检测框的四个顶点坐标

                # =========================================================
                # 根据目标高度动态调整
                # 当目标更近时（检测框更高），使用更大的增益
                # 当目标更远时（检测框更低），使用较小的增益
                normalized_height = height / self.image_height  # 归一化的高度值(0-1)
                # 使用平方反比关系调整增益，确保近距离时增益更大
                scale_factor = (4*normalized_height ** 2 + 0.75)  # 添加偏移量避免增益过小

                # 使用pid_height_controller控制水平方向的速度
                if self.change_target:
                    self.filtered_height = height
                    self.initial_height = height  # 更新初始高度为新目标的当前高度
                    self.desired_height_temp = self.initial_height
                    self.last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
                    print(f"切换跟踪目标至 ID: {selected_target_id}, 新初始高度: {self.initial_height}")

                self.filtered_height = apply_filter(height, self.filtered_height, alpha=1.0)
                self.desired_height_temp = apply_filter((self.image_height) // 2, self.desired_height_temp, alpha=0.5)

                # 计算误差
                error_z = self.desired_height_temp - self.filtered_height  # 期望高度 - 实际高度
                control_signal_x = self.pid_height_controller.update(error_z, max_integral=100) * scale_factor
                velocity_x = control_signal_x * self.VELOCITY_GAIN_X

                # 仅使用相机坐标系下的velocity_x计算ENU速度
                enu_vx, enu_vy, _ = self.transform_camera_to_enu(velocity_x, 0, 0)

                # 使用pid_angle_controller控制z轴速度（高度调整）
                # 计算图像中心点坐标
                center_image_x = self.image_width / 2
                center_image_y = self.image_height / 2
                # 计算像素误差（从图像中心到目标）
                pixel_error_x = center_x - center_image_x  # 正值表示目标在图像中心右侧
                pixel_error_y = center_y - center_image_y  # 正值表示目标在图像中心下方

                # 计算角度误差（相机坐标系）
                yaw_error, pitch_error = self.pixel_to_angle(
                    pixel_error_x, pixel_error_y, self.image_width, self.image_height)

                # 将角度误差转换为相机坐标系下的单位向量
                cam_target_vector = self.angle_to_vector(yaw_error, pitch_error)

                # 将相机坐标系下的目标方向向量转换到ENU世界坐标系
                # 获取无人机当前姿态
                body_target_vector = self.transform_vector_camera_to_body(cam_target_vector)
                enu_vector = self.transform_vector_body_to_enu(body_target_vector)

                # 对向量应用滤波器
                if self.change_target:
                    self.last_enu_target_vector = enu_vector
                    self.last_tracked_target_id = selected_target_id  # 更新上一次跟踪的目标ID
                self.last_enu_target_vector = apply_vector_filter(enu_vector, self.last_enu_target_vector, alpha=1.0)
                enu_vector = self.last_enu_target_vector

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
                    edge_threshold = 0.01  # 边缘阈值，距离图像边缘1%范围内认为是边缘
                    if y2 >= self.image_height * (1 - edge_threshold):
                        scale_factor = 2*scale_factor

                    # 初始化临时目标角度（如果尚未定义）
                    if not hasattr(self, 'temporary_target_angle'):
                        self.temporary_target_angle = angle_diff
                    elif self.change_target:
                        # 当切换目标时，更新临时目标角度为当前角度
                        self.temporary_target_angle = angle_diff

                    # 使用简单的低通滤波器使临时目标角度缓慢追踪期望目标角度
                    self.temporary_target_angle = apply_filter(
                        self.desired_target_angle, self.temporary_target_angle, alpha=0.01)

                    # 计算角度误差
                    error_angle = self.temporary_target_angle - angle_diff
                    print(f"绝对水平面角度差: {math.degrees(error_angle):.4f} 期望目标角度: {math.degrees(self.desired_target_angle):.4f} 临时目标角度: {math.degrees(self.temporary_target_angle):.4f} 当前角度: {math.degrees(angle_diff):.4f}")

                    # 角度过大则下降高度，角度过小则抬升高度
                    enu_vz = self.pid_angle_controller.update(
                        error_angle, max_integral=10/180*math.pi) * self.VELOCITY_GAIN_XY * scale_factor

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

                if x1 <= self.image_width * edge_threshold or x2 >= self.image_width * (1 - edge_threshold):
                    # 当检测框贴住左右边缘时，使用最大的角速度使无人机快速转向，直到目标回到视野中心
                    # 根据目标在哪一侧决定旋转方向，使目标快速远离边缘
                    if x1 <= self.image_width * edge_threshold:
                        # 目标在左侧边缘，需要快速向右转（顺时针，负角速度）使目标移向画面中央
                        enu_wz = min(math.pi, self.last_enu_wz+0.1)
                        print(f"目标在左侧边缘，快速向右转")
                    elif x2 >= self.image_width * (1 - edge_threshold):
                        # 目标在右侧边缘，需要快速向左转（逆时针，正角速度）使目标移向画面中央
                        enu_wz = max(-math.pi, self.last_enu_wz-0.1)
                        print(f"目标在右侧边缘，快速向左转")

                    print(f"目标贴住边缘，使用最大角速度快速调整: {enu_wz:.4f} rad/s")
                else:
                    print(f"yaw方向像素差: {error_x:.4f}")
                    control_signal_y = self.pid_yaw_controller.update(error_x, max_integral=100/180*math.pi)
                    enu_wz = control_signal_y * scale_factor * self.ANGULAR_GAIN
                self.last_enu_wz = enu_wz

            self.change_target = False
        else:
            self.change_target = True

        # 确保飞行高度不低于默认高度且不超过最大高度
        error_enu_z = self.MINIMUM_ALTITUDE - self.current_altitude
        if error_enu_z > 0:
            enu_vz = enu_vz + self.pid_altitude_controller.update(error_enu_z)

        # 添加最大高度限制
        error_max_altitude = self.current_altitude - self.MAXIMUM_ALTITUDE
        if error_max_altitude > 0:
            enu_vz = enu_vz - self.pid_altitude_controller.update(error_max_altitude)

        print(f"当前相对高度: {self.current_altitude:.2f} 米")

        # 添加最大速度限制
        speed_magnitude = math.sqrt(enu_vx**2 + enu_vy**2 + enu_vz**2)
        if speed_magnitude > self.MAX_VELOCITY:
            scale_factor = self.MAX_VELOCITY / speed_magnitude
            enu_vx *= scale_factor
            enu_vy *= scale_factor
            enu_vz *= scale_factor
            print(f"速度超过限制，已缩放至{self.MAX_VELOCITY} m/s以内")

        return enu_vx, enu_vy, enu_vz, enu_wx, enu_wy, enu_wz

    def reset(self):
        """重置控制系统状态"""
        self.change_target = True
        self.last_enu_wz = 0.0
        self.last_control_signal_x = 0.0
        self.filtered_height = 0.0
        self.initial_height = None
        self.desired_height_temp = None
        self.last_angle_diff = math.radians(87)
        self.last_enu_target_vector = np.array([0.0, 0.0, 0.0])
        
        # 重置PID控制器
        self.pid_height_controller.reset()
        self.pid_angle_controller.reset()
        self.pid_yaw_controller.reset()
        self.pid_vertical_controller.reset()
        self.pid_altitude_controller.reset()