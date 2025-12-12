#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PID控制器模块
用于封装PID控制逻辑
"""

import time


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