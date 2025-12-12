#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
鼠标目标选择器模块
用于处理鼠标点击选择跟踪目标的功能
"""

import cv2
import time

class MouseTargetSelector:
    """
    鼠标目标选择器类
    处理鼠标点击事件，选择要跟踪的目标
    """
    
    def __init__(self):
        """初始化鼠标目标选择器"""
        self.selected_target_id = None  # 当前选中的目标ID
        self.last_seen_target_box = None
        self.target_selection_message = "Please click on the target"  # 显示在图像上的提示信息
        self.message_display_time = 0  # 消息显示时间戳
        self.tracking_info_cache = []  # 缓存跟踪信息供鼠标回调使用
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数，用于选择跟踪目标
        
        Args:
            event: 鼠标事件
            x, y: 鼠标坐标
            flags: 鼠标事件标志
            param: 附加参数
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.select_target_by_coordinates(x, y)

        elif event == cv2.EVENT_MBUTTONDOWN:
            # 中键点击，取消选择
            self._cancel_selection()
            
    def select_target_by_coordinates(self, x, y):
        """
        通过坐标选择目标
        
        Args:
            x (int): x坐标
            y (int): y坐标
        """
        # 查找点击位置对应的目标
        if self.tracking_info_cache:
            found_target = False
            for info in self.tracking_info_cache:
                x1, y1, x2, y2 = info['xyxy']
                # 检查点击位置是否在边界框内
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._select_target(info['id'],info['xyxy'])
                    found_target = True
                    break
            
            if not found_target:
                self._no_target_found()

    def _select_target(self, target_id, target_box):
        """
        选择目标的内部方法
        
        Args:
            target_id (int): 要选择的目标ID
        """
        self.selected_target_id = target_id
        self.last_seen_target_box= target_box  # 更新最后看到的目标ID
        self.target_selection_message = f"Tracking ID: {self.selected_target_id}"
        self.message_display_time = time.time()
        print(f"Tracking ID: {self.selected_target_id}")

    def _cancel_selection(self):
        """取消选择的内部方法"""
        self.selected_target_id = None
        self.target_selection_message = "Selection cancelled"
        self.message_display_time = time.time()
        print("Selection cancelled")
        
    def _no_target_found(self):
        """未找到目标时的处理方法"""
        self.selected_target_id = None
        self.target_selection_message = "Please click on the target"
        self.message_display_time = time.time()
        print("Please click on the target")
        # pass
    
    def draw_tracking_results(self, frame, tracking_info):
        """
        在图像上绘制跟踪结果，选中的目标框更粗
        
        Args:
            frame: 图像帧
            tracking_info: 跟踪信息列表
            
        Returns:
            frame: 绘制了跟踪结果的图像帧
        """
        # 绘制所有检测到的目标
        for info in tracking_info:
            x1, y1, x2, y2 = map(int, info['xyxy'])
            target_id = info['id']
            
            # 根据是否选中目标，设置框的粗细
            if target_id == self.selected_target_id:
                # 选中的目标框更粗（绿色）
                color = (0, 255, 0)
                thickness = 5
            
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)                
        
        return frame
    
    def update_tracking_info(self, tracking_info):
        """
        更新跟踪信息缓存
        
        Args:
            tracking_info: 跟踪信息列表
        """
        self.tracking_info_cache = tracking_info
    
    def check_target_exists(self):
        """
        检查选中的目标是否还存在
        
        Returns:
            bool: 如果目标存在返回True，否则返回False
        """
        # 如果当前有选中的目标
        if self.selected_target_id is not None:
            # 检查目标是否还在当前帧中
            target_exists = any(info['id'] == self.selected_target_id for info in self.tracking_info_cache)
            
            if not target_exists:
                # 获取上一次选中目标的边界框信息
                x1, y1, x2, y2 = self.last_seen_target_box
                # 计算边界框的中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # 使用中心点坐标重新选择目标
                self.select_target_by_coordinates(center_x, center_y)
                
                # # 目标不存在，直接取消选中
                # print(f"Target ID {self.selected_target_id} lost")
                # self.selected_target_id = None
    
    def draw_selection_message(self, frame):
        """
        在图像上绘制选择提示信息
        
        Args:
            frame: 图像帧
            
        Returns:
            frame: 绘制了提示信息的图像帧
        """
        # 显示主要提示信息
        cv2.putText(frame, self.target_selection_message, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) #红色
        
        return frame
    
    def get_selected_target_id(self):
        """
        获取当前选中的目标ID
        
        Returns:
            int or None: 选中的目标ID，如果没有选中目标则返回None
        """
        return self.selected_target_id
    
    def get_selected_bbox(self):
        """
        获取当前选中目标的边界框坐标
        
        Returns:
            list or None: 选中目标的边界框坐标 [x1, y1, x2, y2]，如果没有选中目标则返回None
        """
        if self.selected_target_id is not None:
            for info in self.tracking_info_cache:
                if info['id'] == self.selected_target_id:
                    return info['xyxy']
        return None
    
    def reset_selection(self):
        """重置选择状态"""
        self.selected_target_id = None
        self.target_selection_message = "Please click on the target to select for tracking"
        self.message_display_time = time.time()