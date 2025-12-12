#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标追踪器模块
用于封装目标追踪相关功能
"""

import numpy as np
import torch
from ultralytics.trackers.bot_sort import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace


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