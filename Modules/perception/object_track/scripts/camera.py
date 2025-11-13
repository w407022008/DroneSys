#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB灰度相机V4L2读取脚本
通过V4L2读取USB灰度相机并转换为OpenCV RGB格式
"""
import cv2
import numpy as np

def main():
    # 设置摄像头设备（根据实际情况修改设备号）
    device = "/dev/video4"
    
    # 创建VideoCapture对象
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print(f"无法打开摄像头设备 {device}")
        return
    
    # 设置摄像头参数（根据相机支持的分辨率调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 设置视频格式为MJPEG
    # 注意：MJPEG的fourcc编码是'MJPG'
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # 检查是否成功设置MJPEG格式
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # 将整数转换为字符表示
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"实际使用的视频格式: {fourcc_str}")

    print("开始采集图像，按 'q' 键退出...")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法获取帧")
            break

        # 显示图像
        cv2.imshow("USB Camera", frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
