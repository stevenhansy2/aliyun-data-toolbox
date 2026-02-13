#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: depthImage16BitConversion.py
日期: 2025-07-24
描述: 
    该脚本用于将8位分离的高低位深度图像复原为16位深度图，并使用无损 mkv 格式保存视频。
    保存后的视频可通过 ffprobe 工具验证位深，例如：
        ffprobe output.mkv

        # 输出示例：
        # Stream #0:0: Video: ffv1 (FFV1 / 0x31564646), gray, 848x480, ...
        # 表示为8位灰度图像

        # Stream #0:0: Video: ffv1 (FFV1 / 0x31564646), gray16le, 848x480, ...
        # 表示为16位小端灰度图像

版本: 1.0.0
Copyright (c) 2025 乐聚机器人
"""

import rosbag
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import os
from tqdm import tqdm
from collections import deque
import traceback
import random
from sklearn.cluster import KMeans

bag_path = "/home/ma/convert/bag/image.bag"
depth_topic_name = "/cam_h/depth/image_raw/compressed"
color_topic_name = "/cam_h/color/image_raw/compressed"
output_video_path = "output.mkv"
depth_temp_dir = "depth_frames"
color_temp_dir = "color_frames"
# 设置帧率
framerate = 30
# 将小于这个值的像素视为边界
threshold_value = 0.2
# 默认摄像头的最远距离值
default_max_range = int(3600)

def gamma_transform(image_16bit, max_range = default_max_range, gamma=0.55):
    """
    Gamma曲线压缩高动态范围，保留暗部细节

    Args:
        image_16bit: 输入的16位图像
        max_range: 最大范围值，默认为3600
        gamma: Gamma值，默认为0.55
    Returns:
        处理后的图像
    """
    threshold = max_range * threshold_value
    # 添加噪声
    max_range = max_range + random.randint(0, 50)
    normalized = image_16bit.astype(np.float32) / 65280.0
    # 应用Gamma变换
    gamma_corrected = np.power(normalized, gamma)
    # 线性放大，并添加噪声
    gamma_corrected = gamma_corrected * max_range + random.randint(0, 50)
    # 阈值处理，将小于阈值的像素设为0
    gamma_corrected = np.where(gamma_corrected < threshold, 0, gamma_corrected)
    return gamma_corrected.astype(np.uint16)

def compress_range(image, min_val = default_max_range/4, max_val = default_max_range*3/4):
    """
    范化图像值域到指定范围

    Args:
        image: 输入的16图像
        min_val: 范化后的最小值,默认为摄像机的1/3范围
        max_val: 范化后的最大值,默认为摄像机的2/3范围
    Returns:
        压缩后的图像
    """
    # 原始值域范围
    original_range = default_max_range
    # 目标值域范围
    target_range = max_val - min_val
    # 线性变换:压缩值域并偏移
    compressed = (image * (target_range / original_range)) + min_val + random.randint(0, int(default_max_range* 0.2))
    # 确保值在目标范围内
    compressed = np.clip(compressed, min_val, max_val)
    
    return compressed.astype(np.uint16)

from sklearn.cluster import KMeans

def smear_by_color_blocks(color_image, fill_image, n_clusters=8):
    # 将彩色图像展平成二维数组，每个像素一个特征向量
    h, w, c = color_image.shape
    pixels = color_image.reshape(-1, c).astype(np.float32)
    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=1)
    labels = kmeans.fit_predict(pixels)
    labels = labels.reshape(h, w)
    result = np.zeros_like(fill_image)
    for label in range(n_clusters):
        mask = (labels == label)
        if np.any(mask):
            block_value = max(np.mean(fill_image[mask])*0.8 - random.randint(0, 20), default_max_range * threshold_value)
            result[mask] = block_value
    return result

def process_depth_image(color_image, depth_image):
    """
    使用彩色图像引导处理深度图像

    Args:
        color_image: 彩色图像
        depth_image: 深度图像
    Return:
        处理后的深度图像
    """
    # 提取深度图的边缘信息
    side_image = filtered_image = cv2.ximgproc.jointBilateralFilter(
        joint = color_image,
        src = depth_image,
        d = 0,
        sigmaColor=5,
        sigmaSpace=9,
        borderType=cv2.BORDER_CONSTANT
    )

    # 二值化处理
    _, binary_image = cv2.threshold(
    src=side_image,
    thresh=127,
    maxval=255,
    type=cv2.THRESH_BINARY
    )

    # 填充深度图的深度,最耗时的过程
    filtered_image = cv2.ximgproc.jointBilateralFilter(
        joint = color_image,
        src = depth_image,
        d = 0,
        sigmaColor = 80,
        sigmaSpace = 12,
        borderType=cv2.BORDER_CONSTANT
    )

    # 线性放大
    filtered_image_16bit = (filtered_image * 256).astype(np.uint16)
    binary_image_16bit = (binary_image * 256).astype(np.uint16)
    # gamma衰减
    gamma_image = gamma_transform(filtered_image_16bit, gamma = 24)
    # 深度变换
    gamma_image = compress_range(gamma_image)
    # 合并边缘信息
    gamma_image = np.where(binary_image_16bit < 10, 0, gamma_image)
    # 使用块填充
    gamma_image = smear_by_color_blocks(color_image, gamma_image)
    # 高斯模糊
    gamma_image = cv2.GaussianBlur(gamma_image, (5,5), sigmaX=2)
    # 再次合并边缘信息
    gamma_image = np.where(binary_image_16bit < 10, 0, gamma_image)

    return gamma_image

def extract_and_save_frames(bag_path, depth_topic_name, color_topic_name, output_dir):
    """
    从ROS包中提取深度和彩色图像帧，并保存处理后的结果
    Args:
        bag_path: ROS包路径
        depth_topic_name: 深度图像话题名称
        color_topic_name: 彩色图像话题名称
        output_dir: 输出目录
    Returns:
        提取的帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    bag = rosbag.Bag(bag_path, 'r')
    # 帧索引
    index = 0
    
    print(f"从{depth_topic_name}中提取压缩信息...")
    # 正确获取消息数量
    total_depth_msgs = bag.get_message_count(topic_filters=[depth_topic_name])
    total_color_msgs = bag.get_message_count(topic_filters=[color_topic_name])
    
    # 分别读取深度和彩色消息流
    depth_msgs = bag.read_messages(topics=[depth_topic_name])
    color_msgs = bag.read_messages(topics=[color_topic_name])
    
    # 创建迭代器
    depth_iter = depth_msgs.__iter__()
    color_iter = color_msgs.__iter__()
    
    # 使用zip同步两个消息流
    for (_, depth_msg, *_), (_, color_msg, *_) in tqdm(
        zip(depth_iter, color_iter), 
        total=min(total_depth_msgs, total_color_msgs),
        desc="提取帧"
    ):
        try:
            # 处理深度图像
            np_arr = np.frombuffer(depth_msg.data, np.uint8)
            depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            # # 处理彩色图像
            # np_arr = np.frombuffer(color_msg.data, np.uint8)
            # color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # # 关联双边滤波处理
            # result_image = process_depth_image(color_image, depth_image)
            result_image = depth_image
            result_filename = os.path.join(output_dir, f"result_{index:05d}.png")
            cv2.imwrite(result_filename, result_image)
            print(f"保存滤波图帧 #{index}")
            index += 1
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
    
    bag.close()
    print(f"成功保存 {index} 个帧对")
    return index

def encode_video_with_ffmpeg(depth_temp_dir, output_video_path, framerate = 25):
    """
    使用FFmpeg将提取的帧编码为视频
    Args:
        depth_temp_dir: 深度图像帧目录
        color_temp_dir: 彩色图像帧目录
        output_video_path: 输出视频路径
        framerate: 视频帧率
    """
    cmd = f"ffmpeg -y -framerate {framerate} -i {depth_temp_dir}/result_%05d.png -c:v ffv1 -pix_fmt gray16le {output_video_path}"
    print(f"运行命令: {cmd}")
    os.system(cmd)

def main():
    total = extract_and_save_frames(bag_path, depth_topic_name, color_topic_name, depth_temp_dir)
    print(f"共提取了深度图{total}帧.")

    if total > 0:
        encode_video_with_ffmpeg(depth_temp_dir, output_video_path, framerate = framerate)
    else:
        print("No frames extracted. Check topic and bag file.")

if __name__ == '__main__':
    main()
