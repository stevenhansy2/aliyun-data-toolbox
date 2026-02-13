#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import os
from tqdm import tqdm
import traceback
import random
from sklearn.cluster import MiniBatchKMeans
import multiprocessing as mp
from multiprocessing import Queue, Process, cpu_count
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import threading
import matplotlib.pyplot as plt

# 配置参数
bag_path = "./1.bag"
depth_topic_name = "/cam_h/depth/image_raw/compressed"
color_topic_name = "/cam_h/color/image_raw/compressed"
output_video_path = "output_new.mkv"
depth_temp_dir = "depth_frames"
color_temp_dir = "color_frames"
framerate = 30
threshold_value = 0.2
default_max_range = int(3600)

# NUM_WORKERS = cpu_count() - 1
# UEUE_SIZE = 100
# BATCH_SIZE = 10


def gamma_transform(image_16bit, max_range=default_max_range, gamma=0.55, alpha=1.0):
    """
    Gamma曲线压缩高动态范围，保留暗部细节
    """
    threshold = max_range * threshold_value
    # 使用numpy向量化操作，避免循环
    noise = np.random.randint(0, 51, size=1)[0]
    max_range_noisy = max_range + noise

    normalized = image_16bit.astype(np.float32) / 65280.0
    gamma_corrected = alpha * np.power(normalized, gamma)
    gamma_corrected = gamma_corrected * max_range_noisy + np.random.randint(0, 51)

    # 向量化阈值处理
    gamma_corrected[gamma_corrected < threshold] = 0
    return gamma_corrected.astype(np.uint16)


def compress_range(
    image, min_val=default_max_range / 4, max_val=default_max_range * 3 / 4
):
    """
    范化图像值域到指定范围
    """
    original_range = default_max_range
    target_range = max_val - min_val

    # 向量化处理
    compressed = (
        (image * (target_range / original_range))
        + min_val
        + np.random.randint(0, int(default_max_range * 0.2))
    )
    compressed = np.clip(compressed, min_val, max_val)

    return compressed.astype(np.uint16)


def smear_by_gray_blocks_optimized(gray_image, fill_image, n_clusters=8):
    """
    优化的KMeans聚类填充，使用MiniBatchKMeans加速
    """
    # 将彩色图像转换为灰度图像

    h, w = gray_image.shape

    scale = 2
    small_gray = cv2.resize(
        gray_image, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST
    )
    small_fill = cv2.resize(
        fill_image, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST
    )

    # 将灰度图像展平并转换为浮点数
    pixels = small_gray.reshape(-1, 1).astype(np.float32)

    # 使用 MiniBatchKMeans 进行聚类
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=0, batch_size=256, n_init=1
    )
    labels = kmeans.fit_predict(pixels)
    labels = labels.reshape(h // scale, w // scale)

    # 将标签扩展回原始尺寸
    labels_full = cv2.resize(
        labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    # 初始化结果图像
    result = np.zeros_like(fill_image)

    # 遍历每个簇，计算填充值
    for label in range(n_clusters):
        mask = labels_full == label
        if np.any(mask):
            block_value = max(
                np.mean(fill_image[mask]) * 0.8 - np.random.randint(0, 21),
                default_max_range * threshold_value,
            )
            result[mask] = block_value

    return result


def smear_by_color_blocks_optimized(color_image, fill_image, n_clusters=8):
    """
    优化的KMeans聚类填充，使用MiniBatchKMeans加速
    """
    h, w, c = color_image.shape

    scale = 2
    small_color = cv2.resize(
        color_image, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST
    )
    small_fill = cv2.resize(
        fill_image, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST
    )

    pixels = small_color.reshape(-1, c).astype(np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=0, batch_size=100, n_init=1
    )
    labels = kmeans.fit_predict(pixels)
    labels = labels.reshape(h // scale, w // scale)

    labels_full = cv2.resize(
        labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    result = np.zeros_like(fill_image)
    for label in range(n_clusters):
        mask = labels_full == label
        if np.any(mask):
            block_value = max(
                np.mean(fill_image[mask]) * 0.8 - np.random.randint(0, 21),
                default_max_range * threshold_value,
            )
            result[mask] = block_value

    return result


def process_depth_image_optimized(color_image, depth_image):
    """
    优化的深度图像处理函数
    """
    # side_image = cv2.ximgproc.jointBilateralFilter(
    #     joint = color_image,
    #     src = depth_image,
    #     d = 0,
    #     sigmaColor=5,
    #     sigmaSpace=9,
    #     borderType=cv2.BORDER_CONSTANT
    # )
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(
        src=depth_image, thresh=127, maxval=255, type=cv2.THRESH_BINARY
    )

    # filtered_image = cv2.ximgproc.jointBilateralFilter(
    #     joint = gray_image,
    #     src = depth_image,
    #     d = 0,
    #     sigmaColor = 80,
    #     sigmaSpace = 12,
    #     borderType=cv2.BORDER_CONSTANT
    # )

    filtered_image = cv2.ximgproc.guidedFilter(
        guide=color_image,  # 引导图像
        src=depth_image,  # 目标图像
        radius=8,  # 滤波窗口半径
        eps=1e-3,  # 正则化参数
    )

    filtered_image_16bit = (filtered_image * 256).astype(np.uint16)
    binary_image_16bit = (binary_image * 256).astype(np.uint16)

    gamma_image = gamma_transform(filtered_image_16bit, gamma=24)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("depth Image")
    # plt.imshow(depth_image, cmap="gray")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.title("Gamma Image (After Smearing)")
    # plt.imshow(gamma_image, cmap="gray")
    # plt.axis("off")

    # plt.show()

    gamma_image = compress_range(gamma_image)
    gamma_image = np.where(binary_image_16bit < 10, 0, gamma_image)

    gamma_image = smear_by_color_blocks_optimized(color_image, gamma_image)

    # gamma_image = smear_by_gray_blocks_optimized(gray_image, gamma_image)

    gamma_image = cv2.GaussianBlur(gamma_image, (5, 5), sigmaX=1)
    gamma_image = np.where(binary_image_16bit < 10, 0, gamma_image)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Color Image")
    # plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以正确显示
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.title("Gray Image")
    # plt.imshow(gray_image, cmap="gray")
    # plt.axis("off")

    # plt.show()
    # gamma_image =

    return gamma_image


def extract_and_save_frames(bag_path, depth_topic_name, color_topic_name, output_dir):
    """
    从ROS包中提取深度和彩色图像帧，并保存处理后的结果（来自原 depthImage8to16BitConversion.py 中的 extract_and_save_frames 函数）
    Args:
        bag_path: ROS包路径
        depth_topic_name: 深度图像话题名称
        color_topic_name: 彩色图像话题名称
        output_dir: 输出目录
    Returns:
        提取的帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    bag = rosbag.Bag(bag_path, "r")
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
        desc="提取帧",
    ):
        try:
            # 处理深度图像
            np_arr = np.frombuffer(depth_msg.data, np.uint8)
            depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            # 处理彩色图像
            np_arr = np.frombuffer(color_msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # 关联双边滤波处理
            result_image = process_depth_image_optimized(color_image, depth_image)

            result_filename = os.path.join(output_dir, f"result_{index:05d}.png")
            cv2.imwrite(result_filename, result_image)
            print(f"保存滤波图帧 #{index}")
            index += 1
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")

    bag.close()
    print(f"成功保存 {index} 个帧对")
    return index


def encode_video_with_ffmpeg(depth_temp_dir, output_video_path, framerate=25):
    """
    使用FFmpeg将提取的帧编码为视频
    """
    cmd = f"ffmpeg -y -framerate {framerate} -i {depth_temp_dir}/result_%05d.png -c:v ffv1 -pix_fmt gray16le {output_video_path}"
    print(f"运行命令: {cmd}")
    os.system(cmd)


def main():
    start_time = time.time()

    # total = extract_and_save_frames_parallel(bag_path, depth_topic_name, color_topic_name, depth_temp_dir)
    total = extract_and_save_frames(
        bag_path, depth_topic_name, color_topic_name, depth_temp_dir
    )

    elapsed_time = time.time() - start_time
    print(f"共提取了深度图{total}帧，耗时: {elapsed_time:.2f}秒")
    print(f"平均FPS: {total/elapsed_time:.2f}")

    if total > 0:
        encode_video_with_ffmpeg(depth_temp_dir, output_video_path, framerate=framerate)
    else:
        print("No frames extracted. Check topic and bag file.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
