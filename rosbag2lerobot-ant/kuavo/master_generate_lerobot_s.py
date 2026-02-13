"""
Script to convert Kuavo rosbag data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
现在id是唯一指示版本变量，修改了入参的结构，添加了描述信息至每个bag的每个step中，添加了使用ks_standard下载bag，通过限制线程个数减少内存占用，为最新版本。对应json入参为 request_new2.json
"""

from merge_batches import (
    merge_parquet_files,
    merge_meta_files,
    get_batch_dirs,
    merge_metadata,
)
from collections import OrderedDict
import custom_patches
import uuid
import psutil
import gc
import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import sys
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import get_feature_stats
import numpy as np
import torch
import tqdm
import json
from config_dataset_slave import Config, load_config_from_json
import argparse
import requests
import time
import uuid
import joblib
import gc
from copy import deepcopy
from slave_utils import (
    move_and_rename_depth_videos,
    save_camera_extrinsic_params,
    save_camera_info_to_json_new,
    save_depth_videos_16U_parallel,
    save_depth_videos_enhanced_parallel,
    swap_left_right_data_if_needed,
    flip_camera_arrays_if_needed,
)
from kuavo_dataset_slave_s import (
    KuavoRosbagReader,
    DEFAULT_JOINT_NAMES_LIST,
    DEFAULT_LEG_JOINT_NAMES,
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_HEAD_JOINT_NAMES,
    # DEFAULT_CAMERA_NAMES,
    DEFAULT_JOINT_NAMES,
    DEFAULT_LEJUCLAW_JOINT_NAMES,
    DEFAULT_DEXHAND_JOINT_NAMES,
    PostProcessorUtils,
)
import zipfile
import datetime
import einops
from math import ceil
from copy import deepcopy
import rosbag
import cv2
import os
import shutil
import concurrent.futures
import tempfile
from pathlib import Path

import requests
import time
import uuid
from copy import deepcopy
import logging

import os
import time
import subprocess
import multiprocessing
import queue
import threading

LEROBOT_HOME = HF_LEROBOT_HOME


def save_image_bytes_to_temp(
    imgs_per_cam: dict, imgs_per_cam_depth: dict, temp_base_dir: str, batch_id: int
):
    """
    直接保存图像字节流到临时目录（不解码、不缩放，传入什么尺寸就保存什么尺寸）

    Args:
        imgs_per_cam: 彩色图像字节流 {camera: [bytes, ...]}
        imgs_per_cam_depth: 深度图像字节流 {camera: [bytes, ...]}
        temp_base_dir: 临时目录基路径
        batch_id: 批次ID
    """
    import os

    cam_stats = {}
    # 保存彩色图像
    for camera, jpeg_list in imgs_per_cam.items():
        camera_dir = os.path.join(temp_base_dir, "color", camera)
        os.makedirs(camera_dir, exist_ok=True)

        for i, jpeg_bytes in enumerate(jpeg_list):
            frame_path = os.path.join(
                camera_dir, f"batch_{batch_id:04d}_frame_{i:06d}.jpg"
            )
            with open(frame_path, "wb") as f:
                f.write(jpeg_bytes)
            if i == 0:
                img_np_bgr = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if img_np_bgr is None:
                    raise ValueError(
                        f"Failed to decode color image for camera {camera} at frame {i}"
                    )
                h0, w0 = img_np_bgr.shape[:2]
                img_np = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                img_np = einops.rearrange(img_np, "h w c -> c h w")
                key = f"observation.images.{camera}"
                cam_stats[key] = get_feature_stats(
                    [img_np], axis=(0, 2, 3), keepdims=True
                )
                cam_stats[key] = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                    for k, v in cam_stats[key].items()
                }
                # 额外记录实际图像高宽，供 info.json 使用
                cam_stats[key]["height"] = int(h0)
                cam_stats[key]["width"] = int(w0)

    # 保存深度图像（PNG字节流）
    if imgs_per_cam_depth is not None:
        for camera, png_list in imgs_per_cam_depth.items():
            camera_dir = os.path.join(temp_base_dir, "depth", camera)
            os.makedirs(camera_dir, exist_ok=True)

            for i, png_bytes in enumerate(png_list):
                frame_path = os.path.join(
                    camera_dir, f"batch_{batch_id:04d}_frame_{i:06d}.png"
                )
                png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
                if isinstance(png_bytes, bytes):
                    idx_png = png_bytes.find(png_magic)
                    if idx_png != -1:
                        png_data = png_bytes[idx_png:]
                        with open(frame_path, "wb") as f:
                            f.write(png_data)

    print(f"[TEMP] 批次{batch_id} 图像字节流已保存到临时目录")
    return cam_stats


def _encode_color_camera_worker(
    camera_dir: str, camera: str, out_path: str, train_hz: int, stats_output_dir: str
):
    """
    子进程：编码彩色视频（PyAV）
    """
    import av
    from PIL import Image
    import glob
    import shutil
    import gc

    try:
        frame_files = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
        print(f"[VIDEO][COLOR] {camera}: 帧数 {len(frame_files)}")
        if len(frame_files) == 0:
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            print(f"[VIDEO][COLOR] {camera}: 无帧，已清理")
            return

        video_options = {
            "g": "2",
            "crf": "30",
            "svtav1-params": "threads=6:lp=4",
        }
        first_img = Image.open(frame_files[0])
        width, height = first_img.size

        with av.open(str(out_path), "w") as output:
            stream = output.add_stream("libx264", train_hz, options=video_options)
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height

            for frame_file in frame_files:
                img = Image.open(frame_file).convert("RGB")
                frame = av.VideoFrame.from_image(img)
                packet = stream.encode(frame)
                if packet:
                    output.mux(packet)
            packet = stream.encode()
            if packet:
                output.mux(packet)
        print(f"[VIDEO][COLOR] ✅ {camera} 完成: {out_path}")
    except Exception as e:
        print(f"[VIDEO][COLOR] ❌ {camera} 失败: {e}")
    finally:
        shutil.rmtree(camera_dir, ignore_errors=True)
        gc.collect()
        print(f"[VIDEO][COLOR] 🗑️  {camera} 临时文件已清理")


def _encode_depth_camera_worker(
    camera_dir: str, camera: str, out_path: str, train_hz: int, apply_denoise: bool
):
    """
    子进程：编码深度视频（ffmpeg + 可选去噪）
    """
    import glob
    import shutil
    import tempfile
    import gc
    import numpy as np
    import cv2
    import subprocess

    try:
        frame_files = sorted(glob.glob(os.path.join(camera_dir, "*.png")))
        print(f"[VIDEO][DEPTH] {camera}: 帧数 {len(frame_files)}")
        if len(frame_files) == 0:
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            print(f"[VIDEO][DEPTH] {camera}: 无帧，已清理")
            return

        is_hand_camera = "wrist_cam" in camera
        if is_hand_camera and apply_denoise:
            print(f"[VIDEO][DEPTH] {camera}: 应用去噪")

        with tempfile.TemporaryDirectory() as processed_dir:
            for idx, frame_file in enumerate(frame_files):
                img = cv2.imread(frame_file, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim > 2:
                    img = img[:, :, 0]
                if img.dtype != np.uint16:
                    img = img.astype(np.uint16)

                if is_hand_camera and apply_denoise:
                    try:
                        from video_denoising import repair_depth_noise_focused

                        img = repair_depth_noise_focused(
                            img,
                            max_valid_depth=10000,
                            median_kernel=5,
                            detect_white_spots=True,
                            spot_size_range=(10, 1000),
                        )
                    except Exception:
                        pass

                processed_path = os.path.join(processed_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(processed_path, img)
                if idx % 50 == 0:
                    gc.collect()

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(train_hz),
                "-i",
                os.path.join(processed_dir, "frame_%06d.png"),
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "gray16le",
                out_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[VIDEO][DEPTH] ✅ {camera} 完成: {out_path}")
    except Exception as e:
        print(f"[VIDEO][DEPTH] ❌ {camera} 失败: {e}")
    finally:
        shutil.rmtree(camera_dir, ignore_errors=True)
        gc.collect()
        print(f"[VIDEO][DEPTH] 🗑️  {camera} 临时文件已清理")


# ==================== 流式视频编码器 ====================


class StreamingColorVideoEncoder:
    """
    单相机流式视频编码器：通过有界队列接收帧，Worker 线程实时解码并编码到 PyAV 容器。

    特点：
    - PyAV 容器从初始化时就保持打开
    - 有界队列实现背压（队列满时阻塞入队）
    - 无需临时文件，直接内存编码

    用法:
        encoder = StreamingColorVideoEncoder("camera_top", output_path, train_hz=30)
        for batch_frames in batches:
            for frame_bytes in batch_frames:
                encoder.put(frame_bytes)
        stats = encoder.finish()
    """

    SENTINEL = object()  # 结束信号

    def __init__(
        self, camera: str, output_path: str, train_hz: int = 30, queue_limit: int = 100
    ):
        """
        Args:
            camera: 相机名称
            output_path: 输出视频路径
            train_hz: 帧率
            queue_limit: 队列上限（背压控制）
        """
        self.camera = camera
        self.output_path = output_path
        self.train_hz = train_hz
        self.queue_limit = queue_limit

        # 统计
        self._frame_count = 0  # 入队帧数
        self._encoded_count = 0  # 已编码帧数
        self._block_count = 0  # 入队阻塞次数
        self._start_time = time.time()

        # 状态
        self._finished = False
        self._error = None
        self._width = None
        self._height = None

        # 有界队列
        self._queue = queue.Queue(maxsize=queue_limit)

        # PyAV 容器（延迟初始化，在第一帧时确定尺寸）
        self._container = None
        self._stream = None
        self._container_lock = threading.Lock()

        # 启动工作线程
        self._worker = threading.Thread(
            target=self._encode_worker, name=f"StreamEnc-{camera}", daemon=True
        )
        self._worker.start()

        print(f"[STREAMING][{camera}] 编码器已启动 (队列上限={queue_limit})")

    def put(self, frame_bytes: bytes):
        """
        将帧放入队列（阻塞式，实现背压）

        Args:
            frame_bytes: JPEG 图像字节流
        """
        if self._finished:
            raise RuntimeError(
                f"StreamingColorVideoEncoder({self.camera}) already finished"
            )

        if self._error:
            raise self._error

        # 记录阻塞
        if self._queue.full():
            self._block_count += 1

        self._queue.put(frame_bytes)  # 阻塞等待空位
        self._frame_count += 1

    def finish(self) -> dict:
        """
        发送结束信号，等待编码完成，关闭容器。

        Returns:
            统计信息字典
        """
        if self._finished:
            return self._get_stats()

        # 发送结束信号
        self._queue.put(self.SENTINEL)
        self._worker.join()
        self._finished = True

        # 关闭容器
        with self._container_lock:
            if self._container is not None:
                try:
                    # 刷新编码器
                    if self._stream is not None:
                        packet = self._stream.encode(None)
                        if packet:
                            self._container.mux(packet)
                    self._container.close()
                except Exception as e:
                    print(f"[STREAMING][{self.camera}] 关闭容器时出错: {e}")

        if self._error:
            raise self._error

        elapsed = time.time() - self._start_time
        print(
            f"[STREAMING][{self.camera}] 完成: {self._encoded_count} 帧, 阻塞 {self._block_count} 次, 耗时 {elapsed:.1f}s"
        )

        return self._get_stats()

    def _get_stats(self) -> dict:
        return {
            "camera": self.camera,
            "frame_count": self._frame_count,
            "encoded_count": self._encoded_count,
            "block_count": self._block_count,
            "elapsed": time.time() - self._start_time,
        }

    def _encode_worker(self):
        """工作线程：从队列取帧，解码并编码"""
        import av
        from PIL import Image
        import io

        try:
            while True:
                item = self._queue.get()
                if item is self.SENTINEL:
                    break

                frame_bytes = item

                # 解码 JPEG
                try:
                    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                except Exception as e:
                    print(
                        f"[STREAMING][{self.camera}] 帧 {self._encoded_count} 解码失败: {e}"
                    )
                    continue

                # 延迟初始化容器（第一帧时确定尺寸）
                if self._container is None:
                    self._init_container(img.width, img.height)

                # 编码
                with self._container_lock:
                    if self._container is not None and self._stream is not None:
                        frame = av.VideoFrame.from_image(img)
                        frame.pts = self._encoded_count
                        packet = self._stream.encode(frame)
                        if packet:
                            self._container.mux(packet)
                        self._encoded_count += 1

        except Exception as e:
            self._error = e
            print(f"[STREAMING][{self.camera}] 编码错误: {e}")

    def _init_container(self, width: int, height: int):
        """初始化 PyAV 容器"""
        import av

        with self._container_lock:
            if self._container is not None:
                return

            self._width = width
            self._height = height

            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

            # 创建容器
            self._container = av.open(self.output_path, mode="w")

            # 创建视频流
            video_options = {
                "g": "2",
                "crf": "30",
            }
            self._stream = self._container.add_stream(
                "libx264", self.train_hz, options=video_options
            )
            self._stream.pix_fmt = "yuv420p"
            self._stream.width = width
            self._stream.height = height

            print(
                f"[STREAMING][{self.camera}] 容器已初始化: {width}x{height} @ {self.train_hz}fps"
            )


class StreamingVideoEncoderManager:
    """
    多相机流式编码管理器：管理多个 StreamingColorVideoEncoder 实例。

    职责：
    - 为每个相机创建编码器
    - 提供 feed_batch() 方法批量喂入帧
    - 提供 finalize() 方法等待所有编码完成
    - 错误传播：任一编码器失败则终止

    用法:
        manager = StreamingVideoEncoderManager(cameras, output_dir, uuid, train_hz)
        for batch_id, imgs_per_cam in batches:
            manager.feed_batch(imgs_per_cam, batch_id)
        manager.finalize()
    """

    def __init__(
        self,
        cameras: list,
        video_output_dir: str,
        uuid_str: str,
        train_hz: int = 30,
        queue_limit: int = 100,
    ):
        """
        Args:
            cameras: 相机名称列表
            video_output_dir: 视频输出目录
            uuid_str: 数据集 UUID
            train_hz: 帧率
            queue_limit: 每个相机的队列上限
        """
        self.cameras = cameras
        self.video_output_dir = video_output_dir
        self.uuid_str = uuid_str
        self.train_hz = train_hz
        self.queue_limit = queue_limit

        self._encoders = {}
        self._start_time = time.time()
        self._total_frames = 0
        self._batches_fed = 0
        self._cam_stats = {}  # 存储第一批次的 cam_stats（用于 meta 文件）

        # 为每个相机创建编码器
        for camera in cameras:
            output_path = os.path.join(
                video_output_dir,
                "videos",
                "chunk-000",
                f"observation.images.{camera}",
                "episode_000000.mp4",
            )
            self._encoders[camera] = StreamingColorVideoEncoder(
                camera=camera,
                output_path=output_path,
                train_hz=train_hz,
                queue_limit=queue_limit,
            )

        print(
            f"[STREAMING] 初始化流式编码管理器: {len(cameras)} 相机, 队列上限={queue_limit}"
        )

    def feed_batch(self, imgs_per_cam: dict, batch_id: int) -> dict:
        """
        将一个批次的帧喂入所有相机编码器。

        Args:
            imgs_per_cam: 每个相机的帧列表 {camera: [frame_bytes, ...]}
            batch_id: 批次ID

        Returns:
            cam_stats: 图像统计信息，格式与 save_image_bytes_to_temp() 兼容
        """
        import cv2
        import numpy as np
        import einops

        cam_stats = {}
        batch_total = 0
        batch_blocks = 0
        block_details = {}

        for camera, frame_list in imgs_per_cam.items():
            if camera not in self._encoders:
                continue

            encoder = self._encoders[camera]
            before_blocks = encoder._block_count

            # 计算第一帧的图像统计（用于 meta 文件），并记录实际高宽
            if len(frame_list) > 0 and batch_id == 1:
                first_frame_bytes = frame_list[0]
                img_np_bgr = cv2.imdecode(
                    np.frombuffer(first_frame_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if img_np_bgr is not None:
                    h0, w0 = img_np_bgr.shape[:2]
                    img_np = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                    img_np = einops.rearrange(img_np, "h w c -> c h w")
                    key = f"observation.images.{camera}"
                    cam_stats[key] = get_feature_stats(
                        [img_np], axis=(0, 2, 3), keepdims=True
                    )
                    cam_stats[key] = {
                        k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                        for k, v in cam_stats[key].items()
                    }
                    cam_stats[key]["height"] = int(h0)
                    cam_stats[key]["width"] = int(w0)

            # 喂入所有帧到编码器
            for frame_bytes in frame_list:
                encoder.put(frame_bytes)

            after_blocks = encoder._block_count
            blocks_this_batch = after_blocks - before_blocks
            batch_blocks += blocks_this_batch
            batch_total += len(frame_list)
            block_details[camera] = blocks_this_batch

        self._total_frames += batch_total
        self._batches_fed += 1

        # 保存第一批次的 cam_stats（用于 meta 文件生成）
        if batch_id == 1 and cam_stats:
            self._cam_stats = cam_stats

        # 日志
        if batch_blocks > 0:
            block_info = ", ".join(
                [f"{c}: 阻塞{b}次" for c, b in block_details.items() if b > 0]
            )
            print(
                f"[STREAMING] Batch {batch_id}: 已喂入 {batch_total // len(imgs_per_cam)} 帧/相机 ({block_info})"
            )
        else:
            print(
                f"[STREAMING] Batch {batch_id}: 已喂入 {batch_total // len(imgs_per_cam)} 帧/相机"
            )

        return self._cam_stats

    def finalize(self) -> dict:
        """
        等待所有编码器完成，收集统计信息。

        Returns:
            汇总统计信息
        """
        print(f"[STREAMING] 等待编码完成...")

        all_stats = {}
        total_blocks = 0
        total_encoded = 0

        for camera, encoder in self._encoders.items():
            try:
                stats = encoder.finish()
                all_stats[camera] = stats
                total_blocks += stats["block_count"]
                total_encoded += stats["encoded_count"]
            except Exception as e:
                print(f"[STREAMING][{camera}] 完成时出错: {e}")
                raise

        elapsed = time.time() - self._start_time
        print(
            f"[STREAMING] 全部完成: {total_encoded} 帧, 总阻塞 {total_blocks} 次, 总耗时 {elapsed:.1f}s"
        )

        return {
            "cameras": all_stats,
            "total_frames": self._total_frames,
            "total_encoded": total_encoded,
            "total_blocks": total_blocks,
            "batches_fed": self._batches_fed,
            "elapsed": elapsed,
        }


def _encode_batch_segment_color(
    batch_id: int,
    camera: str,
    temp_dir: str,
    segment_dir: str,
    train_hz: int,
    chunk_size: int = 800,
) -> str:
    """
    编码单个 batch 的彩色视频片段

    Args:
        batch_id: 批次ID
        camera: 相机名称
        temp_dir: 临时帧目录 (包含 batch_XXXX_frame_XXXXXX.jpg)
        segment_dir: 片段输出目录
        train_hz: 视频帧率
        chunk_size: 每批次帧数，用于计算全局帧起始位置

    Returns:
        片段文件路径，失败返回 None
    """
    import av
    from PIL import Image
    import glob
    import time as time_module

    start_time = time_module.time()

    # 查找该 batch 的帧
    pattern = os.path.join(temp_dir, f"batch_{batch_id:04d}_frame_*.jpg")
    frame_files = sorted(glob.glob(pattern))

    if len(frame_files) == 0:
        print(f"[PIPELINE][{camera}] Batch {batch_id}: 无帧，跳过")
        return None

    # 输出片段路径
    os.makedirs(segment_dir, exist_ok=True)
    segment_path = os.path.join(segment_dir, f"segment_{batch_id:04d}.mp4")

    # 计算该 batch 的全局帧起始位置 (batch_id 从 1 开始)
    global_frame_start = (batch_id - 1) * chunk_size

    try:
        video_options = {
            "g": "2",
            "crf": "30",
        }
        first_img = Image.open(frame_files[0])
        width, height = first_img.size

        with av.open(str(segment_path), "w") as output:
            stream = output.add_stream("libx264", train_hz, options=video_options)
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height

            for local_idx, frame_file in enumerate(frame_files):
                img = Image.open(frame_file).convert("RGB")
                frame = av.VideoFrame.from_image(img)
                # 设置全局 PTS，确保拼接后时间戳连续
                frame.pts = global_frame_start + local_idx
                packet = stream.encode(frame)
                if packet:
                    output.mux(packet)
            packet = stream.encode()
            if packet:
                output.mux(packet)

        elapsed_ms = (time_module.time() - start_time) * 1000
        print(
            f"[PIPELINE][{camera}] Batch {batch_id}: {len(frame_files)} 帧 (起始帧={global_frame_start}) → {segment_path} ({elapsed_ms:.0f}ms)"
        )
        return segment_path

    except Exception as e:
        print(f"[PIPELINE][{camera}] Batch {batch_id} 编码失败: {e}")
        return None


def _concat_segments_ffmpeg(
    segment_paths: list, output_path: str, train_hz: int = 30
) -> bool:
    """
    使用 ffmpeg 拼接视频片段（重新编码以确保时间戳正确）

    Args:
        segment_paths: 片段文件路径列表（已排序）
        output_path: 最终输出路径
        train_hz: 视频帧率

    Returns:
        成功返回 True
    """
    import tempfile
    import time as time_module

    if not segment_paths:
        print(f"[PIPELINE] 拼接失败: 无片段")
        return False

    start_time = time_module.time()

    # 创建 filelist.txt
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg in segment_paths:
            f.write(f"file '{seg}'\n")
        filelist_path = f.name

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 使用重新编码确保时间戳严格连续
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            filelist_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "30",
            "-g",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(train_hz),
            "-vsync",
            "cfr",  # 强制恒定帧率，确保时间戳精确
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[PIPELINE] ffmpeg 拼接失败: {result.stderr}")
            return False

        elapsed_ms = (time_module.time() - start_time) * 1000
        print(
            f"[PIPELINE] 拼接完成: {len(segment_paths)} 片段 → {output_path} ({elapsed_ms:.0f}ms)"
        )
        return True

    finally:
        os.unlink(filelist_path)


class BatchSegmentEncoder:
    """
    批次分段视频编码器 - 实现批处理与视频编码的流水线并行

    工作流程:
    1. 主线程调用 submit_batch(batch_id) 提交编码任务
    2. 工作线程池异步编码各批次的视频片段
    3. 主线程调用 finalize() 等待编码完成并拼接最终视频
    """

    def __init__(
        self,
        temp_base_dir: str,
        segment_base_dir: str,
        video_output_dir: str,
        cameras: list,
        train_hz: int,
        uuid_str: str,
        chunk_size: int = 800,
        max_workers: int = 3,
    ):
        """
        Args:
            temp_base_dir: 临时帧目录 (包含 color/{camera}/)
            segment_base_dir: 片段临时目录
            video_output_dir: 最终视频输出目录
            cameras: 相机列表
            train_hz: 视频帧率
            uuid_str: 数据集 UUID
            chunk_size: 每批次帧数，用于计算全局帧 PTS
            max_workers: 最大并行编码数
        """
        import queue
        import threading

        self.temp_base_dir = temp_base_dir
        self.segment_base_dir = segment_base_dir
        self.video_output_dir = video_output_dir
        self.cameras = cameras
        self.train_hz = train_hz
        self.uuid_str = uuid_str
        self.chunk_size = chunk_size
        self.max_workers = max_workers

        # 任务队列和结果存储
        self.task_queue = queue.Queue()
        self.segments = {cam: [] for cam in cameras}  # {camera: [segment_path, ...]}
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()

        # 统计
        self.batches_submitted = 0
        self.batches_encoded = 0
        self.start_time = None

        # 错误状态
        self.error_flag = threading.Event()
        self.error_message = None

        # 启动工作线程
        self.workers = []
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"SegmentEncoder-{i}", daemon=True
            )
            t.start()
            self.workers.append(t)

        self.start_time = time.time()
        print(
            f"[PIPELINE] 初始化分段编码器: {len(cameras)} 相机, {max_workers} 工作线程"
        )

    def _worker_loop(self):
        """工作线程主循环"""
        while not self.stop_flag.is_set() and not self.error_flag.is_set():
            try:
                task = self.task_queue.get(timeout=0.5)
                if task is None:  # 结束信号
                    break

                # 如果已经有错误，跳过处理
                if self.error_flag.is_set():
                    self.task_queue.task_done()
                    continue

                batch_id = task

                # 为每个相机编码该批次
                for camera in self.cameras:
                    if self.error_flag.is_set():
                        break

                    temp_dir = os.path.join(self.temp_base_dir, "color", camera)
                    segment_dir = os.path.join(self.segment_base_dir, camera)

                    segment_path = _encode_batch_segment_color(
                        batch_id,
                        camera,
                        temp_dir,
                        segment_dir,
                        self.train_hz,
                        self.chunk_size,
                    )

                    if segment_path:
                        with self.lock:
                            self.segments[camera].append((batch_id, segment_path))
                    else:
                        # 编码失败，设置错误标志并终止
                        with self.lock:
                            self.error_flag.set()
                            self.error_message = (
                                f"Batch {batch_id} camera {camera} 编码失败"
                            )
                        print(f"[PIPELINE][ERROR] {self.error_message}，终止流水线")
                        break

                with self.lock:
                    self.batches_encoded += 1

                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                with self.lock:
                    self.error_flag.set()
                    self.error_message = f"工作线程异常: {e}"
                print(f"[PIPELINE][ERROR] {self.error_message}")

    def submit_batch(self, batch_id: int):
        """提交一个批次的编码任务"""
        self.task_queue.put(batch_id)
        self.batches_submitted += 1
        print(f"[PIPELINE] Batch {batch_id} 已提交编码队列")

    def finalize(self, use_depth: bool = False) -> bool:
        """
        等待所有编码完成，拼接最终视频

        Returns:
            成功返回 True

        Raises:
            RuntimeError: 如果编码过程中发生错误
        """
        import time as time_module

        print(f"[PIPELINE] 等待 {self.batches_submitted} 个批次编码完成...")

        # 等待队列清空
        self.task_queue.join()

        # 停止工作线程
        self.stop_flag.set()
        for _ in self.workers:
            self.task_queue.put(None)
        for t in self.workers:
            t.join(timeout=5)

        encode_time = time.time() - self.start_time

        # 检查是否有错误发生
        if self.error_flag.is_set():
            error_msg = self.error_message or "未知错误"
            print(f"[PIPELINE][FATAL] 编码失败: {error_msg}")
            # 清理临时目录
            if os.path.exists(self.segment_base_dir):
                shutil.rmtree(self.segment_base_dir)
            if os.path.exists(self.temp_base_dir):
                shutil.rmtree(self.temp_base_dir)
            raise RuntimeError(f"视频编码流水线失败: {error_msg}")

        print(
            f"[PIPELINE] 所有批次编码完成 ({self.batches_encoded}/{self.batches_submitted}), 耗时 {encode_time:.1f}s"
        )

        # 拼接各相机的视频
        concat_start = time.time()
        color_out_dir = os.path.join(self.video_output_dir, "videos", "chunk-000")

        success = True
        for camera in self.cameras:
            # 按 batch_id 排序
            with self.lock:
                sorted_segments = sorted(self.segments[camera], key=lambda x: x[0])
                segment_paths = [path for _, path in sorted_segments]

            if not segment_paths:
                print(f"[PIPELINE][{camera}] 无片段可拼接")
                continue

            output_path = os.path.join(
                color_out_dir, f"observation.images.{camera}", "episode_000000.mp4"
            )
            if not _concat_segments_ffmpeg(segment_paths, output_path, self.train_hz):
                success = False

        concat_time = time.time() - concat_start
        total_time = time.time() - self.start_time

        # 清理片段临时目录
        if os.path.exists(self.segment_base_dir):
            shutil.rmtree(self.segment_base_dir)

        # 清理原始帧临时目录
        if os.path.exists(self.temp_base_dir):
            shutil.rmtree(self.temp_base_dir)

        print(f"[PIPELINE] ========== 流水线完成 ==========")
        print(
            f"[PIPELINE] 编码耗时: {encode_time:.1f}s, 拼接耗时: {concat_time:.1f}s, 总计: {total_time:.1f}s"
        )

        return success


# ==================== 原有视频编码函数 ====================


def encode_complete_videos_from_temp(
    temp_base_dir: str,
    video_output_dir: str,
    uuid: str,
    raw_config: Config,
    use_depth: bool = True,
):
    """
    从临时帧目录合成完整视频（所有batch合并为一个视频）
    逐个相机处理，处理完立即清理，控制内存占用

    Args:
        temp_base_dir: 临时帧目录
        video_output_dir: 视频输出目录
        uuid: 数据集UUID
        raw_config: 配置对象
    """
    import shutil
    import av
    from PIL import Image
    import glob

    print("[VIDEO] ========== 开始合成完整视频 ==========")

    # 创建输出目录
    stats_output_dir = os.path.join(video_output_dir, "meta", "episodes_stats.jsonl")
    color_out_dir = os.path.join(video_output_dir, "videos", "chunk-000")

    os.makedirs(color_out_dir, exist_ok=True)

    # === 彩色：每相机一个子进程 ===
    color_temp_dir = os.path.join(temp_base_dir, "color")
    color_procs = []
    if os.path.exists(color_temp_dir):
        for camera in os.listdir(color_temp_dir):
            camera_dir = os.path.join(color_temp_dir, camera)
            if not os.path.isdir(camera_dir):
                continue
            video_path = os.path.join(
                color_out_dir, f"observation.images.{camera}", "episode_000000.mp4"
            )
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            p = multiprocessing.Process(
                target=_encode_color_camera_worker,
                args=(
                    camera_dir,
                    camera,
                    video_path,
                    raw_config.train_hz,
                    stats_output_dir,
                ),
                daemon=False,
            )
            p.start()
            color_procs.append(p)

    # === 深度：每相机一个子进程（受 use_depth 控制） ===
    depth_temp_dir = os.path.join(temp_base_dir, "depth")
    depth_procs = []
    if use_depth and os.path.exists(depth_temp_dir):
        depth_out_dir = os.path.join(video_output_dir, "depth", "chunk-000")
        os.makedirs(depth_out_dir, exist_ok=True)
        apply_denoise = getattr(raw_config, "denoise_enabled", True)
        apply_denoise = False  # 保持原逻辑关闭
        for camera in os.listdir(depth_temp_dir):
            camera_dir = os.path.join(depth_temp_dir, camera)
            if not os.path.isdir(camera_dir):
                continue
            video_path = os.path.join(depth_out_dir, f"{camera}.mkv")
            p = multiprocessing.Process(
                target=_encode_depth_camera_worker,
                args=(
                    camera_dir,
                    camera,
                    video_path,
                    raw_config.train_hz,
                    apply_denoise,
                ),
                daemon=False,
            )
            p.start()
            depth_procs.append(p)
    elif not use_depth and os.path.exists(depth_temp_dir):
        shutil.rmtree(depth_temp_dir, ignore_errors=True)
        print("[VIDEO] 跳过深度视频处理（use_depth=false），已清理深度临时目录")

    # 等待所有子进程完成
    for p in color_procs:
        p.join()
    for p in depth_procs:
        p.join()

    # 清理整个临时目录
    if os.path.exists(temp_base_dir):
        shutil.rmtree(temp_base_dir)
        print("[VIDEO] ========== 所有视频编码完成，临时目录已清理 ==========")
        print(f"[VIDEO] 视频保存位置: {video_output_dir}/{uuid}")


def encode_complete_videos_from_temp1(
    temp_base_dir: str,
    video_output_dir: str,
    uuid: str,
    raw_config: Config,
):
    """
    从临时帧目录合成完整视频（所有batch合并为一个视频）
    逐个相机处理，处理完立即清理，控制内存占用

    Args:
        temp_base_dir: 临时帧目录
        video_output_dir: 视频输出目录
        uuid: 数据集UUID
        raw_config: 配置对象
    """
    import shutil
    import av
    from PIL import Image
    import glob
    import concurrent.futures
    import multiprocessing

    print("[VIDEO] ========== 开始合成完整视频 ==========")

    # 创建输出目录
    color_out_dir = os.path.join(video_output_dir, uuid, "color")
    depth_out_dir = os.path.join(video_output_dir, uuid, "depth")
    os.makedirs(color_out_dir, exist_ok=True)
    os.makedirs(depth_out_dir, exist_ok=True)

    # 并发线程数
    max_workers = getattr(raw_config, "video_encoding_workers", None)
    if not isinstance(max_workers, int) or max_workers <= 0:
        max_workers = max(1, multiprocessing.cpu_count())

    # 彩色相机编码的工作函数
    def _encode_color_camera(camera_dir: str, camera: str):
        print(f"[VIDEO] 处理彩色相机: {camera}")
        frame_files = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
        print(f"[VIDEO]   发现 {len(frame_files)} 帧")
        if len(frame_files) == 0:
            # 没帧也清理目录
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            return f"{camera}: skipped(empty)"

        video_path = os.path.join(color_out_dir, f"{camera}.mp4")
        try:
            video_options = {
                "g": "2",
                "crf": "30",
                "svtav1-params": "threads=6:lp=4",
            }
            first_img = Image.open(frame_files[0])
            width, height = first_img.size
            with av.open(str(video_path), "w") as output:
                stream = output.add_stream(
                    "libx264", raw_config.train_hz, options=video_options
                )
                stream.pix_fmt = "yuv420p"
                stream.width = width
                stream.height = height

                for frame_file in frame_files:
                    img = Image.open(frame_file).convert("RGB")
                    frame = av.VideoFrame.from_image(img)
                    packet = stream.encode(frame)
                    if packet:
                        output.mux(packet)
                packet = stream.encode()
                if packet:
                    output.mux(packet)
            ret = f"{camera}: ok -> {video_path}"
        except Exception as e:
            ret = f"{camera}: fail -> {e}"
        finally:
            # 无论成功失败都清理临时帧目录
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            print(f"[VIDEO]   🗑️  {camera} 临时文件已清理")
        return ret

    # 深度相机编码的工作函数（ffmpeg）
    def _encode_depth_camera(camera_dir: str, camera: str, apply_denoise: bool):
        print(f"[VIDEO] 处理深度相机: {camera}")
        frame_files = sorted(glob.glob(os.path.join(camera_dir, "*.png")))
        print(f"[VIDEO]   发现 {len(frame_files)} 深度帧")
        if len(frame_files) == 0:
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            return f"{camera}: skipped(empty)"

        is_hand_camera = "wrist_cam" in camera
        if is_hand_camera and apply_denoise:
            print(f"[VIDEO]   将应用深度去噪")

        import tempfile

        try:
            with tempfile.TemporaryDirectory() as processed_dir:
                for idx, frame_file in enumerate(frame_files):
                    img = cv2.imread(frame_file, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim > 2:
                        img = img[:, :, 0]
                    if img.dtype != np.uint16:
                        img = img.astype(np.uint16)

                    if is_hand_camera and apply_denoise:
                        try:
                            from video_denoising import repair_depth_noise_focused

                            img = repair_depth_noise_focused(
                                img,
                                max_valid_depth=10000,
                                median_kernel=5,
                                detect_white_spots=True,
                                spot_size_range=(10, 1000),
                            )
                        except Exception:
                            pass

                    processed_path = os.path.join(processed_dir, f"frame_{idx:06d}.png")
                    cv2.imwrite(processed_path, img)
                    if idx % 50 == 0:
                        gc.collect()

                video_path = os.path.join(depth_out_dir, f"{camera}.mkv")
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(raw_config.train_hz),
                    "-i",
                    os.path.join(processed_dir, "frame_%06d.png"),
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "gray16le",
                    video_path,
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                ret = f"{camera}: ok -> {video_path}"
        except Exception as e:
            ret = f"{camera}: fail -> {e}"
        finally:
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            print(f"[VIDEO]   🗑️  {camera} 临时文件已清理")
        return ret

    # === 并发处理彩色视频 ===
    color_temp_dir = os.path.join(temp_base_dir, "color")
    if os.path.exists(color_temp_dir):
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for camera in os.listdir(color_temp_dir):
                camera_dir = os.path.join(color_temp_dir, camera)
                if not os.path.isdir(camera_dir):
                    continue
                futures.append(
                    executor.submit(_encode_color_camera, camera_dir, camera)
                )
            for f in concurrent.futures.as_completed(futures):
                print(f"[VIDEO]   结果: {f.result()}")

    # === 并发处理深度视频 ===
    depth_temp_dir = os.path.join(temp_base_dir, "depth")
    if os.path.exists(depth_temp_dir):
        apply_denoise = getattr(raw_config, "denoise_enabled", True)
        # 现有逻辑强制关闭
        apply_denoise = False
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for camera in os.listdir(depth_temp_dir):
                camera_dir = os.path.join(depth_temp_dir, camera)
                if not os.path.isdir(camera_dir):
                    continue
                futures.append(
                    executor.submit(
                        _encode_depth_camera, camera_dir, camera, apply_denoise
                    )
                )
            for f in concurrent.futures.as_completed(futures):
                print(f"[VIDEO]   结果: {f.result()}")

    # 清理整个临时目录
    if os.path.exists(temp_base_dir):
        shutil.rmtree(temp_base_dir)
        print("[VIDEO] ========== 所有视频编码完成，临时目录已清理 ==========")
        print(f"[VIDEO] 视频保存位置: {video_output_dir}/{uuid}")


def get_nested_value(data, path, i=None, default=None):
    """
    从嵌套字典中通过路径字符串提取数据，并支持按帧索引和默认值。
    path: 例如 "state.head.position"
    i: 帧索引，如果为 None 则返回整个数组
    default: 默认值（如 [0.0]*2）
    """
    keys = path.split(".")
    v = data
    try:
        for k in keys:
            v = v[k]
        if i is not None:
            if v is not None and len(v) > i:
                v = v[i]
            else:
                v = default
        if v is None:
            v = default
        if isinstance(v, torch.Tensor):
            return v.float()
        else:
            return torch.tensor(v, dtype=torch.float32)
    except Exception:
        return torch.tensor(default, dtype=torch.float32)


# 用法示例：
# get_nested_value(all_low_dim_data, "state.head.position", i, [0.0]*2)


def is_valid_hand_data(arr, expected_shape=None):
    arr = np.array(arr) if arr is not None else None
    if arr is None or arr.size == 0:
        return False
    if expected_shape is not None and arr.shape[1:] != expected_shape:
        return False
    return True


def calculate_action_frames(
    rosbag_actual_start_time,  # 实际数据开始时间
    rosbag_actual_end_time,  # 实际数据结束时间
    rosbag_original_start_time,  # 原始bag开始时间
    rosbag_original_end_time,  # 原始bag结束时间
    action_original_start_time,  # 动作原始开始时间
    action_duration,  # 动作持续时间
    frame_rate,  # 帧率
    total_frames,  # 总帧数
):
    """
    计算动作的开始帧和结束帧

    策略：
    1. 计算动作在原始时间轴上的绝对时间范围
    2. 将这个时间范围映射到实际数据的时间范围
    3. 根据实际数据的时间范围计算对应的帧数
    """

    # 1. 计算动作的绝对时间范围
    action_start_time = action_original_start_time
    action_end_time = action_original_start_time + action_duration

    # 2. 检查动作时间是否在实际数据范围内
    if (
        action_end_time < rosbag_actual_start_time
        or action_start_time > rosbag_actual_end_time
    ):
        # 动作完全在实际数据范围之外
        return None, None

    # 3. 将动作时间范围限制在实际数据范围内
    clipped_action_start = max(action_start_time, rosbag_actual_start_time)
    clipped_action_end = min(action_end_time, rosbag_actual_end_time)

    # 4. 计算相对于实际数据开始时间的偏移
    start_offset = clipped_action_start - rosbag_actual_start_time
    end_offset = clipped_action_end - rosbag_actual_start_time

    # 5. 根据实际数据的时间范围计算帧数
    actual_data_duration = rosbag_actual_end_time - rosbag_actual_start_time

    # 方法1：按时间比例计算
    start_frame = int((start_offset / actual_data_duration) * total_frames)
    end_frame = int((end_offset / actual_data_duration) * total_frames)

    # 方法2：按帧率计算（更精确）
    # start_frame = int(start_offset * frame_rate)
    # end_frame = int(end_offset * frame_rate)

    # 6. 确保帧数在有效范围内
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    return start_frame, end_frame


def merge_metadata_and_moment(
    metadata_path,
    moment_path,
    output_path,
    uuid,
    raw_config,
    bag_time_info=None,
    main_time_line_timestamps=None,
):
    """
    合并 metadata 和 moment 数据，并添加 bag 时间信息和计算帧数
    支持两种格式：
    1. 旧格式：metadata.json + moments.json 两个文件
    2. 新格式：只有一个 metadata.json，包含 marks 数组
    
    Args:
        metadata_path: metadata.json 文件路径
        moment_path: moment.json 文件路径（新格式下可为 None）
        output_path: 输出文件路径
        uuid: 唯一标识符
        raw_config: 原始配置对象
        bag_time_info: bag时间信息字典（可选）
        main_time_line_timestamps: 经过帧率对齐后的时间戳数组（纳秒）
    """
    frequency = raw_config.train_hz if hasattr(raw_config, "train_hz") else 30

    # 读取 metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)

    # 检测新格式：如果 metadata.json 中有 marks 字段，使用新格式
    is_new_format = "marks" in raw_metadata and isinstance(raw_metadata.get("marks"), list)
    
    if is_new_format:
        print("[FORMAT] 检测到新格式 metadata.json（包含 marks 数组）")
        marks = raw_metadata.get("marks", [])
        moment = None  # 新格式不需要 moment.json
    else:
        print("[FORMAT] 使用旧格式（metadata.json + moments.json）")
        # 读取 moment.json
        if moment_path and os.path.exists(moment_path):
            with open(moment_path, "r", encoding="utf-8") as f:
                moment = json.load(f)
        else:
            print(f"[WARN] moment.json 不存在: {moment_path}")
            moment = {"moments": []}

    # 转换新格式 metadata 为旧格式
    converted_metadata = {}
    
    if is_new_format:
        # 新格式字段映射
        converted_metadata["scene_name"] = raw_metadata.get("primaryScene", "")
        converted_metadata["sub_scene_name"] = raw_metadata.get("tertiaryScene", "")
        converted_metadata["init_scene_text"] = raw_metadata.get("initSceneText", "")
        converted_metadata["english_init_scene_text"] = raw_metadata.get("englishInitSceneText", "")
        
        # task_name 优先 taskGroupName，其次 taskName
        task_name = raw_metadata.get("taskGroupName")
        if not task_name:
            task_name = raw_metadata.get("taskName", "")
        converted_metadata["task_name"] = task_name
        
        # english_task_name 优先 taskGroupCode，其次 taskCode
        english_task_name = raw_metadata.get("taskGroupCode")
        if not english_task_name:
            english_task_name = raw_metadata.get("taskCode", "")
        converted_metadata["english_task_name"] = english_task_name
        if isinstance(english_task_name, str) and "_" in english_task_name:
            english_task_name = english_task_name.replace("_", " ")
        converted_metadata["english_task_name"] = english_task_name
        
        converted_metadata["sn_code"] = raw_metadata.get("deviceSn", "")
    else:
        # 旧格式字段映射
        converted_metadata["scene_name"] = raw_metadata.get("scene_code", "")
        converted_metadata["sub_scene_name"] = raw_metadata.get("sub_scene_code", "")
        converted_metadata["init_scene_text"] = raw_metadata.get("sub_scene_zh_dec", "")
        converted_metadata["english_init_scene_text"] = raw_metadata.get("sub_scene_en_dec", "")
        
        task_name = raw_metadata.get("task_group_name")
        if not task_name:
            task_name = raw_metadata.get("task_name", "")
        converted_metadata["task_name"] = task_name
        
        english_task_name = raw_metadata.get("task_group_code")
        if not english_task_name:
            english_task_name = raw_metadata.get("task_code", "")
        converted_metadata["english_task_name"] = english_task_name
        if isinstance(english_task_name, str) and "_" in english_task_name:
            english_task_name = english_task_name.replace("_", " ")
        converted_metadata["english_task_name"] = english_task_name
        
        converted_metadata["sn_code"] = raw_metadata.get("device_sn", "")

    # 默认值字段
    converted_metadata["data_type"] = "常规"
    converted_metadata["episode_status"] = "approved"
    converted_metadata["data_gen_mode"] = "real_machine"
    converted_metadata["sn_name"] = "乐聚机器人"

    print(f"Metadata 字段转换结果:")
    for key, value in converted_metadata.items():
        print(f"  {key}: '{value}'")

    # 使用转换后的 metadata
    metadata = converted_metadata

    # 获取时间信息
    rosbag_actual_start_time = None
    rosbag_actual_end_time = None
    rosbag_original_start_time = None
    rosbag_original_end_time = None
    total_frames = 0

    # 实际数据时间范围
    if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
        # 调试：打印原始时间戳
        print(f"原始时间戳前3个: {main_time_line_timestamps[:3]}")
        print(f"原始时间戳后3个: {main_time_line_timestamps[-3:]}")

        # 检查时间戳是否已经是秒格式还是纳秒格式
        if main_time_line_timestamps[0] > 1e12:  # 如果大于1e12，认为是纳秒格式
            timestamps_seconds = main_time_line_timestamps / 1e9
            print("时间戳格式：纳秒 -> 秒")
        else:
            timestamps_seconds = main_time_line_timestamps
            print("时间戳格式：已经是秒")

        rosbag_actual_start_time = timestamps_seconds[0]
        rosbag_actual_end_time = timestamps_seconds[-1]
        total_frames = len(main_time_line_timestamps)

        # 调试：打印转换后的时间戳
        print(f"转换后时间戳前3个: {timestamps_seconds[:3]}")
        print(f"转换后时间戳后3个: {timestamps_seconds[-3:]}")

        # 验证时间戳转换
        start_datetime = datetime.datetime.fromtimestamp(
            rosbag_actual_start_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )
        end_datetime = datetime.datetime.fromtimestamp(
            rosbag_actual_end_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )

        print(f"实际开始时间验证: {start_datetime.isoformat()}")
        print(f"实际结束时间验证: {end_datetime.isoformat()}")

    # 原始bag时间范围
    if bag_time_info:
        rosbag_original_start_time = bag_time_info.get("unix_timestamp")
        rosbag_original_end_time = bag_time_info.get("end_time")

    # 构造 action_config
    print(f"时间信息:")
    if rosbag_original_start_time and rosbag_original_end_time:
        print(
            f"  原始bag时间: {rosbag_original_start_time:.6f}s - {rosbag_original_end_time:.6f}s"
        )
    if rosbag_actual_start_time and rosbag_actual_end_time:
        print(
            f"  实际数据时间: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
        )
    print(f"  总帧数: {total_frames}")

    action_config = []

    # 根据格式选择数据源
    if is_new_format:
        # 新格式：从 marks 数组读取
        data_source = marks
    else:
        # 旧格式：从 moments 数组读取
        data_source = moment.get("moments", [])

    for m in data_source:
        if is_new_format:
            # 新格式：直接从 mark 对象读取
            mark_start = m.get("markStart", "")
            mark_end = m.get("markEnd", "")
            duration = m.get("duration", 0.0)  # 已经是数字，单位秒
            
            # 转换时间格式：从 "2026-01-06 09:41:20.781" 转为 ISO 格式
            try:
                # 解析 markStart 时间
                if mark_start:
                    # 尝试解析 "2026-01-06 09:41:20.781" 格式
                    if " " in mark_start:
                        dt_str, time_str = mark_start.split(" ", 1)
                        # 转换为 ISO 格式：2026-01-06T09:41:20.781+08:00
                        formatted_trigger_time = f"{dt_str}T{time_str}+08:00"
                    else:
                        formatted_trigger_time = mark_start
                else:
                    formatted_trigger_time = ""
            except Exception as e:
                print(f"[WARN] 解析 markStart 时间失败: {mark_start}, 错误: {e}")
                formatted_trigger_time = ""
            
            skill_atomic = m.get("skillAtomic", "")
            skill_detail = m.get("skillDetail", "")
            en_skill_detail = m.get("enSkillDetail", "")
            mark_type = m.get("markType", "step")
            
            # 判断是否为错误动作（retry 类型）
            is_mistake = (mark_type == "retry")
            
            print(f"处理动作数据（新格式）:")
            print(f"  skill_atomic: {skill_atomic}")
            print(f"  skill_detail: {skill_detail}")
            print(f"  en_skill_detail: {en_skill_detail}")
            print(f"  markStart: {mark_start}")
            print(f"  markEnd: {mark_end}")
            print(f"  duration: {duration}s")
            print(f"  markType: {mark_type} (is_mistake={is_mistake})")
        else:
            # 旧格式：从 customFieldValues 中提取数据
            custom_fields = m.get("customFieldValues", {})
            trigger_time = m.get("triggerTime", "")
            duration_str = m.get("duration", "0s")
            
            # 格式化时间戳：将 "Z" 替换为 "+00:00"
            formatted_trigger_time = (
                trigger_time.replace("Z", "+00:00") if trigger_time else ""
            )
            
            skill_atomic = custom_fields.get("skill_atomic_en", "")
            skill_detail = custom_fields.get("skill_detail", "")
            en_skill_detail = custom_fields.get("en_skill_detail", "")
            is_mistake = False  # 旧格式默认不是错误
            
            print(f"处理动作数据（旧格式）:")
            print(f"  skill_atomic_en: {skill_atomic}")
            print(f"  skill_detail: {skill_detail}")
            print(f"  en_skill_detail: {en_skill_detail}")
            print(f"  原始时间戳: {trigger_time}")
            print(f"  格式化时间戳: {formatted_trigger_time}")

        start_frame = None
        end_frame = None

        if (
            rosbag_actual_start_time is not None
            and rosbag_actual_end_time is not None
            and formatted_trigger_time
        ):
            try:
                if is_new_format:
                    # 新格式：使用 markStart 作为触发时间
                    # 解析 markStart 时间（格式：2026-01-06 09:41:20.781）
                    if mark_start and " " in mark_start:
                        dt_str, time_str = mark_start.split(" ", 1)
                        # 转换为 datetime 对象（假设是本地时间，+08:00）
                        trigger_datetime = datetime.datetime.fromisoformat(
                            f"{dt_str}T{time_str}+08:00"
                        )
                    else:
                        trigger_datetime = datetime.datetime.fromisoformat(
                            formatted_trigger_time
                        )
                    action_original_start_time = trigger_datetime.timestamp()
                    action_duration = float(duration)  # 已经是数字
                else:
                    # 旧格式：使用 triggerTime
                    trigger_datetime = datetime.datetime.fromisoformat(
                        formatted_trigger_time
                    )
                    action_original_start_time = trigger_datetime.timestamp()
                    # 解析持续时间
                    action_duration = 0
                    if duration_str.endswith("s"):
                        action_duration = float(duration_str[:-1])

                # 计算帧数
                start_frame, end_frame = calculate_action_frames(
                    rosbag_actual_start_time=rosbag_actual_start_time,
                    rosbag_actual_end_time=rosbag_actual_end_time,
                    rosbag_original_start_time=rosbag_original_start_time,
                    rosbag_original_end_time=rosbag_original_end_time,
                    action_original_start_time=action_original_start_time,
                    action_duration=action_duration,
                    frame_rate=frequency,
                    total_frames=total_frames,
                )

                print(f"动作: {skill_detail}")
                print(f"  动作时间: {trigger_datetime.isoformat()}")
                print(f"  原始开始时间: {action_original_start_time:.6f}s")
                print(
                    f"  原始结束时间: {action_original_start_time + action_duration:.6f}s"
                )
                print(f"  持续时间: {action_duration:.3f}s")
                print(f"  计算得到帧数: {start_frame} - {end_frame}")

                # 更详细的调试信息
                if start_frame is None or end_frame is None:
                    print(f"  调试信息:")
                    print(
                        f"    实际数据范围: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
                    )
                    print(
                        f"    动作时间范围: {action_original_start_time:.6f}s - {action_original_start_time + action_duration:.6f}s"
                    )
                    print(
                        f"    动作是否在数据范围内: {action_original_start_time >= rosbag_actual_start_time and action_original_start_time + action_duration <= rosbag_actual_end_time}"
                    )
                    print(
                        f"    动作开始是否在数据范围后: {action_original_start_time > rosbag_actual_end_time}"
                    )
                    print(
                        f"    动作结束是否在数据范围前: {action_original_start_time + action_duration < rosbag_actual_start_time}"
                    )

                # 验证计算结果
                if start_frame is not None and end_frame is not None:
                    actual_start_time = rosbag_actual_start_time + (
                        start_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    actual_end_time = rosbag_actual_start_time + (
                        end_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    print(f"  验证-实际开始时间: {actual_start_time:.6f}s")
                    print(f"  验证-实际结束时间: {actual_end_time:.6f}s")

                print("-" * 50)

            except Exception as e:
                print(f"计算帧数时出错: {e}")
                import traceback

                traceback.print_exc()

        # 构造新的 action 对象
        action = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "timestamp_utc": formatted_trigger_time,
            "is_mistake": is_mistake,
            "skill": skill_atomic,
            "action_text": skill_detail,
            "english_action_text": en_skill_detail,
        }
        action_config.append(action)

    # 按照 timestamp_utc 排序
    action_config = sorted(
        action_config, key=lambda x: x["timestamp_utc"] if x["timestamp_utc"] else ""
    )

    # 构造新json，episode_id放在最前
    new_json = OrderedDict()
    new_json["episode_id"] = uuid

    # 使用转换后的 metadata
    for k, v in metadata.items():
        new_json[k] = v

    if "label_info" not in new_json:
        new_json["label_info"] = {}
    new_json["label_info"]["action_config"] = action_config
    if "key_frame" not in new_json["label_info"]:
        new_json["label_info"]["key_frame"] = []

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)
    print(f"已保存到 {output_path}")


def get_time_range_from_moments(moments_json_path, metadata_json_path=None):
    """
    从 moments.json 或 metadata.json（新格式）文件中读取时间范围
    支持两种格式：
    1. 旧格式：从 moments.json 的 moments 数组中读取 start_position/end_position
    2. 新格式：从 metadata.json 的 marks 数组中读取 startPosition/endPosition

    Args:
        moments_json_path: moments.json 文件路径（旧格式）
        metadata_json_path: metadata.json 文件路径（新格式，可选）

    Returns:
        tuple: (start_time, end_time) 或 (None, None) 如果失败
    """
    # 优先尝试从新格式的 metadata.json 读取
    if metadata_json_path and os.path.exists(metadata_json_path):
        try:
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                metadata_data = json.load(f)
            
            # 检查是否为新格式（包含 marks 字段）
            if "marks" in metadata_data and isinstance(metadata_data.get("marks"), list):
                marks = metadata_data.get("marks", [])
                if not marks:
                    print(f"[MOMENTS] metadata.json中未找到marks数据")
                else:
                    start_positions = []
                    end_positions = []
                    
                    for mark in marks:
                        start_pos = mark.get("startPosition")
                        end_pos = mark.get("endPosition")
                        
                        if start_pos is not None:
                            try:
                                start_positions.append(float(start_pos))
                            except (ValueError, TypeError):
                                print(f"[MOMENTS] 无效的startPosition值: {start_pos}")
                                pass
                        
                        if end_pos is not None:
                            try:
                                end_positions.append(float(end_pos))
                            except (ValueError, TypeError):
                                print(f"[MOMENTS] 无效的endPosition值: {end_pos}")
                                pass
                    
                    if start_positions and end_positions:
                        moments_start_time = min(start_positions)
                        moments_end_time = max(end_positions)
                        
                        print(
                            f"[MOMENTS] 从metadata.json（新格式）获取时间范围: {moments_start_time} - {moments_end_time}"
                        )
                        print(
                            f"[MOMENTS] 找到 {len(start_positions)} 个startPosition, {len(end_positions)} 个endPosition"
                        )
                        
                        return moments_start_time, moments_end_time
                    else:
                        print(f"[MOMENTS] metadata.json中未找到有效的时间位置信息")
        except Exception as e:
            print(f"[MOMENTS] 读取metadata.json时出错: {e}")
    
    # 回退到旧格式：从 moments.json 读取
    if not moments_json_path or not os.path.exists(moments_json_path):
        return None, None

    try:
        with open(moments_json_path, "r", encoding="utf-8") as f:
            moments_data = json.load(f)

        moments = moments_data.get("moments", [])
        if not moments:
            print(f"[MOMENTS] moments.json中未找到moments数据")
            return None, None

        start_positions = []
        end_positions = []

        for moment in moments:
            custom_fields = moment.get("customFieldValues", {})
            start_pos = custom_fields.get("start_position")
            end_pos = custom_fields.get("end_position")

            if start_pos is not None:
                try:
                    start_positions.append(float(start_pos))
                except (ValueError, TypeError):
                    print(f"[MOMENTS] 无效的start_position值: {start_pos}")
                    pass

            if end_pos is not None:
                try:
                    end_positions.append(float(end_pos))
                except (ValueError, TypeError):
                    print(f"[MOMENTS] 无效的end_position值: {end_pos}")
                    pass

        # 使用最早的start_position和最晚的end_position
        if start_positions and end_positions:
            moments_start_time = min(start_positions)
            moments_end_time = max(end_positions)

            print(
                f"[MOMENTS] 从moments.json获取时间范围: {moments_start_time} - {moments_end_time}"
            )
            print(
                f"[MOMENTS] 找到 {len(start_positions)} 个start_position, {len(end_positions)} 个end_position"
            )

            return moments_start_time, moments_end_time
        else:
            print(f"[MOMENTS] moments.json中未找到有效的时间位置信息")
            return None, None

    except Exception as e:
        print(f"[MOMENTS] 读取moments.json时出错: {e}")
        return None, None


def get_bag_time_info(bag_path: str) -> dict:
    """
    获取 rosbag 包的时间信息

    Args:
        bag_path: rosbag 文件路径

    Returns:
        dict: 包含时间信息的字典，包括：
            - unix_timestamp: Unix时间戳（秒）
            - iso_format: ISO格式时间字符串（东八区）
            - nanoseconds: 纳秒格式时间戳
            - duration: bag持续时间（秒）
            - end_time: 结束时间Unix时间戳
    """
    try:
        bag = rosbag.Bag(bag_path, "r")
        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        bag_duration = bag_end_time - bag_start_time
        bag.close()

        # 转换为带时区的ISO格式（东八区）
        start_datetime = datetime.datetime.fromtimestamp(
            bag_start_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )
        start_iso = start_datetime.isoformat()

        # 转换为纳秒
        start_nanoseconds = int(bag_start_time * 1e9)

        return {
            "unix_timestamp": bag_start_time,
            "iso_format": start_iso,
            "nanoseconds": start_nanoseconds,
            "duration": bag_duration,
            "end_time": bag_end_time,
        }

    except Exception as e:
        print(f"获取bag时间信息失败: {e}")
        return {
            "unix_timestamp": None,
            "iso_format": None,
            "nanoseconds": None,
            "duration": None,
            "end_time": None,
        }


def list_bag_files_auto(raw_dir):
    bag_files = []
    for i, fname in enumerate(sorted(os.listdir(raw_dir))):
        if fname.endswith(".bag"):
            bag_files.append(
                {
                    "link": "",  # 保持为空
                    "start": 0,  # 批量设置为0
                    "end": 1,  # 批量设置为1
                    "local_path": os.path.join(raw_dir, fname),
                }
            )
    return bag_files


def load_raw_depth_lerobot(
    bag_data: dict, default_camera_names: list[str]
) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in default_camera_names:
        key = f"{camera}_depth"
        imgs_per_cam[camera] = np.array([msg["data"] for msg in bag_data[key]])
        # print(f"camera {camera} image", imgs_per_cam[camera].shape)

    return imgs_per_cam


def load_raw_depth_images_per_camera(bag_data: dict, default_camera_names: list[str]):
    imgs_per_cam = {}
    compressed_per_cam = {}
    for camera in default_camera_names:
        key = f"{camera}_depth"
        imgs_per_cam[camera] = [msg["data"] for msg in bag_data[key]]
        # 只取第一帧的压缩状态（假设所有帧一致）
        if bag_data[key]:
            compressed_per_cam[camera] = bag_data[key][0].get("compressed", None)
        else:
            compressed_per_cam[camera] = None
    print("+" * 20, compressed_per_cam)
    return imgs_per_cam, compressed_per_cam


def load_camera_info_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict:
    info_per_cam = {}
    distortion_model = {}
    for camera in default_camera_names:
        info_per_cam[camera] = np.array(
            [msg["data"] for msg in bag_data[f"{camera}_camera_info"]], dtype=np.float32
        )
        distortion_model[camera] = [
            msg["distortion_model"] for msg in bag_data[f"{camera}_camera_info"]
        ]
    return info_per_cam, distortion_model


def load_raw_images_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict[str, list]:
    imgs_per_cam = {}
    for camera in default_camera_names:
        imgs_per_cam[camera] = [msg["data"] for msg in bag_data[camera]]
    return imgs_per_cam


def load_raw_episode_data(
    raw_config: Config,
    ep_path: Path,
    start_time: float = 0,
    end_time: float = 1,
    action_config=None,
    min_duration: float = 5.0,
    metadata_json_dir: str = None,
):
    sn_code = None
    if metadata_json_dir and os.path.exists(metadata_json_dir):
        try:
            with open(metadata_json_dir, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
            sn_code = raw_metadata.get("device_sn", "")
        except Exception as e:
            print(f"[WARN] 读取metadata.json失败: {e})")
    bag_reader = KuavoRosbagReader(raw_config)
    bag_data = bag_reader.process_rosbag(
        ep_path, start_time=start_time, end_time=end_time, action_config=action_config
    )
    if sn_code is not None:
        main_time_line_timestamps = None
        if "camera_top" in bag_data and len(bag_data["camera_top"]) > 0:
            main_time_line_timestamps = np.array(
                [msg["timestamp"] for msg in bag_data["camera_top"]]
            )
        else:
            main_time_line_timestamps = None
        swap_left_right_data_if_needed(bag_data, sn_code, main_time_line_timestamps)
    # 1. 处理完 bag_data 后立即提取所需数据并清理
    sensors_data_raw__joint_q = state = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_q"]],
        dtype=np.float32,
    )
    joint_cmd__joint_q = action = np.array(
        [msg["data"] for msg in bag_data["action.joint_cmd.joint_q"]],
        dtype=np.float32,
    )
    kuavo_arm_traj__position = action_kuavo_arm_traj = np.array(
        [msg["data"] for msg in bag_data["action.kuavo_arm_traj"]],
        dtype=np.float32,
    )

    # 手部数据
    leju_claw_state__position = claw_state = np.array(
        [msg["data"] for msg in bag_data["observation.claw"]],
        dtype=np.float32,
    )
    leju_claw_command__position = claw_action = np.array(
        [msg["data"] for msg in bag_data["action.claw"]],
        dtype=np.float32,
    )

    # control_robot_hand_position_state_both = qiangnao_state = np.array(
    #     [msg["data"] for msg in bag_data["observation.qiangnao"]], dtype=np.float32,
    # )
    # control_robot_hand_position_both = qiangnao_action = np.array(
    #     [msg["data"] for msg in bag_data["action.qiangnao"]], dtype=np.float32,
    # )
    qiangnao_state = None
    try:
        qiangnao_state = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao"]],
            dtype=np.float32,
        )
    except KeyError:
        print("[WARN] 未找到 'observation.qiangnao' 数据")
    qiangnao_action = None
    try:
        qiangnao_action = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao"]],
            dtype=np.float32,
        )
    except KeyError:
        print("[WARN] 未找到 'action.qiangnao' 数据")

    hand_state_left = None
    hand_state_right = None
    hand_action_left = None
    hand_action_right = None

    if "observation.qiangnao_left" in bag_data:
        hand_state_left = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao_left"]],
            dtype=np.float32,
        )
    if "observation.qiangnao_right" in bag_data:
        hand_state_right = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao_right"]],
            dtype=np.float32,
        )
    if "action.qiangnao_left" in bag_data:
        hand_action_left = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao_left"]],
            dtype=np.float32,
        )
    if "action.qiangnao_right" in bag_data:
        hand_action_right = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao_right"]],
            dtype=np.float32,
        )

    if (
        (hand_state_left is None or hand_state_right is None)
        and qiangnao_state is not None
    ):
        split_left, split_right = _split_dexhand_lr(qiangnao_state)
        if split_left is not None:
            hand_state_left = split_left
        if split_right is not None:
            hand_state_right = split_right
    if (
        (hand_action_left is None or hand_action_right is None)
        and qiangnao_action is not None
    ):
        split_left, split_right = _split_dexhand_lr(qiangnao_action)
        if split_left is not None:
            hand_action_left = split_left
        if split_right is not None:
            hand_action_right = split_right

    # 速度和电流数据
    sensors_data_raw__joint_v = state_joint_v = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_v"]],
        dtype=np.float32,
    )
    state_joint_current = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_current"]],
        dtype=np.float32,
    )

    # 图像数据
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"[内存] 提取图像前: {mem_before:.1f} MB")

    imgs_per_cam = load_raw_images_per_camera(bag_data, raw_config.default_camera_names)
    mem_after_color = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 彩色图像提取后: {mem_after_color:.1f} MB (增长 {mem_after_color - mem_before:.1f} MB)"
    )

    imgs_per_cam_depth, compressed = load_raw_depth_images_per_camera(
        bag_data, raw_config.default_camera_names
    )
    mem_after_depth = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 深度图像提取后: {mem_after_depth:.1f} MB (增长 {mem_after_depth - mem_after_color:.1f} MB)"
    )

    info_per_cam, distortion_model = load_camera_info_per_camera(
        bag_data, raw_config.default_camera_names
    )
    mem_after_info = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 相机信息提取后: {mem_after_info:.1f} MB (增长 {mem_after_info - mem_after_depth:.1f} MB)"
    )
    main_time_line_timestamps = np.array(
        [msg["timestamp"] for msg in bag_data["camera_top"]]
    )
    if sn_code is not None:
        imgs_per_cam, imgs_per_cam_depth = flip_camera_arrays_if_needed(
            imgs_per_cam, imgs_per_cam_depth, sn_code, main_time_line_timestamps[0]
        )
    else:
        print("[WARN] 未提供sn_code，跳过相机翻转检测")
    # 时间戳和相机外参

    head_extrinsics = bag_data.get("head_camera_extrinsics", [])
    left_extrinsics = bag_data.get("left_hand_camera_extrinsics", [])
    right_extrinsics = bag_data.get("right_hand_camera_extrinsics", [])
    end_position = np.array(
        [msg["data"] for msg in bag_data["end.position"]],
        dtype=np.float32,
    )
    end_orientation = np.array(
        [msg["data"] for msg in bag_data["end.orientation"]],
        dtype=np.float32,
    )
    sensors_data_raw__imu_data = state_joint_imu = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.imu"]],
        dtype=np.float32,
    )

    # 2. 立即清理 bag_data 和 bag_reader
    mem_before_del = process.memory_info().rss / 1024 / 1024
    print(f"[内存] 删除 bag_data 前: {mem_before_del:.1f} MB")

    del bag_data
    del bag_reader
    gc.collect()

    mem_after_del = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 删除 bag_data 后: {mem_after_del:.1f} MB (释放 {mem_before_del - mem_after_del:.1f} MB)"
    )

    # 3. 处理电机数据（这些计算比较消耗内存）
    action[:, 12:26] = action_kuavo_arm_traj
    del action_kuavo_arm_traj  # 立即删除临时变量

    sensors_data_raw__joint_effort = state_joint_effort = (
        PostProcessorUtils.current_to_torque_batch(
            state_joint_current,
            MOTOR_C2T=[
                2,
                1.05,
                1.05,
                2,
                2.1,
                2.1,
                2,
                1.05,
                1.05,
                2,
                2.1,
                2.1,
                1.05,
                5,
                2.3,
                5,
                4.7,
                4.7,
                4.7,
                1.05,
                5,
                2.3,
                5,
                4.7,
                4.7,
                4.7,
                0.21,
                4.7,
            ],
        )
    )

    sensors_data_raw__joint_current = PostProcessorUtils.torque_to_current_batch(
        state_joint_current,
        MOTOR_C2T=[
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            0.21,
            4.7,
        ],
    )

    # 4. 提取子数组并清理原始数组
    head_effort = sensors_data_raw__joint_effort[:, 26:28]
    head_current = sensors_data_raw__joint_current[:, 26:28]
    joint_effort = sensors_data_raw__joint_effort[:, 12:26]
    joint_current = sensors_data_raw__joint_current[:, 12:26]

    # 清理一些不再需要的临时变量
    del state_joint_current
    gc.collect()

    # 5. 处理时间戳
    main_time_line_timestamps_ns = (main_time_line_timestamps * 1e9).astype(np.int64)

    velocity = None
    effort = None

    # 6. 构建 all_low_dim_data（这是返回的主要数据结构）
    all_low_dim_data = {
        "timestamps": main_time_line_timestamps_ns,
        "action": {
            "effector": {
                "position(gripper)": leju_claw_command__position,
                "index": main_time_line_timestamps_ns,
            },
            "hand_left": {
                "position": hand_action_left,
                "index": main_time_line_timestamps_ns,
            },
            "hand_right": {
                "position": hand_action_right,
                "index": main_time_line_timestamps_ns,
            },
            "joint": {
                "position": kuavo_arm_traj__position,
                "index": main_time_line_timestamps_ns,
            },
            "head": {
                "position": joint_cmd__joint_q[:, 26:28],
                "index": main_time_line_timestamps_ns,
            },
            "leg": {
                "position": joint_cmd__joint_q[:, :12],
                "index": main_time_line_timestamps_ns,
            },
        },
        "state": {
            "effector": {
                "position(gripper)": leju_claw_state__position,
            },
            "hand_left": {
                "position": hand_state_left,
            },
            "hand_right": {
                "position": hand_state_right,
            },
            "head": {
                "current_value": head_current,
                "effort": head_effort,
                "position": sensors_data_raw__joint_q[:, 26:28],
                "velocity": sensors_data_raw__joint_v[:, 26:28],
            },
            "joint": {
                "current_value": joint_current,
                "effort": joint_effort,
                "position": sensors_data_raw__joint_q[:, 12:26],
                "velocity": sensors_data_raw__joint_v[:, 12:26],
            },
            "end": {
                "position": end_position,
                "orientation": end_orientation,
            },
            "leg": {
                "current_value": sensors_data_raw__joint_current[:, :12],
                "effort": sensors_data_raw__joint_effort[:, :12],
                "position": sensors_data_raw__joint_q[:, 0:12],
                "velocity": sensors_data_raw__joint_v[:, 0:12],
            },
        },
        "imu": {
            "gyro_xyz": sensors_data_raw__imu_data[:, 0:3],
            "acc_xyz": sensors_data_raw__imu_data[:, 3:6],
            "free_acc_xyz": sensors_data_raw__imu_data[:, 6:9],
            "quat_xyzw": sensors_data_raw__imu_data[:, 9:13],
        },
    }

    # 7. 返回前最后一次内存清理
    del kuavo_arm_traj__position
    gc.collect()

    return (
        imgs_per_cam,
        imgs_per_cam_depth,
        info_per_cam,
        all_low_dim_data,
        main_time_line_timestamps,
        distortion_model,
        head_extrinsics,
        left_extrinsics,
        right_extrinsics,
        compressed,
        state,
        action,
        claw_state,
        claw_action,
        qiangnao_state,
        qiangnao_action,
    )


import multiprocessing


def load_raw_episode_worker(raw_config, ep_path, start_time, end_time, queue):
    try:
        result = load_raw_episode_data(
            raw_config=raw_config,
            ep_path=ep_path,
            start_time=start_time,
            end_time=end_time,
        )
        queue.put({"ok": True, "data": result})
    except Exception as e:
        import traceback

        queue.put({"ok": False, "error": str(e), "traceback": traceback.format_exc()})


def load_hand_data_worker(config, first_bag_path, first_start, first_end, queue):
    try:
        claw_state, claw_action, qiangnao_state, qiangnao_action = process_rosbag_eef(
            config, first_bag_path, start_time=first_start, end_time=first_end
        )
        queue.put(
            {
                "ok": True,
                "data": (claw_state, claw_action, qiangnao_state, qiangnao_action),
            }
        )
    except Exception as e:
        import traceback

        queue.put({"ok": False, "error": str(e), "traceback": traceback.format_exc()})


def process_rosbag_eef(config, bag_path, start_time=0, end_time=1):
    """
    只读取手部相关数据，不做时间戳对齐和话题筛选。
    只遍历需要的话题，返回 claw_state, claw_action, qiangnao_state, qiangnao_action
    """
    import rosbag
    import numpy as np

    claw_state = []
    claw_action = []
    qiangnao_state = []
    qiangnao_action = []
    cb_left_state = []
    cb_right_state = []
    cb_left_action = []
    cb_right_action = []

    # 话题名根据你的实际定义
    topic_claw_state = "/leju_claw_state"
    topic_claw_action = "/leju_claw_command"
    topic_qiangnao_state = "/control_robot_hand_position_state"
    topic_qiangnao_action = "/control_robot_hand_position"
    topic_cb_left_state = "/cb_left_hand_state"
    topic_cb_right_state = "/cb_right_hand_state"
    topic_cb_left_action = "/cb_left_hand_control_cmd"
    topic_cb_right_action = "/cb_right_hand_control_cmd"

    bag = rosbag.Bag(bag_path, "r")
    bag_start = bag.get_start_time()
    bag_end = bag.get_end_time()
    bag_duration = bag_end - bag_start

    abs_start = bag_start + start_time * bag_duration
    abs_end = bag_start + end_time * bag_duration

    # 只遍历需要的话题
    for topic, msg, t in bag.read_messages(
        topics=[
            topic_claw_state,
            topic_claw_action,
            topic_qiangnao_state,
            topic_qiangnao_action,
            topic_cb_left_state,
            topic_cb_right_state,
            topic_cb_left_action,
            topic_cb_right_action,
        ]
    ):
        if t.to_sec() < abs_start or t.to_sec() > abs_end:
            continue
        if topic == topic_claw_state:
            try:
                claw_state.append(np.array(msg.data.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_claw_action:
            try:
                claw_action.append(np.array(msg.data.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_qiangnao_state:
            try:
                state = list(msg.left_hand_position) + list(msg.right_hand_position)
                qiangnao_state.append(np.array(state, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_qiangnao_action:
            try:
                position = list(msg.left_hand_position) + list(msg.right_hand_position)
                qiangnao_action.append(np.array(position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_cb_left_state:
            try:
                cb_left_state.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_cb_right_state:
            try:
                cb_right_state.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_cb_left_action:
            try:
                cb_left_action.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_cb_right_action:
            try:
                cb_right_action.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass

    bag.close()

    claw_state = np.array(claw_state)
    claw_action = np.array(claw_action)
    qiangnao_state = np.array(qiangnao_state)
    qiangnao_action = np.array(qiangnao_action)
    cb_left_state = np.array(cb_left_state)
    cb_right_state = np.array(cb_right_state)
    cb_left_action = np.array(cb_left_action)
    cb_right_action = np.array(cb_right_action)

    if qiangnao_state.size == 0:
        if cb_left_state.size > 0:
            qiangnao_state = cb_left_state
        elif cb_right_state.size > 0:
            qiangnao_state = cb_right_state
    if qiangnao_action.size == 0:
        if cb_left_action.size > 0:
            qiangnao_action = cb_left_action
        elif cb_right_action.size > 0:
            qiangnao_action = cb_right_action

    return claw_state, claw_action, qiangnao_state, qiangnao_action


def _split_dexhand_lr(arr):
    if arr is None:
        return None, None
    arr = np.array(arr)
    if arr.ndim != 2 or arr.shape[1] < 12:
        return None, None
    left = arr[:, :6]
    right = arr[:, 6:12]
    return left, right


def port_kuavo_rosbag(
    raw_config: Config,
    repo_id: str = "lerobot/kuavo",
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    mode: Literal["video", "image"] = "video",
    processed_files: list[dict[str, str]] | list[str] = [],
    moment_json_DIR: str | None = None,
    metadata_json_DIR: str | None = None,
    lerobot_dir: str | None = None,
    use_depth: bool = True,
):

    from kuavo_dataset_slave_s import (
        KuavoRosbagReader,
        DEFAULT_JOINT_NAMES_LIST,
        DEFAULT_LEG_JOINT_NAMES,
        DEFAULT_ARM_JOINT_NAMES,
        DEFAULT_HEAD_JOINT_NAMES,
        DEFAULT_JOINT_NAMES,
        DEFAULT_LEJUCLAW_JOINT_NAMES,
        DEFAULT_DEXHAND_JOINT_NAMES,
        PostProcessorUtils,
    )

    config = raw_config

    # 处理并行 ROSbag 读取环境变量
    env_parallel = os.environ.get("USE_PARALLEL_ROSBAG_READ", "").lower()
    if env_parallel in ("true", "1", "yes"):
        config.use_parallel_rosbag_read = True
        print(
            "[CONFIG] 并行 ROSbag 读取已通过环境变量启用 (USE_PARALLEL_ROSBAG_READ=true)"
        )
    elif env_parallel in ("false", "0", "no"):
        config.use_parallel_rosbag_read = False

    env_workers = os.environ.get("PARALLEL_ROSBAG_WORKERS", "")
    if env_workers.isdigit():
        config.parallel_rosbag_workers = int(env_workers)
        print(f"[CONFIG] 并行 worker 数量: {config.parallel_rosbag_workers}")

    RAW_DIR = config.raw_dir
    ID = config.id
    CONTROL_HAND_SIDE = config.which_arm
    SLICE_ROBOT = config.slice_robot
    SLICE_DEX = config.dex_slice
    SLICE_CLAW = config.claw_slice
    IS_BINARY = config.is_binary
    DELTA_ACTION = config.delta_action
    RELATIVE_START = config.relative_start
    ONLY_HALF_UP_BODY = config.only_arm
    USE_LEJU_CLAW = config.use_leju_claw
    USE_QIANGNAO = config.use_qiangnao
    SEPARATE_HAND_FIELDS = getattr(config, "separate_hand_fields", False)
    MERGE_HAND_POSITION = getattr(config, "merge_hand_position", False)

    DEFAULT_JOINT_NAMES_LIST_ORIGIN = DEFAULT_JOINT_NAMES_LIST
    DEFAULT_ARM_JOINT_NAMES_ORIGIN = DEFAULT_ARM_JOINT_NAMES

    # 为整次导出创建 uuid 根目录
    episode_uuid = str(uuid.uuid4())
    base_root = os.path.join(lerobot_dir, episode_uuid)
    if os.path.exists(base_root):
        shutil.rmtree(base_root)
    os.makedirs(base_root, exist_ok=True)

    # 1) 读取第一个 bag，检测实际手型（与原逻辑一致）
    first_bag_info = processed_files[0]
    first_bag_path = (
        first_bag_info["local_path"]
        if isinstance(first_bag_info, dict)
        else first_bag_info
    )
    first_start = (
        first_bag_info.get("start", 0) if isinstance(first_bag_info, dict) else 0
    )
    first_end = first_bag_info.get("end", 1) if isinstance(first_bag_info, dict) else 1

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=load_hand_data_worker,
        args=(config, first_bag_path, first_start, first_end, queue),
    )
    p.start()
    result = queue.get()
    p.join()
    if not result.get("ok"):
        print("子进程异常退出！")
        print(result.get("error"))
        print(result.get("traceback"))
        sys.exit(1)

    (
        claw_state_probe,
        claw_action_probe,
        qiangnao_state_probe,
        qiangnao_action_probe,
    ) = result["data"]
    USE_LEJU_CLAW = is_valid_hand_data(claw_state_probe) or is_valid_hand_data(
        claw_action_probe
    )
    USE_QIANGNAO = is_valid_hand_data(qiangnao_state_probe) or is_valid_hand_data(
        qiangnao_action_probe
    )
    print(f"检测到手部类型: USE_LEJU_CLAW={USE_LEJU_CLAW}, USE_QIANGNAO={USE_QIANGNAO}")

    half_arm = len(DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(DEFAULT_DEXHAND_JOINT_NAMES) // 2
    UP_START_INDEX = 12
    if ONLY_HALF_UP_BODY:
        if SEPARATE_HAND_FIELDS:
            DEFAULT_ARM_JOINT_NAMES = DEFAULT_ARM_JOINT_NAMES_ORIGIN
        if USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            )
            arm_slice = [
                (
                    SLICE_ROBOT[0][0] - UP_START_INDEX,
                    SLICE_ROBOT[0][-1] - UP_START_INDEX,
                ),
                (SLICE_CLAW[0][0] + half_arm, SLICE_CLAW[0][-1] + half_arm),
                (
                    SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw,
                    SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw,
                ),
                (SLICE_CLAW[1][0] + half_arm * 2, SLICE_CLAW[1][-1] + half_arm * 2),
            ]
        elif USE_QIANGNAO and not SEPARATE_HAND_FIELDS:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]
            )
            arm_slice = [
                (
                    SLICE_ROBOT[0][0] - UP_START_INDEX,
                    SLICE_ROBOT[0][-1] - UP_START_INDEX,
                ),
                (SLICE_DEX[0][0] + half_arm, SLICE_DEX[0][-1] + half_arm),
                (
                    SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand,
                    SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand,
                ),
                (SLICE_DEX[1][0] + half_arm * 2, SLICE_DEX[1][-1] + half_arm * 2),
            ]
        if USE_QIANGNAO and SEPARATE_HAND_FIELDS:
            DEFAULT_JOINT_NAMES_LIST = DEFAULT_ARM_JOINT_NAMES
        else:
            DEFAULT_JOINT_NAMES_LIST = [
                DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)
            ]
    else:
        if SEPARATE_HAND_FIELDS:
            DEFAULT_ARM_JOINT_NAMES = DEFAULT_ARM_JOINT_NAMES_ORIGIN
        if USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            )
        elif USE_QIANGNAO and not SEPARATE_HAND_FIELDS:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]
            )
        DEFAULT_JOINT_NAMES_LIST = (
            DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES
        )
    if MERGE_HAND_POSITION:
        DEFAULT_JOINT_NAMES_LIST = (
            list(DEFAULT_JOINT_NAMES_LIST)
            + DEFAULT_DEXHAND_JOINT_NAMES[:6]
            + DEFAULT_DEXHAND_JOINT_NAMES[6:12]
        )

    @dataclasses.dataclass(frozen=True)
    class DatasetConfig:
        use_videos: bool = True
        tolerance_s: float = 0.0001
        image_writer_processes: int = 6
        image_writer_threads: int = 12
        video_backend: str | None = None

    DEFAULT_DATASET_CONFIG = DatasetConfig()
    dataset_config = DEFAULT_DATASET_CONFIG

    def create_empty_dataset(
        repo_id: str,
        robot_type: str,
        mode: Literal["video", "image"] = "video",
        eef_type: Literal["leju_claw", "dex_hand"] = "dex_hand",
        *,
        has_depth_image: bool = False,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
        root: str,
        extra_features: bool = True,
        raw_config: Config,
    ) -> LeRobotDataset:
        dexhand = [
            "left_linkerhand_1",
            "left_linkerhand_2",
            "left_linkerhand_3",
            "left_linkerhand_4",
            "left_linkerhand_5",
            "left_linkerhand_6",
            "right_linkerhand_1",
            "right_linkerhand_2",
            "right_linkerhand_3",
            "right_linkerhand_4",
            "right_linkerhand_5",
            "right_linkerhand_6",
        ]
        lejuclaw = [
            "left_claw",
            "right_claw",
        ]
        leg = [
            "l_leg_roll",
            "l_leg_yaw",
            "l_leg_pitch",
            "l_knee",
            "l_foot_pitch",
            "l_foot_roll",
            "r_leg_roll",
            "r_leg_yaw",
            "r_leg_pitch",
            "r_knee",
            "r_foot_pitch",
            "r_foot_roll",
        ]
        arm = [
            "zarm_l1_link",
            "zarm_l2_link",
            "zarm_l3_link",
            "zarm_l4_link",
            "zarm_l5_link",
            "zarm_l6_link",
            "zarm_l7_link",
            "zarm_r1_link",
            "zarm_r2_link",
            "zarm_r3_link",
            "zarm_r4_link",
            "zarm_r5_link",
            "zarm_r6_link",
            "zarm_r7_link",
        ]
        head = ["head_yaw", "head_pitch"]
        cameras = raw_config.default_camera_names
        imu_acc = ["acc_x", "acc_y", "acc_z"]
        imu_free_acc = ["free_acc_x", "ree_acc_y", "free_acc_z"]
        imu_gyro_acc = ["gyro_x", "gyro_y", "gyro_z"]
        imu_quat_acc = ["quat_x", "quat_y", "quat_z", "quat_w"]
        end_orientation = [
            "left_x",
            "left_y",
            "left_z",
            "left_w",
            "right_x",
            "right_y",
            "right_z",
            "right_w",
        ]
        end_position = ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]
        # 根据末端执行器类型定义特征
        features = {
            "observation.state.arm.position": {
                "dtype": "float32",
                "shape": (14,),
                "names": arm,
            },
            "observation.state.arm.effort": {
                "dtype": "float32",
                "shape": (14,),
                "names": arm,
            },
            "observation.state.arm.velocity": {
                "dtype": "float32",
                "shape": (14,),
                "names": arm,
            },
            "observation.state.arm.current_value": {
                "dtype": "float32",
                "shape": (14,),
                "names": arm,
            },
            "observation.state.end.position": {
                "dtype": "float32",
                "shape": (6,),
                "names": end_position,
            },
            "observation.state.end.orientation": {
                "dtype": "float32",
                "shape": (8,),
                "names": end_orientation,
            },
            # "observation.state.head.position" : {"dtype": "float32", "shape": (2,), "names": head},
            "observation.state.head.effort": {
                "dtype": "float32",
                "shape": (2,),
                "names": head,
            },
            "observation.state.head.position": {
                "dtype": "float32",
                "shape": (2,),
                "names": head,
            },
            "observation.state.head.velocity": {
                "dtype": "float32",
                "shape": (2,),
                "names": head,
            },
            "observation.state.leg.effort": {
                "dtype": "float32",
                "shape": (12,),
                "names": leg,
            },
            "observation.state.leg.position": {
                "dtype": "float32",
                "shape": (12,),
                "names": leg,
            },
            "observation.state.leg.velocity": {
                "dtype": "float32",
                "shape": (12,),
                "names": leg,
            },
            "observation.state.leg.current_value": {
                "dtype": "float32",
                "shape": (12,),
                "names": leg,
            },
            "action.head.position": {"dtype": "float32", "shape": (2,), "names": head},
            "action.arm.position": {"dtype": "float32", "shape": (14,), "names": arm},
            "action.leg.position": {"dtype": "float32", "shape": (12,), "names": leg},
            "imu.acc_xyz": {"dtype": "float32", "shape": (3,), "names": imu_acc},
            "imu.free_acc_xyz": {
                "dtype": "float32",
                "shape": (3,),
                "names": imu_free_acc,
            },
            "imu.gyro_xyz": {
                "dtype": "float32",
                "shape": (3,),
                "names": imu_gyro_acc,
            },
            "imu.quat_xyzw": {
                "dtype": "float32",
                "shape": (4,),
                "names": imu_quat_acc,
            },
        }

        # 根据末端执行器类型添加相应的特征
        if eef_type == "leju_claw":
            features.update(
                {
                    "action.effector.position": {
                        "dtype": "float32",
                        "shape": (2,),
                        "names": lejuclaw,
                    },
                    "observation.state.effector.position": {
                        "dtype": "float32",
                        "shape": (2,),
                        "names": lejuclaw,
                    },
                }
            )
        elif eef_type == "dex_hand":
            features.update(
                {
                    "action.hand_left.position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": dexhand[:6],
                    },
                    "action.hand_right.position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": dexhand[6:],
                    },
                    "observation.state.hand_left.position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": dexhand[:6],
                    },
                    "observation.state.hand_right.position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": dexhand[6:],
                    },
                    "observation.state.hand_left.force_torque": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": [
                            "force_x",
                            "force_y",
                            "force_z",
                            "torque_x",
                            "torque_y",
                            "torque_z",
                        ],
                    },
                    "observation.state.hand_right.force_torque": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": [
                            "force_x",
                            "force_y",
                            "force_z",
                            "torque_x",
                            "torque_y",
                            "torque_z",
                        ],
                    },
                    "observation.state.hand_left.touch_matrix": {
                        "dtype": "float32",
                        "shape": (360,),
                        "names": None,
                    },
                    "observation.state.hand_right.touch_matrix": {
                        "dtype": "float32",
                        "shape": (360,),
                        "names": None,
                    },
                }
            )

        # 相机特征：如果视频单独存储，不添加图像features
        separate_video_storage = getattr(raw_config, "separate_video_storage", False)

        if not separate_video_storage:
            # 原有逻辑：添加图像/视频features
            for cam in cameras:
                features[f"observation.images.{cam}"] = {
                    "dtype": mode,
                    "shape": (3, 480, 848),
                    "names": ["channels", "height", "width"],
                }
                if has_depth_image:
                    features[f"observation.images.depth.{cam}"] = {
                        "dtype": mode,
                        "shape": (480, 848),
                        "names": ["height", "width"],
                    }

        for cam in cameras:
            features[f"observation.camera_params.rotation_matrix_flat.{cam}"] = {
                "dtype": "float32",
                "shape": (9,),
                "names": None,
            }
            features[f"observation.camera_params.translation_vector.{cam}"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": None,
            }
        if extra_features:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (len(DEFAULT_JOINT_NAMES_LIST),),
                "names": DEFAULT_JOINT_NAMES_LIST,
            }
            features["action"] = {
                "dtype": "float32",
                "shape": (len(DEFAULT_JOINT_NAMES_LIST),),
                "names": DEFAULT_JOINT_NAMES_LIST,
            }
            print("DEFAULT_JOINT_NAMES_LIST", DEFAULT_JOINT_NAMES_LIST)

        if Path(LEROBOT_HOME / repo_id).exists():
            shutil.rmtree(LEROBOT_HOME / repo_id)

        # 如果视频单独存储，features中已经没有video类型，use_videos保持原值即可
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=raw_config.train_hz,
            robot_type=robot_type,
            features=features,
            use_videos=dataset_config.use_videos,
            tolerance_s=dataset_config.tolerance_s,
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads,
            video_backend=dataset_config.video_backend,
            root=root,
        )

    def populate_dataset_stream(
        raw_config: Config,
        bag_files: list,
        task: str,
        moment_json_dir: str | None,
        base_root: str,
        metadata_json_dir: str | None = None,
        pipeline_encoder: "BatchSegmentEncoder | None" = None,
        streaming_encoder: "StreamingVideoEncoderManager | None" = None,
    ):
        import psutil

        process = psutil.Process()

        # 读取 metadata.json 获取 sn_code（相机左右翻转判定）
        sn_code = None
        if metadata_json_dir and os.path.exists(metadata_json_dir):
            try:
                with open(metadata_json_dir, "r", encoding="utf-8") as f:
                    raw_metadata = json.load(f)
                # 支持新格式（deviceSn）和旧格式（device_sn）
                sn_code = raw_metadata.get("deviceSn") or raw_metadata.get("device_sn", "")
            except Exception as e:
                print(f"[WARN] 读取metadata.json失败: {e})")

        if len(bag_files) == 0:
            print("[WARN] 无 bag 文件")
            return None, None

        # 遍历每个 bag
        for ep_idx, bag_info in enumerate(bag_files):
            if isinstance(bag_info, dict):
                ep_path = bag_info["local_path"]
                start_time = bag_info.get("start", 0)
                end_time = bag_info.get("end", 1)
            else:
                ep_path = bag_info
                start_time = 0
                end_time = 1

            # moments.json 或 metadata.json（新格式）覆盖时间窗
            moments_start_time, moments_end_time = get_time_range_from_moments(
                moment_json_dir, metadata_json_path=metadata_json_dir
            )
            if moments_start_time is not None and moments_end_time is not None:
                print(
                    f"[MOMENTS] 覆盖使用标注文件时间范围: {moments_start_time} - {moments_end_time}"
                )
                start_time = moments_start_time
                end_time = moments_end_time

            # bag 时间信息（用于 metadata 合并）
            bag_time_info = get_bag_time_info(ep_path)
            if bag_time_info["iso_format"]:
                print(f"Bag开始时间: {bag_time_info['iso_format']}")
                print(f"Bag持续时间: {bag_time_info['duration']:.2f}秒")

            # 流式 reader
            reader = KuavoRosbagReader(raw_config, use_depth)
            extrinsics_dict = {}
            # 逐批消费
            batch_id = 0
            _t_prev_batch_end = time.time()  # 用于计算 generator yield 耗时

            # 提前获取配置，避免循环体内未定义
            separate_video_storage = getattr(
                raw_config, "separate_video_storage", False
            )
            cam_stats = {}  # 初始化，避免无 batch 时未定义

            # 选择串行或并行读取
            use_parallel = getattr(raw_config, "use_parallel_rosbag_read", False)
            num_workers = getattr(raw_config, "parallel_rosbag_workers", 2)

            if use_parallel:
                print(f"[STREAM] 启用并行 ROSbag 读取 ({num_workers} workers)")
                batch_iter = reader.process_rosbag_parallel(
                    str(ep_path),
                    start_time=start_time,
                    end_time=end_time,
                    action_config=None,
                    chunk_size=800,
                    num_workers=num_workers,
                )
            else:
                batch_iter = reader.process_rosbag(
                    str(ep_path),
                    start_time=start_time,
                    end_time=end_time,
                    action_config=None,
                    chunk_size=800,
                )
            for aligned_batch in batch_iter:
                batch_id += 1
                _t_batch_start = time.time()
                _t_rosbag_read = (
                    _t_batch_start - _t_prev_batch_end
                )  # ROSbag读取+对齐时间
                main_key = getattr(reader, "MAIN_TIMESTAMP_TOPIC", "camera_top")
                if main_key not in aligned_batch or len(aligned_batch[main_key]) == 0:
                    print(f"[STREAM][WARN] 批次{batch_id} 无主时间线，跳过")
                    continue

                # 主时间戳
                main_ts = np.array(
                    [it["timestamp"] for it in aligned_batch[main_key]],
                    dtype=np.float64,
                )

                first_ts = float(main_ts[0])
                last_ts = float(main_ts[-1])

                # 每批提取相机外参（按时间窗）
                if batch_id == 1:
                    try:
                        extrinsics = reader.extract_and_format_camera_extrinsics(
                            str(ep_path), abs_start=first_ts, abs_end=last_ts
                        )
                        head_extrinsics = extrinsics.get("head_camera_extrinsics", [])
                        left_extrinsics = extrinsics.get(
                            "left_hand_camera_extrinsics", []
                        )
                        right_extrinsics = extrinsics.get(
                            "right_hand_camera_extrinsics", []
                        )
                    except Exception as e:
                        print(f"[WARN] 批次{batch_id} 外参提取失败: {e}")
                        head_extrinsics, left_extrinsics, right_extrinsics = [], [], []

                # 颜色/深度/相机信息
                _t_extract_start = time.time()
                cameras = raw_config.default_camera_names
                imgs_per_cam = {
                    cam: [x["data"] for x in aligned_batch.get(cam, [])]
                    for cam in cameras
                }
                if use_depth:
                    imgs_per_cam_depth = {
                        cam: [x["data"] for x in aligned_batch.get(f"{cam}_depth", [])]
                        for cam in cameras
                    }
                    compressed = {
                        cam: (
                            aligned_batch.get(f"{cam}_depth", [])[0].get(
                                "compressed", None
                            )
                            if len(aligned_batch.get(f"{cam}_depth", [])) > 0
                            else None
                        )
                        for cam in cameras
                    }
                else:
                    imgs_per_cam_depth = None
                    compressed = None
                info_per_cam = {
                    cam: [
                        np.array(x["data"], dtype=np.float32)
                        for x in aligned_batch.get(f"{cam}_camera_info", [])
                    ]
                    for cam in cameras
                }
                distortion_model = {
                    cam: [
                        x.get("distortion_model", None)
                        for x in aligned_batch.get(f"{cam}_camera_info", [])
                    ]
                    for cam in cameras
                }

                # 相机翻转（基于 sn_code）
                # if sn_code is not None and len(main_ts) > 0:
                #     imgs_per_cam, imgs_per_cam_depth = flip_camera_arrays_if_needed(
                #         imgs_per_cam, imgs_per_cam_depth, sn_code, main_ts[0]
                #     )

                # 低维数据/末端位姿
                def get_arr(key, dflt_shape=None):
                    items = aligned_batch.get(key, [])
                    if not items:
                        return None
                    return np.array([x["data"] for x in items], dtype=np.float32)

                # print(get_arr("observation.sensorsData.joint_q").shape)
                state = get_arr(
                    "observation.sensorsData.joint_q"
                )  # or np.zeros((0, 28), dtype=np.float32)
                # sensors_data_raw__joint_v = get_arr("observation.sensorsData.joint_v") #or np.zeros((len(state), 28), dtype=np.float32)
                state_joint_current = get_arr(
                    "observation.sensorsData.joint_current"
                )  # or np.zeros((len(state), 28), dtype=np.float32)
                action = get_arr(
                    "action.joint_cmd.joint_q"
                )  # or np.zeros((0, 28), dtype=np.float32)
                action_kuavo_arm_traj = get_arr(
                    "action.kuavo_arm_traj"
                )  # or np.zeros((0, 14), dtype=np.float32)
                sensors_data_raw__joint_v = get_arr(
                    "observation.sensorsData.joint_v"
                )  # or np.zeros((len(state), 28), dtype=np.float32)
                state_joint_current_arr = get_arr(
                    "observation.sensorsData.joint_current"
                )  # or np.zeros((len(state), 28), dtype=np.float32)
                sensors_data_raw__imu_data = get_arr(
                    "observation.sensorsData.imu"
                )  # or np.zeros((len(state), 13), dtype=np.float32)

                claw_state = get_arr(
                    "observation.claw"
                )  # or np.zeros((len(state), 2), dtype=np.float32)
                claw_action = get_arr(
                    "action.claw"
                )  # or np.zeros((len(state), 2), dtype=np.float32)
                qiangnao_state = get_arr("observation.qiangnao")
                qiangnao_action = get_arr("action.qiangnao")
                hand_state_left = get_arr("observation.qiangnao_left")
                hand_state_right = get_arr("observation.qiangnao_right")
                hand_action_left = get_arr("action.qiangnao_left")
                hand_action_right = get_arr("action.qiangnao_right")
                hand_force_left = get_arr("observation.state.hand_left.force_torque")
                hand_force_right = get_arr("observation.state.hand_right.force_torque")
                hand_touch_left = get_arr("observation.state.hand_left.touch_matrix")
                hand_touch_right = get_arr("observation.state.hand_right.touch_matrix")
                if (
                    (hand_state_left is None or hand_state_right is None)
                    and qiangnao_state is not None
                ):
                    split_left, split_right = _split_dexhand_lr(qiangnao_state)
                    if split_left is not None:
                        hand_state_left = split_left
                    if split_right is not None:
                        hand_state_right = split_right
                if (
                    (hand_action_left is None or hand_action_right is None)
                    and qiangnao_action is not None
                ):
                    split_left, split_right = _split_dexhand_lr(qiangnao_action)
                    if split_left is not None:
                        hand_action_left = split_left
                    if split_right is not None:
                        hand_action_right = split_right

                end_position = get_arr(
                    "end.position"
                )  # or np.zeros((len(state), 6), dtype=np.float32)
                end_orientation = get_arr(
                    "end.orientation"
                )  # or np.zeros((len(state), 8), dtype=np.float32)

                # 填充 action 的关节子段（12:26）为 kuavo_arm_traj
                if action.size > 0 and action_kuavo_arm_traj.size > 0:
                    min_rows = min(len(action), len(action_kuavo_arm_traj))
                    action[:min_rows, 12:26] = action_kuavo_arm_traj[:min_rows]

                sensors_data_raw__joint_effort = state_joint_effort = (
                    PostProcessorUtils.current_to_torque_batch(
                        state_joint_current,
                        MOTOR_C2T=[
                            2,
                            1.05,
                            1.05,
                            2,
                            2.1,
                            2.1,
                            2,
                            1.05,
                            1.05,
                            2,
                            2.1,
                            2.1,
                            1.05,
                            5,
                            2.3,
                            5,
                            4.7,
                            4.7,
                            4.7,
                            1.05,
                            5,
                            2.3,
                            5,
                            4.7,
                            4.7,
                            4.7,
                            0.21,
                            4.7,
                        ],
                    )
                )

                sensors_data_raw__joint_current = (
                    PostProcessorUtils.torque_to_current_batch(
                        state_joint_current,
                        MOTOR_C2T=[
                            2,
                            1.05,
                            1.05,
                            2,
                            2.1,
                            2.1,
                            2,
                            1.05,
                            1.05,
                            2,
                            2.1,
                            2.1,
                            1.05,
                            5,
                            2.3,
                            5,
                            4.7,
                            4.7,
                            4.7,
                            1.05,
                            5,
                            2.3,
                            5,
                            4.7,
                            4.7,
                            4.7,
                            0.21,
                            4.7,
                        ],
                    )
                )

                # 4. 提取子数组并清理原始数组
                head_effort = sensors_data_raw__joint_effort[:, 26:28]
                head_current = sensors_data_raw__joint_current[:, 26:28]
                joint_effort = sensors_data_raw__joint_effort[:, 12:26]
                joint_current = sensors_data_raw__joint_current[:, 12:26]

                # all_low_dim_data（按批次）
                main_ts_ns = (main_ts * 1e9).astype(np.int64)
                all_low_dim_data = {
                    "timestamps": main_ts_ns,
                    "action": {
                        "effector": {
                            "position": claw_action,
                            "index": main_ts_ns,
                        },
                        "hand_left": {
                            "position": hand_action_left,
                            "index": main_ts_ns,
                        },
                        "hand_right": {
                            "position": hand_action_right,
                            "index": main_ts_ns,
                        },
                        "arm": {
                            "position": action_kuavo_arm_traj,
                            "index": main_ts_ns,
                        },
                        "head": {
                            "position": action[:, 26:28],
                            "index": main_ts_ns,
                        },
                        "leg": {
                            "position": action[:, :12],
                            "index": main_ts_ns,
                        },
                    },
                    "state": {
                        "effector": {
                            "position": claw_state,
                        },
                        "hand_left": {
                            "position": hand_state_left,
                            "force_torque": hand_force_left,
                            "touch_matrix": hand_touch_left,
                        },
                        "hand_right": {
                            "position": hand_state_right,
                            "force_torque": hand_force_right,
                            "touch_matrix": hand_touch_right,
                        },
                        "head": {
                            "current_value": head_current,
                            "effort": head_effort,
                            "position": state[:, 26:28],
                            "velocity": sensors_data_raw__joint_v[:, 26:28],
                        },
                        "arm": {
                            "current_value": joint_current,
                            "effort": joint_effort,
                            "position": state[:, 12:26],
                            "velocity": sensors_data_raw__joint_v[:, 12:26],
                        },
                        "end": {
                            "position": end_position,
                            "orientation": end_orientation,
                        },
                        "leg": {
                            "current_value": sensors_data_raw__joint_current[:, :12],
                            "effort": sensors_data_raw__joint_effort[:, :12],
                            "position": state[:, 0:12],
                            "velocity": sensors_data_raw__joint_v[:, 0:12],
                        },
                    },
                    "imu": {
                        "gyro_xyz": sensors_data_raw__imu_data[:, 0:3],
                        "acc_xyz": sensors_data_raw__imu_data[:, 3:6],
                        "free_acc_xyz": sensors_data_raw__imu_data[:, 6:9],
                        "quat_xyzw": sensors_data_raw__imu_data[:, 9:13],
                    },
                }
                _t_extract_end = time.time()

                # 为该批创建独立数据集 root: {base_root}/batch_{id}
                batch_root = os.path.join(base_root, f"batch_{batch_id:04d}")
                # os.makedirs(batch_root, exist_ok=True)
                use_leju_claw_batch = (
                    USE_LEJU_CLAW
                    and claw_state is not None
                    and claw_action is not None
                    and len(claw_state) > 0
                    and len(claw_action) > 0
                )
                use_qiangnao_batch = (
                    USE_QIANGNAO
                    and hand_state_left is not None
                    and hand_state_right is not None
                    and hand_action_left is not None
                    and hand_action_right is not None
                    and len(hand_state_left) > 0
                    and len(hand_state_right) > 0
                    and len(hand_action_left) > 0
                    and len(hand_action_right) > 0
                )
                eef_type = "leju_claw" if use_leju_claw_batch else "dex_hand"
                _t_create_dataset_start = time.time()
                dataset = create_empty_dataset(
                    repo_id=f"lerobot/kuavo",
                    robot_type="kuavo4pro",
                    mode=mode,
                    eef_type=eef_type,
                    dataset_config=dataset_config,
                    has_depth_image=use_depth,
                    root=batch_root,
                    raw_config=raw_config,
                )
                _t_create_dataset_end = time.time()

                # 帧写入（与原逻辑一致）
                if batch_id == 1:
                    extrinsics_map = {
                        "camera_top": head_extrinsics,
                        "camera_wrist_left": left_extrinsics,
                        "camera_wrist_right": right_extrinsics,
                        "head_cam_h": head_extrinsics,
                        "wrist_cam_l": left_extrinsics,
                        "wrist_cam_r": right_extrinsics,
                    }
                    extrinsics_dict = {
                        cam: extrinsics_map[cam]
                        for cam in cameras
                        if cam in extrinsics_map
                    }

                num_frames = state.shape[0]
                print(f"[STREAM] 批次{batch_id} 写入 {num_frames} 帧")

                _t_frame_loop_start = time.time()
                for i in range(num_frames):
                    # 高效构造 output_state / output_action：预分配 + 切片赋值，避免多次 concatenate/insert
                    if ONLY_HALF_UP_BODY:
                        if use_leju_claw_batch:
                            if CONTROL_HAND_SIDE in ("left", "both"):
                                l0, l1 = SLICE_ROBOT[0][0], SLICE_ROBOT[0][-1]
                                c0, c1 = SLICE_CLAW[0][0], SLICE_CLAW[0][-1]
                                left_len = (l1 - l0) + (c1 - c0)
                                output_state = np.empty((left_len,), dtype=np.float32)
                                output_action = np.empty((left_len,), dtype=np.float32)
                                output_state[: (l1 - l0)] = state[i, l0:l1]
                                output_state[(l1 - l0) :] = claw_state[i, c0:c1]
                                output_action[: (l1 - l0)] = action[i, l0:l1]
                                output_action[(l1 - l0) :] = claw_action[i, c0:c1]
                            if CONTROL_HAND_SIDE in ("right", "both"):
                                r0, r1 = SLICE_ROBOT[1][0], SLICE_ROBOT[1][-1]
                                rc0, rc1 = SLICE_CLAW[1][0], SLICE_CLAW[1][-1]
                                right_len = (r1 - r0) + (rc1 - rc0)
                                right_state = np.empty((right_len,), dtype=np.float32)
                                right_action = np.empty((right_len,), dtype=np.float32)
                                right_state[: (r1 - r0)] = state[i, r0:r1]
                                right_state[(r1 - r0) :] = claw_state[i, rc0:rc1]
                                right_action[: (r1 - r0)] = action[i, r0:r1]
                                right_action[(r1 - r0) :] = claw_action[i, rc0:rc1]
                                if CONTROL_HAND_SIDE == "both":
                                    # 仅一次拼接
                                    output_state = np.concatenate(
                                        (output_state, right_state), axis=0
                                    )
                                    output_action = np.concatenate(
                                        (output_action, right_action), axis=0
                                    )
                                else:
                                    output_state = right_state
                                    output_action = right_action

                        else:
                            if CONTROL_HAND_SIDE in ("left", "both"):
                                l0, l1 = SLICE_ROBOT[0][0], SLICE_ROBOT[0][-1]
                                output_state = np.array(state[i, l0:l1], dtype=np.float32)
                                output_action = np.array(action[i, l0:l1], dtype=np.float32)
                            if CONTROL_HAND_SIDE in ("right", "both"):
                                r0, r1 = SLICE_ROBOT[1][0], SLICE_ROBOT[1][-1]
                                right_state = np.array(state[i, r0:r1], dtype=np.float32)
                                right_action = np.array(action[i, r0:r1], dtype=np.float32)
                                if CONTROL_HAND_SIDE == "both":
                                    output_state = np.concatenate((output_state, right_state), axis=0)
                                    output_action = np.concatenate((output_action, right_action), axis=0)
                                else:
                                    output_state = right_state
                                    output_action = right_action

                    else:
                        if use_leju_claw_batch:
                            # 全身 + 爪：目标长度 = 28 原关节 + 2 爪 = 30
                            output_state = np.empty((30,), dtype=np.float32)
                            output_action = np.empty((30,), dtype=np.float32)
                            # 0:19 原始
                            output_state[0:19] = state[i, 0:19]
                            output_action[0:19] = action[i, 0:19]
                            # 左爪放在索引19
                            output_state[19] = float(claw_state[i, 0])
                            output_action[19] = float(claw_action[i, 0])
                            # 20:27 原 19:26
                            output_state[20:27] = state[i, 19:26]
                            output_action[20:27] = action[i, 19:26]
                            # 右爪放在索引27
                            output_state[27] = float(claw_state[i, 1])
                            output_action[27] = float(claw_action[i, 1])
                            # 28:30 头部
                            output_state[28:30] = state[i, 26:28]
                            output_action[28:30] = action[i, 26:28]

                        else:
                            output_state = np.array(state[i, :], dtype=np.float32)
                            output_action = np.array(action[i, :], dtype=np.float32)

                    if MERGE_HAND_POSITION:
                        left_pos = (
                            hand_state_left[i]
                            if hand_state_left is not None and len(hand_state_left) > i
                            else np.zeros((6,), dtype=np.float32)
                        )
                        right_pos = (
                            hand_state_right[i]
                            if hand_state_right is not None and len(hand_state_right) > i
                            else np.zeros((6,), dtype=np.float32)
                        )
                        left_act = (
                            hand_action_left[i]
                            if hand_action_left is not None and len(hand_action_left) > i
                            else np.zeros((6,), dtype=np.float32)
                        )
                        right_act = (
                            hand_action_right[i]
                            if hand_action_right is not None and len(hand_action_right) > i
                            else np.zeros((6,), dtype=np.float32)
                        )
                        output_state = np.concatenate((output_state, left_pos, right_pos), axis=0)
                        output_action = np.concatenate((output_action, left_act, right_act), axis=0)

                    frame = {
                        "observation.state": torch.from_numpy(output_state).type(
                            torch.float32
                        ),
                        "action": torch.from_numpy(output_action).type(torch.float32),
                        "action.head.position": get_nested_value(
                            all_low_dim_data, "action.head.position", i, [0.0] * 2
                        ),
                        "action.arm.position": get_nested_value(
                            all_low_dim_data, "action.arm.position", i, [0.0] * 14
                        ),
                        "action.leg.position": get_nested_value(
                            all_low_dim_data, "action.leg.position", i, [0.0] * 12
                        ),
                        "observation.state.head.effort": get_nested_value(
                            all_low_dim_data, "state.head.effort", i, [0.0] * 2
                        ),
                        "observation.state.head.position": get_nested_value(
                            all_low_dim_data, "state.head.position", i, [0.0] * 2
                        ),
                        "observation.state.head.velocity": get_nested_value(
                            all_low_dim_data, "state.head.velocity", i, [0.0] * 2
                        ),
                        "observation.state.arm.current_value": get_nested_value(
                            all_low_dim_data, "state.arm.current_value", i, [0.0] * 14
                        ),
                        "observation.state.arm.effort": get_nested_value(
                            all_low_dim_data, "state.arm.effort", i, [0.0] * 14
                        ),
                        "observation.state.arm.position": get_nested_value(
                            all_low_dim_data, "state.arm.position", i, [0.0] * 14
                        ),
                        "observation.state.arm.velocity": get_nested_value(
                            all_low_dim_data, "state.arm.velocity", i, [0.0] * 14
                        ),
                        # 展平末端左右手姿态和位置
                        "observation.state.end.orientation": (
                            get_nested_value(
                                all_low_dim_data, "state.end.orientation", i, [0.0] * 8
                            )
                        ).flatten(),
                        "observation.state.end.position": (
                            get_nested_value(
                                all_low_dim_data, "state.end.position", i, [0.0] * 6
                            )
                        ).flatten(),
                        "observation.state.leg.current_value": get_nested_value(
                            all_low_dim_data, "state.leg.current_value", i, [0.0] * 12
                        ),
                        "observation.state.leg.effort": get_nested_value(
                            all_low_dim_data, "state.leg.effort", i, [0.0] * 12
                        ),
                        "observation.state.leg.position": get_nested_value(
                            all_low_dim_data, "state.leg.position", i, [0.0] * 12
                        ),
                        "observation.state.leg.velocity": get_nested_value(
                            all_low_dim_data, "state.leg.velocity", i, [0.0] * 12
                        ),
                        "imu.acc_xyz": get_nested_value(
                            all_low_dim_data, "imu.acc_xyz", i, [0.0] * 3
                        ),
                        "imu.gyro_xyz": get_nested_value(
                            all_low_dim_data, "imu.gyro_xyz", i, [0.0] * 3
                        ),
                        "imu.free_acc_xyz": get_nested_value(
                            all_low_dim_data, "imu.free_acc_xyz", i, [0.0] * 3
                        ),
                        "imu.quat_xyzw": get_nested_value(
                            all_low_dim_data, "imu.quat_xyzw", i, [0.0] * 4
                        ),
                    }

                    # 末端类型
                    if USE_LEJU_CLAW:
                        frame.update(
                            {
                                "action.effector.position": get_nested_value(
                                    all_low_dim_data,
                                    "action.effector.position",
                                    i,
                                    [0.0] * 2,
                                ),
                                "observation.state.effector.position": get_nested_value(
                                    all_low_dim_data,
                                    "state.effector.position",
                                    i,
                                    [0.0] * 2,
                                ),
                            }
                        )
                    if USE_QIANGNAO:
                        frame.update(
                            {
                                "action.hand_left.position": get_nested_value(
                                    all_low_dim_data,
                                    "action.hand_left.position",
                                    i,
                                    [0.0] * 6,
                                ),
                                "action.hand_right.position": get_nested_value(
                                    all_low_dim_data,
                                    "action.hand_right.position",
                                    i,
                                    [0.0] * 6,
                                ),
                                "observation.state.hand_left.position": get_nested_value(
                                    all_low_dim_data,
                                    "state.hand_left.position",
                                    i,
                                    [0.0] * 6,
                                ),
                                "observation.state.hand_right.position": get_nested_value(
                                    all_low_dim_data,
                                    "state.hand_right.position",
                                    i,
                                    [0.0] * 6,
                                ),
                                "observation.state.hand_left.force_torque": get_nested_value(
                                    all_low_dim_data,
                                    "state.hand_left.force_torque",
                                    i,
                                    [0.0] * 6,
                                ),
                                "observation.state.hand_right.force_torque": get_nested_value(
                                    all_low_dim_data,
                                    "state.hand_right.force_torque",
                                    i,
                                    [0.0] * 6,
                                ),
                                "observation.state.hand_left.touch_matrix": get_nested_value(
                                    all_low_dim_data,
                                    "state.hand_left.touch_matrix",
                                    i,
                                    [0.0] * 360,
                                ),
                                "observation.state.hand_right.touch_matrix": get_nested_value(
                                    all_low_dim_data,
                                    "state.hand_right.touch_matrix",
                                    i,
                                    [0.0] * 360,
                                ),
                            }
                        )

                    # 外参（若可用）
                    for cam_key, extrs in extrinsics_dict.items():
                        if extrs and len(extrs) > i:
                            rot = np.array(
                                extrs[i]["rotation_matrix"], dtype=np.float32
                            ).reshape(-1)
                            trans = np.array(
                                extrs[i]["translation_vector"], dtype=np.float32
                            ).reshape(-1)
                            frame[
                                f"observation.camera_params.rotation_matrix_flat.{cam_key}"
                            ] = rot
                            frame[
                                f"observation.camera_params.translation_vector.{cam_key}"
                            ] = trans

                    # 彩色图（如果视频单独存储，跳过图像处理）
                    separate_video_storage = getattr(
                        raw_config, "separate_video_storage", False
                    )

                    if not separate_video_storage:
                        # 原有逻辑：图像编码到dataset中
                        for camera, img_list in imgs_per_cam.items():
                            if i < len(img_list):
                                img_bytes = img_list[i]
                                img_np = cv2.imdecode(
                                    np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                                )
                                if img_np is None:
                                    raise ValueError(
                                        f"Failed to decode color image for camera {camera} at frame {i}"
                                    )
                                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                                img_np = cv2.resize(
                                    img_np,
                                    (raw_config.resize.width, raw_config.resize.height),
                                )
                                frame[f"observation.images.{camera}"] = img_np

                    dataset.add_frame(frame, task=task)

                    if (
                        i % 800 == 0
                        and hasattr(dataset, "_wait_image_writer")
                        and dataset._wait_image_writer
                    ):
                        if dataset.image_writer.queue.qsize() > 500:
                            dataset._wait_image_writer()
                            gc.collect()

                # 保存一批（低维数据）
                _t_frame_loop_end = time.time()
                _t_save_episode_start = time.time()
                dataset.save_episode()
                _t_save_episode_end = time.time()

                # 根据配置选择视频处理方式
                separate_video_storage = getattr(
                    raw_config, "separate_video_storage", False
                )

                _t_save_images_start = time.time()
                if separate_video_storage:
                    temp_video_dir = os.path.join(
                        "/tmp", "kuavo_video_temp", episode_uuid
                    )

                    # 流式编码模式：彩色帧直接喂入编码器，深度帧保存到临时目录
                    if streaming_encoder is not None:
                        # 喂入彩色帧到流式编码器
                        cam_stats = streaming_encoder.feed_batch(imgs_per_cam, batch_id)

                        # 深度帧仍需保存到临时目录（ffmpeg 编码）
                        if imgs_per_cam_depth:
                            save_image_bytes_to_temp(
                                {}, imgs_per_cam_depth, temp_video_dir, batch_id
                            )
                    else:
                        # 原有逻辑：保存图像字节流到独立临时目录
                        cam_stats = save_image_bytes_to_temp(
                            imgs_per_cam, imgs_per_cam_depth, temp_video_dir, batch_id
                        )

                        # 如果启用流水线编码，立即提交编码任务
                        if pipeline_encoder is not None:
                            pipeline_encoder.submit_batch(batch_id)

                    _t_save_images_end = time.time()
                    # 立即释放图像数据内存
                    del imgs_per_cam, imgs_per_cam_depth
                    gc.collect()
                    print(f"[MEMORY] 批次{batch_id} 图像数据已释放")

                else:
                    _t_save_images_end = None  # 非 separate_video_storage 模式不计时
                    # 原有逻辑：深度视频编码到batch目录
                    depth_dir = os.path.join(batch_root, "depth")
                    os.makedirs(depth_dir, exist_ok=True)
                    compressed_group = {
                        cam: imgs_per_cam_depth[cam]
                        for cam in cameras
                        if compressed.get(cam, None) is True
                    }
                    uncompressed_group = {
                        cam: imgs_per_cam_depth[cam]
                        for cam in cameras
                        if compressed.get(cam, None) is False
                    }

                    if compressed_group:
                        if raw_config.enhance_enabled:
                            save_depth_videos_enhanced_parallel(
                                compressed_group,
                                imgs_per_cam,
                                output_dir=depth_dir,
                                raw_config=raw_config,
                            )
                    if uncompressed_group:
                        save_depth_videos_16U_parallel(
                            uncompressed_group,
                            output_dir=depth_dir,
                            raw_config=raw_config,
                        )
                    move_and_rename_depth_videos(depth_dir, episode_idx=0)

                # 保存参数（camera info 与 extrinsics）
                if batch_id == 1:
                    parameters_dir = os.path.join(batch_root, "parameters")
                    os.makedirs(parameters_dir, exist_ok=True)
                    save_camera_info_to_json_new(
                        info_per_cam, distortion_model, output_dir=parameters_dir
                    )
                    save_camera_extrinsic_params(
                        cameras=cameras, output_dir=parameters_dir
                    )

                # 保存 metadata.json（按批次）
                try:
                    if metadata_json_DIR is not None and os.path.exists(metadata_json_DIR):
                        # 检查是否为新格式（包含 marks 字段）
                        with open(metadata_json_DIR, "r", encoding="utf-8") as f:
                            test_metadata = json.load(f)
                        is_new_format = "marks" in test_metadata and isinstance(test_metadata.get("marks"), list)
                        
                        # 新格式不需要 moment_json_DIR，旧格式需要
                        if is_new_format:
                            # 新格式：只需要 metadata.json
                            merge_metadata_and_moment(
                                metadata_json_DIR,
                                None,  # moment_path 在新格式下为 None
                                os.path.join(batch_root, "metadata.json"),
                                episode_uuid,
                                raw_config,
                                bag_time_info=bag_time_info,
                                main_time_line_timestamps=main_ts,  # 秒
                            )
                        elif moment_json_DIR is not None and os.path.exists(moment_json_DIR):
                            # 旧格式：需要 metadata.json + moments.json
                            merge_metadata_and_moment(
                                metadata_json_DIR,
                                moment_json_DIR,
                                os.path.join(batch_root, "metadata.json"),
                                episode_uuid,
                                raw_config,
                                bag_time_info=bag_time_info,
                                main_time_line_timestamps=main_ts,  # 秒
                            )
                        else:
                            print(
                                f"[WARN] 旧格式需要 moments.json，但未找到: moment_json_DIR={moment_json_DIR}"
                            )
                    else:
                        print(
                            f"[WARN] 未生成批次 metadata.json，metadata_json_DIR={metadata_json_DIR}"
                        )
                except Exception as e:
                    print(f"[ERROR] 合并 metadata 和 moment 失败: {e}")
                    import traceback
                    traceback.print_exc()

                # 释放批次内存
                del dataset, info_per_cam, distortion_model

                # 如果没有在前面删除，这里删除图像数据
                if not separate_video_storage:
                    if "imgs_per_cam" in locals():
                        del imgs_per_cam
                    if "imgs_per_cam_depth" in locals():
                        del imgs_per_cam_depth

                del (
                    state,
                    action,
                    action_kuavo_arm_traj,
                    sensors_data_raw__joint_v,
                    state_joint_current_arr,
                    sensors_data_raw__imu_data,
                )
                del claw_state, claw_action, qiangnao_state, qiangnao_action
                del (
                    end_position,
                    end_orientation,
                    all_low_dim_data,
                )
                if batch_id == 1:
                    del head_extrinsics, left_extrinsics, right_extrinsics
                gc.collect()

                # ===== 计时汇总 =====
                _t_batch_end = time.time()
                _t_total = _t_batch_end - _t_batch_start
                _t_extract = _t_extract_end - _t_extract_start
                _t_create = _t_create_dataset_end - _t_create_dataset_start
                _t_frames = _t_frame_loop_end - _t_frame_loop_start
                _t_save_ep = _t_save_episode_end - _t_save_episode_start
                _t_save_img = (
                    (_t_save_images_end - _t_save_images_start)
                    if _t_save_images_end
                    else 0
                )
                print(
                    f"[TIMING] Batch {batch_id}: ROSbag读取={_t_rosbag_read:.2f}s | "
                    f"数据提取={_t_extract:.2f}s | Dataset创建={_t_create:.2f}s | "
                    f"帧循环={_t_frames:.2f}s | Parquet保存={_t_save_ep:.2f}s | "
                    f"图像保存={_t_save_img:.2f}s | 批次总计={_t_total:.2f}s"
                )
                _t_prev_batch_end = time.time()  # 更新为下一批准备

        if separate_video_storage:
            return cam_stats
        else:
            return None

    # 如果启用流水线编码，创建编码器
    # 环境变量优先于配置文件
    pipeline_encoder = None
    env_pipeline = os.environ.get("USE_PIPELINE_ENCODING", "").lower()
    if env_pipeline in ("true", "1", "yes"):
        use_pipeline_encoding = True
        print("[CONFIG] 流水线编码已通过环境变量启用 (USE_PIPELINE_ENCODING=true)")
    elif env_pipeline in ("false", "0", "no"):
        use_pipeline_encoding = False
    else:
        use_pipeline_encoding = getattr(raw_config, "use_pipeline_encoding", False)

    if use_pipeline_encoding and getattr(raw_config, "separate_video_storage", False):
        temp_video_dir = os.path.join("/tmp", "kuavo_video_temp", episode_uuid)
        segment_dir = os.path.join("/tmp", "kuavo_video_segments", episode_uuid)
        video_output_dir = os.path.join(base_root, episode_uuid, episode_uuid)

        pipeline_encoder = BatchSegmentEncoder(
            temp_base_dir=temp_video_dir,
            segment_base_dir=segment_dir,
            video_output_dir=video_output_dir,
            cameras=raw_config.default_camera_names,
            train_hz=raw_config.train_hz,
            uuid_str=episode_uuid,
            chunk_size=800,  # 固定批次大小
            max_workers=3,  # 3个相机，3个工作线程
        )

    # 如果启用流式编码，创建编码器（优先级高于 pipeline_encoder）
    streaming_encoder = None
    env_streaming = os.environ.get("USE_STREAMING_VIDEO", "").lower()
    if env_streaming in ("true", "1", "yes"):
        use_streaming_video = True
        print("[CONFIG] 流式视频编码已通过环境变量启用 (USE_STREAMING_VIDEO=true)")
    elif env_streaming in ("false", "0", "no"):
        use_streaming_video = False
    else:
        use_streaming_video = getattr(raw_config, "use_streaming_video", False)

    if use_streaming_video and getattr(raw_config, "separate_video_storage", False):
        # 流式编码与流水线编码互斥，流式编码优先
        if pipeline_encoder is not None:
            print("[CONFIG] 流式编码与流水线编码互斥，优先使用流式编码")
            pipeline_encoder = None

        video_output_dir = os.path.join(base_root, episode_uuid, episode_uuid)
        queue_limit = int(
            os.environ.get(
                "VIDEO_QUEUE_LIMIT", getattr(raw_config, "video_queue_limit", 100)
            )
        )

        streaming_encoder = StreamingVideoEncoderManager(
            cameras=raw_config.default_camera_names,
            video_output_dir=video_output_dir,
            uuid_str=episode_uuid,
            train_hz=raw_config.train_hz,
            queue_limit=queue_limit,
        )

    # 执行流式填充（快速生成lerobot数据）
    cam_stats = populate_dataset_stream(
        raw_config=raw_config,
        bag_files=processed_files,
        task=task,
        moment_json_dir=moment_json_DIR,
        base_root=base_root,
        metadata_json_dir=metadata_json_DIR,
        pipeline_encoder=pipeline_encoder,
        streaming_encoder=streaming_encoder,
    )

    print("[INFO] ========== 主数据处理完成 ==========")
    print(f"[INFO] LeRobot数据已保存到: {base_root}")

    # ===== 优化: 提前启动视频编码，与合并并行 =====
    base_path = Path(base_root).resolve()
    output_dir = base_path / episode_uuid / episode_uuid
    encoding_thread = None

    if getattr(raw_config, "separate_video_storage", False):
        temp_video_dir = os.path.join("/tmp", "kuavo_video_temp", episode_uuid)
        video_output_dir = output_dir
        async_encoding = getattr(raw_config, "async_video_encoding", False)

        # 流式/流水线编码器特殊处理（它们需要在合并后finalize）
        if streaming_encoder is None and pipeline_encoder is None and async_encoding:
            # 原有异步编码: 提前启动，与合并并行
            import threading

            print("[VIDEO] ========== 提前启动视频编码（与合并并行）==========")

            def async_encode():
                try:
                    encode_complete_videos_from_temp(
                        temp_video_dir,
                        video_output_dir,
                        episode_uuid,
                        raw_config,
                        use_depth=use_depth,
                    )
                except Exception as e:
                    print(f"[VIDEO] 异步编码出错: {e}")
                    import traceback

                    traceback.print_exc()

            encoding_thread = threading.Thread(target=async_encode, daemon=False)
            encoding_thread.start()
            print("[VIDEO] 视频编码已在后台启动")
            print(f"[VIDEO] 视频将保存到: {video_output_dir}")

    # ===== 合并批次数据（与视频编码并行）=====
    _t_merge_start = time.time()
    print("[INFO] 开始合并批次数据...")
    batch_dirs = get_batch_dirs(base_path)
    total_frames = merge_parquet_files(batch_dirs, output_dir)

    # 先合并生成全局 metadata.json（使用各 batch 的 metadata.json）
    try:
        merge_metadata(batch_dirs, output_dir, total_frames)
    except Exception as e:
        print(f"[WARN] 合并 metadata.json 失败: {e}")

    # 再合并 episodes.jsonl / info.json / tasks.jsonl / episodes_stats.jsonl 等 meta 文件
    # 传入真实保存的视频高宽，用于 info.json 中相机 shape
    video_h = None
    video_w = None
    if getattr(raw_config, "resize", None) is not None:
        video_h = getattr(raw_config.resize, "height", 480)
        video_w = getattr(raw_config.resize, "width", 848)
    merge_meta_files(
        batch_dirs, output_dir, total_frames, cam_stats,
        video_height=video_h, video_width=video_w,
    )
    _t_merge_end = time.time()
    print(f"[INFO] 批次数据合并完成。耗时: {_t_merge_end - _t_merge_start:.2f}s")

    # 合并后删除所有 batch 文件夹
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith("batch_"):
            try:
                shutil.rmtree(d)
                print(f"[INFO] 已删除批次文件夹: {d}")
            except Exception as e:
                print(f"[WARN] 删除批次文件夹失败: {d}, 错误: {e}")

    # ===== 视频编码后续处理 =====
    if getattr(raw_config, "separate_video_storage", False):
        temp_video_dir = os.path.join("/tmp", "kuavo_video_temp", episode_uuid)
        video_output_dir = output_dir

        if streaming_encoder is not None:
            # 流式编码模式：彩色视频已在批处理中编码完成，只需 finalize
            print("[VIDEO] ========== 流式编码模式 ==========")
            streaming_encoder.finalize()
            print(f"[VIDEO] 彩色视频已保存到: {video_output_dir}")

            # 深度视频单独处理（仍然使用 ffmpeg）
            if use_depth:
                depth_temp_dir = os.path.join(temp_video_dir, "depth")
                if os.path.exists(depth_temp_dir):
                    print("[VIDEO] 开始编码深度视频...")
                    depth_out_dir = os.path.join(video_output_dir, "depth", "chunk-000")
                    os.makedirs(depth_out_dir, exist_ok=True)
                    apply_denoise = False  # 保持原逻辑
                    depth_procs = []
                    for camera in os.listdir(depth_temp_dir):
                        camera_dir = os.path.join(depth_temp_dir, camera)
                        if not os.path.isdir(camera_dir):
                            continue
                        video_path = os.path.join(depth_out_dir, f"{camera}.mkv")
                        p = multiprocessing.Process(
                            target=_encode_depth_camera_worker,
                            args=(
                                camera_dir,
                                camera,
                                video_path,
                                raw_config.train_hz,
                                apply_denoise,
                            ),
                            daemon=False,
                        )
                        p.start()
                        depth_procs.append(p)
                    for p in depth_procs:
                        p.join()
                    print("[VIDEO] 深度视频编码完成")
                    # 清理深度临时目录
                    shutil.rmtree(depth_temp_dir, ignore_errors=True)

        elif pipeline_encoder is not None:
            # 流水线模式：等待编码完成并拼接
            print("[VIDEO] ========== 流水线编码模式 ==========")
            pipeline_encoder.finalize(use_depth=use_depth)
            print(f"[VIDEO] 所有视频已保存到: {video_output_dir}")

        elif encoding_thread is not None:
            # 异步编码已提前启动，只需输出状态
            print("[INFO] 主流程已完成，视频编码在后台继续...")

        else:
            # 同步编码（等待完成）
            async_encoding = getattr(raw_config, "async_video_encoding", False)
            if not async_encoding:
                print("[VIDEO] 开始同步编码视频...")
                encode_complete_videos_from_temp(
                    temp_video_dir,
                    video_output_dir,
                    episode_uuid,
                    raw_config,
                    use_depth=use_depth,
                )
                print(f"[VIDEO] 所有视频已保存到: {video_output_dir}")


if __name__ == "__main__":
    import argparse
    import json
    import time

    start = time.time()
    parser = argparse.ArgumentParser(description="Kuavo ROSbag to Lerobot Converter")
    parser.add_argument(
        "--bag_dir",
        default="/home/leju_kuavo/tmp/123/",
        type=str,
        required=False,
        help="Path to ROS bag",
    )
    # parser.add_argument("--bag_dir", default = "./testbag/task24_519_20250519_193043_0.bag", type=str, required=False, help="Path to ROS bag")
    parser.add_argument(
        "--moment_json_dir", type=str, required=False, help="Path to moment.json"
    )
    parser.add_argument(
        "--metadata_json_dir", type=str, required=False, help="Path to metadata.json"
    )
    parser.add_argument(
        "--output_dir",
        default="testoutput/",
        type=str,
        required=False,
        help="Path to output",
    )
    parser.add_argument(
        "--train_frequency",
        type=int,
        help="Training frequency (Hz), overrides config file setting",
    )

    parser.add_argument(
        "--only_arm",
        type=str,
        choices=["true", "false"],
        help="Use only arm data (true/false), overrides config file setting",
    )

    parser.add_argument(
        "--which_arm",
        type=str,
        choices=["left", "right", "both"],
        help="Which arm to use (left/right/both), overrides config file setting",
    )

    parser.add_argument(
        "--dex_dof_needed",
        type=int,
        help="Degrees of freedom needed for dexterous hand, overrides config file setting",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./kuavo/request.json",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--use_depth",
        action="store_true",
        help="如果指定，忽略所有与 metadata.json / moments.json 相关的输入与输出（不读取也不写入）",
    )
    args = parser.parse_args()

    # 加载配置文件
    config = load_config_from_json(args.config)
    # 用命令行参数覆盖配置文件中的设置
    if args.train_frequency is not None:
        config.train_hz = args.train_frequency
        print(f"✅ 覆盖配置: train_hz = {args.train_frequency}")

    if args.only_arm is not None:
        config.only_arm = args.only_arm.lower() == "true"
        print(f"✅ 覆盖配置: only_arm = {config.only_arm}")

    if args.which_arm is not None:
        config.which_arm = args.which_arm
        print(f"✅ 覆盖配置: which_arm = {args.which_arm}")

    if args.dex_dof_needed is not None:
        config.dex_dof_needed = args.dex_dof_needed
        print(f"✅ 覆盖配置: dex_dof_needed = {args.dex_dof_needed}")
    # 从配置获取参数

    if args.bag_dir is not None:
        bag_DIR = args.bag_dir
    print(f"Bag directory: {bag_DIR}")
    moment_json_DIR = None
    metadata_json_DIR = None
    if args.moment_json_dir is not None:
        moment_json_DIR = args.moment_json_dir
    else:
        moment_json_DIR = os.path.join(bag_DIR, "moments.json")

    if args.metadata_json_dir is not None:
        metadata_json_DIR = args.metadata_json_dir
    else:
        metadata_json_DIR = os.path.join(bag_DIR, "metadata.json")
    if args.output_dir is not None:
        output_DIR = args.output_dir

    ID = config.id
    use_depth = args.use_depth
    bag_files = list_bag_files_auto(bag_DIR)
    port_kuavo_rosbag(
        raw_config=config,
        processed_files=bag_files,
        moment_json_DIR=moment_json_DIR,
        metadata_json_DIR=metadata_json_DIR,
        lerobot_dir=output_DIR,
        use_depth=use_depth,
    )
    end = time.time()
    print(f"[INFO] 总用时: {end - start:.2f} 秒")
