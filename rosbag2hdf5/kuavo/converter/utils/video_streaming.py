import os
import shutil
import subprocess
from queue import Queue
from threading import Thread

import cv2
import numpy as np

from converter.image.video_denoising import repair_depth_noise_focused
from converter.utils.camera_utils import get_mapped_filename


class StreamingVideoWriter:
    """
    流式视频写入器：通过 bounded queue 接收帧，worker 线程将帧写入临时目录，
    finish() 时调用 ffmpeg 编码。

    用法:
        writer = StreamingVideoWriter("head_cam_h", output_dir, raw_config, is_depth=False)
        for frame_bytes in frames:
            writer.put(frame_bytes)
        writer.finish()  # 等待编码完成
    """

    SENTINEL = object()  # 结束信号

    def __init__(
        self,
        cam_name: str,
        output_dir: str,
        raw_config,
        is_depth: bool = False,
        is_16bit_depth: bool = False,
        queue_limit: int = 300,
    ):
        self.cam_name = cam_name
        self.output_dir = output_dir
        self.raw_config = raw_config
        self.is_depth = is_depth
        self.is_16bit_depth = is_16bit_depth
        self.queue_limit = queue_limit

        self._queue = Queue(maxsize=queue_limit)
        self._frame_count = 0
        self._written_count = 0
        self._block_count = 0  # put 阻塞次数
        self._width = None
        self._height = None
        self._finished = False
        self._error = None

        # 临时目录
        self._temp_dir = os.path.join(output_dir, f"streaming_frames_{cam_name}")
        os.makedirs(self._temp_dir, exist_ok=True)

        # 启动 worker 线程
        self._worker = Thread(target=self._write_worker, daemon=True)
        self._worker.start()

    def put(self, frame_bytes):
        """将帧放入队列（blocking if full）"""
        if self._finished:
            raise RuntimeError(f"StreamingVideoWriter({self.cam_name}) already finished")

        if self._queue.full():
            self._block_count += 1
            if self._block_count % 100 == 1:
                log_print(
                    f"[StreamingVideoWriter] {self.cam_name} 队列已满，开始阻塞 "
                    f"(block_count={self._block_count}, frame_count={self._frame_count})"
                )
            if not self._worker.is_alive():
                err = self._error
                if err is None:
                    raise RuntimeError(
                        f"StreamingVideoWriter({self.cam_name}) worker 已退出，未知原因"
                    )
                raise RuntimeError(
                    f"StreamingVideoWriter({self.cam_name}) worker 异常退出: {err}"
                )

        self._queue.put(frame_bytes)  # blocking
        self._frame_count += 1

    def finish(self) -> dict:
        """
        发送结束信号，等待 worker 完成，调用 ffmpeg 编码。
        返回统计信息。
        """
        if self._finished:
            return self._get_stats()

        # 发送结束信号
        self._queue.put(self.SENTINEL)
        self._worker.join()
        self._finished = True

        if self._error:
            raise self._error

        # 调用 ffmpeg 编码
        if self._written_count > 0:
            self._encode_video()

        # 清理临时目录
        shutil.rmtree(self._temp_dir, ignore_errors=True)

        return self._get_stats()

    def _get_stats(self) -> dict:
        return {
            "cam_name": self.cam_name,
            "frame_count": self._frame_count,
            "written_count": self._written_count,
            "block_count": self._block_count,
            "is_depth": self.is_depth,
            "is_16bit_depth": self.is_16bit_depth,
        }

    def _write_worker(self):
        """Worker 线程：从队列取帧，解码并写入临时文件"""
        png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        is_hand_camera = "wrist_cam_l" in self.cam_name or "wrist_cam_r" in self.cam_name

        try:
            while True:
                item = self._queue.get()
                if item is self.SENTINEL:
                    break

                frame_bytes = item
                idx = self._written_count

                if self.is_depth:
                    # 深度图处理
                    if self.is_16bit_depth:
                        img = self._decode_depth_16bit(frame_bytes, png_magic, idx, is_hand_camera)
                    else:
                        img = self._decode_depth_8bit(frame_bytes, idx)
                else:
                    # 彩色图处理
                    img = self._decode_color(frame_bytes, idx)

                if img is None:
                    continue

                # 写入临时文件
                img_path = os.path.join(self._temp_dir, f"frame_{idx:05d}.png")
                if self.is_16bit_depth:
                    cv2.imwrite(img_path, img.astype(np.uint16))
                else:
                    cv2.imwrite(img_path, img)

                self._written_count += 1
                if self._written_count % 300 == 0:
                    log_print(
                        f"[StreamingVideoWriter] {self.cam_name} 已写入 {self._written_count} 帧 "
                        f"(frame_count={self._frame_count})"
                    )

        except Exception as e:
            self._error = e

    def _decode_color(self, frame_bytes, idx):
        """解码彩色图像"""
        img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log_print(f"[{self.cam_name}] 第{idx}帧彩色图解码失败，跳过")
            return None
        if self.raw_config is not None and hasattr(self.raw_config, "resize"):
            width, height = self.raw_config.resize.width, self.raw_config.resize.height
            img = cv2.resize(img, (width, height))
            if self._width is None:
                self._width, self._height = width, height
        else:
            if self._width is None:
                self._height, self._width = img.shape[:2]
        return img

    def _decode_depth_8bit(self, frame_bytes, idx):
        """解码8位深度图像"""
        img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            log_print(f"[{self.cam_name}] 第{idx}帧深度图解码失败，跳过")
            return None
        if img.ndim > 2:
            img = img[:, :, 0]
        if self.raw_config is not None and hasattr(self.raw_config, "resize"):
            width, height = self.raw_config.resize.width, self.raw_config.resize.height
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
            if self._width is None:
                self._width, self._height = width, height
        else:
            if self._width is None:
                self._height, self._width = img.shape
        return img.astype(np.uint8)

    def _decode_depth_16bit(self, frame_bytes, png_magic, idx, is_hand_camera):
        """解码16位深度图像"""
        if isinstance(frame_bytes, bytes):
            idx_png = frame_bytes.find(png_magic)
            if idx_png == -1:
                log_print(f"[{self.cam_name}] 第{idx}帧未找到PNG头，跳过")
                return None
            png_data = frame_bytes[idx_png:]
        else:
            log_print(f"[{self.cam_name}] 第{idx}帧数据类型异常，跳过")
            return None

        img = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            log_print(f"[{self.cam_name}] 第{idx}帧解码失败，跳过")
            return None
        if img.ndim > 2:
            img = img[:, :, 0]
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)

        if self.raw_config is not None and hasattr(self.raw_config, "resize"):
            width, height = self.raw_config.resize.width, self.raw_config.resize.height
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
            if self._width is None:
                self._width, self._height = width, height
        else:
            if self._width is None:
                self._height, self._width = img.shape

        # 手部相机去噪
        if is_hand_camera:
            try:
                img = repair_depth_noise_focused(
                    img,
                    max_valid_depth=10000,
                    median_kernel=5,
                    detect_white_spots=True,
                    spot_size_range=(10, 2000),
                )
            except Exception as e:
                log_print(f"[DENOISE] {self.cam_name} 第{idx}帧去噪失败: {e}")

        return img

    def _encode_video(self):
        """调用 ffmpeg 编码视频"""
        if self.is_depth:
            if self.is_16bit_depth:
                self._encode_depth_16bit()
            else:
                self._encode_depth_8bit()
        else:
            self._encode_color()

    def _encode_color(self):
        """编码彩色视频 (h264)"""
        output_filename = get_mapped_filename(self.cam_name, ".mp4")
        video_path = os.path.join(self.output_dir, output_filename)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", os.path.join(self._temp_dir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            video_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            log_print(f"[StreamingVideoWriter] 已保存彩色视频: {video_path} ({self._written_count} 帧)")
        except subprocess.CalledProcessError as e:
            log_print(f"[StreamingVideoWriter] ffmpeg编码失败 {self.cam_name}: {e.stderr.decode() if e.stderr else e}")

    def _encode_depth_8bit(self):
        """编码8位深度视频 (ffv1 gray)"""
        output_filename = get_mapped_filename(self.cam_name, ".mkv")
        video_path = os.path.join(self.output_dir, output_filename)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", os.path.join(self._temp_dir, "frame_%05d.png"),
            "-c:v", "ffv1",
            "-pix_fmt", "gray",
            "-level", "3",
            "-g", "1",
            "-slicecrc", "1",
            "-slices", "16",
            "-an",
            video_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            log_print(f"[StreamingVideoWriter] 已保存深度视频: {video_path} ({self._written_count} 帧)")
        except subprocess.CalledProcessError as e:
            log_print(f"[StreamingVideoWriter] ffmpeg编码失败 {self.cam_name}: {e.stderr.decode() if e.stderr else e}")

    def _encode_depth_16bit(self):
        """编码16位深度视频 (ffv1 gray16le)"""
        output_filename = get_mapped_filename(self.cam_name, ".mkv")
        video_path = os.path.join(self.output_dir, output_filename)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", os.path.join(self._temp_dir, "frame_%05d.png"),
            "-c:v", "ffv1",
            "-pix_fmt", "gray16le",
            video_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            log_print(f"[StreamingVideoWriter] 已保存16位深度视频: {video_path} ({self._written_count} 帧)")
        except subprocess.CalledProcessError as e:
            log_print(f"[StreamingVideoWriter] ffmpeg编码失败 {self.cam_name}: {e.stderr.decode() if e.stderr else e}")


def save_color_videos_streaming(
    imgs_per_cam_color: dict,
    output_dir: str = "./",
    raw_config = None,
    queue_limit: int = 300,
) -> dict:
    """
    使用流式写入器保存彩色视频。

    相比 save_color_videos_ffmpeg_parallel，内存占用更低（bounded queue），
    且帧处理与写入并行。

    Args:
        imgs_per_cam_color: {cam_name: [img_bytes, ...]}
        output_dir: 输出目录
        raw_config: 配置对象
        queue_limit: 队列大小限制

    Returns:
        统计信息 dict: {cam_name: stats_dict}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建所有 writers
    writers = {}
    for cam_name in imgs_per_cam_color.keys():
        writers[cam_name] = StreamingVideoWriter(
            cam_name=cam_name,
            output_dir=output_dir,
            raw_config=raw_config,
            is_depth=False,
            is_16bit_depth=False,
            queue_limit=queue_limit,
        )

    # 2. 并行喂帧（每个 writer 的 worker 线程会异步写入）
    for cam_name, imgs in imgs_per_cam_color.items():
        writer = writers[cam_name]
        for frame_bytes in imgs:
            writer.put(frame_bytes)

    # 3. 完成所有 writers
    all_stats = {}
    for cam_name, writer in writers.items():
        stats = writer.finish()
        all_stats[cam_name] = stats
        if stats["block_count"] > 0:
            log_print(f"[StreamingVideoWriter] {cam_name} 队列阻塞 {stats['block_count']} 次")

    return all_stats


def save_depth_videos_16U_streaming(
    imgs_per_cam_depth: dict,
    output_dir: str = "./",
    raw_config = None,
    queue_limit: int = 300,
) -> dict:
    """
    使用流式写入器保存16位深度视频。

    相比 save_depth_videos_16U_parallel，内存占用更低（bounded queue），
    且帧处理与写入并行。

    Args:
        imgs_per_cam_depth: {cam_name_depth: [img_bytes, ...]}
        output_dir: 输出目录
        raw_config: 配置对象
        queue_limit: 队列大小限制

    Returns:
        统计信息 dict: {cam_name: stats_dict}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建所有 writers
    writers = {}
    for cam_name in imgs_per_cam_depth.keys():
        writers[cam_name] = StreamingVideoWriter(
            cam_name=cam_name,
            output_dir=output_dir,
            raw_config=raw_config,
            is_depth=True,
            is_16bit_depth=True,
            queue_limit=queue_limit,
        )

    # 2. 并行喂帧
    for cam_name, imgs in imgs_per_cam_depth.items():
        writer = writers[cam_name]
        for frame_bytes in imgs:
            writer.put(frame_bytes)

    # 3. 完成所有 writers
    all_stats = {}
    for cam_name, writer in writers.items():
        stats = writer.finish()
        all_stats[cam_name] = stats
        if stats["block_count"] > 0:
            log_print(f"[StreamingVideoWriter] {cam_name} 队列阻塞 {stats['block_count']} 次")

    return all_stats
