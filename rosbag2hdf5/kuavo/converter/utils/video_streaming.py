import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import Semaphore, Thread

import cv2
import numpy as np

from converter.image.video_denoising import repair_depth_noise_focused
from converter.utils.camera_utils import get_mapped_filename


@dataclass(frozen=True)
class StageSchedule:
    cores: int
    decode_feed_workers: int
    max_encode_processes: int
    queue_limit: int


_SCHEDULES = {
    1: StageSchedule(cores=1, decode_feed_workers=1, max_encode_processes=1, queue_limit=96),
    2: StageSchedule(cores=2, decode_feed_workers=2, max_encode_processes=1, queue_limit=160),
    4: StageSchedule(cores=4, decode_feed_workers=3, max_encode_processes=2, queue_limit=260),
    8: StageSchedule(cores=8, decode_feed_workers=4, max_encode_processes=3, queue_limit=400),
}


def _resolve_schedule(raw_config, queue_limit_override: int | None = None) -> StageSchedule:
    """
    调度策略（颜色/深度路径）:
    - 1核: decode_feed=1, encode=1
    - 2核: decode_feed=2, encode=1
    - 4核: decode_feed=3, encode=2
    - 8核: decode_feed=4, encode=3
    """
    cfg_cores = int(getattr(raw_config, "schedule_cores", 0) or 0) if raw_config is not None else 0
    env_cores = int(os.getenv("KUAVO_SCHED_CORES", "0") or 0)
    req = env_cores if env_cores in _SCHEDULES else cfg_cores
    if req not in _SCHEDULES:
        host = os.cpu_count() or 1
        req = 8 if host >= 8 else (4 if host >= 4 else (2 if host >= 2 else 1))
    base = _SCHEDULES[req]
    if queue_limit_override is None:
        return base
    return StageSchedule(
        cores=base.cores,
        decode_feed_workers=base.decode_feed_workers,
        max_encode_processes=base.max_encode_processes,
        queue_limit=queue_limit_override,
    )


class StreamingVideoWriter:
    """
    流式视频写入器：通过 bounded queue 接收帧，worker 线程解码后直接通过
    ffmpeg stdin pipe 编码（不落盘 PNG 临时帧）。

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
        encoder_semaphore: Semaphore | None = None,
    ):
        self.cam_name = cam_name
        self.output_dir = output_dir
        self.raw_config = raw_config
        self.is_depth = is_depth
        self.is_16bit_depth = is_16bit_depth
        self.queue_limit = queue_limit
        self._encoder_semaphore = encoder_semaphore

        self._queue = Queue(maxsize=queue_limit)
        self._frame_count = 0
        self._written_count = 0
        self._block_count = 0  # put 阻塞次数
        self._width = None
        self._height = None
        self._finished = False
        self._error = None
        self._ffmpeg_proc = None
        self._encoder_sem_acquired = False

        # 统一采用 ffmpeg pipe 直写，不使用临时帧目录
        self._temp_dir = None

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

        self._finalize_encoder()

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

                self._start_encoder_if_needed(img.shape[1], img.shape[0])
                if self.is_depth:
                    if self.is_16bit_depth:
                        self._ffmpeg_proc.stdin.write(img.astype(np.uint16, copy=False).tobytes())
                    else:
                        self._ffmpeg_proc.stdin.write(img.astype(np.uint8, copy=False).tobytes())
                else:
                    self._ffmpeg_proc.stdin.write(img.tobytes())

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

        # 手部相机去噪（可开关）
        denoise_enabled = bool(
            getattr(self.raw_config, "enable_depth_denoise", True)
            if self.raw_config is not None
            else True
        )
        if is_hand_camera and denoise_enabled:
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

    def _start_encoder_if_needed(self, width: int, height: int):
        if self._ffmpeg_proc is not None:
            return
        if self._encoder_semaphore is not None:
            self._encoder_semaphore.acquire()
            self._encoder_sem_acquired = True

        output_ext = ".mkv" if self.is_depth else ".mp4"
        output_filename = get_mapped_filename(self.cam_name, output_ext)
        video_path = os.path.join(self.output_dir, output_filename)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le" if self.is_depth and self.is_16bit_depth else ("gray" if self.is_depth else "bgr24"),
            "-s",
            f"{width}x{height}",
            "-r",
            "30",
            "-i",
            "-",
        ]
        if self.is_depth:
            ffmpeg_cmd.extend(
                [
                    "-an",
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "gray16le" if self.is_16bit_depth else "gray",
                    video_path,
                ]
            )
        else:
            preset = getattr(self.raw_config, "color_video_preset", "fast")
            crf = str(getattr(self.raw_config, "color_video_crf", 18))
            ffmpeg_cmd.extend(
                [
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    crf,
                    video_path,
                ]
            )
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _finalize_encoder(self):
        if self._ffmpeg_proc is None:
            if self._encoder_sem_acquired and self._encoder_semaphore is not None:
                self._encoder_semaphore.release()
                self._encoder_sem_acquired = False
            return
        try:
            if self._ffmpeg_proc.stdin is not None:
                self._ffmpeg_proc.stdin.close()
            rc = self._ffmpeg_proc.wait()
            err = b""
            if self._ffmpeg_proc.stderr is not None:
                err = self._ffmpeg_proc.stderr.read()
            if rc != 0:
                err = err.decode("utf-8", errors="ignore")
                raise RuntimeError(f"ffmpeg编码失败 {self.cam_name}: {err}")
            output_ext = ".mkv" if self.is_depth else ".mp4"
            output_filename = get_mapped_filename(self.cam_name, output_ext)
            video_path = os.path.join(self.output_dir, output_filename)
            if self.is_depth:
                log_print(f"[StreamingVideoWriter] 已保存深度视频: {video_path} ({self._written_count} 帧)")
            else:
                log_print(f"[StreamingVideoWriter] 已保存彩色视频: {video_path} ({self._written_count} 帧)")
        finally:
            if self._encoder_sem_acquired and self._encoder_semaphore is not None:
                self._encoder_semaphore.release()
                self._encoder_sem_acquired = False


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
    schedule = _resolve_schedule(raw_config, queue_limit_override=queue_limit)
    encoder_semaphore = Semaphore(schedule.max_encode_processes)
    log_print(
        "[SCHEDULE] color streaming:"
        f" cores={schedule.cores}"
        f" decode_feed_workers={schedule.decode_feed_workers}"
        f" encode_processes={schedule.max_encode_processes}"
        f" queue_limit={schedule.queue_limit}"
    )

    # 1. 创建所有 writers
    writers = {}
    for cam_name in imgs_per_cam_color.keys():
        writers[cam_name] = StreamingVideoWriter(
            cam_name=cam_name,
            output_dir=output_dir,
            raw_config=raw_config,
            is_depth=False,
            is_16bit_depth=False,
            queue_limit=schedule.queue_limit,
            encoder_semaphore=encoder_semaphore,
        )

    # 2. 三路（多路）并行喂帧（避免按camera串行）
    def _feed_one_camera(cam_name: str, imgs):
        writer = writers[cam_name]
        for frame_bytes in imgs:
            writer.put(frame_bytes)

    with ThreadPoolExecutor(max_workers=schedule.decode_feed_workers) as ex:
        futures = [ex.submit(_feed_one_camera, cam_name, imgs) for cam_name, imgs in imgs_per_cam_color.items()]
        for f in futures:
            f.result()

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
    schedule = _resolve_schedule(raw_config, queue_limit_override=queue_limit)
    encoder_semaphore = Semaphore(schedule.max_encode_processes)
    log_print(
        "[SCHEDULE] depth streaming:"
        f" cores={schedule.cores}"
        f" decode_feed_workers={schedule.decode_feed_workers}"
        f" encode_processes={schedule.max_encode_processes}"
        f" queue_limit={schedule.queue_limit}"
    )

    # 1. 创建所有 writers
    writers = {}
    for cam_name in imgs_per_cam_depth.keys():
        writers[cam_name] = StreamingVideoWriter(
            cam_name=cam_name,
            output_dir=output_dir,
            raw_config=raw_config,
            is_depth=True,
            is_16bit_depth=True,
            queue_limit=schedule.queue_limit,
            encoder_semaphore=encoder_semaphore,
        )

    # 2. 并行喂帧
    def _feed_one_camera(cam_name: str, imgs):
        writer = writers[cam_name]
        for frame_bytes in imgs:
            writer.put(frame_bytes)

    with ThreadPoolExecutor(max_workers=schedule.decode_feed_workers) as ex:
        futures = [ex.submit(_feed_one_camera, cam_name, imgs) for cam_name, imgs in imgs_per_cam_depth.items()]
        for f in futures:
            f.result()

    # 3. 完成所有 writers
    all_stats = {}
    for cam_name, writer in writers.items():
        stats = writer.finish()
        all_stats[cam_name] = stats
        if stats["block_count"] > 0:
            log_print(f"[StreamingVideoWriter] {cam_name} 队列阻塞 {stats['block_count']} 次")

    return all_stats
