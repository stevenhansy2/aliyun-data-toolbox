"""Video encoding and temporary frame staging utilities for Kuavo -> LeRobot conversion."""

import os
import queue
import shutil
import threading
import time

import cv2
import einops
import numpy as np
from converter.configs import Config
from converter.media.video_workers import (
    _encode_color_camera_worker as _encode_color_camera_worker_impl,
    _encode_depth_camera_worker as _encode_depth_camera_worker_impl,
    save_image_bytes_to_temp as save_image_bytes_to_temp_impl,
)
from converter.media.video_segments import (
    _concat_segments_ffmpeg as _concat_segments_ffmpeg_impl,
    _encode_batch_segment_color as _encode_batch_segment_color_impl,
)
from converter.media.video_finalize import (
    encode_complete_videos_from_temp as encode_complete_videos_from_temp_impl,
)
from lerobot.datasets.compute_stats import get_feature_stats


def save_image_bytes_to_temp(
    imgs_per_cam: dict, imgs_per_cam_depth: dict, temp_base_dir: str, batch_id: int
):
    return save_image_bytes_to_temp_impl(
        imgs_per_cam, imgs_per_cam_depth, temp_base_dir, batch_id
    )


def _encode_color_camera_worker(
    camera_dir: str, camera: str, out_path: str, train_hz: int, stats_output_dir: str
):
    return _encode_color_camera_worker_impl(
        camera_dir, camera, out_path, train_hz, stats_output_dir
    )


def _encode_depth_camera_worker(
    camera_dir: str, camera: str, out_path: str, train_hz: int, apply_denoise: bool
):
    return _encode_depth_camera_worker_impl(
        camera_dir, camera, out_path, train_hz, apply_denoise
    )


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
    chunk_size: int,
):
    return _encode_batch_segment_color_impl(
        temp_dir, batch_id, camera, train_hz, segment_dir
    )


def _concat_segments_ffmpeg(
    segment_paths: list,
    output_path: str,
    train_hz: int,
):
    if not segment_paths:
        return False
    segment_dir = os.path.dirname(segment_paths[0])
    return _concat_segments_ffmpeg_impl(segment_dir, output_path, train_hz)

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



# ==================== 原有视频编码函数 ====================


def encode_complete_videos_from_temp(
    temp_base_dir: str,
    output_dir: str,
    episode_uuid: str,
    raw_config: Config,
    use_depth: bool = True,
):
    return encode_complete_videos_from_temp_impl(
        temp_base_dir, output_dir, episode_uuid, raw_config, use_depth
    )

