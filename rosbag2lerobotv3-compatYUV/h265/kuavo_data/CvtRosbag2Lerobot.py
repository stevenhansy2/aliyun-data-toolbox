"""
分块流式rosbag转换器 - 低内存版本

核心优化（参考Diffusion Policy的按需读取方式）：
1. 第一遍扫描：只读取时间戳（内存占用几MB）
2. 第二遍扫描：按时间窗口分块读取+对齐+写入dataset

与原始CvtRosbag2Lerobot.py的区别：
- 原始：一次性加载整个rosbag到内存 → 对齐 → 写入（内存峰值巨大）
- 本版：分块读取 → 即时对齐 → 即时写入 → 释放内存（内存可控）

使用方法：
    python CvtRosbag2Lerobot_chunked.py --config-name=KuavoRosbag2Lerobot \
        rosbag.rosbag_dir=/path/to/rosbag \
        rosbag.lerobot_dir=/path/to/output \
        rosbag.chunk_size=100
"""
import custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import json
import os
import gc
import shutil
import subprocess
import tempfile
import rosbag
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info
import dataclasses
from common import kuavo_dataset as kuavo
from h265_rewriter import H265StreamRewriter
from reference_video_pipeline import CameraSyncState, ReferenceH265VideoEncoder
import logging


class SimpleLogger:
    def info(self, msg, *args):
        print('[INFO] ' + (msg % args if args else str(msg)), flush=True)

    def warning(self, msg, *args):
        print('[WARN] ' + (msg % args if args else str(msg)), flush=True)

    def error(self, msg, *args):
        print('[ERROR] ' + (msg % args if args else str(msg)), flush=True)

    def debug(self, msg, *args):
        print('[DEBUG] ' + (msg % args if args else str(msg)), flush=True)


DEFAULT_JOINT_NAMES_LIST = kuavo.DEFAULT_JOINT_NAMES_LIST
RESUME_STATE_FILENAME = "_resume_state.json"

# helpers for metadata-driven cropping and merging
from converter.data.metadata_merge import get_time_range_from_metadata
from converter.data.metadata_merge import merge_metadata_and_moment



def extract_actions_for_chunk(metadata, chunk_start_frame, chunk_end_frame, total_frames):
    """
    从 metadata 的 marks 中提取在指定帧范围内的动作
    """
    marks = metadata.get("marks", [])
    chunk_marks = []

    for mark in marks:
        sp = mark.get("startPosition", 0)
        ep = mark.get("endPosition", 0)

        mark_start_frame = int(sp * total_frames)
        mark_end_frame = int(ep * total_frames)

        if mark_end_frame >= chunk_start_frame and mark_start_frame <= chunk_end_frame:
            chunk_marks.append(mark)

    return chunk_marks


def save_chunk_metadata(output_path, metadata, chunk_marks, total_frames, episode_uuid, raw_config, metadata_json_path=None):
    """
    保存该 chunk 的 metadata.json
    """
    temp_metadata = metadata.copy()
    temp_metadata["marks"] = chunk_marks

    merge_metadata_and_moment(
        metadata_path=metadata_json_path,
        moment_path=None,
        output_path=str(output_path),
        uuid=episode_uuid,
        raw_config=raw_config,
        total_frames=total_frames,
    )


def calculate_total_frames(chunk_dirs):
    """计算所有 chunk 的总帧数"""
    import pyarrow.parquet as pq

    total = 0
    for chunk_dir in chunk_dirs:
        parquet_files = list(Path(chunk_dir).glob("data/**/episode_*.parquet"))
        for pf in parquet_files:
            try:
                table = pq.ParquetFile(pf)
                total += table.metadata.num_rows
            except Exception as e:
                log_print.warning(f"读取 parquet 文件失败 {pf}: {e}")
    return total


def _resume_state_path(root: str | Path) -> Path:
    return Path(root) / RESUME_STATE_FILENAME


def _load_resume_state(root: str | Path) -> dict:
    state_path = _resume_state_path(root)
    if not state_path.is_file():
        return {"completed_bags": []}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        log_print.warning(f"Failed to load resume state {state_path}: {e}")
        return {"completed_bags": []}
    completed = state.get("completed_bags", []) if isinstance(state, dict) else []
    if not isinstance(completed, list):
        completed = []
    return {"completed_bags": completed}


def _save_resume_state(root: str | Path, completed_bags: list[str]) -> None:
    state_path = _resume_state_path(root)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"completed_bags": completed_bags}, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


log_print = SimpleLogger()


def setup_logging():
    return None


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset_chunked(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
) -> LeRobotDataset:
    motors = DEFAULT_JOINT_NAMES_LIST
    cameras = kuavo.DEFAULT_CAMERA_NAMES

    action_dim = (len(motors),)
    action_name = motors

    state_dim = (len(motors),)
    state_name = (
        kuavo.DEFAULT_ARM_JOINT_NAMES[: len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2]
        + ["gripper_l"]
        + kuavo.DEFAULT_ARM_JOINT_NAMES[len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2 :]
        + ["gripper_r"]
    )

    if not kuavo.ONLY_HALF_UP_BODY:
        action_dim = (action_dim[0] + 3 + 1,)
        action_name += ["cmd_pos_x", "cmd_pos_y", "cmd_pos_yaw", "ctrl_change_cmd"]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_dim,
            "names": {"state_names": state_name},
        },
        "action": {
            "dtype": "float32",
            "shape": action_dim,
            "names": {"action_names": action_name},
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    for cam in cameras:
        feature_key = _video_feature_key(cam)
        features[feature_key] = {
            "dtype": mode,
            "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
            "names": ["channels", "height", "width"],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=kuavo.TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )


def _create_or_load_dataset_chunked(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    resume: bool = False,
) -> LeRobotDataset:
    root_path = Path(root)
    info_path = root_path / "meta" / "info.json"
    if resume and info_path.is_file():
        log_print.info(f"Resume mode: loading existing dataset at {root_path}")
        return LeRobotDataset(repo_id=repo_id, root=root_path)

    if root_path.exists():
        shutil.rmtree(root_path)

    return create_empty_dataset_chunked(
        repo_id=repo_id,
        robot_type=robot_type,
        mode=mode,
        has_velocity=has_velocity,
        has_effort=has_effort,
        dataset_config=dataset_config,
        root=root,
    )

def _video_feature_key(cam_key: str) -> str:
    return f"observation.{cam_key}" if "depth" in cam_key else f"observation.images.{cam_key}"


def _build_h265_mp4(packets: list[dict], fps: int, temp_root: str | Path, stem: str) -> Path | None:
    if not packets:
        return None
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{stem}_", dir=str(temp_root)))
    bitstream_path = temp_dir / f"{stem}.h265"
    mp4_path = temp_dir / f"{stem}.mp4"
    with open(bitstream_path, "wb") as f:
        for packet in packets:
            payload = packet["payload"]
            if not (payload.startswith(b"\x00\x00\x00\x01") or payload.startswith(b"\x00\x00\x01")):
                payload = b"\x00\x00\x00\x01" + payload
            f.write(payload)
    commands = [
        ["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts", "-framerate", str(fps), "-f", "hevc", "-i", str(bitstream_path), "-c", "copy", str(mp4_path)],
        ["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts", "-framerate", str(fps), "-f", "hevc", "-i", str(bitstream_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
            return mp4_path
    shutil.rmtree(temp_dir, ignore_errors=True)
    return None


def _build_rawvideo_mp4(packets: list[dict], fps: int, temp_root: str | Path, stem: str) -> Path | None:
    if not packets:
        return None
    first = packets[0]
    width = first.get("width")
    height = first.get("height")
    pix_fmt = first.get("pix_fmt")
    if not width or not height or not pix_fmt:
        return None
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{stem}_", dir=str(temp_root)))
    raw_path = temp_dir / f"{stem}.raw"
    mp4_path = temp_dir / f"{stem}.mp4"
    with open(raw_path, "wb") as f:
        for packet in packets:
            f.write(packet["payload"])
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", str(raw_path),
        "-vf", f"scale={kuavo.RESIZE_W}:{kuavo.RESIZE_H}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
        return mp4_path
    shutil.rmtree(temp_dir, ignore_errors=True)
    return None


def _yuv_pix_fmt_from_encoding(encoding: str) -> str | None:
    encoding = (encoding or '').lower()
    mapping = {
        'yuv422': 'yuyv422',
        'yuv422_yuy2': 'yuyv422',
        'yuyv': 'yuyv422',
        'yuyv422': 'yuyv422',
        'uyvy': 'uyvy422',
        'uyvy422': 'uyvy422',
        'nv12': 'nv12',
        'nv21': 'nv21',
        'yuv420': 'yuv420p',
        'i420': 'yuv420p',
        'yuv420p': 'yuv420p',
    }
    return mapping.get(encoding)


def _run_ffmpeg_commands(
    cam_key: str,
    commands: list[list[str]],
    mp4_path: Path,
    timeouts: list[int],
    modes: list[str],
) -> bool:
    for idx, cmd in enumerate(commands):
        timeout_sec = timeouts[min(idx, len(timeouts) - 1)]
        mode = modes[min(idx, len(modes) - 1)]
        print(f'[INFO] [{cam_key}] ffmpeg start mode={mode} timeout={timeout_sec}s', flush=True)
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            _stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print(f'[WARN] [{cam_key}] ffmpeg timeout in mode={mode}, killing process', flush=True)
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.communicate(timeout=2)
                except Exception:
                    pass
            continue

        if proc.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 0:
            print(f'[INFO] [{cam_key}] ffmpeg done mode={mode} size={mp4_path.stat().st_size}', flush=True)
            return True

        err = stderr.decode('utf-8', errors='ignore').strip().splitlines()[-1:]
        err_msg = err[0] if err else f'returncode={proc.returncode}'
        print(f'[WARN] [{cam_key}] ffmpeg failed mode={mode}: {err_msg}', flush=True)

    return False


def _build_full_stream_videos(bag_file: Path, output_root: str | Path) -> dict[str, dict]:
    reader = kuavo.KuavoRosbagReader()
    bag = rosbag.Bag(str(bag_file))
    topic_info = bag.get_type_and_topic_info().topics

    cam_topics: dict[str, str] = {}
    reverse_cam_topics: dict[str, str] = {}
    for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
        if 'depth' in cam_key:
            continue
        config = reader._topic_process_map.get(cam_key)
        if config is None:
            continue
        topic = next((candidate for candidate in config.get('topic_candidates', [config['topic']]) if candidate in topic_info), None)
        if topic is not None:
            cam_topics[cam_key] = topic
            reverse_cam_topics[topic] = cam_key

    temp_root = Path(tempfile.gettempdir()) / "rosbag2lerobotv3_source_videos"
    work_dir = temp_root / bag_file.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Building full-stream source videos for {bag_file.name}', flush=True)
    print(f'[INFO] Camera topics detected: {reverse_cam_topics}', flush=True)

    if not cam_topics:
        bag.close()
        return {}

    head_cam_key = 'head_cam_h' if 'head_cam_h' in cam_topics else next(iter(cam_topics.keys()))
    sync_state = CameraSyncState(
        head_camera=head_cam_key,
        observation_topics=cam_topics,
        max_time_diff=0.04,
    )

    encoders: dict[str, ReferenceH265VideoEncoder] = {}
    start_ts: float | None = None

    for topic, msg, t in bag.read_messages(topics=list(reverse_cam_topics.keys())):
        cam_key = reverse_cam_topics.get(topic)
        if cam_key is None:
            continue
        if hasattr(msg, 'encoding'):
            continue

        payload = bytes(msg.data)
        fmt = str(getattr(msg, 'format', '') or '').lower()
        is_h265 = ('h265' in fmt or 'hevc' in fmt or kuavo.KuavoMsgProcesser._looks_like_h26x(payload))
        if not is_h265:
            continue
        if not (payload.startswith(b'\x00\x00\x00\x01') or payload.startswith(b'\x00\x00\x01')):
            payload = b'\x00\x00\x00\x01' + payload

        sync_state.update_camera(cam_key, t.to_sec(), payload, is_h265=True)
        if cam_key != head_cam_key:
            continue

        aligned = sync_state.try_build_aligned_frame(t.to_sec())
        if not aligned:
            continue

        if start_ts is None:
            start_ts = t.to_sec()
            print(f'[INFO] [reference] all cameras IDR ready at T={start_ts:.3f}', flush=True)

        for out_cam_key, frame in aligned.items():
            encoder = encoders.get(out_cam_key)
            if encoder is None:
                encoder = ReferenceH265VideoEncoder(work_dir / out_cam_key, kuavo.MAIN_TIMELINE_FPS)
                encoders[out_cam_key] = encoder
            encoder.write_frame(frame.payload, is_duplicate=frame.is_duplicate)

    bag.close()

    sources = {}
    for cam_key, encoder in encoders.items():
        mp4_path = encoder.close()
        if mp4_path is not None and mp4_path.exists():
            print(f'[INFO] [{cam_key}] reference pipeline video ready size={mp4_path.stat().st_size} path={mp4_path}', flush=True)
            sources[cam_key] = {'path': str(mp4_path), 'start_ts': start_ts or 0.0}
        else:
            print(f'[WARN] [{cam_key}] reference pipeline failed to produce mp4', flush=True)

    return sources


def populate_dataset_chunked(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    chunk_size: int = 800,
    metadata: dict | None = None,
    root: str = None,  # 输出根目录
    repo_id: str = None,  # 数据集 ID
    metadata_json_path: str = None,  # metadata.json 文件路径
    resume: bool = False,
) -> LeRobotDataset:
    """
    使用分块流式处理填充数据集
    
    核心优化：
    1. 第一遍扫描只读取时间戳（内存几MB）
    2. 第二遍扫描按时间窗口分块读取+对齐+写入
    3. 每个chunk处理完立即保存并释放内存
    
    Args:
        dataset: LeRobotDataset实例
        bag_files: rosbag文件路径列表
        task: 任务描述
        episodes: 要处理的episode索引列表
        chunk_size: 每个chunk包含的帧数（默认100帧）
    """
    if episodes is None:
        episodes = range(len(bag_files))
    
    failed_bags = []
    completed_bags = set(_load_resume_state(root).get("completed_bags", [])) if resume else set()
    log_print.info(f"Total episodes to process: {len(episodes)}")
    bag_reader = kuavo.KuavoRosbagReader()
    
    # 内存监控
    process = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except ImportError:
        pass
    
    def log_memory(prefix: str):
        if process:
            mem_mb = process.memory_info().rss / 1024 / 1024
            log_print.debug(f"{prefix} Memory: {mem_mb:.2f} MB")
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = Path(bag_files[ep_idx])
        log_print.warning(f"Processing {ep_path}")
        log_memory("Before processing")

        # 计算裁剪范围 (fractional positions 0.0-1.0)
        crop_range = None
        start_frame = None
        end_frame = None

        if metadata:
            # 优先使用 explicit frame indices（来自已计算好的 label_info）
            try:
                label_info = metadata.get("label_info") or {}
                action_cfg = label_info.get("action_config") or []
                starts = [a.get("start_frame") for a in action_cfg if a.get("start_frame") is not None]
                ends = [a.get("end_frame") for a in action_cfg if a.get("end_frame") is not None]
                if starts and ends:
                    start_frame = min(starts)
                    end_frame = max(ends)
                    log_print.info(
                        f"Using explicit frame range from label_info: {start_frame}-{end_frame}"
                    )
            except Exception:
                pass

            # 如果没有 explicit frame indices，使用 fractional positions（来自 marks）
            if start_frame is None:
                frac_range = get_time_range_from_metadata(metadata)
                if frac_range is not None:
                    crop_range = frac_range  # (min_start_position, max_end_position)
                    log_print.info(
                        f"Using fractional crop range from marks: {crop_range[0]:.3f}-{crop_range[1]:.3f}"
                    )

        try:
            direct_video_sources = _build_full_stream_videos(ep_path, root)
            # 收集当前episode的所有帧
            frames_buffer = []
            frame_count = [0]
            chunk_frame_ranges = []  # 记录每个 chunk 的帧范围 [(start, end), ...]
            current_chunk_frames = 0  # 当前 chunk 已处理的帧数
            total_frames = 0  # 初始化总帧数（用于 metadata 处理）

            def on_frame(aligned_frame: dict, frame_idx: int):
                nonlocal current_chunk_frames, total_frames
                """处理单帧对齐数据"""
                # 如果使用 explicit frame indices 进行裁剪（双重保险）
                if start_frame is not None:
                    if frame_idx < start_frame or (end_frame is not None and frame_idx >= end_frame):
                        return
                """处理单帧对齐数据"""

                def get_array(key, dtype, default_empty=True):
                    item = aligned_frame.get(key)
                    if item is None:
                        return np.array([], dtype=dtype) if default_empty else None
                    return np.array(item.get("data", []), dtype=dtype)

                # =========================
                # 1. state / action
                # =========================
                state = get_array('observation.state', np.float32)
                action = get_array('action', np.float32)
                

                if state.size == 0 or action.size == 0:
                    return

                # =========================
                # 2. arm trajectory（alt 优先）
                # =========================
                arm_traj     = get_array("action.kuavo_arm_traj", np.float32)
                arm_traj_alt = get_array("action.kuavo_arm_traj_alt", np.float32)
                if arm_traj_alt.size == 0 and arm_traj.size == 0:
                    return
                action[12:26] = arm_traj_alt if arm_traj_alt.size else arm_traj
                
                # 接口留用
                velocity = None
                effort = None

                # =========================
                # 3. 手部数据读取
                # =========================
                claw_state     = get_array("observation.claw", np.float64)
                claw_action    = get_array("action.claw", np.float64)
                qiangnao_state = get_array("observation.qiangnao", np.float64)
                qiangnao_action= get_array("action.qiangnao", np.float64)
                rq2f85_state   = get_array("observation.rq2f85", np.float64)
                rq2f85_action  = get_array("action.rq2f85", np.float64)

                if claw_state.size == 0 and qiangnao_state.size == 0 and rq2f85_state.size==0:
                    return 
                if claw_action.size == 0 and qiangnao_action.size==0 and rq2f85_action.size ==0:
                    return
                # =========================
                # 4. 手部归一化（保持原逻辑）
                # =========================
                if kuavo.IS_BINARY:
                    qiangnao_state  = np.where(qiangnao_state > 50, 1, 0)
                    qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
                    claw_state      = np.where(claw_state > 50, 1, 0)
                    claw_action     = np.where(claw_action > 50, 1, 0)
                    rq2f85_state    = np.where(rq2f85_state > 0.4, 1, 0)
                    rq2f85_action   = np.where(rq2f85_action > 70, 1, 0)
                    # rq2f85_state = np.where(rq2f85_state > 0.1, 1, 0)
                    # rq2f85_action = np.where(rq2f85_action > 128, 1, 0)
                else:
                    if claw_state.size:      claw_state /= 100
                    if claw_action.size:     claw_action /= 100
                    if qiangnao_state.size:  qiangnao_state /= 100
                    if qiangnao_action.size: qiangnao_action /= 100
                    if rq2f85_state.size:    rq2f85_state /= 0.8
                    if rq2f85_action.size:   rq2f85_action /= 255
                    # rq2f85_state = rq2f85_state / 0.8
                    # rq2f85_action = rq2f85_action / 255

                if claw_action.size == 0 and qiangnao_action.size == 0:
                    claw_action = rq2f85_action
                    claw_state  = rq2f85_state

                # =========================
                # 5. 构建最终 state / action
                # =========================
                if kuavo.USE_LEJU_CLAW or kuavo.USE_QIANGNAO:
                    hand_type = "LEJU" if kuavo.USE_LEJU_CLAW else "QIANGNAO"
                    s_list, a_list = [], []

                    def get_hand_slice(hand_side):
                        s_slice = kuavo.SLICE_ROBOT[hand_side]

                        if hand_type == "LEJU":
                            c_slice = kuavo.SLICE_CLAW[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                claw_state[c_slice[0]:c_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]],
                                                claw_action[c_slice[0]:c_slice[-1]]))
                        else:
                            d_slice = kuavo.SLICE_DEX[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                qiangnao_state[d_slice[0]:d_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]],
                                                qiangnao_action[d_slice[0]:d_slice[-1]]))
                        return s, a

                    if kuavo.CONTROL_HAND_SIDE in ("left", "both"):
                        s, a = get_hand_slice(0)
                        s_list.append(s)
                        a_list.append(a)

                    if kuavo.CONTROL_HAND_SIDE in ("right", "both"):
                        s, a = get_hand_slice(1)
                        s_list.append(s)
                        a_list.append(a)

                    final_state  = np.concatenate(s_list).astype(np.float32)
                    final_action = np.concatenate(a_list).astype(np.float32)
                else:
                    raise ValueError(f"eef type are not supported! ")

                # =========================
                # 6. cmd_pos_world & gap_flag
                # =========================
                if not kuavo.ONLY_HALF_UP_BODY:
                    cmd_pos_world = get_array(
                        "action.cmd_pos_world", np.float32
                    )
                    if cmd_pos_world.size == 0:
                        raise ValueError(f"kuavo.ONLY_HALF_UP_BODY is {kuavo.ONLY_HALF_UP_BODY}, but no action.cmd_pos_world found! ")
                    gap_flag = 1.0 if arm_traj.size and np.any(arm_traj == 999.0) else 0.0

                    final_action = np.concatenate(
                        [final_action, cmd_pos_world, np.array([gap_flag], np.float32)],
                        axis=0
                    )

                # =========================
                # 7. 构建 frame
                # =========================
                frame = {
                    "observation.state": torch.from_numpy(final_state).type(torch.float32),
                    "action": torch.from_numpy(final_action).type(torch.float32),
                }

                frame["timestamp"] = float(aligned_frame.get("timestamp", frame_idx / kuavo.TRAIN_HZ))

                for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
                    cam_data = aligned_frame.get(cam_key)
                    if "depth" not in cam_key and cam_key in direct_video_sources:
                        source = direct_video_sources[cam_key]
                        rel_ts = max(float(aligned_frame.get("timestamp", 0.0)) - float(source.get("start_ts", 0.0)), 0.0)
                        frame[f"observation.images.{cam_key}"] = {
                            "path": str(source["path"]),
                            "timestamp": rel_ts,
                        }
                        continue

                    if cam_data and "data" in cam_data:
                        img = cam_data["data"]
                        if "depth" in cam_key:
                            min_d, max_d = kuavo.DEPTH_RANGE
                            depth = np.clip(img, min_d, max_d)
                            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
                            depth_uint8 = (depth_norm * 255).astype(np.uint8)
                            frame[f"observation.{cam_key}"] = depth_uint8[..., None].repeat(3, -1)
                        else:
                            frame[f"observation.images.{cam_key}"] = img
                    else:
                        return
                
                if velocity is not None:
                    frame["observation.velocity"] = velocity
                if effort is not None:
                    frame["observation.effort"] = effort
                
                frames_buffer.append(frame)
                frame_count[0] += 1
                total_frames += 1  # 更新总帧数

            
            def on_chunk_done():
                """每个chunk处理完后的回调：保存并释放内存"""
                nonlocal root, repo_id, metadata_json_path
                if len(frames_buffer) == 0:
                    return

                # 将所有缓存的帧添加到dataset
                for frame in frames_buffer:
                    frame["task"] = task
                    dataset.add_frame(frame)

                # 保存当前chunk
                dataset.save_episode()
                dataset.hf_dataset = dataset.create_hf_dataset()

                # 记录当前 chunk 的帧范围
                chunk_start_frame = frame_count[0] - len(frames_buffer)
                chunk_end_frame = frame_count[0] - 1
                chunk_frame_ranges.append((chunk_start_frame, chunk_end_frame))
                chunk_idx = len(chunk_frame_ranges) - 1

                # 如果有 metadata，保存该 batch 的 metadata
                if metadata and total_frames > 0:
                    try:
                        # 提取该 chunk 范围内的 marks
                        chunk_marks = extract_actions_for_chunk(
                            metadata=metadata,
                            chunk_start_frame=chunk_start_frame,
                            chunk_end_frame=chunk_end_frame,
                            total_frames=total_frames
                        )

                        if chunk_marks:
                            # 创建 batch 目录
                            batch_dir = Path(root) / f"batch_{chunk_idx:04d}"
                            batch_dir.mkdir(parents=True, exist_ok=True)

                            # 保存该 chunk 的 metadata
                            save_chunk_metadata(
                                output_path=batch_dir / "metadata.json",
                                metadata=metadata,
                                chunk_marks=chunk_marks,
                                total_frames=len(frames_buffer),  # 当前 chunk 的帧数
                                episode_uuid=repo_id.split("/")[-1],
                                raw_config=None,
                                metadata_json_path=metadata_json_path  # 传递原始 metadata.json 路径
                            )

                            # 复制 data 和 meta 目录到 batch 目录
                            chunk_data_dir = Path(root) / "data"
                            if chunk_data_dir.exists():
                                import shutil
                                batch_data_dir = batch_dir / "data"
                                if batch_data_dir.exists():
                                    shutil.rmtree(batch_data_dir)
                                shutil.copytree(chunk_data_dir, batch_data_dir)

                            batch_meta_dir = Path(root) / "meta"
                            if batch_meta_dir.exists():
                                batch_meta_out_dir = batch_dir / "meta"
                                if batch_meta_out_dir.exists():
                                    shutil.rmtree(batch_meta_out_dir)
                                shutil.copytree(batch_meta_dir, batch_meta_out_dir)

                            log_print.info(f"✓ 保存 batch {chunk_idx} metadata: {len(chunk_marks)} 个动作, 帧范围 {chunk_start_frame}-{chunk_end_frame}")
                        else:
                            log_print.debug(f"Batch {chunk_idx}: 无相关的 marks")
                    except Exception as e:
                        log_print.warning(f"保存 batch metadata 失败: {e}")
                        import traceback
                        traceback.print_exc()

                # 清空buffer并释放内存
                frames_buffer.clear()
                gc.collect()

                log_memory(f"After saving chunk (total frames: {frame_count[0]})")
            
            # 使用分块流式处理（传递裁剪范围）
            bag_reader.process_rosbag_chunked(
                bag_file=str(ep_path),
                frame_callback=on_frame,
                chunk_size=chunk_size,
                save_callback=on_chunk_done,
                crop_range=crop_range  # 传递裁剪范围，如果为 None 则不裁剪
            )
             
            # 处理剩余的帧
            if len(frames_buffer) > 0:
                for frame in frames_buffer:
                    dataset.add_frame(frame, task=task)
                dataset.save_episode()
                dataset.hf_dataset = dataset.create_hf_dataset()
                frames_buffer.clear()
                gc.collect()
            
            log_print.info(f"Episode {ep_idx} completed: {frame_count[0]} frames")
            if resume:
                completed_bags.add(str(ep_path.resolve()))
                _save_resume_state(root, sorted(completed_bags))
                log_print.info(f"Resume checkpoint saved for bag: {ep_path.name}")
            
        except Exception as e:
            log_print.error(f"Error processing {ep_path}: {e}")
            import traceback
            traceback.print_exc()
            failed_bags.append(str(ep_path))
            continue
        
        log_memory("After episode")
        gc.collect()
    
    if failed_bags:
        with open("error.txt", "w") as f:
            for bag in failed_bags:
                f.write(bag + "\n")
        log_print.error(f"{len(failed_bags)} failed bags written to error.txt")

    # 如果生成了 batch 目录，执行批次合并
    if chunk_frame_ranges and len(chunk_frame_ranges) > 1:
        log_print.info(f"\n{'='*60}")
        log_print.info(f"检测到 {len(chunk_frame_ranges)} 个 batch，开始合并...")
        log_print.info(f"{'='*60}")

        try:
            from converter.pipeline.batch_merger import merge_all_batches

            # 合并所有 batch
            merge_all_batches(
                input_dir=str(root),
                output_dir=str(root)
            )

            log_print.info(f"\n✓ 批次合并完成！")

        except Exception as e:
            log_print.error(f"批次合并失败: {e}")
            import traceback
            traceback.print_exc()

    # 清理残留的 batch 目录
    base_path = Path(root)
    if base_path.exists():
        for d in base_path.iterdir():
            if d.is_dir() and d.name.startswith("batch_"):
                try:
                    shutil.rmtree(d)
                    log_print.info("已删除批次文件夹: %s", d)
                except Exception as e:
                    log_print.warning("删除批次文件夹失败: %s, 错误: %s", d, e)

    return dataset


def _sanitize_dataset_name(name: str) -> str:
    """将 bag 文件名转换为合法的数据集目录名"""
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "bag"


def port_kuavo_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    n: int | None = None,
    chunk_size: int = 800,
    metadata: dict | None = None,
    metadata_json_path: str = None,
    bag_files_override: list[Path] | None = None,
    resume: bool = False,
):
    """
    分块流式转换rosbag到LeRobot格式
    
    Args:
        raw_dir: rosbag目录
        repo_id: 输出数据集ID
        task: 任务描述
        chunk_size: 每个chunk的帧数（默认100）
    """
    if bag_files_override is not None:
        bag_files = bag_files_override
    else:
        bag_reader = kuavo.KuavoRosbagReader()
        bag_files = bag_reader.list_bag_files(raw_dir)

        if isinstance(n, int) and n > 0:
            num_available_bags = len(bag_files)
            if n > num_available_bags:
                log_print.warning(f"Requested {n} bags, but only {num_available_bags} available. Using all available bags.")
                n = num_available_bags
            select_idx = np.random.choice(num_available_bags, n, replace=False)
            bag_files = [bag_files[i] for i in select_idx]
    
    state = _load_resume_state(root) if resume else {"completed_bags": []}
    completed_bags = {str(Path(p).resolve()) for p in state.get("completed_bags", [])}
    if resume and not completed_bags and Path(root, "meta", "info.json").is_file():
        try:
            existing_dataset = LeRobotDataset(repo_id=repo_id, root=root)
            inferred_count = min(existing_dataset.meta.total_episodes, len(bag_files))
            if inferred_count > 0:
                completed_bags = {str(Path(bf).resolve()) for bf in bag_files[:inferred_count]}
                _save_resume_state(root, sorted(completed_bags))
                log_print.warning(
                    f"Resume state missing; inferred {inferred_count} completed bags from existing dataset episodes."
                )
        except Exception as e:
            log_print.warning(f"Failed inferring completed bags from existing dataset: {e}")

    original_bag_count = len(bag_files)
    if resume and completed_bags:
        bag_files = [bf for bf in bag_files if str(Path(bf).resolve()) not in completed_bags]
        log_print.info(
            f"Resume mode: skipping {original_bag_count - len(bag_files)} completed bags, remaining {len(bag_files)}"
        )

    dataset = _create_or_load_dataset_chunked(
        repo_id,
        robot_type="kuavo4pro",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
        root=root,
        resume=resume,
    )

    if len(bag_files) == 0:
        log_print.info("No remaining bags to process.")
        return dataset
    
    try:
        dataset = populate_dataset_chunked(
            dataset,
            bag_files,
            task=task,
            episodes=episodes,
            chunk_size=chunk_size,
            metadata=metadata,
            root=root,
            repo_id=repo_id,
            metadata_json_path=metadata_json_path,
            resume=resume,
        )
    finally:
        try:
            dataset.meta._flush_metadata_buffer()
        except Exception:
            pass
        try:
            dataset.consolidate()
        except Exception:
            pass
    # 如果提供了 metadata 字典，则生成完整的 metadata（包含 label_info.action_config）
    try:
        if metadata:
            # 使用 root 路径而不是 LEROBOT_HOME
            out_dir = Path(root)
            log_print.info(f"✓ 使用实际输出目录: {out_dir}")

            # 计算总帧数
            calculated_frames = calculate_total_frames([out_dir])

            # 优先使用 metadata.json 中的 total_frames（如果有）
            total_frames = metadata.get("total_frames", calculated_frames)

            log_print.info(f"✓ 计算的总帧数: {calculated_frames}")
            log_print.info(f"✓ 使用的总帧数: {total_frames} (优先使用 metadata.total_frames)")

            if total_frames > 0:
                # 生成完整的 metadata.json（包含转换后的 action_config）
                from converter.data.metadata_merge import merge_metadata_and_moment
                import json
                import tempfile

                # 保存 metadata 字典为临时文件（移除 total_frames 避免污染输出）
                temp_meta = metadata.copy()
                temp_meta.pop("total_frames", None)  # 移除 total_frames

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
                    json.dump(temp_meta, tmp, ensure_ascii=False, indent=2)
                    tmp_metadata_path = tmp.name

                try:
                    final_metadata_path = out_dir / "metadata.json"
                    merge_metadata_and_moment(
                        metadata_path=tmp_metadata_path,
                        moment_path=None,
                        output_path=str(final_metadata_path),
                        uuid=repo_id.split("/")[-1],
                        raw_config=None,
                        total_frames=total_frames,  # 使用原始总帧数
                    )

                    log_print.info(f"Wrote metadata to {final_metadata_path}")
                finally:
                    # 清理临时文件
                    import os
                    if os.path.exists(tmp_metadata_path):
                        os.unlink(tmp_metadata_path)
                    import os
                    if os.path.exists(tmp_metadata_path):
                        os.unlink(tmp_metadata_path)
            else:
                # 如果无法计算总帧数，只保存原始 metadata
                meta_path = out_dir / "metadata.json"
                import json
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                log_print.info(f"Wrote raw metadata to {meta_path}")
    except Exception as e:
        log_print.warning(f"Failed to process metadata.json: {e}")
        import traceback
        traceback.print_exc()

    try:
        dataset.stop_image_writer()
    except Exception as e:
        log_print.warning(f"Failed to stop image writer cleanly: {e}")

    try:
        dataset.finalize()
    except Exception as e:
        log_print.warning(f"Failed to finalize dataset cleanly: {e}")

    dataset.hf_dataset = None
    gc.collect()
    return dataset


@hydra.main(
    config_path="./configs/",
    config_name="KuavoRosbag2Lerobot",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    分块流式转换入口
    
    使用方法：
        python CvtRosbag2Lerobot_chunked.py \
            rosbag.rosbag_dir=/path/to/rosbag \
            rosbag.lerobot_dir=/path/to/output \
            rosbag.chunk_size=100
    """
    import time
    start_time = time.time()
    setup_logging()  # set logger 

    global DEFAULT_JOINT_NAMES_LIST
    kuavo.init_parameters(cfg)

    n = cfg.rosbag.num_used
    raw_dir = cfg.rosbag.rosbag_dir
    version = cfg.rosbag.lerobot_dir

    task_name = os.path.basename(raw_dir)
    lerobot_base_dir = os.path.realpath(version)
    os.makedirs(lerobot_base_dir, exist_ok=True)

    chunk_size = cfg.rosbag.get("chunk_size", 800)
    resume_enabled = os.environ.get("RESUME_CONVERSION", "true").strip().lower() not in ("0", "false", "no")
    # 尝试从配置/命令行参数读取 metadata.json 路径并加载
    metadata = None
    meta_path = None
    try:
        meta_path = cfg.rosbag.get("metadata_json", None)
        if meta_path:
            if os.path.isfile(meta_path):
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                log_print.info(f"Loaded metadata.json from {meta_path}")
            else:
                log_print.warning(f"metadata_json path not found: {meta_path}")
    except Exception as e:
        log_print.warning(f"Failed to load metadata.json: {e}")

    # 如果 metadata 中包含控制字段，可以用来覆盖 kuavo 参数（影响相机选择与切片）
    if metadata:
        try:
            if "whichArm" in metadata or "which_arm" in metadata:
                side = metadata.get("whichArm", metadata.get("which_arm"))
                if side in ("left", "right", "both"):
                    kuavo.CONTROL_HAND_SIDE = side
                    log_print.info(f"Override CONTROL_HAND_SIDE from metadata: {side}")
            if "useDepth" in metadata:
                kuavo.USE_DEPTH = bool(metadata.get("useDepth"))
                log_print.info(f"Override USE_DEPTH from metadata: {kuavo.USE_DEPTH}")
            if "eefType" in metadata or "eef_type" in metadata:
                eef = metadata.get("eefType", metadata.get("eef_type"))
                if isinstance(eef, str):
                    if "claw" in eef.lower():
                        kuavo.USE_LEJU_CLAW = True
                        kuavo.USE_QIANGNAO = False
                        log_print.info("Override to use LEJU_CLAW based on metadata.eefType")
                    elif "dex" in eef.lower() or "qiang" in eef.lower():
                        kuavo.USE_QIANGNAO = True
                        kuavo.USE_LEJU_CLAW = False
                        log_print.info("Override to use QIANGNAO based on metadata.eefType")
        except Exception as e:
            log_print.warning(f"Failed applying metadata overrides to kuavo params: {e}")

    # 重新计算 joint names 列表（可能被 metadata 覆盖）
    half_arm = len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(kuavo.DEFAULT_DEXHAND_JOINT_NAMES) // 2
    UP_START_INDEX = 12
    if kuavo.USE_LEJU_CLAW:
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_CLAW[0][0] + half_arm, kuavo.SLICE_CLAW[0][-1] + half_arm),
            (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw), (kuavo.SLICE_CLAW[1][0] + half_arm * 2, kuavo.SLICE_CLAW[1][-1] + half_arm * 2)
            ]
    elif kuavo.USE_QIANGNAO:
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_DEX[0][0] + half_arm, kuavo.SLICE_DEX[0][-1] + half_arm),
            (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand), (kuavo.SLICE_DEX[1][0] + half_arm * 2, kuavo.SLICE_DEX[1][-1] + half_arm * 2)
            ]
    DEFAULT_JOINT_NAMES_LIST = [DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)]

    # 发现所有 bag 文件
    bag_reader = kuavo.KuavoRosbagReader()
    all_bag_files = bag_reader.list_bag_files(raw_dir)

    if isinstance(n, int) and n > 0:
        num_available = len(all_bag_files)
        if n > num_available:
            n = num_available
        select_idx = np.random.choice(num_available, n, replace=False)
        all_bag_files = [all_bag_files[i] for i in select_idx]

    dataset_name = _sanitize_dataset_name(task_name)
    repo_id = f"lerobot/{dataset_name}"
    lerobot_dir = lerobot_base_dir

    if (not resume_enabled) and os.path.exists(lerobot_dir):
        shutil.rmtree(lerobot_dir)
    os.makedirs(lerobot_dir, exist_ok=True)

    log_print.info(f"=== Chunked Streaming Rosbag Converter ===")
    log_print.info(f"Rosbag dir: {raw_dir}")
    log_print.info(f"Output dir: {lerobot_dir}")
    log_print.info(f"Dataset name: {dataset_name}")
    log_print.info(f"Total bags to process: {len(all_bag_files)}")
    log_print.info(f"Chunk size: {chunk_size}")
    log_print.info(f"Resume conversion: {resume_enabled}")

    port_kuavo_rosbag_chunked(
        raw_dir=raw_dir,
        repo_id=repo_id,
        task=kuavo.TASK_DESCRIPTION,
        mode="video",
        root=lerobot_dir,
        n=None,
        chunk_size=chunk_size,
        metadata=metadata,
        metadata_json_path=meta_path if metadata else None,
        bag_files_override=all_bag_files,
        resume=resume_enabled,
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    log_print.info(f"Conversion completed! Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    log_print.info("Starting Kuavo rosbag to LeRobot conversion...")
    main()
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
