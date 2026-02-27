"""Main conversion pipeline orchestration for Kuavo -> LeRobot."""

import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid
from typing import Literal

from converter.configs import Config
from converter.pipeline.dataset_builder import DatasetConfig
from converter.pipeline.batch_processor import populate_dataset_stream
from converter.reader.reader_entry import (
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_DEXHAND_JOINT_NAMES,
    DEFAULT_HEAD_JOINT_NAMES,
    DEFAULT_JOINT_NAMES_LIST,
    DEFAULT_LEG_JOINT_NAMES,
    DEFAULT_LEJUCLAW_JOINT_NAMES,
)
from converter.pipeline.batch_merger import (
    get_batch_dirs,
    merge_meta_files,
    merge_metadata,
    merge_parquet_files,
)
from converter.data.bag_discovery import get_bag_time_info
from converter.data.common_utils import is_valid_hand_data
from converter.data.episode_loader import load_hand_data_worker
from converter.data.metadata_merge import get_time_range_from_moments
from converter.media.video_orchestrator import (
    BatchSegmentEncoder,
    StreamingVideoEncoderManager,
    _encode_depth_camera_worker,
    encode_complete_videos_from_temp,
)
from converter.media.schedule import resolve_video_process_timeout_sec, resolve_video_schedule
from converter.media.video_finalize import _join_with_timeout_or_raise

logger = logging.getLogger(__name__)

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
    dataset_name: str | None = None,
):

    from converter.reader.reader_entry import (
        KuavoRosbagReader,
        DEFAULT_JOINT_NAMES_LIST,
        DEFAULT_LEG_JOINT_NAMES,
        DEFAULT_ARM_JOINT_NAMES,
        DEFAULT_HEAD_JOINT_NAMES,
        DEFAULT_JOINT_NAMES,
        DEFAULT_LEJUCLAW_JOINT_NAMES,
        DEFAULT_DEXHAND_JOINT_NAMES,
    )
    from converter.reader.postprocess_utils import PostProcessorUtils

    config = raw_config

    # 处理并行 ROSbag 读取环境变量
    env_parallel = os.environ.get("USE_PARALLEL_ROSBAG_READ", "").lower()
    if env_parallel in ("true", "1", "yes"):
        config.use_parallel_rosbag_read = True
        logger.info(
            "[CONFIG] 并行 ROSbag 读取已通过环境变量启用 (USE_PARALLEL_ROSBAG_READ=true)"
        )
    elif env_parallel in ("false", "0", "no"):
        config.use_parallel_rosbag_read = False

    env_workers = os.environ.get("PARALLEL_ROSBAG_WORKERS", "")
    if env_workers.isdigit():
        config.parallel_rosbag_workers = int(env_workers)
        logger.info("[CONFIG] 并行 worker 数量: %s", config.parallel_rosbag_workers)

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

    # 为整次导出创建目录（优先使用指定数据名）
    episode_uuid = dataset_name or str(uuid.uuid4())
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
        logger.error("子进程异常退出: %s", result.get("error"))
        if result.get("traceback"):
            logger.error("子进程 traceback:\n%s", result.get("traceback"))
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
    logger.info(
        "检测到手部类型: USE_LEJU_CLAW=%s, USE_QIANGNAO=%s",
        USE_LEJU_CLAW,
        USE_QIANGNAO,
    )

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

    dataset_config = DatasetConfig()


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

    schedule = resolve_video_schedule(raw_config)
    logger.info(
        "[SCHEDULE] 视频调度: cores=%s pipeline_workers=%s encode_processes=%s queue_limit=%s",
        schedule.cores,
        schedule.pipeline_workers,
        schedule.max_encode_processes,
        schedule.queue_limit,
    )

    if use_pipeline_encoding and getattr(raw_config, "separate_video_storage", False):
        temp_video_dir = os.path.join("/tmp", "kuavo_video_temp", episode_uuid)
        segment_dir = os.path.join("/tmp", "kuavo_video_segments", episode_uuid)
        video_output_dir = base_root

        pipeline_encoder = BatchSegmentEncoder(
            temp_base_dir=temp_video_dir,
            segment_base_dir=segment_dir,
            video_output_dir=video_output_dir,
            cameras=raw_config.default_camera_names,
            train_hz=raw_config.train_hz,
            uuid_str=episode_uuid,
            chunk_size=800,  # 固定批次大小
            max_workers=schedule.pipeline_workers,
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

        video_output_dir = base_root
        default_queue_limit = getattr(raw_config, "video_queue_limit", schedule.queue_limit)
        queue_limit = int(os.environ.get("VIDEO_QUEUE_LIMIT", default_queue_limit))

        streaming_encoder = StreamingVideoEncoderManager(
            cameras=raw_config.default_camera_names,
            video_output_dir=video_output_dir,
            uuid_str=episode_uuid,
            train_hz=raw_config.train_hz,
            queue_limit=queue_limit,
        )

    # 执行流式填充（快速生成lerobot数据）
    stream_context = {
        "use_depth": use_depth,
        "episode_uuid": episode_uuid,
        "dataset_config": dataset_config,
        "only_half_up_body": ONLY_HALF_UP_BODY,
        "control_hand_side": CONTROL_HAND_SIDE,
        "slice_robot": SLICE_ROBOT,
        "slice_claw": SLICE_CLAW,
        "merge_hand_position": MERGE_HAND_POSITION,
        "use_leju_claw": USE_LEJU_CLAW,
        "use_qiangnao": USE_QIANGNAO,
        "default_joint_names_list": DEFAULT_JOINT_NAMES_LIST,
    }
    cam_stats = populate_dataset_stream(
        raw_config=raw_config,
        bag_files=processed_files,
        task=task,
        mode=mode,
        moment_json_dir=moment_json_DIR,
        base_root=base_root,
        context=stream_context,
        metadata_json_dir=metadata_json_DIR,
        pipeline_encoder=pipeline_encoder,
        streaming_encoder=streaming_encoder,
    )

    logger.info("[INFO] ========== 主数据处理完成 ==========")
    logger.info("[INFO] LeRobot数据已保存到: %s", base_root)

    # ===== 优化: 提前启动视频编码，与合并并行 =====
    base_path = Path(base_root).resolve()
    output_dir = base_path
    encoding_thread = None
    encoding_error = []

    if getattr(raw_config, "separate_video_storage", False):
        temp_video_dir = os.path.join("/tmp", "kuavo_video_temp", episode_uuid)
        video_output_dir = output_dir
        async_encoding = getattr(raw_config, "async_video_encoding", False)

        # 流式/流水线编码器特殊处理（它们需要在合并后finalize）
        if streaming_encoder is None and pipeline_encoder is None and async_encoding:
            # 原有异步编码: 提前启动，与合并并行
            logger.info("[VIDEO] ========== 提前启动视频编码（与合并并行）==========")

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
                    encoding_error.append(e)
                    logger.exception("[VIDEO] 异步编码出错: %s", e)

            encoding_thread = threading.Thread(target=async_encode, daemon=False)
            encoding_thread.start()
            logger.info("[VIDEO] 视频编码已在后台启动")
            logger.info("[VIDEO] 视频将保存到: %s", video_output_dir)

    # ===== 合并批次数据（与视频编码并行）=====
    _t_merge_start = time.time()
    logger.info("[INFO] 开始合并批次数据...")
    batch_dirs = get_batch_dirs(base_path)
    total_frames = merge_parquet_files(batch_dirs, output_dir)

    # 先合并生成全局 metadata.json（使用各 batch 的 metadata.json）
    try:
        merge_metadata(batch_dirs, output_dir, total_frames)
    except Exception as e:
        logger.exception("[WARN] 合并 metadata.json 失败: %s", e)

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
    logger.info("[INFO] 批次数据合并完成。耗时: %.2fs", _t_merge_end - _t_merge_start)

    # 合并后删除所有 batch 文件夹
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith("batch_"):
            try:
                shutil.rmtree(d)
                logger.info("[INFO] 已删除批次文件夹: %s", d)
            except Exception as e:
                logger.exception("[WARN] 删除批次文件夹失败: %s, 错误: %s", d, e)

    # ===== 视频编码后续处理 =====
    if getattr(raw_config, "separate_video_storage", False):
        temp_video_dir = os.path.join("/tmp", "kuavo_video_temp", episode_uuid)
        video_output_dir = output_dir

        if streaming_encoder is not None:
            # 流式编码模式：彩色视频已在批处理中编码完成，只需 finalize
            logger.info("[VIDEO] ========== 流式编码模式 ==========")
            streaming_encoder.finalize()
            logger.info("[VIDEO] 彩色视频已保存到: %s", video_output_dir)

            # 深度视频单独处理（仍然使用 ffmpeg）
            if use_depth:
                depth_temp_dir = os.path.join(temp_video_dir, "depth")
                if os.path.exists(depth_temp_dir):
                    logger.info("[VIDEO] 开始编码深度视频...")
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
                    _join_with_timeout_or_raise(
                        depth_procs,
                        "DEPTH",
                        resolve_video_process_timeout_sec(raw_config),
                    )
                    logger.info("[VIDEO] 深度视频编码完成")
                    # 清理深度临时目录
                    shutil.rmtree(depth_temp_dir, ignore_errors=True)

        elif pipeline_encoder is not None:
            # 流水线模式：等待编码完成并拼接
            print("[VIDEO] ========== 流水线编码模式 ==========")
            pipeline_encoder.finalize(use_depth=use_depth)
            print(f"[VIDEO] 所有视频已保存到: {video_output_dir}")

        elif encoding_thread is not None:
            # 异步编码已提前启动，此处等待结束并向上抛出错误
            print("[INFO] 主流程已完成，等待后台视频编码结束...")
            encoding_thread.join()
            if encoding_error:
                raise RuntimeError(f"异步视频编码失败: {encoding_error[0]}")
            print(f"[VIDEO] 所有视频已保存到: {video_output_dir}")

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
