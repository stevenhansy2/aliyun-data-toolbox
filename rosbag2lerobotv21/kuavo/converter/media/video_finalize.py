"""Finalize videos from temporary image frames."""

import multiprocessing
import os

from converter.configs import Config
from converter.media.schedule import (
    resolve_video_process_timeout_sec,
    resolve_video_schedule,
)
from converter.media.video_workers import _encode_color_camera_worker, _encode_depth_camera_worker


def _join_with_timeout_or_raise(proc_list, kind: str, process_timeout_sec: int):
    errors = []
    for p in proc_list:
        p.join(timeout=process_timeout_sec)
        if p.is_alive():
            msg = (
                f"[VIDEO][{kind}] 子进程 pid={p.pid} 超时 {process_timeout_sec}s，"
                "将终止并整体失败退出"
            )
            print(msg)
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                print(f"[VIDEO][{kind}] 子进程 pid={p.pid} terminate 后仍存活，执行 kill")
                p.kill()
                p.join(timeout=5)
            errors.append(msg)
            continue

        if p.exitcode not in (0, None):
            errors.append(
                f"[VIDEO][{kind}] 子进程 pid={p.pid} 异常退出，exitcode={p.exitcode}"
            )

    if errors:
        raise RuntimeError("; ".join(errors))


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
    process_timeout_sec = resolve_video_process_timeout_sec(raw_config)
    schedule = resolve_video_schedule(raw_config)
    max_parallel = max(1, schedule.max_encode_processes)
    print(
        f"[VIDEO] 调度配置: cores={schedule.cores}, max_encode_processes={max_parallel}, "
        f"timeout={process_timeout_sec}s"
    )

    # 创建输出目录
    stats_output_dir = os.path.join(video_output_dir, "meta", "episodes_stats.jsonl")
    color_out_dir = os.path.join(video_output_dir, "videos", "chunk-000")

    os.makedirs(color_out_dir, exist_ok=True)

    # === 彩色：每相机一个子进程 ===
    color_temp_dir = os.path.join(temp_base_dir, "color")
    color_jobs = []
    if os.path.exists(color_temp_dir):
        for camera in os.listdir(color_temp_dir):
            camera_dir = os.path.join(color_temp_dir, camera)
            if not os.path.isdir(camera_dir):
                continue
            video_path = os.path.join(
                color_out_dir, f"observation.images.{camera}", "episode_000000.mp4"
            )
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            color_jobs.append((camera_dir, camera, video_path, raw_config.train_hz, stats_output_dir))

    # === 深度：每相机一个子进程（受 use_depth 控制） ===
    depth_temp_dir = os.path.join(temp_base_dir, "depth")
    depth_jobs = []
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
            depth_jobs.append((camera_dir, camera, video_path, raw_config.train_hz, apply_denoise))
    elif not use_depth and os.path.exists(depth_temp_dir):
        shutil.rmtree(depth_temp_dir, ignore_errors=True)
        print("[VIDEO] 跳过深度视频处理（use_depth=false），已清理深度临时目录")

    def _run_jobs_in_rounds(job_list, kind: str, target):
        for i in range(0, len(job_list), max_parallel):
            round_jobs = job_list[i : i + max_parallel]
            procs = []
            for args in round_jobs:
                p = multiprocessing.Process(target=target, args=args, daemon=False)
                p.start()
                procs.append(p)
            _join_with_timeout_or_raise(procs, kind, process_timeout_sec)

    _run_jobs_in_rounds(color_jobs, "COLOR", _encode_color_camera_worker)
    _run_jobs_in_rounds(depth_jobs, "DEPTH", _encode_depth_camera_worker)

    # 清理整个临时目录
    if os.path.exists(temp_base_dir):
        shutil.rmtree(temp_base_dir)
        print("[VIDEO] ========== 所有视频编码完成，临时目录已清理 ==========")
        print(f"[VIDEO] 视频保存位置: {video_output_dir}/{uuid}")
