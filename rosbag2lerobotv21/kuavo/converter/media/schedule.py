"""Shared concurrency scheduling helpers for the conversion pipeline."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RuntimeParallelism:
    cores: int
    pipeline_workers: int
    max_encode_processes: int
    queue_limit: int
    parallel_rosbag_workers: int
    image_writer_processes: int
    image_writer_threads: int
    codec_threads: int
    ffmpeg_threads: int


@dataclass(frozen=True)
class VideoSchedule:
    cores: int
    pipeline_workers: int
    max_encode_processes: int
    queue_limit: int


def resolve_assigned_cores(raw_config=None) -> int:
    """Resolve the CPU cores assigned to this job."""
    cfg_cores = (
        int(getattr(raw_config, "schedule_cores", 0) or 0)
        if raw_config is not None
        else 0
    )
    env_cores = int(os.getenv("KUAVO_SCHED_CORES", "0") or 0)
    cores = env_cores if env_cores > 0 else cfg_cores
    if cores <= 0:
        cores = os.cpu_count() or 1
    return max(1, cores)


def _split_image_writer_workers(cores: int) -> tuple[int, int]:
    """
    Keep total image writer concurrency roughly bounded by assigned cores.
    """
    processes = min(max(1, cores), 4)
    threads = max(1, cores // processes)
    return processes, threads


def resolve_runtime_parallelism(raw_config=None) -> RuntimeParallelism:
    cores = resolve_assigned_cores(raw_config)
    image_writer_processes, image_writer_threads = _split_image_writer_workers(cores)

    return RuntimeParallelism(
        cores=cores,
        pipeline_workers=cores,
        max_encode_processes=cores,
        queue_limit=max(96, cores * 80),
        parallel_rosbag_workers=cores,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        codec_threads=1,
        ffmpeg_threads=1,
    )


def resolve_video_schedule(
    raw_config=None, queue_limit_override: int | None = None
) -> VideoSchedule:
    parallelism = resolve_runtime_parallelism(raw_config)
    queue_limit = (
        parallelism.queue_limit
        if queue_limit_override is None
        else queue_limit_override
    )
    return VideoSchedule(
        cores=parallelism.cores,
        pipeline_workers=parallelism.pipeline_workers,
        max_encode_processes=parallelism.max_encode_processes,
        queue_limit=queue_limit,
    )


def resolve_video_process_timeout_sec(raw_config=None) -> int:
    env_val = os.getenv("VIDEO_PROCESS_TIMEOUT_SEC")
    if env_val:
        try:
            return max(1, int(env_val))
        except Exception:
            pass
    cfg_val = (
        int(getattr(raw_config, "video_process_timeout_sec", 600) or 600)
        if raw_config is not None
        else 600
    )
    return max(1, cfg_val)
