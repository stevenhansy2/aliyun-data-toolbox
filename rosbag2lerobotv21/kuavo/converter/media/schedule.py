"""Video scheduling helpers shared by lerobot conversion pipeline."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class VideoSchedule:
    cores: int
    pipeline_workers: int
    max_encode_processes: int
    queue_limit: int


_SCHEDULES = {
    1: VideoSchedule(
        cores=1, pipeline_workers=1, max_encode_processes=1, queue_limit=96
    ),
    2: VideoSchedule(
        cores=2, pipeline_workers=2, max_encode_processes=1, queue_limit=160
    ),
    4: VideoSchedule(
        cores=4, pipeline_workers=4, max_encode_processes=2, queue_limit=260
    ),
    8: VideoSchedule(
        cores=8, pipeline_workers=6, max_encode_processes=3, queue_limit=400
    ),
}


def _pick_bucket(cores: int) -> int:
    if cores >= 8:
        return 8
    if cores >= 4:
        return 4
    if cores >= 2:
        return 2
    return 1


def resolve_video_schedule(raw_config=None, queue_limit_override: int | None = None) -> VideoSchedule:
    """
    Resolve schedule from:
    1) KUAVO_SCHED_CORES env
    2) config.schedule_cores
    3) host cpu count
    """
    cfg_cores = int(getattr(raw_config, "schedule_cores", 0) or 0) if raw_config is not None else 0
    env_cores = int(os.getenv("KUAVO_SCHED_CORES", "0") or 0)
    req = env_cores if env_cores > 0 else cfg_cores

    if req <= 0:
        req = os.cpu_count() or 1
    bucket = _pick_bucket(req)
    base = _SCHEDULES[bucket]
    if queue_limit_override is None:
        return base
    return VideoSchedule(
        cores=base.cores,
        pipeline_workers=base.pipeline_workers,
        max_encode_processes=base.max_encode_processes,
        queue_limit=queue_limit_override,
    )


def resolve_video_process_timeout_sec(raw_config=None) -> int:
    env_val = os.getenv("VIDEO_PROCESS_TIMEOUT_SEC")
    if env_val:
        try:
            return max(1, int(env_val))
        except Exception:
            pass
    cfg_val = int(getattr(raw_config, "video_process_timeout_sec", 600) or 600) if raw_config is not None else 600
    return max(1, cfg_val)
