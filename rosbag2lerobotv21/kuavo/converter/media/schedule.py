"""Shared concurrency scheduling helpers for the conversion pipeline."""

from dataclasses import dataclass
import math
import os


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_cpu_set(cpu_set: str | None) -> int | None:
    if not cpu_set:
        return None

    count = 0
    for part in cpu_set.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                return None
            if end < start:
                return None
            count += end - start + 1
            continue
        try:
            int(part)
        except ValueError:
            return None
        count += 1
    return count or None


def _detect_affinity_cores() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _detect_cgroup_quota_cores() -> int | None:
    # cgroup v2
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        try:
            quota_str, period_str = cpu_max.split()
            if quota_str != "max":
                quota = int(quota_str)
                period = int(period_str)
                if quota > 0 and period > 0:
                    return max(1, math.ceil(quota / period))
        except (ValueError, TypeError):
            pass

    # cgroup v1
    quota_str = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_str = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_str and period_str:
        try:
            quota = int(quota_str)
            period = int(period_str)
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
        except ValueError:
            pass

    return None


def _detect_cpuset_cores() -> int | None:
    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "/sys/fs/cgroup/cpuset.cpus",
    ):
        parsed = _parse_cpu_set(_read_text(path))
        if parsed:
            return parsed
    return None


def _detect_container_cpu_limit() -> int:
    detected = [n for n in (
        os.cpu_count(),
        _detect_affinity_cores(),
        _detect_cgroup_quota_cores(),
        _detect_cpuset_cores(),
    ) if n and n > 0]
    return max(1, min(detected)) if detected else 1


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
        cores = _detect_container_cpu_limit()
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
