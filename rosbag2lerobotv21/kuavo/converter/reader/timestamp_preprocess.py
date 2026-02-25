"""Timestamp preprocessing helpers: dedup, gap checks, interpolation, nearest-index."""

from __future__ import annotations

from typing import Callable

import numpy as np


def find_closest_indices_vectorized(timestamps, target_timestamps):
    """向量化查找最近时间戳索引"""
    timestamps = np.array(timestamps)
    target_timestamps = np.array(target_timestamps)

    indices = np.searchsorted(timestamps, target_timestamps)
    indices = np.clip(indices, 0, len(timestamps) - 1)

    valid_left = indices > 0
    left_indices = np.where(valid_left, indices - 1, indices)

    left_diffs = np.abs(timestamps[left_indices] - target_timestamps)
    right_diffs = np.abs(timestamps[indices] - target_timestamps)
    closer_indices = np.where(left_diffs < right_diffs, left_indices, indices)
    return closer_indices


def remove_duplicate_timestamps(data_list: list, key: str) -> list:
    """去除重复时间戳及对应数据（使用纳秒精度）"""
    if len(data_list) <= 1:
        return data_list

    deduplicated = []
    seen_timestamps = set()
    duplicate_count = 0

    for item in data_list:
        timestamp_seconds = item["timestamp"]
        timestamp_ns = int(timestamp_seconds * 1e9)

        if timestamp_ns not in seen_timestamps:
            seen_timestamps.add(timestamp_ns)
            deduplicated.append(item)
        else:
            duplicate_count += 1

    if duplicate_count > 0:
        print(f"  {key}: 删除 {duplicate_count} 个重复时间戳")

    return deduplicated


def check_actual_time_gaps(
    data_list: list,
    key: str,
    *,
    max_gap_duration: float,
    error_cls: type[Exception],
) -> None:
    """检测去重后数据的实际时间间隔卡顿"""
    if len(data_list) <= 1:
        return

    timestamps_seconds = np.array([item["timestamp"] for item in data_list])
    timestamps_ns = (timestamps_seconds * 1e9).astype(np.int64)
    time_diffs_ns = np.diff(timestamps_ns)
    time_diffs_seconds = time_diffs_ns / 1e9
    large_gaps = time_diffs_seconds > max_gap_duration

    if np.any(large_gaps):
        max_gap_seconds = np.max(time_diffs_seconds)
        gap_indices = np.where(large_gaps)[0]

        error_msg = (
            f"时间间隔卡顿检测：{key} 话题存在 {len(gap_indices)} 个超过{max_gap_duration}s的时间间隔，"
            f"最大间隔 {max_gap_seconds:.3f}s，数据质量异常，终止处理"
        )
        print(f"[ERROR] {error_msg}")

        for i, gap_idx in enumerate(gap_indices[:3]):
            start_time = timestamps_seconds[gap_idx]
            end_time = timestamps_seconds[gap_idx + 1]
            gap_duration = time_diffs_seconds[gap_idx]
            print(f"  间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={gap_duration:.3f}s")

        raise error_cls(
            message=error_msg,
            topic=key,
            stuck_timestamp=timestamps_seconds[gap_indices[0]],
            stuck_duration=max_gap_seconds,
            stuck_frame_count=len(gap_indices),
            threshold=max_gap_duration,
        )

    max_gap_seconds = np.max(time_diffs_seconds) if len(time_diffs_seconds) > 0 else 0
    print(f"  {key}: ✓ 时间间隔正常，最大间隔 {max_gap_seconds:.3f}s")


def interpolate_timestamps_and_data(
    data_list: list,
    key: str,
    *,
    time_tolerance: float,
    create_interpolated_data_point: Callable[[dict, float, str], dict],
    error_cls: type[Exception],
) -> list:
    """时间戳插值和数据填充（严格时间间隔检查版）"""
    if len(data_list) <= 1:
        return data_list

    timestamps_seconds = np.array([item["timestamp"] for item in data_list])
    timestamps_ns = (timestamps_seconds * 1e9).astype(np.int64)
    time_diffs_ns = np.diff(timestamps_ns)
    time_diffs_seconds = time_diffs_ns / 1e9

    max_gap_seconds = np.max(time_diffs_seconds)
    large_gaps_2s = time_diffs_seconds > time_tolerance

    if np.any(large_gaps_2s):
        gap_indices = np.where(large_gaps_2s)[0]
        error_msg = (
            f"插值阶段发现严重时间间隔：{key} 话题存在 {len(gap_indices)} 个超过{time_tolerance}s的时间间隔，"
            f"最大间隔 {max_gap_seconds:.3f}s，数据质量异常，终止处理"
        )
        print(f"[ERROR] {error_msg}")

        for i, gap_idx in enumerate(gap_indices[:3]):
            start_time = timestamps_seconds[gap_idx]
            end_time = timestamps_seconds[gap_idx + 1]
            gap_duration = time_diffs_seconds[gap_idx]
            print(f"  严重间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={gap_duration:.3f}s")

        raise error_cls(
            message=error_msg,
            topic=key,
            stuck_timestamp=timestamps_seconds[gap_indices[0]],
            stuck_duration=max_gap_seconds,
            stuck_frame_count=len(gap_indices),
            threshold=2.0,
        )

    if any(cam in key for cam in ["top"]) and "depth" not in key:
        target_interval_ns = int(32 * 1e6)
        max_allowed_interval_ns = int(39.8 * 1e6)
        data_type = "video"
    elif any(cam in key for cam in ["wrist"]) and "depth" not in key:
        target_interval_ns = int(32 * 1e6)
        max_allowed_interval_ns = int(8 * 1e6)
        data_type = "video"
    elif "depth" in key:
        target_interval_ns = int(32 * 1e6)
        max_allowed_interval_ns = int(8 * 1e6)
        data_type = "depth"
    else:
        target_interval_ns = int(10 * 1e6)
        max_allowed_interval_ns = int(4 * 1e6)
        data_type = "sensor"

    print(f"  {key}: 目标间隔 {target_interval_ns/1e6:.1f}ms, 最大允许间隔 {max_allowed_interval_ns/1e6:.1f}ms")
    interpolation_threshold_ns = max_allowed_interval_ns
    large_gaps = time_diffs_ns > interpolation_threshold_ns

    if not np.any(large_gaps):
        print(f"  {key}: 无需插值，最大间隔 {np.max(time_diffs_ns)/1e6:.1f}ms")
        return data_list

    print(f"  {key}: 发现 {np.sum(large_gaps)} 个需要插值的时间间隔")
    print(f"  {key}: 目标间隔 {target_interval_ns/1e6:.1f}ms, 最大允许间隔 {max_allowed_interval_ns/1e6:.1f}ms")

    interpolated_data = []
    for i in range(len(data_list)):
        interpolated_data.append(data_list[i])
        if i >= len(data_list) - 1 or not large_gaps[i]:
            continue

        current_time_ns = timestamps_ns[i]
        next_time_ns = timestamps_ns[i + 1]
        gap_duration_ns = next_time_ns - current_time_ns
        gap_duration_seconds = gap_duration_ns / 1e9

        if gap_duration_seconds > time_tolerance:
            error_msg = (
                f"插值过程中发现超过{time_tolerance}秒的间隔：{key} 在索引{i}处有{gap_duration_seconds:.3f}s间隔"
            )
            print(f"[ERROR] {error_msg}")
            raise error_cls(
                message=error_msg,
                topic=key,
                stuck_timestamp=current_time_ns / 1e9,
                stuck_duration=gap_duration_seconds,
                stuck_frame_count=1,
                threshold=2.0,
            )

        num_segments_needed = int(np.ceil(gap_duration_ns / max_allowed_interval_ns))
        if num_segments_needed <= 1:
            continue

        num_interpolations = num_segments_needed - 1
        interp_times_ns = np.linspace(
            current_time_ns,
            next_time_ns,
            num_interpolations + 2,
            dtype=np.int64,
        )[1:-1]

        for interp_time_ns in interp_times_ns:
            interp_time_seconds = interp_time_ns / 1e9
            interpolated_item = create_interpolated_data_point(
                data_list[i], interp_time_seconds, data_type
            )
            interpolated_data.append(interpolated_item)

    final_timestamps = np.array([item["timestamp"] for item in interpolated_data])
    final_timestamps_ns = (final_timestamps * 1e9).astype(np.int64)
    final_intervals_ns = np.diff(final_timestamps_ns)
    final_intervals_ms = final_intervals_ns / 1e6
    max_final_interval = np.max(final_intervals_ms)
    print(f"  {key}: 插值完成，最大间隔 {max_final_interval:.1f}ms")

    return interpolated_data


def preprocess_timestamps_only_deduplicate(
    data: dict,
    *,
    time_tolerance: float,
    error_cls: type[Exception],
) -> dict:
    """预处理时间戳和数据：只去重和检测卡顿，不插值（按需插值策略）"""
    preprocessed_data = {}

    for key, data_list in data.items():
        if len(data_list) == 0:
            preprocessed_data[key] = []
            continue

        print(f"预处理 {key}: 原始长度 {len(data_list)}")
        deduplicated_data = remove_duplicate_timestamps(data_list, key)
        check_actual_time_gaps(
            deduplicated_data,
            key,
            max_gap_duration=time_tolerance,
            error_cls=error_cls,
        )
        preprocessed_data[key] = deduplicated_data
        print(f"预处理 {key}: 去重后 {len(deduplicated_data)} 帧（未插值）")

    return preprocessed_data


def preprocess_timestamps_and_data(
    data: dict,
    *,
    time_tolerance: float,
    create_interpolated_data_point: Callable[[dict, float, str], dict],
    error_cls: type[Exception],
) -> dict:
    """预处理时间戳和数据：去重、检测实际卡顿、插值"""
    preprocessed_data = {}

    for key, data_list in data.items():
        if len(data_list) == 0:
            preprocessed_data[key] = []
            continue

        print(f"预处理 {key}: 原始长度 {len(data_list)}")
        deduplicated_data = remove_duplicate_timestamps(data_list, key)
        check_actual_time_gaps(
            deduplicated_data,
            key,
            max_gap_duration=time_tolerance,
            error_cls=error_cls,
        )
        interpolated_data = interpolate_timestamps_and_data(
            deduplicated_data,
            key,
            time_tolerance=time_tolerance,
            create_interpolated_data_point=create_interpolated_data_point,
            error_cls=error_cls,
        )
        preprocessed_data[key] = interpolated_data
        print(f"预处理 {key}: 去重后 {len(deduplicated_data)}, 插值后 {len(interpolated_data)}")

    return preprocessed_data

