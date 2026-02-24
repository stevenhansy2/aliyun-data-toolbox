"""On-demand interpolation helpers for multimodal alignment."""

from __future__ import annotations

import time as time_module

import numpy as np


def create_interpolated_data_point(
    reference_item: dict, new_timestamp: float, data_type: str
) -> dict:
    """创建插值数据点"""
    interpolated_item = reference_item.copy()
    interpolated_item["timestamp"] = new_timestamp

    if data_type in ["video", "depth"]:
        interpolated_item["interpolated"] = True
    elif data_type == "sensor":
        interpolated_item["interpolated"] = True

    return interpolated_item


def interpolate_on_demand(
    aligned_data: list,
    time_errors_ms: np.ndarray,
    original_data_list: list,
    original_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    key: str,
) -> list:
    """
    按需插值：只对误差 >10ms 的帧进行插值修正（向量化版本）
    """
    start_time = time_module.time()

    data_type = (
        "depth"
        if "depth" in key
        else ("video" if any(cam in key for cam in ["head_cam", "wrist_cam"]) else "sensor")
    )

    need_interp_mask = time_errors_ms > 10
    need_interp_indices = np.where(need_interp_mask)[0]

    if len(need_interp_indices) == 0:
        print("    [按需插值] 无需插值，所有帧误差 ≤10ms")
        return aligned_data

    target_ts_subset = target_timestamps[need_interp_indices]
    all_diffs_ms = np.abs(original_timestamps[None, :] - target_ts_subset[:, None]) * 1000
    min_diffs = np.min(all_diffs_ms, axis=1)
    argmin_diffs = np.argmin(all_diffs_ms, axis=1)
    can_use_original = min_diffs < 10

    interpolated_count = 0
    created_count = 0

    for local_idx, global_idx in enumerate(need_interp_indices):
        if can_use_original[local_idx]:
            best_orig_idx = argmin_diffs[local_idx]
            aligned_data[global_idx] = original_data_list[best_orig_idx]
            interpolated_count += 1
        else:
            closest_idx = argmin_diffs[local_idx]
            reference_frame = original_data_list[closest_idx]
            target_ts = target_timestamps[global_idx]
            interpolated_frame = create_interpolated_data_point(
                reference_frame, target_ts, data_type
            )
            aligned_data[global_idx] = interpolated_frame
            created_count += 1

    elapsed_ms = (time_module.time() - start_time) * 1000

    if interpolated_count > 0:
        print(f"    [按需插值] 从原始数据选择了 {interpolated_count} 帧")
    if created_count > 0:
        print(f"    [按需插值] 创建了 {created_count} 个插值帧（复制最近帧）")

    total_processed = interpolated_count + created_count
    print(
        f"    [按需插值-验证] 需插值: {len(need_interp_indices)}, 已处理: {total_processed}, 耗时: {elapsed_ms:.1f}ms"
    )

    return aligned_data

