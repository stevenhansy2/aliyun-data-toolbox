"""Timestamp quality validation helpers."""

from __future__ import annotations

import numpy as np


def validate_timestamp_quality(
    timestamps: np.ndarray,
    data_name: str,
    *,
    error_cls: type[Exception],
) -> None:
    """验证时间戳质量（纳秒精度）并在关键错误时抛异常。"""
    if len(timestamps) <= 1:
        return

    timestamps_ns = (timestamps * 1e9).astype(np.int64)
    time_diffs_ns = np.diff(timestamps_ns)
    time_diffs_ms = time_diffs_ns / 1e6

    mean_interval_ms = np.mean(time_diffs_ms)
    max_interval_ms = np.max(time_diffs_ms)
    min_interval_ms = np.min(time_diffs_ms)
    std_interval_ms = np.std(time_diffs_ms)

    print(f"  {data_name} 时间戳质量:")
    print(f"    平均间隔: {mean_interval_ms:.1f}ms")
    print(f"    最大间隔: {max_interval_ms:.1f}ms")
    print(f"    最小间隔: {min_interval_ms:.1f}ms")
    print(f"    标准差: {std_interval_ms:.1f}ms")

    critical_errors = []
    warnings = []

    if max_interval_ms > 40:
        critical_errors.append(f"最大时间间隔过大: {max_interval_ms:.1f}ms")
    if min_interval_ms < 0.1:
        critical_errors.append(f"最小时间间隔过小: {min_interval_ms:.1f}ms")
    if std_interval_ms > 15:
        warnings.append(f"时间间隔波动过大: {std_interval_ms:.1f}ms")

    unique_timestamps = np.unique(timestamps_ns)
    if len(unique_timestamps) < len(timestamps_ns):
        duplicate_count = len(timestamps_ns) - len(unique_timestamps)
        critical_errors.append(f"仍存在 {duplicate_count} 个重复时间戳")

    if critical_errors:
        print(f"    ❌ 关键错误: {'; '.join(critical_errors)}")
        if data_name == "主时间戳":
            error_msg = f"{data_name} 存在关键质量问题: {'; '.join(critical_errors)}"
            raise error_cls(
                message=error_msg,
                topic=data_name,
                stuck_timestamp=None,
                stuck_duration=max_interval_ms / 1000,
                stuck_frame_count=len(critical_errors),
                threshold=0.04,
            )
    elif warnings:
        print(f"    ⚠️  警告: {'; '.join(warnings)}")
    else:
        print("    ✓ 时间戳质量良好")

