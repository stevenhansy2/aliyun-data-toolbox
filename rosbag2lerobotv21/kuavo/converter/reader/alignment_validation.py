"""Validation helpers for aligned multimodal timestamps."""

from __future__ import annotations

import numpy as np


def final_alignment_validation(
    aligned_data: dict,
    main_timestamps: np.ndarray,
    *,
    error_cls: type[Exception],
) -> None:
    """Validate aligned data quality and raise ``error_cls`` on hard failures."""
    print("  验证0: 主时间戳基本要求检查")

    min_required_frames = 300
    if len(main_timestamps) < min_required_frames:
        error_msg = f"主时间戳长度 {len(main_timestamps)} 小于最低要求 {min_required_frames} 帧"
        print(f"    ❌ {error_msg}")
        raise error_cls(
            message=f"数据长度不足: {error_msg}",
            topic="main_timeline_length",
            stuck_timestamp=(main_timestamps[0] if len(main_timestamps) > 0 else None),
            stuck_duration=None,
            stuck_frame_count=len(main_timestamps),
            threshold=min_required_frames,
        )
    print(f"    ✓ 主时间戳长度验证通过: {len(main_timestamps)} 帧 (>= {min_required_frames})")

    min_required_duration = 10.0
    if len(main_timestamps) > 1:
        time_span = main_timestamps[-1] - main_timestamps[0]
        if time_span < min_required_duration:
            error_msg = f"主时间戳时间跨度 {time_span:.3f}s 小于最低要求 {min_required_duration}s"
            print(f"    ❌ {error_msg}")
            raise error_cls(
                message=f"数据时间跨度不足: {error_msg}",
                topic="main_timeline_duration",
                stuck_timestamp=main_timestamps[0],
                stuck_duration=time_span,
                stuck_frame_count=len(main_timestamps),
                threshold=min_required_duration,
            )
        print(f"    ✓ 主时间戳时间跨度验证通过: {time_span:.3f}s (>= {min_required_duration}s)")

        actual_fps = len(main_timestamps) / time_span
        max_required_fps = 30.095
        min_required_fps = 29.905
        if actual_fps > max_required_fps:
            error_msg = f"主时间戳频率 {actual_fps:.2f}Hz 大于最大要求 {max_required_fps}Hz"
            print(f"    ❌ {error_msg}")
            raise error_cls(
                message=f"数据频率过大: {error_msg}",
                topic="main_timeline_fps",
                stuck_timestamp=main_timestamps[0],
                stuck_duration=time_span,
                stuck_frame_count=len(main_timestamps),
                threshold=max_required_fps,
            )
        if actual_fps < min_required_fps:
            error_msg = f"主时间戳频率 {actual_fps:.2f}Hz 小于最低要求 {min_required_fps}Hz"
            print(f"    ❌ {error_msg}")
            raise error_cls(
                message=f"数据频率不足: {error_msg}",
                topic="main_timeline_fps",
                stuck_timestamp=main_timestamps[0],
                stuck_duration=time_span,
                stuck_frame_count=len(main_timestamps),
                threshold=min_required_fps,
            )
        print(
            f"    ✓ 主时间戳频率验证通过: {min_required_fps}Hz <={actual_fps:.2f}Hz (<= {max_required_fps}Hz)"
        )
    else:
        error_msg = f"主时间戳只有 {len(main_timestamps)} 个，无法计算时间跨度和频率"
        print(f"    ❌ {error_msg}")
        raise error_cls(
            message=f"数据不足以计算时间跨度和频率: {error_msg}",
            topic="main_timeline_duration",
            stuck_timestamp=(main_timestamps[0] if len(main_timestamps) > 0 else None),
            stuck_duration=0,
            stuck_frame_count=len(main_timestamps),
            threshold=min_required_duration,
        )

    print("  验证1: 主时间戳间隔检查")
    if len(main_timestamps) > 1:
        main_timestamps_ns = (main_timestamps * 1e9).astype(np.int64)
        main_intervals_ns = np.diff(main_timestamps_ns)
        main_intervals_ms = main_intervals_ns / 1e6
        max_main_interval = np.max(main_intervals_ms)
        if max_main_interval > 40:
            error_msg = f"主时间戳最大间隔 {max_main_interval:.1f}ms 超过40ms阈值"
            print(f"    ❌ {error_msg}")
            large_interval_indices = np.where(main_intervals_ms > 40)[0]
            print("    主时间戳大间隔详情:")
            for idx in large_interval_indices[:3]:
                print(
                    f"      间隔{idx}: {main_intervals_ms[idx]:.1f}ms "
                    f"({main_timestamps[idx]:.6f}s -> {main_timestamps[idx+1]:.6f}s)"
                )
            raise error_cls(
                message=f"时间戳间隔验证失败: {error_msg}",
                topic="main_timeline",
                stuck_timestamp=None,
                stuck_duration=max_main_interval / 1000,
                stuck_frame_count=None,
                threshold=0.04,
            )
        print(f"    ✓ 主时间戳间隔验证通过: 最大间隔 {max_main_interval:.1f}ms")

    valid_modalities = {}
    for key, data_list in aligned_data.items():
        if (
            len(data_list) == 0
            or key.endswith("_extrinsics")
            or key.endswith("_camera_info")
            or key.startswith("end_")
        ):
            continue
        aligned_timestamps = np.array([item["timestamp"] for item in data_list])
        if len(aligned_timestamps) != len(main_timestamps):
            print(f"  ❌ {key}: 长度不匹配 ({len(aligned_timestamps)} vs {len(main_timestamps)})")
            continue
        valid_modalities[key] = aligned_timestamps

    if not valid_modalities:
        print("  ⚠️ 没有找到有效的数据模态进行验证")
        return

    print(f"  开始验证 {len(valid_modalities)} 个数据模态的时间戳同步...")
    print("  验证2: 每个时刻多模态间的时间戳差值")
    frame_sync_errors = []
    all_timestamps_ns = {
        key: (timestamps * 1e9).astype(np.int64) for key, timestamps in valid_modalities.items()
    }
    max_frame_spread = 0
    worst_frame_idx = -1
    for frame_idx in range(len(main_timestamps)):
        frame_timestamps_ns = []
        frame_keys = []
        for key, timestamps_ns in all_timestamps_ns.items():
            if frame_idx < len(timestamps_ns):
                frame_timestamps_ns.append(timestamps_ns[frame_idx])
                frame_keys.append(key)
        if len(frame_timestamps_ns) <= 1:
            continue
        min_ts_ns = np.min(frame_timestamps_ns)
        max_ts_ns = np.max(frame_timestamps_ns)
        spread_ms = (max_ts_ns - min_ts_ns) / 1e6
        if spread_ms > max_frame_spread:
            max_frame_spread = spread_ms
            worst_frame_idx = frame_idx
        if spread_ms > 20:
            frame_sync_errors.append(
                {
                    "frame_idx": frame_idx,
                    "spread_ms": spread_ms,
                    "timestamps": frame_timestamps_ns,
                    "keys": frame_keys,
                }
            )

    if frame_sync_errors:
        error_msg = f"发现 {len(frame_sync_errors)} 个时刻的多模态时间戳差值超过20ms阈值"
        print(f"    ❌ {error_msg}")
        sorted_errors = sorted(frame_sync_errors, key=lambda x: x["spread_ms"], reverse=True)
        for error in sorted_errors[:3]:
            frame_idx = error["frame_idx"]
            spread_ms = error["spread_ms"]
            timestamps_s = [ts_ns / 1e9 for ts_ns in error["timestamps"]]
            keys = error["keys"]
            print(f"      时刻{frame_idx}: 时间戳差值 {spread_ms:.1f}ms")
            for key, ts_s in zip(keys, timestamps_s):
                print(f"        {key}: {ts_s:.6f}s")
        detailed_msg = (
            "严格对齐验证失败:\n"
            f"- 最大帧内时间戳差值: {max_frame_spread:.1f}ms\n"
            f"- 验证错误: 多模态同步: {error_msg}，最大差值 {max_frame_spread:.1f}ms\n"
            f"- 参与验证的数据模态数: {len(valid_modalities)}\n"
            f"- 主时间戳长度: {len(main_timestamps)}"
        )
        print(f"[ERROR] {detailed_msg}")
        raise error_cls(
            message=f"严格对齐验证失败: 多模态同步: {error_msg}，最大差值 {max_frame_spread:.1f}ms",
            topic="strict_alignment_validation",
            stuck_timestamp=(main_timestamps[worst_frame_idx] if worst_frame_idx >= 0 else None),
            stuck_duration=max_frame_spread / 1000,
            stuck_frame_count=1,
            threshold=0.02,
        )

    print(f"    ✓ 多模态时间戳同步验证通过，最大差值 {max_frame_spread:.1f}ms")
    print("=" * 60)
    print("✓ 严格对齐验证通过!")
    time_span = main_timestamps[-1] - main_timestamps[0] if len(main_timestamps) > 1 else 0
    actual_fps = len(main_timestamps) / time_span if time_span > 0 else 0
    print(f"  - 主时间戳长度: {len(main_timestamps)} 帧 (要求 >= {min_required_frames} )")
    print(f"  - 主时间戳时间跨度: {time_span:.3f}s (要求 >= {min_required_duration}s)")
    print(f"  - 主时间戳频率: {actual_fps:.2f}Hz (要求 29.95~30.05Hz)")
    extrinsics_modalities = [
        k for k in aligned_data.keys() if k.endswith("_extrinsics") and len(aligned_data[k]) > 0
    ]
    camera_info_modalities = [
        k for k in aligned_data.keys() if k.endswith("_camera_info") and len(aligned_data[k]) > 0
    ]
    print(f"  - 参与验证的数据模态数: {len(valid_modalities)}")
    print(f"  - 跳过验证的外参模态数: {len(extrinsics_modalities)}")
    print(f"  - 跳过验证的相机信息模态数: {len(camera_info_modalities)}")
    print(f"  - 最大帧内时间戳差值: {max_frame_spread:.1f}ms")
    if len(main_timestamps) > 1:
        print(f"  - 主时间戳平均间隔: {np.mean(main_intervals_ms):.1f}ms")
        print(f"  - 主时间戳最大间隔: {max_main_interval:.1f}ms")
    if extrinsics_modalities:
        print(f"  - 外参模态: {extrinsics_modalities}")
    if camera_info_modalities:
        print(f"  - 相机信息模态: {camera_info_modalities}")
    print("=" * 60)

