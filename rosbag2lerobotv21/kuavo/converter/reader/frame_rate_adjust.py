"""Frame-rate adjustment helpers for aligned multimodal timestamp sequences."""

from __future__ import annotations

from typing import Callable

import numpy as np


def adjust_frame_rate_to_30fps(
    aligned_data: dict,
    main_timestamps: np.ndarray,
    *,
    insert_fn: Callable[[np.ndarray, dict, float, float], tuple[np.ndarray, dict]],
    remove_fn: Callable[[np.ndarray, dict, float, float], tuple[np.ndarray, dict]],
) -> tuple[dict, np.ndarray]:
    """Adjust frame rate to target range around 30fps by insert/remove strategy."""
    print("=" * 60)
    print("开始帧率调整到30fps...")

    if len(main_timestamps) < 2:
        print("  ⚠️ 时间戳数量不足，跳过帧率调整")
        return aligned_data, main_timestamps

    time_span = main_timestamps[-1] - main_timestamps[0]
    current_fps = len(main_timestamps) / time_span
    target_fps_min = 29.905
    target_fps_max = 30.095

    print(f"  当前帧率: {current_fps:.2f}Hz")
    print(f"  目标范围: {target_fps_min:.2f}-{target_fps_max:.2f}Hz")
    print(f"  当前帧数: {len(main_timestamps)}")
    print(f"  时间跨度: {time_span:.3f}s")

    if target_fps_min <= current_fps <= target_fps_max:
        print("  ✓ 帧率已在目标范围内，无需调整")
        return aligned_data, main_timestamps

    main_timestamps = np.array(main_timestamps)
    valid_modalities = {k: list(v) for k, v in aligned_data.items() if len(v) > 0}

    if current_fps < target_fps_min:
        print("  帧率过低，开始插帧...")
        main_timestamps, valid_modalities = insert_fn(
            main_timestamps, valid_modalities, target_fps_min, time_span
        )
    elif current_fps > target_fps_max:
        print("  帧率过高，开始抽帧...")
        main_timestamps, valid_modalities = remove_fn(
            main_timestamps, valid_modalities, target_fps_max, time_span
        )

    final_time_span = main_timestamps[-1] - main_timestamps[0]
    final_fps = len(main_timestamps) / final_time_span

    print(f"  调整后帧率: {final_fps:.2f}Hz")
    print(f"  调整后帧数: {len(main_timestamps)}")
    print(f"  调整后时间跨度: {final_time_span:.3f}s")
    if target_fps_min <= final_fps <= target_fps_max:
        print("  ✓ 帧率调整成功！")
    else:
        print("  ⚠️ 帧率调整后仍不在目标范围内")

    aligned_data_adjusted = dict(valid_modalities)
    for key, data_list in aligned_data.items():
        if key not in aligned_data_adjusted:
            aligned_data_adjusted[key] = []

    print("=" * 60)
    return aligned_data_adjusted, main_timestamps


def insert_frames_to_increase_fps(
    main_timestamps: np.ndarray,
    valid_modalities: dict,
    target_fps: float,
    time_span: float,
) -> tuple[np.ndarray, dict]:
    """Increase fps by inserting frames at largest intervals."""
    target_frame_count = int(time_span * target_fps)
    current_frame_count = len(main_timestamps)
    frames_to_insert = target_frame_count - current_frame_count

    print(f"    需要插入 {frames_to_insert} 帧")
    if frames_to_insert <= 0:
        return main_timestamps, valid_modalities

    insertion_threshold_ms = 33.0
    inserted_count = 0
    max_iterations = frames_to_insert * 2
    iteration = 0

    while inserted_count < frames_to_insert and iteration < max_iterations:
        iteration += 1
        time_intervals = np.diff(main_timestamps) * 1000
        max_interval_idx = np.argmax(time_intervals)
        max_interval_ms = time_intervals[max_interval_idx]

        if max_interval_ms <= insertion_threshold_ms:
            print(f"    无法找到超过{insertion_threshold_ms}ms的间隔进行插帧")
            break

        insert_pos = max_interval_idx + 1
        prev_timestamp = main_timestamps[max_interval_idx]
        next_timestamp = main_timestamps[max_interval_idx + 1]
        new_timestamp = (prev_timestamp + next_timestamp) / 2
        main_timestamps = np.insert(main_timestamps, insert_pos, new_timestamp)

        for _, data_list in valid_modalities.items():
            if insert_pos <= len(data_list):
                reference_frame = data_list[max_interval_idx].copy()
                reference_frame["timestamp"] = new_timestamp
                reference_frame["frame_inserted"] = True
                data_list.insert(insert_pos, reference_frame)

        inserted_count += 1
        if inserted_count % 10 == 0:
            current_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])
            print(f"    已插入 {inserted_count} 帧，当前帧率: {current_fps:.2f}Hz")

    print(f"    实际插入了 {inserted_count} 帧")
    return main_timestamps, valid_modalities


def remove_frames_to_decrease_fps(
    main_timestamps: np.ndarray,
    valid_modalities: dict,
    target_fps: float,
    time_span: float,
    *,
    reaverage_fn: Callable[[np.ndarray, float, float], np.ndarray],
    execute_window_fn: Callable[[np.ndarray, dict, dict], tuple[np.ndarray, bool]],
    error_cls: type[Exception],
) -> tuple[np.ndarray, dict]:
    """Lower fps with sliding-window removal and local timestamp re-averaging."""
    target_frame_count = int(time_span * target_fps)
    current_frame_count = len(main_timestamps)
    frames_to_remove = current_frame_count - target_frame_count

    print(
        f"    需要删除 {frames_to_remove} 帧 (从 {current_frame_count} 帧降到 {target_frame_count} 帧)"
    )
    if frames_to_remove <= 0:
        return main_timestamps, valid_modalities

    removed_count = 0
    max_iterations = frames_to_remove * 3
    iteration = 0

    window_size = 5
    max_window_size = 15
    max_interval_threshold_ms = 40.0
    print(f"    使用滑动窗口删除+重新平均算法，初始窗口大小: {window_size}")

    while removed_count < frames_to_remove and iteration < max_iterations:
        iteration += 1
        if len(main_timestamps) <= window_size + 2:
            print(f"    剩余帧数过少({len(main_timestamps)})，无法继续删除")
            break

        best_candidate = None
        best_score = float("inf")
        candidates_found = 0

        for start_idx in range(len(main_timestamps) - window_size + 1):
            end_idx = start_idx + window_size
            center_idx = start_idx + window_size // 2
            if center_idx <= 1 or center_idx >= len(main_timestamps) - 2:
                continue

            window_timestamps = main_timestamps[start_idx:end_idx]
            timestamps_after_removal = np.concatenate(
                [
                    window_timestamps[: window_size // 2],
                    window_timestamps[window_size // 2 + 1 :],
                ]
            )
            reaveraged_timestamps = reaverage_fn(
                timestamps_after_removal, window_timestamps[0], window_timestamps[-1]
            )

            if len(reaveraged_timestamps) > 1:
                reaveraged_intervals_ms = np.diff(reaveraged_timestamps) * 1000
                max_reaveraged_interval = np.max(reaveraged_intervals_ms)
                if max_reaveraged_interval <= max_interval_threshold_ms:
                    candidates_found += 1
                    interval_score = max_reaveraged_interval
                    uniformity_score = np.std(reaveraged_intervals_ms) * 2
                    density_score = -np.mean(np.diff(window_timestamps) * 1000)
                    total_score = interval_score + uniformity_score + density_score
                    if total_score < best_score:
                        best_score = total_score
                        best_candidate = {
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "remove_idx": center_idx,
                            "window_size": window_size,
                            "original_timestamps": window_timestamps,
                            "reaveraged_timestamps": reaveraged_timestamps,
                            "max_interval_after": max_reaveraged_interval,
                            "score": total_score,
                        }

        if best_candidate is not None:
            new_timestamps, success = execute_window_fn(
                main_timestamps, valid_modalities, best_candidate
            )
            if success:
                main_timestamps = new_timestamps
                removed_count += 1
                if removed_count % 10 == 0:
                    current_fps = len(main_timestamps) / (
                        main_timestamps[-1] - main_timestamps[0]
                    )
                    print(f"      已删除 {removed_count} 帧，当前帧率: {current_fps:.2f}Hz")
                    print(
                        f"      最新删除: 窗口{best_candidate['start_idx']}-{best_candidate['end_idx']}, "
                        f"删除后最大间隔: {best_candidate['max_interval_after']:.1f}ms"
                    )
            else:
                print("    执行删除失败，跳过此候选")
        else:
            print(f"    第{iteration}轮: 窗口大小{window_size}下找到 {candidates_found} 个候选")
            if window_size < max_window_size:
                window_size += 2
                print(f"    扩大窗口大小到: {window_size}，继续尝试")
                continue
            print(f"    窗口大小已达到最大({window_size})，无法找到更多可删除位置")
            break

    final_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])
    print("    删除完成统计:")
    print(f"      目标删除: {frames_to_remove} 帧")
    print(f"      实际删除: {removed_count} 帧")
    print(f"      最终帧率: {final_fps:.3f}Hz")

    if len(main_timestamps) > 1:
        final_intervals_ms = np.diff(main_timestamps) * 1000
        max_final_interval = np.max(final_intervals_ms)
        avg_final_interval = np.mean(final_intervals_ms)
        std_final_interval = np.std(final_intervals_ms)
        print("      最终时间戳质量:")
        print(f"        最大间隔: {max_final_interval:.1f}ms")
        print(f"        平均间隔: {avg_final_interval:.1f}ms")
        print(f"        间隔标准差: {std_final_interval:.1f}ms")

        large_intervals = final_intervals_ms > max_interval_threshold_ms
        if np.any(large_intervals):
            large_count = np.sum(large_intervals)
            worst_interval = np.max(final_intervals_ms)
            error_msg = (
                f"删除后验证失败：仍有 {large_count} 个间隔超过{max_interval_threshold_ms}ms，"
                f"最大间隔{worst_interval:.1f}ms"
            )
            print(f"        ❌ {error_msg}")
            problem_indices = np.where(large_intervals)[0]
            for i, idx in enumerate(problem_indices[:3]):
                interval_value = final_intervals_ms[idx]
                start_time = main_timestamps[idx]
                end_time = main_timestamps[idx + 1]
                print(
                    f"          问题间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={interval_value:.1f}ms"
                )
            raise error_cls(
                message=f"严格间隔验证失败: {error_msg}",
                topic="strict_interval_validation",
                stuck_timestamp=main_timestamps[problem_indices[0]],
                stuck_duration=worst_interval / 1000,
                stuck_frame_count=large_count,
                threshold=max_interval_threshold_ms / 1000,
            )
        print(f"        ✓ 所有间隔都在{max_interval_threshold_ms}ms以内")

    if removed_count < frames_to_remove:
        shortfall = frames_to_remove - removed_count
        error_msg = (
            f"删除未完成：需要删除 {frames_to_remove} 帧，实际删除 {removed_count} 帧，"
            f"还差 {shortfall} 帧。当前帧率: {final_fps:.3f}Hz，目标: ≤{target_fps:.3f}Hz"
        )
        print(f"    ❌ {error_msg}")
        raise error_cls(
            message=f"严格帧率调整失败: {error_msg}",
            topic="strict_frame_rate_adjustment",
            stuck_timestamp=None,
            stuck_duration=None,
            stuck_frame_count=shortfall,
            threshold=target_fps,
        )

    if final_fps > target_fps:
        fps_excess = final_fps - target_fps
        error_msg = (
            f"最终帧率验证失败：当前帧率 {final_fps:.3f}Hz 仍然超过目标 {target_fps:.3f}Hz，"
            f"超出 {fps_excess:.3f}Hz"
        )
        print(f"    ❌ {error_msg}")
        raise error_cls(
            message=f"严格帧率目标未达成: {error_msg}",
            topic="strict_fps_target",
            stuck_timestamp=None,
            stuck_duration=None,
            stuck_frame_count=frames_to_remove - removed_count,
            threshold=target_fps,
        )

    print("    ✓ 滑动窗口删除+重新平均成功完成")
    print(f"      最终帧率: {final_fps:.3f}Hz ≤ {target_fps:.3f}Hz")
    print(f"      最大时间间隔: {max_final_interval:.1f}ms ≤ {max_interval_threshold_ms}ms")
    return main_timestamps, valid_modalities

