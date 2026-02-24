"""Post-alignment trimming and start-alignment correction helpers."""

from __future__ import annotations

import numpy as np


def trim_aligned_data_by_frames(
    aligned_data: dict,
    main_timestamps: np.ndarray,
    head_trim_frames: int,
    tail_trim_frames: int,
) -> tuple[dict, np.ndarray]:
    """按帧数裁剪对齐后的数据"""
    trimmed_data = {}
    original_length = len(main_timestamps)
    start_idx = max(0, head_trim_frames)
    end_idx = min(original_length, original_length - tail_trim_frames if tail_trim_frames > 0 else original_length)

    if start_idx < end_idx:
        trimmed_main_timestamps = main_timestamps[start_idx:end_idx]
    else:
        trimmed_main_timestamps = np.array([])
        print("    警告: 主时间戳裁剪后为空")

    for key, data_list in aligned_data.items():
        if isinstance(data_list, list) and len(data_list) > 0:
            original_data_length = len(data_list)
            if start_idx < end_idx and start_idx < original_data_length:
                actual_end_idx = min(end_idx, original_data_length)
                trimmed_data[key] = data_list[start_idx:actual_end_idx]
                print(
                    f"    {key}: {original_data_length} -> {len(trimmed_data[key])} (-{original_data_length - len(trimmed_data[key])})"
                )
            else:
                trimmed_data[key] = []
                print(f"    警告: {key} 裁剪后为空")
        else:
            trimmed_data[key] = data_list

    return trimmed_data, trimmed_main_timestamps


def detect_and_trim_aligned_data(
    aligned_data: dict,
    main_timestamps: np.ndarray,
    *,
    action_config=None,
    default_camera_names: list[str],
    train_hz: int,
) -> tuple[dict, np.ndarray]:
    """
    检测并裁剪对齐后数据中的静止区域，头尾裁剪上限由首尾动作持续帧数的一半决定。
    """
    from converter.slave_utils import detect_stillness_from_image_data

    motion_threshold = 4.5
    stillness_ratio = 1
    check_duration = 10.0
    fps = train_hz or 30

    camera_keys = [c for c in default_camera_names if c in aligned_data and len(aligned_data[c]) > 0]
    if not camera_keys:
        print("  未找到有效的相机数据，跳过静止检测")
        return aligned_data, main_timestamps

    print(f"  基于 {len(camera_keys)} 个相机检测静止区域: {camera_keys}")

    max_head_trim_limit = None
    max_tail_trim_limit = None
    total_frames = len(main_timestamps)
    if action_config and len(action_config) > 0:
        first_action = None
        last_action = None
        min_start = None
        max_end = None
        for act in action_config:
            custom_fields = act.get("customFieldValues", {})
            try:
                sp = float(custom_fields.get("start_position", None))
                if min_start is None or sp < min_start:
                    min_start = sp
                    first_action = act
            except Exception:
                pass
            try:
                ep = float(custom_fields.get("end_position", None))
                if max_end is None or ep > max_end:
                    max_end = ep
                    last_action = act
            except Exception:
                pass

        if first_action is not None and last_action is not None:
            first_sp = float(first_action["customFieldValues"]["start_position"])
            first_ep = float(first_action["customFieldValues"]["end_position"])
            last_sp = float(last_action["customFieldValues"]["start_position"])
            last_ep = float(last_action["customFieldValues"]["end_position"])

            first_start_idx = int(round((first_sp - min_start) / (max_end - min_start) * (total_frames - 1)))
            first_end_idx = int(round((first_ep - min_start) / (max_end - min_start) * (total_frames - 1)))
            last_start_idx = int(round((last_sp - min_start) / (max_end - min_start) * (total_frames - 1)))
            last_end_idx = int(round((last_ep - min_start) / (max_end - min_start) * (total_frames - 1)))
            first_len = max(0, first_end_idx - first_start_idx)
            last_len = max(0, last_end_idx - last_start_idx)
            max_head_trim_limit = max(0, int(first_len / 2))
            max_tail_trim_limit = max(0, int(last_len / 2))
            print(f"  首动作帧区间: {first_start_idx}-{first_end_idx}，长度: {first_len}")
            print(f"  尾动作帧区间: {last_start_idx}-{last_end_idx}，长度: {last_len}")
            print(f"  首动作最大裁剪上限: {max_head_trim_limit} 帧，尾动作最大裁剪上限: {max_tail_trim_limit} 帧")
        else:
            print("  未找到有效的动作首尾裁剪上限")
    else:
        print("  未找到有效的动作首尾裁剪上限")

    all_stillness_results = {}
    for camera_key in camera_keys:
        frames_data = aligned_data[camera_key]
        print(f"  分析 {camera_key}: 总帧数 {len(frames_data)}")
        head_stillness, tail_stillness = detect_stillness_from_image_data(
            frames_data,
            camera_key,
            motion_threshold,
            stillness_ratio,
            check_duration,
            fps,
        )
        all_stillness_results[camera_key] = {"head_frames": head_stillness, "tail_frames": tail_stillness}
        print(f"    {camera_key}: 开头静止 {head_stillness} 帧, 结尾静止 {tail_stillness} 帧")

    if all_stillness_results:
        max_head_trim = max(result["head_frames"] for result in all_stillness_results.values())
        max_tail_trim = max(result["tail_frames"] for result in all_stillness_results.values())
    else:
        max_head_trim = 0
        max_tail_trim = 0

    if max_head_trim_limit is not None and max_head_trim > max_head_trim_limit:
        print(f"  开头静止裁剪帧数 {max_head_trim} 超过首动作上限 {max_head_trim_limit}，已覆盖")
        max_head_trim = max_head_trim_limit
    if max_tail_trim_limit is not None and max_tail_trim > max_tail_trim_limit:
        print(f"  结尾静止裁剪帧数 {max_tail_trim} 超过尾动作上限 {max_tail_trim_limit}，已覆盖")
        max_tail_trim = max_tail_trim_limit

    print(f"  最终裁剪决定: 开头 {max_head_trim} 帧, 结尾 {max_tail_trim} 帧")

    if max_head_trim > 0 or max_tail_trim > 0:
        trimmed_aligned_data, trimmed_main_timestamps = trim_aligned_data_by_frames(
            aligned_data, main_timestamps, max_head_trim, max_tail_trim
        )
        print(f"  裁剪完成: 主时间戳 {len(main_timestamps)} -> {len(trimmed_main_timestamps)} 帧")
        return trimmed_aligned_data, trimmed_main_timestamps

    print("  无需裁剪")
    return aligned_data, main_timestamps


def fix_severely_stuck_timestamps(
    preprocessed_data: dict,
    key: str,
    main_timestamps: np.ndarray,
    tolerance_ms: float = 20,
) -> None:
    """修复严重卡住的数据模态的时间戳"""
    print(f"  开始修复严重卡住的数据模态: {key}")

    data_list = preprocessed_data[key]
    if len(data_list) == 0:
        return

    original_timestamps = np.array([item["timestamp"] for item in data_list])
    normal_start_index = None
    for i in range(len(original_timestamps)):
        if i < len(main_timestamps):
            main_ts = main_timestamps[i]
            data_ts = original_timestamps[i]
            diff_ms = abs(data_ts - main_ts) * 1000
            if diff_ms <= tolerance_ms:
                normal_start_index = i
                print(f"    在索引 {i} 处找到正常时间戳，偏差 {diff_ms:.1f}ms")
                break

    if normal_start_index is None:
        print("    未找到正常时间戳，寻找数据开始变化的位置...")
        for i in range(1, min(len(original_timestamps), len(main_timestamps))):
            time_change = abs(original_timestamps[i] - original_timestamps[0])
            if time_change > 1.0:
                expected_main_ts = (
                    main_timestamps[i]
                    if i < len(main_timestamps)
                    else main_timestamps[-1] + (i - len(main_timestamps) + 1) * 0.033
                )
                diff_ms = abs(original_timestamps[i] - expected_main_ts) * 1000
                if diff_ms <= tolerance_ms * 5:
                    normal_start_index = i
                    print(f"    在索引 {i} 处找到数据变化点，开始正常同步，偏差 {diff_ms:.1f}ms")
                    break

    replaced_count = 0
    if normal_start_index is not None and normal_start_index > 0:
        for i in range(min(normal_start_index, len(main_timestamps))):
            if i < len(data_list):
                old_timestamp = data_list[i]["timestamp"]
                new_timestamp = main_timestamps[i]
                data_list[i]["timestamp"] = new_timestamp
                data_list[i]["timestamp_replaced"] = True
                data_list[i]["original_timestamp"] = old_timestamp
                replaced_count += 1
        print(f"    ✓ 替换了前 {replaced_count} 个时间戳")
        print(f"    从索引 {normal_start_index} 开始使用原始时间戳")
    else:
        max_replace_count = min(50, len(data_list), len(main_timestamps))
        print(f"    无法找到同步点，强制替换前 {max_replace_count} 个时间戳")
        for i in range(max_replace_count):
            if i < len(data_list):
                old_timestamp = data_list[i]["timestamp"]
                new_timestamp = main_timestamps[i]
                data_list[i]["timestamp"] = new_timestamp
                data_list[i]["timestamp_replaced"] = True
                data_list[i]["original_timestamp"] = old_timestamp
                replaced_count += 1
        print(f"    ⚠️ 强制替换了前 {replaced_count} 个时间戳")

    print(f"  验证 {key} 修复效果:")
    new_timestamps = np.array([item["timestamp"] for item in data_list])
    check_frames = min(5, len(main_timestamps), len(new_timestamps))
    if check_frames > 0:
        main_subset = main_timestamps[:check_frames]
        data_subset = new_timestamps[:check_frames]
        time_diffs_ms = np.abs(data_subset - main_subset) * 1000
        max_diff = np.max(time_diffs_ms)
        avg_diff = np.mean(time_diffs_ms)
        print(f"    修复后开头{check_frames}帧: 最大偏差 {max_diff:.1f}ms, 平均偏差 {avg_diff:.1f}ms")
        if max_diff <= tolerance_ms:
            print(f"    ✓ {key} 修复成功，开头偏差已控制在 {tolerance_ms}ms 内")
        else:
            print(f"    ⚠️ {key} 修复后仍有偏差，但已显著改善")

    preprocessed_data[key] = data_list


def fix_multimodal_start_alignment(
    main_timestamps: np.ndarray,
    preprocessed_data: dict,
    *,
    find_closest_indices_fn,
    fix_severely_stuck_fn,
) -> np.ndarray:
    """修正多模态开头时间戳偏差问题（与原逻辑一致）。"""
    if len(main_timestamps) == 0:
        return main_timestamps

    valid_keys = []
    for key in preprocessed_data.keys():
        if (
            not key.endswith("_extrinsics")
            and not key.endswith("_camera_info")
            and len(preprocessed_data[key]) > 0
        ):
            valid_keys.append(key)

    if len(valid_keys) <= 1:
        print("  ✓ 只有一个或零个数据模态，无需修正")
        return main_timestamps

    print(f"  检查 {len(valid_keys)} 个数据模态的开头对齐情况")

    max_alignment_tolerance_ms = 20
    severe_stuck_threshold_ms = 1000
    aligned_timestamps_by_key = {}
    for key in valid_keys:
        timestamps = np.array([item["timestamp"] for item in preprocessed_data[key]])
        check_frames = min(5, len(main_timestamps), len(timestamps))
        if check_frames == 0:
            continue
        main_subset = main_timestamps[:check_frames]
        closest_indices = find_closest_indices_fn(timestamps, main_subset)
        aligned_timestamps = timestamps[closest_indices]
        aligned_timestamps_by_key[key] = aligned_timestamps

    check_frames = min(5, len(main_timestamps))
    frame_spreads = []
    severely_stuck_keys = []

    for frame_idx in range(check_frames):
        frame_timestamps = []
        frame_keys = []
        for key in valid_keys:
            if key in aligned_timestamps_by_key and frame_idx < len(aligned_timestamps_by_key[key]):
                frame_timestamps.append(aligned_timestamps_by_key[key][frame_idx])
                frame_keys.append(key)
        if len(frame_timestamps) > 1:
            frame_timestamps = np.array(frame_timestamps)
            min_ts = np.min(frame_timestamps)
            max_ts = np.max(frame_timestamps)
            spread_ms = (max_ts - min_ts) * 1000
            frame_spreads.append(
                {
                    "frame_idx": frame_idx,
                    "spread_ms": spread_ms,
                    "timestamps": frame_timestamps,
                    "keys": frame_keys,
                    "main_timestamp": main_timestamps[frame_idx],
                }
            )

    if frame_spreads:
        max_spread = max(spread["spread_ms"] for spread in frame_spreads)
        avg_spread = np.mean([spread["spread_ms"] for spread in frame_spreads])
        print(f"    开头{len(frame_spreads)}帧模态间时间戳差值分析:")
        print(f"      最大差值: {max_spread:.1f}ms")
        print(f"      平均差值: {avg_spread:.1f}ms")
        for spread_info in frame_spreads:
            frame_idx = spread_info["frame_idx"]
            spread_ms = spread_info["spread_ms"]
            if spread_ms > max_alignment_tolerance_ms:
                print(f"      帧{frame_idx}: 差值 {spread_ms:.1f}ms (超过阈值)")
                timestamps = spread_info["timestamps"]
                keys = spread_info["keys"]
                main_ts = spread_info["main_timestamp"]
                for ts, key in zip(timestamps, keys):
                    diff_ms = abs(ts - main_ts) * 1000
                    if diff_ms > severe_stuck_threshold_ms and key not in severely_stuck_keys:
                        severely_stuck_keys.append(key)
                        print(f"        {key}: 与主时间戳偏差 {diff_ms:.1f}ms (严重卡住)")
            else:
                print(f"      帧{frame_idx}: 差值 {spread_ms:.1f}ms (正常)")

    problematic_frames = [s for s in frame_spreads if s["spread_ms"] > max_alignment_tolerance_ms]

    if severely_stuck_keys:
        print(f"  发现 {len(severely_stuck_keys)} 个严重卡住的数据模态，需要特殊处理:")
        severely_stuck_by_type = {}
        for key in severely_stuck_keys:
            if any(cam in key for cam in ["head_cam", "wrist_cam"]):
                modality_type = "相机"
            elif "action." in key:
                modality_type = "动作"
            elif "observation." in key:
                modality_type = "观测"
            else:
                modality_type = "其他"
            severely_stuck_by_type.setdefault(modality_type, []).append(key)
        print("  严重卡住模态分布:")
        for mod_type, keys in severely_stuck_by_type.items():
            print(f"    {mod_type}: {len(keys)} 个 - {keys}")
        for key in severely_stuck_keys:
            fix_severely_stuck_fn(preprocessed_data, key, main_timestamps, max_alignment_tolerance_ms)

    if not problematic_frames or len(severely_stuck_keys) == len(valid_keys):
        if severely_stuck_keys:
            print("  ✓ 严重卡住数据已处理，其他模态开头对齐良好")
        else:
            print("  ✓ 所有模态开头对齐良好，无需修正")
        return main_timestamps

    print(f"  发现开头 {len(problematic_frames)} 帧存在模态间对齐偏差过大，开始修正...")

    normal_problematic_keys = set()
    for spread_info in problematic_frames:
        for key in spread_info["keys"]:
            if key not in severely_stuck_keys:
                normal_problematic_keys.add(key)

    if normal_problematic_keys:
        problematic_by_type = {}
        for key in normal_problematic_keys:
            if any(cam in key for cam in ["head_cam", "wrist_cam"]):
                modality_type = "相机"
            elif "action." in key:
                modality_type = "动作"
            elif "observation." in key:
                modality_type = "观测"
            else:
                modality_type = "其他"
            problematic_by_type.setdefault(modality_type, []).append(key)
        print("  问题模态分布:")
        for mod_type, keys in problematic_by_type.items():
            print(f"    {mod_type}: {len(keys)} 个 - {keys}")

    best_start_idx = 0
    min_max_spread = float("inf")
    search_range = min(20, len(main_timestamps))
    normal_valid_keys = [key for key in valid_keys if key not in severely_stuck_keys]

    for start_candidate in range(search_range):
        if start_candidate >= len(main_timestamps):
            break
        candidate_timestamps = main_timestamps[start_candidate:]
        if len(candidate_timestamps) < 10:
            break

        check_frames_candidate = min(5, len(candidate_timestamps))
        candidate_subset = candidate_timestamps[:check_frames_candidate]
        max_spread_at_this_start = 0
        valid_alignment = True

        for frame_idx in range(check_frames_candidate):
            frame_timestamps = []
            for key in normal_valid_keys:
                timestamps = np.array([item["timestamp"] for item in preprocessed_data[key]])
                valid_indices = np.where(timestamps >= candidate_subset[0])[0]
                if len(valid_indices) < len(candidate_subset):
                    valid_alignment = False
                    break
                closest_indices = find_closest_indices_fn(timestamps, candidate_subset)
                if frame_idx < len(closest_indices) and closest_indices[frame_idx] < len(timestamps):
                    frame_timestamps.append(timestamps[closest_indices[frame_idx]])
            if not valid_alignment:
                break
            if len(frame_timestamps) > 1:
                frame_timestamps = np.array(frame_timestamps)
                spread_ms = (np.max(frame_timestamps) - np.min(frame_timestamps)) * 1000
                max_spread_at_this_start = max(max_spread_at_this_start, spread_ms)

        if valid_alignment and max_spread_at_this_start < min_max_spread:
            min_max_spread = max_spread_at_this_start
            best_start_idx = start_candidate
        if min_max_spread <= max_alignment_tolerance_ms:
            break

    if best_start_idx > 0:
        original_length = len(main_timestamps)
        main_timestamps = main_timestamps[best_start_idx:]
        removed_frames = original_length - len(main_timestamps)
        worst_before = max(spread["spread_ms"] for spread in problematic_frames)
        print(f"  ✓ 修正完成：移除开头 {removed_frames} 帧")
        print(f"    最大模态间差值: {worst_before:.1f}ms -> {min_max_spread:.1f}ms")
        print(f"    修正后主时间戳长度: {len(main_timestamps)}")
    else:
        print("  ⚠️ 无法找到满意的修正方案，保持原始时间戳")
        print(f"  建议检查数据质量，当前最小模态间最大差值: {min_max_spread:.1f}ms")

    return main_timestamps

