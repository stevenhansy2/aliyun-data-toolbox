from collections import defaultdict
from copy import deepcopy

import numpy as np

from converter.reader.constants import DEFAULT_DEXHAND_JOINT_NAMES, DEFAULT_LEJUCLAW_JOINT_NAMES
from converter.reader.msg_processor import TimestampStuckError


class ReaderAlignmentCoreMixin:
    def align_frame_data_optimized(
        self,
        data: dict,
        drop_head: bool,
        drop_tail: bool,
        action_config=None,
        min_duration: float = 5.0,
    ):
        """优化的时间戳对齐函数，支持去重和插值"""
        log_print("开始优化版本的时间戳对齐...")
        aligned_data = defaultdict(list)

        # 1. 预处理：检测严重卡顿、去重和插值
        log_print("步骤1: 预处理数据 - 检测卡顿、去重和插值")
        log_print("---------------------------ooooooooooooooooooooooooooooooooooooooooooo-------------------")
        preprocessed_data = self._preprocess_timestamps_and_data(data)
        # 2. 生成统一的主时间戳基准
        main_timeline = getattr(self, "MAIN_TIMESTAMP_TOPIC", "head_cam_h")
        if (
            main_timeline not in preprocessed_data
            or len(preprocessed_data[main_timeline]) == 0
        ):
            main_timeline = max(
                self.DEFAULT_CAMERA_NAMES,
                key=lambda cam_k: len(preprocessed_data.get(cam_k, [])),
            )
            log_print(f"警告：主时间戳话题不存在，使用降级话题: {main_timeline}")

        # 3. 生成主时间戳序列
        jump = self.MAIN_TIMELINE_FPS // self.TRAIN_HZ
        main_img_timestamps = [t["timestamp"] for t in preprocessed_data[main_timeline]]

        # 根据传入参数裁剪首尾
        start_idx = self.SAMPLE_DROP if drop_head else 0
        end_idx = -self.SAMPLE_DROP if drop_tail else None
        main_img_timestamps = main_img_timestamps[start_idx:end_idx][::jump]

        # 4. 时间戳边界过滤
        data_with_content = {k: v for k, v in preprocessed_data.items() if len(v) > 0}
        if not data_with_content:
            return aligned_data

        min_end = min([data[k][-1]["timestamp"] for k in data_with_content.keys()])
        main_img_timestamps = [t for t in main_img_timestamps if t < min_end]
        main_img_timestamps = np.array(main_img_timestamps)

        log_print(f"主时间线: {main_timeline}, 预处理后长度: {len(main_img_timestamps)}")

        # 5. 多模态开头时间戳修正
        log_print("步骤5: 多模态开头时间戳修正")
        main_img_timestamps = self._fix_multimodal_start_alignment(
            main_img_timestamps, preprocessed_data
        )

        # 6. 验证时间戳质量
        self._validate_timestamp_quality(main_img_timestamps, "主时间戳")

        # 7. 向量化对齐处理
        for key, data_list in preprocessed_data.items():
            if len(data_list) == 0:
                aligned_data[key] = []
                continue

            timestamps = np.array([frame["timestamp"] for frame in data_list])
            closest_indices = self.find_closest_indices_vectorized(
                timestamps, main_img_timestamps
            )
            aligned_data[key] = [data_list[idx] for idx in closest_indices]

            # 验证对齐质量
            aligned_timestamps = timestamps[closest_indices]
            max_diff = np.max(np.abs(aligned_timestamps - main_img_timestamps)) * 1000
            if max_diff > 20:
                log_print(f"警告: {key} 时间戳对齐偏差过大: {max_diff:.1f}ms")
        # === 新增步骤8: 静止区域检测和裁剪 ===
        log_print("步骤8: 静止区域检测和裁剪（暂时跳过）")
        # aligned_data, main_img_timestamps = self._detect_and_trim_aligned_data(aligned_data, main_img_timestamps,action_config=action_config)
        # === 新增步骤8: 帧率调整到30fps ===
        log_print("步骤8: 帧率调整到30fps")
        aligned_data, main_img_timestamps = self._adjust_frame_rate_to_30fps1(
            aligned_data, main_img_timestamps
        )

        log_print("步骤9: 最终验证对齐质量")
        self._final_alignment_validation(
            aligned_data, main_img_timestamps, min_duration=min_duration
        )

        return aligned_data

    def _detect_and_trim_aligned_data(
        self, aligned_data: dict, main_timestamps: np.ndarray, action_config=None
    ):
        """
        检测并裁剪对齐后数据中的静止区域，头尾裁剪上限由首尾动作持续帧数的一半决定
        """
        from converter.utils.facade import (
            detect_stillness_from_image_data,
            analyze_stillness_frames,
        )

        motion_threshold = 4.5
        stillness_ratio = 1
        check_duration = 10.0
        fps = self.TRAIN_HZ or 30

        camera_keys = [
            c
            for c in self.DEFAULT_CAMERA_NAMES
            if c in aligned_data and len(aligned_data[c]) > 0
        ]
        if not camera_keys:
            log_print("  未找到有效的相机数据，跳过静止检测")
            return aligned_data, main_timestamps

        log_print(f"  基于 {len(camera_keys)} 个相机检测静止区域: {camera_keys}")

        # === 计算首尾动作的持续帧数 ===
        max_head_trim_limit = None
        max_tail_trim_limit = None
        total_frames = len(main_timestamps)
        if action_config and len(action_config) > 0:
            # moments.json格式，需从customFieldValues中提取start_position和end_position
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

            # 计算帧区间（直接用比例乘以总帧数）
            if first_action is not None and last_action is not None:
                first_sp = float(first_action["customFieldValues"]["start_position"])
                first_ep = float(first_action["customFieldValues"]["end_position"])
                last_sp = float(last_action["customFieldValues"]["start_position"])
                last_ep = float(last_action["customFieldValues"]["end_position"])

                # 按比例映射到帧索引
                first_start_idx = int(
                    round(
                        (first_sp - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )
                first_end_idx = int(
                    round(
                        (first_ep - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )
                last_start_idx = int(
                    round(
                        (last_sp - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )
                last_end_idx = int(
                    round(
                        (last_ep - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )

                first_len = max(0, first_end_idx - first_start_idx)
                last_len = max(0, last_end_idx - last_start_idx)
                max_head_trim_limit = max(0, int(first_len / 2))
                max_tail_trim_limit = max(0, int(last_len / 2))
                log_print(
                    f"  首动作帧区间: {first_start_idx}-{first_end_idx}，长度: {first_len}"
                )
                log_print(
                    f"  尾动作帧区间: {last_start_idx}-{last_end_idx}，长度: {last_len}"
                )
                log_print(
                    f"  首动作最大裁剪上限: {max_head_trim_limit} 帧，尾动作最大裁剪上限: {max_tail_trim_limit} 帧"
                )
            else:
                log_print("  未找到有效的动作首尾裁剪上限")
        else:
            log_print("  未找到有效的动作首尾裁剪上限")

        # === 静止检测 ===
        all_stillness_results = {}
        for camera_key in camera_keys:
            frames_data = aligned_data[camera_key]
            log_print(f"  分析 {camera_key}: 总帧数 {len(frames_data)}")
            head_stillness, tail_stillness = detect_stillness_from_image_data(
                frames_data,
                camera_key,
                motion_threshold,
                stillness_ratio,
                check_duration,
                fps,
            )
            all_stillness_results[camera_key] = {
                "head_frames": head_stillness,
                "tail_frames": tail_stillness,
            }
            log_print(
                f"    {camera_key}: 开头静止 {head_stillness} 帧, 结尾静止 {tail_stillness} 帧"
            )

        # 计算最终裁剪帧数（取所有相机的最大值确保一致性）
        if all_stillness_results:
            max_head_trim = max(
                result["head_frames"] for result in all_stillness_results.values()
            )
            max_tail_trim = max(
                result["tail_frames"] for result in all_stillness_results.values()
            )
        else:
            max_head_trim = 0
            max_tail_trim = 0

        # === 应用首尾裁剪上限 ===
        if max_head_trim_limit is not None:
            if max_head_trim > max_head_trim_limit:
                log_print(
                    f"  开头静止裁剪帧数 {max_head_trim} 超过首动作上限 {max_head_trim_limit}，已覆盖"
                )
                max_head_trim = max_head_trim_limit
        if max_tail_trim_limit is not None:
            if max_tail_trim > max_tail_trim_limit:
                log_print(
                    f"  结尾静止裁剪帧数 {max_tail_trim} 超过尾动作上限 {max_tail_trim_limit}，已覆盖"
                )
                max_tail_trim = max_tail_trim_limit

        log_print(f"  最终裁剪决定: 开头 {max_head_trim} 帧, 结尾 {max_tail_trim} 帧")

        # 应用裁剪到所有数据
        if max_head_trim > 0 or max_tail_trim > 0:
            trimmed_aligned_data, trimmed_main_timestamps = (
                self._trim_aligned_data_by_frames(
                    aligned_data, main_timestamps, max_head_trim, max_tail_trim
                )
            )
            log_print(
                f"  裁剪完成: 主时间戳 {len(main_timestamps)} -> {len(trimmed_main_timestamps)} 帧"
            )
            return trimmed_aligned_data, trimmed_main_timestamps
        else:
            log_print("  无需裁剪")
            return aligned_data, main_timestamps

    def _trim_aligned_data_by_frames(
        self,
        aligned_data: dict,
        main_timestamps: np.ndarray,
        head_trim_frames: int,
        tail_trim_frames: int,
    ):
        """按帧数裁剪对齐后的数据"""
        trimmed_data = {}

        # 裁剪主时间戳
        original_length = len(main_timestamps)
        start_idx = head_trim_frames
        if tail_trim_frames > 0:
            end_idx = original_length - tail_trim_frames
        else:
            end_idx = original_length

        # 确保索引有效
        start_idx = max(0, start_idx)
        end_idx = min(original_length, end_idx)

        if start_idx < end_idx:
            trimmed_main_timestamps = main_timestamps[start_idx:end_idx]
        else:
            trimmed_main_timestamps = np.array([])
            log_print("    警告: 主时间戳裁剪后为空")

        # 裁剪所有对齐后的数据
        for key, data_list in aligned_data.items():
            if isinstance(data_list, list) and len(data_list) > 0:
                original_data_length = len(data_list)

                if start_idx < end_idx and start_idx < original_data_length:
                    actual_end_idx = min(end_idx, original_data_length)
                    trimmed_data[key] = data_list[start_idx:actual_end_idx]
                    log_print(
                        f"    {key}: {original_data_length} -> {len(trimmed_data[key])} (-{original_data_length - len(trimmed_data[key])})"
                    )
                else:
                    trimmed_data[key] = []
                    log_print(f"    警告: {key} 裁剪后为空")
            else:
                # 非列表数据或空数据保持不变
                trimmed_data[key] = data_list

        return trimmed_data, trimmed_main_timestamps

    def _fix_multimodal_start_alignment(
        self, main_timestamps: np.ndarray, preprocessed_data: dict
    ) -> np.ndarray:
        """修正多模态开头时间戳偏差问题 - 使用与最终验证一致的逻辑"""
        if len(main_timestamps) == 0:
            return main_timestamps

        # 识别所有有效的数据模态（排除外参数据和空数据）
        valid_keys = []
        for key in preprocessed_data.keys():
            if (
                not key.endswith("_extrinsics")
                and not key.endswith("_camera_info")
                and len(preprocessed_data[key]) > 0
            ):
                valid_keys.append(key)

        if len(valid_keys) <= 1:
            log_print("  ✓ 只有一个或零个数据模态，无需修正")
            return main_timestamps

        log_print(f"  检查 {len(valid_keys)} 个数据模态的开头对齐情况")

        # 分析每个数据模态的开头时刻模态间最大最小差值
        alignment_info = []
        max_alignment_tolerance_ms = 20  # 20ms容差
        severe_stuck_threshold_ms = 1000  # 1秒阈值，认为是严重卡住

        # 先进行向量化对齐，获取对齐后的时间戳
        aligned_timestamps_by_key = {}
        for key in valid_keys:
            timestamps = np.array(
                [item["timestamp"] for item in preprocessed_data[key]]
            )

            # 检查前5帧的对齐情况
            check_frames = min(5, len(main_timestamps), len(timestamps))
            if check_frames == 0:
                continue

            main_subset = main_timestamps[:check_frames]

            # 找到最近的对齐索引
            closest_indices = self.find_closest_indices_vectorized(
                timestamps, main_subset
            )
            aligned_timestamps = timestamps[closest_indices]

            aligned_timestamps_by_key[key] = aligned_timestamps

        # 逐帧检查开头几帧的模态间时间戳差值
        check_frames = min(5, len(main_timestamps))
        frame_spreads = []
        severely_stuck_keys = []

        for frame_idx in range(check_frames):
            # 收集该帧所有模态的时间戳
            frame_timestamps = []
            frame_keys = []

            for key in valid_keys:
                if key in aligned_timestamps_by_key and frame_idx < len(
                    aligned_timestamps_by_key[key]
                ):
                    frame_timestamps.append(aligned_timestamps_by_key[key][frame_idx])
                    frame_keys.append(key)

            if len(frame_timestamps) > 1:
                # 计算该帧所有模态时间戳的最大最小差值
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

        # 分析开头对齐质量
        if frame_spreads:
            max_spread = max(spread["spread_ms"] for spread in frame_spreads)
            avg_spread = np.mean([spread["spread_ms"] for spread in frame_spreads])

            log_print(f"    开头{len(frame_spreads)}帧模态间时间戳差值分析:")
            log_print(f"      最大差值: {max_spread:.1f}ms")
            log_print(f"      平均差值: {avg_spread:.1f}ms")

            # 显示每帧的详细情况
            for spread_info in frame_spreads:
                frame_idx = spread_info["frame_idx"]
                spread_ms = spread_info["spread_ms"]
                if spread_ms > max_alignment_tolerance_ms:
                    log_print(f"      帧{frame_idx}: 差值 {spread_ms:.1f}ms (超过阈值)")

                    # 检查是否有严重卡住的模态
                    timestamps = spread_info["timestamps"]
                    keys = spread_info["keys"]
                    main_ts = spread_info["main_timestamp"]

                    for i, (ts, key) in enumerate(zip(timestamps, keys)):
                        diff_ms = abs(ts - main_ts) * 1000
                        if diff_ms > severe_stuck_threshold_ms:
                            if key not in severely_stuck_keys:
                                severely_stuck_keys.append(key)
                                log_print(
                                    f"        {key}: 与主时间戳偏差 {diff_ms:.1f}ms (严重卡住)"
                                )
                else:
                    log_print(f"      帧{frame_idx}: 差值 {spread_ms:.1f}ms (正常)")

        # 识别不同类型的问题模态
        problematic_frames = [
            s for s in frame_spreads if s["spread_ms"] > max_alignment_tolerance_ms
        ]

        # 处理严重卡住的数据模态
        if severely_stuck_keys:
            log_print(
                f"  发现 {len(severely_stuck_keys)} 个严重卡住的数据模态，需要特殊处理:"
            )

            # 按模态类型分类显示
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

                if modality_type not in severely_stuck_by_type:
                    severely_stuck_by_type[modality_type] = []
                severely_stuck_by_type[modality_type].append(key)

            log_print("  严重卡住模态分布:")
            for mod_type, keys in severely_stuck_by_type.items():
                log_print(f"    {mod_type}: {len(keys)} 个 - {keys}")

            # 对严重卡住的数据模态进行时间戳替换
            for key in severely_stuck_keys:
                self._fix_severely_stuck_timestamps(
                    preprocessed_data, key, main_timestamps, max_alignment_tolerance_ms
                )

        # 检查是否需要常规修正（排除已处理的严重卡住数据）
        if not problematic_frames or len(severely_stuck_keys) == len(valid_keys):
            if severely_stuck_keys:
                log_print("  ✓ 严重卡住数据已处理，其他模态开头对齐良好")
            else:
                log_print("  ✓ 所有模态开头对齐良好，无需修正")
            return main_timestamps

        log_print(
            f"  发现开头 {len(problematic_frames)} 帧存在模态间对齐偏差过大，开始修正..."
        )

        # 统计问题模态（排除严重卡住的）
        normal_problematic_keys = set()
        for spread_info in problematic_frames:
            for key in spread_info["keys"]:
                if key not in severely_stuck_keys:
                    normal_problematic_keys.add(key)

        if normal_problematic_keys:
            # 按类型分组显示
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

                if modality_type not in problematic_by_type:
                    problematic_by_type[modality_type] = []
                problematic_by_type[modality_type].append(key)

            log_print("  问题模态分布:")
            for mod_type, keys in problematic_by_type.items():
                log_print(f"    {mod_type}: {len(keys)} 个 - {keys}")

        # 策略：找到所有正常数据模态都能良好对齐的时间范围
        best_start_idx = 0
        min_max_spread = float("inf")

        # 在前20帧中寻找最佳起始点
        search_range = min(20, len(main_timestamps))

        # 只考虑非严重卡住的数据模态
        normal_valid_keys = [
            key for key in valid_keys if key not in severely_stuck_keys
        ]

        for start_candidate in range(search_range):
            if start_candidate >= len(main_timestamps):
                break

            candidate_timestamps = main_timestamps[start_candidate:]
            if len(candidate_timestamps) < 10:  # 至少保留10帧
                break

            # 检查从这个起始点开始的对齐情况
            check_frames_candidate = min(5, len(candidate_timestamps))
            candidate_subset = candidate_timestamps[:check_frames_candidate]

            # 计算这个起始点的模态间最大差值
            max_spread_at_this_start = 0
            valid_alignment = True

            for frame_idx in range(check_frames_candidate):
                # 收集该帧所有正常模态的时间戳
                frame_timestamps = []

                for key in normal_valid_keys:
                    timestamps = np.array(
                        [item["timestamp"] for item in preprocessed_data[key]]
                    )

                    # 找到能覆盖候选时间戳的数据
                    valid_indices = np.where(timestamps >= candidate_subset[0])[0]
                    if len(valid_indices) < len(candidate_subset):
                        valid_alignment = False
                        break

                    closest_indices = self.find_closest_indices_vectorized(
                        timestamps, candidate_subset
                    )
                    if frame_idx < len(closest_indices) and closest_indices[
                        frame_idx
                    ] < len(timestamps):
                        frame_timestamps.append(timestamps[closest_indices[frame_idx]])

                if not valid_alignment:
                    break

                if len(frame_timestamps) > 1:
                    frame_timestamps = np.array(frame_timestamps)
                    spread_ms = (
                        np.max(frame_timestamps) - np.min(frame_timestamps)
                    ) * 1000
                    max_spread_at_this_start = max(max_spread_at_this_start, spread_ms)

            if valid_alignment and max_spread_at_this_start < min_max_spread:
                min_max_spread = max_spread_at_this_start
                best_start_idx = start_candidate

            # 如果找到了很好的对齐点，提前退出
            if min_max_spread <= max_alignment_tolerance_ms:
                break

        # 应用修正
        if best_start_idx > 0:
            original_length = len(main_timestamps)
            main_timestamps = main_timestamps[best_start_idx:]
            removed_frames = original_length - len(main_timestamps)

            worst_before = max(spread["spread_ms"] for spread in problematic_frames)

            log_print(f"  ✓ 修正完成：移除开头 {removed_frames} 帧")
            log_print(f"    最大模态间差值: {worst_before:.1f}ms -> {min_max_spread:.1f}ms")
            log_print(f"    修正后主时间戳长度: {len(main_timestamps)}")

            # 重新验证修正效果
            log_print("  验证修正效果:")
            check_frames_verify = min(3, len(main_timestamps))

            for frame_idx in range(check_frames_verify):
                frame_timestamps = []
                frame_keys = []

                for key in normal_valid_keys[:5]:  # 只验证前5个模态
                    timestamps = np.array(
                        [item["timestamp"] for item in preprocessed_data[key]]
                    )
                    closest_indices = self.find_closest_indices_vectorized(
                        timestamps, main_timestamps[:check_frames_verify]
                    )
                    if frame_idx < len(closest_indices) and closest_indices[
                        frame_idx
                    ] < len(timestamps):
                        frame_timestamps.append(timestamps[closest_indices[frame_idx]])
                        frame_keys.append(key)

                if len(frame_timestamps) > 1:
                    frame_timestamps = np.array(frame_timestamps)
                    spread_ms = (
                        np.max(frame_timestamps) - np.min(frame_timestamps)
                    ) * 1000
                    log_print(f"    帧{frame_idx}: 修正后模态间差值 {spread_ms:.1f}ms")

            if len(normal_valid_keys) > 5:
                log_print(f"    ... 其余 {len(normal_valid_keys) - 5} 个正常模态也已修正")

        else:
            log_print(f"  ⚠️ 无法找到满意的修正方案，保持原始时间戳")
            log_print(f"  建议检查数据质量，当前最小模态间最大差值: {min_max_spread:.1f}ms")

        return main_timestamps

    def _fix_severely_stuck_timestamps(
        self,
        preprocessed_data: dict,
        key: str,
        main_timestamps: np.ndarray,
        tolerance_ms: float = 20,
    ):
        """修复严重卡住的数据模态的时间戳"""
        log_print(f"  开始修复严重卡住的数据模态: {key}")

        data_list = preprocessed_data[key]
        if len(data_list) == 0:
            return

        # 获取原始时间戳
        original_timestamps = np.array([item["timestamp"] for item in data_list])

        # 找到第一个正常时间戳的位置
        normal_start_index = None
        for i in range(len(original_timestamps)):
            if i < len(main_timestamps):
                main_ts = main_timestamps[i]
                data_ts = original_timestamps[i]
                diff_ms = abs(data_ts - main_ts) * 1000

                if diff_ms <= tolerance_ms:
                    normal_start_index = i
                    log_print(f"    在索引 {i} 处找到正常时间戳，偏差 {diff_ms:.1f}ms")
                    break

        if normal_start_index is None:
            # 如果没有找到正常的时间戳，寻找数据开始变化的位置
            log_print(f"    未找到正常时间戳，寻找数据开始变化的位置...")

            # 寻找时间戳开始明显变化的位置
            for i in range(1, min(len(original_timestamps), len(main_timestamps))):
                time_change = abs(original_timestamps[i] - original_timestamps[0])
                if time_change > 1.0:  # 时间戳变化超过1秒
                    # 检查这个位置是否能与主时间戳对齐
                    expected_main_ts = (
                        main_timestamps[i]
                        if i < len(main_timestamps)
                        else main_timestamps[-1]
                        + (i - len(main_timestamps) + 1) * 0.033
                    )
                    diff_ms = abs(original_timestamps[i] - expected_main_ts) * 1000

                    if diff_ms <= tolerance_ms * 5:  # 放宽5倍容差
                        normal_start_index = i
                        log_print(
                            f"    在索引 {i} 处找到数据变化点，开始正常同步，偏差 {diff_ms:.1f}ms"
                        )
                        break

        # 执行时间戳替换
        replaced_count = 0
        if normal_start_index is not None and normal_start_index > 0:
            # 从开头到normal_start_index，使用主时间戳替换
            for i in range(min(normal_start_index, len(main_timestamps))):
                if i < len(data_list):
                    old_timestamp = data_list[i]["timestamp"]
                    new_timestamp = main_timestamps[i]
                    data_list[i]["timestamp"] = new_timestamp
                    data_list[i]["timestamp_replaced"] = True
                    data_list[i]["original_timestamp"] = old_timestamp
                    replaced_count += 1

            log_print(f"    ✓ 替换了前 {replaced_count} 个时间戳")
            log_print(f"    从索引 {normal_start_index} 开始使用原始时间戳")

        else:
            # 如果始终无法同步，替换更多的开头时间戳
            max_replace_count = min(
                50, len(data_list), len(main_timestamps)
            )  # 最多替换50帧

            log_print(f"    无法找到同步点，强制替换前 {max_replace_count} 个时间戳")

            for i in range(max_replace_count):
                if i < len(data_list):
                    old_timestamp = data_list[i]["timestamp"]
                    new_timestamp = main_timestamps[i]
                    data_list[i]["timestamp"] = new_timestamp
                    data_list[i]["timestamp_replaced"] = True
                    data_list[i]["original_timestamp"] = old_timestamp
                    replaced_count += 1

            log_print(f"    ⚠️ 强制替换了前 {replaced_count} 个时间戳")

        # 验证修复效果
        log_print(f"  验证 {key} 修复效果:")
        new_timestamps = np.array([item["timestamp"] for item in data_list])
        check_frames = min(5, len(main_timestamps), len(new_timestamps))

        if check_frames > 0:
            main_subset = main_timestamps[:check_frames]
            data_subset = new_timestamps[:check_frames]
            time_diffs_ms = np.abs(data_subset - main_subset) * 1000
            max_diff = np.max(time_diffs_ms)
            avg_diff = np.mean(time_diffs_ms)

            log_print(
                f"    修复后开头{check_frames}帧: 最大偏差 {max_diff:.1f}ms, 平均偏差 {avg_diff:.1f}ms"
            )

            if max_diff <= tolerance_ms:
                log_print(f"    ✓ {key} 修复成功，开头偏差已控制在 {tolerance_ms}ms 内")
            else:
                log_print(f"    ⚠️ {key} 修复后仍有偏差，但已显著改善")

        # 更新预处理数据
        preprocessed_data[key] = data_list

