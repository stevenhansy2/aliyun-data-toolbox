import numpy as np


class ReaderAlignmentFpsMixin:
    def _adjust_frame_rate_to_30fps(
        self, aligned_data: dict, main_timestamps: np.ndarray
    ):
        """
        将主时间戳调整到30fps范围（29.905~30.095Hz）。
        不修改各模态的原始时间戳，仅基于调整后的主时间戳重新选取最近帧，避免打乱已对齐数据。
        """
        log_print("=" * 60)
        log_print("开始帧率调整到30fps（不修改各模态时间戳）...")

        if len(main_timestamps) < 2:
            log_print("  ⚠️ 时间戳数量不足，跳过帧率调整")
            return aligned_data, main_timestamps

        # 当前帧率
        time_span = main_timestamps[-1] - main_timestamps[0]
        current_fps = len(main_timestamps) / time_span if time_span > 0 else 0.0
        target_fps_min = 29.905
        target_fps_max = 30.095
        target_fps = 30.0

        log_print(f"  当前帧率: {current_fps:.3f}Hz, 帧数: {len(main_timestamps)}, 时间跨度: {time_span:.3f}s")

        # 已在目标范围内则直接返回
        if target_fps_min <= current_fps <= target_fps_max:
            log_print("  ✓ 帧率已在目标范围内，无需调整")
            return aligned_data, main_timestamps

        # 仅调整主时间戳，不改动各模态时间戳
        new_main = np.array(main_timestamps, dtype=float)

        if current_fps < target_fps_min:
            # 帧率过低：在最大间隔处插入时间戳（中点），不修改各模态时间戳
            log_print("  帧率过低 -> 插入主时间戳")
            need = int(time_span * target_fps) - len(new_main)
            need = max(need, 0)
            inserted = 0
            while inserted < need:
                intervals = np.diff(new_main)
                if len(intervals) == 0:
                    break
                idx = int(np.argmax(intervals))
                mid = (new_main[idx] + new_main[idx + 1]) / 2.0
                new_main = np.insert(new_main, idx + 1, mid)
                inserted += 1
            log_print(f"  已插入主时间戳 {inserted} 个 -> 新帧数 {len(new_main)}")
        elif current_fps > target_fps_max:
            # 帧率过高：均匀抽帧（不对各模态时间戳做任何更改）
            log_print("  帧率过高 -> 抽取主时间戳")
            target_count = int(time_span * target_fps)  # 近似30fps数量
            target_count = max(target_count, 2)
            # 均匀下采样索引
            idxs = np.linspace(0, len(new_main) - 1, target_count).astype(int)
            new_main = new_main[idxs]
            log_print(f"  抽帧后主时间戳数: {len(new_main)}")

        new_span = new_main[-1] - new_main[0] if len(new_main) > 1 else 0.0
        new_fps = len(new_main) / new_span if new_span > 0 else 0.0
        log_print(f"  调整后帧率: {new_fps:.3f}Hz, 帧数: {len(new_main)}, 时间跨度: {new_span:.3f}s")

        # 额外保障：夹逼主时间戳大间隔到 <= 40ms
        max_interval_sec = 0.010
        if len(new_main) > 1:
            inserted_total = 0
            while True:
                intervals = np.diff(new_main)
                if len(intervals) == 0:
                    break
                worst_idx = int(np.argmax(intervals))
                worst_gap = float(intervals[worst_idx])
                if worst_gap <= max_interval_sec:
                    break
                mid = (new_main[worst_idx] + new_main[worst_idx + 1]) / 2.0
                new_main = np.insert(new_main, worst_idx + 1, mid)
                inserted_total += 1
            if inserted_total:
                log_print(f"  夹逼完成：插入 {inserted_total} 个主时间戳以消除 >40ms 间隔")

        # 收尾：统一用等间隔网格重建主时间戳，确保同时满足频率与40ms间隔限制
        if len(new_main) > 1:
            target_fps_min = 29.905
            target_fps_max = 30.095
            T = new_main[-1] - new_main[0]
            n_min = max(2, int(np.ceil(T * target_fps_min)))
            n_max = max(n_min, int(np.floor(T * target_fps_max)))
            # 在允许范围内，尽量贴近当前帧数
            desired = len(new_main)
            desired = min(max(desired, n_min), n_max)
            if desired != len(new_main):
                log_print(f"  统一重建主时间戳为等间隔: {len(new_main)} -> {desired}")
            new_main = np.linspace(new_main[0], new_main[-1], desired)
            # 复核
            span_chk = new_main[-1] - new_main[0]
            fps_chk = len(new_main) / span_chk if span_chk > 0 else 0.0
            max_gap_ms = (np.max(np.diff(new_main)) * 1000) if len(new_main) > 1 else 0.0
            log_print(f"  复核: fps={fps_chk:.3f}Hz, max_gap={max_gap_ms:.2f}ms")

        # 基于调整后的主时间戳重新选择各模态最近帧
        aligned_data_adjusted = {}

        # 局部函数：用 vectorized 最近索引（允许对 main_ts 施加小偏移）
        def pick_indices_with_offset(ts: np.ndarray, main: np.ndarray, offset_s: float = 0.0):
            return self.find_closest_indices_vectorized(ts, main + offset_s)

        # 估计每个模态的全局偏移（用中位数鲁棒估计，限制在 ±20ms），再据此选帧
        per_key_debug = []
        for key, data_list in aligned_data.items():
            if (
                len(data_list) == 0
                or key.endswith("_extrinsics")
                or key.endswith("_camera_info")
            ):
                aligned_data_adjusted[key] = data_list
                continue

            ts = np.array([item["timestamp"] for item in data_list], dtype=float)
            # 先不加偏移估计一次最近帧
            idx0 = self.find_closest_indices_vectorized(ts, new_main)
            diffs0 = ts[idx0] - new_main
            # 用中位数估计该模态相对主时间戳的全局偏移（秒），并裁剪到 ±20ms
            est_offset = float(np.median(diffs0))
            est_offset = float(np.clip(est_offset, -0.020, 0.020))

            # 按偏移重选
            idx = pick_indices_with_offset(ts, new_main, est_offset)
            sel = [data_list[int(i)] for i in idx]

            # 统计选择后的误差
            diffs = ts[idx] - new_main
            max_abs_ms = float(np.max(np.abs(diffs)) * 1000) if len(diffs) else 0.0
            mean_abs_ms = float(np.mean(np.abs(diffs)) * 1000) if len(diffs) else 0.0
            per_key_debug.append((key, est_offset * 1000.0, max_abs_ms, mean_abs_ms))

            aligned_data_adjusted[key] = sel

        # 调试摘要：按最大绝对误差排序，便于定位问题模态
        if per_key_debug:
            per_key_debug.sort(key=lambda x: x[2], reverse=True)
            log_print("  模态偏移与误差摘要（前10）:")
            for key, off_ms, max_ms, mean_ms in per_key_debug[:10]:
                log_print(f"    {key}: offset={off_ms:+.1f}ms, max|Δ|={max_ms:.1f}ms, mean|Δ|={mean_ms:.1f}ms")

        # 逐帧邻域微调：将每帧多模态时间戳向该帧的中位数靠拢，减小帧内最大差值
        def _refine_alignment_spread(new_main_ts: np.ndarray, data_dict: dict, initial_indices: dict,
                                     threshold_ms: float = 20.0):
            keys = [k for k in initial_indices.keys()]
            # 每个模态的原始时间戳数组
            ts_by_key = {k: np.array([it["timestamp"] for it in data_dict[k]], dtype=float) for k in keys}
            # 拷贝索引（形状与 new_main_ts 等长）
            idx_by_key = {k: initial_indices[k].copy() for k in keys}
            L = len(new_main_ts)
            if L == 0 or not keys:
                return idx_by_key

            for i in range(L):
                # 当前帧选择的各模态时间戳
                frame_ts = []
                for k in keys:
                    idx_i = int(idx_by_key[k][i])
                    idx_i = max(0, min(idx_i, len(ts_by_key[k]) - 1))
                    idx_by_key[k][i] = idx_i
                    frame_ts.append(ts_by_key[k][idx_i])
                frame_ts = np.array(frame_ts, dtype=float)
                spread_ms = (np.max(frame_ts) - np.min(frame_ts)) * 1000.0
                if spread_ms <= threshold_ms:
                    continue

                center = float(np.median(frame_ts))
                # 尝试将极端值向中位数靠拢（只在相邻索引±1内微调，且保证索引单调不回退）
                for k in keys:
                    cur = int(idx_by_key[k][i])
                    candidates = [cur]
                    if cur > 0:
                        candidates.append(cur - 1)
                    if cur + 1 < len(ts_by_key[k]):
                        candidates.append(cur + 1)
                    # 单调性：不回退（相对于上一帧）
                    if i > 0:
                        prev_idx = int(idx_by_key[k][i - 1])
                        candidates = [c for c in candidates if c >= prev_idx]
                        if not candidates:
                            candidates = [cur]  # 回退保护
                    # 选离中位数更近的邻居
                    best = min(candidates, key=lambda c: abs(ts_by_key[k][c] - center))
                    idx_by_key[k][i] = best

            return idx_by_key

        # 准备初始索引（来自偏移后的最近帧选择）
        initial_indices = {}
        for key, data_list in aligned_data.items():
            if (
                len(data_list) == 0
                or key.endswith("_extrinsics")
                or key.endswith("_camera_info")
            ):
                continue
            ts = np.array([item["timestamp"] for item in data_list], dtype=float)
            # 先与 new_main 对齐求最近索引
            idx0 = self.find_closest_indices_vectorized(ts, new_main)
            # 基于对齐后的差值估计全局偏移（秒），裁剪到 ±20ms
            diffs0 = ts[idx0] - new_main
            est_offset = float(np.clip(np.median(diffs0), -0.020, 0.020))
            # 按偏移后的主时间戳重算索引，作为微调起点
            idx_adj = self.find_closest_indices_vectorized(ts, new_main + est_offset)
            initial_indices[key] = idx_adj.astype(int)

        # 执行微调
        refined_indices = _refine_alignment_spread(new_main, aligned_data_adjusted, initial_indices, threshold_ms=20.0)

        # 用微调后的索引重建 aligned_data_adjusted
        for key, data_list in aligned_data_adjusted.items():
            if (
                len(data_list) == 0
                or key.endswith("_extrinsics")
                or key.endswith("_camera_info")
            ):
                continue
            idx = refined_indices.get(key, None)
            if idx is None:
                continue
            source_list = aligned_data[key]  # 以原对齐前列表作为索引来源
            aligned_data_adjusted[key] = [source_list[int(i)] for i in idx]

        # 再次统计帧内最大跨模态差值
        try:
            keys_for_check = [k for k in aligned_data_adjusted.keys()
                              if not (k.endswith('_extrinsics') or k.endswith('_camera_info')) and len(aligned_data_adjusted[k]) > 0]
            if keys_for_check:
                max_spread_ms = 0.0
                worst_idx = -1
                for i in range(len(new_main)):
                    frame_ts = []
                    for k in keys_for_check:
                        frame_ts.append(aligned_data_adjusted[k][i]["timestamp"])
                    if len(frame_ts) >= 2:
                        frame_ts = np.array(frame_ts, dtype=float)
                        spread_ms = (np.max(frame_ts) - np.min(frame_ts)) * 1000.0
                        if spread_ms > max_spread_ms:
                            max_spread_ms = spread_ms
                            worst_idx = i
                log_print(f"  微调后帧内最大跨模态差值: {max_spread_ms:.1f}ms (索引 {worst_idx})")
        except Exception:
            pass

        log_print("  ✓ 帧率与间隔钳制完成，已重新对齐各模态（含每模态全局偏移补偿与逐帧微调）")
        log_print("=" * 60)
        return aligned_data_adjusted, new_main

    def _adjust_frame_rate_to_30fps1(
        self, aligned_data: dict, main_timestamps: np.ndarray
    ):
        """
        调整帧率到30fps范围内（29.95-30.05Hz）
        通过插帧或抽帧来达到目标帧率
        """
        log_print("=" * 60)
        log_print("开始帧率调整到30fps...")

        if len(main_timestamps) < 2:
            log_print("  ⚠️ 时间戳数量不足，跳过帧率调整")
            return aligned_data, main_timestamps

        # 计算当前帧率
        time_span = main_timestamps[-1] - main_timestamps[0]
        current_fps = len(main_timestamps) / time_span
        target_fps_min = 29.905
        target_fps_max = 30.095

        log_print(f"  当前帧率: {current_fps:.2f}Hz")
        log_print(f"  目标范围: {target_fps_min:.2f}-{target_fps_max:.2f}Hz")
        log_print(f"  当前帧数: {len(main_timestamps)}")
        log_print(f"  时间跨度: {time_span:.3f}s")

        # 检查是否在目标范围内
        if target_fps_min <= current_fps <= target_fps_max:
            log_print(f"  ✓ 帧率已在目标范围内，无需调整")
            return aligned_data, main_timestamps

        # 转换为numpy数组便于操作
        main_timestamps = np.array(main_timestamps)

        # 收集所有有效的数据模态
        valid_modalities = {}
        for key, data_list in aligned_data.items():
            if len(data_list) > 0:
                valid_modalities[key] = list(data_list)  # 转换为列表便于插入/删除

        if current_fps < target_fps_min:
            # 帧率太低，需要插帧
            log_print(f"  帧率过低，开始插帧...")
            main_timestamps, valid_modalities = self._insert_frames_to_increase_fps(
                main_timestamps, valid_modalities, target_fps_min, time_span
            )
        elif current_fps > target_fps_max:
            # 帧率太高，需要抽帧
            log_print(f"  帧率过高，开始抽帧...")
            main_timestamps, valid_modalities = self._remove_frames_to_decrease_fps(
                main_timestamps, valid_modalities, target_fps_max, time_span
            )

        # 验证调整结果
        final_time_span = main_timestamps[-1] - main_timestamps[0]
        final_fps = len(main_timestamps) / final_time_span

        log_print(f"  调整后帧率: {final_fps:.2f}Hz")
        log_print(f"  调整后帧数: {len(main_timestamps)}")
        log_print(f"  调整后时间跨度: {final_time_span:.3f}s")

        if target_fps_min <= final_fps <= target_fps_max:
            log_print(f"  ✓ 帧率调整成功！")
        else:
            log_print(f"  ⚠️ 帧率调整后仍不在目标范围内")

        # 转换回原格式
        aligned_data_adjusted = {}
        for key, data_list in valid_modalities.items():
            aligned_data_adjusted[key] = data_list

        # 添加空的模态数据
        for key, data_list in aligned_data.items():
            if key not in aligned_data_adjusted:
                aligned_data_adjusted[key] = []

        log_print("=" * 60)
        return aligned_data_adjusted, main_timestamps

    def _insert_frames_to_increase_fps(
        self,
        main_timestamps: np.ndarray,
        valid_modalities: dict,
        target_fps: float,
        time_span: float,
    ):
        """
        通过插帧来提高帧率到目标值
        """
        target_frame_count = int(time_span * target_fps)
        current_frame_count = len(main_timestamps)
        frames_to_insert = target_frame_count - current_frame_count

        log_print(f"    需要插入 {frames_to_insert} 帧")

        if frames_to_insert <= 0:
            return main_timestamps, valid_modalities

        # 计算时间间隔
        time_intervals = np.diff(main_timestamps) * 1000  # 转换为毫秒
        insertion_threshold_ms = 33.0  # 33ms阈值

        inserted_count = 0
        max_iterations = frames_to_insert * 2  # 防止无限循环
        iteration = 0

        while inserted_count < frames_to_insert and iteration < max_iterations:
            iteration += 1

            # 重新计算间隔（因为插入会改变）
            time_intervals = np.diff(main_timestamps) * 1000

            # 找到最大的间隔
            max_interval_idx = np.argmax(time_intervals)
            max_interval_ms = time_intervals[max_interval_idx]

            if max_interval_ms <= insertion_threshold_ms:
                log_print(f"    无法找到超过{insertion_threshold_ms}ms的间隔进行插帧")
                break

            # 在最大间隔处插入一帧
            insert_pos = max_interval_idx + 1

            # 计算插入时间戳（两帧中间）
            prev_timestamp = main_timestamps[max_interval_idx]
            next_timestamp = main_timestamps[max_interval_idx + 1]
            new_timestamp = (prev_timestamp + next_timestamp) / 2

            # 插入主时间戳
            main_timestamps = np.insert(main_timestamps, insert_pos, new_timestamp)

            # 为所有模态插入数据（复制前一帧）
            for key, data_list in valid_modalities.items():
                if insert_pos <= len(data_list):
                    # 复制前一帧数据
                    reference_frame = data_list[max_interval_idx].copy()
                    reference_frame["timestamp"] = new_timestamp
                    reference_frame["frame_inserted"] = True  # 标记为插入帧
                    data_list.insert(insert_pos, reference_frame)

            inserted_count += 1

            if inserted_count % 10 == 0:  # 每插入10帧输出一次进度
                current_fps = len(main_timestamps) / (
                    main_timestamps[-1] - main_timestamps[0]
                )
                log_print(f"    已插入 {inserted_count} 帧，当前帧率: {current_fps:.2f}Hz")

        log_print(f"    实际插入了 {inserted_count} 帧")

        return main_timestamps, valid_modalities

    # def _remove_frames_to_decrease_fps(self, main_timestamps: np.ndarray, valid_modalities: dict,
    #                                 target_fps: float, time_span: float):
    #     """
    #     通过抽帧来降低帧率到目标值
    #     """
    #     target_frame_count = int(time_span * target_fps)
    #     current_frame_count = len(main_timestamps)
    #     frames_to_remove = current_frame_count - target_frame_count

    #     log_print(f"    需要删除 {frames_to_remove} 帧")

    #     if frames_to_remove <= 0:
    #         return main_timestamps, valid_modalities

    #     removal_threshold_ms = 40.0  # 40ms阈值
    #     removed_count = 0
    #     max_iterations = frames_to_remove * 2  # 防止无限循环
    #     iteration = 0

    #     # 创建可删除帧的候选列表（排除首尾帧）
    #     removable_indices = list(range(1, len(main_timestamps) - 1))

    #     while removed_count < frames_to_remove and iteration < max_iterations and removable_indices:
    #         iteration += 1

    #         # 寻找可以安全删除的帧
    #         best_remove_idx = None
    #         min_max_interval = float('inf')

    #         for candidate_idx in removable_indices[:]:  # 使用切片创建副本
    #             if candidate_idx <= 0 or candidate_idx >= len(main_timestamps) - 1:
    #                 removable_indices.remove(candidate_idx)
    #                 continue

    #             # 计算删除该帧后前后帧的间隔
    #             prev_timestamp = main_timestamps[candidate_idx - 1]
    #             next_timestamp = main_timestamps[candidate_idx + 1]
    #             resulting_interval_ms = (next_timestamp - prev_timestamp) * 1000

    #             # 检查是否满足删除条件
    #             if resulting_interval_ms < removal_threshold_ms:
    #                 # 选择删除后间隔最小的帧（更安全）
    #                 if resulting_interval_ms < min_max_interval:
    #                     min_max_interval = resulting_interval_ms
    #                     best_remove_idx = candidate_idx

    #         if best_remove_idx is None:
    #             log_print(f"    无法找到可以安全删除的帧（删除后间隔需小于{removal_threshold_ms}ms）")
    #             break

    #         # 删除选中的帧
    #         # 删除主时间戳
    #         main_timestamps = np.delete(main_timestamps, best_remove_idx)

    #         # 删除所有模态的对应数据
    #         for key, data_list in valid_modalities.items():
    #             if best_remove_idx < len(data_list):
    #                 del data_list[best_remove_idx]

    #         # 更新可删除索引列表（调整索引值）
    #         removable_indices = [idx - 1 if idx > best_remove_idx else idx for idx in removable_indices]
    #         removable_indices = [idx for idx in removable_indices if 0 < idx < len(main_timestamps) - 1]

    #         removed_count += 1

    #         if removed_count % 10 == 0:  # 每删除10帧输出一次进度
    #             current_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])
    #             log_print(f"    已删除 {removed_count} 帧，当前帧率: {current_fps:.2f}Hz")

    #     # 检查是否成功达到目标
    #     if removed_count < frames_to_remove:
    #         final_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])
    #         error_msg = (
    #             f"无法通过抽帧达到目标帧率。需要删除 {frames_to_remove} 帧，"
    #             f"但只能安全删除 {removed_count} 帧。当前帧率: {final_fps:.2f}Hz"
    #         )
    #         log_print(f"    ❌ {error_msg}")

    #         raise TimestampStuckError(
    #             message=f"帧率调整失败: {error_msg}",
    #             topic="frame_rate_adjustment",
    #             stuck_timestamp=None,
    #             stuck_duration=None,
    #             stuck_frame_count=frames_to_remove - removed_count,
    #             threshold=target_fps
    #         )

    #     log_print(f"    实际删除了 {removed_count} 帧")

    #     return main_timestamps, valid_modalities
    def _remove_frames_to_decrease_fps(
        self,
        main_timestamps: np.ndarray,
        valid_modalities: dict,
        target_fps: float,
        time_span: float,
    ):
        """
        通过滑动窗口删除+局部时间戳重新平均来降低帧率到目标值
        删除后对窗口内时间戳重新平均分布，并同步调整所有模态
        """
        target_frame_count = int(time_span * target_fps)
        current_frame_count = len(main_timestamps)
        frames_to_remove = current_frame_count - target_frame_count

        log_print(
            f"    需要删除 {frames_to_remove} 帧 (从 {current_frame_count} 帧降到 {target_frame_count} 帧)"
        )

        if frames_to_remove <= 0:
            return main_timestamps, valid_modalities

        removed_count = 0
        max_iterations = frames_to_remove * 3  # 防止无限循环
        iteration = 0

        # 滑动窗口参数
        window_size = 5  # 初始窗口大小（必须为奇数）
        max_window_size = 15  # 最大窗口大小
        max_interval_threshold_ms = 40.0  # 最大间隔阈值

        log_print(f"    使用滑动窗口删除+重新平均算法，初始窗口大小: {window_size}")

        while removed_count < frames_to_remove and iteration < max_iterations:
            iteration += 1

            if len(main_timestamps) <= window_size + 2:  # 保证至少有足够的帧数
                log_print(f"    剩余帧数过少({len(main_timestamps)})，无法继续删除")
                break

            # 寻找最佳删除候选
            best_candidate = None
            best_score = float("inf")
            candidates_found = 0

            # 滑动窗口遍历所有可能的删除位置
            for start_idx in range(len(main_timestamps) - window_size + 1):
                end_idx = start_idx + window_size
                center_idx = start_idx + window_size // 2  # 窗口中心索引

                # 跳过首尾帧附近的窗口
                if center_idx <= 1 or center_idx >= len(main_timestamps) - 2:
                    continue

                # 提取窗口时间戳
                window_timestamps = main_timestamps[start_idx:end_idx]

                # 模拟删除窗口中心帧
                timestamps_after_removal = np.concatenate(
                    [
                        window_timestamps[: window_size // 2],  # 中心帧之前
                        window_timestamps[window_size // 2 + 1 :],  # 中心帧之后
                    ]
                )

                # 对删除后的时间戳进行重新平均分布
                reaveraged_timestamps = self._reaverage_timestamps_in_window(
                    timestamps_after_removal,
                    window_timestamps[0],
                    window_timestamps[-1],
                )

                # 检查重新平均后的最大时间间隔
                if len(reaveraged_timestamps) > 1:
                    reaveraged_intervals_ms = np.diff(reaveraged_timestamps) * 1000
                    max_reaveraged_interval = np.max(reaveraged_intervals_ms)

                    # 检查是否满足40ms限制
                    if max_reaveraged_interval <= max_interval_threshold_ms:
                        candidates_found += 1

                        # 计算评分（优先选择重新平均后间隔最小且最均匀的）
                        interval_score = max_reaveraged_interval
                        uniformity_score = np.std(reaveraged_intervals_ms) * 2
                        density_score = -np.mean(
                            np.diff(window_timestamps) * 1000
                        )  # 优先删除密集区域

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

            # 如果找到了合适的删除候选
            if best_candidate is not None:
                # 执行删除和重新平均
                # 修改后的代码
                new_timestamps, success = self._execute_window_removal_and_reaverage(
                    main_timestamps, valid_modalities, best_candidate
                )

                if success:
                    main_timestamps = new_timestamps  # 更新时间戳数组
                    removed_count += 1

                    if removed_count % 10 == 0:
                        current_fps = len(main_timestamps) / (
                            main_timestamps[-1] - main_timestamps[0]
                        )
                        log_print(
                            f"      已删除 {removed_count} 帧，当前帧率: {current_fps:.2f}Hz"
                        )
                        log_print(
                            f"      最新删除: 窗口{best_candidate['start_idx']}-{best_candidate['end_idx']}, "
                            f"删除后最大间隔: {best_candidate['max_interval_after']:.1f}ms"
                        )
                else:
                    log_print(f"    执行删除失败，跳过此候选")

            else:
                # 没有找到合适的删除候选
                log_print(
                    f"    第{iteration}轮: 窗口大小{window_size}下找到 {candidates_found} 个候选"
                )

                # 扩大窗口大小再尝试
                if window_size < max_window_size:
                    window_size += 2  # 保持奇数
                    log_print(f"    扩大窗口大小到: {window_size}，继续尝试")
                    continue
                else:
                    # 窗口已经最大，无法继续
                    log_print(
                        f"    窗口大小已达到最大({window_size})，无法找到更多可删除位置"
                    )
                    break

        # 最终验证和统计
        final_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])

        log_print(f"    删除完成统计:")
        log_print(f"      目标删除: {frames_to_remove} 帧")
        log_print(f"      实际删除: {removed_count} 帧")
        log_print(f"      最终帧率: {final_fps:.3f}Hz")

        # 验证最终时间戳质量
        if len(main_timestamps) > 1:
            final_intervals_ms = np.diff(main_timestamps) * 1000
            max_final_interval = np.max(final_intervals_ms)
            avg_final_interval = np.mean(final_intervals_ms)
            std_final_interval = np.std(final_intervals_ms)

            log_print(f"      最终时间戳质量:")
            log_print(f"        最大间隔: {max_final_interval:.1f}ms")
            log_print(f"        平均间隔: {avg_final_interval:.1f}ms")
            log_print(f"        间隔标准差: {std_final_interval:.1f}ms")

            # 严格验证：检查是否仍有超过40ms的间隔
            large_intervals = final_intervals_ms > max_interval_threshold_ms
            if np.any(large_intervals):
                large_count = np.sum(large_intervals)
                worst_interval = np.max(final_intervals_ms)

                error_msg = (
                    f"删除后验证失败：仍有 {large_count} 个间隔超过{max_interval_threshold_ms}ms，"
                    f"最大间隔{worst_interval:.1f}ms"
                )
                log_print(f"        ❌ {error_msg}")

                # 显示具体问题间隔
                problem_indices = np.where(large_intervals)[0]
                for i, idx in enumerate(problem_indices[:3]):  # 只显示前3个
                    interval_value = final_intervals_ms[idx]
                    start_time = main_timestamps[idx]
                    end_time = main_timestamps[idx + 1]
                    log_print(
                        f"          问题间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={interval_value:.1f}ms"
                    )

                raise TimestampStuckError(
                    message=f"严格间隔验证失败: {error_msg}",
                    topic="strict_interval_validation",
                    stuck_timestamp=main_timestamps[problem_indices[0]],
                    stuck_duration=worst_interval / 1000,
                    stuck_frame_count=large_count,
                    threshold=max_interval_threshold_ms / 1000,
                )
            else:
                log_print(f"        ✓ 所有间隔都在{max_interval_threshold_ms}ms以内")

        # 严格检查：是否达到目标帧率
        if removed_count < frames_to_remove:
            shortfall = frames_to_remove - removed_count
            error_msg = (
                f"删除未完成：需要删除 {frames_to_remove} 帧，实际删除 {removed_count} 帧，"
                f"还差 {shortfall} 帧。当前帧率: {final_fps:.3f}Hz，目标: ≤{target_fps:.3f}Hz"
            )

            log_print(f"    ❌ {error_msg}")

            raise TimestampStuckError(
                message=f"严格帧率调整失败: {error_msg}",
                topic="strict_frame_rate_adjustment",
                stuck_timestamp=None,
                stuck_duration=None,
                stuck_frame_count=shortfall,
                threshold=target_fps,
            )

        # 最终检查：验证帧率是否达到目标
        if final_fps > target_fps:
            fps_excess = final_fps - target_fps
            error_msg = (
                f"最终帧率验证失败：当前帧率 {final_fps:.3f}Hz 仍然超过目标 {target_fps:.3f}Hz，"
                f"超出 {fps_excess:.3f}Hz"
            )

            log_print(f"    ❌ {error_msg}")

            raise TimestampStuckError(
                message=f"严格帧率目标未达成: {error_msg}",
                topic="strict_fps_target",
                stuck_timestamp=None,
                stuck_duration=None,
                stuck_frame_count=frames_to_remove - removed_count,
                threshold=target_fps,
            )

        log_print(f"    ✓ 滑动窗口删除+重新平均成功完成")
        log_print(f"      最终帧率: {final_fps:.3f}Hz ≤ {target_fps:.3f}Hz")
        log_print(
            f"      最大时间间隔: {max_final_interval:.1f}ms ≤ {max_interval_threshold_ms}ms"
        )

        return main_timestamps, valid_modalities

    def _reaverage_timestamps_in_window(
        self,
        timestamps_after_removal: np.ndarray,
        window_start_time: float,
        window_end_time: float,
    ) -> np.ndarray:
        """
        对删除帧后的窗口内时间戳进行重新平均分布
        修正版：只对内部时间戳重新平均，保持两端不变

        Args:
            timestamps_after_removal: 删除中心帧后的时间戳数组
            window_start_time: 窗口开始时间（保持不变）
            window_end_time: 窗口结束时间（保持不变）

        Returns:
            重新平均分布后的时间戳数组
        """
        if len(timestamps_after_removal) <= 2:
            # 如果只有2个或更少的点，无法进行内部重新平均
            return timestamps_after_removal

        # 使用传入的窗口起始和结束时间，而不是数组的首尾时间
        start_time = window_start_time
        end_time = window_end_time

        # 内部点数量
        num_internal_points = len(timestamps_after_removal) - 2

        if num_internal_points <= 0:
            # 没有内部点，只返回首尾时间戳
            return np.array([start_time, end_time])

        # 重新平均：在起始和结束时间之间均匀分布内部点
        internal_timestamps = np.linspace(
            start_time, end_time, num_internal_points + 2
        )[1:-1]

        # 构建完整的重新平均时间戳数组
        reaveraged_timestamps = np.concatenate(
            [
                [start_time],  # 起始点保持不变
                internal_timestamps,  # 内部点重新平均
                [end_time],  # 结束点保持不变
            ]
        )

        return reaveraged_timestamps

    def _execute_window_removal_and_reaverage(
        self, main_timestamps: np.ndarray, valid_modalities: dict, candidate: dict
    ) -> bool:
        """
        执行窗口删除和重新平均操作，同步更新所有模态
        修正版：只对窗口内部时间戳重新平均，两端保持不变；子时间戳使用变化量同步
        """
        try:
            start_idx = candidate["start_idx"]
            end_idx = candidate["end_idx"]
            remove_idx = candidate["remove_idx"]
            window_size = candidate["window_size"]

            # 确保窗口大小至少为5（删除后至少4个点，内部至少2个点可以平均）
            if window_size < 5:
                log_print(f"    窗口大小 {window_size} < 5，无法安全进行内部重新平均")
                return main_timestamps, False

            # 1. 删除主时间戳的中心帧
            main_timestamps_list = main_timestamps.tolist()
            del main_timestamps_list[remove_idx]

            # 2. 同步删除所有模态的对应帧
            for key, data_list in valid_modalities.items():
                if remove_idx < len(data_list):
                    del data_list[remove_idx]

            # 3. 更新主时间戳数组
            new_main_timestamps = np.array(main_timestamps_list)

            # 4. 重新计算窗口范围（删除后索引会变化）
            window_start_idx = start_idx
            window_end_idx = end_idx - 1  # 删除了一帧，所以end_idx要减1

            # 确保索引范围有效
            if window_start_idx >= len(new_main_timestamps) or window_end_idx > len(
                new_main_timestamps
            ):
                log_print(f"    删除后窗口索引超出范围，跳过此次操作")
                return main_timestamps, False

            # 5. 提取窗口内的时间戳进行重新平均（只平均内部点，保持两端不变）
            window_timestamps = new_main_timestamps[window_start_idx:window_end_idx]

            if len(window_timestamps) < 3:
                log_print(
                    f"    删除后窗口内时间戳过少({len(window_timestamps)})，无法重新平均"
                )
                return main_timestamps, False

            # 6. 重新平均：只平均内部点，保持首尾不变
            start_time = window_timestamps[0]  # 窗口起始时间（保持不变）
            end_time = window_timestamps[-1]  # 窗口结束时间（保持不变）

            # 计算内部点的新时间戳（均匀分布）
            num_internal_points = len(window_timestamps) - 2  # 内部点数量

            if num_internal_points > 0:
                # 在起始和结束时间之间均匀分布内部点
                internal_new_timestamps = np.linspace(
                    start_time, end_time, num_internal_points + 2
                )[1:-1]

                # 构建完整的重新平均时间戳数组
                reaveraged_timestamps = np.concatenate(
                    [
                        [start_time],  # 起始点保持不变
                        internal_new_timestamps,  # 内部点重新平均
                        [end_time],  # 结束点保持不变
                    ]
                )
            else:
                # 如果没有内部点，直接使用原时间戳
                reaveraged_timestamps = window_timestamps

            # 验证重新平均后的间隔
            if len(reaveraged_timestamps) > 1:
                reaveraged_intervals_ms = np.diff(reaveraged_timestamps) * 1000
                max_reaveraged_interval = np.max(reaveraged_intervals_ms)

                if max_reaveraged_interval > 40:
                    log_print(
                        f"    重新平均后最大间隔 {max_reaveraged_interval:.1f}ms 仍超过40ms"
                    )
                    return main_timestamps, False

            # 7. 更新窗口内的主时间戳
            for i, new_timestamp in enumerate(reaveraged_timestamps):
                global_idx = window_start_idx + i
                if global_idx < len(new_main_timestamps):
                    old_timestamp = new_main_timestamps[global_idx]
                    timestamp_delta = new_timestamp - old_timestamp

                    # 更新主时间戳
                    new_main_timestamps[global_idx] = new_timestamp

                    # 8. 同步更新所有模态对应帧的时间戳（使用变化量）
                    for key, data_list in valid_modalities.items():
                        if global_idx < len(data_list):
                            if "timestamp" in data_list[global_idx]:
                                # 保存原始时间戳
                                original_modality_timestamp = data_list[global_idx][
                                    "timestamp"
                                ]

                                # 应用相同的时间戳变化量（保持各模态间的相对关系）
                                new_modality_timestamp = (
                                    original_modality_timestamp + timestamp_delta
                                )

                                # 更新时间戳
                                data_list[global_idx][
                                    "timestamp"
                                ] = new_modality_timestamp

                                # 添加调试信息
                                data_list[global_idx]["timestamp_reaveraged"] = True
                                data_list[global_idx][
                                    "original_timestamp"
                                ] = original_modality_timestamp
                                data_list[global_idx][
                                    "timestamp_delta"
                                ] = timestamp_delta
                                data_list[global_idx][
                                    "main_timestamp_new"
                                ] = new_timestamp

            # 9. 将更新后的主时间戳数组复制回原数组
            # main_timestamps[:] = new_main_timestamps[:]

            # 10. 最终验证操作结果
            if len(new_main_timestamps) > 1:
                # 验证整个窗口的间隔
                window_intervals_ms = (
                    np.diff(new_main_timestamps[window_start_idx:window_end_idx]) * 1000
                )
                max_interval = (
                    np.max(window_intervals_ms) if len(window_intervals_ms) > 0 else 0
                )

                if max_interval > 40:
                    log_print(f"    ⚠️ 窗口重新平均后间隔仍然过大: {max_interval:.1f}ms")
                    return main_timestamps, False

                # 输出调试信息
                avg_interval = (
                    np.mean(window_intervals_ms) if len(window_intervals_ms) > 0 else 0
                )
                log_print(
                    f"    ✓ 窗口重新平均成功: 平均间隔 {avg_interval:.1f}ms, 最大间隔 {max_interval:.1f}ms"
                )

            return new_main_timestamps, True

        except Exception as e:
            log_print(f"    执行窗口删除和重新平均时出错: {e}")
            return main_timestamps, False

