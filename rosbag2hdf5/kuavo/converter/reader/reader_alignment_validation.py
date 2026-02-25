import numpy as np


class ReaderAlignmentValidationMixin:
    def _final_alignment_validation(
        self, aligned_data: dict, main_timestamps: np.ndarray, min_duration: float = 5.0
    ):
        """最终验证对齐后的数据质量，不满足要求则抛出异常"""

        # === 新增验证：主时间戳长度和时间跨度检查 ===
        # === 新增验证：主时间戳长度和时间跨度检查 ===
        log_print("  验证0: 主时间戳基本要求检查")
        if len(main_timestamps) < max(2, int(min_duration * (self.TRAIN_HZ or 30))):
            raise TimestampStuckError(
                message=f"主时间戳长度过短: {len(main_timestamps)} 帧，小于最小要求",
                topic="main_timestamps_length",
                stuck_timestamp=None,
                stuck_duration=0,
                stuck_frame_count=len(main_timestamps),
                threshold=min_duration,
            )

        # 构建有效模态的时间戳视图（排除外参和 camera_info）
        valid_modalities = {}
        for key, data_list in aligned_data.items():
            if (
                isinstance(data_list, list)
                and len(data_list) > 0
                and not key.endswith("_extrinsics")
                and not key.endswith("_camera_info")
                and "timestamp" in data_list[0]
            ):
                try:
                    ts = np.array([item["timestamp"] for item in data_list], dtype=float)
                    # 长度需与主时间戳一致，若不一致则截断到最短，避免越界
                    L = min(len(ts), len(main_timestamps))
                    if L > 0:
                        valid_modalities[key] = ts[:L]
                except Exception:
                    pass

        # 如果没有有效模态，跳过验证
        if not valid_modalities:
            log_print("  ⚠️ 没有找到有效的数据模态进行验证")
            return

        log_print(f"  开始验证 {len(valid_modalities)} 个数据模态的时间戳同步...")

        alignment_errors = []        # === 新增验证：主时间戳长度和时间跨度检查 ===
        log_print("  验证0: 主时间戳基本要求检查")
        if len(main_timestamps) < max(2, int(min_duration * (self.TRAIN_HZ or 30))):
            raise TimestampStuckError(
                message=f"主时间戳长度过短: {len(main_timestamps)} 帧，小于最小要求",
                topic="main_timestamps_length",
                stuck_timestamp=None,
                stuck_duration=0,
                stuck_frame_count=len(main_timestamps),
                threshold=min_duration,
            )

        # 构建有效模态的时间戳视图（排除外参和 camera_info）
        valid_modalities = {}
        for key, data_list in aligned_data.items():
            if (
                isinstance(data_list, list)
                and len(data_list) > 0
                and not key.endswith("_extrinsics")
                and not key.endswith("_camera_info")
                and "timestamp" in data_list[0]
            ):
                try:
                    ts = np.array([item["timestamp"] for item in data_list], dtype=float)
                    # 长度需与主时间戳一致，若不一致则截断到最短，避免越界
                    L = min(len(ts), len(main_timestamps))
                    if L > 0:
                        valid_modalities[key] = ts[:L]
                except Exception:
                    pass

        # 如果没有有效模态，跳过验证
        if not valid_modalities:
            log_print("  ⚠️ 没有找到有效的数据模态进行验证")
            return

        log_print(f"  开始验证 {len(valid_modalities)} 个数据模态的时间戳同步...")

        alignment_errors = []

        def run_spread_check_and_fix_once():
            """执行一次帧内差值检查，若发现超过20ms的帧，则将该帧所有模态时间戳替换为主时间戳"""
            frame_sync_errors_local = []
            # 纳秒精度缓存
            all_timestamps_ns_local = {
                key: (ts * 1e9).astype(np.int64) for key, ts in valid_modalities.items()
            }
            max_frame_spread_local = 0.0
            worst_frame_idx_local = -1

            for frame_idx in range(len(main_timestamps)):
                frame_ts_ns = []
                frame_keys = []
                for key, ts_ns in all_timestamps_ns_local.items():
                    if frame_idx < len(ts_ns):
                        frame_ts_ns.append(ts_ns[frame_idx])
                        frame_keys.append(key)
                if len(frame_ts_ns) > 1:
                    min_ts_ns = np.min(frame_ts_ns)
                    max_ts_ns = np.max(frame_ts_ns)
                    spread_ms = (max_ts_ns - min_ts_ns) / 1e6
                    if spread_ms > max_frame_spread_local:
                        max_frame_spread_local = spread_ms
                        worst_frame_idx_local = frame_idx
                    if spread_ms > 20.0:
                        frame_sync_errors_local.append(
                            {"frame_idx": frame_idx, "spread_ms": spread_ms, "keys": frame_keys}
                        )

            # 若存在超阈值帧，执行替换：将该帧所有模态的时间戳设为主时间戳
            if frame_sync_errors_local:
                log_print(f"    ⚠️ 发现 {len(frame_sync_errors_local)} 个时刻差值>20ms，执行主时间戳替换修正")
                for err in frame_sync_errors_local:
                    i = err["frame_idx"]
                    ts_main = float(main_timestamps[i])
                    for key in err["keys"]:
                        # 更新 aligned_data 中该帧的时间戳
                        if key in aligned_data and i < len(aligned_data[key]) and "timestamp" in aligned_data[key][i]:
                            old_ts = aligned_data[key][i]["timestamp"]
                            aligned_data[key][i]["timestamp"] = ts_main
                            aligned_data[key][i]["timestamp_replaced_by_main"] = True
                            aligned_data[key][i]["original_timestamp_before_replace"] = old_ts
                        # 更新 valid_modalities 视图
                        if key in valid_modalities and i < len(valid_modalities[key]):
                            valid_modalities[key][i] = ts_main
                # 返回修正后需要再次检查
                return True, max_frame_spread_local, worst_frame_idx_local
            # 无错误
            return False, max_frame_spread_local, worst_frame_idx_local

        # 尝试最多两轮修正与复检
        fixed_once, max_frame_spread, worst_frame_idx = run_spread_check_and_fix_once()
        if fixed_once:
            log_print("    ✓ 第一轮修正完成，重新验证...")
            fixed_twice, max_frame_spread, worst_frame_idx = run_spread_check_and_fix_once()
            if fixed_twice:
                log_print("    ℹ️ 第二轮仍存在超阈值帧，已再次替换为主时间戳后继续验证")

        # 替换后最终检查
        frame_sync_errors = []
        all_timestamps_ns = {key: (ts * 1e9).astype(np.int64) for key, ts in valid_modalities.items()}
        max_frame_spread = 0.0
        worst_frame_idx = -1

        for frame_idx in range(len(main_timestamps)):
            frame_timestamps_ns = []
            frame_keys = []
            for key, timestamps_ns in all_timestamps_ns.items():
                if frame_idx < len(timestamps_ns):
                    frame_timestamps_ns.append(timestamps_ns[frame_idx])
                    frame_keys.append(key)
            if len(frame_timestamps_ns) > 1:
                min_ts_ns = np.min(frame_timestamps_ns)
                max_ts_ns = np.max(frame_timestamps_ns)
                spread_ms = (max_ts_ns - min_ts_ns) / 1e6
                if spread_ms > max_frame_spread:
                    max_frame_spread = spread_ms
                    worst_frame_idx = frame_idx
                if spread_ms > 20.0:
                    frame_sync_errors.append(
                        {"frame_idx": frame_idx, "spread_ms": spread_ms, "timestamps": frame_timestamps_ns, "keys": frame_keys}
                    )

        if frame_sync_errors:
            error_msg = f"发现 {len(frame_sync_errors)} 个时刻的多模态时间戳差值超过20ms阈值"
            log_print(f"    ❌ {error_msg}")
            sorted_errors = sorted(frame_sync_errors, key=lambda x: x["spread_ms"], reverse=True)
            for i, error in enumerate(sorted_errors[:3]):
                frame_idx = error["frame_idx"]
                spread_ms = error["spread_ms"]
                timestamps_s = [ts_ns / 1e9 for ts_ns in error["timestamps"]]
                keys = error["keys"]
                log_print(f"      时刻{frame_idx}: 时间戳差值 {spread_ms:.1f}ms")
                for key, ts_s in zip(keys, timestamps_s):
                    log_print(f"        {key}: {ts_s:.6f}s")
            alignment_errors.append(f"多模态同步: {error_msg}，最大差值 {max_frame_spread:.1f}ms")
        else:
            log_print(f"    ✓ 多模态时间戳同步验证通过，最大差值 {max_frame_spread:.1f}ms")

        # 如果有验证错误，抛出异常
        if alignment_errors:
            error_summary = "; ".join(alignment_errors)
            detailed_msg = (
                f"严格对齐验证失败:\n"
                f"- 最大帧内时间戳差值: {max_frame_spread:.1f}ms\n"
                f"- 验证错误: {error_summary}\n"
                f"- 参与验证的数据模态数: {len(valid_modalities)}\n"
                f"- 主时间戳长度: {len(main_timestamps)}"
            )
            log_print(f"[ERROR] {detailed_msg}")
            raise TimestampStuckError(
                message=f"严格对齐验证失败: {error_summary}",
                topic="strict_alignment_validation",
                stuck_timestamp=(main_timestamps[worst_frame_idx] if worst_frame_idx >= 0 else None),
                stuck_duration=max_frame_spread / 1000,
                stuck_frame_count=len(alignment_errors),
                threshold=0.02,
            )

        # 验证通过后的总结输出保持不变
        log_print("=" * 60)
        log_print("✓ 严格对齐验证通过!")

