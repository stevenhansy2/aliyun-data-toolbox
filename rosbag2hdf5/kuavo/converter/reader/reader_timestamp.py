import numpy as np


class ReaderTimestampMixin:
    def find_closest_indices_vectorized(self, timestamps, target_timestamps):
        """向量化查找最近时间戳索引"""
        timestamps = np.array(timestamps)
        target_timestamps = np.array(target_timestamps)

        # 使用 searchsorted 进行高效查找
        indices = np.searchsorted(timestamps, target_timestamps)

        # 处理边界情况
        indices = np.clip(indices, 0, len(timestamps) - 1)

        # 检查左右邻居，选择更近的
        valid_left = indices > 0
        left_indices = np.where(valid_left, indices - 1, indices)

        left_diffs = np.abs(timestamps[left_indices] - target_timestamps)
        right_diffs = np.abs(timestamps[indices] - target_timestamps)

        # 选择距离更近的索引
        closer_indices = np.where(left_diffs < right_diffs, left_indices, indices)

        return closer_indices

    def _preprocess_timestamps_and_data(self, data: dict) -> dict:
        """预处理时间戳和数据：去重、检测实际卡顿、插值"""
        preprocessed_data = {}

        for key, data_list in data.items():
            if len(data_list) == 0:
                preprocessed_data[key] = []
                continue

            log_print(f"预处理 {key}: 原始长度 {len(data_list)}")

            # 步骤1: 去除重复时间戳
            deduplicated_data = self._remove_duplicate_timestamps(data_list, key)

            # 步骤2: 检测去重后数据的实际时间间隔卡顿（更准确）
            self._check_actual_time_gaps(
                deduplicated_data, key, max_gap_duration=self.TIME_TOLERANCE
            )

            # 步骤3: 时间戳插值和数据填充
            interpolated_data = self._interpolate_timestamps_and_data(
                deduplicated_data, key
            )

            preprocessed_data[key] = interpolated_data
            log_print(
                f"预处理 {key}: 去重后 {len(deduplicated_data)}, 插值后 {len(interpolated_data)}"
            )

        return preprocessed_data

    def _check_actual_time_gaps(
        self, data_list: list, key: str, max_gap_duration: float = 2.0
    ):
        """检测去重后数据的实际时间间隔卡顿"""
        if len(data_list) <= 1:
            return

        timestamps_seconds = np.array([item["timestamp"] for item in data_list])
        timestamps_ns = (timestamps_seconds * 1e9).astype(np.int64)

        # 计算实际时间间隔
        time_diffs_ns = np.diff(timestamps_ns)
        time_diffs_seconds = time_diffs_ns / 1e9

        # 找出超过阈值的时间间隔
        large_gaps = time_diffs_seconds > max_gap_duration

        if np.any(large_gaps):
            max_gap_seconds = np.max(time_diffs_seconds)
            gap_indices = np.where(large_gaps)[0]

            error_msg = (
                f"时间间隔卡顿检测：{key} 话题存在 {len(gap_indices)} 个超过{max_gap_duration}s的时间间隔，"
                f"最大间隔 {max_gap_seconds:.3f}s，数据质量异常，终止处理"
            )
            log_print(f"[ERROR] {error_msg}")

            # 显示具体的问题间隔
            for i, gap_idx in enumerate(gap_indices[:3]):  # 只显示前3个
                start_time = timestamps_seconds[gap_idx]
                end_time = timestamps_seconds[gap_idx + 1]
                gap_duration = time_diffs_seconds[gap_idx]
                log_print(
                    f"  间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={gap_duration:.3f}s"
                )

            raise TimestampStuckError(
                message=error_msg,
                topic=key,
                stuck_timestamp=timestamps_seconds[gap_indices[0]],
                stuck_duration=max_gap_seconds,
                stuck_frame_count=len(gap_indices),
                threshold=max_gap_duration,
            )
        else:
            max_gap_seconds = (
                np.max(time_diffs_seconds) if len(time_diffs_seconds) > 0 else 0
            )
            log_print(f"  {key}: ✓ 时间间隔正常，最大间隔 {max_gap_seconds:.3f}s")

    def _remove_duplicate_timestamps(self, data_list: list, key: str) -> list:
        """去除重复时间戳及对应数据（使用纳秒精度）"""
        if len(data_list) <= 1:
            return data_list

        deduplicated = []
        seen_timestamps = set()
        duplicate_count = 0

        for item in data_list:
            timestamp_seconds = item["timestamp"]
            # 转换为纳秒精度避免浮点精度问题
            timestamp_ns = int(timestamp_seconds * 1e9)

            if timestamp_ns not in seen_timestamps:
                seen_timestamps.add(timestamp_ns)
                deduplicated.append(item)
            else:
                duplicate_count += 1

        if duplicate_count > 0:
            log_print(f"  {key}: 删除 {duplicate_count} 个重复时间戳")

        return deduplicated

    def _interpolate_timestamps_and_data(self, data_list: list, key: str) -> list:
        """时间戳插值和数据填充（修复版本 - 严格控制间隔，超过2秒直接抛异常）"""
        if len(data_list) <= 1:
            return data_list

        timestamps_seconds = np.array([item["timestamp"] for item in data_list])
        timestamps_ns = (timestamps_seconds * 1e9).astype(np.int64)

        # 首先检查是否有超过2秒的间隔，如果有直接抛出异常
        time_diffs_ns = np.diff(timestamps_ns)
        time_diffs_seconds = time_diffs_ns / 1e9

        max_gap_seconds = np.max(time_diffs_seconds)
        large_gaps_2s = time_diffs_seconds > self.TIME_TOLERANCE  # 2秒阈值

        if np.any(large_gaps_2s):
            gap_indices = np.where(large_gaps_2s)[0]
            error_msg = (
                f"插值阶段发现严重时间间隔：{key} 话题存在 {len(gap_indices)} 个超过{self.TIME_TOLERANCE}s的时间间隔，"
                f"最大间隔 {max_gap_seconds:.3f}s，数据质量异常，终止处理"
            )
            log_print(f"[ERROR] {error_msg}")

            # 显示具体的问题间隔
            for i, gap_idx in enumerate(gap_indices[:3]):  # 只显示前3个
                start_time = timestamps_seconds[gap_idx]
                end_time = timestamps_seconds[gap_idx + 1]
                gap_duration = time_diffs_seconds[gap_idx]
                log_print(
                    f"  严重间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={gap_duration:.3f}s"
                )

            raise TimestampStuckError(
                message=error_msg,
                topic=key,
                stuck_timestamp=timestamps_seconds[gap_indices[0]],
                stuck_duration=max_gap_seconds,
                stuck_frame_count=len(gap_indices),
                threshold=2.0,
            )

        # 确定插值间隔（纳秒）
        if any(cam in key for cam in ["head_cam"]) and "depth" not in key:
            # 彩色视频：33ms间隔 (30fps)
            target_interval_ns = int(32 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(39.8 * 1e6)  # 37ms最大允许间隔
            data_type = "video"
        elif any(cam in key for cam in ["wrist_cam"]) and "depth" not in key:
            # 彩色视频：33ms间隔 (30fps)
            target_interval_ns = int(32 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(2 * 1e6)  # 38ms最大允许间隔
            data_type = "video"
        elif "depth" in key:
            # 深度视频：33ms间隔 (30fps)
            target_interval_ns = int(32 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(2 * 1e6)  # 38ms最大允许间隔
            data_type = "depth"

        else:
            # 传感器数据：5ms间隔 (100hz)
            target_interval_ns = int(10 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(2 * 1e6)  # 5ms最大允许间隔（传感器更严格）
            data_type = "sensor"

        # 检测需要插值的位置（使用更严格的阈值）
        interpolation_threshold_ns = (
            max_allowed_interval_ns  # 直接使用最大允许间隔作为阈值
        )

        large_gaps = time_diffs_ns > interpolation_threshold_ns

        if not np.any(large_gaps):
            # 无需插值
            log_print(f"  {key}: 无需插值，最大间隔 {np.max(time_diffs_ns)/1e6:.1f}ms")
            return data_list

        log_print(f"  {key}: 发现 {np.sum(large_gaps)} 个需要插值的时间间隔")
        log_print(
            f"  {key}: 目标间隔 {target_interval_ns/1e6:.1f}ms, 最大允许间隔 {max_allowed_interval_ns/1e6:.1f}ms"
        )

        # 构建插值后的数据
        interpolated_data = []

        for i in range(len(data_list)):
            # 添加当前数据点
            interpolated_data.append(data_list[i])

            # 检查是否需要在当前点和下一点之间插值
            if i < len(data_list) - 1 and large_gaps[i]:
                current_time_ns = timestamps_ns[i]
                next_time_ns = timestamps_ns[i + 1]
                gap_duration_ns = next_time_ns - current_time_ns
                gap_duration_seconds = gap_duration_ns / 1e9

                # 双重保险：再次检查间隔是否超过self.TIME_TOLERANCE
                if gap_duration_seconds > self.TIME_TOLERANCE:
                    error_msg = f"插值过程中发现超过{self.TIME_TOLERANCE}秒的间隔：{key} 在索引{i}处有{gap_duration_seconds:.3f}s间隔"
                    log_print(f"[ERROR] {error_msg}")
                    raise TimestampStuckError(
                        message=error_msg,
                        topic=key,
                        stuck_timestamp=current_time_ns / 1e9,
                        stuck_duration=gap_duration_seconds,
                        stuck_frame_count=1,
                        threshold=2.0,
                    )

                # log_print(f"    间隔{i}: {gap_duration_ns/1e6:.1f}ms 需要插值")

                # 计算需要插入多少个点来满足最大间隔要求
                num_segments_needed = int(
                    np.ceil(gap_duration_ns / max_allowed_interval_ns)
                )

                if num_segments_needed > 1:
                    # 需要插值
                    num_interpolations = num_segments_needed - 1

                    # 生成均匀分布的插值时间戳
                    interp_times_ns = np.linspace(
                        current_time_ns,
                        next_time_ns,
                        num_interpolations + 2,  # +2 包含起点和终点
                        dtype=np.int64,
                    )[
                        1:-1
                    ]  # 去掉起点和终点

                    # log_print(f"    插入 {len(interp_times_ns)} 个点，平均间隔 {gap_duration_ns/(num_interpolations+1)/1e6:.1f}ms")

                    # 插入数据点
                    for interp_time_ns in interp_times_ns:
                        interp_time_seconds = interp_time_ns / 1e9  # 转回秒
                        interpolated_item = self._create_interpolated_data_point(
                            data_list[i], interp_time_seconds, data_type
                        )
                        interpolated_data.append(interpolated_item)

        # 验证插值结果
        final_timestamps = np.array([item["timestamp"] for item in interpolated_data])
        final_timestamps_ns = (final_timestamps * 1e9).astype(np.int64)
        final_intervals_ns = np.diff(final_timestamps_ns)
        final_intervals_ms = final_intervals_ns / 1e6

        max_final_interval = np.max(final_intervals_ms)
        log_print(f"  {key}: 插值完成，最大间隔 {max_final_interval:.1f}ms")

        # # 最终检查：如果插值后仍然存在超过阈值的间隔，抛出异常
        # if max_final_interval > max_allowed_interval_ns / 1e6:
        #     problematic_indices = np.where(final_intervals_ms > max_allowed_interval_ns / 1e6)[0]
        #     error_msg = f"插值后验证失败：{key} 仍有 {len(problematic_indices)} 个间隔超过{max_allowed_interval_ns/1e6:.1f}ms阈值，最大间隔{max_final_interval:.1f}ms"
        #     log_print(f"[ERROR] {error_msg}")

        #     # 显示具体问题
        #     for idx in problematic_indices[:3]:  # 只显示前3个
        #         log_print(f"    问题间隔{idx}: {final_intervals_ms[idx]:.1f}ms")

        #     raise TimestampStuckError(
        #         message=f"插值后质量验证失败: {error_msg}",
        #         topic=key,
        #         stuck_timestamp=final_timestamps[problematic_indices[0]],
        #         stuck_duration=max_final_interval/1000,
        #         stuck_frame_count=len(problematic_indices),
        #         threshold=max_allowed_interval_ns/1e6/1000  # 转换为秒
        #     )

        return interpolated_data

    def _create_interpolated_data_point(
        self, reference_item: dict, new_timestamp: float, data_type: str
    ) -> dict:
        """创建插值数据点"""
        interpolated_item = reference_item.copy()
        interpolated_item["timestamp"] = new_timestamp

        # 根据数据类型处理数据字段
        if data_type in ["video", "depth"]:
            # 图像数据：复制参考帧的数据
            interpolated_item["interpolated"] = True
        elif data_type == "sensor":
            # 传感器数据：保持相同的数值（零阶保持插值）
            interpolated_item["interpolated"] = True

        return interpolated_item

    def _validate_timestamp_quality(self, timestamps: np.ndarray, data_name: str):
        """验证时间戳质量（使用纳秒精度）- 增强版本"""
        if len(timestamps) <= 1:
            return

        # 转换为纳秒进行精确计算
        timestamps_ns = (timestamps * 1e9).astype(np.int64)
        time_diffs_ns = np.diff(timestamps_ns)
        time_diffs_ms = time_diffs_ns / 1e6  # 转换为毫秒显示

        # 检查时间间隔
        mean_interval_ms = np.mean(time_diffs_ms)
        max_interval_ms = np.max(time_diffs_ms)
        min_interval_ms = np.min(time_diffs_ms)
        std_interval_ms = np.std(time_diffs_ms)

        log_print(f"  {data_name} 时间戳质量:")
        log_print(f"    平均间隔: {mean_interval_ms:.1f}ms")
        log_print(f"    最大间隔: {max_interval_ms:.1f}ms")
        log_print(f"    最小间隔: {min_interval_ms:.1f}ms")
        log_print(f"    标准差: {std_interval_ms:.1f}ms")

        # 严格的质量检查 - 修改为更严格的验证
        critical_errors = []
        warnings = []

        if max_interval_ms > 40:  # 40ms阈值 - 关键错误
            critical_errors.append(f"最大时间间隔过大: {max_interval_ms:.1f}ms")

        if min_interval_ms < 0.1:  # 0.1ms阈值 - 关键错误
            critical_errors.append(f"最小时间间隔过小: {min_interval_ms:.1f}ms")

        if std_interval_ms > 15:  # 15ms标准差阈值 - 警告
            warnings.append(f"时间间隔波动过大: {std_interval_ms:.1f}ms")

        # 检查重复时间戳
        unique_timestamps = np.unique(timestamps_ns)
        if len(unique_timestamps) < len(timestamps_ns):
            duplicate_count = len(timestamps_ns) - len(unique_timestamps)
            critical_errors.append(f"仍存在 {duplicate_count} 个重复时间戳")

        # 输出结果
        if critical_errors:
            log_print(f"    ❌ 关键错误: {'; '.join(critical_errors)}")
            # 对于主时间戳的关键错误，抛出异常
            if data_name == "主时间戳":
                error_msg = (
                    f"{data_name} 存在关键质量问题: {'; '.join(critical_errors)}"
                )
                raise TimestampStuckError(
                    message=error_msg,
                    topic=data_name,
                    stuck_timestamp=None,
                    stuck_duration=max_interval_ms / 1000,
                    stuck_frame_count=len(critical_errors),
                    threshold=0.04,
                )
        elif warnings:
            log_print(f"    ⚠️  警告: {'; '.join(warnings)}")
        else:
            log_print(f"    ✓ 时间戳质量良好")

