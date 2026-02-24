"""Main timeline continuity and prescan helpers."""

from __future__ import annotations

import numpy as np
import rosbag


def enforce_batch_continuity_main(
    self,
    streaming_state,
    main_timestamps: np.ndarray,
    target_interval: float = 0.033,
    tolerance: float = 0.008,
) -> np.ndarray:
    """后续批次主时间线连续性校验与必要平移。"""
    if len(main_timestamps) == 0 or streaming_state.last_main_timestamp is None:
        return main_timestamps

    last_prev = streaming_state.last_main_timestamp
    first_curr = float(main_timestamps[0])
    gap = first_curr - last_prev

    if gap <= 0:
        shift = (last_prev + target_interval) - first_curr
        main_timestamps = main_timestamps + shift
        print(
            f"[STREAM] 主批首帧倒退/重叠，整体前移 {shift:.6f}s 保持连续 (→ {last_prev + target_interval:.6f}s)"
        )
        return main_timestamps

    if abs(gap - target_interval) <= tolerance:
        return main_timestamps

    if gap < target_interval - tolerance:
        shift = (last_prev + target_interval) - first_curr
        main_timestamps = main_timestamps + shift
        print(
            f"[STREAM] 主批首帧间隔偏小 {gap:.6f}s，平移 {shift:.6f}s -> 间隔≈{target_interval:.3f}s"
        )
    elif gap > target_interval + tolerance:
        if gap > 3 * target_interval:
            shift = (last_prev + target_interval) - first_curr
            main_timestamps = main_timestamps + shift
            print(
                f"[STREAM][WARN] 间隔过大 {gap:.3f}s (>3×33ms)，采用平移保持连续，平移 {shift:.6f}s"
            )
        else:
            print(f"[STREAM][INFO] 间隔偏大 {gap:.6f}s，保持原始跨度不平移")
    return main_timestamps


def prescan_main_timeline(
    self,
    bag_file: str,
    abs_start: float,
    abs_end: float,
) -> np.ndarray:
    """预扫描 bag，提取并构建全局主时间线（去重/插值/降采样）。"""
    import rospy

    print("[PRESCAN] 开始预扫描主时间线...")
    print(f"[PRESCAN] 时间窗: [{abs_start:.3f}, {abs_end:.3f}]")

    main_key = getattr(self, "MAIN_TIMESTAMP_TOPIC", "camera_top")
    main_topic_info = self._topic_process_map.get(main_key)
    if main_topic_info is None:
        for cam in self.DEFAULT_CAMERA_NAMES:
            if cam in self._topic_process_map:
                main_key = cam
                main_topic_info = self._topic_process_map[main_key]
                print(
                    f"[PRESCAN][WARN] 主时间线 {self.MAIN_TIMESTAMP_TOPIC} 不存在，使用 {main_key}"
                )
                break

    if main_topic_info is None:
        print("[PRESCAN][ERROR] 无法找到主相机 topic")
        return np.array([])

    main_topic = main_topic_info["topic"]
    print(f"[PRESCAN] 主相机: {main_key}, topic: {main_topic}")

    bag = rosbag.Bag(bag_file, "r")
    timestamps = []
    try:
        for _, _, t in bag.read_messages(
            topics=[main_topic],
            start_time=rospy.Time.from_sec(abs_start),
            end_time=rospy.Time.from_sec(abs_end),
        ):
            timestamps.append(t.to_sec())
    finally:
        bag.close()

    if not timestamps:
        print("[PRESCAN][ERROR] 未读取到任何主相机时间戳")
        return np.array([])

    print(f"[PRESCAN] 原始帧数: {len(timestamps)}")
    data_list = [{"timestamp": ts, "data": None} for ts in timestamps]
    dedup_list = self._remove_duplicate_timestamps(data_list, main_key)
    print(f"[PRESCAN] 去重后帧数: {len(dedup_list)}")
    interpolated_list = self._interpolate_timestamps_and_data(dedup_list, main_key)
    print(f"[PRESCAN] 插值后帧数: {len(interpolated_list)}")

    main_full = [x["timestamp"] for x in interpolated_list]
    start_idx = self.SAMPLE_DROP
    end_idx = -self.SAMPLE_DROP if self.SAMPLE_DROP > 0 else None
    main_cut = main_full[start_idx:end_idx]

    jump = max(1, self.MAIN_TIMELINE_FPS // self.TRAIN_HZ)
    main_cut = main_cut[::jump]
    global_timeline = np.array(main_cut, dtype=np.float64)

    print(f"[PRESCAN] 全局主时间线生成完成: {len(global_timeline)} 帧")
    if len(global_timeline) > 0:
        print(f"[PRESCAN] 时间范围: [{global_timeline[0]:.3f}, {global_timeline[-1]:.3f}]")
        duration = global_timeline[-1] - global_timeline[0]
        avg_interval = duration / (len(global_timeline) - 1) if len(global_timeline) > 1 else 0
        print(f"[PRESCAN] 持续时间: {duration:.2f}s, 平均间隔: {avg_interval*1000:.1f}ms")

    return global_timeline

