"""Alignment engine for frame-wise multimodal data synchronization."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def align_frame_data_optimized(
    self,
    data: dict,
    drop_head: bool,
    drop_tail: bool,
    action_config=None,
    streaming_state=None,
    external_main_timestamps: np.ndarray | None = None,
):
    """
    流式适配：
      - 当提供 external_main_timestamps 时，直接使用该时间线进行对齐（全局主时间线模式）
      - 否则回退到原有逻辑（兼容旧代码）
    """
    aligned_data = defaultdict(list)
    main_key_cfg = getattr(self, "MAIN_TIMESTAMP_TOPIC", "camera_top")

    preprocessed = {}
    for key, lst in data.items():
        if not lst:
            preprocessed[key] = []
            continue
        dedup = self._remove_duplicate_timestamps(lst, key)
        self._check_actual_time_gaps(dedup, key, max_gap_duration=self.TIME_TOLERANCE)
        if key == main_key_cfg and external_main_timestamps is None:
            dedup = self._interpolate_timestamps_and_data(dedup, key)
        preprocessed[key] = dedup

    if external_main_timestamps is not None:
        main_ts_np = external_main_timestamps
        print(f"[ALIGN] 使用外部全局主时间线: {len(main_ts_np)} 帧")
    else:
        if main_key_cfg not in preprocessed or len(preprocessed[main_key_cfg]) == 0:
            candidates = [k for k, v in preprocessed.items() if v]
            if not candidates:
                return aligned_data
            main_key_cfg = candidates[0]
            print(f"[STREAM][WARN] 主时间线缺失，降级使用 {main_key_cfg}")

        main_full = [x["timestamp"] for x in preprocessed[main_key_cfg]]
        start_idx = (
            self.SAMPLE_DROP
            if (drop_head and (streaming_state is None or streaming_state.batch_index == 0))
            else 0
        )
        end_idx = (
            -self.SAMPLE_DROP
            if (drop_tail and (streaming_state is None or streaming_state.batch_index == 0))
            else None
        )
        main_cut = main_full[start_idx:end_idx]
        jump = max(1, self.MAIN_TIMELINE_FPS // self.TRAIN_HZ)
        main_cut = main_cut[::jump]

        data_with_content = {k: v for k, v in preprocessed.items() if v}
        if data_with_content:
            min_end = min(v[-1]["timestamp"] for v in data_with_content.values())
            main_cut = [t for t in main_cut if t < min_end]

        main_ts_np = np.array(main_cut)

        if streaming_state is not None and streaming_state.batch_index > 0:
            main_ts_np = self._enforce_batch_continuity_main(streaming_state, main_ts_np)
        else:
            print("[STREAM] 首批执行多模态开头对齐修正")
            main_ts_np = self._fix_multimodal_start_alignment(main_ts_np, preprocessed)

    for key, lst in preprocessed.items():
        if not lst:
            aligned_data[key] = []
            continue
        ts_arr = np.array([f["timestamp"] for f in lst])
        idxs = self.find_closest_indices_vectorized(ts_arr, main_ts_np)
        aligned_data[key] = [lst[i] for i in idxs]

        aligned_timestamps = ts_arr[idxs]
        time_errors_ms = np.abs(aligned_timestamps - main_ts_np) * 1000
        max_diff = np.max(time_errors_ms)
        mean_diff = np.mean(time_errors_ms)
        errors_gt_10ms = np.sum(time_errors_ms > 10)
        errors_gt_15ms = np.sum(time_errors_ms > 15)
        errors_gt_20ms = np.sum(time_errors_ms > 20)

        print(f"  {key}: 对齐完成 {len(aligned_data[key])} 帧")
        print(f"    时间戳误差: 平均 {mean_diff:.1f}ms, 最大 {max_diff:.1f}ms")
        print(
            f"    误差分布: >10ms={errors_gt_10ms}, >15ms={errors_gt_15ms}, >20ms={errors_gt_20ms}"
        )

        if errors_gt_10ms > 0:
            print(f"    [按需插值] 发现 {errors_gt_10ms} 帧误差 >10ms，进行插值修正")
            aligned_data[key] = self._interpolate_on_demand(
                aligned_data[key], time_errors_ms, lst, ts_arr, main_ts_np, key
            )
        else:
            print("    ✅ 所有帧误差 <10ms，无需插值")

    if streaming_state is not None:
        streaming_state.update(main_ts_np, aligned_data)

    return aligned_data

