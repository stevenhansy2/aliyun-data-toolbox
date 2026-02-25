"""Utility operations for timestamp window re-averaging and sync updates."""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def reaverage_timestamps_in_window(
    timestamps_after_removal: np.ndarray,
    window_start_time: float,
    window_end_time: float,
) -> np.ndarray:
    """Re-average internal timestamps while keeping the window endpoints fixed."""
    if len(timestamps_after_removal) <= 2:
        return timestamps_after_removal

    start_time = window_start_time
    end_time = window_end_time
    num_internal_points = len(timestamps_after_removal) - 2

    if num_internal_points <= 0:
        return np.array([start_time, end_time])

    internal_timestamps = np.linspace(start_time, end_time, num_internal_points + 2)[
        1:-1
    ]
    return np.concatenate([[start_time], internal_timestamps, [end_time]])


def execute_window_removal_and_reaverage(
    main_timestamps: np.ndarray,
    valid_modalities: dict,
    candidate: dict,
    *,
    max_interval_ms: float = 40.0,
) -> tuple[np.ndarray, bool]:
    """
    Delete one frame in window center and re-average window timestamps.

    The same timestamp delta is propagated to all modalities to preserve relative offsets.
    """
    try:
        start_idx = candidate["start_idx"]
        end_idx = candidate["end_idx"]
        remove_idx = candidate["remove_idx"]
        window_size = candidate["window_size"]

        if window_size < 5:
            logger.warning("窗口大小 %s < 5，无法安全进行内部重新平均", window_size)
            return main_timestamps, False

        main_timestamps_list = main_timestamps.tolist()
        del main_timestamps_list[remove_idx]

        for _, data_list in valid_modalities.items():
            if remove_idx < len(data_list):
                del data_list[remove_idx]

        new_main_timestamps = np.array(main_timestamps_list)
        window_start_idx = start_idx
        window_end_idx = end_idx - 1

        if window_start_idx >= len(new_main_timestamps) or window_end_idx > len(
            new_main_timestamps
        ):
            logger.warning("删除后窗口索引超出范围，跳过此次操作")
            return main_timestamps, False

        window_timestamps = new_main_timestamps[window_start_idx:window_end_idx]
        if len(window_timestamps) < 3:
            logger.warning("删除后窗口内时间戳过少(%s)，无法重新平均", len(window_timestamps))
            return main_timestamps, False

        start_time = window_timestamps[0]
        end_time = window_timestamps[-1]
        num_internal_points = len(window_timestamps) - 2

        if num_internal_points > 0:
            internal_new_timestamps = np.linspace(
                start_time, end_time, num_internal_points + 2
            )[1:-1]
            reaveraged_timestamps = np.concatenate(
                [[start_time], internal_new_timestamps, [end_time]]
            )
        else:
            reaveraged_timestamps = window_timestamps

        if len(reaveraged_timestamps) > 1:
            reaveraged_intervals_ms = np.diff(reaveraged_timestamps) * 1000
            max_reaveraged_interval = np.max(reaveraged_intervals_ms)
            if max_reaveraged_interval > max_interval_ms:
                logger.warning(
                    "重新平均后最大间隔 %.1fms 仍超过%.0fms",
                    max_reaveraged_interval,
                    max_interval_ms,
                )
                return main_timestamps, False

        for i, new_timestamp in enumerate(reaveraged_timestamps):
            global_idx = window_start_idx + i
            if global_idx >= len(new_main_timestamps):
                continue
            old_timestamp = new_main_timestamps[global_idx]
            timestamp_delta = new_timestamp - old_timestamp
            new_main_timestamps[global_idx] = new_timestamp

            for _, data_list in valid_modalities.items():
                if global_idx < len(data_list) and "timestamp" in data_list[global_idx]:
                    original_modality_timestamp = data_list[global_idx]["timestamp"]
                    new_modality_timestamp = original_modality_timestamp + timestamp_delta
                    data_list[global_idx]["timestamp"] = new_modality_timestamp
                    data_list[global_idx]["timestamp_reaveraged"] = True
                    data_list[global_idx]["original_timestamp"] = (
                        original_modality_timestamp
                    )
                    data_list[global_idx]["timestamp_delta"] = timestamp_delta
                    data_list[global_idx]["main_timestamp_new"] = new_timestamp

        if len(new_main_timestamps) > 1:
            window_intervals_ms = (
                np.diff(new_main_timestamps[window_start_idx:window_end_idx]) * 1000
            )
            max_interval = np.max(window_intervals_ms) if len(window_intervals_ms) > 0 else 0
            if max_interval > max_interval_ms:
                logger.warning("窗口重新平均后间隔仍然过大: %.1fms", max_interval)
                return main_timestamps, False
            avg_interval = np.mean(window_intervals_ms) if len(window_intervals_ms) > 0 else 0
            logger.debug(
                "窗口重新平均成功: 平均间隔 %.1fms, 最大间隔 %.1fms",
                avg_interval,
                max_interval,
            )

        return new_main_timestamps, True

    except Exception as exc:
        logger.exception("执行窗口删除和重新平均时出错: %s", exc)
        return main_timestamps, False
