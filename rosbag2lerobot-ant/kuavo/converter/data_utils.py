"""Compatibility exports for dataset/bag utility functions."""

import torch
import numpy as np

from converter.data.metadata_merge import (
    calculate_action_frames,
    merge_metadata_and_moment,
    get_time_range_from_moments,
)
from converter.data.bag_discovery import (
    get_bag_time_info,
    list_bag_files_auto,
    _sanitize_dataset_name,
    _resolve_sidecar_path,
    discover_bag_tasks_auto,
)
from converter.data.episode_loader import (
    load_raw_depth_images_per_camera,
    load_camera_info_per_camera,
    load_raw_images_per_camera,
    load_raw_episode_data,
    load_hand_data_worker,
    process_rosbag_eef,
    _split_dexhand_lr,
)


def get_nested_value(data, path, i=None, default=None):
    keys = path.split(".")
    v = data
    try:
        for k in keys:
            v = v[k]
        if i is not None:
            if v is not None and len(v) > i:
                v = v[i]
            else:
                v = default
        if v is None:
            v = default
        if isinstance(v, torch.Tensor):
            return v.float()
        return torch.tensor(v, dtype=torch.float32)
    except Exception:
        return torch.tensor(default, dtype=torch.float32)


def is_valid_hand_data(arr, expected_shape=None):
    arr = np.array(arr) if arr is not None else None
    if arr is None or arr.size == 0:
        return False
    if expected_shape is not None and arr.shape[1:] != expected_shape:
        return False
    return True


__all__ = [
    "get_nested_value",
    "is_valid_hand_data",
    "calculate_action_frames",
    "merge_metadata_and_moment",
    "get_time_range_from_moments",
    "get_bag_time_info",
    "list_bag_files_auto",
    "_sanitize_dataset_name",
    "_resolve_sidecar_path",
    "discover_bag_tasks_auto",
    "load_raw_depth_images_per_camera",
    "load_camera_info_per_camera",
    "load_raw_images_per_camera",
    "load_raw_episode_data",
    "load_hand_data_worker",
    "process_rosbag_eef",
    "_split_dexhand_lr",
]
