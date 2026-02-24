"""Common lightweight helpers used across pipeline/data modules."""

from __future__ import annotations

import numpy as np
import torch


def get_nested_value(data, path, i=None, default=None):
    """Fetch nested value by dotted path and return as float32 tensor."""
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

