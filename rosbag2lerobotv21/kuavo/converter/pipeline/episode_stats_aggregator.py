"""Aggregate batch-level episode_stats into one merged stats payload."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from lerobot.datasets.utils import serialize_dict

logger = logging.getLogger(__name__)


def _agg_init(shape_len: int) -> dict:
    return {
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "_sum": [0.0] * shape_len,
        "_sum_sq": [0.0] * shape_len,
        "_n": 0,
    }


def _resize_acc(acc: dict, new_len: int) -> None:
    def _res(arr, fill):
        if arr is None:
            return [fill] * new_len
        if len(arr) < new_len:
            return arr + [fill] * (new_len - len(arr))
        return arr[:new_len]

    acc["_sum"] = _res(acc["_sum"], 0.0)
    acc["_sum_sq"] = _res(acc["_sum_sq"], 0.0)
    acc["min"] = _res(acc["min"], float("inf"))
    acc["max"] = _res(acc["max"], float("-inf"))
    acc["mean"] = _res(acc["mean"], 0.0)
    acc["std"] = _res(acc["std"], 0.0)


def _as_list(v):
    if v is None:
        return None
    return v if isinstance(v, list) else [v]


def _reduce_stats(acc: dict, part: dict, key_name: str) -> None:
    vals = part.get(key_name)
    if vals is None:
        return
    vals = _as_list(vals)

    if len(acc["_sum"]) != len(vals):
        _resize_acc(acc, max(len(acc["_sum"]), len(vals)))

    if key_name == "min":
        if acc["min"] is None:
            acc["min"] = vals.copy()
        else:
            acc["min"] = [min(a, b) for a, b in zip(acc["min"], vals)]
    elif key_name == "max":
        if acc["max"] is None:
            acc["max"] = vals.copy()
        else:
            acc["max"] = [max(a, b) for a, b in zip(acc["max"], vals)]
    elif key_name == "mean":
        part_count = part.get("count")
        if part_count is None:
            return
        n = _as_list(part_count)[0]
        for i in range(len(vals)):
            acc["_sum"][i] += float(vals[i]) * n
        acc["_n"] += n
    elif key_name == "std":
        part_count = part.get("count")
        part_mean = part.get("mean")
        if part_count is None or part_mean is None:
            return
        n = _as_list(part_count)[0]
        means = _as_list(part_mean)
        if len(means) != len(vals):
            if len(means) < len(vals):
                means = means + [0.0] * (len(vals) - len(means))
            else:
                means = means[: len(vals)]
        for i in range(len(vals)):
            s = float(vals[i])
            m = float(means[i])
            acc["_sum_sq"][i] += (s * s + m * m) * n


def _finalize(acc: dict) -> None:
    n = acc["_n"]
    if n <= 0:
        acc["mean"] = [0.0] * len(acc["_sum"])
        acc["std"] = [0.0] * len(acc["_sum"])
    else:
        means, stds = [], []
        for i in range(len(acc["_sum"])):
            mean_v = acc["_sum"][i] / n
            var = max(acc["_sum_sq"][i] / n - mean_v * mean_v, 0.0)
            means.append(mean_v)
            stds.append(var**0.5)
        acc["mean"] = means
        acc["std"] = stds
    for k in ("_sum", "_sum_sq", "_n"):
        acc.pop(k, None)


def aggregate_episode_stats(
    batch_dirs: list[Path], total_frames: int, cam_stats: dict, fps: float = 30.0
) -> dict:
    stats_agg = {"episode_index": 0, "stats": {}}
    cumulative_count = 0

    for batch_dir in batch_dirs:
        stats_path = batch_dir / "meta" / "episodes_stats.jsonl"
        if not stats_path.exists():
            continue
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if not line:
                    continue
                data = json.loads(line)
            part_stats = data.get("stats", {})

            ts_offset = cumulative_count / fps
            batch_count = int(part_stats.get("timestamp", {}).get("count", [0])[0])
            cumulative_count += batch_count

            for metric_name, metric_vals in part_stats.items():
                adj = dict(metric_vals)
                if metric_name == "timestamp":
                    for kk in ("min", "max", "mean"):
                        if kk in adj and adj[kk] is not None:
                            value = adj[kk]
                            if isinstance(value, list):
                                adj[kk] = [float(x) + ts_offset for x in value]
                            else:
                                adj[kk] = float(value) + ts_offset

                sample_len = 1
                for kk in ("min", "max", "mean", "std"):
                    value = adj.get(kk)
                    if value is not None:
                        sample_len = len(value) if isinstance(value, list) else 1
                        break

                acc = stats_agg["stats"].get(metric_name)
                if acc is None:
                    acc = _agg_init(sample_len)
                    stats_agg["stats"][metric_name] = acc
                elif len(acc["_sum"]) != sample_len:
                    _resize_acc(acc, max(len(acc["_sum"]), sample_len))

                _reduce_stats(acc, adj, "min")
                _reduce_stats(acc, adj, "max")
                _reduce_stats(acc, adj, "mean")
                _reduce_stats(acc, adj, "std")
        except Exception as exc:
            logger.exception("读取 %s 失败: %s", stats_path, exc)

    for _, acc in stats_agg["stats"].items():
        _finalize(acc)
        acc["count"] = [int(total_frames)]

    cam_stats = {
        key: {
            metric: np.array([int(total_frames)]) if metric == "count" else value
            for metric, value in values.items()
        }
        for key, values in cam_stats.items()
    }
    stats_agg["stats"].update(serialize_dict(cam_stats))
    return stats_agg

