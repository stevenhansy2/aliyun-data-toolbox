import json
import jsonlines
import numpy as np
import shutil
from pathlib import Path
import os
import pandas as pd

INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
# STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"
META_PATH = "metadata.json"


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def get_nested_item(obj, flattened_key: str, sep: str = "/"):
    split_keys = flattened_key.split(sep)
    getter = obj[split_keys[0]]
    if len(split_keys) == 1:
        return getter

    for key in split_keys[1:]:
        getter = getter[key]

    return getter


def serialize_dict(stats: dict[str, np.ndarray | dict]) -> dict:
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, np.ndarray):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def load_json(fpath: Path):
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Path):
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def write_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def append_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)


def write_info(info: dict, local_dir: Path):
    write_json(info, local_dir / INFO_PATH)


def load_info(local_dir: Path) -> dict:
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info

def load_meta(local_dir: Path) -> dict:
    meta = load_json(local_dir / META_PATH)
    return meta

def write_meta(meta: dict, local_dir: Path):
    write_json(meta, local_dir / META_PATH)

def write_task(task_index: int, task: dict, local_dir: Path):
    task_dict = {
        "task_index": task_index,
        "task": task,
    }
    append_jsonlines(task_dict, local_dir / TASKS_PATH)


def load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks = load_jsonlines(local_dir / TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def write_episode(episode: dict, local_dir: Path):
    append_jsonlines(episode, local_dir / EPISODES_PATH)


def load_episodes(local_dir: Path) -> dict:
    episodes = load_jsonlines(local_dir / EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def write_episode_stats(episode_index: int, episode_stats: dict, local_dir: Path):
    # We wrap episode_stats in a dictionary since `episode_stats["episode_index"]`
    # is a dictionary of stats and not an integer.
    episode_stats = {"episode_index": episode_index, "stats": serialize_dict(episode_stats)}
    append_jsonlines(episode_stats, local_dir / EPISODES_STATS_PATH)


def load_episodes_stats(local_dir: Path) -> dict:
    episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                # if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                #     raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means ** 2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def mv_data_idx(src_dir, tgt_dir, idx_delta):
    tgt_dir.mkdir(parents=True, exist_ok=True)
    for file_name in sorted(os.listdir(src_dir)):
        file = Path(src_dir) / file_name
        if not file.is_file():
            continue
        episode_index = int(file.stem.split('_')[-1])
        new_index = episode_index + idx_delta
        new_name = f'episode_{new_index:06d}{file.suffix}'
        target_file = tgt_dir / new_name
        if file.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(file)
            df['episode_index'] = new_index
            df.to_parquet(target_file, index=False)
            file.unlink()  # 删除原 parquet 文件
        else:
            shutil.move(str(file), target_file)

def mv_data_dir_idx(src_dir, tgt_dir, idx):
    tgt_dir.mkdir(parents=True, exist_ok=True)
    for dir_name in sorted(os.listdir(src_dir)):
        file = Path(src_dir) / dir_name
        if file.is_file():
            continue
        new_index = idx
        new_name = f'episode_{new_index:06d}'
        target_dir = tgt_dir / new_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(file), target_dir)

def mv_data(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file_name in sorted(os.listdir(src_dir)):
        file = Path(src_dir) / file_name
        if file.is_file():
            shutil.move(str(file), dst_dir / file.name)


def append_ep_idx(episodes, idx_delta, out_path):
    for ep_idx, episode in episodes.items():
        episode['episode_index'] = ep_idx + idx_delta
        write_episode(episode, Path(out_path))


def append_ep_sts_idx(episodes_stats, idx_delta, out_path):
    for ep_idx, episode_stats in episodes_stats.items():
        # episode_stats['episode_index'] = ep_idx + idx_delta
        write_episode_stats(ep_idx + idx_delta, episode_stats, Path(out_path))


def append_tasks_idx(tasks, out_path):
    for idx, task in tasks.items():
        write_task(idx, task, Path(out_path))
