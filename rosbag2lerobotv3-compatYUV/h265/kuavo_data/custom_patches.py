from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import sys
import importlib
import concurrent.futures
import logging

import cv2
import shutil
import numpy as np
import torch


lerobot_config_types = importlib.import_module("lerobot.configs.types")


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    RGB = "RGB"
    DEPTH = "DEPTH"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


lerobot_config_types.FeatureType = FeatureType
lerobot_config_types.PolicyFeature = PolicyFeature
sys.modules["lerobot.configs.types"] = lerobot_config_types


from lerobot.datasets.compute_stats import (
    auto_downsample_height_width,
    get_feature_stats,
    load_image_as_numpy,
    sample_indices,
)
from lerobot.datasets.image_writer import write_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, validate_episode_buffer, _encode_video_worker
from lerobot.datasets.utils import validate_frame, write_info


lerobot_datasets_compute_stats = importlib.import_module("lerobot.datasets.compute_stats")
lerobot_datasets_utils = importlib.import_module("lerobot.datasets.utils")
lerobot_datasets_lerobot_dataset = importlib.import_module("lerobot.datasets.lerobot_dataset")


@dataclass
class DirectVideoRef:
    path: str
    timestamp: float


@dataclass
class DirectVideoPayload:
    kind: str
    payload: bytes
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None


def _is_video_ref(value: Any) -> bool:
    return isinstance(value, DirectVideoRef) or (
        isinstance(value, dict) and "path" in value and "timestamp" in value
    ) or (isinstance(value, str) and value.lower().endswith(".mp4"))


def _normalize_video_ref(value: Any, default_timestamp: float) -> dict[str, Any]:
    if isinstance(value, DirectVideoRef):
        return {"path": value.path, "timestamp": float(value.timestamp)}
    if isinstance(value, dict) and "path" in value and "timestamp" in value:
        return {"path": str(value["path"]), "timestamp": float(value["timestamp"])}
    if isinstance(value, str):
        return {"path": value, "timestamp": float(default_timestamp)}
    raise TypeError(f"Unsupported direct video ref type: {type(value)}")


def register_direct_episode_videos(dataset: LeRobotDataset, episode_index: int, video_paths: dict[str, str]) -> None:
    mapping = getattr(dataset, "_direct_episode_video_paths", None)
    if mapping is None:
        mapping = {}
        dataset._direct_episode_video_paths = mapping
    for video_key, path in video_paths.items():
        mapping[(episode_index, video_key)] = str(path)


def _read_video_frame(video_ref: Any, fallback_index: int) -> np.ndarray:
    if isinstance(video_ref, dict):
        path = str(video_ref["path"])
        timestamp = float(video_ref.get("timestamp", 0.0))
    elif isinstance(video_ref, DirectVideoRef):
        path = video_ref.path
        timestamp = float(video_ref.timestamp)
    else:
        path = str(video_ref)
        timestamp = 0.0

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video for stats sampling: {path}")

    target_msec = max(timestamp, 0.0) * 1000.0
    if target_msec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_index)

    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_index)
        ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise ValueError(f"Failed to sample frame from video: {path} @ {timestamp:.3f}s")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)


def custom_sample_images(image_paths: list[Any], sampled_indices) -> np.ndarray:
    images = None
    for i, idx in enumerate(sampled_indices):
        item = image_paths[idx]
        if _is_video_ref(item):
            img = _read_video_frame(item, fallback_index=int(idx))
        else:
            img = load_image_as_numpy(item, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def custom_sample_depth(data, sampled_indices):
    sampled_depth_maps = []
    for idx in sampled_indices:
        sampled_depth_maps.append(auto_downsample_height_width(data[idx, :, :, :]))
    return np.array(sampled_depth_maps)


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    sampled_indices = sample_indices(len(episode_data["action"]))
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = custom_sample_images(data, sampled_indices)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        elif features[key]["dtype"] == "uint16":
            ep_ft_array = custom_sample_depth(data, sampled_indices)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }
        if features[key]["dtype"] == "uint16":
            ep_stats[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()}
    return ep_stats


lerobot_datasets_compute_stats.compute_episode_stats = compute_episode_stats
lerobot_datasets_compute_stats.sample_images = custom_sample_images
lerobot_datasets_lerobot_dataset.compute_episode_stats = compute_episode_stats
sys.modules["lerobot.datasets.compute_stats"] = lerobot_datasets_compute_stats


_original_validate_feature_image_or_video = lerobot_datasets_utils.validate_feature_image_or_video


def patched_validate_feature_image_or_video(name: str, expected_shape: list[str], value):
    if _is_video_ref(value):
        return ""
    return _original_validate_feature_image_or_video(name, expected_shape, value)


lerobot_datasets_utils.validate_feature_image_or_video = patched_validate_feature_image_or_video
sys.modules["lerobot.datasets.utils"] = lerobot_datasets_utils


def patched_add_frame(self, frame: dict) -> None:
    for name in list(frame.keys()):
        if isinstance(frame[name], torch.Tensor):
            frame[name] = frame[name].numpy()

    frame_index = self.episode_buffer["size"] if self.episode_buffer is not None else 0
    timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps

    validate_frame(frame, self.features)

    if self.episode_buffer is None:
        self.episode_buffer = self.create_episode_buffer()
        frame_index = self.episode_buffer["size"]

    self.episode_buffer["frame_index"].append(frame_index)
    self.episode_buffer["timestamp"].append(timestamp)
    self.episode_buffer["task"].append(frame.pop("task"))

    for key in frame:
        if key not in self.features:
            raise ValueError(
                f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
            )

        value = frame[key]
        if self.features[key]["dtype"] in ["image", "video"]:
            if self.features[key]["dtype"] == "video" and _is_video_ref(value):
                self.episode_buffer[key].append(_normalize_video_ref(value, default_timestamp=timestamp))
            else:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                compress_level = 1 if self.features[key]["dtype"] == "video" else 6
                if self.image_writer is None:
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    write_image(value, img_path, compress_level=compress_level)
                else:
                    self.image_writer.save_image(image=value, fpath=img_path, compress_level=compress_level)
                self.episode_buffer[key].append(str(img_path))
        else:
            self.episode_buffer[key].append(value)

    self.episode_buffer["size"] += 1


_original_save_episode_video = LeRobotDataset._save_episode_video
_original_encode_temporary_episode_video = LeRobotDataset._encode_temporary_episode_video


def patched_save_episode_video(self, video_key: str, episode_index: int, temp_path=None):
    refs = getattr(self, "episode_buffer", {}).get(video_key, None)
    if refs and all(isinstance(ref, dict) and "path" in ref and "timestamp" in ref for ref in refs):
        src_path = Path(refs[0]["path"]).resolve()
        if src_path.exists():
            registry = getattr(self, "_direct_video_registry", None)
            if registry is None:
                registry = {}
                self._direct_video_registry = registry
            per_key = registry.setdefault(video_key, {})
            src_key = str(src_path)
            if src_key not in per_key:
                source_count = len(per_key)
                chunk_idx = source_count // self.meta.chunks_size
                file_idx = source_count % self.meta.chunks_size
                dst_path = self.root / self.meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path != dst_path.resolve():
                    shutil.copy2(src_path, dst_path)
                per_key[src_key] = (chunk_idx, file_idx, str(dst_path), str(src_path))
                if episode_index == 0 or source_count == 0:
                    self.meta.update_video_info(video_key)
                    write_info(self.meta.info, self.meta.root)
            chunk_idx, file_idx, _, _ = per_key[src_key]
            timestamps = [float(ref["timestamp"]) for ref in refs]
            from_ts = min(timestamps) if timestamps else 0.0
            to_ts = (max(timestamps) + 1.0 / self.fps) if timestamps else 0.0
            return {
                "episode_index": episode_index,
                f"videos/{video_key}/chunk_index": chunk_idx,
                f"videos/{video_key}/file_index": file_idx,
                f"videos/{video_key}/from_timestamp": from_ts,
                f"videos/{video_key}/to_timestamp": to_ts,
            }
    return _original_save_episode_video(self, video_key, episode_index, temp_path=temp_path)


def patched_encode_temporary_episode_video(self, video_key: str, episode_index: int):
    direct_video_paths = getattr(self, "_direct_episode_video_paths", None)
    if direct_video_paths is not None:
        direct_path = direct_video_paths.pop((episode_index, video_key), None)
        if direct_path is not None:
            return Path(direct_path)

    refs = getattr(self, "episode_buffer", {}).get(video_key, None)
    if refs and all(isinstance(ref, dict) and "path" in ref for ref in refs):
        src_path = Path(refs[0]["path"]).resolve()
        if src_path.exists():
            return src_path

    return _original_encode_temporary_episode_video(self, video_key, episode_index)



_original_save_episode = LeRobotDataset.save_episode


def patched_save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True):
    episode_buffer = episode_data if episode_data is not None else self.episode_buffer

    validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = episode_buffer["episode_index"]

    episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
    self.meta.save_episode_tasks(episode_tasks)
    episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

    for key, ft in self.features.items():
        if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
            continue
        episode_buffer[key] = np.stack(episode_buffer[key])

    self._wait_image_writer()
    ep_stats = compute_episode_stats(episode_buffer, self.features)
    ep_metadata = self._save_episode_data(episode_buffer)
    has_video_keys = len(self.meta.video_keys) > 0
    use_batched_encoding = self.batch_encoding_size > 1

    def _is_direct_video_key(video_key: str) -> bool:
        refs = episode_buffer.get(video_key)
        return bool(refs) and all(isinstance(ref, dict) and "path" in ref and "timestamp" in ref for ref in refs)

    if has_video_keys and not use_batched_encoding:
        direct_keys = [video_key for video_key in self.meta.video_keys if _is_direct_video_key(video_key)]
        encoded_keys = [video_key for video_key in self.meta.video_keys if video_key not in direct_keys]

        if parallel_encoding and len(encoded_keys) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(encoded_keys)) as executor:
                future_to_key = {
                    executor.submit(_encode_video_worker, video_key, episode_index, self.root, self.fps): video_key
                    for video_key in encoded_keys
                }
                results = {}
                for future in concurrent.futures.as_completed(future_to_key):
                    video_key = future_to_key[future]
                    try:
                        results[video_key] = future.result()
                    except Exception as exc:
                        logging.error(f"Video encoding failed for {video_key}: {exc}")
                        raise exc
            for video_key in encoded_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index, temp_path=results[video_key]))
        else:
            for video_key in encoded_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        for video_key in direct_keys:
            ep_metadata.update(self._save_episode_video(video_key, episode_index))

    self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

    if has_video_keys and use_batched_encoding:
        self.episodes_since_last_encoding += 1
        if self.episodes_since_last_encoding == self.batch_encoding_size:
            start_ep = self.num_episodes - self.batch_encoding_size
            end_ep = self.num_episodes
            self._batch_save_episode_video(start_ep, end_ep)
            self.episodes_since_last_encoding = 0

    if not episode_data:
        self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)


_original_finalize = getattr(LeRobotDataset, "finalize", None)


def patched_finalize(self, *args, **kwargs):
    try:
        if _original_finalize is not None:
            return _original_finalize(self, *args, **kwargs)
        return None
    finally:
        registry = getattr(self, "_direct_video_registry", None)
        if registry:
            for per_key in registry.values():
                for _, _, _, src_path in per_key.values():
                    src_dir = Path(src_path).resolve().parent
                    if "rosbag2lerobotv3_source_videos" in str(src_dir):
                        shutil.rmtree(src_dir, ignore_errors=True)

LeRobotDataset.add_frame = patched_add_frame
LeRobotDataset.save_episode = patched_save_episode
LeRobotDataset._save_episode_video = patched_save_episode_video
LeRobotDataset._encode_temporary_episode_video = patched_encode_temporary_episode_video
if _original_finalize is not None:
    LeRobotDataset.finalize = patched_finalize
lerobot_datasets_lerobot_dataset.LeRobotDataset.add_frame = patched_add_frame
lerobot_datasets_lerobot_dataset.LeRobotDataset.save_episode = patched_save_episode
lerobot_datasets_lerobot_dataset.LeRobotDataset._save_episode_video = patched_save_episode_video
lerobot_datasets_lerobot_dataset.LeRobotDataset._encode_temporary_episode_video = patched_encode_temporary_episode_video
if _original_finalize is not None:
    lerobot_datasets_lerobot_dataset.LeRobotDataset.finalize = patched_finalize
sys.modules["lerobot.datasets.lerobot_dataset"] = lerobot_datasets_lerobot_dataset


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video", "uint16"]:
            if "depth" in key or "DEPTH" in key:
                type = FeatureType.DEPTH
            else:
                type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

            names = ft["names"]
            if names[2] in ["channel", "channels"]:
                shape = (shape[2], shape[0], shape[1])
        elif key == "observation.environment_state":
            type = FeatureType.ENV
        elif key.startswith("observation"):
            type = FeatureType.STATE
        elif key == "action":
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(type=type, shape=shape)

    return policy_features


lerobot_datasets_utils.dataset_to_policy_features = dataset_to_policy_features
sys.modules["lerobot.datasets.utils"] = lerobot_datasets_utils
