"""CLI entry for converting Kuavo rosbag datasets to HDF5."""

import argparse
import json
import os
import time

import h5py
import numpy as np
import tqdm

from converter.configs.runtime_config import Config, load_config_from_json
from converter.reader.reader_entry import PostProcessorUtils
from converter.data.bag_discovery import list_bag_files_auto
from converter.data.episode_loader import load_raw_episode_data
from converter.data.metadata_ops import (
    create_file_structure,
    get_bag_time_info,
    merge_metadata_and_moment,
)
from converter.utils.facade import (
    recursive_filter_and_position,
    save_camera_extrinsic_params,
    save_camera_info_to_json_new,
    save_color_videos_parallel,
    save_color_videos_streaming,
    save_depth_videos_16U_parallel,
    save_depth_videos_16U_streaming,
    save_depth_videos_enhanced_parallel,
    validate_episode_data_consistency,
)


class DatasetConfig:
    def __init__(self):
        self.use_videos = True
        self.image_writer_processes = 0
        self.image_writer_threads = 4
        self.video_backend = "pyav"


DEFAULT_DATASET_CONFIG = DatasetConfig()


def generate_dataset_file(
    raw_config: Config,
    bag_files: list,
    metadata_json_dir: str,
    output_dir: str,
    scene: str,
    sub_scene: str,
    continuous_action: str,
    mode: str,
    min_duration=5.0,
):
    t0 = time.time()
    episodes = range(len(bag_files))

    for ep_idx in tqdm.tqdm(episodes):
        action_config = []
        if os.path.exists(metadata_json_dir):
            with open(metadata_json_dir, "r", encoding="utf-8") as f:
                moments_data = json.load(f) or {}
            action_config = moments_data.get("masks", [])

        bag_info = bag_files[ep_idx]
        if isinstance(bag_info, dict):
            ep_path = bag_info["local_path"]
            start_time = bag_info.get("start", 0)
            end_time = bag_info.get("end", 1)
        else:
            ep_path = bag_info
            start_time = 0
            end_time = 1

        log_print(f"Processing {ep_path} (time range: {start_time}-{end_time})")

        bag_time_info = get_bag_time_info(ep_path)
        if bag_time_info.get("iso_format"):
            log_print(f"Bag开始时间: {bag_time_info['iso_format']}")
            log_print(f"Bag持续时间: {bag_time_info['duration']:.2f}秒")

        (
            imgs_per_cam,
            imgs_per_cam_depth,
            info_per_cam,
            all_low_dim_data,
            main_time_line_timestamps,
            distortion_model,
            head_extrinsics,
            left_extrinsics,
            right_extrinsics,
            compressed,
            all_low_dim_data_original,
        ) = load_raw_episode_data(
            raw_config=raw_config,
            ep_path=ep_path,
            start_time=start_time,
            end_time=end_time,
            action_config=action_config,
            min_duration=min_duration,
            metadata_json_dir=metadata_json_dir,
        )
        log_print(f"[TIME] load_raw_episode_data: {time.time() - t0:.2f}s since start")

        uuid, task_info_dir, depth_dir, video_dir, parameters_dir, proprio_stats_dir = (
            create_file_structure(
                scene=scene,
                sub_scene=sub_scene,
                continuous_action=continuous_action,
                bag_path=ep_path,
                save_dir=output_dir,
                mode=mode,
            )
        )

        recursive_filter_and_position(all_low_dim_data)

        def count_rows(d):
            total = 0
            for _, v in d.items():
                if isinstance(v, dict):
                    total += count_rows(v)
                elif hasattr(v, "__len__") and not isinstance(v, str):
                    total += len(v)
            return total

        total_rows = count_rows(all_low_dim_data)
        log_print(f"[HDF5] 准备写入 aligned data: 约 {total_rows} 行数据")

        PostProcessorUtils.save_to_hdf5(
            all_low_dim_data,
            os.path.join(proprio_stats_dir, "proprio_stats.hdf5"),
        )

        if all_low_dim_data_original is not None:
            total_rows_raw = count_rows(all_low_dim_data_original)
            log_print(f"[HDF5] 准备写入 raw data: 约 {total_rows_raw} 行数据")
            PostProcessorUtils.save_to_hdf5(
                all_low_dim_data_original,
                os.path.join(proprio_stats_dir, "proprio_stats_original.hdf5"),
            )

        extrinsic_hdf5_path = os.path.join(proprio_stats_dir, "proprio_stats.hdf5")
        with h5py.File(extrinsic_hdf5_path, "a") as f:
            group = f.require_group("camera_extrinsic_params")
            for cam_key, extrinsics in [
                ("head_camera", head_extrinsics),
                ("left_hand_camera", left_extrinsics),
                ("right_hand_camera", right_extrinsics),
            ]:
                if not extrinsics:
                    continue
                rot = np.array([x["rotation_matrix"] for x in extrinsics], dtype=np.float32)
                trans = np.array(
                    [x["translation_vector"] for x in extrinsics], dtype=np.float32
                )
                ts_seconds = np.array([x["timestamp"] for x in extrinsics], dtype=np.float64)
                ts_nanoseconds = (ts_seconds * 1e9).astype(np.int64)
                cam_group = group.require_group(cam_key)
                cam_group.create_dataset("camera_rotation_matrix", data=rot)
                cam_group.create_dataset("camera_translation_vector", data=trans)
                cam_group.create_dataset("index", data=ts_nanoseconds)

        compressed_group = {}
        uncompressed_group = {}
        for cam, imgs in imgs_per_cam_depth.items():
            is_compressed = compressed.get(cam)
            if is_compressed is True:
                compressed_group[cam] = imgs
            elif is_compressed is False:
                uncompressed_group[cam] = imgs

        if compressed_group and raw_config.enhance_enabled:
            log_print(
                f"[INFO] 以下相机为压缩深度，将使用增强处理输出16位视频: {list(compressed_group.keys())}"
            )
            save_depth_videos_enhanced_parallel(
                compressed_group,
                imgs_per_cam,
                output_dir=depth_dir,
                raw_config=raw_config,
            )

        use_streaming = getattr(raw_config, "use_streaming_video", True)
        queue_limit = getattr(raw_config, "video_queue_limit", 300)

        if uncompressed_group:
            if use_streaming:
                depth_stats = save_depth_videos_16U_streaming(
                    uncompressed_group,
                    output_dir=depth_dir,
                    raw_config=raw_config,
                    queue_limit=queue_limit,
                )
                for cam, stats in depth_stats.items():
                    log_print(
                        f"  [{cam}] 帧数: {stats['written_count']}, 阻塞次数: {stats['block_count']}"
                    )
            else:
                save_depth_videos_16U_parallel(
                    uncompressed_group,
                    output_dir=depth_dir,
                    raw_config=raw_config,
                )

        save_camera_info_to_json_new(info_per_cam, distortion_model, output_dir=parameters_dir)

        if use_streaming:
            color_stats = save_color_videos_streaming(
                imgs_per_cam,
                output_dir=video_dir,
                raw_config=raw_config,
                queue_limit=queue_limit,
            )
            for cam, stats in color_stats.items():
                log_print(
                    f"  [{cam}] 帧数: {stats['written_count']}, 阻塞次数: {stats['block_count']}"
                )
        else:
            save_color_videos_parallel(imgs_per_cam, output_dir=video_dir, raw_config=raw_config)

        save_camera_extrinsic_params(cameras=["head_cam_h", "wrist_cam_r", "wrist_cam_l"], output_dir=parameters_dir)

        merge_metadata_and_moment(
            metadata_json_dir,
            os.path.join(task_info_dir, "metadata.json"),
            uuid,
            raw_config,
            bag_time_info=bag_time_info,
            main_time_line_timestamps=main_time_line_timestamps,
            output_dir=output_dir,
        )

        temp_uuid_path = os.path.join(output_dir, uuid)
        if not os.path.exists(temp_uuid_path):
            raise Exception(f"Episode {uuid} 临时路径不存在: {temp_uuid_path}")

        log_print(f"开始验证 episode {uuid} 的数据一致性...")
        validation_result = validate_episode_data_consistency(temp_uuid_path)
        if validation_result is None:
            raise Exception(f"Episode {uuid} 数据一致性验证无法完成")
        if not validation_result["is_consistent"]:
            inconsistencies = validation_result.get("inconsistencies", [])
            error_details = []
            for inc in inconsistencies:
                error_details.append(
                    f"{inc['type']} {inc['camera']}: 期望{inc['expected']}帧, 实际{inc['actual']}帧, 差异{inc['difference']:+d}帧"
                )
            raise Exception(
                f"Episode {uuid} 数据一致性验证失败: {'; '.join(error_details)}"
            )

        log_print(f"Episode {uuid} 数据一致性验证通过 ✓")


def port_kuavo_rosbag(
    raw_config: Config,
    bag_dir: str,
    metadata_json_dir: str,
    output_dir: str,
    scene: str,
    sub_scene: str,
    continuous_action: str,
    mode: str,
    min_duration=5.0,
):
    log_print(
        "[CONFIG] dataset:"
        f" use_videos={DEFAULT_DATASET_CONFIG.use_videos}"
        f" image_writer_processes={DEFAULT_DATASET_CONFIG.image_writer_processes}"
        f" image_writer_threads={DEFAULT_DATASET_CONFIG.image_writer_threads}"
        f" video_backend={DEFAULT_DATASET_CONFIG.video_backend}"
    )

    bag_files = list_bag_files_auto(bag_dir)
    generate_dataset_file(
        bag_files=bag_files,
        raw_config=raw_config,
        metadata_json_dir=metadata_json_dir,
        output_dir=output_dir,
        scene=scene,
        sub_scene=sub_scene,
        continuous_action=continuous_action,
        mode=mode,
        min_duration=min_duration,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Kuavo ROSbag to HDF5 Converter")
    parser.add_argument("--bag_dir", default="test/", type=str, help="Path to ROS bag dir")
    parser.add_argument("--metadata_json_dir", type=str, help="Path to metadata.json")
    parser.add_argument("--output_dir", default="testoutput/", type=str, help="Path to output")
    parser.add_argument("--scene", default="test_scene", type=str)
    parser.add_argument("--sub_scene", default="test_sub_scene", type=str)
    parser.add_argument("--continuous_action", default="test_continuous_action", type=str)
    parser.add_argument("--mode", default="simplified", type=str)
    parser.add_argument("--min_duration", type=float, default=5.0)
    parser.add_argument("-v", "--process_ID", default="v0", type=str)
    parser.add_argument("--config", type=str, default="configs/request.json")
    return parser.parse_args()


def main():
    args = _parse_args()
    config_path = args.config
    if not os.path.exists(config_path) and os.path.basename(config_path) == "request.json":
        fallback_path = os.path.join("configs", "request.json")
        if os.path.exists(fallback_path):
            log_print(f"[INFO] 配置文件 {config_path} 不存在，自动回退到 {fallback_path}")
            config_path = fallback_path
    config = load_config_from_json(config_path)
    config.id = args.process_ID

    bag_dir = args.bag_dir
    metadata_json_dir = args.metadata_json_dir or os.path.join(bag_dir, "metadata.json")

    if args.mode not in {"complete", "simplified"}:
        raise ValueError("Invalid mode. Choose either 'complete' or 'simplified'.")

    port_kuavo_rosbag(
        config,
        bag_dir,
        metadata_json_dir,
        args.output_dir,
        args.scene,
        args.sub_scene,
        args.continuous_action,
        args.mode,
        args.min_duration,
    )


if __name__ == "__main__":
    main()
