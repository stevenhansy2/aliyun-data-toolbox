"""Finalization helpers for each streamed batch."""

from __future__ import annotations

import gc
import json
import os
import time

from converter.data_utils import merge_metadata_and_moment
from converter.slave_utils import (
    move_and_rename_depth_videos,
    save_camera_extrinsic_params,
    save_camera_info_to_json_new,
    save_depth_videos_16U_parallel,
    save_depth_videos_enhanced_parallel,
)
from converter.video_pipeline import save_image_bytes_to_temp


def persist_batch_media(
    raw_config,
    *,
    episode_uuid: str,
    batch_id: int,
    batch_root: str,
    cameras: list[str],
    compressed: dict | None,
    imgs_per_cam: dict,
    imgs_per_cam_depth: dict | None,
    pipeline_encoder,
    streaming_encoder,
    cam_stats,
):
    separate_video_storage = getattr(raw_config, "separate_video_storage", False)
    save_images_end_time = None

    if separate_video_storage:
        temp_video_dir = os.path.join("/tmp", "kuavo_video_temp", episode_uuid)
        if streaming_encoder is not None:
            cam_stats = streaming_encoder.feed_batch(imgs_per_cam, batch_id)
            if imgs_per_cam_depth:
                save_image_bytes_to_temp({}, imgs_per_cam_depth, temp_video_dir, batch_id)
        else:
            cam_stats = save_image_bytes_to_temp(
                imgs_per_cam, imgs_per_cam_depth, temp_video_dir, batch_id
            )
            if pipeline_encoder is not None:
                pipeline_encoder.submit_batch(batch_id)

        save_images_end_time = time.time()
        del imgs_per_cam, imgs_per_cam_depth
        gc.collect()
        print(f"[MEMORY] 批次{batch_id} 图像数据已释放")
        imgs_per_cam = None
        imgs_per_cam_depth = None
    else:
        depth_dir = os.path.join(batch_root, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        compressed_group = {
            cam: imgs_per_cam_depth[cam] for cam in cameras if compressed.get(cam, None) is True
        }
        uncompressed_group = {
            cam: imgs_per_cam_depth[cam]
            for cam in cameras
            if compressed.get(cam, None) is False
        }

        if compressed_group and raw_config.enhance_enabled:
            save_depth_videos_enhanced_parallel(
                compressed_group,
                imgs_per_cam,
                output_dir=depth_dir,
                raw_config=raw_config,
            )
        if uncompressed_group:
            save_depth_videos_16U_parallel(
                uncompressed_group,
                output_dir=depth_dir,
                raw_config=raw_config,
            )
        move_and_rename_depth_videos(depth_dir, episode_idx=0)

    return (
        cam_stats,
        save_images_end_time,
        separate_video_storage,
        imgs_per_cam,
        imgs_per_cam_depth,
    )


def save_first_batch_parameters(
    *,
    batch_id: int,
    batch_root: str,
    info_per_cam: dict,
    distortion_model: dict,
    cameras: list[str],
):
    if batch_id != 1:
        return

    parameters_dir = os.path.join(batch_root, "parameters")
    os.makedirs(parameters_dir, exist_ok=True)
    save_camera_info_to_json_new(info_per_cam, distortion_model, output_dir=parameters_dir)
    save_camera_extrinsic_params(cameras=cameras, output_dir=parameters_dir)


def save_batch_metadata_json(
    *,
    metadata_json_path: str | None,
    moment_json_path: str | None,
    batch_root: str,
    episode_uuid: str,
    raw_config,
    bag_time_info: dict,
    main_ts,
):
    try:
        if metadata_json_path is not None and os.path.exists(metadata_json_path):
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                test_metadata = json.load(f)
            is_new_format = "marks" in test_metadata and isinstance(
                test_metadata.get("marks"), list
            )

            if is_new_format:
                merge_metadata_and_moment(
                    metadata_json_path,
                    None,
                    os.path.join(batch_root, "metadata.json"),
                    episode_uuid,
                    raw_config,
                    bag_time_info=bag_time_info,
                    main_time_line_timestamps=main_ts,
                )
            elif moment_json_path is not None and os.path.exists(moment_json_path):
                merge_metadata_and_moment(
                    metadata_json_path,
                    moment_json_path,
                    os.path.join(batch_root, "metadata.json"),
                    episode_uuid,
                    raw_config,
                    bag_time_info=bag_time_info,
                    main_time_line_timestamps=main_ts,
                )
            else:
                print(
                    f"[WARN] 旧格式需要 moments.json，但未找到: moment_json_DIR={moment_json_path}"
                )
        else:
            print(
                f"[WARN] 未生成批次 metadata.json，metadata_json_DIR={metadata_json_path}"
            )
    except Exception as e:
        print(f"[ERROR] 合并 metadata 和 moment 失败: {e}")
        import traceback

        traceback.print_exc()
