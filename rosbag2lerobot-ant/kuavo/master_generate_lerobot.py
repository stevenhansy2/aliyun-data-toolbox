"""
Script to convert Kuavo rosbag data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
现在id是唯一指示版本变量，修改了入参的结构，添加了描述信息至每个bag的每个step中，添加了使用ks_standard下载bag，通过限制线程个数减少内存占用，为最新版本。对应json入参为 request_new2.json
"""

from collections import OrderedDict
import custom_patches
import uuid
import psutil
import gc
import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import sys
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import json
from config_dataset_slave import Config, load_config_from_json
import argparse
import requests
import time
import uuid
import joblib
import gc
from copy import deepcopy
from slave_utils import (
    move_and_rename_depth_videos,
    save_camera_extrinsic_params,
    save_camera_info_to_json_new,
    save_depth_videos_16U_parallel,
    save_depth_videos_enhanced_parallel,
    swap_left_right_data_if_needed,
    flip_camera_arrays_if_needed,
)
from kuavo_dataset_slave import (
    KuavoRosbagReader,
    DEFAULT_JOINT_NAMES_LIST,
    DEFAULT_LEG_JOINT_NAMES,
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_HEAD_JOINT_NAMES,
    # DEFAULT_CAMERA_NAMES,
    DEFAULT_JOINT_NAMES,
    DEFAULT_LEJUCLAW_JOINT_NAMES,
    DEFAULT_DEXHAND_JOINT_NAMES,
    PostProcessorUtils,
)
import zipfile
import datetime
import einops
from math import ceil
from copy import deepcopy
import rosbag
import cv2
import os
import shutil
import concurrent.futures
import tempfile
from pathlib import Path

import requests
import time
import uuid
from copy import deepcopy
import logging

import os
import time
import subprocess

LEROBOT_HOME = HF_LEROBOT_HOME


def get_nested_value(data, path, i=None, default=None):
    """
    从嵌套字典中通过路径字符串提取数据，并支持按帧索引和默认值。
    path: 例如 "state.head.position"
    i: 帧索引，如果为 None 则返回整个数组
    default: 默认值（如 [0.0]*2）
    """
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
        else:
            return torch.tensor(v, dtype=torch.float32)
    except Exception:
        return torch.tensor(default, dtype=torch.float32)


# 用法示例：
# get_nested_value(all_low_dim_data, "state.head.position", i, [0.0]*2)


def is_valid_hand_data(arr, expected_shape=None):
    arr = np.array(arr) if arr is not None else None
    if arr is None or arr.size == 0:
        return False
    if expected_shape is not None and arr.shape[1:] != expected_shape:
        return False
    return True


def calculate_action_frames(
    rosbag_actual_start_time,  # 实际数据开始时间
    rosbag_actual_end_time,  # 实际数据结束时间
    rosbag_original_start_time,  # 原始bag开始时间
    rosbag_original_end_time,  # 原始bag结束时间
    action_original_start_time,  # 动作原始开始时间
    action_duration,  # 动作持续时间
    frame_rate,  # 帧率
    total_frames,  # 总帧数
):
    """
    计算动作的开始帧和结束帧

    策略：
    1. 计算动作在原始时间轴上的绝对时间范围
    2. 将这个时间范围映射到实际数据的时间范围
    3. 根据实际数据的时间范围计算对应的帧数
    """

    # 1. 计算动作的绝对时间范围
    action_start_time = action_original_start_time
    action_end_time = action_original_start_time + action_duration

    # 2. 检查动作时间是否在实际数据范围内
    if (
        action_end_time < rosbag_actual_start_time
        or action_start_time > rosbag_actual_end_time
    ):
        # 动作完全在实际数据范围之外
        return None, None

    # 3. 将动作时间范围限制在实际数据范围内
    clipped_action_start = max(action_start_time, rosbag_actual_start_time)
    clipped_action_end = min(action_end_time, rosbag_actual_end_time)

    # 4. 计算相对于实际数据开始时间的偏移
    start_offset = clipped_action_start - rosbag_actual_start_time
    end_offset = clipped_action_end - rosbag_actual_start_time

    # 5. 根据实际数据的时间范围计算帧数
    actual_data_duration = rosbag_actual_end_time - rosbag_actual_start_time

    # 方法1：按时间比例计算
    start_frame = int((start_offset / actual_data_duration) * total_frames)
    end_frame = int((end_offset / actual_data_duration) * total_frames)

    # 方法2：按帧率计算（更精确）
    # start_frame = int(start_offset * frame_rate)
    # end_frame = int(end_offset * frame_rate)

    # 6. 确保帧数在有效范围内
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    return start_frame, end_frame


def merge_metadata_and_moment(
    metadata_path,
    moment_path,
    output_path,
    uuid,
    raw_config,
    bag_time_info=None,
    main_time_line_timestamps=None,
):
    """
    合并 metadata 和 moment 数据，并添加 bag 时间信息和计算帧数
    Args:
        metadata_path: metadata.json 文件路径
        moment_path: moment.json 文件路径
        output_path: 输出文件路径
        uuid: 唯一标识符
        raw_config: 原始配置对象
        bag_time_info: bag时间信息字典（可选）
        main_time_line_timestamps: 经过帧率对齐后的时间戳数组（纳秒）
    """
    frequency = raw_config.train_hz if hasattr(raw_config, "train_hz") else 30

    # 读取 metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)

    # 读取 moment.json
    with open(moment_path, "r", encoding="utf-8") as f:
        moment = json.load(f)

    # 转换新格式 metadata 为旧格式
    converted_metadata = {}

    # scene_name 对应 scene_code
    converted_metadata["scene_name"] = raw_metadata.get("scene_code", "")

    # sub_scene_name 对应 sub_scene_code (如果不存在则为空)
    converted_metadata["sub_scene_name"] = raw_metadata.get("sub_scene_code", "")

    # init_scene_text 对应 sub_scene_code (如果不存在则为空)
    converted_metadata["init_scene_text"] = raw_metadata.get("sub_scene_zh_dec", "")

    # english_init_scene_text 对应 scene_en_dec
    converted_metadata["english_init_scene_text"] = raw_metadata.get(
        "sub_scene_en_dec", ""
    )

    # task_name 优先 task_group_name 其次 task_name
    task_name = raw_metadata.get("task_group_name")
    if not task_name:  # 如果为空或不存在
        task_name = raw_metadata.get("task_name", "")
    converted_metadata["task_name"] = task_name

    # english_task_name 优先 task_group_code 其次 task_code
    english_task_name = raw_metadata.get("task_group_code")
    if not english_task_name:  # 如果为空或不存在
        english_task_name = raw_metadata.get("task_code", "")
    converted_metadata["english_task_name"] = english_task_name
    if isinstance(english_task_name, str) and "_" in english_task_name:
        english_task_name = english_task_name.replace("_", " ")
    converted_metadata["english_task_name"] = english_task_name

    # 默认值字段
    converted_metadata["data_type"] = "常规"
    converted_metadata["episode_status"] = "approved"
    converted_metadata["data_gen_mode"] = "real_machine"

    # sn_code 对应 device_sn
    converted_metadata["sn_code"] = raw_metadata.get("device_sn", "")

    # sn_name 默认值
    converted_metadata["sn_name"] = "乐聚机器人"

    print(f"Metadata 字段转换结果:")
    for key, value in converted_metadata.items():
        print(f"  {key}: '{value}'")

    # 使用转换后的 metadata
    metadata = converted_metadata

    # 获取时间信息
    rosbag_actual_start_time = None
    rosbag_actual_end_time = None
    rosbag_original_start_time = None
    rosbag_original_end_time = None
    total_frames = 0

    # 实际数据时间范围
    if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
        # 调试：打印原始时间戳
        print(f"原始时间戳前3个: {main_time_line_timestamps[:3]}")
        print(f"原始时间戳后3个: {main_time_line_timestamps[-3:]}")

        # 检查时间戳是否已经是秒格式还是纳秒格式
        if main_time_line_timestamps[0] > 1e12:  # 如果大于1e12，认为是纳秒格式
            timestamps_seconds = main_time_line_timestamps / 1e9
            print("时间戳格式：纳秒 -> 秒")
        else:
            timestamps_seconds = main_time_line_timestamps
            print("时间戳格式：已经是秒")

        rosbag_actual_start_time = timestamps_seconds[0]
        rosbag_actual_end_time = timestamps_seconds[-1]
        total_frames = len(main_time_line_timestamps)

        # 调试：打印转换后的时间戳
        print(f"转换后时间戳前3个: {timestamps_seconds[:3]}")
        print(f"转换后时间戳后3个: {timestamps_seconds[-3:]}")

        # 验证时间戳转换
        start_datetime = datetime.datetime.fromtimestamp(
            rosbag_actual_start_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )
        end_datetime = datetime.datetime.fromtimestamp(
            rosbag_actual_end_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )

        print(f"实际开始时间验证: {start_datetime.isoformat()}")
        print(f"实际结束时间验证: {end_datetime.isoformat()}")

    # 原始bag时间范围
    if bag_time_info:
        rosbag_original_start_time = bag_time_info.get("unix_timestamp")
        rosbag_original_end_time = bag_time_info.get("end_time")

    # 构造 action_config
    print(f"时间信息:")
    if rosbag_original_start_time and rosbag_original_end_time:
        print(
            f"  原始bag时间: {rosbag_original_start_time:.6f}s - {rosbag_original_end_time:.6f}s"
        )
    if rosbag_actual_start_time and rosbag_actual_end_time:
        print(
            f"  实际数据时间: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
        )
    print(f"  总帧数: {total_frames}")

    action_config = []

    for m in moment.get("moments", []):
        # 从新格式的 customFieldValues 中提取数据
        custom_fields = m.get("customFieldValues", {})

        trigger_time = m.get("triggerTime", "")
        duration_str = m.get("duration", "0s")

        # 格式化时间戳：将 "Z" 替换为 "+00:00"
        formatted_trigger_time = (
            trigger_time.replace("Z", "+00:00") if trigger_time else ""
        )

        # 添加调试信息
        print(f"处理动作数据:")
        print(f"  skill_atomic_en: {custom_fields.get('skill_atomic_en', '')}")
        print(f"  skill_detail: {custom_fields.get('skill_detail', '')}")
        print(f"  en_skill_detail: {custom_fields.get('en_skill_detail', '')}")
        print(f"  原始时间戳: {trigger_time}")
        print(f"  格式化时间戳: {formatted_trigger_time}")

        start_frame = None
        end_frame = None

        if (
            rosbag_actual_start_time is not None
            and rosbag_actual_end_time is not None
            and trigger_time
        ):

            try:
                # 解析动作时间
                trigger_datetime = datetime.datetime.fromisoformat(
                    trigger_time.replace("Z", "+00:00")
                )
                action_original_start_time = trigger_datetime.timestamp()

                # 解析持续时间
                action_duration = 0
                if duration_str.endswith("s"):
                    action_duration = float(duration_str[:-1])

                # 计算帧数
                start_frame, end_frame = calculate_action_frames(
                    rosbag_actual_start_time=rosbag_actual_start_time,
                    rosbag_actual_end_time=rosbag_actual_end_time,
                    rosbag_original_start_time=rosbag_original_start_time,
                    rosbag_original_end_time=rosbag_original_end_time,
                    action_original_start_time=action_original_start_time,
                    action_duration=action_duration,
                    frame_rate=frequency,
                    total_frames=total_frames,
                )

                print(f"动作: {custom_fields.get('skill_detail', '')}")
                print(f"  动作时间: {trigger_datetime.isoformat()}")
                print(f"  原始开始时间: {action_original_start_time:.6f}s")
                print(
                    f"  原始结束时间: {action_original_start_time + action_duration:.6f}s"
                )
                print(f"  持续时间: {action_duration:.3f}s")
                print(f"  计算得到帧数: {start_frame} - {end_frame}")

                # 更详细的调试信息
                if start_frame is None or end_frame is None:
                    print(f"  调试信息:")
                    print(
                        f"    实际数据范围: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
                    )
                    print(
                        f"    动作时间范围: {action_original_start_time:.6f}s - {action_original_start_time + action_duration:.6f}s"
                    )
                    print(
                        f"    动作是否在数据范围内: {action_original_start_time >= rosbag_actual_start_time and action_original_start_time + action_duration <= rosbag_actual_end_time}"
                    )
                    print(
                        f"    动作开始是否在数据范围后: {action_original_start_time > rosbag_actual_end_time}"
                    )
                    print(
                        f"    动作结束是否在数据范围前: {action_original_start_time + action_duration < rosbag_actual_start_time}"
                    )

                # 验证计算结果
                if start_frame is not None and end_frame is not None:
                    actual_start_time = rosbag_actual_start_time + (
                        start_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    actual_end_time = rosbag_actual_start_time + (
                        end_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    print(f"  验证-实际开始时间: {actual_start_time:.6f}s")
                    print(f"  验证-实际结束时间: {actual_end_time:.6f}s")

                print("-" * 50)

            except Exception as e:
                print(f"计算帧数时出错: {e}")
                import traceback

                traceback.print_exc()

        # 构造新的 action 对象
        action = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "timestamp_utc": formatted_trigger_time,  # 使用格式化后的时间戳
            "is_mistake": False,  # 直接设置为 False
            "skill": custom_fields.get(
                "skill_atomic_en", ""
            ),  # 从 skill_atomic_en 获取
            "action_text": custom_fields.get(
                "skill_detail", ""
            ),  # 从 skill_detail 获取
            "english_action_text": custom_fields.get(
                "en_skill_detail", ""
            ),  # 从 en_skill_detail 获取
        }
        action_config.append(action)

    # 按照 timestamp_utc 排序
    action_config = sorted(
        action_config, key=lambda x: x["timestamp_utc"] if x["timestamp_utc"] else ""
    )

    # 构造新json，episode_id放在最前
    new_json = OrderedDict()
    new_json["episode_id"] = uuid

    # 使用转换后的 metadata
    for k, v in metadata.items():
        new_json[k] = v

    if "label_info" not in new_json:
        new_json["label_info"] = {}
    new_json["label_info"]["action_config"] = action_config
    if "key_frame" not in new_json["label_info"]:
        new_json["label_info"]["key_frame"] = []

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)
    print(f"已保存到 {output_path}")


def get_time_range_from_moments(moments_json_path):
    """
    从 moments.json 文件中读取时间范围

    Args:
        moments_json_path: moments.json 文件路径

    Returns:
        tuple: (start_time, end_time) 或 (None, None) 如果失败
    """
    if not moments_json_path or not os.path.exists(moments_json_path):
        return None, None

    try:
        with open(moments_json_path, "r", encoding="utf-8") as f:
            moments_data = json.load(f)

        moments = moments_data.get("moments", [])
        if not moments:
            print(f"[MOMENTS] moments.json中未找到moments数据")
            return None, None

        start_positions = []
        end_positions = []

        for moment in moments:
            custom_fields = moment.get("customFieldValues", {})
            start_pos = custom_fields.get("start_position")
            end_pos = custom_fields.get("end_position")

            if start_pos is not None:
                try:
                    start_positions.append(float(start_pos))
                except (ValueError, TypeError):
                    print(f"[MOMENTS] 无效的start_position值: {start_pos}")
                    pass

            if end_pos is not None:
                try:
                    end_positions.append(float(end_pos))
                except (ValueError, TypeError):
                    print(f"[MOMENTS] 无效的end_position值: {end_pos}")
                    pass

        # 使用最早的start_position和最晚的end_position
        if start_positions and end_positions:
            moments_start_time = min(start_positions)
            moments_end_time = max(end_positions)

            print(
                f"[MOMENTS] 从moments.json获取时间范围: {moments_start_time} - {moments_end_time}"
            )
            print(
                f"[MOMENTS] 找到 {len(start_positions)} 个start_position, {len(end_positions)} 个end_position"
            )

            return moments_start_time, moments_end_time
        else:
            print(f"[MOMENTS] moments.json中未找到有效的时间位置信息")
            return None, None

    except Exception as e:
        print(f"[MOMENTS] 读取moments.json时出错: {e}")
        return None, None


def get_bag_time_info(bag_path: str) -> dict:
    """
    获取 rosbag 包的时间信息

    Args:
        bag_path: rosbag 文件路径

    Returns:
        dict: 包含时间信息的字典，包括：
            - unix_timestamp: Unix时间戳（秒）
            - iso_format: ISO格式时间字符串（东八区）
            - nanoseconds: 纳秒格式时间戳
            - duration: bag持续时间（秒）
            - end_time: 结束时间Unix时间戳
    """
    try:
        bag = rosbag.Bag(bag_path, "r")
        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        bag_duration = bag_end_time - bag_start_time
        bag.close()

        # 转换为带时区的ISO格式（东八区）
        start_datetime = datetime.datetime.fromtimestamp(
            bag_start_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )
        start_iso = start_datetime.isoformat()

        # 转换为纳秒
        start_nanoseconds = int(bag_start_time * 1e9)

        return {
            "unix_timestamp": bag_start_time,
            "iso_format": start_iso,
            "nanoseconds": start_nanoseconds,
            "duration": bag_duration,
            "end_time": bag_end_time,
        }

    except Exception as e:
        print(f"获取bag时间信息失败: {e}")
        return {
            "unix_timestamp": None,
            "iso_format": None,
            "nanoseconds": None,
            "duration": None,
            "end_time": None,
        }


def list_bag_files_auto(raw_dir):
    bag_files = []
    for i, fname in enumerate(sorted(os.listdir(raw_dir))):
        if fname.endswith(".bag"):
            bag_files.append(
                {
                    "link": "",  # 保持为空
                    "start": 0,  # 批量设置为0
                    "end": 1,  # 批量设置为1
                    "local_path": os.path.join(raw_dir, fname),
                }
            )
    return bag_files


def load_raw_depth_lerobot(
    bag_data: dict, default_camera_names: list[str]
) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in default_camera_names:
        key = f"{camera}_depth"
        imgs_per_cam[camera] = np.array([msg["data"] for msg in bag_data[key]])
        # print(f"camera {camera} image", imgs_per_cam[camera].shape)

    return imgs_per_cam


def load_raw_depth_images_per_camera(bag_data: dict, default_camera_names: list[str]):
    imgs_per_cam = {}
    compressed_per_cam = {}
    for camera in default_camera_names:
        key = f"{camera}_depth"
        imgs_per_cam[camera] = [msg["data"] for msg in bag_data[key]]
        # 只取第一帧的压缩状态（假设所有帧一致）
        if bag_data[key]:
            compressed_per_cam[camera] = bag_data[key][0].get("compressed", None)
        else:
            compressed_per_cam[camera] = None
    print("+" * 20, compressed_per_cam)
    return imgs_per_cam, compressed_per_cam


def load_camera_info_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict:
    info_per_cam = {}
    distortion_model = {}
    for camera in default_camera_names:
        info_per_cam[camera] = np.array(
            [msg["data"] for msg in bag_data[f"{camera}_camera_info"]], dtype=np.float32
        )
        distortion_model[camera] = [
            msg["distortion_model"] for msg in bag_data[f"{camera}_camera_info"]
        ]
    return info_per_cam, distortion_model


def load_raw_images_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict[str, list]:
    imgs_per_cam = {}
    for camera in default_camera_names:
        imgs_per_cam[camera] = [msg["data"] for msg in bag_data[camera]]
    return imgs_per_cam


def load_raw_episode_data(
    raw_config: Config,
    ep_path: Path,
    start_time: float = 0,
    end_time: float = 1,
    action_config=None,
    min_duration: float = 5.0,
    metadata_json_dir: str = None,
):
    sn_code = None
    if metadata_json_dir and os.path.exists(metadata_json_dir):
        try:
            with open(metadata_json_dir, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
            sn_code = raw_metadata.get("device_sn", "")
        except Exception as e:
            print(f"[WARN] 读取metadata.json失败: {e})")
    bag_reader = KuavoRosbagReader(raw_config)
    bag_data = bag_reader.process_rosbag(
        ep_path, start_time=start_time, end_time=end_time, action_config=action_config
    )
    if sn_code is not None:
        main_time_line_timestamps = None
        if "head_cam_h" in bag_data and len(bag_data["head_cam_h"]) > 0:
            main_time_line_timestamps = np.array(
                [msg["timestamp"] for msg in bag_data["head_cam_h"]]
            )
        else:
            main_time_line_timestamps = None
        swap_left_right_data_if_needed(bag_data, sn_code, main_time_line_timestamps)
    # 1. 处理完 bag_data 后立即提取所需数据并清理
    sensors_data_raw__joint_q = state = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_q"]],
        dtype=np.float32,
    )
    joint_cmd__joint_q = action = np.array(
        [msg["data"] for msg in bag_data["action.joint_cmd.joint_q"]], dtype=np.float32
    )
    kuavo_arm_traj__position = action_kuavo_arm_traj = np.array(
        [msg["data"] for msg in bag_data["action.kuavo_arm_traj"]], dtype=np.float32
    )

    # 手部数据
    leju_claw_state__position = claw_state = np.array(
        [msg["data"] for msg in bag_data["observation.claw"]], dtype=np.float32
    )
    leju_claw_command__position = claw_action = np.array(
        [msg["data"] for msg in bag_data["action.claw"]], dtype=np.float32
    )

    # control_robot_hand_position_state_both = qiangnao_state = np.array(
    #     [msg["data"] for msg in bag_data["observation.qiangnao"]], dtype=np.float32
    # )
    # control_robot_hand_position_both = qiangnao_action = np.array(
    #     [msg["data"] for msg in bag_data["action.qiangnao"]], dtype=np.float32
    # )
    try:
        control_robot_hand_position_state_both = qiangnao_state = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao"]], dtype=np.float32
        )
    except KeyError:
        print("[WARN] 未找到 'observation.qiangnao' 数据，使用空值")
        control_robot_hand_position_state_both = qiangnao_state = None

    try:
        control_robot_hand_position_both = qiangnao_action = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao"]], dtype=np.float32
        )
    except KeyError:
        print("[WARN] 未找到 'action.qiangnao' 数据，使用空值")
        control_robot_hand_position_both = qiangnao_action = None

    # 速度和电流数据
    sensors_data_raw__joint_v = state_joint_v = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_v"]],
        dtype=np.float32,
    )
    state_joint_current = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_current"]],
        dtype=np.float32,
    )

    # 图像数据
    import psutil

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"[内存] 提取图像前: {mem_before:.1f} MB")

    imgs_per_cam = load_raw_images_per_camera(bag_data, raw_config.default_camera_names)
    mem_after_color = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 彩色图像提取后: {mem_after_color:.1f} MB (增长 {mem_after_color - mem_before:.1f} MB)"
    )

    imgs_per_cam_depth, compressed = load_raw_depth_images_per_camera(
        bag_data, raw_config.default_camera_names
    )
    mem_after_depth = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 深度图像提取后: {mem_after_depth:.1f} MB (增长 {mem_after_depth - mem_after_color:.1f} MB)"
    )

    info_per_cam, distortion_model = load_camera_info_per_camera(
        bag_data, raw_config.default_camera_names
    )
    mem_after_info = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 相机信息提取后: {mem_after_info:.1f} MB (增长 {mem_after_info - mem_after_depth:.1f} MB)"
    )
    main_time_line_timestamps = np.array(
        [msg["timestamp"] for msg in bag_data["head_cam_h"]]
    )
    if sn_code is not None:
        imgs_per_cam, imgs_per_cam_depth = flip_camera_arrays_if_needed(
            imgs_per_cam, imgs_per_cam_depth, sn_code, main_time_line_timestamps[0]
        )
    else:
        print("[WARN] 未提供sn_code，跳过相机翻转检测")
    # 时间戳和相机外参

    head_extrinsics = bag_data.get("head_camera_extrinsics", [])
    left_extrinsics = bag_data.get("left_hand_camera_extrinsics", [])
    right_extrinsics = bag_data.get("right_hand_camera_extrinsics", [])
    end_position = np.array(
        [msg["data"] for msg in bag_data["end.position"]], dtype=np.float32
    )
    end_orientation = np.array(
        [msg["data"] for msg in bag_data["end.orientation"]], dtype=np.float32
    )
    sensors_data_raw__imu_data = state_joint_imu = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.imu"]],
        dtype=np.float32,
    )

    # 2. 立即清理 bag_data 和 bag_reader
    mem_before_del = process.memory_info().rss / 1024 / 1024
    print(f"[内存] 删除 bag_data 前: {mem_before_del:.1f} MB")

    del bag_data
    del bag_reader
    gc.collect()

    mem_after_del = process.memory_info().rss / 1024 / 1024
    print(
        f"[内存] 删除 bag_data 后: {mem_after_del:.1f} MB (释放 {mem_before_del - mem_after_del:.1f} MB)"
    )

    # 3. 处理电机数据（这些计算比较消耗内存）
    action[:, 12:26] = action_kuavo_arm_traj
    del action_kuavo_arm_traj  # 立即删除临时变量

    sensors_data_raw__joint_effort = state_joint_effort = (
        PostProcessorUtils.current_to_torque_batch(
            state_joint_current,
            MOTOR_C2T=[
                2,
                1.05,
                1.05,
                2,
                2.1,
                2.1,
                2,
                1.05,
                1.05,
                2,
                2.1,
                2.1,
                1.05,
                5,
                2.3,
                5,
                4.7,
                4.7,
                4.7,
                1.05,
                5,
                2.3,
                5,
                4.7,
                4.7,
                4.7,
                0.21,
                4.7,
            ],
        )
    )

    sensors_data_raw__joint_current = PostProcessorUtils.torque_to_current_batch(
        state_joint_current,
        MOTOR_C2T=[
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            2,
            1.05,
            1.05,
            2,
            2.1,
            2.1,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            1.05,
            5,
            2.3,
            5,
            4.7,
            4.7,
            4.7,
            0.21,
            4.7,
        ],
    )

    # 4. 提取子数组并清理原始数组
    head_effort = sensors_data_raw__joint_effort[:, 26:28]
    head_current = sensors_data_raw__joint_current[:, 26:28]
    joint_effort = sensors_data_raw__joint_effort[:, 12:26]
    joint_current = sensors_data_raw__joint_current[:, 12:26]

    # 清理一些不再需要的临时变量
    del state_joint_current
    gc.collect()

    # 5. 处理时间戳
    main_time_line_timestamps_ns = (main_time_line_timestamps * 1e9).astype(np.int64)

    velocity = None
    effort = None

    # 6. 构建 all_low_dim_data（这是返回的主要数据结构）
    all_low_dim_data = {
        "timestamps": main_time_line_timestamps_ns,
        "action": {
            "effector": {
                "position(gripper)": leju_claw_command__position,
                "position(dexhand)": control_robot_hand_position_both,
                "index": main_time_line_timestamps_ns,
            },
            "joint": {
                "position": kuavo_arm_traj__position,
                "index": main_time_line_timestamps_ns,
            },
            "head": {
                "position": joint_cmd__joint_q[:, 26:28],
                "index": main_time_line_timestamps_ns,
            },
            "leg": {
                "position": joint_cmd__joint_q[:, :12],
                "index": main_time_line_timestamps_ns,
            },
        },
        "state": {
            "effector": {
                "position(gripper)": leju_claw_state__position,
                "position(dexhand)": control_robot_hand_position_state_both,
            },
            "head": {
                "current_value": head_current,
                "effort": head_effort,
                "position": sensors_data_raw__joint_q[:, 26:28],
                "velocity": sensors_data_raw__joint_v[:, 26:28],
            },
            "joint": {
                "current_value": joint_current,
                "effort": joint_effort,
                "position": sensors_data_raw__joint_q[:, 12:26],
                "velocity": sensors_data_raw__joint_v[:, 12:26],
            },
            "end": {
                "position": end_position,
                "orientation": end_orientation,
            },
            "leg": {
                "current_value": sensors_data_raw__joint_current[:, :12],
                "effort": sensors_data_raw__joint_effort[:, :12],
                "position": sensors_data_raw__joint_q[:, 0:12],
                "velocity": sensors_data_raw__joint_v[:, 0:12],
            },
        },
        "imu": {
            "gyro_xyz": sensors_data_raw__imu_data[:, 0:3],
            "acc_xyz": sensors_data_raw__imu_data[:, 3:6],
            "free_acc_xyz": sensors_data_raw__imu_data[:, 6:9],
            "quat_xyzw": sensors_data_raw__imu_data[:, 9:13],
        },
    }

    # 7. 返回前最后一次内存清理
    del kuavo_arm_traj__position
    gc.collect()

    return (
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
        state,
        action,
        claw_state,
        claw_action,
        qiangnao_state,
        qiangnao_action,
    )


import multiprocessing


def load_raw_episode_worker(raw_config, ep_path, start_time, end_time, queue):
    try:
        result = load_raw_episode_data(
            raw_config=raw_config,
            ep_path=ep_path,
            start_time=start_time,
            end_time=end_time,
        )
        queue.put({"ok": True, "data": result})
    except Exception as e:
        import traceback

        queue.put({"ok": False, "error": str(e), "traceback": traceback.format_exc()})


def load_hand_data_worker(config, first_bag_path, first_start, first_end, queue):
    try:
        claw_state, claw_action, qiangnao_state, qiangnao_action = process_rosbag_eef(
            config, first_bag_path, start_time=first_start, end_time=first_end
        )
        queue.put(
            {
                "ok": True,
                "data": (claw_state, claw_action, qiangnao_state, qiangnao_action),
            }
        )
    except Exception as e:
        import traceback

        queue.put({"ok": False, "error": str(e), "traceback": traceback.format_exc()})


def process_rosbag_eef(config, bag_path, start_time=0, end_time=1):
    """
    只读取手部相关数据，不做时间戳对齐和话题筛选。
    只遍历需要的话题，返回 claw_state, claw_action, qiangnao_state, qiangnao_action
    """
    import rosbag
    import numpy as np

    claw_state = []
    claw_action = []
    qiangnao_state = []
    qiangnao_action = []

    # 话题名根据你的实际定义
    topic_claw_state = "/leju_claw_state"
    topic_claw_action = "/leju_claw_command"
    topic_qiangnao_state = "/control_robot_hand_position_state"
    topic_qiangnao_action = "/control_robot_hand_position"

    bag = rosbag.Bag(bag_path, "r")
    bag_start = bag.get_start_time()
    bag_end = bag.get_end_time()
    bag_duration = bag_end - bag_start

    abs_start = bag_start + start_time * bag_duration
    abs_end = bag_start + end_time * bag_duration

    # 只遍历需要的话题
    for topic, msg, t in bag.read_messages(
        topics=[
            topic_claw_state,
            topic_claw_action,
            topic_qiangnao_state,
            topic_qiangnao_action,
        ]
    ):
        if t.to_sec() < abs_start or t.to_sec() > abs_end:
            continue
        if topic == topic_claw_state:
            try:
                claw_state.append(np.array(msg.data.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_claw_action:
            try:
                claw_action.append(np.array(msg.data.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_qiangnao_state:
            try:
                state = list(msg.left_hand_position) + list(msg.right_hand_position)
                qiangnao_state.append(np.array(state, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_qiangnao_action:
            try:
                position = list(msg.left_hand_position) + list(msg.right_hand_position)
                qiangnao_action.append(np.array(position, dtype=np.float64))
            except Exception:
                pass

    bag.close()

    claw_state = np.array(claw_state)
    claw_action = np.array(claw_action)
    qiangnao_state = np.array(qiangnao_state)
    qiangnao_action = np.array(qiangnao_action)

    return claw_state, claw_action, qiangnao_state, qiangnao_action


def port_kuavo_rosbag(
    raw_config: Config,
    repo_id: str = "lerobot/kuavo",
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    mode: Literal["video", "image"] = "video",
    processed_files: list[dict[str, str]] | list[str] = [],
    moment_json_DIR: str | None = None,
    metadata_json_DIR: str | None = None,
    lerobot_dir: str | None = None,
):
    from kuavo_dataset_slave import (
        KuavoRosbagReader,
        DEFAULT_JOINT_NAMES_LIST,
        DEFAULT_LEG_JOINT_NAMES,
        DEFAULT_ARM_JOINT_NAMES,
        DEFAULT_HEAD_JOINT_NAMES,
        # DEFAULT_CAMERA_NAMES,
        DEFAULT_JOINT_NAMES,
        DEFAULT_LEJUCLAW_JOINT_NAMES,
        DEFAULT_DEXHAND_JOINT_NAMES,
        PostProcessorUtils,
    )

    config = raw_config

    RAW_DIR = config.raw_dir
    ID = config.id
    CONTROL_HAND_SIDE = config.which_arm
    SLICE_ROBOT = config.slice_robot
    SLICE_DEX = config.dex_slice
    SLICE_CLAW = config.claw_slice
    IS_BINARY = config.is_binary
    DELTA_ACTION = config.delta_action
    RELATIVE_START = config.relative_start
    ONLY_HALF_UP_BODY = config.only_arm
    USE_LEJU_CLAW = config.use_leju_claw
    USE_QIANGNAO = config.use_qiangnao

    DEFAULT_JOINT_NAMES_LIST_ORIGIN = DEFAULT_JOINT_NAMES_LIST

    repo_id = f"lerobot/kuavo"
    episode_uuid = str(uuid.uuid4())
    root = os.path.join(lerobot_dir, episode_uuid)
    n = None  # 可以通过修改yaml文件添加num_of_bag配置项
    if os.path.exists(lerobot_dir):
        shutil.rmtree(lerobot_dir)

    # 1. 先读取第一个 bag，判断手部类型
    first_bag_info = processed_files[0]
    first_bag_path = (
        first_bag_info["local_path"]
        if isinstance(first_bag_info, dict)
        else first_bag_info
    )
    first_start = (
        first_bag_info.get("start", 0) if isinstance(first_bag_info, dict) else 0
    )
    first_end = first_bag_info.get("end", 1) if isinstance(first_bag_info, dict) else 1

    # 在主进程中调用
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=load_hand_data_worker,
        args=(config, first_bag_path, first_start, first_end, queue),
    )
    p.start()
    result = queue.get()
    p.join()

    if not result.get("ok"):
        print("子进程异常退出！")
        print(result.get("error"))
        print(result.get("traceback"))
        sys.exit(1)

    claw_state, claw_action, qiangnao_state, qiangnao_action = result["data"]
    print(
        claw_state.shape, claw_action.shape, qiangnao_state.shape, qiangnao_action.shape
    )
    USE_LEJU_CLAW = is_valid_hand_data(claw_state) or is_valid_hand_data(claw_action)
    USE_QIANGNAO = is_valid_hand_data(qiangnao_state) or is_valid_hand_data(
        qiangnao_action
    )
    print(f"检测到手部类型: USE_LEJU_CLAW={USE_LEJU_CLAW}, USE_QIANGNAO={USE_QIANGNAO}")

    half_arm = len(DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(DEFAULT_DEXHAND_JOINT_NAMES) // 2
    UP_START_INDEX = 12
    if ONLY_HALF_UP_BODY:
        if USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            )
            arm_slice = [
                (
                    SLICE_ROBOT[0][0] - UP_START_INDEX,
                    SLICE_ROBOT[0][-1] - UP_START_INDEX,
                ),
                (SLICE_CLAW[0][0] + half_arm, SLICE_CLAW[0][-1] + half_arm),
                (
                    SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw,
                    SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw,
                ),
                (SLICE_CLAW[1][0] + half_arm * 2, SLICE_CLAW[1][-1] + half_arm * 2),
            ]
        elif USE_QIANGNAO:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]
            )
            arm_slice = [
                (
                    SLICE_ROBOT[0][0] - UP_START_INDEX,
                    SLICE_ROBOT[0][-1] - UP_START_INDEX,
                ),
                (SLICE_DEX[0][0] + half_arm, SLICE_DEX[0][-1] + half_arm),
                (
                    SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand,
                    SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand,
                ),
                (SLICE_DEX[1][0] + half_arm * 2, SLICE_DEX[1][-1] + half_arm * 2),
            ]
        DEFAULT_JOINT_NAMES_LIST = [
            DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)
        ]
    else:
        if USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            )
        elif USE_QIANGNAO:
            DEFAULT_ARM_JOINT_NAMES = (
                DEFAULT_ARM_JOINT_NAMES[:half_arm]
                + DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand]
                + DEFAULT_ARM_JOINT_NAMES[half_arm:]
                + DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]
            )
        DEFAULT_JOINT_NAMES_LIST = (
            DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES
        )

    @dataclasses.dataclass(frozen=True)
    class DatasetConfig:
        use_videos: bool = True
        tolerance_s: float = 0.0001
        image_writer_processes: int = 0
        image_writer_threads: int = 3
        video_backend: str | None = None

    DEFAULT_DATASET_CONFIG = DatasetConfig()
    dataset_config = DEFAULT_DATASET_CONFIG

    def create_empty_dataset(
        repo_id: str,
        robot_type: str,
        mode: Literal["video", "image"] = "video",
        eef_type: Literal["leju_claw", "dex_hand"] = "dex_hand",
        *,
        has_depth_image: bool = False,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
        root: str,
        extra_features: bool = True,
        raw_config: Config,
    ) -> LeRobotDataset:
        dexhand = [
            "left_qiangnao_1",
            "left_qiangnao_2",
            "left_qiangnao_3",
            "left_qiangnao_4",
            "left_qiangnao_5",
            "left_qiangnao_6",
            "right_qiangnao_1",
            "right_qiangnao_2",
            "right_qiangnao_3",
            "right_qiangnao_4",
            "right_qiangnao_5",
            "right_qiangnao_6",
        ]
        lejuclaw = [
            "left_claw",
            "right_claw",
        ]
        leg = [
            "l_leg_roll",
            "l_leg_yaw",
            "l_leg_pitch",
            "l_knee",
            "l_foot_pitch",
            "l_foot_roll",
            "r_leg_roll",
            "r_leg_yaw",
            "r_leg_pitch",
            "r_knee",
            "r_foot_pitch",
            "r_foot_roll",
        ]
        arm = [
            "zarm_l1_link",
            "zarm_l2_link",
            "zarm_l3_link",
            "zarm_l4_link",
            "zarm_l5_link",
            "zarm_l6_link",
            "zarm_l7_link",
            "zarm_r1_link",
            "zarm_r2_link",
            "zarm_r3_link",
            "zarm_r4_link",
            "zarm_r5_link",
            "zarm_r6_link",
            "zarm_r7_link",
        ]
        head = ["head_yaw", "head_pitch"]
        cameras = raw_config.default_camera_names
        imu_acc = ["acc_x", "acc_y", "acc_z"]
        imu_free_acc = ["free_acc_x", "ree_acc_y", "free_acc_z"]
        imu_gyro_acc = ["gyro_x", "gyro_y", "gyro_z"]
        imu_quat_acc = ["quat_x", "quat_y", "quat_z", "quat_w"]
        end_orientation = [
            "left_x",
            "left_y",
            "left_z",
            "left_w",
            "right_x",
            "right_y",
            "right_z",
            "right_w",
        ]
        end_position = ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]
        # 根据末端执行器类型定义特征
        features = {
            "action.head.position": {"dtype": "float32", "shape": (2,), "names": head},
            "action.joint.position": {"dtype": "float32", "shape": (14,), "names": arm},
            "action.leg.position": {"dtype": "float32", "shape": (12,), "names": leg},
            "state.head.effort": {"dtype": "float32", "shape": (2,), "names": head},
            "state.head.position": {"dtype": "float32", "shape": (2,), "names": head},
            "state.head.velocity": {"dtype": "float32", "shape": (2,), "names": head},
            "state.joint.current_value": {
                "dtype": "float32",
                "shape": (14,),
                "names": arm,
            },
            "state.joint.effort": {"dtype": "float32", "shape": (14,), "names": arm},
            "state.joint.position": {"dtype": "float32", "shape": (14,), "names": arm},
            "state.joint.velocity": {"dtype": "float32", "shape": (14,), "names": arm},
            "state.robot.orientation": {
                "dtype": "float32",
                "shape": (4,),
                "names": imu_quat_acc,
            },
            "state.leg.current_value": {
                "dtype": "float32",
                "shape": (12,),
                "names": leg,
            },
            "state.leg.effort": {"dtype": "float32", "shape": (12,), "names": leg},
            "state.leg.position": {"dtype": "float32", "shape": (12,), "names": leg},
            "state.leg.velocity": {"dtype": "float32", "shape": (12,), "names": leg},
            "state.end.orientation": {
                "dtype": "float32",
                "shape": (8,),
                "names": end_orientation,
            },
            "state.end.position": {
                "dtype": "float32",
                "shape": (6,),
                "names": end_position,
            },
            "state.robot.orientation": {
                "dtype": "float32",
                "shape": (4,),
                "names": imu_quat_acc,
            },
            "imu.acc_xyz": {"dtype": "float32", "shape": (3,), "names": imu_acc},
            "imu.free_acc_xyz": {
                "dtype": "float32",
                "shape": (3,),
                "names": imu_free_acc,
            },
            "imu.gyro_acc_xyz": {
                "dtype": "float32",
                "shape": (3,),
                "names": imu_gyro_acc,
            },
            "imu.quat_acc_xyzw": {
                "dtype": "float32",
                "shape": (4,),
                "names": imu_quat_acc,
            },
        }

        # 根据末端执行器类型添加相应的特征
        if eef_type == "leju_claw":
            features.update(
                {
                    "action.effector.position": {
                        "dtype": "float32",
                        "shape": (2,),
                        "names": lejuclaw,
                    },
                    "state.effector.position": {
                        "dtype": "float32",
                        "shape": (2,),
                        "names": lejuclaw,
                    },
                }
            )
        elif eef_type == "dex_hand":
            features.update(
                {
                    "action.effector.position": {
                        "dtype": "float32",
                        "shape": (12,),
                        "names": dexhand,
                    },
                    "state.effector.position": {
                        "dtype": "float32",
                        "shape": (12,),
                        "names": dexhand,
                    },
                }
            )

        # 相机特征保持不变
        for cam in cameras:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, 480, 848),
                "names": ["channels", "height", "width"],
            }
            if has_depth_image:
                features[f"observation.images.depth.{cam}"] = {
                    "dtype": mode,
                    "shape": (480, 848),
                    "names": ["height", "width"],
                }

        for cam in cameras:
            features[f"observation.camera_params.rotation_matrix_flat.{cam}"] = {
                "dtype": "float32",
                "shape": (9,),
                "names": None,
            }
            features[f"observation.camera_params.translation_vector.{cam}"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": None,
            }
        if extra_features:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (len(DEFAULT_JOINT_NAMES_LIST),),
                "names": DEFAULT_JOINT_NAMES_LIST,
            }
            features["action"] = {
                "dtype": "float32",
                "shape": (len(DEFAULT_JOINT_NAMES_LIST),),
                "names": DEFAULT_JOINT_NAMES_LIST,
            }
            print("DEFAULT_JOINT_NAMES_LIST", DEFAULT_JOINT_NAMES_LIST)

        if Path(LEROBOT_HOME / repo_id).exists():
            shutil.rmtree(LEROBOT_HOME / repo_id)

        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=raw_config.train_hz,
            robot_type=robot_type,
            features=features,
            use_videos=dataset_config.use_videos,
            tolerance_s=dataset_config.tolerance_s,
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads,
            video_backend=dataset_config.video_backend,
            root=root,
        )

    def populate_dataset(
        raw_config: Config,
        dataset: LeRobotDataset,
        bag_files: list,
        task: str,
        episodes: list[int] | None = None,
        moment_json_dir: str | None = None,
        root: str | None = None,
        metadata_json_dir: str | None = None,
    ) -> LeRobotDataset:
        # 内存监控
        import psutil

        process = psutil.Process()

        if episodes is None:
            episodes = range(len(bag_files))
        for ep_idx in tqdm.tqdm(episodes):
            if os.path.exists(moment_json_dir):
                with open(moment_json_dir, "r", encoding="utf-8") as f:
                    moments_data = json.load(f)
                    action_config = moments_data.get("moments", [])
            bag_info = bag_files[ep_idx]
            if isinstance(bag_info, dict):
                ep_path = bag_info["local_path"]
                start_time = bag_info.get("start", 0)
                end_time = bag_info.get("end", 1)
            else:
                ep_path = bag_info
                start_time = 0
                end_time = 1

            moments_start_time, moments_end_time = get_time_range_from_moments(
                moment_json_dir
            )

            if moments_start_time is not None and moments_end_time is not None:
                print(f"[MOMENTS] 原始时间范围: {start_time} - {end_time}")
                print(
                    f"[MOMENTS] 覆盖使用moments.json时间范围: {moments_start_time} - {moments_end_time}"
                )

                start_time = moments_start_time
                end_time = moments_end_time
            else:
                print(
                    f"[MOMENTS] 未找到有效的moments.json时间范围，使用原始时间范围: {start_time} - {end_time}"
                )

            from termcolor import colored

            print(
                colored(
                    f"Processing {ep_path} (time range: {start_time}-{end_time})",
                    "yellow",
                    attrs=["bold"],
                )
            )
            # 获取bag时间信息
            bag_time_info = get_bag_time_info(ep_path)

            if bag_time_info["iso_format"]:
                print(f"Bag开始时间: {bag_time_info['iso_format']}")
                print(f"Bag持续时间: {bag_time_info['duration']:.2f}秒")

            # # 在 populate_dataset 的循环中
            # queue = multiprocessing.Queue()
            # p = multiprocessing.Process(target=load_raw_episode_worker, args=(raw_config, ep_path, start_time, end_time, queue))
            # p.start()
            # result = queue.get()
            # p.join()
            # if not result.get("ok"):
            #     print("子进程异常退出！")
            #     print(result.get("error"))
            #     sys.exit(1)
            # (imgs_per_cam, imgs_per_cam_depth, info_per_cam, all_low_dim_data,
            # main_time_line_timestamps, distortion_model, head_extrinsics,
            # left_extrinsics, right_extrinsics, compressed, state, action,
            # claw_state, claw_action, qiangnao_state, qiangnao_action) = result["data"]
            # del result
            # gc.collect()
            result = load_raw_episode_data(
                raw_config=raw_config,
                ep_path=ep_path,
                start_time=start_time,
                end_time=end_time,
                metadata_json_dir=metadata_json_dir,
            )

            # # 解包返回数据
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
                state,
                action,
                claw_state,
                claw_action,
                qiangnao_state,
                qiangnao_action,
            ) = result
            del result
            gc.collect()

            compressed_group = {}
            uncompressed_group = {}
            for cam, imgs in imgs_per_cam_depth.items():
                is_compressed = compressed.get(cam, None)
                if is_compressed is True:
                    compressed_group[cam] = imgs
                elif is_compressed is False:
                    uncompressed_group[cam] = imgs
                else:
                    print(f"[WARN] {cam} 的压缩状态未知，跳过")
            depth_dir = os.path.join(root, "depth/")

            # 内存监控：深度编码前
            mem_before_depth_encode = process.memory_info().rss / 1024 / 1024
            print(f"[内存] 深度视频编码前: {mem_before_depth_encode:.1f} MB")

            if compressed_group:
                if raw_config.enhance_enabled:
                    print(
                        f"[INFO] 以下相机为压缩深度，将使用增强处理输出16位视频: {list(compressed_group.keys())}"
                    )
                    # 使用新的增强处理函数，传入彩色图像数据
                    save_depth_videos_enhanced_parallel(
                        compressed_group,
                        imgs_per_cam,
                        output_dir=depth_dir,
                        raw_config=raw_config,
                    )
                    del compressed_group
            if uncompressed_group:
                print(
                    f"[INFO] 以下相机为未压缩深度，将用 save_depth_videos_16U_parallel: {list(uncompressed_group.keys())}"
                )
                save_depth_videos_16U_parallel(
                    uncompressed_group, output_dir=depth_dir, raw_config=raw_config
                )
                del uncompressed_group
            gc.collect()

            mem_after_depth_encode = process.memory_info().rss / 1024 / 1024
            print(f"[内存] 深度视频编码后: {mem_after_depth_encode:.1f} MB")

            move_and_rename_depth_videos(os.path.join(root, "depth"), episode_idx=0)

            mem_before_del_depth = process.memory_info().rss / 1024 / 1024
            print(f"[内存] 删除深度图像前: {mem_before_del_depth:.1f} MB")

            del imgs_per_cam_depth
            gc.collect()

            mem_after_del_depth = process.memory_info().rss / 1024 / 1024
            print(
                f"[内存] 删除深度图像后: {mem_after_del_depth:.1f} MB (释放 {mem_before_del_depth - mem_after_del_depth:.1f} MB)"
            )

            cameras = raw_config.default_camera_names
            extrinsics_map = {
                "head_cam_h": head_extrinsics,
                "wrist_cam_l": left_extrinsics,
                "wrist_cam_r": right_extrinsics,
            }
            extrinsics_dict = {
                cam: extrinsics_map[cam] for cam in cameras if cam in extrinsics_map
            }
            # 对手部进行二值化处理
            if IS_BINARY:
                qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
                qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
                claw_state = np.where(claw_state > 50, 1, 0)
                claw_action = np.where(claw_action > 50, 1, 0)
            else:
                # 进行数据归一化处理
                claw_state = claw_state / 100
                claw_action = claw_action / 100
                qiangnao_state = qiangnao_state / 100
                qiangnao_action = qiangnao_action / 100
            ########################
            # delta 处理
            ########################
            # =====================
            # 为了解决零点问题，将每帧与第一帧相减
            if RELATIVE_START:
                # 每个state, action与他们的第一帧相减
                state = state - state[0]
                action = action - action[0]

            # ===只处理delta action
            mem_before_delta = process.memory_info().rss / 1024 / 1024
            print(f"[内存] DELTA_ACTION 处理前: {mem_before_delta:.1f} MB")

            if DELTA_ACTION:
                delta_action = action[1:] - state[:-1]
                trim = lambda x: x[1:] if (x is not None) and (len(x) > 0) else x
                (
                    state,
                    action,
                    velocity,
                    effort,
                    claw_state,
                    claw_action,
                    qiangnao_state,
                    qiangnao_action,
                ) = map(
                    trim,
                    [
                        state,
                        action,
                        velocity,
                        effort,
                        claw_state,
                        claw_action,
                        qiangnao_state,
                        qiangnao_action,
                    ],
                )
                for camera, img_array in imgs_per_cam.items():
                    imgs_per_cam[camera] = img_array[1:]

                action = delta_action

            mem_after_delta = process.memory_info().rss / 1024 / 1024
            print(f"[内存] DELTA_ACTION 处理后: {mem_after_delta:.1f} MB")

            num_frames = state.shape[0]

            mem_before_loop = process.memory_info().rss / 1024 / 1024
            print(f"[内存] 主循环开始前: {mem_before_loop:.1f} MB")
            print(f"[主循环] 开始处理 {num_frames} 帧")

            for i in range(num_frames):
                if ONLY_HALF_UP_BODY:
                    if USE_LEJU_CLAW:
                        # 使用lejuclaw进行上半身关节数据转换
                        if CONTROL_HAND_SIDE == "left" or CONTROL_HAND_SIDE == "both":
                            output_state = state[
                                i, SLICE_ROBOT[0][0] : SLICE_ROBOT[0][-1]
                            ]
                            output_state = np.concatenate(
                                (
                                    output_state,
                                    claw_state[
                                        i, SLICE_CLAW[0][0] : SLICE_CLAW[0][-1]
                                    ].astype(np.float32),
                                ),
                                axis=0,
                            )
                            output_action = action[
                                i, SLICE_ROBOT[0][0] : SLICE_ROBOT[0][-1]
                            ]
                            output_action = np.concatenate(
                                (
                                    output_action,
                                    claw_action[
                                        i, SLICE_CLAW[0][0] : SLICE_CLAW[0][-1]
                                    ].astype(np.float32),
                                ),
                                axis=0,
                            )
                        if CONTROL_HAND_SIDE == "right" or CONTROL_HAND_SIDE == "both":
                            if CONTROL_HAND_SIDE == "both":
                                output_state = np.concatenate(
                                    (
                                        output_state,
                                        state[
                                            i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                        ],
                                    ),
                                    axis=0,
                                )
                                output_state = np.concatenate(
                                    (
                                        output_state,
                                        claw_state[
                                            i, SLICE_CLAW[1][0] : SLICE_CLAW[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )
                                output_action = np.concatenate(
                                    (
                                        output_action,
                                        action[
                                            i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                        ],
                                    ),
                                    axis=0,
                                )
                                output_action = np.concatenate(
                                    (
                                        output_action,
                                        claw_action[
                                            i, SLICE_CLAW[1][0] : SLICE_CLAW[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )
                            else:
                                output_state = state[
                                    i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                ]
                                output_state = np.concatenate(
                                    (
                                        output_state,
                                        claw_state[
                                            i, SLICE_CLAW[1][0] : SLICE_CLAW[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )
                                output_action = action[
                                    i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                ]
                                output_action = np.concatenate(
                                    (
                                        output_action,
                                        claw_action[
                                            i, SLICE_CLAW[1][0] : SLICE_CLAW[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )

                    elif USE_QIANGNAO:
                        # 类型: kuavo_sdk/robotHandPosition
                        # left_hand_position (list of float): 左手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                        # right_hand_position (list of float): 右手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                        # 构造qiangnao类型的output_state的数据结构的长度应该为26
                        if CONTROL_HAND_SIDE == "left" or CONTROL_HAND_SIDE == "both":
                            output_state = state[
                                i, SLICE_ROBOT[0][0] : SLICE_ROBOT[0][-1]
                            ]
                            output_state = np.concatenate(
                                (
                                    output_state,
                                    qiangnao_state[
                                        i, SLICE_DEX[0][0] : SLICE_DEX[0][-1]
                                    ].astype(np.float32),
                                ),
                                axis=0,
                            )

                            output_action = action[
                                i, SLICE_ROBOT[0][0] : SLICE_ROBOT[0][-1]
                            ]
                            output_action = np.concatenate(
                                (
                                    output_action,
                                    qiangnao_action[
                                        i, SLICE_DEX[0][0] : SLICE_DEX[0][-1]
                                    ].astype(np.float32),
                                ),
                                axis=0,
                            )
                        if CONTROL_HAND_SIDE == "right" or CONTROL_HAND_SIDE == "both":
                            if CONTROL_HAND_SIDE == "both":
                                output_state = np.concatenate(
                                    (
                                        output_state,
                                        state[
                                            i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                        ],
                                    ),
                                    axis=0,
                                )
                                output_state = np.concatenate(
                                    (
                                        output_state,
                                        qiangnao_state[
                                            i, SLICE_DEX[1][0] : SLICE_DEX[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )
                                output_action = np.concatenate(
                                    (
                                        output_action,
                                        action[
                                            i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                        ],
                                    ),
                                    axis=0,
                                )
                                output_action = np.concatenate(
                                    (
                                        output_action,
                                        qiangnao_action[
                                            i, SLICE_DEX[1][0] : SLICE_DEX[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )
                            else:
                                output_state = state[
                                    i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                ]
                                output_state = np.concatenate(
                                    (
                                        output_state,
                                        qiangnao_state[
                                            i, SLICE_DEX[1][0] : SLICE_DEX[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )
                                output_action = action[
                                    i, SLICE_ROBOT[1][0] : SLICE_ROBOT[1][-1]
                                ]
                                output_action = np.concatenate(
                                    (
                                        output_action,
                                        qiangnao_action[
                                            i, SLICE_DEX[1][0] : SLICE_DEX[1][-1]
                                        ].astype(np.float32),
                                    ),
                                    axis=0,
                                )
                        # output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)

                else:
                    if USE_LEJU_CLAW:
                        # 使用lejuclaw进行全身关节数据转换
                        # 原始的数据是28个关节的数据对应原始的state和action数据的长度为28
                        # 数据顺序:
                        # 前 12 个数据为下肢电机数据:
                        #     0~5 为左下肢数据 (l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll)
                        #     6~11 为右下肢数据 (r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll)
                        # 接着 14 个数据为手臂电机数据:
                        #     12~18 左臂电机数据 ("l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll")
                        #     19~25 为右臂电机数据 ("r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll")
                        # 最后 2 个为头部电机数据: head_yaw 和 head_pitch

                        # TODO：构造目标切片
                        output_state = state[i, 0:19]
                        output_state = np.insert(
                            output_state, 19, claw_state[i, 0].astype(np.float32)
                        )
                        output_state = np.concatenate(
                            (output_state, state[i, 19:26]), axis=0
                        )
                        output_state = np.insert(
                            output_state, 19, claw_state[i, 1].astype(np.float32)
                        )
                        output_state = np.concatenate(
                            (output_state, state[i, 26:28]), axis=0
                        )

                        output_action = action[i, 0:19]
                        output_action = np.insert(
                            output_action, 19, claw_action[i, 0].astype(np.float32)
                        )
                        output_action = np.concatenate(
                            (output_action, action[i, 19:26]), axis=0
                        )
                        output_action = np.insert(
                            output_action, 19, claw_action[i, 1].astype(np.float32)
                        )
                        output_action = np.concatenate(
                            (output_action, action[i, 26:28]), axis=0
                        )

                    elif USE_QIANGNAO:
                        output_state = state[i, 0:19]
                        output_state = np.concatenate(
                            (output_state, qiangnao_state[i, 0:6].astype(np.float32)),
                            axis=0,
                        )
                        output_state = np.concatenate(
                            (output_state, state[i, 19:26]), axis=0
                        )
                        output_state = np.concatenate(
                            (output_state, qiangnao_state[i, 6:12].astype(np.float32)),
                            axis=0,
                        )
                        output_state = np.concatenate(
                            (output_state, state[i, 26:28]), axis=0
                        )

                        output_action = action[i, 0:19]
                        output_action = np.concatenate(
                            (output_action, qiangnao_action[i, 0:6].astype(np.float32)),
                            axis=0,
                        )
                        output_action = np.concatenate(
                            (output_action, action[i, 19:26]), axis=0
                        )
                        output_action = np.concatenate(
                            (
                                output_action,
                                qiangnao_action[i, 6:12].astype(np.float32),
                            ),
                            axis=0,
                        )
                        output_action = np.concatenate(
                            (output_action, action[i, 26:28]), axis=0
                        )

                frame = {
                    "observation.state": torch.from_numpy(output_state).type(
                        torch.float32
                    ),
                    "action": torch.from_numpy(output_action).type(torch.float32),
                    "action.head.position": get_nested_value(
                        all_low_dim_data, "action.head.position", i, [0.0] * 2
                    ),
                    "action.joint.position": get_nested_value(
                        all_low_dim_data, "action.joint.position", i, [0.0] * 14
                    ),
                    "action.leg.position": get_nested_value(
                        all_low_dim_data, "action.leg.position", i, [0.0] * 12
                    ),
                    "state.head.effort": get_nested_value(
                        all_low_dim_data, "state.head.effort", i, [0.0] * 2
                    ),
                    "state.head.position": get_nested_value(
                        all_low_dim_data, "state.head.position", i, [0.0] * 2
                    ),
                    "state.head.velocity": get_nested_value(
                        all_low_dim_data, "state.head.velocity", i, [0.0] * 2
                    ),
                    "state.joint.current_value": get_nested_value(
                        all_low_dim_data, "state.joint.current_value", i, [0.0] * 14
                    ),
                    "state.joint.effort": get_nested_value(
                        all_low_dim_data, "state.joint.effort", i, [0.0] * 14
                    ),
                    "state.joint.position": get_nested_value(
                        all_low_dim_data, "state.joint.position", i, [0.0] * 14
                    ),
                    "state.joint.velocity": get_nested_value(
                        all_low_dim_data, "state.joint.velocity", i, [0.0] * 14
                    ),
                    # 展平末端左右手姿态和位置
                    "state.end.orientation": (
                        get_nested_value(
                            all_low_dim_data, "state.end.orientation", i, [0.0] * 8
                        )
                    ).flatten(),
                    "state.end.position": (
                        get_nested_value(
                            all_low_dim_data, "state.end.position", i, [0.0] * 6
                        )
                    ).flatten(),
                    "state.leg.current_value": get_nested_value(
                        all_low_dim_data, "state.leg.current_value", i, [0.0] * 12
                    ),
                    "state.leg.effort": get_nested_value(
                        all_low_dim_data, "state.leg.effort", i, [0.0] * 12
                    ),
                    "state.leg.position": get_nested_value(
                        all_low_dim_data, "state.leg.position", i, [0.0] * 12
                    ),
                    "state.leg.velocity": get_nested_value(
                        all_low_dim_data, "state.leg.velocity", i, [0.0] * 12
                    ),
                    "state.robot.orientation": get_nested_value(
                        all_low_dim_data, "state.robot.orientation", i, [0.0] * 4
                    ),
                    "imu.acc_xyz": get_nested_value(
                        all_low_dim_data, "imu.acc_xyz", i, [0.0] * 3
                    ),
                    "imu.gyro_acc_xyz": get_nested_value(
                        all_low_dim_data, "imu.gyro_acc_xyz", i, [0.0] * 3
                    ),
                    "imu.free_acc_xyz": get_nested_value(
                        all_low_dim_data, "imu.free_acc_xyz", i, [0.0] * 3
                    ),
                    "imu.quat_acc_xyzw": get_nested_value(
                        all_low_dim_data, "imu.quat_acc_xyzw", i, [0.0] * 4
                    ),
                }
                # 根据末端执行器类型添加相应数据
                if USE_LEJU_CLAW:
                    frame.update(
                        {
                            "action.effector.position": get_nested_value(
                                all_low_dim_data,
                                "action.effector.position(gripper)",
                                i,
                                [0.0] * 2,
                            ),
                            "state.effector.position": get_nested_value(
                                all_low_dim_data,
                                "state.effector.position(gripper)",
                                i,
                                [0.0] * 2,
                            ),
                        }
                    )
                if USE_QIANGNAO:
                    frame.update(
                        {
                            "action.effector.position": get_nested_value(
                                all_low_dim_data,
                                "action.effector.position(dexhand)",
                                i,
                                [0.0] * 12,
                            ),
                            "state.effector.position": get_nested_value(
                                all_low_dim_data,
                                "state.effector.position(dexhand)",
                                i,
                                [0.0] * 12,
                            ),
                        }
                    )

                for cam_key, extrinsics in extrinsics_dict.items():
                    if extrinsics and len(extrinsics) > i:
                        rot = np.array(
                            extrinsics[i]["rotation_matrix"], dtype=np.float32
                        ).reshape(
                            -1
                        )  # (3,3) -> (9,)
                        trans = np.array(
                            extrinsics[i]["translation_vector"], dtype=np.float32
                        ).reshape(
                            -1
                        )  # (3,)
                        frame[
                            f"observation.camera_params.rotation_matrix_flat.{cam_key}"
                        ] = rot
                        frame[
                            f"observation.camera_params.translation_vector.{cam_key}"
                        ] = trans

                for camera, img_array in imgs_per_cam.items():
                    img_bytes = img_array[i]
                    # 解码为RGB图像
                    img_np = cv2.imdecode(
                        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                    )
                    if img_np is None:
                        raise ValueError(
                            f"Failed to decode color image for camera {camera} at frame {i}"
                        )
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    img_np = cv2.resize(
                        img_np, (raw_config.resize.width, raw_config.resize.height)
                    )
                    frame[f"observation.images.{camera}"] = img_np
                    del img_bytes, img_np

                dataset.add_frame(frame, task=task)
                del frame
                if "output_state" in locals():
                    del output_state
                if "output_action" in locals():
                    del output_action

                # 定期等待图像写入完成并GC
                if i % 200 == 0:
                    # 等待 AsyncImageWriter 队列清空
                    if hasattr(dataset, '_wait_image_writer') and dataset._wait_image_writer:
                        if dataset.image_writer.queue.qsize() > 500:
                            dataset._wait_image_writer()
                            print(f"[主循环] 第 {i} 帧：等待图像写入完成")
                            gc.collect()

                # 如果是循环的最后一帧，可以回收更多变量
                if i == num_frames - 1:
                    del imgs_per_cam, all_low_dim_data
                    del (
                        state,
                        action,
                        claw_state,
                        claw_action,
                        qiangnao_state,
                        qiangnao_action,
                    )
                    del head_extrinsics, left_extrinsics, right_extrinsics
                    gc.collect()
                    # 清理解码变量
            dataset.save_episode()
            parameters_dir = os.path.join(root, "parameters")
            if not os.path.exists(parameters_dir):
                os.makedirs(parameters_dir)
            cameras = raw_config.default_camera_names
            save_camera_info_to_json_new(
                info_per_cam, distortion_model, output_dir=parameters_dir
            )
            save_camera_extrinsic_params(cameras=cameras, output_dir=parameters_dir)
            merge_metadata_and_moment(
                metadata_json_dir,
                moment_json_dir,
                os.path.join(root, "metadata.json"),
                episode_uuid,
                raw_config,
                bag_time_info=bag_time_info,
                main_time_line_timestamps=main_time_line_timestamps,
            )
        return dataset

    bag_files = processed_files
    eef_type = "leju_claw" if USE_LEJU_CLAW else "dex_hand"
    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type="kuavo4pro",
        mode=mode,
        eef_type=eef_type,
        dataset_config=dataset_config,
        root=root,
        raw_config=raw_config,
    )
    dataset = populate_dataset(
        dataset=dataset,
        bag_files=bag_files,
        raw_config=raw_config,
        task=task,
        episodes=episodes,
        moment_json_dir=moment_json_DIR,
        metadata_json_dir=metadata_json_DIR,
        root=root,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Kuavo ROSbag to Lerobot Converter")
    parser.add_argument(
        "--bag_dir",
        default="/home/leju_kuavo/zxr/rosbag2lerobot-cb/test/",
        type=str,
        required=False,
        help="Path to ROS bag",
    )
    # parser.add_argument("--bag_dir", default = "./testbag/task24_519_20250519_193043_0.bag", type=str, required=False, help="Path to ROS bag")
    parser.add_argument(
        "--moment_json_dir", type=str, required=False, help="Path to moment.json"
    )
    parser.add_argument(
        "--metadata_json_dir", type=str, required=False, help="Path to metadata.json"
    )
    parser.add_argument(
        "--output_dir",
        default="testoutput/",
        type=str,
        required=False,
        help="Path to output",
    )
    parser.add_argument(
        "--train_frequency",
        type=int,
        help="Training frequency (Hz), overrides config file setting",
    )

    parser.add_argument(
        "--only_arm",
        type=str,
        choices=["true", "false"],
        help="Use only arm data (true/false), overrides config file setting",
    )

    parser.add_argument(
        "--which_arm",
        type=str,
        choices=["left", "right", "both"],
        help="Which arm to use (left/right/both), overrides config file setting",
    )

    parser.add_argument(
        "--dex_dof_needed",
        type=int,
        help="Degrees of freedom needed for dexterous hand, overrides config file setting",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./kuavo/request.json",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # 加载配置文件
    config = load_config_from_json(args.config)
    # 用命令行参数覆盖配置文件中的设置
    if args.train_frequency is not None:
        config.train_hz = args.train_frequency
        print(f"✅ 覆盖配置: train_hz = {args.train_frequency}")

    if args.only_arm is not None:
        config.only_arm = args.only_arm.lower() == "true"
        print(f"✅ 覆盖配置: only_arm = {config.only_arm}")

    if args.which_arm is not None:
        config.which_arm = args.which_arm
        print(f"✅ 覆盖配置: which_arm = {args.which_arm}")

    if args.dex_dof_needed is not None:
        config.dex_dof_needed = args.dex_dof_needed
        print(f"✅ 覆盖配置: dex_dof_needed = {args.dex_dof_needed}")
    # 从配置获取参数

    if args.bag_dir is not None:
        bag_DIR = args.bag_dir
    print(f"Bag directory: {bag_DIR}")
    if args.moment_json_dir is not None:
        moment_json_DIR = args.moment_json_dir
    else:
        moment_json_DIR = os.path.join(bag_DIR, "moments.json")

    if args.metadata_json_dir is not None:
        metadata_json_DIR = args.metadata_json_dir
    else:
        metadata_json_DIR = os.path.join(bag_DIR, "metadata.json")
    if args.output_dir is not None:
        output_DIR = args.output_dir

    ID = config.id

    bag_files = list_bag_files_auto(bag_DIR)
    port_kuavo_rosbag(
        raw_config=config,
        processed_files=bag_files,
        moment_json_DIR=moment_json_DIR,
        metadata_json_DIR=metadata_json_DIR,
        lerobot_dir=output_DIR,
    )
