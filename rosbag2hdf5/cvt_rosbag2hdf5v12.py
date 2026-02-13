"""
Script to convert Kuavo rosbag data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import argparse
import dataclasses
import datetime
import json
import os
import shutil
import time
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rosbag
import tqdm
from slave_utils import (
    detect_and_trim_bag_data,
    flip_camera_arrays_if_needed,
    load_camera_info_per_camera,
    load_raw_depth_images_per_camera,
    load_raw_images_per_camera,
    recursive_filter_and_position,
    save_camera_extrinsic_params,
    save_camera_info_to_json_new,
    save_color_videos_parallel,
    save_depth_videos_16U_parallel,
    save_depth_videos_enhanced_parallel,
    save_depth_videos_parallel,
    swap_left_right_data_if_needed,
    validate_episode_data_consistency,
)
from config_dataset_slave import Config, load_config_from_json
from kuavo_dataset_slave import (
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_DEXHAND_JOINT_NAMES,
    DEFAULT_HEAD_JOINT_NAMES,
    # DEFAULT_CAMERA_NAMES,
    DEFAULT_JOINT_NAMES,
    DEFAULT_JOINT_NAMES_LIST,
    DEFAULT_LEG_JOINT_NAMES,
    DEFAULT_LEJUCLAW_JOINT_NAMES,
    KuavoRosbagReader,
    PostProcessorUtils,
)


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


def format_size_gb(size_bytes):
    size_gb = size_bytes / (1024**3)
    int_part = int(size_gb)
    frac_part = int(round((size_gb - int_part) * 100))
    return f"{int_part}p{frac_part:02d}"


def create_file_structure(
    scene, sub_scene, continuous_action, bag_path, save_dir, mode="simplified"
):
    if mode not in ["simplified", "complete"]:
        raise ValueError("mode must be either 'simplified' or 'complete'")
    elif mode == "complete":
        # 1. 统计bag大小和数量
        bag_files = [bag_path] if isinstance(bag_path, str) else bag_path
        total_size = sum(os.path.getsize(f) for f in bag_files)
        count = len(bag_files)
        size_str = format_size_gb(total_size)
        hour_str = "0p00h"

        # 2. 目录名
        main_dir = f"{scene}"
        sub_dir = f"{sub_scene}"
        action_dir = f"{continuous_action}"

        # 3. 生成UUID
        episode_uuid = str(uuid.uuid4())

        # 4. 拼接完整路径
        base_path = os.path.join(save_dir, main_dir, sub_dir, action_dir, episode_uuid)
        camera_dir = os.path.join(base_path, "camera/")
        depth_dir = os.path.join(camera_dir, "depth/")
        video_dir = os.path.join(camera_dir, "video/")
        parameters_dir = os.path.join(base_path, "parameters/")
        proprio_stats_dir = os.path.join(base_path, "proprio_stats/")
        audio_dir = os.path.join(base_path, "audio/")
        task_info_dir = os.path.join(save_dir, "task_info/")

        # 5. 创建所有目录
        for d in [
            task_info_dir,
            base_path,
            camera_dir,
            depth_dir,
            video_dir,
            parameters_dir,
            proprio_stats_dir,
            audio_dir,
        ]:
            os.makedirs(d, exist_ok=True)
    else:
        # 简化模式
        episode_uuid = str(uuid.uuid4())
        base_path = os.path.join(save_dir, episode_uuid)
        camera_dir = os.path.join(base_path, "camera/")
        depth_dir = os.path.join(camera_dir, "depth/")
        video_dir = os.path.join(camera_dir, "video/")
        parameters_dir = os.path.join(base_path, "parameters/")
        proprio_stats_dir = os.path.join(base_path, "proprio_stats/")

        # 创建目录
        for d in [base_path, depth_dir, video_dir, parameters_dir, proprio_stats_dir]:
            os.makedirs(d, exist_ok=True)

    print(f"已创建目录结构：{base_path}")
    return (
        episode_uuid,
        base_path + "/",
        depth_dir,
        video_dir,
        parameters_dir,
        proprio_stats_dir,
    )


def merge_metadata_and_moment(
    metadata_path,
    moment_path,
    output_path,
    uuid,
    raw_config,
    bag_time_info=None,
    main_time_line_timestamps=None,
    output_dir=None,
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

    # 验证 metadata.json 关键字段是否存在且非空
    required_metadata_fields = {
        "scene_code": "场景编码",
        "sub_scene_code": "子场景编码",
        "sub_scene_zh_dec": "子场景中文描述",
        "sub_scene_en_dec": "子场景英文描述",
        "device_sn": "设备序列号",
    }

    for field, desc in required_metadata_fields.items():
        value = raw_metadata.get(field)
        if not value or (isinstance(value, str) and value.strip() == ""):
            error_msg = f"❌ metadata.json 缺失关键字段 '{field}' ({desc})，数据不完整，终止处理"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

    # 验证 task 相关字段至少有一个有效
    task_group_name = raw_metadata.get("task_group_name", "").strip()
    task_name = raw_metadata.get("task_name", "").strip()
    task_group_code = raw_metadata.get("task_group_code", "").strip()
    task_code = raw_metadata.get("task_code", "").strip()

    if not (task_group_name or task_name):
        error_msg = f"❌ metadata.json 缺失任务名称字段 'task_group_name' 和 'task_name' 都为空，数据不完整，终止处理"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    if not (task_group_code or task_code):
        error_msg = f"❌ metadata.json 缺失任务编码字段 'task_group_code' 和 'task_code' 都为空，数据不完整，终止处理"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    # 验证 moment.json 数据有效性
    moments = moment.get("moments", [])
    if not moments:
        error_msg = (
            f"❌ moment.json 中未找到有效的 moments 数据，标注信息缺失，终止处理"
        )
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    # 验证每个 moment 的关键字段
    for i, m in enumerate(moments):
        custom_fields = m.get("customFieldValues", {})
        trigger_time = m.get("triggerTime", "").strip()

        # 验证时间戳
        if not trigger_time:
            error_msg = f"❌ moment.json 第{i+1}个动作缺失 'triggerTime' 时间戳，标注信息不完整，终止处理"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # 验证动作描述字段
        required_moment_fields = {
            "skill_atomic_en": "技能原子英文名",
            "skill_detail": "技能详细描述",
            "en_skill_detail": "技能英文详细描述",
        }

        for field, desc in required_moment_fields.items():
            value = custom_fields.get(field, "").strip()
            if not value:
                error_msg = f"❌ moment.json 第{i+1}个动作缺失 '{field}' ({desc})，标注信息不完整，终止处理"
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)

    # 转换新格式 metadata 为旧格式
    converted_metadata = {}

    # scene_name 对应 scene_code
    converted_metadata["scene_name"] = raw_metadata.get("scene_code")

    # sub_scene_name 对应 sub_scene_code
    converted_metadata["sub_scene_name"] = raw_metadata.get("sub_scene_code")

    # init_scene_text 对应 sub_scene_zh_dec
    converted_metadata["init_scene_text"] = raw_metadata.get("sub_scene_zh_dec")

    # english_init_scene_text 对应 sub_scene_en_dec
    converted_metadata["english_init_scene_text"] = raw_metadata.get("sub_scene_en_dec")

    # task_name 优先 task_group_name 其次 task_name
    if task_group_name:
        converted_metadata["task_name"] = task_group_name
    else:
        converted_metadata["task_name"] = task_name

    # english_task_name 优先 task_group_code 其次 task_code
    if task_group_code:
        english_task_name = task_group_code
    else:
        english_task_name = task_code

    if isinstance(english_task_name, str) and "_" in english_task_name:
        english_task_name = english_task_name.replace("_", " ")
    converted_metadata["english_task_name"] = english_task_name

    # 默认值字段
    converted_metadata["data_type"] = "常规"
    converted_metadata["episode_status"] = "approved"
    converted_metadata["data_gen_mode"] = "real_machine"

    # sn_code 对应 device_sn
    converted_metadata["sn_code"] = raw_metadata.get("device_sn")

    # sn_name 默认值
    # TODO: 确认格式：机器厂家-机器人本体型号(和urdf名字保持对应)-末端执行器类型
    converted_metadata["sn_name"] = f"乐聚-biped_s49-{raw_metadata.get('eef_type', 'unknown')}"

    # 新增：验证转换后的关键字段不能为空
    final_required_fields = {
        "scene_name": "场景名称",
        "sub_scene_name": "子场景名称",
        "init_scene_text": "场景中文描述",
        "english_init_scene_text": "场景英文描述",
        "task_name": "任务名称",
        "english_task_name": "任务英文名称",
        "sn_code": "设备序列号",
    }

    for field, desc in final_required_fields.items():
        value = converted_metadata.get(field)
        if not value or (isinstance(value, str) and value.strip() == ""):
            error_msg = f"❌ 转换后字段 '{field}' ({desc}) 为空，数据不完整，终止处理"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

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

        # 新增：时间戳有效性验证（下界与上界）
        import datetime

        year_2025_timestamp = datetime.datetime(2025, 1, 1).timestamp()
        year_2040_timestamp = datetime.datetime(2040, 1, 1).timestamp()

        if rosbag_actual_start_time < year_2025_timestamp:
            start_datetime = datetime.datetime.fromtimestamp(rosbag_actual_start_time)
            error_msg = f"❌ 数据开始时间戳异常: {rosbag_actual_start_time} ({start_datetime.isoformat()})，早于2025年，数据可能损坏"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if rosbag_actual_end_time < year_2025_timestamp:
            end_datetime = datetime.datetime.fromtimestamp(rosbag_actual_end_time)
            error_msg = f"❌ 数据结束时间戳异常: {rosbag_actual_end_time} ({end_datetime.isoformat()})，早于2025年，数据可能损坏"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # 新增：大于等于 2040 年也视为异常
        if rosbag_actual_start_time >= year_2040_timestamp:
            start_datetime = datetime.datetime.fromtimestamp(rosbag_actual_start_time)
            error_msg = f"❌ 数据开始时间戳异常: {rosbag_actual_start_time} ({start_datetime.isoformat()})，晚于2040年，数据可能损坏"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if rosbag_actual_end_time >= year_2040_timestamp:
            end_datetime = datetime.datetime.fromtimestamp(rosbag_actual_end_time)
            error_msg = f"❌ 数据结束时间戳异常: {rosbag_actual_end_time} ({end_datetime.isoformat()})，晚于2040年，数据可能损坏"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

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

    for m in moments:
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

                # 下界校验
                if action_original_start_time < year_2025_timestamp:
                    error_msg = f"❌ 动作时间戳异常: {trigger_time} ({trigger_datetime.isoformat()})，早于2025年，数据可能损坏"
                    print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)

                # 新增：上界校验
                if action_original_start_time >= year_2040_timestamp:
                    error_msg = f"❌ 动作时间戳异常: {trigger_time} ({trigger_datetime.isoformat()})，晚于2040年，数据可能损坏"
                    print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)

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

                # 新增：验证计算的帧数有效性
                if start_frame is None or end_frame is None:
                    error_msg = f"❌ 动作帧数计算失败: 动作'{custom_fields.get('skill_detail', '')}' 时间戳 {trigger_time} 计算出的开始帧={start_frame}, 结束帧={end_frame}"
                    print(f"[ERROR] {error_msg}")
                    print(f"  调试信息:")
                    print(
                        f"    实际数据范围: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
                    )
                    print(
                        f"    动作时间范围: {action_original_start_time:.6f}s - {action_original_start_time + action_duration:.6f}s"
                    )
                    raise ValueError(error_msg)

                print(f"动作: {custom_fields.get('skill_detail', '')}")
                print(f"  动作时间: {trigger_datetime.isoformat()}")
                print(f"  原始开始时间: {action_original_start_time:.6f}s")
                print(
                    f"  原始结束时间: {action_original_start_time + action_duration:.6f}s"
                )
                print(f"  持续时间: {action_duration:.3f}s")
                print(f"  计算得到帧数: {start_frame} - {end_frame}")

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
                print(f"[ERROR] 计算帧数时出错: {e}")
                import traceback

                traceback.print_exc()
                raise  # 重新抛出异常，终止程序

        # 获取动作字段值并验证不为空
        skill_value = custom_fields.get("skill_atomic_en", "").strip()
        action_text_value = custom_fields.get("skill_detail", "").strip()
        english_action_text_value = custom_fields.get("en_skill_detail", "").strip()

        # 新增：验证动作配置中的关键字段不能为空
        if not skill_value:
            error_msg = f"❌ 动作配置中 skill 字段为空，动作标注不完整，终止处理"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if not action_text_value:
            error_msg = f"❌ 动作配置中 action_text 字段为空，动作标注不完整，终止处理"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if not english_action_text_value:
            error_msg = (
                f"❌ 动作配置中 english_action_text 字段为空，动作标注不完整，终止处理"
            )
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # 构造新的 action 对象，使用验证过的非空值
        action = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "timestamp_utc": formatted_trigger_time,
            "is_mistake": False,
            "skill": skill_value,
            "action_text": action_text_value,
            "english_action_text": english_action_text_value,
        }
        action_config.append(action)

    # 按照 timestamp_utc 排序
    action_config = sorted(
        action_config, key=lambda x: x["timestamp_utc"] if x["timestamp_utc"] else ""
    )

    # 1. 时长统计
    file_duration = None
    if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 1:
        duration_sec = main_time_line_timestamps[-1] - main_time_line_timestamps[0]
        file_duration = round(float(duration_sec), 2)

    # 2. 大小统计
    def get_dir_size_gb(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        return round(total_size / (1024**3), 6)

    file_size = get_dir_size_gb(output_dir)

    # ...构造 new_json ...
    new_json = OrderedDict()
    new_json["episode_id"] = uuid

    # 使用转换后的 metadata
    for k, v in metadata.items():
        new_json[k] = v

    # 新增统计字段
    new_json["file_duration"] = file_duration
    new_json["file_size"] = file_size

    if "label_info" not in new_json:
        new_json["label_info"] = {}
    new_json["label_info"]["action_config"] = action_config
    if "key_frame" not in new_json["label_info"]:
        new_json["label_info"]["key_frame"] = []

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)
    print(f"已保存到 {output_path}")


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def load_raw_episode_data(
    raw_config: Config,
    ep_path: Path,
    start_time: float = 0,
    end_time: float = 1,
    action_config=None,
    min_duration: float = 5.0,
    metadata_json_dir: str = None,  # 新增参数
):

    bag_reader = KuavoRosbagReader(raw_config)
    bag_data = bag_reader.process_rosbag(
        ep_path,
        start_time=start_time,
        end_time=end_time,
        action_config=action_config,
        min_duration=min_duration,
    )
    ori_bag_data = bag_reader.process_rosbag(
        ep_path,
        start_time=start_time,
        end_time=end_time,
        action_config=action_config,
        min_duration=min_duration,
        is_align=False
    )
    sn_code = None
    is_wheel_arm = False
    if metadata_json_dir and os.path.exists(metadata_json_dir):
        try:
            with open(metadata_json_dir, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
            sn_code = raw_metadata.get("device_sn", "")
            is_wheel_arm = sn_code.startswith("LB")
            print(f"[INFO] 检测到设备序列号: {sn_code}, 是否为轮臂: {is_wheel_arm}")
        except Exception as e:
            print(f"[WARN] 读取metadata.json失败: {e}, 默认包含腿部数据")
    # 检测部分数据中左右手数据颠倒的问题
    if sn_code is not None:
        main_time_line_timestamps = None
        if "head_cam_h" in bag_data and len(bag_data["head_cam_h"]) > 0:
            main_time_line_timestamps = np.array(
                [msg["timestamp"] for msg in bag_data["head_cam_h"]]
            )
        else:
            main_time_line_timestamps = None
        swap_left_right_data_if_needed(bag_data, sn_code, main_time_line_timestamps)

    # 常用数据
    sensors_data_raw__joint_q = state = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_q"]],
        dtype=np.float32,
    )  # 原 observation.state
    joint_cmd__joint_q = action = np.array(
        [msg["data"] for msg in bag_data["action.joint_cmd.joint_q"]], dtype=np.float32
    )  # 原action
    joint_cmd__joint_v = action_joint_v = np.array(
        [msg["data"] for msg in bag_data["action.joint_cmd.joint_v"]], dtype=np.float32
    )
    kuavo_arm_traj__position = action_kuavo_arm_traj = np.array(
        [msg["data"] for msg in bag_data["action.kuavo_arm_traj"]], dtype=np.float32
    )
    leju_claw_state__position = claw_state = np.array(
        [msg["data"] for msg in bag_data["observation.claw"]], dtype=np.float32
    )
    leju_claw_command__position = claw_action = np.array(
        [msg["data"] for msg in bag_data["action.claw"]], dtype=np.float32
    )
    # print("==========================0000000000000000000000===========================",'\n',leju_claw_command__position)
    # TODO: 夹爪添加velocity和effort数据

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
    action[:, 12:26] = action_kuavo_arm_traj

    # 新增数据
    sensors_data_raw__joint_v = state_joint_v = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_v"]],
        dtype=np.float32,
    )
    # sensors_data_raw__joint_vd = state_joint_vd = np.array([msg['data'] for msg in bag_data['observation.sensorsData.joint_vd']], dtype=np.float32)

    state_joint_current = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.joint_current"]],
        dtype=np.float32,
    )
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
    head_extrinsics = bag_data.get("head_camera_extrinsics", [])
    left_extrinsics = bag_data.get("left_hand_camera_extrinsics", [])
    right_extrinsics = bag_data.get("right_hand_camera_extrinsics", [])
    end_position = np.array(
        [msg["data"] for msg in bag_data["end.position"]], dtype=np.float32
    )
    end_orientation = np.array(
        [msg["data"] for msg in bag_data["end.orientation"]], dtype=np.float32
    )
    head_effort = sensors_data_raw__joint_effort[:, 26:28]  # 头部关节的effort
    head_current = sensors_data_raw__joint_current[:, 26:28]  # 头部关节的current
    joint_effort = sensors_data_raw__joint_effort[:, 12:26]  # 其他关节的effort
    joint_current = sensors_data_raw__joint_current[:, 12:26]  # 其他关节的current
    sensors_data_raw__imu_data = state_joint_imu = np.array(
        [msg["data"] for msg in bag_data["observation.sensorsData.imu"]],
        dtype=np.float32,
    )
    velocity = None
    effort = None

    imgs_per_cam = load_raw_images_per_camera(bag_data, raw_config.default_camera_names)

    imgs_per_cam_depth, compressed = load_raw_depth_images_per_camera(
        bag_data, raw_config.default_camera_names
    )
    info_per_cam, distortion_model = load_camera_info_per_camera(
        bag_data, raw_config.default_camera_names
    )

    # 在 load_raw_episode_data_hdf5 函数中，修改时间戳处理部分

    main_time_line_timestamps = np.array(
        [msg["timestamp"] for msg in bag_data["head_cam_h"]]
    )

    # 新增：自动翻转相机数据（根据设备号和主时间戳）
    imgs_per_cam, imgs_per_cam_depth = flip_camera_arrays_if_needed(
        imgs_per_cam, imgs_per_cam_depth, sn_code, main_time_line_timestamps[0]
    )
    main_time_line_timestamps_ns = (main_time_line_timestamps * 1e9).astype(np.int64)
    main_time_line_timestamps_ns_head_camera = main_time_line_timestamps_ns
    main_time_line_timestamps_head_camera_depth = np.array(
        [msg["timestamp"] for msg in bag_data["head_cam_h_depth"]]
    )
    main_time_line_timestamps_ns_head_camera_depth = (
        main_time_line_timestamps_head_camera_depth * 1e9
    ).astype(np.int64)

    # 检查左右相机数据是否存在
    main_time_line_timestamps_ns_left_camera = None
    main_time_line_timestamps_ns_right_camera = None

    if "wrist_cam_l" in bag_data and len(bag_data["wrist_cam_l"]) > 0:
        main_time_line_timestamps_left_camera = np.array(
            [msg["timestamp"] for msg in bag_data["wrist_cam_l"]]
        )
        main_time_line_timestamps_ns_left_camera = (
            main_time_line_timestamps_left_camera * 1e9
        ).astype(np.int64)
        main_time_line_timestamps_left_camera_depth = np.array(
            [msg["timestamp"] for msg in bag_data["wrist_cam_l_depth"]]
        )
        main_time_line_timestamps_ns_left_camera_depth = (
            main_time_line_timestamps_left_camera_depth * 1e9
        ).astype(np.int64)

    if "wrist_cam_r" in bag_data and len(bag_data["wrist_cam_r"]) > 0:
        main_time_line_timestamps_right_camera = np.array(
            [msg["timestamp"] for msg in bag_data["wrist_cam_r"]]
        )
        main_time_line_timestamps_ns_right_camera = (
            main_time_line_timestamps_right_camera * 1e9
        ).astype(np.int64)
        main_time_line_timestamps_right_camera_depth = np.array(
            [msg["timestamp"] for msg in bag_data["wrist_cam_r_depth"]]
        )
        main_time_line_timestamps_ns_right_camera_depth = (
            main_time_line_timestamps_right_camera_depth * 1e9
        ).astype(np.int64)

    # 其他时间戳处理
    main_time_line_timestamps_head = np.array(
        [msg["timestamp"] for msg in bag_data["observation.sensorsData.joint_q"]]
    )
    main_time_line_timestamps_ns_head = (main_time_line_timestamps_head * 1e9).astype(
        np.int64
    )
    main_time_line_timestamps_ns_extrinsic = main_time_line_timestamps_ns_head
    main_time_line_timestamps_joint = np.array(
        [msg["timestamp"] for msg in bag_data["observation.sensorsData.joint_q"]]
    )
    main_time_line_timestamps_ns_joint = (main_time_line_timestamps_joint * 1e9).astype(
        np.int64
    )

    # 检查效果器数据存在性（参考 recursive_filter_and_position 函数的逻辑）
    has_dexhand = "action.qiangnao" in bag_data and len(bag_data["action.qiangnao"]) > 0
    has_lejuclaw = "action.claw" in bag_data and len(bag_data["action.claw"]) > 0

    main_time_line_timestamps_ns_effector_dexhand = None
    main_time_line_timestamps_ns_effector_lejuclaw = None

    if has_dexhand:
        main_time_line_timestamps_effector_dexhand = np.array(
            [msg["timestamp"] for msg in bag_data["action.qiangnao"]]
        )
        main_time_line_timestamps_ns_effector_dexhand = (
            main_time_line_timestamps_effector_dexhand * 1e9
        ).astype(np.int64)

    if has_lejuclaw:
        main_time_line_timestamps_effector_lejuclaw = np.array(
            [msg["timestamp"] for msg in bag_data["action.claw"]]
        )
        main_time_line_timestamps_ns_effector_lejuclaw = (
            main_time_line_timestamps_effector_lejuclaw * 1e9
        ).astype(np.int64)

    # 构建基础的 all_low_dim_data（保持原有结构不变）
    # TODO:
    all_low_dim_data = {
        "timestamps": main_time_line_timestamps_ns,
        "head_color_mp4_timestamps": main_time_line_timestamps_ns_head_camera,
        "head_depth_mkv_timestamps": main_time_line_timestamps_ns_head_camera_depth,
        "camera_extrinsics_timestamps": main_time_line_timestamps_ns_extrinsic,
        "joint_timestamps": main_time_line_timestamps_ns_joint,
        "head_timestamps": main_time_line_timestamps_ns_head,
        "action": {
            "effector": {
                "position": control_robot_hand_position_both if control_robot_hand_position_both is not None else leju_claw_command__position,
                # "index": main_time_line_timestamps_ns,
                "names": ["l_thumbMCP", "l_thumbCMC", "l_indexMCP", "l_middleMCP", "l_ringMCP", "l_littleMCP", "r_thumbMCP", "r_thumbCMC", "r_indexMCP", "r_middleMCP", "r_ringMCP", "r_littleMCP"] if control_robot_hand_position_both is not None else ["right_outer_finger", "left_outer_finger"],
            },
            "joint": {
                "position": kuavo_arm_traj__position,
                "velocity": joint_cmd__joint_v[:, 12:26],
                "names": ["zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint","zarm_l7_joint", "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint","zarm_r7_joint",],
            },
            "head": {
                "position": joint_cmd__joint_q[:, 26:28],
                "velocity": joint_cmd__joint_v[:, 26:28],
                "names": ["zhead_1_joint", "zhead_2_joint"],
            },
        },
        "state": {
            "effector": {
                "position": control_robot_hand_position_state_both if control_robot_hand_position_state_both is not None else leju_claw_state__position,
                "names": ["l_thumbMCP", "l_thumbCMC", "l_indexMCP", "l_middleMCP", "l_ringMCP", "l_littleMCP", "r_thumbMCP", "r_thumbCMC", "r_indexMCP", "r_middleMCP", "r_ringMCP", "r_littleMCP"] if control_robot_hand_position_state_both is not None else ["right_outer_finger", "left_outer_finger"],
            },
            "head": {
                "effort": head_effort,
                "position": sensors_data_raw__joint_q[:, 26:28],
                "velocity": sensors_data_raw__joint_v[:, 26:28],
                "naems": ["zhead_1_joint", "zhead_2_joint"],
            },
            "joint": {
                "current_value": joint_current,
                "effort": joint_effort,
                "position": sensors_data_raw__joint_q[:, 12:26],
                "velocity": sensors_data_raw__joint_v[:, 12:26],
                "names": ["zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint","zarm_l7_joint", "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint","zarm_r7_joint",],
            },
            "end": {
                "position": end_position,
                "orientation": end_orientation,
            },
        },
        "imu": {
            "gyro_xyz": sensors_data_raw__imu_data[:, 0:3],
            "acc_xyz": sensors_data_raw__imu_data[:, 3:6],
            "free_acc_xyz": sensors_data_raw__imu_data[:, 6:9],
            "quat_xyzw": sensors_data_raw__imu_data[:, 9:13],
        },
    }
    # 条件性添加腿部数据：只有非轮臂设备才添加腿部数据
    if not is_wheel_arm:
        print(f"[INFO] 设备类型为非轮臂，添加腿部数据到HDF5")
        all_low_dim_data["leg_timestamps"] = main_time_line_timestamps_ns_head
        all_low_dim_data["action"]["leg"] = {
            "position": joint_cmd__joint_q[:, :12],
            "velocity": joint_cmd__joint_v[:, :12],
            "names": ["leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint", "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint"],
        }
        all_low_dim_data["state"]["leg"] = {
            "position": sensors_data_raw__joint_q[:, 0:12],
            "velocity": sensors_data_raw__joint_v[:, 0:12],
            "names": ["leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint", "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint"],
        }
    else:
        print(f"[INFO] 设备类型为轮臂(LB开头)，跳过腿部数据，不添加到HDF5")
    # 条件添加左相机时间戳
    if main_time_line_timestamps_ns_left_camera is not None:
        all_low_dim_data["hand_left_color_mp4_timestamps"] = (
            main_time_line_timestamps_ns_left_camera
        )
        all_low_dim_data["hand_left_depth_mkv_timestamps"] = (
            main_time_line_timestamps_ns_left_camera_depth
        )

    # 条件添加右相机时间戳
    if main_time_line_timestamps_ns_right_camera is not None:
        all_low_dim_data["hand_right_color_mp4_timestamps"] = (
            main_time_line_timestamps_ns_right_camera
        )
        all_low_dim_data["hand_right_depth_mkv_timestamps"] = (
            main_time_line_timestamps_ns_right_camera_depth
        )

    # 条件添加效果器时间戳（只添加存在的那个，参考 recursive_filter_and_position 的逻辑）
    if main_time_line_timestamps_ns_effector_dexhand is not None:
        all_low_dim_data["effector_dexhand_timestamps"] = (
            main_time_line_timestamps_ns_effector_dexhand
        )

    if main_time_line_timestamps_ns_effector_lejuclaw is not None:
        all_low_dim_data["effector_lejuclaw_timestamps"] = (
            main_time_line_timestamps_ns_effector_lejuclaw
        )

    # ===== 新增：为 ori_bag_data 构建与 all_low_dim_data 对应的原始（未对齐）结构，并返回 =====
    try:
        # safe-get helper
        def _arr_from(bd, key, dtype=np.float32):
            return np.array([msg["data"] for msg in bd.get(key, [])], dtype=dtype)

        # 时间戳（原始）
        if "head_cam_h" in ori_bag_data and len(ori_bag_data["head_cam_h"]) > 0:
            main_time_line_timestamps_ori = np.array([msg["timestamp"] for msg in ori_bag_data["head_cam_h"]])
            main_time_line_timestamps_ns_ori = (main_time_line_timestamps_ori * 1e9).astype(np.int64)
        else:
            main_time_line_timestamps_ori = None
            main_time_line_timestamps_ns_ori = None

        main_time_line_timestamps_ori_head_camera_depth = np.array(
            [msg["timestamp"] for msg in ori_bag_data["head_cam_h_depth"]]
        )
        main_time_line_timestamps_ns_ori_head_camera_depth= (
            main_time_line_timestamps_ori_head_camera_depth * 1e9
        ).astype(np.int64)

        main_time_line_timestamps_ori_head = np.array(
            [msg["timestamp"] for msg in ori_bag_data["observation.sensorsData.joint_q"]]
        )
        main_time_line_timestamps_ns_ori_head = (main_time_line_timestamps_ori_head * 1e9).astype(
            np.int64
        )
        main_time_line_timestamps_ns_ori_extrinsic = main_time_line_timestamps_ns_ori_head

        main_time_line_timestamps_ori_joint = np.array(
            [msg["timestamp"] for msg in ori_bag_data["observation.sensorsData.joint_q"]]
        )
        main_time_line_timestamps_ns_ori_joint = (main_time_line_timestamps_ori_joint * 1e9).astype(
            np.int64
        )
        # 构建常用原始数组（与上方同名但后缀 _ori）
        sensors_data_raw__joint_q_ori = _arr_from(ori_bag_data, "observation.sensorsData.joint_q")
        sensors_data_raw__joint_v_ori = _arr_from(ori_bag_data, "observation.sensorsData.joint_v")
        sensors_data_raw__joint_effort_ori = (
            PostProcessorUtils.current_to_torque_batch(
                np.array([msg["data"] for msg in ori_bag_data.get("observation.sensorsData.joint_current", [])], dtype=np.float32),
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
            if "observation.sensorsData.joint_current" in ori_bag_data
            else np.array([], dtype=np.float32)
        )
        sensors_data_raw__joint_current_ori = PostProcessorUtils.torque_to_current_batch(
            np.array([msg["data"] for msg in ori_bag_data.get("observation.sensorsData.joint_current", [])], dtype=np.float32),
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
        joint_current_ori = sensors_data_raw__joint_current_ori[:, 12:26] 
        joint_cmd__joint_q_ori = _arr_from(ori_bag_data, "action.joint_cmd.joint_q")
        joint_cmd__joint_v_ori = _arr_from(ori_bag_data, "action.joint_cmd.joint_v")
        kuavo_arm_traj__position_ori = _arr_from(ori_bag_data, "action.kuavo_arm_traj")
        leju_claw_state__position_ori = _arr_from(ori_bag_data, "observation.claw")
        leju_claw_command__position_ori = _arr_from(ori_bag_data, "action.claw")
        sensors_data_raw__imu_data_ori = _arr_from(ori_bag_data, "observation.sensorsData.imu")
        end_position_ori = np.array([msg["data"] for msg in ori_bag_data.get("end.position", [])], dtype=np.float32) if "end.position" in ori_bag_data else np.array([], dtype=np.float32)
        end_orientation_ori = np.array([msg["data"] for msg in ori_bag_data.get("end.orientation", [])], dtype=np.float32) if "end.orientation" in ori_bag_data else np.array([], dtype=np.float32)
        dex_hand_command__position_ori = _arr_from(ori_bag_data, "action.qiangnao")
        try:
            control_robot_hand_position_state_both_ori = qiangnao_state = np.array(
                [msg["data"] for msg in ori_bag_data["observation.qiangnao"]], dtype=np.float32
            )
        except KeyError:
            print("[WARN] 未找到 'observation.qiangnao' 数据，使用空值")
            control_robot_hand_position_state_both_ori = qiangnao_state = None
        # 左右相机 extrinsics 原始
        head_extrinsics_ori = ori_bag_data.get("head_camera_extrinsics", [])
        left_extrinsics_ori = ori_bag_data.get("left_hand_camera_extrinsics", [])
        right_extrinsics_ori = ori_bag_data.get("right_hand_camera_extrinsics", [])

        # 构建 all_low_dim_data_original（字段尽量与 all_low_dim_data 对齐）
        all_low_dim_data_original = {
            "timestamps": main_time_line_timestamps_ns_ori,
            "head_color_mp4_timestamps": main_time_line_timestamps_ns_ori,
            "head_depth_mkv_timestamps": main_time_line_timestamps_ns_ori_head_camera_depth,
            "camera_extrinsics_timestamps": main_time_line_timestamps_ns_ori_extrinsic,
            "joint_timestamps": main_time_line_timestamps_ns_ori_joint,
            "head_timestamps": main_time_line_timestamps_ns_ori_head,
            "action": {
                "effector": {
                    "position": leju_claw_command__position_ori if leju_claw_command__position_ori.size else dex_hand_command__position_ori,
                    "names": ["right_outer_finger", "left_outer_finger"] if leju_claw_command__position_ori.size else ["l_thumbMCP", "l_thumbCMC", "l_indexMCP", "l_middleMCP", "l_ringMCP", "l_littleMCP", "r_thumbMCP", "r_thumbCMC", "r_indexMCP", "r_middleMCP", "r_ringMCP", "r_littleMCP"],
                },
                "joint": {
                    "position": kuavo_arm_traj__position_ori, 
                    "velocity": joint_cmd__joint_v_ori[:, 12:26],
                    "names": ["zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint","zarm_l7_joint", "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint","zarm_r7_joint",],
                },
                "head": {
                    "position": joint_cmd__joint_q_ori[:, 26:28] if joint_cmd__joint_q_ori.size else np.array([]),
                    "velocity": joint_cmd__joint_v_ori[:, 26:28] if joint_cmd__joint_v_ori.size else np.array([]),
                    "names": ["zhead_1_joint", "zhead_2_joint"],
                },
            },
            "state": {
                "effector": {
                    "position": control_robot_hand_position_state_both_ori if control_robot_hand_position_state_both_ori is not None else leju_claw_state__position_ori,
                    "names": ["l_thumbMCP", "l_thumbCMC", "l_indexMCP", "l_middleMCP", "l_ringMCP", "l_littleMCP", "r_thumbMCP", "r_thumbCMC", "r_indexMCP", "r_middleMCP", "r_ringMCP", "r_littleMCP"] if control_robot_hand_position_state_both_ori is not None else ["right_outer_finger", "left_outer_finger"],
                },
                "head": {
                    "effort": sensors_data_raw__joint_effort_ori[:, 26:28] if sensors_data_raw__joint_effort_ori.size else np.array([]),
                    "position": sensors_data_raw__joint_q_ori[:, 26:28] if sensors_data_raw__joint_q_ori.size else np.array([]),
                    "velocity": sensors_data_raw__joint_v_ori[:, 26:28] if sensors_data_raw__joint_v_ori.size else np.array([]),
                    "naems": ["zhead_1_joint", "zhead_2_joint"],
                },
                "joint": {
                    "current_value": joint_current_ori,
                    "effort": sensors_data_raw__joint_effort_ori[:, 12:26] if sensors_data_raw__joint_effort_ori.size else np.array([]),
                    "position": sensors_data_raw__joint_q_ori[:, 12:26] if sensors_data_raw__joint_q_ori.size else np.array([]),
                    "velocity": sensors_data_raw__joint_v_ori[:, 12:26] if sensors_data_raw__joint_v_ori.size else np.array([]),
                    "names": ["zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint","zarm_l7_joint", "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint","zarm_r7_joint",],
                },
                "end": {"position": end_position_ori, "orientation": end_orientation_ori},
            },
            "imu": {
                "gyro_xyz": sensors_data_raw__imu_data_ori[:, 0:3] if sensors_data_raw__imu_data_ori.size else np.array([]),
                "acc_xyz": sensors_data_raw__imu_data_ori[:, 3:6] if sensors_data_raw__imu_data_ori.size else np.array([]),
                "free_acc_xyz": sensors_data_raw__imu_data_ori[:, 6:9] if sensors_data_raw__imu_data_ori.size else np.array([]),
                "quat_xyzw": sensors_data_raw__imu_data_ori[:, 9:13] if sensors_data_raw__imu_data_ori.size else np.array([]),
            },
        }

        # 条件性添加腿部数据（原始）
        if not is_wheel_arm and sensors_data_raw__joint_q_ori.size:
            all_low_dim_data_original["leg_timestamps"] = (np.array([msg["timestamp"] for msg in ori_bag_data.get("observation.sensorsData.joint_q", [])]) * 1e9).astype(np.int64)
            all_low_dim_data_original["action"]["leg"] = {
                "position": joint_cmd__joint_q_ori[:, :12] if joint_cmd__joint_q_ori.size else np.array([]),
                "velocity": joint_cmd__joint_v_ori[:, :12] if joint_cmd__joint_v_ori.size else np.array([]),
                "names": ["leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint", "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint"],
            }
            all_low_dim_data_original["state"]["leg"] = {
                "position": sensors_data_raw__joint_q_ori[:, 0:12] if sensors_data_raw__joint_q_ori.size else np.array([]),
                "velocity": sensors_data_raw__joint_v_ori[:, 0:12] if sensors_data_raw__joint_v_ori.size else np.array([]),
                "names": ["leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint", "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint"],
            }
    except Exception as _e:
        # 若构建原始数据失败，设为 None 并继续（不应阻断主流程）
        print(f"[WARN] 构建 all_low_dim_data_original 时出错: {_e}")
        all_low_dim_data_original = None
    # ===== 结束新增 =====

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
        all_low_dim_data_original,  # 新增返回
    )


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


def diagnose_frame_data(data):
    for k, v in data.items():
        print(f"Field: {k}")
        print(f"  Shape    : {v.shape}")
        print(f"  Dtype    : {v.dtype}")
        print(f"  Type     : {type(v).__name__}")
        print("-" * 40)


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


def generate_dataset_file(
    raw_config: Config,
    bag_files: list,
    moment_json_dir: str,
    metadata_json_dir: str,
    output_dir: str,
    scene: str,
    sub_scene: str,
    continuous_action: str,
    mode: str,
    min_duration=5.0,
):

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

        # recursive_filter_and_position(all_low_dim_data)
        output_file = PostProcessorUtils.save_to_hdf5(
            all_low_dim_data, proprio_stats_dir + "proprio_stats.hdf5"
        )
        # 保存原始（未对齐）proprio 数据到 proprio_stats_original.hdf5（若存在）
        if all_low_dim_data_original is not None:
            PostProcessorUtils.save_to_hdf5(
                all_low_dim_data_original,
                proprio_stats_dir + "proprio_stats_original.hdf5",
            )
        import h5py

        extrinsic_hdf5_path = proprio_stats_dir + "proprio_stats.hdf5"

        with h5py.File(extrinsic_hdf5_path, "a") as f:
            group = f.require_group("camera_extrinsic_params")
            for cam_key, extrinsics in [
                ("head_camera", head_extrinsics),
                ("left_hand_camera", left_extrinsics),
                ("right_hand_camera", right_extrinsics),
            ]:
                if extrinsics:
                    rot = np.array(
                        [x["rotation_matrix"] for x in extrinsics], dtype=np.float32
                    )
                    trans = np.array(
                        [x["translation_vector"] for x in extrinsics], dtype=np.float32
                    )
                    # 修复：将秒级时间戳转换为纳秒级
                    ts_seconds = np.array(
                        [x["timestamp"] for x in extrinsics], dtype=np.float64
                    )
                    ts_nanoseconds = (ts_seconds * 1e9).astype(np.int64)
                    cam_group = cam_group = group.require_group(cam_key)
                    cam_group.create_dataset("camera_rotation_matrix", data=rot)
                    cam_group.create_dataset("camera_translation_vector", data=trans)
                    cam_group.create_dataset("index", data=ts_nanoseconds)
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
        if compressed_group:
            starttime = time.time()
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
            endtime = time.time()
            print(f"[TIME] 处理压缩深度视频耗时: {endtime - starttime:.2f}秒")
        if uncompressed_group:
            starttime = time.time()
            print(
                f"[INFO] 以下相机为未压缩深度，将用 save_depth_videos_16U_parallel: {list(uncompressed_group.keys())}"
            )
            save_depth_videos_16U_parallel(
                uncompressed_group, output_dir=depth_dir, raw_config=raw_config
            )
            endtime = time.time()
            print(f"[TIME] 处理无压缩深度视频耗时: {endtime - starttime:.2f}秒")
        save_camera_info_to_json_new(
            info_per_cam, distortion_model, output_dir=parameters_dir
        )
        starttime = time.time()
        save_color_videos_parallel(
            imgs_per_cam, output_dir=video_dir, raw_config=raw_config
        )
        endtime = time.time()
        print(f"[TIME] 处理彩色视频耗时: {endtime - starttime:.2f}秒")
        cameras = ["head_cam_h", "wrist_cam_r", "wrist_cam_l"]
        save_camera_extrinsic_params(cameras=cameras, output_dir=parameters_dir)
        merge_metadata_and_moment(
            metadata_json_dir,
            moment_json_dir,
            task_info_dir + "metadata.json",
            uuid,
            raw_config,
            bag_time_info=bag_time_info,
            main_time_line_timestamps=main_time_line_timestamps,
            output_dir=output_dir,
        )

        # 新增：数据一致性验证
        temp_uuid_path = os.path.join(output_dir, uuid)
        if os.path.exists(temp_uuid_path):
            print(f"开始验证 episode {uuid} 的数据一致性...")
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

                error_msg = (
                    f"Episode {uuid} 数据一致性验证失败: {'; '.join(error_details)}"
                )
                raise Exception(error_msg)

            print(f"Episode {uuid} 数据一致性验证通过 ✓")

        else:
            raise Exception(f"Episode {uuid} 临时路径不存在: {temp_uuid_path}")


def port_kuavo_rosbag(
    raw_config: Config,
    bag_dir: str,
    moment_json_dir: str,
    metadata_json_dir: str,
    output_dir: str,
    scene: str,
    sub_scene: str,
    continuous_action: str,
    mode: str,
    min_duration=5.0,
):

    # 测试代码：直接指定bagfiles
    bag_files = list_bag_files_auto(bag_dir)

    generate_dataset_file(
        bag_files=bag_files,
        raw_config=raw_config,
        moment_json_dir=moment_json_dir,
        metadata_json_dir=metadata_json_dir,
        output_dir=output_dir,
        scene=scene,
        sub_scene=sub_scene,
        continuous_action=continuous_action,
        mode=mode,
        min_duration=min_duration,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Kuavo ROSbag to HDF5 Converter")
    parser.add_argument(
        "--bag_dir",
        default="/home/leju_kuavo/Data/长三角一体化示范区智能机器人训练中心-XGZSL_16_001_P4-199_20250903180715",
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
        "--scene", default="test_scene", type=str, required=False, help="scene"
    )
    parser.add_argument(
        "--sub_scene",
        default="test_sub_scene",
        type=str,
        required=False,
        help="sub_scene",
    )
    parser.add_argument(
        "--continuous_action",
        default="test_continuous_action",
        type=str,
        required=False,
        help="continuous_action",
    )

    parser.add_argument(
        "--mode",
        default="simplified",
        type=str,
        required=False,
        help="file structure mode, either 'complete' or 'simplified'. Default is 'simplified'.",
    )
    # parser.add_argument("--use_current_topic", default = False, type=bool, required=False, help="Choose the topic name of joint current state. Old bags use 'joint_current', new bags use 'joint_torque'. Default is False, which means using 'joint_torque'.")
    parser.add_argument(
        "--min_duration", type=float, default=5.0, help="最小时长（秒），影响帧数要求"
    )
    parser.add_argument("-v", "--process_ID", default="v0", type=str, help="process ID")
    parser.add_argument(
        "--config", type=str, default="/home/leju_kuavo/kuavo_cb/data_cvt/kuavo-data-toolbox/coscene_action/rosbag2hdf5/request.json", help="Path to config YAML file"
    )
    args = parser.parse_args()

    # 加载配置文件
    config = load_config_from_json(args.config)

    # 从配置获取参数
    if args.bag_dir is not None:
        bag_DIR = args.bag_dir
    print(f"Bag directory: {bag_DIR}")
    if args.process_ID is not None:
        config.id = args.process_ID
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
    if args.scene is not None:
        scene = args.scene
    if args.sub_scene is not None:
        sub_scene = args.sub_scene
    if args.continuous_action is not None:
        continuous_action = args.continuous_action
    if args.mode is not None:
        if args.mode == "complete":
            mode = "complete"
        elif args.mode == "simplified":
            mode = "simplified"
        else:
            raise ValueError("Invalid mode. Choose either 'complete' or 'simplified'.")
    # if args.use_current_topic is not None:
    #     USE_JOINT_CURRENT_STATE= args.use_current_topic

    ID = config.id
    min_duration = args.min_duration

    port_kuavo_rosbag(
        config,
        bag_DIR,
        moment_json_DIR,
        metadata_json_DIR,
        output_DIR,
        scene,
        sub_scene,
        continuous_action,
        mode,
        min_duration,
    )
