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
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rosbag
import tqdm
from slave_utils import detect_and_trim_bag_data, load_camera_info_per_camera, load_raw_depth_images_per_camera, load_raw_images_per_camera, recursive_filter_and_position, save_camera_extrinsic_params, save_camera_info_to_json_new, save_color_videos_parallel, save_depth_videos_16U_parallel, save_depth_videos_enhanced_parallel, save_depth_videos_parallel, validate_episode_data_consistency
from config_dataset_slave import Config,  load_config_from_json
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
    rosbag_actual_start_time,     # 实际数据开始时间
    rosbag_actual_end_time,       # 实际数据结束时间
    rosbag_original_start_time,   # 原始bag开始时间
    rosbag_original_end_time,     # 原始bag结束时间
    action_original_start_time,   # 动作原始开始时间
    action_duration,              # 动作持续时间
    frame_rate,                   # 帧率
    total_frames                  # 总帧数
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
    if action_end_time < rosbag_actual_start_time or action_start_time > rosbag_actual_end_time:
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
        bag = rosbag.Bag(bag_path, 'r')
        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        bag_duration = bag_end_time - bag_start_time
        bag.close()
        
        # 转换为带时区的ISO格式（东八区）
        start_datetime = datetime.datetime.fromtimestamp(
            bag_start_time, 
            tz=datetime.timezone(datetime.timedelta(hours=8))
        )
        start_iso = start_datetime.isoformat()
        
        # 转换为纳秒
        start_nanoseconds = int(bag_start_time * 1e9)
        
        return {
            'unix_timestamp': bag_start_time,
            'iso_format': start_iso,
            'nanoseconds': start_nanoseconds,
            'duration': bag_duration,
            'end_time': bag_end_time
        }
        
    except Exception as e:
        print(f"获取bag时间信息失败: {e}")
        return {
            'unix_timestamp': None,
            'iso_format': None,
            'nanoseconds': None,
            'duration': None,
            'end_time': None
        }

def format_size_gb(size_bytes):
    size_gb = size_bytes / (1024 ** 3)
    int_part = int(size_gb)
    frac_part = int(round((size_gb - int_part) * 100))
    return f"{int_part}p{frac_part:02d}"

def create_file_structure(scene, sub_scene, continuous_action, bag_path, save_dir,mode="simplified"):
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
        main_dir = f"{scene}-{size_str}GB_{count}counts_{hour_str}"
        sub_dir = f"{sub_scene}-{size_str}GB_{count}counts_{hour_str}"
        action_dir = f"{continuous_action}-{size_str}GB_{count}counts_{hour_str}"

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
            audio_dir
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
        for d in [ base_path, depth_dir, video_dir, parameters_dir, proprio_stats_dir]:
            os.makedirs(d, exist_ok=True)

    print(f"已创建目录结构：{base_path}")
    return episode_uuid,base_path+'/' ,depth_dir,video_dir,parameters_dir,proprio_stats_dir
    
# def merge_metadata_and_moment(metadata_path, moment_path, output_path,uuid,raw_config,bag_time_info=None,main_time_line_timestamps=None):
#     """
#     合并 metadata 和 moment 数据，并添加 bag 时间信息和计算帧数
#     Args:
#         metadata_path: metadata.json 文件路径
#         moment_path: moment.json 文件路径
#         output_path: 输出文件路径
#         uuid: 唯一标识符
#         raw_config: 原始配置对象
#         bag_time_info: bag时间信息字典（可选）
#         main_time_line_timestamps: 经过帧率对齐后的时间戳数组（纳秒）
#     """
#     frequency = raw_config.train_hz if hasattr(raw_config, 'train_hz') else 30
#     # 读取 metadata.json
#     with open(metadata_path, 'r', encoding='utf-8') as f:
#         metadata = json.load(f)

#     # 读取 moment.json
#     with open(moment_path, 'r', encoding='utf-8') as f:
#         moment = json.load(f)
#     # 获取时间信息
#     rosbag_actual_start_time = None
#     rosbag_actual_end_time = None
#     rosbag_original_start_time = None
#     rosbag_original_end_time = None
#     total_frames = 0
#     # 实际数据时间范围
#     # 实际数据时间范围
#     if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
#         # 调试：打印原始时间戳
#         print(f"原始时间戳前3个: {main_time_line_timestamps[:3]}")
#         print(f"原始时间戳后3个: {main_time_line_timestamps[-3:]}")
        
#         # 检查时间戳是否已经是秒格式还是纳秒格式
#         if main_time_line_timestamps[0] > 1e12:  # 如果大于1e12，认为是纳秒格式
#             timestamps_seconds = main_time_line_timestamps / 1e9
#             print("时间戳格式：纳秒 -> 秒")
#         else:
#             timestamps_seconds = main_time_line_timestamps
#             print("时间戳格式：已经是秒")
            
#         rosbag_actual_start_time = timestamps_seconds[0]
#         rosbag_actual_end_time = timestamps_seconds[-1]
#         total_frames = len(main_time_line_timestamps)
        
#         # 调试：打印转换后的时间戳
#         print(f"转换后时间戳前3个: {timestamps_seconds[:3]}")
#         print(f"转换后时间戳后3个: {timestamps_seconds[-3:]}")
        
#         # 验证时间戳转换
#         start_datetime = datetime.datetime.fromtimestamp(rosbag_actual_start_time, tz=datetime.timezone(datetime.timedelta(hours=8)))
#         end_datetime = datetime.datetime.fromtimestamp(rosbag_actual_end_time, tz=datetime.timezone(datetime.timedelta(hours=8)))
        
#         print(f"实际开始时间验证: {start_datetime.isoformat()}")
#         print(f"实际结束时间验证: {end_datetime.isoformat()}")
    
#     # 原始bag时间范围
#     if bag_time_info:
#         rosbag_original_start_time = bag_time_info.get('unix_timestamp')
#         rosbag_original_end_time = bag_time_info.get('end_time')
#     # 构造 action_config
#     print(f"时间信息:")
#     if rosbag_original_start_time and rosbag_original_end_time:
#         print(f"  原始bag时间: {rosbag_original_start_time:.6f}s - {rosbag_original_end_time:.6f}s")
#     if rosbag_actual_start_time and rosbag_actual_end_time:
#         print(f"  实际数据时间: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s")
#     print(f"  总帧数: {total_frames}")
#     action_config = []

#     for m in moment.get("moments", []):
#         # 兼容两种字段名称：优先使用 customizedFields，如果不存在则使用 attribute
#         attribute = m.get("customizedFields", {})
#         if not attribute:  # 如果 customizedFields 为空或不存在
#             attribute = m.get("attribute", {})
        
#         trigger_time = m.get("triggerTime", "")
#         duration_str = m.get("duration", "0s")
        
#         # 添加调试信息，帮助确认读取的字段来源
#         field_source = "customizedFields" if m.get("customizedFields") else "attribute"
#         print(f"从 {field_source} 读取动作信息:")
#         print(f"  skill: {attribute.get('skill', '')}")
#         print(f"  action_text: {attribute.get('action_text', '')}")
#         print(f"  english_action_text: {attribute.get('english_action_text', '')}")
        
#         start_frame = None
#         end_frame = None
        
#         if (rosbag_actual_start_time is not None and 
#             rosbag_actual_end_time is not None and 
#             trigger_time):
            
#             try:
#                 # 解析动作时间
#                 trigger_datetime = datetime.datetime.fromisoformat(trigger_time.replace('Z', '+00:00'))
#                 action_original_start_time = trigger_datetime.timestamp()
                
#                 # 解析持续时间
#                 action_duration = 0
#                 if duration_str.endswith('s'):
#                     action_duration = float(duration_str[:-1])
                
#                 # 计算帧数
#                 start_frame, end_frame = calculate_action_frames(
#                     rosbag_actual_start_time=rosbag_actual_start_time,
#                     rosbag_actual_end_time=rosbag_actual_end_time,
#                     rosbag_original_start_time=rosbag_original_start_time,
#                     rosbag_original_end_time=rosbag_original_end_time,
#                     action_original_start_time=action_original_start_time,
#                     action_duration=action_duration,
#                     frame_rate=frequency,
#                     total_frames=total_frames
#                 )
                
#                 print(f"动作: {attribute.get('action_text', '')}")
#                 print(f"  动作时间: {trigger_datetime.isoformat()}")
#                 print(f"  原始开始时间: {action_original_start_time:.6f}s")
#                 print(f"  原始结束时间: {action_original_start_time + action_duration:.6f}s")
#                 print(f"  持续时间: {action_duration:.3f}s")
#                 print(f"  计算得到帧数: {start_frame} - {end_frame}")
                
#                 # 更详细的调试信息
#                 if start_frame is None or end_frame is None:
#                     print(f"  调试信息:")
#                     print(f"    实际数据范围: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s")
#                     print(f"    动作时间范围: {action_original_start_time:.6f}s - {action_original_start_time + action_duration:.6f}s")
#                     print(f"    动作是否在数据范围内: {action_original_start_time >= rosbag_actual_start_time and action_original_start_time + action_duration <= rosbag_actual_end_time}")
#                     print(f"    动作开始是否在数据范围后: {action_original_start_time > rosbag_actual_end_time}")
#                     print(f"    动作结束是否在数据范围前: {action_original_start_time + action_duration < rosbag_actual_start_time}")
                
#                 # 验证计算结果
#                 if start_frame is not None and end_frame is not None:
#                     actual_start_time = rosbag_actual_start_time + (start_frame / total_frames) * (rosbag_actual_end_time - rosbag_actual_start_time)
#                     actual_end_time = rosbag_actual_start_time + (end_frame / total_frames) * (rosbag_actual_end_time - rosbag_actual_start_time)
#                     print(f"  验证-实际开始时间: {actual_start_time:.6f}s")
#                     print(f"  验证-实际结束时间: {actual_end_time:.6f}s")
                
#                 print("-" * 50)
                
#             except Exception as e:
#                 print(f"计算帧数时出错: {e}")
#                 import traceback
#                 traceback.print_exc()
        
#         action = {
#             "start_frame": start_frame,
#             "end_frame": end_frame,
#             "timestamp_utc": trigger_time,
#             "duration": duration_str,
#             "skill": attribute.get("skill", ""),
#             "action_text": attribute.get("action_text", ""),
#             "english_action_text": attribute.get("english_action_text", "")
#         }
#         action_config.append(action)
#     # 按照 timestamp_utc 排序
#     action_config = sorted(
#         action_config,
#         key=lambda x: x["timestamp_utc"] if x["timestamp_utc"] else ""
#     )

#     # 构造新json，episode_id放在最前
#     new_json = OrderedDict()
#     new_json["episode_id"] = uuid  

#     for k, v in metadata.items():
#         new_json[k] = v
#     if "label_info" not in new_json:
#         new_json["label_info"] = {}
#     new_json["label_info"]["action_config"] = action_config
#     if "key_frame" not in new_json["label_info"]:
#         new_json["label_info"]["key_frame"] = []

#     # 保存
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(new_json, f, ensure_ascii=False, indent=4)
#     print(f"已保存到 {output_path}")
def merge_metadata_and_moment(metadata_path, moment_path, output_path,uuid,raw_config,bag_time_info=None,main_time_line_timestamps=None):
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
    frequency = raw_config.train_hz if hasattr(raw_config, 'train_hz') else 30
    
    # 读取 metadata.json
    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw_metadata = json.load(f)

    # 读取 moment.json
    with open(moment_path, 'r', encoding='utf-8') as f:
        moment = json.load(f)
    
    # 转换新格式 metadata 为旧格式
    converted_metadata = {}
    
    # scene_name 对应 scene_code
    converted_metadata["scene_name"] = raw_metadata.get("scene_code", "")
    
    # sub_scene_name 对应 sub_scene_code (如果不存在则为空)
    converted_metadata["sub_scene_name"] = raw_metadata.get("sub_scene_code", "")
    
    # init_scene_text 对应 sub_scene_code (如果不存在则为空)
    converted_metadata["init_scene_text"] = raw_metadata.get("scene_zh_dec", "")
    
    # english_init_scene_text 对应 scene_en_dec
    converted_metadata["english_init_scene_text"] = raw_metadata.get("scene_en_dec", "")
    
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
        start_datetime = datetime.datetime.fromtimestamp(rosbag_actual_start_time, tz=datetime.timezone(datetime.timedelta(hours=8)))
        end_datetime = datetime.datetime.fromtimestamp(rosbag_actual_end_time, tz=datetime.timezone(datetime.timedelta(hours=8)))
        
        print(f"实际开始时间验证: {start_datetime.isoformat()}")
        print(f"实际结束时间验证: {end_datetime.isoformat()}")
    
    # 原始bag时间范围
    if bag_time_info:
        rosbag_original_start_time = bag_time_info.get('unix_timestamp')
        rosbag_original_end_time = bag_time_info.get('end_time')
    
    # 构造 action_config
    print(f"时间信息:")
    if rosbag_original_start_time and rosbag_original_end_time:
        print(f"  原始bag时间: {rosbag_original_start_time:.6f}s - {rosbag_original_end_time:.6f}s")
    if rosbag_actual_start_time and rosbag_actual_end_time:
        print(f"  实际数据时间: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s")
    print(f"  总帧数: {total_frames}")
    
    action_config = []

    for m in moment.get("moments", []):
        # 从新格式的 customFieldValues 中提取数据
        custom_fields = m.get("customFieldValues", {})
        
        trigger_time = m.get("triggerTime", "")
        duration_str = m.get("duration", "0s")
        
        # 格式化时间戳：将 "Z" 替换为 "+00:00"
        formatted_trigger_time = trigger_time.replace('Z', '+00:00') if trigger_time else ""
        
        # 添加调试信息
        print(f"处理动作数据:")
        print(f"  skill_atomic_en: {custom_fields.get('skill_atomic_en', '')}")
        print(f"  skill_detail: {custom_fields.get('skill_detail', '')}")
        print(f"  en_skill_detail: {custom_fields.get('en_skill_detail', '')}")
        print(f"  原始时间戳: {trigger_time}")
        print(f"  格式化时间戳: {formatted_trigger_time}")
        
        start_frame = None
        end_frame = None
        
        if (rosbag_actual_start_time is not None and 
            rosbag_actual_end_time is not None and 
            trigger_time):
            
            try:
                # 解析动作时间
                trigger_datetime = datetime.datetime.fromisoformat(trigger_time.replace('Z', '+00:00'))
                action_original_start_time = trigger_datetime.timestamp()
                
                # 解析持续时间
                action_duration = 0
                if duration_str.endswith('s'):
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
                    total_frames=total_frames
                )
                
                print(f"动作: {custom_fields.get('skill_detail', '')}")
                print(f"  动作时间: {trigger_datetime.isoformat()}")
                print(f"  原始开始时间: {action_original_start_time:.6f}s")
                print(f"  原始结束时间: {action_original_start_time + action_duration:.6f}s")
                print(f"  持续时间: {action_duration:.3f}s")
                print(f"  计算得到帧数: {start_frame} - {end_frame}")
                
                # 更详细的调试信息
                if start_frame is None or end_frame is None:
                    print(f"  调试信息:")
                    print(f"    实际数据范围: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s")
                    print(f"    动作时间范围: {action_original_start_time:.6f}s - {action_original_start_time + action_duration:.6f}s")
                    print(f"    动作是否在数据范围内: {action_original_start_time >= rosbag_actual_start_time and action_original_start_time + action_duration <= rosbag_actual_end_time}")
                    print(f"    动作开始是否在数据范围后: {action_original_start_time > rosbag_actual_end_time}")
                    print(f"    动作结束是否在数据范围前: {action_original_start_time + action_duration < rosbag_actual_start_time}")
                
                # 验证计算结果
                if start_frame is not None and end_frame is not None:
                    actual_start_time = rosbag_actual_start_time + (start_frame / total_frames) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    actual_end_time = rosbag_actual_start_time + (end_frame / total_frames) * (rosbag_actual_end_time - rosbag_actual_start_time)
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
            "skill": custom_fields.get("skill_atomic_en", ""),  # 从 skill_atomic_en 获取
            "action_text": custom_fields.get("skill_detail", ""),  # 从 skill_detail 获取
            "english_action_text": custom_fields.get("en_skill_detail", "")  # 从 en_skill_detail 获取
        }
        action_config.append(action)
    
    # 按照 timestamp_utc 排序
    action_config = sorted(
        action_config,
        key=lambda x: x["timestamp_utc"] if x["timestamp_utc"] else ""
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
    with open(output_path, 'w', encoding='utf-8') as f:
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
    action_config=None
):
    
    bag_reader = KuavoRosbagReader(raw_config)
    bag_data = bag_reader.process_rosbag(ep_path, start_time=start_time, end_time=end_time, action_config=action_config)
    print("开始检测视频静止区域并进行裁剪...")
    #bag_data = detect_and_trim_bag_data(bag_data, raw_config)
    
    # state = np.array([msg['data'] for msg in bag_data['observation.state']], dtype=np.float32)
    # action = np.array([msg['data'] for msg in bag_data['action']], dtype=np.float32)

    # 常用数据
    sensors_data_raw__joint_q   = state = np.array([msg['data'] for msg in bag_data['observation.sensorsData.joint_q']], dtype=np.float32) # 原 observation.state
    joint_cmd__joint_q          = action = np.array([msg['data'] for msg in bag_data['action.joint_cmd.joint_q']], dtype=np.float32)# 原action
    kuavo_arm_traj__position    = action_kuavo_arm_traj = np.array([msg['data'] for msg in bag_data['action.kuavo_arm_traj']], dtype=np.float32)
    leju_claw_state__position   = claw_state = np.array([msg['data'] for msg in bag_data['observation.claw']], dtype=np.float32)
    leju_claw_command__position = claw_action= np.array([msg['data'] for msg in bag_data['action.claw']], dtype=np.float32)
    # TODO: 夹爪添加velocity和effort数据
    
    control_robot_hand_position_state_both = qiangnao_state = np.array([msg['data'] for msg in bag_data['observation.qiangnao']], dtype=np.float32)
    control_robot_hand_position_both = qiangnao_action = np.array([msg['data'] for msg in bag_data['action.qiangnao']], dtype=np.float32)
    action[:, 12:26] = action_kuavo_arm_traj

    #新增数据
    sensors_data_raw__joint_v = state_joint_v = np.array([msg['data'] for msg in bag_data['observation.sensorsData.joint_v']], dtype=np.float32)
    #sensors_data_raw__joint_vd = state_joint_vd = np.array([msg['data'] for msg in bag_data['observation.sensorsData.joint_vd']], dtype=np.float32)

    state_joint_current = np.array([msg['data'] for msg in bag_data['observation.sensorsData.joint_current']], dtype=np.float32)
    sensors_data_raw__joint_effort = state_joint_effort = PostProcessorUtils.current_to_torque_batch(state_joint_current, 
                                                               MOTOR_C2T=[2, 1.05, 1.05, 2, 2.1, 2.1, 
                                                                            2, 1.05, 1.05, 2, 2.1, 2.1,
                                                                            1.05, 5, 2.3, 5, 4.7, 4.7, 4.7,
                                                                            1.05, 5, 2.3, 5, 4.7, 4.7, 4.7,
                                                                            0.21, 4.7])
    sensors_data_raw__joint_current = PostProcessorUtils.torque_to_current_batch(state_joint_current, 
                                                               MOTOR_C2T=[2, 1.05, 1.05, 2, 2.1, 2.1, 
                                                                            2, 1.05, 1.05, 2, 2.1, 2.1,
                                                                            1.05, 5, 2.3, 5, 4.7, 4.7, 4.7,
                                                                            1.05, 5, 2.3, 5, 4.7, 4.7, 4.7,
                                                                            0.21, 4.7])
        
    head_extrinsics = bag_data.get("head_camera_extrinsics", [])
    left_extrinsics = bag_data.get("left_hand_camera_extrinsics", [])
    right_extrinsics = bag_data.get("right_hand_camera_extrinsics", []) 
    end_position=np.array([msg['data'] for msg in bag_data['end.position']], dtype=np.float32)
    end_orientation=np.array([msg['data'] for msg in bag_data['end.orientation']], dtype=np.float32)
    head_effort = sensors_data_raw__joint_effort[:, 26:28] # 头部关节的effort
    head_current = sensors_data_raw__joint_current[:, 26:28] # 头部关节的current
    joint_effort = sensors_data_raw__joint_effort[:, 12:26] # 其他关节的effort
    joint_current= sensors_data_raw__joint_current[:, 12:26] # 其他关节的current
    sensors_data_raw__imu_data = state_joint_imu = np.array([msg['data'] for msg in bag_data['observation.sensorsData.imu']], dtype=np.float32)
    
    # 注释库帕斯数据中不需要的话题
    
    # joint_cmd__joint_v = action_joint_v = np.array([msg['data'] for msg in bag_data['action.joint_cmd.joint_v']], dtype=np.float32)
    # joint_cmd__tau = action_tau = np.array([msg['data'] for msg in bag_data['action.joint_cmd.tau']], dtype=np.float32)
    # joint_cmd__tau_max = action_tau_max = np.array([msg['data'] for msg in bag_data['action.joint_cmd.tau_max']], dtype=np.float32)
    # joint_cmd__tau_ratio = action_tau_ratio = np.array([msg['data'] for msg in bag_data['action.joint_cmd.tau_ratio']], dtype=np.float32)
    # joint_cmd__tau_joint_kp = action_tau_joint_kp = np.array([msg['data'] for msg in bag_data['action.joint_cmd.tau_joint_kp']], dtype=np.float32)
    # joint_cmd__tau_joint_kd = action_tau_joint_kd = np.array([msg['data'] for msg in bag_data['action.joint_cmd.tau_joint_kd']], dtype=np.float32)
    # joint_cmd__control_modes = action_control_modes = np.array([msg['data'] for msg in bag_data['action.joint_cmd.control_modes']], dtype=np.int32)    



    velocity = None
    effort = None

    imgs_per_cam = load_raw_images_per_camera(bag_data, raw_config.default_camera_names)

    imgs_per_cam_depth,compressed = load_raw_depth_images_per_camera(bag_data, raw_config.default_camera_names)
    info_per_cam ,distortion_model= load_camera_info_per_camera(bag_data, raw_config.default_camera_names)
    #Post-process the data
  # 在 load_raw_episode_data_hdf5 函数中，修改时间戳处理部分

    #Post-process the data
    main_time_line_timestamps = np.array([msg['timestamp'] for msg in bag_data['head_cam_h']])
    main_time_line_timestamps_ns = (main_time_line_timestamps * 1e9).astype(np.int64)
    main_time_line_timestamps_ns_head_camera = main_time_line_timestamps_ns
    main_time_line_timestamps_head_camera_depth = np.array([msg['timestamp'] for msg in bag_data['head_cam_h_depth']])
    main_time_line_timestamps_ns_head_camera_depth = (main_time_line_timestamps_head_camera_depth * 1e9).astype(np.int64)

    # 检查左右相机数据是否存在
    main_time_line_timestamps_ns_left_camera = None
    main_time_line_timestamps_ns_right_camera = None

    if 'wrist_cam_l' in bag_data and len(bag_data['wrist_cam_l']) > 0:
        main_time_line_timestamps_left_camera = np.array([msg['timestamp'] for msg in bag_data['wrist_cam_l']])
        main_time_line_timestamps_ns_left_camera = (main_time_line_timestamps_left_camera * 1e9).astype(np.int64)
        main_time_line_timestamps_left_camera_depth = np.array([msg['timestamp'] for msg in bag_data['wrist_cam_l_depth']])
        main_time_line_timestamps_ns_left_camera_depth = (main_time_line_timestamps_left_camera_depth * 1e9).astype(np.int64)

    if 'wrist_cam_r' in bag_data and len(bag_data['wrist_cam_r']) > 0:
        main_time_line_timestamps_right_camera = np.array([msg['timestamp'] for msg in bag_data['wrist_cam_r']])
        main_time_line_timestamps_ns_right_camera = (main_time_line_timestamps_right_camera * 1e9).astype(np.int64)
        main_time_line_timestamps_right_camera_depth = np.array([msg['timestamp'] for msg in bag_data['wrist_cam_r_depth']])
        main_time_line_timestamps_ns_right_camera_depth = (main_time_line_timestamps_right_camera_depth * 1e9).astype(np.int64)

    # 其他时间戳处理
    main_time_line_timestamps_head = np.array([msg['timestamp'] for msg in bag_data['observation.sensorsData.joint_q']])
    main_time_line_timestamps_ns_head = (main_time_line_timestamps_head * 1e9).astype(np.int64)
    main_time_line_timestamps_ns_extrinsic = main_time_line_timestamps_ns_head
    main_time_line_timestamps_joint = np.array([msg['timestamp'] for msg in bag_data['action.kuavo_arm_traj']])
    main_time_line_timestamps_ns_joint = (main_time_line_timestamps_joint * 1e9).astype(np.int64)

    # 检查效果器数据存在性（参考 recursive_filter_and_position 函数的逻辑）
    has_dexhand = 'action.qiangnao' in bag_data and len(bag_data['action.qiangnao']) > 0
    has_lejuclaw = 'action.claw' in bag_data and len(bag_data['action.claw']) > 0

    main_time_line_timestamps_ns_effector_dexhand = None
    main_time_line_timestamps_ns_effector_lejuclaw = None

    if has_dexhand:
        main_time_line_timestamps_effector_dexhand = np.array([msg['timestamp'] for msg in bag_data['action.qiangnao']])
        main_time_line_timestamps_ns_effector_dexhand = (main_time_line_timestamps_effector_dexhand * 1e9).astype(np.int64)

    if has_lejuclaw:
        main_time_line_timestamps_effector_lejuclaw = np.array([msg['timestamp'] for msg in bag_data['action.claw']])
        main_time_line_timestamps_ns_effector_lejuclaw = (main_time_line_timestamps_effector_lejuclaw * 1e9).astype(np.int64)

    # 构建基础的 all_low_dim_data（保持原有结构不变）
    all_low_dim_data = {
        "timestamps": main_time_line_timestamps_ns,
        "head_color_mp4_camera_timestamps": main_time_line_timestamps_ns_head_camera,
        "head_depth_mkv_camera_timestamps": main_time_line_timestamps_ns_head_camera_depth,
        "camera_extrinsics_timestamps": main_time_line_timestamps_ns_extrinsic,
        "joint_timestamps": main_time_line_timestamps_ns_joint,
        "head_timestamps": main_time_line_timestamps_ns_head,
        "action": {
            "effector": {
                "position(gripper)": leju_claw_command__position,
                "position(dexhand)": control_robot_hand_position_both,
                "index": main_time_line_timestamps_ns
            },
            "joint": {
                "position": kuavo_arm_traj__position, 
                "index": main_time_line_timestamps_ns
            },
            "head": {
                "position": joint_cmd__joint_q[:, 26:28],  
                "index": main_time_line_timestamps_ns
            },
        },
        'state':{
            "effector": {
                "position(gripper)": leju_claw_state__position,
                "position(dexhand)": control_robot_hand_position_state_both,
            },
            "head": {
                "current_value":head_current,
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
        },
        'imu': {
            "gyro_xyz": sensors_data_raw__imu_data[:, 0:3],
            "acc_xyz": sensors_data_raw__imu_data[:, 3:6],
            "free_acc_xyz": sensors_data_raw__imu_data[:, 6:9],
            "quat_xyzw": sensors_data_raw__imu_data[:, 9:13],
        }
    }

    # 条件添加左相机时间戳
    if main_time_line_timestamps_ns_left_camera is not None:
        all_low_dim_data["hand_left_color_mp4_timestamps"] = main_time_line_timestamps_ns_left_camera
        all_low_dim_data["hand_left_depth_mkv_timestamps"] = main_time_line_timestamps_ns_left_camera_depth

    # 条件添加右相机时间戳
    if main_time_line_timestamps_ns_right_camera is not None:
        all_low_dim_data["hand_right_color_mp4_timestamps"] = main_time_line_timestamps_ns_right_camera
        all_low_dim_data["hand_right_depth_mkv_timestamps"] = main_time_line_timestamps_ns_right_camera_depth

    # 条件添加效果器时间戳（只添加存在的那个，参考 recursive_filter_and_position 的逻辑）
    if main_time_line_timestamps_ns_effector_dexhand is not None:
        all_low_dim_data["effector_dexhand_timestamps"] = main_time_line_timestamps_ns_effector_dexhand

    if main_time_line_timestamps_ns_effector_lejuclaw is not None:
        all_low_dim_data["effector_lejuclaw_timestamps"] = main_time_line_timestamps_ns_effector_lejuclaw
   
    return imgs_per_cam, imgs_per_cam_depth, info_per_cam, all_low_dim_data, main_time_line_timestamps, distortion_model, head_extrinsics, left_extrinsics, right_extrinsics,compressed

def list_bag_files_auto(raw_dir):
    bag_files = []
    for i, fname in enumerate(sorted(os.listdir(raw_dir))):
        if fname.endswith('.bag'):
            bag_files.append({
                'link': '',  # 保持为空
                'start': 0,  # 批量设置为0
                'end': 1,    # 批量设置为1
                'local_path': os.path.join(raw_dir, fname)
            })
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
        with open(moments_json_path, 'r', encoding='utf-8') as f:
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
            
            print(f"[MOMENTS] 从moments.json获取时间范围: {moments_start_time} - {moments_end_time}")
            print(f"[MOMENTS] 找到 {len(start_positions)} 个start_position, {len(end_positions)} 个end_position")
            
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
    moment_json_dir:str,
    metadata_json_dir:str,
    output_dir :str,
    scene:str,
    sub_scene:str,
    continuous_action:str,
    mode:str,
):

    episodes = range(len(bag_files))
    for ep_idx in tqdm.tqdm(episodes):
            if os.path.exists(moment_json_dir):
                with open(moment_json_dir, 'r', encoding='utf-8') as f:
                    moments_data = json.load(f)
                    action_config = moments_data.get("moments", [])
            bag_info = bag_files[ep_idx]
            if isinstance(bag_info, dict):
                ep_path = bag_info['local_path']
                start_time = bag_info.get('start', 0)
                end_time = bag_info.get('end', 1)
            else:
                ep_path = bag_info
                start_time = 0
                end_time = 1
            
            moments_start_time, moments_end_time = get_time_range_from_moments(moment_json_dir)
        
            if moments_start_time is not None and moments_end_time is not None:
                print(f"[MOMENTS] 原始时间范围: {start_time} - {end_time}")
                print(f"[MOMENTS] 覆盖使用moments.json时间范围: {moments_start_time} - {moments_end_time}")
                
                start_time = moments_start_time
                end_time = moments_end_time
            else:
                print(f"[MOMENTS] 未找到有效的moments.json时间范围，使用原始时间范围: {start_time} - {end_time}")
        
            from termcolor import colored
            print(colored(f"Processing {ep_path} (time range: {start_time}-{end_time})", "yellow", attrs=["bold"]))
                    # 获取bag时间信息
            bag_time_info = get_bag_time_info(ep_path)
        
            if bag_time_info['iso_format']:
                print(f"Bag开始时间: {bag_time_info['iso_format']}")
                print(f"Bag持续时间: {bag_time_info['duration']:.2f}秒")



            imgs_per_cam, imgs_per_cam_depth, info_per_cam, all_low_dim_data, main_time_line_timestamps, distortion_model, head_extrinsics, left_extrinsics, right_extrinsics,compressed= load_raw_episode_data(
                raw_config=raw_config, 
                ep_path=ep_path,
                start_time=start_time,
                end_time=end_time,
                action_config=action_config
            )

            uuid,task_info_dir,depth_dir,video_dir,parameters_dir,proprio_stats_dir=create_file_structure(scene=scene,sub_scene=sub_scene,continuous_action=continuous_action,bag_path=ep_path,save_dir=output_dir,mode=mode)
            #merge_metadata_and_moment(metadata_json_dir,moment_json_dir,task_info_dir+scene+"_"+sub_scene+"_"+continuous_action+".json",uuid)
            merge_metadata_and_moment(metadata_json_dir,moment_json_dir,task_info_dir+"metadata.json",uuid,raw_config,bag_time_info=bag_time_info,main_time_line_timestamps=main_time_line_timestamps)
            recursive_filter_and_position(all_low_dim_data)
            output_file = PostProcessorUtils.save_to_hdf5(all_low_dim_data, proprio_stats_dir+"proprio_stats.hdf5")
            import h5py
            extrinsic_hdf5_path = proprio_stats_dir + "proprio_stats.hdf5"

            with h5py.File(extrinsic_hdf5_path, "a") as f:
                group = f.require_group("camera_extrinsic_params")
                for cam_key, extrinsics in [
                    ("head_camera", head_extrinsics),
                    ("left_hand_camera", left_extrinsics),
                    ("right_hand_camera", right_extrinsics)
                ]:
                    if extrinsics:
                        rot = np.array([x["rotation_matrix"] for x in extrinsics], dtype=np.float32)
                        trans = np.array([x["translation_vector"] for x in extrinsics], dtype=np.float32)
                        # 修复：将秒级时间戳转换为纳秒级
                        ts_seconds = np.array([x["timestamp"] for x in extrinsics], dtype=np.float64)
                        ts_nanoseconds = (ts_seconds * 1e9).astype(np.int64)
                        cam_group = group.require_group(cam_key)
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
                if raw_config.enhance_enabled:
                    print(f"[INFO] 以下相机为压缩深度，将使用增强处理输出16位视频: {list(compressed_group.keys())}")
                    # 使用新的增强处理函数，传入彩色图像数据
                    save_depth_videos_enhanced_parallel(compressed_group, imgs_per_cam, output_dir=depth_dir, raw_config=raw_config)
            if uncompressed_group:
                print(f"[INFO] 以下相机为未压缩深度，将用 save_depth_videos_16U_parallel: {list(uncompressed_group.keys())}")
                save_depth_videos_16U_parallel(uncompressed_group, output_dir=depth_dir, raw_config=raw_config)
            save_camera_info_to_json_new(info_per_cam, distortion_model, output_dir=parameters_dir)
            save_color_videos_parallel(imgs_per_cam, output_dir=video_dir, raw_config=raw_config)
            cameras = ['head_cam_h', 'wrist_cam_r', 'wrist_cam_l']
            save_camera_extrinsic_params(cameras=cameras,output_dir=parameters_dir)


            # 新增：数据一致性验证
            temp_uuid_path = os.path.join(output_dir, uuid)
            if os.path.exists(temp_uuid_path):
                print(f"开始验证 episode {uuid} 的数据一致性...")
                validation_result = validate_episode_data_consistency(temp_uuid_path)
                
                if validation_result is None:
                    raise Exception(f"Episode {uuid} 数据一致性验证无法完成")
                
                if not validation_result['is_consistent']:
                    inconsistencies = validation_result.get('inconsistencies', [])
                    error_details = []
                    for inc in inconsistencies:
                        error_details.append(f"{inc['type']} {inc['camera']}: 期望{inc['expected']}帧, 实际{inc['actual']}帧, 差异{inc['difference']:+d}帧")
                    
                    error_msg = f"Episode {uuid} 数据一致性验证失败: {'; '.join(error_details)}"
                    raise Exception(error_msg)
                
                print(f"Episode {uuid} 数据一致性验证通过 ✓")

            else:
                raise Exception(f"Episode {uuid} 临时路径不存在: {temp_uuid_path}")
            
def port_kuavo_rosbag(
    raw_config: Config,
    bag_dir:str,
    moment_json_dir:str,
    metadata_json_dir:str,
    output_dir :str,
    scene:str,
    sub_scene:str,
    continuous_action:str,
    mode:str
):

    # 测试代码：直接指定bagfiles 
    bag_files = list_bag_files_auto(bag_dir)

    generate_dataset_file(
        bag_files = bag_files,
        raw_config=raw_config,
        moment_json_dir=moment_json_dir,
        metadata_json_dir=metadata_json_dir,
        output_dir=output_dir,
        scene= scene,
        sub_scene=sub_scene,
        continuous_action=continuous_action,
        mode=mode
    )
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Kuavo ROSbag to HDF5 Converter")
    parser.add_argument("--bag_dir", default = "物料分拣_20250627_132152_0.bag/", type=str, required=False, help="Path to ROS bag")
    #parser.add_argument("--bag_dir", default = "./testbag/task24_519_20250519_193043_0.bag", type=str, required=False, help="Path to ROS bag")
    parser.add_argument("--moment_json_dir",  type=str, required=False, help="Path to moment.json")
    parser.add_argument("--metadata_json_dir", type=str, required=False, help="Path to metadata.json")
    parser.add_argument("--output_dir", default = "testoutput/", type=str, required=False, help="Path to output")
    parser.add_argument("--scene", default = "test_scene", type=str, required=False, help="scene")
    parser.add_argument("--sub_scene", default = "test_sub_scene", type=str, required=False, help="sub_scene")
    parser.add_argument("--continuous_action", default = "test_continuous_action", type=str, required=False, help="continuous_action")
    parser.add_argument("--mode", default = "simplified", type=str, required=False, help="file structure mode, either 'complete' or 'simplified'. Default is 'simplified'.")
    #parser.add_argument("--use_current_topic", default = False, type=bool, required=False, help="Choose the topic name of joint current state. Old bags use 'joint_current', new bags use 'joint_torque'. Default is False, which means using 'joint_torque'.")
    parser.add_argument(
        '-v', '--process_ID',
        default = 'v0',
        type = str,
        help="process ID"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='request.json',
        help="Path to config YAML file"
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
        output_DIR  = args.output_dir
    if args.scene is not None:
        scene  = args.scene
    if args.sub_scene is not None:
        sub_scene  = args.sub_scene
    if args.continuous_action is not None:
        continuous_action  = args.continuous_action
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


    port_kuavo_rosbag(config,bag_DIR,moment_json_DIR,metadata_json_DIR,output_DIR,scene ,sub_scene,continuous_action,mode)
