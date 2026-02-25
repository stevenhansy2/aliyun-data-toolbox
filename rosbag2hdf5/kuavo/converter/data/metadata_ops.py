import dataclasses
import datetime
import json
import os
import uuid
from collections import OrderedDict

import numpy as np
import rosbag


def _to_float_seconds(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Invalid duration value: {value}") from exc


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
        log_print(f"获取bag时间信息失败: {e}")
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

    log_print(f"已创建目录结构：{base_path}")
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

    # 验证 metadata.json 关键字段是否存在且非空
    required_metadata_fields = {
        "sceneCode": "场景编码",
        "subSceneCode": "子场景编码",
        "initSceneText": "子场景中文描述",
        "englishInitSceneText": "子场景英文描述",
        "deviceSn": "设备序列号",
    }

    for field, desc in required_metadata_fields.items():
        value = raw_metadata.get(field)
        if not value or (isinstance(value, str) and value.strip() == ""):
            error_msg = f"❌ metadata.json 缺失关键字段 '{field}' ({desc})，数据不完整，终止处理"
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

    # 验证 task 相关字段至少有一个有效
    task_group_name = raw_metadata.get("taskGroupName", "").strip()
    task_name = raw_metadata.get("taskName", "").strip()
    task_group_code = raw_metadata.get("taskGroupCode", "").strip()
    task_code = raw_metadata.get("taskCode", "").strip()

    if not (task_group_name or task_name):
        error_msg = f"❌ metadata.json 缺失任务名称字段 'task_group_name' 和 'task_name' 都为空，数据不完整，终止处理"
        log_print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    if not (task_group_code or task_code):
        error_msg = f"❌ metadata.json 缺失任务编码字段 'task_group_code' 和 'task_code' 都为空，数据不完整，终止处理"
        log_print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    # 验证 moment.json 数据有效性
    moments = raw_metadata.get("marks", [])
    if not moments:
        error_msg = (
            f"❌ moment.json 中未找到有效的 moments 数据，标注信息缺失，终止处理"
        )
        log_print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    # 验证每个 moment 的关键字段
    for i, m in enumerate(moments):
        custom_fields = m
        trigger_time = m.get("markStart", "").strip()

        # 验证时间戳
        if not trigger_time:
            error_msg = f"❌ moment.json 第{i+1}个动作缺失 'markStart' 时间戳，标注信息不完整，终止处理"
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # 验证动作描述字段
        required_moment_fields = {
            "skillAtomic": "技能原子英文名",
            "skillDetail": "技能详细描述",
            "enSkillDetail": "技能英文详细描述",
        }

        for field, desc in required_moment_fields.items():
            value = custom_fields.get(field, "").strip()
            if not value:
                error_msg = f"❌ moment.json 第{i+1}个动作缺失 '{field}' ({desc})，标注信息不完整，终止处理"
                log_print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)

    # 转换新格式 metadata 为旧格式
    converted_metadata = {}

    # scene_name 对应 scene_code
    converted_metadata["scene_name"] = raw_metadata.get("sceneCode")

    # sub_scene_name 对应 sub_scene_code
    converted_metadata["sub_scene_name"] = raw_metadata.get("subSceneCode")

    # init_scene_text 对应 sub_scene_zh_dec
    converted_metadata["init_scene_text"] = raw_metadata.get("initSceneText")

    # english_init_scene_text 对应 sub_scene_en_dec
    converted_metadata["english_init_scene_text"] = raw_metadata.get(
        "englishInitSceneText"
    )

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
    converted_metadata["sn_code"] = raw_metadata.get("deviceSn")

    # sn_name 默认值
    converted_metadata["sn_name"] = "乐聚机器人"

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
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

    log_print(f"Metadata 字段转换结果:")
    for key, value in converted_metadata.items():
        log_print(f"  {key}: '{value}'")

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
        log_print(f"原始时间戳前3个: {main_time_line_timestamps[:3]}")
        log_print(f"原始时间戳后3个: {main_time_line_timestamps[-3:]}")

        # 检查时间戳是否已经是秒格式还是纳秒格式
        if main_time_line_timestamps[0] > 1e12:  # 如果大于1e12，认为是纳秒格式
            timestamps_seconds = main_time_line_timestamps / 1e9
            log_print("时间戳格式：纳秒 -> 秒")
        else:
            timestamps_seconds = main_time_line_timestamps
            log_print("时间戳格式：已经是秒")

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
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if rosbag_actual_end_time < year_2025_timestamp:
            end_datetime = datetime.datetime.fromtimestamp(rosbag_actual_end_time)
            error_msg = f"❌ 数据结束时间戳异常: {rosbag_actual_end_time} ({end_datetime.isoformat()})，早于2025年，数据可能损坏"
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # 新增：大于等于 2040 年也视为异常
        if rosbag_actual_start_time >= year_2040_timestamp:
            start_datetime = datetime.datetime.fromtimestamp(rosbag_actual_start_time)
            error_msg = f"❌ 数据开始时间戳异常: {rosbag_actual_start_time} ({start_datetime.isoformat()})，晚于2040年，数据可能损坏"
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if rosbag_actual_end_time >= year_2040_timestamp:
            end_datetime = datetime.datetime.fromtimestamp(rosbag_actual_end_time)
            error_msg = f"❌ 数据结束时间戳异常: {rosbag_actual_end_time} ({end_datetime.isoformat()})，晚于2040年，数据可能损坏"
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # 调试：打印转换后的时间戳
        log_print(f"转换后时间戳前3个: {timestamps_seconds[:3]}")
        log_print(f"转换后时间戳后3个: {timestamps_seconds[-3:]}")

        # 验证时间戳转换
        start_datetime = datetime.datetime.fromtimestamp(
            rosbag_actual_start_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )
        end_datetime = datetime.datetime.fromtimestamp(
            rosbag_actual_end_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )

        log_print(f"实际开始时间验证: {start_datetime.isoformat()}")
        log_print(f"实际结束时间验证: {end_datetime.isoformat()}")

    # 原始bag时间范围
    if bag_time_info:
        rosbag_original_start_time = bag_time_info.get("unix_timestamp")
        rosbag_original_end_time = bag_time_info.get("end_time")

    # 构造 action_config
    log_print(f"时间信息:")
    if rosbag_original_start_time and rosbag_original_end_time:
        log_print(
            f"  原始bag时间: {rosbag_original_start_time:.6f}s - {rosbag_original_end_time:.6f}s"
        )
    if rosbag_actual_start_time and rosbag_actual_end_time:
        log_print(
            f"  实际数据时间: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
        )
    log_print(f"  总帧数: {total_frames}")

    action_config = []

    for m in moments:
        # 从新格式的 customFieldValues 中提取数据
        custom_fields = m

        trigger_time = m.get("markStart", "")
        duration_str = m.get("duration", "0")

        # 格式化时间戳：将 "Z" 替换为 "+00:00"
        formatted_trigger_time = (trigger_time + "+08:00") if trigger_time else ""

        # 添加调试信息
        log_print(f"处理动作数据:")
        log_print(f"  skill_atomic_en: {custom_fields.get('skillAtomic', '')}")
        log_print(f"  skill_detail: {custom_fields.get('skillDetail', '')}")
        log_print(f"  en_skill_detail: {custom_fields.get('enSkillDetail', '')}")
        log_print(f"  原始时间戳: {trigger_time}")
        log_print(f"  格式化时间戳: {formatted_trigger_time}")

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
                    trigger_time + "+08:00"
                )
                action_original_start_time = trigger_datetime.timestamp()

                # 下界校验
                if action_original_start_time < year_2025_timestamp:
                    error_msg = f"❌ 动作时间戳异常: {trigger_time} ({trigger_datetime.isoformat()})，早于2025年，数据可能损坏"
                    log_print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)

                # 新增：上界校验
                if action_original_start_time >= year_2040_timestamp:
                    error_msg = f"❌ 动作时间戳异常: {trigger_time} ({trigger_datetime.isoformat()})，晚于2040年，数据可能损坏"
                    log_print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)

                # 解析持续时间
                action_duration = _to_float_seconds(duration_str)

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
                    error_msg = f"❌ 动作帧数计算失败: 动作'{custom_fields.get('skillDetail', '')}' 时间戳 {trigger_time} 计算出的开始帧={start_frame}, 结束帧={end_frame}"
                    log_print(f"[ERROR] {error_msg}")
                    log_print(f"  调试信息:")
                    log_print(
                        f"    实际数据范围: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
                    )
                    log_print(
                        f"    动作时间范围: {action_original_start_time:.6f}s - {action_original_start_time + action_duration:.6f}s"
                    )
                    log_print("[WARN] 跳过该动作（帧数计算失败）")
                    continue

                log_print(f"动作: {custom_fields.get('skillDetail', '')}")
                log_print(f"  动作时间: {trigger_datetime.isoformat()}")
                log_print(f"  原始开始时间: {action_original_start_time:.6f}s")
                log_print(
                    f"  原始结束时间: {action_original_start_time + action_duration:.6f}s"
                )
                log_print(f"  持续时间: {action_duration:.3f}s")
                log_print(f"  计算得到帧数: {start_frame} - {end_frame}")

                # 验证计算结果
                if start_frame is not None and end_frame is not None:
                    actual_start_time = rosbag_actual_start_time + (
                        start_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    actual_end_time = rosbag_actual_start_time + (
                        end_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    log_print(f"  验证-实际开始时间: {actual_start_time:.6f}s")
                    log_print(f"  验证-实际结束时间: {actual_end_time:.6f}s")

                log_print("-" * 50)

            except Exception as e:
                log_print(f"[ERROR] 计算帧数时出错: {e}")
                import traceback

                traceback.print_exc()
                raise  # 重新抛出异常，终止程序

        # 获取动作字段值并验证不为空
        skill_value = custom_fields.get("skillAtomic", "").strip()
        action_text_value = custom_fields.get("skillDetail", "").strip()
        english_action_text_value = custom_fields.get("enSkillDetail", "").strip()

        # 新增：验证动作配置中的关键字段不能为空
        if not skill_value:
            error_msg = f"❌ 动作配置中 skill 字段为空，动作标注不完整，终止处理"
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if not action_text_value:
            error_msg = f"❌ 动作配置中 action_text 字段为空，动作标注不完整，终止处理"
            log_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        if not english_action_text_value:
            error_msg = (
                f"❌ 动作配置中 english_action_text 字段为空，动作标注不完整，终止处理"
            )
            log_print(f"[ERROR] {error_msg}")
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
    log_print(f"已保存到 {output_path}")


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
