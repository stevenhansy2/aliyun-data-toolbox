from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os


@dataclass
class ResizeConfig:
    width: int
    height: int


class KSlink(str):
    """与 str 完全相同的字符串类，但类型名为 KSlink"""

    pass


DEFAULT_CAMERA_TOPIC_SPECS: Dict[str, Dict[str, str]] = {
    "camera_top": {
        "base_topic": "/head_cam_h/image_raw",
        "color_topic": "/cam_h/color/image_raw/compressed",
        "camera_info_topic": "/cam_h/color/camera_info",
        "depth_uncompressed_topic": "/cam_h/depth/image_raw/compressedDepth",
        "depth_compressed_topic": "/cam_h/depth/image_raw/compressed",
    },
    "camera_wrist_left": {
        "base_topic": "/wrist_cam_l/image_raw",
        "color_topic": "/cam_l/color/image_raw/compressed",
        "camera_info_topic": "/cam_l/color/camera_info",
        "depth_uncompressed_topic": "/cam_l/depth/image_rect_raw/compressedDepth",
        "depth_compressed_topic": "/cam_l/depth/image_rect_raw/compressed",
    },
    "camera_wrist_right": {
        "base_topic": "/wrist_cam_r/image_raw",
        "color_topic": "/cam_r/color/image_raw/compressed",
        "camera_info_topic": "/cam_r/color/camera_info",
        "depth_uncompressed_topic": "/cam_r/depth/image_rect_raw/compressedDepth",
        "depth_compressed_topic": "/cam_r/depth/image_rect_raw/compressed",
    },
}

DEFAULT_SOURCE_TOPICS: Dict[str, Any] = {
    "sensors_data_raw": "/sensors_data_raw",
    "arm_traj": "/kuavo_arm_traj",
    "joint_cmd": "/joint_cmd",
    "hand_cmd": "/control_robot_hand_position",
    "hand_state_candidates": [
        "/control_robot_hand_position_state",
        "/dexhand/state",
    ],
    "cb_left_hand_state": "/cb_left_hand_state",
    "cb_right_hand_state": "/cb_right_hand_state",
    "cb_left_hand_control_cmd": "/cb_left_hand_control_cmd",
    "cb_right_hand_control_cmd": "/cb_right_hand_control_cmd",
    "force6d_left_hand_force_torque": "/force6d_left_hand_force_torque",
    "force6d_right_hand_force_torque": "/force6d_right_hand_force_torque",
    "cb_left_hand_matrix_touch_pc2": "/cb_left_hand_matrix_touch_pc2",
    "cb_right_hand_matrix_touch_pc2": "/cb_right_hand_matrix_touch_pc2",
    "leju_claw_state": "/leju_claw_state",
    "leju_claw_command": "/leju_claw_command",
}


def _build_topics_default(
    camera_topic_specs: Dict[str, Dict[str, str]], source_topics: Dict[str, Any]
) -> List[str]:
    topics: List[str] = []
    for key in [
        "arm_traj",
        "sensors_data_raw",
        "joint_cmd",
        "hand_cmd",
        "leju_claw_command",
        "leju_claw_state",
        "cb_left_hand_state",
        "cb_right_hand_state",
        "cb_left_hand_control_cmd",
        "cb_right_hand_control_cmd",
        "force6d_left_hand_force_torque",
        "force6d_right_hand_force_torque",
        "cb_left_hand_matrix_touch_pc2",
        "cb_right_hand_matrix_touch_pc2",
    ]:
        v = source_topics.get(key)
        if isinstance(v, str) and v:
            topics.append(v)
    for v in source_topics.get("hand_state_candidates", []) or []:
        if isinstance(v, str) and v:
            topics.append(v)

    for spec in camera_topic_specs.values():
        for key in [
            "color_topic",
            "depth_uncompressed_topic",
            "depth_compressed_topic",
            "camera_info_topic",
        ]:
            topic = spec.get(key)
            if isinstance(topic, str) and topic:
                topics.append(topic)
        # historical trailing slash support
        ci = spec.get("camera_info_topic")
        if isinstance(ci, str) and ci and not ci.endswith("/"):
            topics.append(ci + "/")

    # preserve order, remove duplicates
    return list(dict.fromkeys(topics))


@dataclass
class Config:
    # Basic settings
    bags: List[dict]  # 输入的ROS bag文件列表（dict格式）
    output_dir: str  # 输出目录路径
    topics: List[str]  # 需要处理的主题列表
    raw_dir: str  # 原始数据目录路径
    id: str  # 数据版本
    only_arm: bool
    eef_type: str  # 'dex_hand' or 'leju_claw'
    which_arm: str  # 'left', 'right', or 'both'
    dex_dof_needed: int  # 通常为1，表示只需要第一个关节作为开合依据
    upload_dir: str  # 上传目录路径

    # 新增场景与任务相关字段

    scene_name: str = ""
    sub_scene_name: str = ""
    init_scene_text: str = ""
    english_init_scene_text: str = ""
    task_name: str = ""
    english_task_name: str = ""
    data_type: str = ""
    episode_status: str = ""
    data_gen_mode: str = ""

    # Timeline settings
    train_hz: int = 30
    main_timeline: str = "/cam_h/color/image_raw/compressed"
    main_timeline_fps: int = 30
    sample_drop: int = 10

    # Processing flags
    is_binary: bool = False
    delta_action: bool = False
    relative_start: bool = False
    enhance_enabled: bool = True
    denoise_enabled: bool = True
    separate_video_storage: bool = True
    separate_hand_fields: bool = False
    merge_hand_position: bool = False
    video_output_dir: str = "videos"
    async_video_encoding: bool = True
    use_pipeline_encoding: bool = False  # 启用流水线编码（批处理与视频编码并行）
    use_streaming_video: bool = False  # 启用流式视频编码（无需临时文件）
    video_queue_limit: int = 0  # 流式编码队列上限，0 表示按核数自动计算
    schedule_cores: int = 0  # 调度档位，0 表示自动（可被 KUAVO_SCHED_CORES 覆盖）
    video_process_timeout_sec: int = 600  # 单个视频编码子进程超时秒数
    use_parallel_rosbag_read: bool = True  # 启用并行 ROSbag 读取，默认自动开启
    parallel_rosbag_workers: int = 0  # 并行读取的 worker 数量，0 表示按核数自动计算
    on_existing_batch: str = "overwrite"  # 已存在 batch 目录处理策略: error|skip|overwrite

    # Image resize settings
    resize: ResizeConfig = None
    slave_bag_savedir: str = None  # 保存转换后的ROS bag文件的目录
    export_type: str = "lerobot"  # 导出类型，默认为pkl
    hdf5_export_mask: bool = False  # 是否导出HDF5格式的mask图片
    robot_model: str = "kuavo_s45"
    urdf_path: str = "./kuavo/assets/urdf/biped_s45.urdf"
    main_timeline_key: str = "camera_top"
    camera_topic_specs: Dict[str, Dict[str, str]] = None
    source_topics: Dict[str, Any] = None
    bag_overrides: List[Dict[str, Any]] = None
    model_profiles: Dict[str, Dict[str, Any]] = None

    @property
    def use_leju_claw(self) -> bool:
        """Determine if using leju claw based on eef_type."""
        return self.eef_type == "leju_claw"

    @property
    def use_qiangnao(self) -> bool:
        """Determine if using qiangnao based on eef_type."""
        return self.eef_type == "dex_hand"

    @property
    def only_half_up_body(self) -> bool:
        """Always true when only using arm."""
        return True

    @property
    def default_camera_names(self) -> List[str]:
        """Get camera names based on which arm is being used."""
        camera_specs = self.camera_topic_specs or DEFAULT_CAMERA_TOPIC_SPECS
        if self.which_arm == "left":
            keys = ["camera_top", "camera_wrist_left"]
        elif self.which_arm == "right":
            keys = ["camera_top", "camera_wrist_right"]
        elif self.which_arm == "both":
            keys = ["camera_top", "camera_wrist_right", "camera_wrist_left"]
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
        return [k for k in keys if k in camera_specs]

    @property
    def default_cameras2topics(self) -> dict:
        """Get camera to topic mapping based on which arm is being used."""
        camera_specs = self.camera_topic_specs or DEFAULT_CAMERA_TOPIC_SPECS
        out = {}
        for cam in self.default_camera_names:
            spec = camera_specs.get(cam, {})
            out[cam] = spec.get("base_topic", "")
        return out

    @property
    def hand_state_topics(self) -> List[str]:
        source_topics = self.source_topics or DEFAULT_SOURCE_TOPICS
        values = source_topics.get("hand_state_candidates", []) or []
        return [v for v in values if isinstance(v, str) and v]

    @property
    def slice_robot(self) -> List[Tuple[int, int]]:
        """Get robot slice based on which arm is being used."""
        if self.which_arm == "left":
            return [(12, 19), (19, 19)]
        elif self.which_arm == "right":
            return [(12, 12), (19, 26)]
        elif self.which_arm == "both":
            return [(12, 19), (19, 26)]
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")

    @property
    def dex_slice(self) -> List[List[int]]:
        """Get dex slice based on which arm and dex_dof_needed."""
        if self.which_arm == "left":
            return [[0, self.dex_dof_needed], [6, 6]]  # 左手使用指定自由度，右手不使用
        elif self.which_arm == "right":
            return [
                [0, 0],
                [6, 6 + self.dex_dof_needed],
            ]  # 左手不使用，右手使用指定自由度
        elif self.which_arm == "both":
            return [
                [0, self.dex_dof_needed],
                [6, 6 + self.dex_dof_needed],
            ]  # 双手都使用指定自由度
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")

    @property
    def claw_slice(self) -> List[List[int]]:
        """Get claw slice based on which arm."""
        if self.which_arm == "left":
            return [[0, 1], [1, 1]]  # 左手使用夹爪，右手不使用
        elif self.which_arm == "right":
            return [[0, 0], [1, 2]]  # 左手不使用，右手使用夹爪
        elif self.which_arm == "both":
            return [[0, 1], [1, 2]]  # 双手都使用夹爪
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")


def load_config_from_json(config_path: str) -> Config:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file (required)

    Returns:
        Config object containing all settings

    Raises:
        FileNotFoundError: If config file does not exist
    """
    import json

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Validate eef_type
    eef_type = config_dict.get("eef_type", "dex_hand")
    if eef_type not in ["dex_hand", "leju_claw"]:
        raise ValueError(
            f"Invalid eef_type: {eef_type}, must be 'dex_hand' or 'leju_claw'"
        )

    # Validate which_arm
    which_arm = config_dict.get("which_arm", "both")
    if which_arm not in ["left", "right", "both"]:
        raise ValueError(
            f"Invalid which_arm: {which_arm}, must be 'left', 'right', or 'both'"
        )

    on_existing_batch = config_dict.get("on_existing_batch", "overwrite")
    if on_existing_batch not in ["error", "skip", "overwrite"]:
        raise ValueError(
            "Invalid on_existing_batch: "
            f"{on_existing_batch}, must be 'error', 'skip', or 'overwrite'"
        )

    # Create ResizeConfig object
    resize_config = ResizeConfig(
        width=config_dict.get("img_resize", {}).get("width", 848),
        height=config_dict.get("img_resize", {}).get("height", 480),
    )

    robot_profile = config_dict.get("robot_profile", {}) or {}
    profile_topics = robot_profile.get("topics", {}) or {}
    profile_cameras = robot_profile.get("cameras", {}) or {}
    profile_model = robot_profile.get("model", "kuavo_s45")
    profile_urdf = robot_profile.get(
        "urdf_path", "./kuavo/assets/urdf/biped_s45.urdf"
    )

    source_topics = dict(DEFAULT_SOURCE_TOPICS)
    source_topics.update(profile_topics)
    source_topics.update(config_dict.get("source_topics", {}) or {})

    camera_topic_specs = dict(DEFAULT_CAMERA_TOPIC_SPECS)
    camera_topic_specs.update(profile_cameras)
    camera_topic_specs.update(config_dict.get("camera_topic_specs", {}) or {})

    topics_default = _build_topics_default(camera_topic_specs, source_topics)

    def get_time_range_from_steps(steps):
        """从steps中获取时间范围"""
        if not steps or len(steps) == 0:
            return None, None  # 返回None表示没有从steps中获取到时间范围

        # 按步骤的start时间排序
        sorted_steps = sorted(steps, key=lambda x: x.get("start", 0))

        first_step_start = sorted_steps[0].get("start", None)
        last_step_end = sorted_steps[-1].get("end", None)

        return first_step_start, last_step_end

    # 解析 bags 字段为 List[dict]，支持每个bag独立场景/任务字段，steps支持Detailed_description等新字段
    bags = []
    for bag in config_dict.get("bags", []):
        bag_dict = dict(bag)

        # 解析steps，支持Detailed_description等新字段
        steps = []
        for step in bag_dict.get("steps") or []:
            step_dict = dict(step)
            steps.append(
                {
                    "index": step_dict.get("index", "default_index"),
                    "mask_link": step_dict.get("mask_link", ""),
                    "start": step_dict.get("start", ""),
                    "end": step_dict.get("end", ""),
                    "skill": step_dict.get("skill", "default_skill"),
                    "action_text": step_dict.get("action_text", "default_action_text"),
                    "english_action_text": step_dict.get(
                        "english_action_text", "default_english_action_text"
                    ),
                    "detailed_description": step_dict.get(
                        "detailed_description", "default_detailed_description"
                    ),
                }
            )
        bag_dict["steps"] = steps if steps else None

        # 新增：处理start和end的优先级逻辑
        steps_start, steps_end = get_time_range_from_steps(steps)

        # 优先使用steps中的时间范围，如果没有则使用bag自带的start/end
        if steps_start is not None and steps_end is not None:
            bag_dict["start"] = steps_start
            bag_dict["end"] = steps_end
            print(
                f"[CONFIG] Bag使用steps时间范围: start={steps_start}, end={steps_end} (从 {len(steps)} 个步骤中获取)"
            )
        else:
            # 保持原有的start/end，如果没有则使用默认值
            original_start = bag_dict.get("start", 0)
            original_end = bag_dict.get("end", 1)
            bag_dict["start"] = original_start
            bag_dict["end"] = original_end
            if steps:
                print(
                    f"[CONFIG] Bag的steps中未找到有效时间范围，使用原始时间范围: start={original_start}, end={original_end}"
                )
            else:
                print(
                    f"[CONFIG] Bag无steps，使用原始时间范围: start={original_start}, end={original_end}"
                )

        # 支持每个bag独立的场景/任务等字段
        for field in [
            "scene_name",
            "sub_scene_name",
            "init_scene_text",
            "english_init_scene_text",
            "task_name",
            "english_task_name",
            "data_type",
            "episode_status",
            "data_gen_mode",
            "sn_code",
            "sn_name",
            "task_code",
            "bag_id",
            "length",  # <-- 新增'task_code'
        ]:
            if field in bag_dict:
                continue
            if field in config_dict:
                bag_dict[field] = config_dict[field]
        bags.append(bag_dict)

    topics = config_dict.get("topics", topics_default)
    if topics is None or (isinstance(topics, list) and len(topics) == 0):
        topics = topics_default

    main_timeline_conf = config_dict.get("main_timeline")
    main_timeline_topic = (
        main_timeline_conf.get("topic", "/cam_h/color/image_raw/compressed")
        if isinstance(main_timeline_conf, dict)
        else config_dict.get("main_timeline", "/cam_h/color/image_raw/compressed")
    )
    main_timeline_fps = (
        main_timeline_conf.get("frequency", 30)
        if isinstance(main_timeline_conf, dict)
        else config_dict.get("main_timeline_fps", 30)
    )
    main_timeline_key = "camera_top"
    if isinstance(main_timeline_conf, dict):
        main_timeline_key = main_timeline_conf.get("camera_key", "camera_top")
    for cam_name, spec in camera_topic_specs.items():
        if spec.get("color_topic") == main_timeline_topic:
            main_timeline_key = cam_name
            break

    # Create main Config object
    return Config(
        output_dir=config_dict.get("output_dir", "/home/c/data/tmp/ks_download"),
        id=config_dict.get("id", "default_id"),
        train_hz=config_dict.get("train_frequency", 30),
        only_arm=config_dict.get("only_arm", False),
        eef_type=eef_type,
        which_arm=which_arm,
        dex_dof_needed=config_dict.get("dex_dof_needed", 1),
        resize=resize_config,
        bags=bags,
        sample_drop=config_dict.get("sample_drop", 10),
        is_binary=config_dict.get("is_binary", False),
        delta_action=config_dict.get("is_delta_action", False),
        relative_start=config_dict.get("is_relative_start", False),
        enhance_enabled=config_dict.get("enhance_enabled", True),
        raw_dir=config_dict.get("raw_dir", None),
        upload_dir=config_dict.get("upload_dir", None),
        topics=topics,
        # 新增场景与任务相关字段
        scene_name=config_dict.get("scene_name", "default_scene_name"),
        sub_scene_name=config_dict.get("sub_scene_name", "default_sub_scene_name"),
        init_scene_text=config_dict.get("init_scene_text", "default_init_scene_text"),
        english_init_scene_text=config_dict.get(
            "english_init_scene_text", "default_english_init_scene_text"
        ),
        task_name=config_dict.get("task_name", "default_task_name"),
        english_task_name=config_dict.get(
            "english_task_name", "default_english_task_name"
        ),
        data_type=config_dict.get("data_type", "常规"),
        episode_status=config_dict.get("episode_status", "approved"),
        data_gen_mode=config_dict.get("data_gen_mode", "real_machine"),
        slave_bag_savedir=config_dict.get("slave_bag_savedir"),
        export_type=config_dict.get("export_type", "lerobot"),  # 默认为"lerobot"
        hdf5_export_mask=config_dict.get(
            "hdf5_export_mask", False
        ),  # 是否导出HDF5格式的mask图片
        main_timeline=main_timeline_topic,
        main_timeline_fps=main_timeline_fps,
        separate_hand_fields=config_dict.get("separate_hand_fields", False),
        merge_hand_position=config_dict.get("merge_hand_position", False),
        use_pipeline_encoding=config_dict.get("use_pipeline_encoding", False),
        use_streaming_video=config_dict.get("use_streaming_video", False),
        video_queue_limit=config_dict.get("video_queue_limit", 0),
        schedule_cores=config_dict.get("schedule_cores", 0),
        video_process_timeout_sec=config_dict.get("video_process_timeout_sec", 600),
        use_parallel_rosbag_read=config_dict.get("use_parallel_rosbag_read", True),
        parallel_rosbag_workers=config_dict.get("parallel_rosbag_workers", 0),
        on_existing_batch=on_existing_batch,
        robot_model=config_dict.get("robot_model", profile_model),
        urdf_path=config_dict.get("urdf_path", profile_urdf),
        main_timeline_key=main_timeline_key,
        camera_topic_specs=camera_topic_specs,
        source_topics=source_topics,
        bag_overrides=config_dict.get("bag_overrides", []) or [],
        model_profiles=config_dict.get("model_profiles", {}) or {},
    )


if __name__ == "__main__":
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "request_new.json")
    try:
        config = load_config_from_json(config_path)
        print(config)
    except Exception as e:
        print(f"Error loading config: {e}")
