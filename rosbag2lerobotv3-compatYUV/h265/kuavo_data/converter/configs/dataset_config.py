from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from omegaconf import OmegaConf


DEFAULT_CAMERA_TOPIC_SPECS: Dict[str, Dict[str, Any]] = {
    "head_cam_h": {
        "base_topic": "/head_cam_h/image_raw",
        "color_topic": "/cam_h/color/image_raw/compressed",
        "color_topic_candidates": [
            "/cam_h/color/image_raw/compressed",
            "/cam_h/color/h265_stream",
        ],
        "camera_info_topic": "/cam_h/color/camera_info",
        "depth_topic": "/cam_h/depth/image_raw/compressedDepth",
        "depth_topic_candidates": [
            "/cam_h/depth/image_raw/compressedDepth",
            "/cam_h/depth/h265_stream",
        ],
    },
    "wrist_cam_l": {
        "base_topic": "/wrist_cam_l/image_raw",
        "color_topic": "/cam_l/color/image_raw/compressed",
        "color_topic_candidates": [
            "/cam_l/color/image_raw/compressed",
            "/cam_l/color/h265_stream",
        ],
        "camera_info_topic": "/cam_l/color/camera_info",
        "depth_topic": "/cam_l/depth/image_rect_raw/compressedDepth",
        "depth_topic_candidates": [
            "/cam_l/depth/image_rect_raw/compressedDepth",
            "/cam_l/depth/h265_stream",
        ],
    },
    "wrist_cam_r": {
        "base_topic": "/wrist_cam_r/image_raw",
        "color_topic": "/cam_r/color/image_raw/compressed",
        "color_topic_candidates": [
            "/cam_r/color/image_raw/compressed",
            "/cam_r/color/h265_stream",
        ],
        "camera_info_topic": "/cam_r/color/camera_info",
        "depth_topic": "/cam_r/depth/image_rect_raw/compressedDepth",
        "depth_topic_candidates": [
            "/cam_r/depth/image_rect_raw/compressedDepth",
            "/cam_r/depth/h265_stream",
        ],
    },
    "depth_h": {
        "depth_topic": "/cam_h/depth/image_raw/compressedDepth",
        "depth_topic_candidates": [
            "/cam_h/depth/image_raw/compressedDepth",
            "/cam_h/depth/h265_stream",
        ],
    },
    "depth_l": {
        "depth_topic": "/cam_l/depth/image_rect_raw/compressedDepth",
        "depth_topic_candidates": [
            "/cam_l/depth/image_rect_raw/compressedDepth",
            "/cam_l/depth/h265_stream",
        ],
    },
    "depth_r": {
        "depth_topic": "/cam_r/depth/image_rect_raw/compressedDepth",
        "depth_topic_candidates": [
            "/cam_r/depth/image_rect_raw/compressedDepth",
            "/cam_r/depth/h265_stream",
        ],
    },
}

DEFAULT_SOURCE_TOPICS: Dict[str, Any] = {
    "sensors_data_raw": "/sensors_data_raw",
    "arm_traj": "/kuavo_arm_traj",
    "arm_traj_alt": "/kuavo_arm_traj_synced",
    "joint_cmd": "/joint_cmd",
    "cmd_pos_world": "/cmd_pose_world_synced",
    "hand_cmd": "/control_robot_hand_position",
    "hand_state_candidates": [
        "/control_robot_hand_position_state",
        "/dexhand/state",
    ],
    "leju_claw_state": "/leju_claw_state",
    "leju_claw_command": "/leju_claw_command",
    "rq2f85_state": "/gripper/state",
    "rq2f85_command": "/gripper/command",
}


@dataclass
class ResizeConfig:
    width: int
    height: int


@dataclass
class Config:
    only_arm: bool
    eef_type: str
    which_arm: str
    use_depth: bool
    depth_range: tuple[int, int]
    dex_dof_needed: int
    train_hz: int
    main_timeline: str
    main_timeline_fps: int
    sample_drop: int
    is_binary: bool
    delta_action: bool
    relative_start: bool
    resize: ResizeConfig
    task_description: str = "Pick and Place Task"
    camera_topic_specs: Dict[str, Dict[str, Any]] = None
    source_topics: Dict[str, Any] = None

    @property
    def use_leju_claw(self) -> bool:
        return "claw" in self.eef_type or self.eef_type == "rq2f85"

    @property
    def use_qiangnao(self) -> bool:
        return self.eef_type == "qiangnao"

    @property
    def only_half_up_body(self) -> bool:
        return self.only_arm

    @property
    def default_camera_names(self) -> List[str]:
        cameras = [
            {"left": ["head_cam_h", "wrist_cam_l"], "right": ["head_cam_h", "wrist_cam_r"], "both": ["head_cam_h", "wrist_cam_l", "wrist_cam_r"]},
            {"left": ["head_cam_h", "depth_h", "wrist_cam_l", "depth_l"], "right": ["head_cam_h", "depth_h", "wrist_cam_r", "depth_r"], "both": ["head_cam_h", "depth_h", "wrist_cam_l", "depth_l", "wrist_cam_r", "depth_r"]},
        ][int(self.use_depth)][self.which_arm]
        return cameras

    @property
    def default_cameras2topics(self) -> dict:
        specs = self.camera_topic_specs or DEFAULT_CAMERA_TOPIC_SPECS
        return {cam: specs.get(cam, {}).get("base_topic", "") for cam in self.default_camera_names}

    @property
    def hand_state_topics(self) -> List[str]:
        topics = (self.source_topics or DEFAULT_SOURCE_TOPICS).get("hand_state_candidates", []) or []
        return [v for v in topics if isinstance(v, str) and v]

    @property
    def slice_robot(self) -> List[Tuple[int, int]]:
        if self.which_arm == 'left':
            return [(12, 19), (19, 19)]
        if self.which_arm == 'right':
            return [(12, 12), (19, 26)]
        if self.which_arm == 'both':
            return [(12, 19), (19, 26)]
        raise ValueError(f"Invalid which_arm: {self.which_arm}")

    @property
    def dex_slice(self) -> List[List[int]]:
        if self.which_arm == 'left':
            return [[0, self.dex_dof_needed], [6, 6]]
        if self.which_arm == 'right':
            return [[0, 0], [6, 6 + self.dex_dof_needed]]
        if self.which_arm == 'both':
            return [[0, self.dex_dof_needed], [6, 6 + self.dex_dof_needed]]
        raise ValueError(f"Invalid which_arm: {self.which_arm}")

    @property
    def claw_slice(self) -> List[List[int]]:
        if self.which_arm == 'left':
            return [[0, 1], [1, 1]]
        if self.which_arm == 'right':
            return [[0, 0], [1, 2]]
        if self.which_arm == 'both':
            return [[0, 1], [1, 2]]
        raise ValueError(f"Invalid which_arm: {self.which_arm}")


def load_config(cfg) -> Config:
    eef_type = OmegaConf.select(cfg, "dataset.eef_type")
    if eef_type not in ['qiangnao', 'leju_claw', 'rq2f85']:
        raise ValueError(f"Invalid eef_type: {eef_type}, must be 'qiangnao' or 'leju_claw','rq2f85' .")

    which_arm = OmegaConf.select(cfg, 'dataset.which_arm')
    if which_arm not in ['left', 'right', 'both']:
        raise ValueError(f"Invalid which_arm: {which_arm}, must be 'left', 'right', or 'both'")

    resize_config = ResizeConfig(
        width=cfg.dataset.resize.width,
        height=cfg.dataset.resize.height,
    )

    camera_topic_specs = dict(DEFAULT_CAMERA_TOPIC_SPECS)
    camera_topic_specs.update(OmegaConf.select(cfg, 'dataset.camera_topic_specs', default={}) or {})

    source_topics = dict(DEFAULT_SOURCE_TOPICS)
    source_topics.update(OmegaConf.select(cfg, 'dataset.source_topics', default={}) or {})

    return Config(
        only_arm=OmegaConf.select(cfg, "dataset.only_arm", default=True),
        eef_type=eef_type,
        which_arm=which_arm,
        use_depth=OmegaConf.select(cfg, 'dataset.use_depth', default=False),
        depth_range=OmegaConf.select(cfg, 'dataset.depth_range', default=(0, 1000)),
        dex_dof_needed=OmegaConf.select(cfg, 'dataset.dex_dof_needed', default=1),
        train_hz=OmegaConf.select(cfg, 'dataset.train_hz', default=10),
        main_timeline=OmegaConf.select(cfg, 'dataset.main_timeline', default='head_cam_h'),
        main_timeline_fps=OmegaConf.select(cfg, 'dataset.main_timeline_fps', default=30),
        sample_drop=OmegaConf.select(cfg, 'dataset.sample_drop', default=0),
        is_binary=OmegaConf.select(cfg, 'dataset.is_binary', default=False),
        delta_action=OmegaConf.select(cfg, 'dataset.delta_action', default=False),
        relative_start=OmegaConf.select(cfg, 'dataset.relative_start', default=False),
        resize=resize_config,
        task_description=OmegaConf.select(cfg, 'dataset.task_description', default="Pick and Place Task"),
        camera_topic_specs=camera_topic_specs,
        source_topics=source_topics,
    )
