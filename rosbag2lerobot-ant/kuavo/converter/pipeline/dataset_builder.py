"""Dataset schema/building helpers."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Literal

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from converter.configs import Config


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 6
    image_writer_threads: int = 12
    video_backend: str | None = None


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    eef_type: Literal["leju_claw", "dex_hand"] = "dex_hand",
    *,
    has_depth_image: bool = False,
    dataset_config: DatasetConfig | None = None,
    root: str,
    extra_features: bool = True,
    raw_config: Config,
    joint_names_list: list[str],
) -> LeRobotDataset:
    if dataset_config is None:
        dataset_config = DatasetConfig()

    dexhand = [
        "left_linkerhand_1",
        "left_linkerhand_2",
        "left_linkerhand_3",
        "left_linkerhand_4",
        "left_linkerhand_5",
        "left_linkerhand_6",
        "right_linkerhand_1",
        "right_linkerhand_2",
        "right_linkerhand_3",
        "right_linkerhand_4",
        "right_linkerhand_5",
        "right_linkerhand_6",
    ]
    lejuclaw = ["left_claw", "right_claw"]
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

    features = {
        "observation.state.arm.position": {"dtype": "float32", "shape": (14,), "names": arm},
        "observation.state.arm.effort": {"dtype": "float32", "shape": (14,), "names": arm},
        "observation.state.arm.velocity": {"dtype": "float32", "shape": (14,), "names": arm},
        "observation.state.arm.current_value": {"dtype": "float32", "shape": (14,), "names": arm},
        "observation.state.end.position": {"dtype": "float32", "shape": (6,), "names": end_position},
        "observation.state.end.orientation": {
            "dtype": "float32",
            "shape": (8,),
            "names": end_orientation,
        },
        "observation.state.head.effort": {"dtype": "float32", "shape": (2,), "names": head},
        "observation.state.head.position": {"dtype": "float32", "shape": (2,), "names": head},
        "observation.state.head.velocity": {"dtype": "float32", "shape": (2,), "names": head},
        "observation.state.leg.effort": {"dtype": "float32", "shape": (12,), "names": leg},
        "observation.state.leg.position": {"dtype": "float32", "shape": (12,), "names": leg},
        "observation.state.leg.velocity": {"dtype": "float32", "shape": (12,), "names": leg},
        "observation.state.leg.current_value": {"dtype": "float32", "shape": (12,), "names": leg},
        "action.head.position": {"dtype": "float32", "shape": (2,), "names": head},
        "action.arm.position": {"dtype": "float32", "shape": (14,), "names": arm},
        "action.leg.position": {"dtype": "float32", "shape": (12,), "names": leg},
        "imu.acc_xyz": {"dtype": "float32", "shape": (3,), "names": imu_acc},
        "imu.free_acc_xyz": {"dtype": "float32", "shape": (3,), "names": imu_free_acc},
        "imu.gyro_xyz": {"dtype": "float32", "shape": (3,), "names": imu_gyro_acc},
        "imu.quat_xyzw": {"dtype": "float32", "shape": (4,), "names": imu_quat_acc},
    }

    if eef_type == "leju_claw":
        features.update(
            {
                "action.effector.position": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": lejuclaw,
                },
                "observation.state.effector.position": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": lejuclaw,
                },
            }
        )
    elif eef_type == "dex_hand":
        features.update(
            {
                "action.hand_left.position": {"dtype": "float32", "shape": (6,), "names": dexhand[:6]},
                "action.hand_right.position": {"dtype": "float32", "shape": (6,), "names": dexhand[6:]},
                "observation.state.hand_left.position": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": dexhand[:6],
                },
                "observation.state.hand_right.position": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": dexhand[6:],
                },
                "observation.state.hand_left.force_torque": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"],
                },
                "observation.state.hand_right.force_torque": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"],
                },
                "observation.state.hand_left.touch_matrix": {
                    "dtype": "float32",
                    "shape": (360,),
                    "names": None,
                },
                "observation.state.hand_right.touch_matrix": {
                    "dtype": "float32",
                    "shape": (360,),
                    "names": None,
                },
            }
        )

    separate_video_storage = getattr(raw_config, "separate_video_storage", False)
    if not separate_video_storage:
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
            "shape": (len(joint_names_list),),
            "names": joint_names_list,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (len(joint_names_list),),
            "names": joint_names_list,
        }
        print("DEFAULT_JOINT_NAMES_LIST", joint_names_list)

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

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

