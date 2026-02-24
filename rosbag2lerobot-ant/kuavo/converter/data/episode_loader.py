"""Episode loading and hand-data extraction helpers."""

import gc
import json
import os
from pathlib import Path

import numpy as np
import psutil

from converter.config import Config
from converter.reader.kuavo_dataset_slave_s import KuavoRosbagReader
from converter.reader.postprocess_utils import PostProcessorUtils
from converter.slave_utils import flip_camera_arrays_if_needed, swap_left_right_data_if_needed

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
        if "camera_top" in bag_data and len(bag_data["camera_top"]) > 0:
            main_time_line_timestamps = np.array(
                [msg["timestamp"] for msg in bag_data["camera_top"]]
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
        [msg["data"] for msg in bag_data["action.joint_cmd.joint_q"]],
        dtype=np.float32,
    )
    kuavo_arm_traj__position = action_kuavo_arm_traj = np.array(
        [msg["data"] for msg in bag_data["action.kuavo_arm_traj"]],
        dtype=np.float32,
    )

    # 手部数据
    leju_claw_state__position = claw_state = np.array(
        [msg["data"] for msg in bag_data["observation.claw"]],
        dtype=np.float32,
    )
    leju_claw_command__position = claw_action = np.array(
        [msg["data"] for msg in bag_data["action.claw"]],
        dtype=np.float32,
    )

    # control_robot_hand_position_state_both = qiangnao_state = np.array(
    #     [msg["data"] for msg in bag_data["observation.qiangnao"]], dtype=np.float32,
    # )
    # control_robot_hand_position_both = qiangnao_action = np.array(
    #     [msg["data"] for msg in bag_data["action.qiangnao"]], dtype=np.float32,
    # )
    qiangnao_state = None
    try:
        qiangnao_state = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao"]],
            dtype=np.float32,
        )
    except KeyError:
        print("[WARN] 未找到 'observation.qiangnao' 数据")
    qiangnao_action = None
    try:
        qiangnao_action = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao"]],
            dtype=np.float32,
        )
    except KeyError:
        print("[WARN] 未找到 'action.qiangnao' 数据")

    hand_state_left = None
    hand_state_right = None
    hand_action_left = None
    hand_action_right = None

    if "observation.qiangnao_left" in bag_data:
        hand_state_left = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao_left"]],
            dtype=np.float32,
        )
    if "observation.qiangnao_right" in bag_data:
        hand_state_right = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao_right"]],
            dtype=np.float32,
        )
    if "action.qiangnao_left" in bag_data:
        hand_action_left = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao_left"]],
            dtype=np.float32,
        )
    if "action.qiangnao_right" in bag_data:
        hand_action_right = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao_right"]],
            dtype=np.float32,
        )

    if (
        (hand_state_left is None or hand_state_right is None)
        and qiangnao_state is not None
    ):
        split_left, split_right = _split_dexhand_lr(qiangnao_state)
        if split_left is not None:
            hand_state_left = split_left
        if split_right is not None:
            hand_state_right = split_right
    if (
        (hand_action_left is None or hand_action_right is None)
        and qiangnao_action is not None
    ):
        split_left, split_right = _split_dexhand_lr(qiangnao_action)
        if split_left is not None:
            hand_action_left = split_left
        if split_right is not None:
            hand_action_right = split_right

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
        [msg["timestamp"] for msg in bag_data["camera_top"]]
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
        [msg["data"] for msg in bag_data["end.position"]],
        dtype=np.float32,
    )
    end_orientation = np.array(
        [msg["data"] for msg in bag_data["end.orientation"]],
        dtype=np.float32,
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
                "index": main_time_line_timestamps_ns,
            },
            "hand_left": {
                "position": hand_action_left,
                "index": main_time_line_timestamps_ns,
            },
            "hand_right": {
                "position": hand_action_right,
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
            },
            "hand_left": {
                "position": hand_state_left,
            },
            "hand_right": {
                "position": hand_state_right,
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
    cb_left_state = []
    cb_right_state = []
    cb_left_action = []
    cb_right_action = []

    # 话题名根据你的实际定义
    topic_claw_state = "/leju_claw_state"
    topic_claw_action = "/leju_claw_command"
    topic_qiangnao_state = "/control_robot_hand_position_state"
    topic_qiangnao_action = "/control_robot_hand_position"
    topic_cb_left_state = "/cb_left_hand_state"
    topic_cb_right_state = "/cb_right_hand_state"
    topic_cb_left_action = "/cb_left_hand_control_cmd"
    topic_cb_right_action = "/cb_right_hand_control_cmd"

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
            topic_cb_left_state,
            topic_cb_right_state,
            topic_cb_left_action,
            topic_cb_right_action,
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
        elif topic == topic_cb_left_state:
            try:
                cb_left_state.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_cb_right_state:
            try:
                cb_right_state.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_cb_left_action:
            try:
                cb_left_action.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass
        elif topic == topic_cb_right_action:
            try:
                cb_right_action.append(np.array(msg.position, dtype=np.float64))
            except Exception:
                pass

    bag.close()

    claw_state = np.array(claw_state)
    claw_action = np.array(claw_action)
    qiangnao_state = np.array(qiangnao_state)
    qiangnao_action = np.array(qiangnao_action)
    cb_left_state = np.array(cb_left_state)
    cb_right_state = np.array(cb_right_state)
    cb_left_action = np.array(cb_left_action)
    cb_right_action = np.array(cb_right_action)

    if qiangnao_state.size == 0:
        if cb_left_state.size > 0:
            qiangnao_state = cb_left_state
        elif cb_right_state.size > 0:
            qiangnao_state = cb_right_state
    if qiangnao_action.size == 0:
        if cb_left_action.size > 0:
            qiangnao_action = cb_left_action
        elif cb_right_action.size > 0:
            qiangnao_action = cb_right_action

    return claw_state, claw_action, qiangnao_state, qiangnao_action


def _split_dexhand_lr(arr):
    if arr is None:
        return None, None
    arr = np.array(arr)
    if arr.ndim != 2 or arr.shape[1] < 12:
        return None, None
    left = arr[:, :6]
    right = arr[:, 6:12]
    return left, right
