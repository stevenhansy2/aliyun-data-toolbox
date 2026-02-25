import json
import os
from pathlib import Path

import numpy as np

from converter.configs.runtime_config import Config
from converter.reader.reader_entry import KuavoRosbagReader, PostProcessorUtils
from converter.utils.facade import (
    flip_camera_arrays_if_needed,
    load_camera_info_per_camera,
    load_raw_depth_images_per_camera,
    load_raw_images_per_camera,
    swap_left_right_data_if_needed,
)


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
    bag_data, ori_bag_data = bag_reader.process_rosbag(
        ep_path,
        start_time=start_time,
        end_time=end_time,
        action_config=action_config,
        min_duration=min_duration,
        is_align=True,
        return_raw=True,
    )
    sn_code = None
    is_wheel_arm = False
    if metadata_json_dir and os.path.exists(metadata_json_dir):
        try:
            with open(metadata_json_dir, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
            sn_code = raw_metadata.get("device_sn", "")
            is_wheel_arm = sn_code.startswith("LB")
            log_print(f"[INFO] 检测到设备序列号: {sn_code}, 是否为轮臂: {is_wheel_arm}")
        except Exception as e:
            log_print(f"[WARN] 读取metadata.json失败: {e}, 默认包含腿部数据")
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
    # log_print("==========================0000000000000000000000===========================",'\n',leju_claw_command__position)
    # TODO: 夹爪添加velocity和effort数据

    try:
        control_robot_hand_position_state_both = qiangnao_state = np.array(
            [msg["data"] for msg in bag_data["observation.qiangnao"]], dtype=np.float32
        )
    except KeyError:
        log_print("[WARN] 未找到 'observation.qiangnao' 数据，使用空值")
        control_robot_hand_position_state_both = qiangnao_state = None

    try:
        control_robot_hand_position_both = qiangnao_action = np.array(
            [msg["data"] for msg in bag_data["action.qiangnao"]], dtype=np.float32
        )
    except KeyError:
        log_print("[WARN] 未找到 'action.qiangnao' 数据，使用空值")
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
        log_print(f"[INFO] 设备类型为非轮臂，添加腿部数据到HDF5")
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
        log_print(f"[INFO] 设备类型为轮臂(LB开头)，跳过腿部数据，不添加到HDF5")
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
            log_print("[WARN] 未找到 'observation.qiangnao' 数据，使用空值")
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
        log_print(f"[WARN] 构建 all_low_dim_data_original 时出错: {_e}")
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

