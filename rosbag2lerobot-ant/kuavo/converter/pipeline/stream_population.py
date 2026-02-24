"""Streaming dataset population for port pipeline."""

import gc
import json
import logging
import os
import time

import numpy as np
from converter.config import Config
from converter.pipeline.dataset_builder import create_empty_dataset
from converter.pipeline.frame_builder import write_batch_frames
from converter.pipeline.stream_finalize import (
    persist_batch_media,
    save_batch_metadata_json,
    save_first_batch_parameters,
)
from converter.reader.kuavo_dataset_slave_s import KuavoRosbagReader
from converter.reader.postprocess_utils import PostProcessorUtils
from converter.data_utils import (
    _split_dexhand_lr,
    get_bag_time_info,
    get_time_range_from_moments,
)

logger = logging.getLogger(__name__)

def populate_dataset_stream(
    raw_config: Config,
    bag_files: list,
    task: str,
    mode: str,
    moment_json_dir: str | None,
    base_root: str,
    context: dict,
    metadata_json_dir: str | None = None,
    pipeline_encoder: "BatchSegmentEncoder | None" = None,
    streaming_encoder: "StreamingVideoEncoderManager | None" = None,
):
    use_depth = context["use_depth"]
    episode_uuid = context["episode_uuid"]
    dataset_config = context["dataset_config"]
    only_half_up_body = context["only_half_up_body"]
    control_hand_side = context["control_hand_side"]
    slice_robot = context["slice_robot"]
    slice_claw = context["slice_claw"]
    merge_hand_position = context["merge_hand_position"]
    use_leju_claw = context["use_leju_claw"]
    use_qiangnao = context["use_qiangnao"]
    default_joint_names_list = context["default_joint_names_list"]

    # 读取 metadata.json 获取 sn_code（相机左右翻转判定）
    sn_code = None
    if metadata_json_dir and os.path.exists(metadata_json_dir):
        try:
            with open(metadata_json_dir, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
            # 支持新格式（deviceSn）和旧格式（device_sn）
            sn_code = raw_metadata.get("deviceSn") or raw_metadata.get("device_sn", "")
        except Exception as e:
            logger.exception("[WARN] 读取metadata.json失败: %s", e)

    if len(bag_files) == 0:
        logger.warning("[WARN] 无 bag 文件")
        return None, None

    # 遍历每个 bag
    for ep_idx, bag_info in enumerate(bag_files):
        if isinstance(bag_info, dict):
            ep_path = bag_info["local_path"]
            start_time = bag_info.get("start", 0)
            end_time = bag_info.get("end", 1)
        else:
            ep_path = bag_info
            start_time = 0
            end_time = 1

        # moments.json 或 metadata.json（新格式）覆盖时间窗
        moments_start_time, moments_end_time = get_time_range_from_moments(
            moment_json_dir, metadata_json_path=metadata_json_dir
        )
        if moments_start_time is not None and moments_end_time is not None:
            logger.info(
                f"[MOMENTS] 覆盖使用标注文件时间范围: {moments_start_time} - {moments_end_time}"
            )
            start_time = moments_start_time
            end_time = moments_end_time

        # bag 时间信息（用于 metadata 合并）
        bag_time_info = get_bag_time_info(ep_path)
        if bag_time_info["iso_format"]:
            logger.info("Bag开始时间: %s", bag_time_info["iso_format"])
            logger.info("Bag持续时间: %.2f秒", bag_time_info["duration"])

        # 流式 reader
        reader = KuavoRosbagReader(raw_config, use_depth)
        extrinsics_dict = {}
        # 逐批消费
        batch_id = 0
        _t_prev_batch_end = time.time()  # 用于计算 generator yield 耗时

        # 提前获取配置，避免循环体内未定义
        separate_video_storage = getattr(
            raw_config, "separate_video_storage", False
        )
        cam_stats = {}  # 初始化，避免无 batch 时未定义

        # 选择串行或并行读取
        use_parallel = getattr(raw_config, "use_parallel_rosbag_read", False)
        num_workers = getattr(raw_config, "parallel_rosbag_workers", 2)

        if use_parallel:
            logger.info("[STREAM] 启用并行 ROSbag 读取 (%s workers)", num_workers)
            batch_iter = reader.process_rosbag_parallel(
                str(ep_path),
                start_time=start_time,
                end_time=end_time,
                action_config=None,
                chunk_size=800,
                num_workers=num_workers,
            )
        else:
            batch_iter = reader.process_rosbag(
                str(ep_path),
                start_time=start_time,
                end_time=end_time,
                action_config=None,
                chunk_size=800,
            )
        for aligned_batch in batch_iter:
            batch_id += 1
            _t_batch_start = time.time()
            _t_rosbag_read = (
                _t_batch_start - _t_prev_batch_end
            )  # ROSbag读取+对齐时间
            main_key = getattr(reader, "MAIN_TIMESTAMP_TOPIC", "camera_top")
            if main_key not in aligned_batch or len(aligned_batch[main_key]) == 0:
                logger.warning("[STREAM] 批次%s 无主时间线，跳过", batch_id)
                continue

            # 主时间戳
            main_ts = np.array(
                [it["timestamp"] for it in aligned_batch[main_key]],
                dtype=np.float64,
            )

            first_ts = float(main_ts[0])
            last_ts = float(main_ts[-1])

            # 每批提取相机外参（按时间窗）
            if batch_id == 1:
                try:
                    extrinsics = reader.extract_and_format_camera_extrinsics(
                        str(ep_path), abs_start=first_ts, abs_end=last_ts
                    )
                    head_extrinsics = extrinsics.get("head_camera_extrinsics", [])
                    left_extrinsics = extrinsics.get(
                        "left_hand_camera_extrinsics", []
                    )
                    right_extrinsics = extrinsics.get(
                        "right_hand_camera_extrinsics", []
                    )
                except Exception as e:
                    logger.exception("[WARN] 批次%s 外参提取失败: %s", batch_id, e)
                    head_extrinsics, left_extrinsics, right_extrinsics = [], [], []

            # 颜色/深度/相机信息
            _t_extract_start = time.time()
            cameras = raw_config.default_camera_names
            imgs_per_cam = {
                cam: [x["data"] for x in aligned_batch.get(cam, [])]
                for cam in cameras
            }
            if use_depth:
                imgs_per_cam_depth = {
                    cam: [x["data"] for x in aligned_batch.get(f"{cam}_depth", [])]
                    for cam in cameras
                }
                compressed = {
                    cam: (
                        aligned_batch.get(f"{cam}_depth", [])[0].get(
                            "compressed", None
                        )
                        if len(aligned_batch.get(f"{cam}_depth", [])) > 0
                        else None
                    )
                    for cam in cameras
                }
            else:
                imgs_per_cam_depth = None
                compressed = None
            info_per_cam = {
                cam: [
                    np.array(x["data"], dtype=np.float32)
                    for x in aligned_batch.get(f"{cam}_camera_info", [])
                ]
                for cam in cameras
            }
            distortion_model = {
                cam: [
                    x.get("distortion_model", None)
                    for x in aligned_batch.get(f"{cam}_camera_info", [])
                ]
                for cam in cameras
            }

            # 相机翻转（基于 sn_code）
            # if sn_code is not None and len(main_ts) > 0:
            #     imgs_per_cam, imgs_per_cam_depth = flip_camera_arrays_if_needed(
            #         imgs_per_cam, imgs_per_cam_depth, sn_code, main_ts[0]
            #     )

            # 低维数据/末端位姿
            def get_arr(key, dflt_shape=None):
                items = aligned_batch.get(key, [])
                if not items:
                    return None
                return np.array([x["data"] for x in items], dtype=np.float32)

            # print(get_arr("observation.sensorsData.joint_q").shape)
            state = get_arr(
                "observation.sensorsData.joint_q"
            )  # or np.zeros((0, 28), dtype=np.float32)
            # sensors_data_raw__joint_v = get_arr("observation.sensorsData.joint_v") #or np.zeros((len(state), 28), dtype=np.float32)
            state_joint_current = get_arr(
                "observation.sensorsData.joint_current"
            )  # or np.zeros((len(state), 28), dtype=np.float32)
            action = get_arr(
                "action.joint_cmd.joint_q"
            )  # or np.zeros((0, 28), dtype=np.float32)
            action_kuavo_arm_traj = get_arr(
                "action.kuavo_arm_traj"
            )  # or np.zeros((0, 14), dtype=np.float32)
            sensors_data_raw__joint_v = get_arr(
                "observation.sensorsData.joint_v"
            )  # or np.zeros((len(state), 28), dtype=np.float32)
            state_joint_current_arr = get_arr(
                "observation.sensorsData.joint_current"
            )  # or np.zeros((len(state), 28), dtype=np.float32)
            sensors_data_raw__imu_data = get_arr(
                "observation.sensorsData.imu"
            )  # or np.zeros((len(state), 13), dtype=np.float32)

            claw_state = get_arr(
                "observation.claw"
            )  # or np.zeros((len(state), 2), dtype=np.float32)
            claw_action = get_arr(
                "action.claw"
            )  # or np.zeros((len(state), 2), dtype=np.float32)
            qiangnao_state = get_arr("observation.qiangnao")
            qiangnao_action = get_arr("action.qiangnao")
            hand_state_left = get_arr("observation.qiangnao_left")
            hand_state_right = get_arr("observation.qiangnao_right")
            hand_action_left = get_arr("action.qiangnao_left")
            hand_action_right = get_arr("action.qiangnao_right")
            hand_force_left = get_arr("observation.state.hand_left.force_torque")
            hand_force_right = get_arr("observation.state.hand_right.force_torque")
            hand_touch_left = get_arr("observation.state.hand_left.touch_matrix")
            hand_touch_right = get_arr("observation.state.hand_right.touch_matrix")
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

            end_position = get_arr(
                "end.position"
            )  # or np.zeros((len(state), 6), dtype=np.float32)
            end_orientation = get_arr(
                "end.orientation"
            )  # or np.zeros((len(state), 8), dtype=np.float32)

            # 填充 action 的关节子段（12:26）为 kuavo_arm_traj
            if action.size > 0 and action_kuavo_arm_traj.size > 0:
                min_rows = min(len(action), len(action_kuavo_arm_traj))
                action[:min_rows, 12:26] = action_kuavo_arm_traj[:min_rows]

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

            sensors_data_raw__joint_current = (
                PostProcessorUtils.torque_to_current_batch(
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

            # 4. 提取子数组并清理原始数组
            head_effort = sensors_data_raw__joint_effort[:, 26:28]
            head_current = sensors_data_raw__joint_current[:, 26:28]
            joint_effort = sensors_data_raw__joint_effort[:, 12:26]
            joint_current = sensors_data_raw__joint_current[:, 12:26]

            # all_low_dim_data（按批次）
            main_ts_ns = (main_ts * 1e9).astype(np.int64)
            all_low_dim_data = {
                "timestamps": main_ts_ns,
                "action": {
                    "effector": {
                        "position": claw_action,
                        "index": main_ts_ns,
                    },
                    "hand_left": {
                        "position": hand_action_left,
                        "index": main_ts_ns,
                    },
                    "hand_right": {
                        "position": hand_action_right,
                        "index": main_ts_ns,
                    },
                    "arm": {
                        "position": action_kuavo_arm_traj,
                        "index": main_ts_ns,
                    },
                    "head": {
                        "position": action[:, 26:28],
                        "index": main_ts_ns,
                    },
                    "leg": {
                        "position": action[:, :12],
                        "index": main_ts_ns,
                    },
                },
                "state": {
                    "effector": {
                        "position": claw_state,
                    },
                    "hand_left": {
                        "position": hand_state_left,
                        "force_torque": hand_force_left,
                        "touch_matrix": hand_touch_left,
                    },
                    "hand_right": {
                        "position": hand_state_right,
                        "force_torque": hand_force_right,
                        "touch_matrix": hand_touch_right,
                    },
                    "head": {
                        "current_value": head_current,
                        "effort": head_effort,
                        "position": state[:, 26:28],
                        "velocity": sensors_data_raw__joint_v[:, 26:28],
                    },
                    "arm": {
                        "current_value": joint_current,
                        "effort": joint_effort,
                        "position": state[:, 12:26],
                        "velocity": sensors_data_raw__joint_v[:, 12:26],
                    },
                    "end": {
                        "position": end_position,
                        "orientation": end_orientation,
                    },
                    "leg": {
                        "current_value": sensors_data_raw__joint_current[:, :12],
                        "effort": sensors_data_raw__joint_effort[:, :12],
                        "position": state[:, 0:12],
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
            _t_extract_end = time.time()

            # 为该批创建独立数据集 root: {base_root}/batch_{id}
            batch_root = os.path.join(base_root, f"batch_{batch_id:04d}")
            # os.makedirs(batch_root, exist_ok=True)
            use_leju_claw_batch = (
                use_leju_claw
                and claw_state is not None
                and claw_action is not None
                and len(claw_state) > 0
                and len(claw_action) > 0
            )
            use_qiangnao_batch = (
                use_qiangnao
                and hand_state_left is not None
                and hand_state_right is not None
                and hand_action_left is not None
                and hand_action_right is not None
                and len(hand_state_left) > 0
                and len(hand_state_right) > 0
                and len(hand_action_left) > 0
                and len(hand_action_right) > 0
            )
            eef_type = "leju_claw" if use_leju_claw_batch else "dex_hand"
            _t_create_dataset_start = time.time()
            dataset = create_empty_dataset(
                repo_id=f"lerobot/kuavo",
                robot_type="kuavo4pro",
                mode=mode,
                eef_type=eef_type,
                dataset_config=dataset_config,
                has_depth_image=use_depth,
                root=batch_root,
                raw_config=raw_config,
                joint_names_list=default_joint_names_list,
            )
            _t_create_dataset_end = time.time()

            # 帧写入（与原逻辑一致）
            if batch_id == 1:
                extrinsics_map = {
                    "camera_top": head_extrinsics,
                    "camera_wrist_left": left_extrinsics,
                    "camera_wrist_right": right_extrinsics,
                    "head_cam_h": head_extrinsics,
                    "wrist_cam_l": left_extrinsics,
                    "wrist_cam_r": right_extrinsics,
                }
                extrinsics_dict = {
                    cam: extrinsics_map[cam]
                    for cam in cameras
                    if cam in extrinsics_map
                }

            num_frames = state.shape[0]
            print(f"[STREAM] 批次{batch_id} 写入 {num_frames} 帧")

            _t_frame_loop_start = time.time()
            write_batch_frames(
                dataset=dataset,
                task=task,
                raw_config=raw_config,
                num_frames=num_frames,
                state=state,
                action=action,
                claw_state=claw_state,
                claw_action=claw_action,
                hand_state_left=hand_state_left,
                hand_state_right=hand_state_right,
                hand_action_left=hand_action_left,
                hand_action_right=hand_action_right,
                all_low_dim_data=all_low_dim_data,
                extrinsics_dict=extrinsics_dict,
                imgs_per_cam=imgs_per_cam,
                    only_half_up_body=only_half_up_body,
                    control_hand_side=control_hand_side,
                    slice_robot=slice_robot,
                    slice_claw=slice_claw,
                    merge_hand_position=merge_hand_position,
                    use_leju_claw_batch=use_leju_claw_batch,
                    use_leju_claw=use_leju_claw,
                    use_qiangnao=use_qiangnao,
                )

            # 保存一批（低维数据）
            _t_frame_loop_end = time.time()
            _t_save_episode_start = time.time()
            dataset.save_episode()
            _t_save_episode_end = time.time()

            # 根据配置选择视频处理方式
            _t_save_images_start = time.time()
            (
                cam_stats,
                _t_save_images_end,
                separate_video_storage,
                imgs_per_cam,
                imgs_per_cam_depth,
            ) = persist_batch_media(
                raw_config,
                episode_uuid=episode_uuid,
                batch_id=batch_id,
                batch_root=batch_root,
                cameras=cameras,
                compressed=compressed,
                imgs_per_cam=imgs_per_cam,
                imgs_per_cam_depth=imgs_per_cam_depth,
                pipeline_encoder=pipeline_encoder,
                streaming_encoder=streaming_encoder,
                cam_stats=cam_stats,
            )

            # 保存参数（camera info 与 extrinsics）
            save_first_batch_parameters(
                batch_id=batch_id,
                batch_root=batch_root,
                info_per_cam=info_per_cam,
                distortion_model=distortion_model,
                cameras=cameras,
            )

            # 保存 metadata.json（按批次）
            save_batch_metadata_json(
                metadata_json_path=metadata_json_dir,
                moment_json_path=moment_json_dir,
                batch_root=batch_root,
                episode_uuid=episode_uuid,
                raw_config=raw_config,
                bag_time_info=bag_time_info,
                main_ts=main_ts,
            )

            # 释放批次内存
            del dataset, info_per_cam, distortion_model

            # 如果没有在前面删除，这里删除图像数据
            if not separate_video_storage:
                if "imgs_per_cam" in locals():
                    del imgs_per_cam
                if "imgs_per_cam_depth" in locals():
                    del imgs_per_cam_depth

            del (
                state,
                action,
                action_kuavo_arm_traj,
                sensors_data_raw__joint_v,
                state_joint_current_arr,
                sensors_data_raw__imu_data,
            )
            del claw_state, claw_action, qiangnao_state, qiangnao_action
            del (
                end_position,
                end_orientation,
                all_low_dim_data,
            )
            if batch_id == 1:
                del head_extrinsics, left_extrinsics, right_extrinsics
            gc.collect()

            # ===== 计时汇总 =====
            _t_batch_end = time.time()
            _t_total = _t_batch_end - _t_batch_start
            _t_extract = _t_extract_end - _t_extract_start
            _t_create = _t_create_dataset_end - _t_create_dataset_start
            _t_frames = _t_frame_loop_end - _t_frame_loop_start
            _t_save_ep = _t_save_episode_end - _t_save_episode_start
            _t_save_img = (
                (_t_save_images_end - _t_save_images_start)
                if _t_save_images_end
                else 0
            )
            print(
                f"[TIMING] Batch {batch_id}: ROSbag读取={_t_rosbag_read:.2f}s | "
                f"数据提取={_t_extract:.2f}s | Dataset创建={_t_create:.2f}s | "
                f"帧循环={_t_frames:.2f}s | Parquet保存={_t_save_ep:.2f}s | "
                f"图像保存={_t_save_img:.2f}s | 批次总计={_t_total:.2f}s"
            )
            _t_prev_batch_end = time.time()  # 更新为下一批准备

    if separate_video_storage:
        return cam_stats
    else:
        return None
