"""Batch frame construction helpers for dataset writing."""

from __future__ import annotations

import gc

import cv2
import numpy as np
import torch

from converter.data_utils import get_nested_value


def _build_output_state_action(
    i: int,
    state: np.ndarray,
    action: np.ndarray,
    claw_state: np.ndarray | None,
    claw_action: np.ndarray | None,
    hand_state_left: np.ndarray | None,
    hand_state_right: np.ndarray | None,
    hand_action_left: np.ndarray | None,
    hand_action_right: np.ndarray | None,
    *,
    only_half_up_body: bool,
    control_hand_side: str,
    slice_robot,
    slice_claw,
    merge_hand_position: bool,
    use_leju_claw_batch: bool,
):
    if only_half_up_body:
        if use_leju_claw_batch:
            if control_hand_side in ("left", "both"):
                l0, l1 = slice_robot[0][0], slice_robot[0][-1]
                c0, c1 = slice_claw[0][0], slice_claw[0][-1]
                left_len = (l1 - l0) + (c1 - c0)
                output_state = np.empty((left_len,), dtype=np.float32)
                output_action = np.empty((left_len,), dtype=np.float32)
                output_state[: (l1 - l0)] = state[i, l0:l1]
                output_state[(l1 - l0) :] = claw_state[i, c0:c1]
                output_action[: (l1 - l0)] = action[i, l0:l1]
                output_action[(l1 - l0) :] = claw_action[i, c0:c1]
            if control_hand_side in ("right", "both"):
                r0, r1 = slice_robot[1][0], slice_robot[1][-1]
                rc0, rc1 = slice_claw[1][0], slice_claw[1][-1]
                right_len = (r1 - r0) + (rc1 - rc0)
                right_state = np.empty((right_len,), dtype=np.float32)
                right_action = np.empty((right_len,), dtype=np.float32)
                right_state[: (r1 - r0)] = state[i, r0:r1]
                right_state[(r1 - r0) :] = claw_state[i, rc0:rc1]
                right_action[: (r1 - r0)] = action[i, r0:r1]
                right_action[(r1 - r0) :] = claw_action[i, rc0:rc1]
                if control_hand_side == "both":
                    output_state = np.concatenate((output_state, right_state), axis=0)
                    output_action = np.concatenate((output_action, right_action), axis=0)
                else:
                    output_state = right_state
                    output_action = right_action
        else:
            if control_hand_side in ("left", "both"):
                l0, l1 = slice_robot[0][0], slice_robot[0][-1]
                output_state = np.array(state[i, l0:l1], dtype=np.float32)
                output_action = np.array(action[i, l0:l1], dtype=np.float32)
            if control_hand_side in ("right", "both"):
                r0, r1 = slice_robot[1][0], slice_robot[1][-1]
                right_state = np.array(state[i, r0:r1], dtype=np.float32)
                right_action = np.array(action[i, r0:r1], dtype=np.float32)
                if control_hand_side == "both":
                    output_state = np.concatenate((output_state, right_state), axis=0)
                    output_action = np.concatenate((output_action, right_action), axis=0)
                else:
                    output_state = right_state
                    output_action = right_action
    else:
        if use_leju_claw_batch:
            output_state = np.empty((30,), dtype=np.float32)
            output_action = np.empty((30,), dtype=np.float32)
            output_state[0:19] = state[i, 0:19]
            output_action[0:19] = action[i, 0:19]
            output_state[19] = float(claw_state[i, 0])
            output_action[19] = float(claw_action[i, 0])
            output_state[20:27] = state[i, 19:26]
            output_action[20:27] = action[i, 19:26]
            output_state[27] = float(claw_state[i, 1])
            output_action[27] = float(claw_action[i, 1])
            output_state[28:30] = state[i, 26:28]
            output_action[28:30] = action[i, 26:28]
        else:
            output_state = np.array(state[i, :], dtype=np.float32)
            output_action = np.array(action[i, :], dtype=np.float32)

    if merge_hand_position:
        left_pos = (
            hand_state_left[i]
            if hand_state_left is not None and len(hand_state_left) > i
            else np.zeros((6,), dtype=np.float32)
        )
        right_pos = (
            hand_state_right[i]
            if hand_state_right is not None and len(hand_state_right) > i
            else np.zeros((6,), dtype=np.float32)
        )
        left_act = (
            hand_action_left[i]
            if hand_action_left is not None and len(hand_action_left) > i
            else np.zeros((6,), dtype=np.float32)
        )
        right_act = (
            hand_action_right[i]
            if hand_action_right is not None and len(hand_action_right) > i
            else np.zeros((6,), dtype=np.float32)
        )
        output_state = np.concatenate((output_state, left_pos, right_pos), axis=0)
        output_action = np.concatenate((output_action, left_act, right_act), axis=0)

    return output_state, output_action


def write_batch_frames(
    dataset,
    task: str,
    raw_config,
    *,
    num_frames: int,
    state: np.ndarray,
    action: np.ndarray,
    claw_state: np.ndarray | None,
    claw_action: np.ndarray | None,
    hand_state_left: np.ndarray | None,
    hand_state_right: np.ndarray | None,
    hand_action_left: np.ndarray | None,
    hand_action_right: np.ndarray | None,
    all_low_dim_data: dict,
    extrinsics_dict: dict,
    imgs_per_cam: dict[str, list[bytes]],
    only_half_up_body: bool,
    control_hand_side: str,
    slice_robot,
    slice_claw,
    merge_hand_position: bool,
    use_leju_claw_batch: bool,
    use_leju_claw: bool,
    use_qiangnao: bool,
):
    separate_video_storage = getattr(raw_config, "separate_video_storage", False)

    for i in range(num_frames):
        output_state, output_action = _build_output_state_action(
            i,
            state,
            action,
            claw_state,
            claw_action,
            hand_state_left,
            hand_state_right,
            hand_action_left,
            hand_action_right,
            only_half_up_body=only_half_up_body,
            control_hand_side=control_hand_side,
            slice_robot=slice_robot,
            slice_claw=slice_claw,
            merge_hand_position=merge_hand_position,
            use_leju_claw_batch=use_leju_claw_batch,
        )

        frame = {
            "observation.state": torch.from_numpy(output_state).type(torch.float32),
            "action": torch.from_numpy(output_action).type(torch.float32),
            "action.head.position": get_nested_value(
                all_low_dim_data, "action.head.position", i, [0.0] * 2
            ),
            "action.arm.position": get_nested_value(
                all_low_dim_data, "action.arm.position", i, [0.0] * 14
            ),
            "action.leg.position": get_nested_value(
                all_low_dim_data, "action.leg.position", i, [0.0] * 12
            ),
            "observation.state.head.effort": get_nested_value(
                all_low_dim_data, "state.head.effort", i, [0.0] * 2
            ),
            "observation.state.head.position": get_nested_value(
                all_low_dim_data, "state.head.position", i, [0.0] * 2
            ),
            "observation.state.head.velocity": get_nested_value(
                all_low_dim_data, "state.head.velocity", i, [0.0] * 2
            ),
            "observation.state.arm.current_value": get_nested_value(
                all_low_dim_data, "state.arm.current_value", i, [0.0] * 14
            ),
            "observation.state.arm.effort": get_nested_value(
                all_low_dim_data, "state.arm.effort", i, [0.0] * 14
            ),
            "observation.state.arm.position": get_nested_value(
                all_low_dim_data, "state.arm.position", i, [0.0] * 14
            ),
            "observation.state.arm.velocity": get_nested_value(
                all_low_dim_data, "state.arm.velocity", i, [0.0] * 14
            ),
            "observation.state.end.orientation": (
                get_nested_value(all_low_dim_data, "state.end.orientation", i, [0.0] * 8)
            ).flatten(),
            "observation.state.end.position": (
                get_nested_value(all_low_dim_data, "state.end.position", i, [0.0] * 6)
            ).flatten(),
            "observation.state.leg.current_value": get_nested_value(
                all_low_dim_data, "state.leg.current_value", i, [0.0] * 12
            ),
            "observation.state.leg.effort": get_nested_value(
                all_low_dim_data, "state.leg.effort", i, [0.0] * 12
            ),
            "observation.state.leg.position": get_nested_value(
                all_low_dim_data, "state.leg.position", i, [0.0] * 12
            ),
            "observation.state.leg.velocity": get_nested_value(
                all_low_dim_data, "state.leg.velocity", i, [0.0] * 12
            ),
            "imu.acc_xyz": get_nested_value(all_low_dim_data, "imu.acc_xyz", i, [0.0] * 3),
            "imu.gyro_xyz": get_nested_value(
                all_low_dim_data, "imu.gyro_xyz", i, [0.0] * 3
            ),
            "imu.free_acc_xyz": get_nested_value(
                all_low_dim_data, "imu.free_acc_xyz", i, [0.0] * 3
            ),
            "imu.quat_xyzw": get_nested_value(
                all_low_dim_data, "imu.quat_xyzw", i, [0.0] * 4
            ),
        }

        if use_leju_claw:
            frame.update(
                {
                    "action.effector.position": get_nested_value(
                        all_low_dim_data, "action.effector.position", i, [0.0] * 2
                    ),
                    "observation.state.effector.position": get_nested_value(
                        all_low_dim_data, "state.effector.position", i, [0.0] * 2
                    ),
                }
            )
        if use_qiangnao:
            frame.update(
                {
                    "action.hand_left.position": get_nested_value(
                        all_low_dim_data, "action.hand_left.position", i, [0.0] * 6
                    ),
                    "action.hand_right.position": get_nested_value(
                        all_low_dim_data, "action.hand_right.position", i, [0.0] * 6
                    ),
                    "observation.state.hand_left.position": get_nested_value(
                        all_low_dim_data, "state.hand_left.position", i, [0.0] * 6
                    ),
                    "observation.state.hand_right.position": get_nested_value(
                        all_low_dim_data, "state.hand_right.position", i, [0.0] * 6
                    ),
                    "observation.state.hand_left.force_torque": get_nested_value(
                        all_low_dim_data, "state.hand_left.force_torque", i, [0.0] * 6
                    ),
                    "observation.state.hand_right.force_torque": get_nested_value(
                        all_low_dim_data, "state.hand_right.force_torque", i, [0.0] * 6
                    ),
                    "observation.state.hand_left.touch_matrix": get_nested_value(
                        all_low_dim_data, "state.hand_left.touch_matrix", i, [0.0] * 360
                    ),
                    "observation.state.hand_right.touch_matrix": get_nested_value(
                        all_low_dim_data, "state.hand_right.touch_matrix", i, [0.0] * 360
                    ),
                }
            )

        for cam_key, extrs in extrinsics_dict.items():
            if extrs and len(extrs) > i:
                rot = np.array(extrs[i]["rotation_matrix"], dtype=np.float32).reshape(-1)
                trans = np.array(extrs[i]["translation_vector"], dtype=np.float32).reshape(
                    -1
                )
                frame[f"observation.camera_params.rotation_matrix_flat.{cam_key}"] = rot
                frame[f"observation.camera_params.translation_vector.{cam_key}"] = trans

        if not separate_video_storage:
            for camera, img_list in imgs_per_cam.items():
                if i < len(img_list):
                    img_bytes = img_list[i]
                    img_np = cv2.imdecode(
                        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                    )
                    if img_np is None:
                        raise ValueError(
                            f"Failed to decode color image for camera {camera} at frame {i}"
                        )
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    img_np = cv2.resize(
                        img_np,
                        (raw_config.resize.width, raw_config.resize.height),
                    )
                    frame[f"observation.images.{camera}"] = img_np

        dataset.add_frame(frame, task=task)

        if (
            i % 800 == 0
            and hasattr(dataset, "_wait_image_writer")
            and dataset._wait_image_writer
        ):
            if dataset.image_writer.queue.qsize() > 500:
                dataset._wait_image_writer()
                gc.collect()

