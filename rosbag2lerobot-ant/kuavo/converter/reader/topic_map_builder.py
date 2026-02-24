"""Topic-map construction helpers for KuavoRosbagReader."""


def build_main_topic_map(
    msg_processer,
    source_topics: dict,
    *,
    joint_current_processor,
    actual_hand_state_topic: str | None,
    default_primary_hand_state: str,
):
    sensors_topic = source_topics.get("sensors_data_raw", "/sensors_data_raw")
    arm_traj_topic = source_topics.get("arm_traj", "/kuavo_arm_traj")
    joint_cmd_topic = source_topics.get("joint_cmd", "/joint_cmd")
    hand_cmd_topic = source_topics.get("hand_cmd", "/control_robot_hand_position")
    cb_left_state_topic = source_topics.get("cb_left_hand_state", "/cb_left_hand_state")
    cb_right_state_topic = source_topics.get(
        "cb_right_hand_state", "/cb_right_hand_state"
    )
    cb_left_cmd_topic = source_topics.get(
        "cb_left_hand_control_cmd", "/cb_left_hand_control_cmd"
    )
    cb_right_cmd_topic = source_topics.get(
        "cb_right_hand_control_cmd", "/cb_right_hand_control_cmd"
    )
    force_left_topic = source_topics.get(
        "force6d_left_hand_force_torque", "/force6d_left_hand_force_torque"
    )
    force_right_topic = source_topics.get(
        "force6d_right_hand_force_torque", "/force6d_right_hand_force_torque"
    )
    touch_left_topic = source_topics.get(
        "cb_left_hand_matrix_touch_pc2", "/cb_left_hand_matrix_touch_pc2"
    )
    touch_right_topic = source_topics.get(
        "cb_right_hand_matrix_touch_pc2", "/cb_right_hand_matrix_touch_pc2"
    )
    claw_state_topic = source_topics.get("leju_claw_state", "/leju_claw_state")
    claw_cmd_topic = source_topics.get("leju_claw_command", "/leju_claw_command")

    main_topic_map = {
        sensors_topic: [
            ("observation.sensorsData.joint_q", msg_processer.process_joint_q_state),
            ("observation.sensorsData.joint_v", msg_processer.process_joint_v_state),
            ("observation.sensorsData.joint_vd", msg_processer.process_joint_vd_state),
            ("observation.sensorsData.joint_current", joint_current_processor),
            (
                "observation.sensorsData.imu",
                msg_processer.process_sensors_data_raw_extract_imu,
            ),
        ],
        arm_traj_topic: [("action.kuavo_arm_traj", msg_processer.process_kuavo_arm_traj)],
        joint_cmd_topic: [
            ("action.joint_cmd.joint_q", msg_processer.process_joint_cmd_joint_q),
            ("action.joint_cmd.joint_v", msg_processer.process_joint_cmd_joint_v),
            ("action.joint_cmd.tau", msg_processer.process_joint_cmd_tau),
            ("action.joint_cmd.tau_max", msg_processer.process_joint_cmd_tau_max),
            ("action.joint_cmd.tau_ratio", msg_processer.process_joint_cmd_tau_ratio),
            ("action.joint_cmd.tau_joint_kp", msg_processer.process_joint_cmd_joint_kp),
            ("action.joint_cmd.tau_joint_kd", msg_processer.process_joint_cmd_joint_kd),
            (
                "action.joint_cmd.control_modes",
                msg_processer.process_joint_cmd_control_modes,
            ),
        ],
        hand_cmd_topic: [("action.qiangnao", msg_processer.process_qiangnao_cmd)],
        cb_left_state_topic: [
            ("observation.qiangnao_left", msg_processer.process_joint_state_position),
        ],
        cb_right_state_topic: [
            ("observation.qiangnao_right", msg_processer.process_joint_state_position),
        ],
        cb_left_cmd_topic: [
            ("action.qiangnao_left", msg_processer.process_joint_state_position),
        ],
        cb_right_cmd_topic: [
            ("action.qiangnao_right", msg_processer.process_joint_state_position),
        ],
        force_left_topic: [
            (
                "observation.state.hand_left.force_torque",
                msg_processer.process_wrench_stamped,
            ),
        ],
        force_right_topic: [
            (
                "observation.state.hand_right.force_torque",
                msg_processer.process_wrench_stamped,
            ),
        ],
        touch_left_topic: [
            (
                "observation.state.hand_left.touch_matrix",
                msg_processer.process_touch_matrix_pc2,
            ),
        ],
        touch_right_topic: [
            (
                "observation.state.hand_right.touch_matrix",
                msg_processer.process_touch_matrix_pc2,
            ),
        ],
        claw_state_topic: [("observation.claw", msg_processer.process_claw_state)],
        claw_cmd_topic: [("action.claw", msg_processer.process_claw_cmd)],
    }

    if actual_hand_state_topic == default_primary_hand_state:
        main_topic_map[actual_hand_state_topic] = [
            ("observation.qiangnao", msg_processer.process_qiangnao_state),
        ]
    elif actual_hand_state_topic == "/dexhand/state":
        def process_dexhand_state(msg):
            return {"data": list(msg.position), "timestamp": msg.header.stamp.to_sec()}

        main_topic_map[actual_hand_state_topic] = [
            ("observation.qiangnao", process_dexhand_state),
        ]

    return main_topic_map

