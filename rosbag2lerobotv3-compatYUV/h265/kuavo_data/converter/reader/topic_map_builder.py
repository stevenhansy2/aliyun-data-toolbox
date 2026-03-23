"""Topic-map construction helpers for the compat line."""


def build_main_topic_process_map(msg_processer, source_topics: dict) -> dict:
    sensors_topic = source_topics.get("sensors_data_raw", "/sensors_data_raw")
    arm_traj_topic = source_topics.get("arm_traj", "/kuavo_arm_traj")
    arm_traj_alt_topic = source_topics.get("arm_traj_alt", "/kuavo_arm_traj_synced")
    joint_cmd_topic = source_topics.get("joint_cmd", "/joint_cmd")
    cmd_pos_world_topic = source_topics.get("cmd_pos_world", "/cmd_pose_world_synced")
    hand_cmd_topic = source_topics.get("hand_cmd", "/control_robot_hand_position")
    hand_state_candidates = source_topics.get("hand_state_candidates", ["/control_robot_hand_position_state", "/dexhand/state"])
    claw_state_topic = source_topics.get("leju_claw_state", "/leju_claw_state")
    claw_cmd_topic = source_topics.get("leju_claw_command", "/leju_claw_command")
    rq2f85_state_topic = source_topics.get("rq2f85_state", "/gripper/state")
    rq2f85_cmd_topic = source_topics.get("rq2f85_command", "/gripper/command")

    return {
        "observation.state": {
            "topic": sensors_topic,
            "msg_process_fn": msg_processer.process_joint_state,
        },
        "action.kuavo_arm_traj": {
            "topic": arm_traj_topic,
            "msg_process_fn": msg_processer.process_kuavo_arm_traj,
        },
        "action.kuavo_arm_traj_alt": {
            "topic": arm_traj_alt_topic,
            "msg_process_fn": msg_processer.process_kuavo_arm_traj,
        },
        "action.cmd_pos_world": {
            "topic": cmd_pos_world_topic,
            "msg_process_fn": msg_processer.process_cmd_pos_world,
        },
        "action": {
            "topic": joint_cmd_topic,
            "msg_process_fn": msg_processer.process_joint_cmd,
        },
        "observation.imu": {
            "topic": sensors_topic,
            "msg_process_fn": msg_processer.process_sensors_data_raw_extract_imu,
        },
        "observation.claw": {
            "topic": claw_state_topic,
            "msg_process_fn": msg_processer.process_claw_state,
        },
        "action.claw": {
            "topic": claw_cmd_topic,
            "msg_process_fn": msg_processer.process_claw_cmd,
        },
        "observation.qiangnao": {
            "topic": hand_state_candidates[0],
            "topic_candidates": hand_state_candidates,
            "msg_process_fn": msg_processer.process_primary_hand_state,
        },
        "action.qiangnao": {
            "topic": hand_cmd_topic,
            "msg_process_fn": msg_processer.process_qiangnao_cmd,
        },
        "observation.rq2f85": {
            "topic": rq2f85_state_topic,
            "msg_process_fn": msg_processer.process_rq2f85_state,
        },
        "action.rq2f85": {
            "topic": rq2f85_cmd_topic,
            "msg_process_fn": msg_processer.process_rq2f85_cmd,
        },
    }
