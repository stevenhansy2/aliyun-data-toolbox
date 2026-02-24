"""End-effector pose data attachment helpers."""

from converter.kinematics.endeffector_pose import extract_and_format_eef_extrinsics


def attach_eef_pose_from_joint_q(data_dict: dict, urdf_path: str):
    """
    在 data_dict 中基于 observation.sensorsData.joint_q 追加:
      - end.position
      - end.orientation
    data_dict 的每个条目元素格式为 {"data": ..., "timestamp": ...}
    """
    joint_q_items = data_dict.get("observation.sensorsData.joint_q", [])
    if not joint_q_items:
        data_dict["end.position"] = []
        data_dict["end.orientation"] = []
        return

    joint_q_list = [item["data"] for item in joint_q_items]
    timestamps = [item["timestamp"] for item in joint_q_items]
    positions, quaternions = extract_and_format_eef_extrinsics(
        [{"joint_q": q} for q in joint_q_list],
        urdf_path=urdf_path,
    )

    data_dict["end.position"] = [
        {"data": positions[i], "timestamp": timestamps[i]} for i in range(len(positions))
    ]
    data_dict["end.orientation"] = [
        {"data": quaternions[i], "timestamp": timestamps[i]}
        for i in range(len(quaternions))
    ]

