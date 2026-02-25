USE_JOINT_CURRENT_STATE = True

DEFAULT_LEG_JOINT_NAMES = [
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
DEFAULT_ARM_JOINT_NAMES = [
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
DEFAULT_HEAD_JOINT_NAMES = ["head_yaw", "head_pitch"]
DEFAULT_DEXHAND_JOINT_NAMES = [
    "left_qiangnao_1",
    "left_qiangnao_2",
    "left_qiangnao_3",
    "left_qiangnao_4",
    "left_qiangnao_5",
    "left_qiangnao_6",
    "right_qiangnao_1",
    "right_qiangnao_2",
    "right_qiangnao_3",
    "right_qiangnao_4",
    "right_qiangnao_5",
    "right_qiangnao_6",
]
DEFAULT_LEJUCLAW_JOINT_NAMES = [
    "left_claw",
    "right_claw",
]

DEFAULT_JOINT_NAMES_LIST = (
    DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + DEFAULT_HEAD_JOINT_NAMES
)

DEFAULT_JOINT_NAMES = {
    "full_joint_names": DEFAULT_LEG_JOINT_NAMES
    + DEFAULT_ARM_JOINT_NAMES
    + DEFAULT_HEAD_JOINT_NAMES,
    "leg_joint_names": DEFAULT_LEG_JOINT_NAMES,
    "arm_joint_names": DEFAULT_ARM_JOINT_NAMES,
    "head_joint_names": DEFAULT_HEAD_JOINT_NAMES,
}
