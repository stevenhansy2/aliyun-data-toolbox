import numpy as np
from pydrake.math import RollPitchYaw
import rosbag
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import BodyIndex


class KuavoPoseCalculator:
    def __init__(self, urdf_path):
        self.plant = MultibodyPlant(0.0)
        Parser(self.plant).AddModelFromFile(urdf_path)

        self.base_link_frame = self.plant.GetFrameByName("base_link")  # 基座坐标系
        self.plant.WeldFrames(self.plant.world_frame(), self.base_link_frame)
        self.plant.Finalize()
        self.context = self.plant.CreateDefaultContext()
        self.nq = self.plant.num_positions()  # 关节数

        # Debug
        log_print(
            "----------------------------------------------------------------------------"
        )
        joint_names = self.plant.GetPositionNames(
            add_model_instance_prefix=False, always_add_suffix=False
        )
        l_leg_joints = [name for name in joint_names if name.startswith("leg_l")]
        r_leg_joints = [name for name in joint_names if name.startswith("leg_r")]
        l_arm_joints = [name for name in joint_names if name.startswith("zarm_l")]
        r_arm_joints = [name for name in joint_names if name.startswith("zarm_r")]
        head_joints = [name for name in joint_names if name.startswith("zhead")]
        log_print("关节数量:", self.nq)
        log_print("左腿关节:", l_leg_joints)
        log_print("右腿关节:", r_leg_joints)
        log_print("左臂关节:", l_arm_joints)
        log_print("右臂关节:", r_arm_joints)
        log_print("头部关节:", head_joints)
        # for body_index in range(self.plant.num_bodies()):
        #     body = self.plant.get_body(BodyIndex(body_index))
        #     link_name = body.name()
        #     log_print(f"Link {body_index}: {link_name}")
        log_print(
            "----------------------------------------------------------------------------"
        )

    def get_camera_pose(self, head_q: list):
        """计算头部相机在基座坐标系中的位姿。

        Args:
            head_q (list): 头部关节角度列表, 长度为2, 分别对应头部的两个关节角度(yaw, pitch), 单位为弧度.

        Returns:
            RigidTransform: 头部相机在基座坐标系中的位姿变换矩阵。

        Raises:
            ValueError: 当head_q长度不为2 或关节角度列表中包含非数字值时抛出异常。
        """
        if len(head_q) != 2:
            raise ValueError(f"head_q must have length 2, but got {len(head_q)}")
        # 检查关节角度列表中是否包含非数字值
        for i, angle in enumerate(head_q):
            if (
                not isinstance(angle, (int, float))
                or np.isnan(angle)
                or np.isinf(angle)
            ):
                raise ValueError(f"head_q[{i}] must be a valid number, but got {angle}")

        q = np.zeros(self.nq)
        q[-2:] = head_q
        self.plant.SetPositions(self.context, q)
        # fk
        camera_pose_in_base = self.plant.GetFrameByName("camera_base").CalcPose(
            self.context, self.base_link_frame
        )
        return camera_pose_in_base

    def get_l_hand_camera_or_eef_pose(self, target_link_name: str, larm_q: list):
        """
        计算左臂相机或末端执行器在基座坐标系中的位姿。

        Args:
            target_link_name (str): 目标链接名称, 可以是 "l_hand_camera" 或 "zarm_l7_end_effector".
            larm_q (list): 左臂关节角度列表, 长度为7, 分别对应左臂的7个关节角度, 单位为弧度.

        Returns:
            RigidTransform: 左臂相机在基座坐标系中的位姿变换矩阵。

        Raises:
            ValueError: 当larm_q长度不为7 或关节角度列表中包含非数字值时抛出异常。
        """
        if len(larm_q) != 7:
            raise ValueError(f"larm_q must have length 7, but got {len(larm_q)}")

        # 检查关节角度列表中是否包含非数字值
        for i, angle in enumerate(larm_q):
            if (
                not isinstance(angle, (int, float))
                or np.isnan(angle)
                or np.isinf(angle)
            ):
                raise ValueError(f"larm_q[{i}] must be a valid number, but got {angle}")

        q = np.zeros(self.nq)
        q[12:19] = larm_q  # 左臂关节索引为 12-18 下标从 0 开始
        self.plant.SetPositions(self.context, q)
        # fk
        camera_pose_in_base = self.plant.GetFrameByName(target_link_name).CalcPose(
            self.context, self.base_link_frame
        )
        return camera_pose_in_base

    def get_r_hand_camera_or_eef_pose(self, target_link_name: str, rarm_q: list):
        """
        计算右臂相机或末端执行器在基座坐标系中的位姿。

        Args:
            target_link_name (str): 目标链接名称, 可以是 "r_hand_camera" 或 "zarm_r7_end_effector".
            rarm_q (list): 右臂关节角度列表, 长度为7, 分别对应右臂的7个关节角度, 单位为弧度.

        Returns:
            RigidTransform: 右臂相机在基座坐标系中的位姿变换矩阵。

        Raises:
            ValueError: 当rarm_q长度不为7 或关节角度列表中包含非数字值时抛出异常。
        """
        if len(rarm_q) != 7:
            raise ValueError(f"rarm_q must have length 7, but got {len(rarm_q)}")

        # 检查关节角度列表中是否包含非数字值
        for i, angle in enumerate(rarm_q):
            if (
                not isinstance(angle, (int, float))
                or np.isnan(angle)
                or np.isinf(angle)
            ):
                raise ValueError(f"rarm_q[{i}] must be a valid number, but got {angle}")

        q = np.zeros(self.nq)
        q[19:26] = rarm_q  # 右臂关节索引为 19-25 下标从 0 开始
        self.plant.SetPositions(self.context, q)
        # fk
        camera_pose_in_base = self.plant.GetFrameByName(target_link_name).CalcPose(
            self.context, self.base_link_frame
        )
        return camera_pose_in_base

    def get_l_hand_tripod_to_camera_pose(self):
        """
        获取左手三脚架到左手相机的位姿变换。

        Returns:
            RigidTransform: 左手三脚架到左手相机的位姿变换矩阵。
        """
        # 获取 l_hand_tripod 和 l_hand_camera 的 frame
        tripod_frame = self.plant.GetFrameByName("l_hand_tripod")
        camera_frame = self.plant.GetFrameByName("l_hand_camera")

        # 计算从 l_hand_tripod 到 l_hand_camera 的位姿变换
        tripod_to_camera_pose = camera_frame.CalcPose(self.context, tripod_frame)

        return tripod_to_camera_pose

    def get_r_hand_tripod_to_camera_pose(self):
        """
        获取右手三脚架到右手相机的位姿变换。

        Returns:
            RigidTransform: 右手三脚架到右手相机的位姿变换矩阵。
        """
        # 获取 r_hand_tripod 和 r_hand_camera 的 frame
        tripod_frame = self.plant.GetFrameByName("r_hand_tripod")
        camera_frame = self.plant.GetFrameByName("r_hand_camera")

        # 计算从 r_hand_tripod 到 r_hand_camera 的位姿变换
        tripod_to_camera_pose = camera_frame.CalcPose(self.context, tripod_frame)

        return tripod_to_camera_pose

    def get_head_link_to_camera_pose(self):
        """
        获取头部链接(zhead_2_link)到相机frame(camera)的转换位姿。

        Returns:
            RigidTransform: 从zhead_2_link到camera的位姿变换矩阵。
        """
        # 获取zhead_2_link frame
        zhead_2_link_frame = self.plant.GetFrameByName("zhead_2_link")
        # 获取camera frame
        camera_frame = self.plant.GetFrameByName("camera")

        # 计算从zhead_2_link到camera的转换
        transform_zhead_to_camera = camera_frame.CalcPose(
            self.context, zhead_2_link_frame
        )
        return transform_zhead_to_camera


def extract_pose_components(rigid_transform):
    position = rigid_transform.translation()
    rotation_matrix = rigid_transform.rotation().matrix()
    rpy = RollPitchYaw(rotation_matrix)
    quaternion = rpy.ToQuaternion()
    # 返回 xyzw 顺序
    return {
        "position": np.array(position, dtype=np.float32),
        "quaternion_xyzw": np.array(
            [quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()],
            dtype=np.float32,
        ),
    }


def extract_eef_pose_arrays(bag_path, urdf_path):
    pose_calculator = KuavoPoseCalculator(urdf_path)
    positions = []
    quaternions = []
    with rosbag.Bag(bag_path, "r") as input_bag:
        for topic, msg, t in input_bag.read_messages(topics=["/sensors_data_raw"]):
            joint_q = msg.joint_data.joint_q
            left_arm_q = joint_q[12:19]
            right_arm_q = joint_q[19:26]
            try:
                left_eef_pose = pose_calculator.get_l_hand_camera_or_eef_pose(
                    "zarm_l7_end_effector", left_arm_q
                )
                right_eef_pose = pose_calculator.get_r_hand_camera_or_eef_pose(
                    "zarm_r7_end_effector", right_arm_q
                )
                left = extract_pose_components(left_eef_pose)
                right = extract_pose_components(right_eef_pose)
                # 按照 (2, 3) 和 (2, 4) 组织
                positions.append([left["position"], right["position"]])
                quaternions.append([left["quaternion_xyzw"], right["quaternion_xyzw"]])
            except ValueError:
                continue
    # 转为 numpy 数组，shape (N, 2, 3) 和 (N, 2, 4)
    positions = np.array(positions, dtype=np.float32)
    quaternions = np.array(quaternions, dtype=np.float32)
    return positions, quaternions


def extract_and_format_eef_extrinsics(sensors_data_raw, urdf_path="./biped_s49.urdf"):
    """
    从 sensors_data_raw 计算左右手末端执行器的位姿（位置和四元数），返回 (N,2,3) 和 (N,2,4) 的 float32 数组
    """
    pose_calculator = KuavoPoseCalculator(urdf_path)
    positions = []
    quaternions = []
    for msg in sensors_data_raw:
        joint_q = msg["joint_q"] if isinstance(msg, dict) else msg.joint_data.joint_q
        left_arm_q = joint_q[12:19]
        right_arm_q = joint_q[19:26]
        try:
            left_eef_pose = pose_calculator.get_l_hand_camera_or_eef_pose(
                "zarm_l7_end_effector", left_arm_q
            )
            right_eef_pose = pose_calculator.get_r_hand_camera_or_eef_pose(
                "zarm_r7_end_effector", right_arm_q
            )
            left = extract_pose_components(left_eef_pose)
            right = extract_pose_components(right_eef_pose)
            positions.append([left["position"], right["position"]])
            quaternions.append([left["quaternion_xyzw"], right["quaternion_xyzw"]])
        except Exception:
            continue
    positions = np.array(positions, dtype=np.float32)
    quaternions = np.array(quaternions, dtype=np.float32)
    return positions, quaternions


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bag_path = os.path.join(script_dir, "111.bag")
    urdf_path = os.path.join(script_dir, "./biped_s49.urdf")
    positions, quaternions = extract_eef_pose_arrays(bag_path, urdf_path)
    log_print("positions shape:", positions.shape)  # (N, 2, 3)
    log_print("quaternions shape:", quaternions.shape)  # (N, 2, 4)
