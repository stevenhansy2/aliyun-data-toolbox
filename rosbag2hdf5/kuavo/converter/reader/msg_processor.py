import numpy as np


class TimestampStuckError(Exception):
    """时间戳卡顿异常"""

    def __init__(
        self,
        message,
        topic=None,
        stuck_timestamp=None,
        stuck_duration=None,
        stuck_frame_count=None,
        threshold=None,
    ):
        super().__init__(message)
        self.topic = topic
        self.stuck_timestamp = stuck_timestamp
        self.stuck_duration = stuck_duration
        self.stuck_frame_count = stuck_frame_count
        self.threshold = threshold

    def __str__(self):
        return f"TimestampStuckError: {super().__str__()}"


class KuavoMsgProcesser:
    def __init__(self, resize_w, resize_h):
        self.RESIZE_W = resize_w
        self.RESIZE_H = resize_h

    @staticmethod
    def process_joint_q_state(msg):
        joint_q = msg.joint_data.joint_q
        return {"data": joint_q, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_v_state(msg):
        joint_v = msg.joint_data.joint_v
        return {"data": joint_v, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_vd_state(msg):
        joint_vd = msg.joint_data.joint_vd
        return {"data": joint_vd, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_current_state(msg):
        joint_current = msg.joint_data.joint_current
        return {"data": joint_current, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_torque_state(msg):
        joint_torque = msg.joint_data.joint_torque
        return {"data": joint_torque, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_sensors_data_raw_extract_imu(msg):
        imu_data = msg.imu_data
        imu = np.array(
            [
                imu_data.gyro.x,
                imu_data.gyro.y,
                imu_data.gyro.z,
                imu_data.acc.x,
                imu_data.acc.y,
                imu_data.acc.z,
                imu_data.free_acc.x,
                imu_data.free_acc.y,
                imu_data.free_acc.z,
                imu_data.quat.x,
                imu_data.quat.y,
                imu_data.quat.z,
                imu_data.quat.w,
            ]
        )
        return {"data": imu, "timestamp": msg.header.stamp.to_sec()}

    # /kuavo_arm_traj

    @staticmethod
    def process_kuavo_arm_traj(msg):
        return {
            "data": np.deg2rad(msg.position),
            "timestamp": msg.header.stamp.to_sec(),
        }

    # /joint_cmd
    @staticmethod
    def process_joint_cmd_joint_q(msg):
        return {"data": msg.joint_q, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd_joint_v(msg):
        return {"data": msg.joint_v, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd_tau(msg):
        return {"data": msg.tau, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd_tau_max(msg):
        return {"data": msg.tau_max, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd_tau_ratio(msg):
        return {"data": msg.tau_ratio, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd_joint_kp(msg):
        return {"data": msg.joint_kp, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd_joint_kd(msg):
        return {"data": msg.joint_kd, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint_cmd_control_modes(msg):
        return {"data": msg.control_modes, "timestamp": msg.header.stamp.to_sec()}

    # /control_robot_hand_position_state

    @staticmethod
    def process_qiangnao_state(msg):
        state = list(msg.left_hand_position) + list(msg.right_hand_position)
        return {"data": state, "timestamp": msg.header.stamp.to_sec()}

    # /control_robot_hand_position
    @staticmethod
    def process_qiangnao_cmd(msg):
        position = list(msg.left_hand_position) + list(msg.right_hand_position)
        return {"data": position, "timestamp": msg.header.stamp.to_sec()}

    # /leju_claw_state
    @staticmethod
    def process_claw_state(msg):
        state = msg.data.position
        return {"data": state, "timestamp": msg.header.stamp.to_sec()}

    # /leju_claw_command
    @staticmethod
    def process_claw_cmd(msg):
        return {"data": msg.data.position, "timestamp": msg.header.stamp.to_sec()}

    def process_color_image(self, msg):
        # 只返回原始bytes
        img_bytes = bytes(msg.data)
        return {"data": img_bytes, "timestamp": msg.header.stamp.to_sec()}

    def process_depth_image(self, msg):
        img_bytes = bytes(msg.data)
        return {
            "data": img_bytes,
            "timestamp": msg.header.stamp.to_sec(),
            "compressed": True,
        }

    def process_depth_image_16U(self, msg):
        img_bytes = bytes(msg.data)
        return {
            "data": img_bytes,
            "timestamp": msg.header.stamp.to_sec(),
            "compressed": False,
        }

    @staticmethod
    def process_camera_metadata(msg):
        return {"data": msg.json_data, "timestamp": msg.header.stamp.to_sec()}

        # /color/metadata

    @staticmethod
    def process_camera_info(msg):
        distortion_model = msg.distortion_model
        D = np.array(msg.D)  # 畸变参数
        K = np.array(msg.K)  # 相机内参矩阵
        R = np.array(msg.R)  # 旋转矩阵
        P = np.array(msg.P)  # 投影矩阵

        # 拼接成一个向量
        # 顺序: 畸变参数D + 内参K（展平） + 旋转矩阵R（展平） + 投影矩阵P（展平）
        camera_vec = np.concatenate(
            [
                D.ravel(),  # 展平畸变参数数组
                K.ravel(),  # 展平内参矩阵
                R.ravel(),  # 展平旋转矩阵
                P.ravel(),  # 展平投影矩阵
            ]
        )
        # log_print("+" * 20,camera_vec.shape, "camera_vec")

        return {
            "data": camera_vec,
            "distortion_model": distortion_model,
            "timestamp": msg.header.stamp.to_sec(),
        }


