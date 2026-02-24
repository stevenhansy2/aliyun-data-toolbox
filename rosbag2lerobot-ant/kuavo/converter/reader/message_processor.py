"""ROS message processors for Kuavo bag conversion."""

import cv2
import numpy as np

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

    @staticmethod
    def process_joint_state_position(msg):
        try:
            pos = list(msg.position)
        except Exception:
            pos = []
        return {"data": pos, "timestamp": msg.header.stamp.to_sec()}

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

    @staticmethod
    def process_wrench_stamped(msg):
        try:
            f = msg.wrench.force
            t = msg.wrench.torque
            data = [f.x, f.y, f.z, t.x, t.y, t.z]
        except Exception:
            data = []
        return {"data": data, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_touch_matrix_pc2(msg):
        try:
            data = list(msg.data)
        except Exception:
            data = []
        return {"data": data, "timestamp": msg.header.stamp.to_sec()}

    def process_depth_image_16U(self, msg):
        img_bytes = bytes(msg.data)
        return {
            "data": img_bytes,
            "timestamp": msg.header.stamp.to_sec(),
            "compressed": False,
        }

    def process_depth(self, msg):
        if not (hasattr(msg, "format") and hasattr(msg, "data")):
            print(f"Skipping invalid message")

        # print(f"message format: {msg.format}")

        png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        idx = msg.data.find(png_magic)
        if idx == -1:
            print("PNG header not found, unable to decode.")
            return None

        png_data = msg.data[idx:]
        np_arr = np.frombuffer(png_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("cv2.imdecode also failed")
            return None

        if image.dtype != np.uint16:
            print(
                "Warning: The decoded image is not a 16-bit image, actual dtype: ",
                image.dtype,
            )
        depth_image = cv2.resize(
            image, (self.RESIZE_W, self.RESIZE_H), interpolation=cv2.INTER_NEAREST
        )

        return {
            "data": depth_image[np.newaxis, ...],
            "timestamp": msg.header.stamp.to_sec(),
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
        # print("+" * 20,camera_vec.shape, "camera_vec")

        return {
            "data": camera_vec,
            "distortion_model": distortion_model,
            "timestamp": msg.header.stamp.to_sec(),
        }


from dataclasses import dataclass, field

