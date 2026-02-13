#!/usr/bin/env python3
"""
优化了数据集对齐逻辑。

"""
import numpy as np
import cv2
import rosbag
from pprint import pprint
import os
import glob
from collections import defaultdict
import yaml
import time
from std_msgs.msg import Float64MultiArray
from config_dataset_slave import Config, load_config_from_json
from kuavo_pose_calculator import (
    extract_camera_poses_from_bag,
    extract_camera_poses_from_bag_with_time,
)


# ================ 机器人关节信息定义 ================
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


class KuavoRosbagReader:
    def __init__(self, config):
        self.DEFAULT_CAMERA_NAMES = config.default_camera_names
        self.TRAIN_HZ = config.train_hz
        self.MAIN_TIMELINE_FPS = config.main_timeline_fps
        self.SAMPLE_DROP = config.sample_drop
        self._msg_processer = KuavoMsgProcesser(
            config.resize.width, config.resize.height
        )
        self.TOPICS = config.topics
        self.EEF_TYPE = config.eef_type
        self.HAND_STATE_TOPICS = [
            "/control_robot_hand_position_state",
            "/dexhand/state",
        ]
        self.MAIN_TIMESTAMP_TOPIC = "head_cam_h"
        self.TIME_TOLERANCE = 180
        # 先不构建main_topic_map，等到process_rosbag时再构建
        self.main_topic_map = None

        # 动态构建topic处理映射
        self._topic_process_map = {}

        for camera in self.DEFAULT_CAMERA_NAMES:
            # 彩色图像（保持原有逻辑）
            color_topic = f"/{camera[-5:]}/color/image_raw/compressed"
            if color_topic in self.TOPICS:
                self._topic_process_map[f"{camera}"] = {
                    "topic": color_topic,
                    "msg_process_fn": self._msg_processer.process_color_image,
                }
                camera_info_topic = f"/{camera[-5:]}/color/camera_info"
                if camera_info_topic in self.TOPICS:
                    self._topic_process_map[f"{camera}_camera_info"] = {
                        "topic": camera_info_topic,
                        "msg_process_fn": self._msg_processer.process_camera_info,
                    }

            # 深度图像（优先未压缩）
            if "wrist" in camera:
                depth_topic_uncompressed = (
                    f"/{camera[-5:]}/depth/image_rect_raw/compressedDepth"
                )
                depth_topic_compressed = (
                    f"/{camera[-5:]}/depth/image_rect_raw/compressed"
                )
            else:  # head_cam_h
                depth_topic_uncompressed = (
                    f"/{camera[-5:]}/depth/image_raw/compressedDepth"
                )
                depth_topic_compressed = f"/{camera[-5:]}/depth/image_raw/compressed"

            if depth_topic_uncompressed in self.TOPICS:
                print(f"[INFO] {camera}: 选择未压缩深度话题 {depth_topic_uncompressed}")
                self._topic_process_map[f"{camera}_depth"] = {
                    "topic": depth_topic_uncompressed,
                    "msg_process_fn": self._msg_processer.process_depth_image_16U,
                    "fallback_topic": depth_topic_compressed,
                    "fallback_fn": self._msg_processer.process_depth_image,
                }
            elif depth_topic_compressed in self.TOPICS:
                print(f"[INFO] {camera}: 仅找到压缩深度话题 {depth_topic_compressed}")
                self._topic_process_map[f"{camera}_depth"] = {
                    "topic": depth_topic_compressed,
                    "msg_process_fn": self._msg_processer.process_depth_image,
                }
            else:
                print(f"[WARN] {camera} 未找到深度话题（未压缩或压缩）")

    def _find_actual_hand_state_topic(self, bag_file):
        """自动检测实际存在的手状态话题"""
        import rosbag

        bag = rosbag.Bag(bag_file)
        bag_topics = set([t for t in bag.get_type_and_topic_info().topics])
        bag.close()
        for t in self.HAND_STATE_TOPICS:
            if t in bag_topics:
                return t
        return None

    def _test_joint_current_availability(self, bag_file):
        """测试bag文件中的/sensors_data_raw消息是否有joint_current字段"""
        try:
            import rosbag

            bag = rosbag.Bag(bag_file)

            # 读取第一条/sensors_data_raw消息进行测试
            for topic, msg, t in bag.read_messages(topics=["/sensors_data_raw"]):
                try:
                    # 尝试访问joint_current字段
                    _ = msg.joint_data.joint_current
                    bag.close()
                    return True  # 有joint_current字段
                except AttributeError:
                    try:
                        # 尝试访问joint_torque字段
                        _ = msg.joint_data.joint_torque
                        bag.close()
                        return False  # 没有joint_current，但有joint_torque
                    except AttributeError:
                        bag.close()
                        return False  # 都没有，默认使用joint_torque
                # 只测试第一条消息
                break

            bag.close()
            return False  # 没有找到消息，默认使用joint_torque

        except Exception as e:
            print(f"测试joint_current可用性时出错: {e}")
            return False  # 出错时默认使用joint_torque

    def extract_and_format_camera_extrinsics(
        self, bag_file, abs_start=None, abs_end=None
    ):
        """提取并格式化相机外参，支持时间裁剪"""
        urdf_path = getattr(self, "urdf_path", None)
        if (
            urdf_path is None
            and hasattr(self, "config")
            and hasattr(self.config, "urdf_path")
        ):
            urdf_path = self.config.urdf_path
        if urdf_path is None:
            urdf_path = "./kuavo/biped_s45.urdf"  # 默认路径

        # 检查话题
        bag = self.load_raw_rosbag(bag_file)
        bag_topics = set([t for t in bag.get_type_and_topic_info().topics])
        bag.close()
        if "/sensors_data_raw" not in bag_topics:
            return {}

        # 新增：带时间裁剪的外参提取
        camera_poses = extract_camera_poses_from_bag_with_time(
            bag_file, urdf_path, abs_start, abs_end
        )

        def rigid_to_dict(rigid, ts):
            return {
                "rotation_matrix": rigid.rotation().matrix().tolist(),
                "translation_vector": rigid.translation().tolist(),
                "timestamp": ts,
            }

        return {
            "head_camera_extrinsics": [
                rigid_to_dict(pose, ts)
                for pose, ts in zip(
                    camera_poses["head_camera_poses"], camera_poses["timestamps"]
                )
            ],
            "left_hand_camera_extrinsics": [
                rigid_to_dict(pose, ts)
                for pose, ts in zip(
                    camera_poses["left_hand_camera_poses"], camera_poses["timestamps"]
                )
            ],
            "right_hand_camera_extrinsics": [
                rigid_to_dict(pose, ts)
                for pose, ts in zip(
                    camera_poses["right_hand_camera_poses"], camera_poses["timestamps"]
                )
            ],
        }

    def _build_main_topic_map(self, bag_file):
        # ...existing code...
        use_joint_current = self._test_joint_current_availability(bag_file)
        if use_joint_current:
            joint_current_processor = self._msg_processer.process_joint_current_state
            print("使用 joint_current 话题")
        else:
            joint_current_processor = self._msg_processer.process_joint_torque_state
            print("使用 joint_torque 话题")

        # 自动适配手状态话题
        actual_hand_state_topic = self._find_actual_hand_state_topic(bag_file)
        if actual_hand_state_topic:
            print(f"使用手部状态话题: {actual_hand_state_topic}")
        else:
            print("[WARN] 未找到手部状态话题，将不会读取手部状态数据。")
        main_topic_map = {
            "/sensors_data_raw": [
                (
                    "observation.sensorsData.joint_q",
                    self._msg_processer.process_joint_q_state,
                ),
                (
                    "observation.sensorsData.joint_v",
                    self._msg_processer.process_joint_v_state,
                ),
                (
                    "observation.sensorsData.joint_vd",
                    self._msg_processer.process_joint_vd_state,
                ),
                ("observation.sensorsData.joint_current", joint_current_processor),
                (
                    "observation.sensorsData.imu",
                    self._msg_processer.process_sensors_data_raw_extract_imu,
                ),
            ],
            "/kuavo_arm_traj": [
                ("action.kuavo_arm_traj", self._msg_processer.process_kuavo_arm_traj),
            ],
            "/joint_cmd": [
                (
                    "action.joint_cmd.joint_q",
                    self._msg_processer.process_joint_cmd_joint_q,
                ),
                (
                    "action.joint_cmd.joint_v",
                    self._msg_processer.process_joint_cmd_joint_v,
                ),
                ("action.joint_cmd.tau", self._msg_processer.process_joint_cmd_tau),
                (
                    "action.joint_cmd.tau_max",
                    self._msg_processer.process_joint_cmd_tau_max,
                ),
                (
                    "action.joint_cmd.tau_ratio",
                    self._msg_processer.process_joint_cmd_tau_ratio,
                ),
                (
                    "action.joint_cmd.tau_joint_kp",
                    self._msg_processer.process_joint_cmd_joint_kp,
                ),
                (
                    "action.joint_cmd.tau_joint_kd",
                    self._msg_processer.process_joint_cmd_joint_kd,
                ),
                (
                    "action.joint_cmd.control_modes",
                    self._msg_processer.process_joint_cmd_control_modes,
                ),
            ],
            # "/control_robot_hand_position_state": [
            #     ("observation.qiangnao", self._msg_processer.process_qiangnao_state),
            # ],
            "/control_robot_hand_position": [
                ("action.qiangnao", self._msg_processer.process_qiangnao_cmd),
            ],
            "/leju_claw_state": [
                ("observation.claw", self._msg_processer.process_claw_state),
            ],
            "/leju_claw_command": [
                ("action.claw", self._msg_processer.process_claw_cmd),
            ],
        }

        # 动态添加手状态话题
        if actual_hand_state_topic == "/control_robot_hand_position_state":
            main_topic_map[actual_hand_state_topic] = [
                ("observation.qiangnao", self._msg_processer.process_qiangnao_state),
            ]
        elif actual_hand_state_topic == "/dexhand/state":
            # 新格式，直接读取12维数组
            def process_dexhand_state(msg):
                # 假设msg.position为12维数组
                return {
                    "data": list(msg.position),
                    "timestamp": msg.header.stamp.to_sec(),
                }

            main_topic_map[actual_hand_state_topic] = [
                ("observation.qiangnao", process_dexhand_state),
            ]
        return main_topic_map

    def load_raw_rosbag(self, bag_file: str):
        return rosbag.Bag(bag_file)

    def print_bag_info(self, bag: rosbag.Bag):
        pprint(bag.get_type_and_topic_info().topics)

    def process_rosbag(
        self,
        bag_file: str,
        start_time: float = 0,
        end_time: float = 1,
        action_config=None,
    ):
        import gc
        import rospy

        # 如果还没有构建main_topic_map，先构建它
        if self.main_topic_map is None:
            self.main_topic_map = self._build_main_topic_map(bag_file)
            # 适配手状态话题
            actual_hand_state_topic = None
            for t in self.HAND_STATE_TOPICS:
                if t in self.main_topic_map:
                    actual_hand_state_topic = t
                    break
            for topic in self.TOPICS:
                # 适配手状态话题
                if topic in self.HAND_STATE_TOPICS:
                    if (
                        actual_hand_state_topic
                        and actual_hand_state_topic in self.main_topic_map
                    ):
                        for key, fn in self.main_topic_map[actual_hand_state_topic]:
                            self._topic_process_map[key] = {
                                "topic": actual_hand_state_topic,
                                "msg_process_fn": fn,
                            }
                elif topic in self.main_topic_map:
                    for key, fn in self.main_topic_map[topic]:
                        self._topic_process_map[key] = {
                            "topic": topic,
                            "msg_process_fn": fn,
                        }

        bag = self.load_raw_rosbag(bag_file)
        data = {}

        # Get bag start time and duration
        bag_start = bag.get_start_time()
        bag_end = bag.get_end_time()
        bag_duration = bag_end - bag_start

        # Calculate absolute start/end times
        abs_start = bag_start + start_time * bag_duration
        abs_end = bag_start + end_time * bag_duration

        print(f"开始处理 bag 文件: {bag_file}")
        print(f"话题数量: {len(self._topic_process_map)}")

        # 分话题处理，每处理完一个话题立即回收内存
        processed_topics = 0
        total_topics = len(self._topic_process_map)

        for key, topic_info in self._topic_process_map.items():
            topic = topic_info["topic"]
            msg_process_fn = topic_info["msg_process_fn"]
            data[key] = []

            print(f"[{processed_topics+1}/{total_topics}] 处理话题: {topic} -> {key}")

            # 临时存储消息数据
            temp_messages = []

            # 先尝试读取主 topic
            frame_count = 0
            for _, msg, t in bag.read_messages(
                topics=[topic],  # 只读取当前话题
                start_time=rospy.Time.from_sec(abs_start),
                end_time=rospy.Time.from_sec(abs_end),
            ):
                msg_data = msg_process_fn(msg)
                correct_timestamp = t.to_sec()
                msg_data["timestamp"] = correct_timestamp
                temp_messages.append(msg_data)
                frame_count += 1

            # 将临时消息数据转移到data中
            data[key] = temp_messages
            # 立即删除临时变量
            del temp_messages

            # 如果是深度话题且没读到消息，且有 fallback，尝试降级
            if (
                len(data[key]) == 0
                and "fallback_topic" in topic_info
                and "fallback_fn" in topic_info
            ):
                print(
                    f"  [WARN] {topic} 未读取到数据，尝试降级到 {topic_info['fallback_topic']}"
                )
                fallback_messages = []
                fallback_count = 0

                for _, msg, t in bag.read_messages(
                    topics=[topic_info["fallback_topic"]],  # 只读取fallback话题
                    start_time=rospy.Time.from_sec(abs_start),
                    end_time=rospy.Time.from_sec(abs_end),
                ):
                    msg_data = topic_info["fallback_fn"](msg)
                    correct_timestamp = t.to_sec()
                    msg_data["timestamp"] = correct_timestamp
                    fallback_messages.append(msg_data)
                    fallback_count += 1

                # 使用fallback数据
                data[key] = fallback_messages
                del fallback_messages

            processed_topics += 1
            print(f"  完成: {key} ({len(data[key])} 帧)")

            # 每处理3个话题进行一次内存回收并删除临时变量
            if processed_topics % 3 == 0:
                # 删除可能的临时变量
                if "msg_data" in locals():
                    del msg_data
                if "correct_timestamp" in locals():
                    del correct_timestamp
                gc.collect()
                print(f"  [内存回收] 已处理 {processed_topics}/{total_topics} 个话题")

        # 2. 立即关闭bag文件释放资源
        bag.close()
        del bag
        # 删除时间相关变量
        gc.collect()
        print(f"✅ bag文件已关闭，基础数据读取完成")

        # 3. 提取相机外参（可能消耗较多内存）
        print("📐 开始提取相机外参...")
        extrinsics = self.extract_and_format_camera_extrinsics(
            bag_file, abs_start, abs_end
        )
        data.update(extrinsics)

        # 清理外参处理中的临时变量
        del extrinsics
        gc.collect()
        print("✅ 相机外参提取完成")

        # 4. 新增：末端执行器位姿计算
        print("🔧 开始计算末端执行器位姿...")
        joint_q_items = data.get("observation.sensorsData.joint_q", [])
        if joint_q_items:
            from endeffector_pose_from_bag import extract_and_format_eef_extrinsics

            joint_q_list = [item["data"] for item in joint_q_items]
            timestamps = [item["timestamp"] for item in joint_q_items]

            positions, quaternions = extract_and_format_eef_extrinsics(
                [{"joint_q": q} for q in joint_q_list],
                urdf_path="./kuavo/biped_s49.urdf",
            )

            # 组装为 [{data: ..., timestamp: ...}, ...]
            data["end.position"] = [
                {"data": positions[i], "timestamp": timestamps[i]}
                for i in range(len(positions))
            ]
            data["end.orientation"] = [
                {"data": quaternions[i], "timestamp": timestamps[i]}
                for i in range(len(quaternions))
            ]

            # 清理末端执行器计算的临时变量
            del joint_q_list, timestamps, positions, quaternions, joint_q_items
            gc.collect()
            print("✅ 末端执行器位姿计算完成")
        else:
            data["end.position"] = []
            data["end.orientation"] = []
            del joint_q_items

        # 5. 计算总体参数
        total_cut_frames = int(
            round((end_time - start_time) * bag_duration * self.TRAIN_HZ)
        )
        drop_head = (
            self.SAMPLE_DROP
            if (start_time * bag_duration * self.TRAIN_HZ) <= self.SAMPLE_DROP
            else 0
        )
        drop_tail = (
            self.SAMPLE_DROP
            if ((1 - end_time) * bag_duration * self.TRAIN_HZ) <= self.SAMPLE_DROP
            else 0
        )

        print(f"🔄 开始时间戳对齐处理...")
        print(f"  预估帧数: {total_cut_frames}")
        print(f"  头部丢弃: {drop_head}, 尾部丢弃: {drop_tail}")

        # 删除不再需要的参数变量
        del total_cut_frames, start_time, end_time

        # 6. 执行对齐（最消耗内存的步骤）
        aligned_data = self.align_frame_data_optimized(
            data, drop_head, drop_tail, action_config=action_config
        )

        # 7. 立即清理原始数据和对齐参数
        del data, drop_head, drop_tail, action_config
        # 删除循环中可能残留的变量
        if "key" in locals():
            del key
        if "topic_info" in locals():
            del topic_info
        if "topic" in locals():
            del topic
        if "msg_process_fn" in locals():
            del msg_process_fn
        if "frame_count" in locals():
            del frame_count
        if "fallback_count" in locals():
            del fallback_count
        if "processed_topics" in locals():
            del processed_topics
        if "total_topics" in locals():
            del total_topics

        gc.collect()
        print("✅ 原始数据已清理，时间戳对齐完成")

        return aligned_data

    def find_closest_indices_vectorized(self, timestamps, target_timestamps):
        """向量化查找最近时间戳索引"""
        timestamps = np.array(timestamps)
        target_timestamps = np.array(target_timestamps)

        # 使用 searchsorted 进行高效查找
        indices = np.searchsorted(timestamps, target_timestamps)

        # 处理边界情况
        indices = np.clip(indices, 0, len(timestamps) - 1)

        # 检查左右邻居，选择更近的
        valid_left = indices > 0
        left_indices = np.where(valid_left, indices - 1, indices)

        left_diffs = np.abs(timestamps[left_indices] - target_timestamps)
        right_diffs = np.abs(timestamps[indices] - target_timestamps)

        # 选择距离更近的索引
        closer_indices = np.where(left_diffs < right_diffs, left_indices, indices)

        return closer_indices

    def _preprocess_timestamps_only_deduplicate(self, data: dict) -> dict:
        """预处理时间戳和数据：只去重和检测卡顿，不插值（按需插值策略）"""
        preprocessed_data = {}

        for key, data_list in data.items():
            if len(data_list) == 0:
                preprocessed_data[key] = []
                continue

            print(f"预处理 {key}: 原始长度 {len(data_list)}")

            # 步骤1: 去除重复时间戳
            deduplicated_data = self._remove_duplicate_timestamps(data_list, key)

            # 步骤2: 检测去重后数据的实际时间间隔卡顿（更准确）
            self._check_actual_time_gaps(
                deduplicated_data, key, max_gap_duration=self.TIME_TOLERANCE
            )

            # 步骤3: 跳过插值（按需插值策略）
            preprocessed_data[key] = deduplicated_data
            print(f"预处理 {key}: 去重后 {len(deduplicated_data)} 帧（未插值）")

        return preprocessed_data

    def _preprocess_timestamps_and_data(self, data: dict) -> dict:
        """预处理时间戳和数据：去重、检测实际卡顿、插值"""
        preprocessed_data = {}

        for key, data_list in data.items():
            if len(data_list) == 0:
                preprocessed_data[key] = []
                continue

            print(f"预处理 {key}: 原始长度 {len(data_list)}")

            # 步骤1: 去除重复时间戳
            deduplicated_data = self._remove_duplicate_timestamps(data_list, key)

            # 步骤2: 检测去重后数据的实际时间间隔卡顿（更准确）
            self._check_actual_time_gaps(
                deduplicated_data, key, max_gap_duration=self.TIME_TOLERANCE
            )

            # 步骤3: 时间戳插值和数据填充
            interpolated_data = self._interpolate_timestamps_and_data(
                deduplicated_data, key
            )

            preprocessed_data[key] = interpolated_data
            print(
                f"预处理 {key}: 去重后 {len(deduplicated_data)}, 插值后 {len(interpolated_data)}"
            )

        return preprocessed_data

    def _check_actual_time_gaps(
        self, data_list: list, key: str, max_gap_duration: float = 2.0
    ):
        """检测去重后数据的实际时间间隔卡顿"""
        if len(data_list) <= 1:
            return

        timestamps_seconds = np.array([item["timestamp"] for item in data_list])
        timestamps_ns = (timestamps_seconds * 1e9).astype(np.int64)

        # 计算实际时间间隔
        time_diffs_ns = np.diff(timestamps_ns)
        time_diffs_seconds = time_diffs_ns / 1e9

        # 找出超过阈值的时间间隔
        large_gaps = time_diffs_seconds > max_gap_duration

        if np.any(large_gaps):
            max_gap_seconds = np.max(time_diffs_seconds)
            gap_indices = np.where(large_gaps)[0]

            error_msg = (
                f"时间间隔卡顿检测：{key} 话题存在 {len(gap_indices)} 个超过{max_gap_duration}s的时间间隔，"
                f"最大间隔 {max_gap_seconds:.3f}s，数据质量异常，终止处理"
            )
            print(f"[ERROR] {error_msg}")

            # 显示具体的问题间隔
            for i, gap_idx in enumerate(gap_indices[:3]):  # 只显示前3个
                start_time = timestamps_seconds[gap_idx]
                end_time = timestamps_seconds[gap_idx + 1]
                gap_duration = time_diffs_seconds[gap_idx]
                print(
                    f"  间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={gap_duration:.3f}s"
                )

            raise TimestampStuckError(
                message=error_msg,
                topic=key,
                stuck_timestamp=timestamps_seconds[gap_indices[0]],
                stuck_duration=max_gap_seconds,
                stuck_frame_count=len(gap_indices),
                threshold=max_gap_duration,
            )
        else:
            max_gap_seconds = (
                np.max(time_diffs_seconds) if len(time_diffs_seconds) > 0 else 0
            )
            print(f"  {key}: ✓ 时间间隔正常，最大间隔 {max_gap_seconds:.3f}s")

    def _remove_duplicate_timestamps(self, data_list: list, key: str) -> list:
        """去除重复时间戳及对应数据（使用纳秒精度）"""
        if len(data_list) <= 1:
            return data_list

        deduplicated = []
        seen_timestamps = set()
        duplicate_count = 0

        for item in data_list:
            timestamp_seconds = item["timestamp"]
            # 转换为纳秒精度避免浮点精度问题
            timestamp_ns = int(timestamp_seconds * 1e9)

            if timestamp_ns not in seen_timestamps:
                seen_timestamps.add(timestamp_ns)
                deduplicated.append(item)
            else:
                duplicate_count += 1

        if duplicate_count > 0:
            print(f"  {key}: 删除 {duplicate_count} 个重复时间戳")

        return deduplicated

    def _interpolate_timestamps_and_data(self, data_list: list, key: str) -> list:
        """时间戳插值和数据填充（修复版本 - 严格控制间隔，超过2秒直接抛异常）"""
        if len(data_list) <= 1:
            return data_list

        timestamps_seconds = np.array([item["timestamp"] for item in data_list])
        timestamps_ns = (timestamps_seconds * 1e9).astype(np.int64)

        # 首先检查是否有超过2秒的间隔，如果有直接抛出异常
        time_diffs_ns = np.diff(timestamps_ns)
        time_diffs_seconds = time_diffs_ns / 1e9

        max_gap_seconds = np.max(time_diffs_seconds)
        large_gaps_2s = time_diffs_seconds > self.TIME_TOLERANCE  # 2秒阈值

        if np.any(large_gaps_2s):
            gap_indices = np.where(large_gaps_2s)[0]
            error_msg = (
                f"插值阶段发现严重时间间隔：{key} 话题存在 {len(gap_indices)} 个超过{self.TIME_TOLERANCE}s的时间间隔，"
                f"最大间隔 {max_gap_seconds:.3f}s，数据质量异常，终止处理"
            )
            print(f"[ERROR] {error_msg}")

            # 显示具体的问题间隔
            for i, gap_idx in enumerate(gap_indices[:3]):  # 只显示前3个
                start_time = timestamps_seconds[gap_idx]
                end_time = timestamps_seconds[gap_idx + 1]
                gap_duration = time_diffs_seconds[gap_idx]
                print(
                    f"  严重间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={gap_duration:.3f}s"
                )

            raise TimestampStuckError(
                message=error_msg,
                topic=key,
                stuck_timestamp=timestamps_seconds[gap_indices[0]],
                stuck_duration=max_gap_seconds,
                stuck_frame_count=len(gap_indices),
                threshold=2.0,
            )

        # 确定插值间隔（纳秒）
        if any(cam in key for cam in ["head_cam"]) and "depth" not in key:
            # 彩色视频：33ms间隔 (30fps)
            target_interval_ns = int(32 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(39.8 * 1e6)  # 37ms最大允许间隔
            data_type = "video"
        elif any(cam in key for cam in ["wrist_cam"]) and "depth" not in key:
            # 彩色视频：33ms间隔 (30fps)
            target_interval_ns = int(32 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(8 * 1e6)  # 38ms最大允许间隔
            data_type = "video"
        elif "depth" in key:
            # 深度视频：33ms间隔 (30fps)
            target_interval_ns = int(32 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(8 * 1e6)  # 38ms最大允许间隔
            data_type = "depth"

        else:
            # 传感器数据：5ms间隔 (100hz)
            target_interval_ns = int(10 * 1e6)  # 纳秒
            max_allowed_interval_ns = int(4 * 1e6)  # 5ms最大允许间隔（传感器更严格）
            data_type = "sensor"

        # 检测需要插值的位置（使用更严格的阈值）
        interpolation_threshold_ns = (
            max_allowed_interval_ns  # 直接使用最大允许间隔作为阈值
        )

        large_gaps = time_diffs_ns > interpolation_threshold_ns

        if not np.any(large_gaps):
            # 无需插值
            print(f"  {key}: 无需插值，最大间隔 {np.max(time_diffs_ns)/1e6:.1f}ms")
            return data_list

        print(f"  {key}: 发现 {np.sum(large_gaps)} 个需要插值的时间间隔")
        print(
            f"  {key}: 目标间隔 {target_interval_ns/1e6:.1f}ms, 最大允许间隔 {max_allowed_interval_ns/1e6:.1f}ms"
        )

        # 构建插值后的数据
        interpolated_data = []

        for i in range(len(data_list)):
            # 添加当前数据点
            interpolated_data.append(data_list[i])

            # 检查是否需要在当前点和下一点之间插值
            if i < len(data_list) - 1 and large_gaps[i]:
                current_time_ns = timestamps_ns[i]
                next_time_ns = timestamps_ns[i + 1]
                gap_duration_ns = next_time_ns - current_time_ns
                gap_duration_seconds = gap_duration_ns / 1e9

                # 双重保险：再次检查间隔是否超过self.TIME_TOLERANCE
                if gap_duration_seconds > self.TIME_TOLERANCE:
                    error_msg = f"插值过程中发现超过{self.TIME_TOLERANCE}秒的间隔：{key} 在索引{i}处有{gap_duration_seconds:.3f}s间隔"
                    print(f"[ERROR] {error_msg}")
                    raise TimestampStuckError(
                        message=error_msg,
                        topic=key,
                        stuck_timestamp=current_time_ns / 1e9,
                        stuck_duration=gap_duration_seconds,
                        stuck_frame_count=1,
                        threshold=2.0,
                    )

                # print(f"    间隔{i}: {gap_duration_ns/1e6:.1f}ms 需要插值")

                # 计算需要插入多少个点来满足最大间隔要求
                num_segments_needed = int(
                    np.ceil(gap_duration_ns / max_allowed_interval_ns)
                )

                if num_segments_needed > 1:
                    # 需要插值
                    num_interpolations = num_segments_needed - 1

                    # 生成均匀分布的插值时间戳
                    interp_times_ns = np.linspace(
                        current_time_ns,
                        next_time_ns,
                        num_interpolations + 2,  # +2 包含起点和终点
                        dtype=np.int64,
                    )[
                        1:-1
                    ]  # 去掉起点和终点

                    # print(f"    插入 {len(interp_times_ns)} 个点，平均间隔 {gap_duration_ns/(num_interpolations+1)/1e6:.1f}ms")

                    # 插入数据点
                    for interp_time_ns in interp_times_ns:
                        interp_time_seconds = interp_time_ns / 1e9  # 转回秒
                        interpolated_item = self._create_interpolated_data_point(
                            data_list[i], interp_time_seconds, data_type
                        )
                        interpolated_data.append(interpolated_item)

        # 验证插值结果
        final_timestamps = np.array([item["timestamp"] for item in interpolated_data])
        final_timestamps_ns = (final_timestamps * 1e9).astype(np.int64)
        final_intervals_ns = np.diff(final_timestamps_ns)
        final_intervals_ms = final_intervals_ns / 1e6

        max_final_interval = np.max(final_intervals_ms)
        print(f"  {key}: 插值完成，最大间隔 {max_final_interval:.1f}ms")

        # # 最终检查：如果插值后仍然存在超过阈值的间隔，抛出异常
        # if max_final_interval > max_allowed_interval_ns / 1e6:
        #     problematic_indices = np.where(final_intervals_ms > max_allowed_interval_ns / 1e6)[0]
        #     error_msg = f"插值后验证失败：{key} 仍有 {len(problematic_indices)} 个间隔超过{max_allowed_interval_ns/1e6:.1f}ms阈值，最大间隔{max_final_interval:.1f}ms"
        #     print(f"[ERROR] {error_msg}")

        #     # 显示具体问题
        #     for idx in problematic_indices[:3]:  # 只显示前3个
        #         print(f"    问题间隔{idx}: {final_intervals_ms[idx]:.1f}ms")

        #     raise TimestampStuckError(
        #         message=f"插值后质量验证失败: {error_msg}",
        #         topic=key,
        #         stuck_timestamp=final_timestamps[problematic_indices[0]],
        #         stuck_duration=max_final_interval/1000,
        #         stuck_frame_count=len(problematic_indices),
        #         threshold=max_allowed_interval_ns/1e6/1000  # 转换为秒
        #     )

        return interpolated_data

    def _interpolate_on_demand(
        self,
        aligned_data: list,
        time_errors_ms: np.ndarray,
        original_data_list: list,
        original_timestamps: np.ndarray,
        target_timestamps: np.ndarray,
        key: str,
    ) -> list:
        """
        按需插值：只对误差 >10ms 的帧进行插值修正

        策略：
        1. 如果原始数据中找不到 <10ms 的帧
        2. 尝试在相邻帧之间找更近的（复制最近帧）
        3. 如果仍然无法满足 <10ms，保持原选择并记录警告
        """
        interpolated_count = 0
        created_count = 0

        # 判断数据类型
        data_type = (
            "depth"
            if "depth" in key
            else (
                "video"
                if any(cam in key for cam in ["head_cam", "wrist_cam"])
                else "sensor"
            )
        )

        for i, error_ms in enumerate(time_errors_ms):
            if error_ms <= 10:  # 误差可接受（<10ms）
                continue

            target_ts = target_timestamps[i]

            # 策略1: 在原始数据中查找能满足 <10ms 的帧
            time_diffs = np.abs(original_timestamps - target_ts) * 1000
            better_candidates = np.where(time_diffs < 10)[0]

            if len(better_candidates) > 0:
                # 找到了更好的原始帧
                best_idx = better_candidates[np.argmin(time_diffs[better_candidates])]
                aligned_data[i] = original_data_list[best_idx]
                interpolated_count += 1
            else:
                # 策略2: 找不到 <10ms 的原始帧，创建插值帧
                # 找到最近的原始帧作为参考
                closest_idx = np.argmin(time_diffs)
                reference_frame = original_data_list[closest_idx]

                # 创建插值帧（复制最近帧，只改时间戳）
                interpolated_frame = self._create_interpolated_data_point(
                    reference_frame, target_ts, data_type
                )

                aligned_data[i] = interpolated_frame
                created_count += 1

        if interpolated_count > 0:
            print(f"    [按需插值] 从原始数据选择了 {interpolated_count} 帧")
        if created_count > 0:
            print(f"    [按需插值] 创建了 {created_count} 个插值帧（复制最近帧）")

        return aligned_data

    def _create_interpolated_data_point(
        self, reference_item: dict, new_timestamp: float, data_type: str
    ) -> dict:
        """创建插值数据点"""
        interpolated_item = reference_item.copy()
        interpolated_item["timestamp"] = new_timestamp

        # 根据数据类型处理数据字段
        if data_type in ["video", "depth"]:
            # 图像数据：复制参考帧的数据
            interpolated_item["interpolated"] = True
        elif data_type == "sensor":
            # 传感器数据：保持相同的数值（零阶保持插值）
            interpolated_item["interpolated"] = True

        return interpolated_item

    def _validate_timestamp_quality(self, timestamps: np.ndarray, data_name: str):
        """验证时间戳质量（使用纳秒精度）- 增强版本"""
        if len(timestamps) <= 1:
            return

        # 转换为纳秒进行精确计算
        timestamps_ns = (timestamps * 1e9).astype(np.int64)
        time_diffs_ns = np.diff(timestamps_ns)
        time_diffs_ms = time_diffs_ns / 1e6  # 转换为毫秒显示

        # 检查时间间隔
        mean_interval_ms = np.mean(time_diffs_ms)
        max_interval_ms = np.max(time_diffs_ms)
        min_interval_ms = np.min(time_diffs_ms)
        std_interval_ms = np.std(time_diffs_ms)

        print(f"  {data_name} 时间戳质量:")
        print(f"    平均间隔: {mean_interval_ms:.1f}ms")
        print(f"    最大间隔: {max_interval_ms:.1f}ms")
        print(f"    最小间隔: {min_interval_ms:.1f}ms")
        print(f"    标准差: {std_interval_ms:.1f}ms")

        # 严格的质量检查 - 修改为更严格的验证
        critical_errors = []
        warnings = []

        if max_interval_ms > 40:  # 40ms阈值 - 关键错误
            critical_errors.append(f"最大时间间隔过大: {max_interval_ms:.1f}ms")

        if min_interval_ms < 0.1:  # 0.1ms阈值 - 关键错误
            critical_errors.append(f"最小时间间隔过小: {min_interval_ms:.1f}ms")

        if std_interval_ms > 15:  # 15ms标准差阈值 - 警告
            warnings.append(f"时间间隔波动过大: {std_interval_ms:.1f}ms")

        # 检查重复时间戳
        unique_timestamps = np.unique(timestamps_ns)
        if len(unique_timestamps) < len(timestamps_ns):
            duplicate_count = len(timestamps_ns) - len(unique_timestamps)
            critical_errors.append(f"仍存在 {duplicate_count} 个重复时间戳")

        # 输出结果
        if critical_errors:
            print(f"    ❌ 关键错误: {'; '.join(critical_errors)}")
            # 对于主时间戳的关键错误，抛出异常
            if data_name == "主时间戳":
                error_msg = (
                    f"{data_name} 存在关键质量问题: {'; '.join(critical_errors)}"
                )
                raise TimestampStuckError(
                    message=error_msg,
                    topic=data_name,
                    stuck_timestamp=None,
                    stuck_duration=max_interval_ms / 1000,
                    stuck_frame_count=len(critical_errors),
                    threshold=0.04,
                )
        elif warnings:
            print(f"    ⚠️  警告: {'; '.join(warnings)}")
        else:
            print(f"    ✓ 时间戳质量良好")

    def align_frame_data_optimized(
        self, data: dict, drop_head: bool, drop_tail: bool, action_config=None
    ):
        """优化的时间戳对齐函数，按需插值策略"""
        print("开始优化版本的时间戳对齐（按需插值）...")
        aligned_data = defaultdict(list)

        # 1. 预处理：只去重和检测卡顿，不插值
        print("步骤1: 预处理数据 - 去重和检测卡顿（非主时间线跳过插值）")
        preprocessed_data = {}

        for key, data_list in data.items():
            if len(data_list) == 0:
                preprocessed_data[key] = []
                continue

            print(f"预处理 {key}: 原始长度 {len(data_list)}")

            # 去重
            deduplicated_data = self._remove_duplicate_timestamps(data_list, key)

            # 检测卡顿
            self._check_actual_time_gaps(
                deduplicated_data, key, max_gap_duration=self.TIME_TOLERANCE
            )

            # 特殊处理：主时间线需要插值（填补丢帧，保证完整性）
            main_timeline = getattr(self, "MAIN_TIMESTAMP_TOPIC", "head_cam_h")
            if key == main_timeline:
                print(f"  [主时间线] 对 {key} 进行插值，填补丢帧")
                interpolated_data = self._interpolate_timestamps_and_data(
                    deduplicated_data, key
                )
                preprocessed_data[key] = interpolated_data
                print(
                    f"预处理 {key}: 去重后 {len(deduplicated_data)}, 插值后 {len(interpolated_data)} 帧（主时间线）"
                )
            else:
                # 其他话题：只去重，不插值（按需对齐）
                preprocessed_data[key] = deduplicated_data
                print(f"预处理 {key}: 去重后 {len(deduplicated_data)} 帧（未插值）")

        print(f"[优化] 主时间线已插值，其他话题保留原始数据")

        # 2. 生成统一的主时间戳基准
        main_timeline = getattr(self, "MAIN_TIMESTAMP_TOPIC", "head_cam_h")
        if (
            main_timeline not in preprocessed_data
            or len(preprocessed_data[main_timeline]) == 0
        ):
            main_timeline = max(
                self.DEFAULT_CAMERA_NAMES,
                key=lambda cam_k: len(preprocessed_data.get(cam_k, [])),
            )
            print(f"警告：主时间戳话题不存在，使用降级话题: {main_timeline}")

        # 3. 生成主时间戳序列
        jump = self.MAIN_TIMELINE_FPS // self.TRAIN_HZ
        main_img_timestamps = [t["timestamp"] for t in preprocessed_data[main_timeline]]

        # 根据传入参数裁剪首尾
        start_idx = self.SAMPLE_DROP if drop_head else 0
        end_idx = -self.SAMPLE_DROP if drop_tail else None
        main_img_timestamps = main_img_timestamps[start_idx:end_idx][::jump]

        # 4. 时间戳边界过滤
        data_with_content = {k: v for k, v in preprocessed_data.items() if len(v) > 0}
        if not data_with_content:
            return aligned_data

        min_end = min([data[k][-1]["timestamp"] for k in data_with_content.keys()])
        main_img_timestamps = [t for t in main_img_timestamps if t < min_end]
        main_img_timestamps = np.array(main_img_timestamps)

        print(f"主时间线: {main_timeline}, 预处理后长度: {len(main_img_timestamps)}")

        # 5. 多模态开头时间戳修正
        print("步骤5: 多模态开头时间戳修正")
        main_img_timestamps = self._fix_multimodal_start_alignment(
            main_img_timestamps, preprocessed_data
        )

        # 6. 验证时间戳质量
        # self._validate_timestamp_quality(main_img_timestamps, "主时间戳")

        # 7. 向量化对齐处理（先用原始数据对齐）
        print("步骤7: 对齐处理（先用原始数据）")
        for key, data_list in preprocessed_data.items():
            if len(data_list) == 0:
                aligned_data[key] = []
                continue

            timestamps = np.array([frame["timestamp"] for frame in data_list])
            closest_indices = self.find_closest_indices_vectorized(
                timestamps, main_img_timestamps
            )
            aligned_data[key] = [data_list[idx] for idx in closest_indices]

            # 验证对齐质量
            aligned_timestamps = timestamps[closest_indices]
            time_errors_ms = np.abs(aligned_timestamps - main_img_timestamps) * 1000
            max_diff = np.max(time_errors_ms)
            mean_diff = np.mean(time_errors_ms)

            # 统计误差分布
            errors_gt_10ms = np.sum(time_errors_ms > 10)
            errors_gt_15ms = np.sum(time_errors_ms > 15)
            errors_gt_20ms = np.sum(time_errors_ms > 20)

            print(f"  {key}: 对齐完成 {len(aligned_data[key])} 帧")
            print(f"    时间戳误差: 平均 {mean_diff:.1f}ms, 最大 {max_diff:.1f}ms")
            print(
                f"    误差分布: >10ms={errors_gt_10ms}, >15ms={errors_gt_15ms}, >20ms={errors_gt_20ms}"
            )

            # 按需插值：只对误差 >10ms 的帧进行插值修正
            if errors_gt_10ms > 0:
                print(
                    f"    [按需插值] 发现 {errors_gt_10ms} 帧误差 >10ms，进行插值修正"
                )
                aligned_data[key] = self._interpolate_on_demand(
                    aligned_data[key],
                    time_errors_ms,
                    data_list,
                    timestamps,
                    main_img_timestamps,
                    key,
                )
            else:
                print(f"    ✅ 所有帧误差 <10ms，无需插值")
        # === 新增步骤8: 静止区域检测和裁剪 ===
        print("步骤8: 静止区域检测和裁剪")
        # aligned_data, main_img_timestamps = self._detect_and_trim_aligned_data(aligned_data, main_img_timestamps,action_config=action_config)
        # === 新增步骤8: 帧率调整到30fps ===
        print("步骤8: 帧率调整到30fps")
        # aligned_data, main_img_timestamps = self._adjust_frame_rate_to_30fps(aligned_data, main_img_timestamps)

        # print("步骤9: 最终验证对齐质量")
        # self._final_alignment_validation(aligned_data, main_img_timestamps)

        return aligned_data

    def _detect_and_trim_aligned_data(
        self, aligned_data: dict, main_timestamps: np.ndarray, action_config=None
    ):
        """
        检测并裁剪对齐后数据中的静止区域，头尾裁剪上限由首尾动作持续帧数的一半决定
        """
        from slave_utils import (
            detect_stillness_from_image_data,
            analyze_stillness_frames,
        )

        motion_threshold = 4.5
        stillness_ratio = 1
        check_duration = 10.0
        fps = self.TRAIN_HZ or 30

        camera_keys = [
            c
            for c in self.DEFAULT_CAMERA_NAMES
            if c in aligned_data and len(aligned_data[c]) > 0
        ]
        if not camera_keys:
            print("  未找到有效的相机数据，跳过静止检测")
            return aligned_data, main_timestamps

        print(f"  基于 {len(camera_keys)} 个相机检测静止区域: {camera_keys}")

        # === 计算首尾动作的持续帧数 ===
        max_head_trim_limit = None
        max_tail_trim_limit = None
        total_frames = len(main_timestamps)
        if action_config and len(action_config) > 0:
            # moments.json格式，需从customFieldValues中提取start_position和end_position
            first_action = None
            last_action = None
            min_start = None
            max_end = None
            for act in action_config:
                custom_fields = act.get("customFieldValues", {})
                try:
                    sp = float(custom_fields.get("start_position", None))
                    if min_start is None or sp < min_start:
                        min_start = sp
                        first_action = act
                except Exception:
                    pass
                try:
                    ep = float(custom_fields.get("end_position", None))
                    if max_end is None or ep > max_end:
                        max_end = ep
                        last_action = act
                except Exception:
                    pass

            # 计算帧区间（直接用比例乘以总帧数）
            if first_action is not None and last_action is not None:
                first_sp = float(first_action["customFieldValues"]["start_position"])
                first_ep = float(first_action["customFieldValues"]["end_position"])
                last_sp = float(last_action["customFieldValues"]["start_position"])
                last_ep = float(last_action["customFieldValues"]["end_position"])

                # 按比例映射到帧索引
                first_start_idx = int(
                    round(
                        (first_sp - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )
                first_end_idx = int(
                    round(
                        (first_ep - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )
                last_start_idx = int(
                    round(
                        (last_sp - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )
                last_end_idx = int(
                    round(
                        (last_ep - min_start)
                        / (max_end - min_start)
                        * (total_frames - 1)
                    )
                )

                first_len = max(0, first_end_idx - first_start_idx)
                last_len = max(0, last_end_idx - last_start_idx)
                max_head_trim_limit = max(0, int(first_len / 2))
                max_tail_trim_limit = max(0, int(last_len / 2))
                print(
                    f"  首动作帧区间: {first_start_idx}-{first_end_idx}，长度: {first_len}"
                )
                print(
                    f"  尾动作帧区间: {last_start_idx}-{last_end_idx}，长度: {last_len}"
                )
                print(
                    f"  首动作最大裁剪上限: {max_head_trim_limit} 帧，尾动作最大裁剪上限: {max_tail_trim_limit} 帧"
                )
            else:
                print("  未找到有效的动作首尾裁剪上限")
        else:
            print("  未找到有效的动作首尾裁剪上限")

        # === 静止检测 ===
        all_stillness_results = {}
        for camera_key in camera_keys:
            frames_data = aligned_data[camera_key]
            print(f"  分析 {camera_key}: 总帧数 {len(frames_data)}")
            head_stillness, tail_stillness = detect_stillness_from_image_data(
                frames_data,
                camera_key,
                motion_threshold,
                stillness_ratio,
                check_duration,
                fps,
            )
            all_stillness_results[camera_key] = {
                "head_frames": head_stillness,
                "tail_frames": tail_stillness,
            }
            print(
                f"    {camera_key}: 开头静止 {head_stillness} 帧, 结尾静止 {tail_stillness} 帧"
            )

        # 计算最终裁剪帧数（取所有相机的最大值确保一致性）
        if all_stillness_results:
            max_head_trim = max(
                result["head_frames"] for result in all_stillness_results.values()
            )
            max_tail_trim = max(
                result["tail_frames"] for result in all_stillness_results.values()
            )
        else:
            max_head_trim = 0
            max_tail_trim = 0

        # === 应用首尾裁剪上限 ===
        if max_head_trim_limit is not None:
            if max_head_trim > max_head_trim_limit:
                print(
                    f"  开头静止裁剪帧数 {max_head_trim} 超过首动作上限 {max_head_trim_limit}，已覆盖"
                )
                max_head_trim = max_head_trim_limit
        if max_tail_trim_limit is not None:
            if max_tail_trim > max_tail_trim_limit:
                print(
                    f"  结尾静止裁剪帧数 {max_tail_trim} 超过尾动作上限 {max_tail_trim_limit}，已覆盖"
                )
                max_tail_trim = max_tail_trim_limit

        print(f"  最终裁剪决定: 开头 {max_head_trim} 帧, 结尾 {max_tail_trim} 帧")

        # 应用裁剪到所有数据
        if max_head_trim > 0 or max_tail_trim > 0:
            trimmed_aligned_data, trimmed_main_timestamps = (
                self._trim_aligned_data_by_frames(
                    aligned_data, main_timestamps, max_head_trim, max_tail_trim
                )
            )
            print(
                f"  裁剪完成: 主时间戳 {len(main_timestamps)} -> {len(trimmed_main_timestamps)} 帧"
            )
            return trimmed_aligned_data, trimmed_main_timestamps
        else:
            print("  无需裁剪")
            return aligned_data, main_timestamps

    def _trim_aligned_data_by_frames(
        self,
        aligned_data: dict,
        main_timestamps: np.ndarray,
        head_trim_frames: int,
        tail_trim_frames: int,
    ):
        """按帧数裁剪对齐后的数据"""
        trimmed_data = {}

        # 裁剪主时间戳
        original_length = len(main_timestamps)
        start_idx = head_trim_frames
        if tail_trim_frames > 0:
            end_idx = original_length - tail_trim_frames
        else:
            end_idx = original_length

        # 确保索引有效
        start_idx = max(0, start_idx)
        end_idx = min(original_length, end_idx)

        if start_idx < end_idx:
            trimmed_main_timestamps = main_timestamps[start_idx:end_idx]
        else:
            trimmed_main_timestamps = np.array([])
            print("    警告: 主时间戳裁剪后为空")

        # 裁剪所有对齐后的数据
        for key, data_list in aligned_data.items():
            if isinstance(data_list, list) and len(data_list) > 0:
                original_data_length = len(data_list)

                if start_idx < end_idx and start_idx < original_data_length:
                    actual_end_idx = min(end_idx, original_data_length)
                    trimmed_data[key] = data_list[start_idx:actual_end_idx]
                    print(
                        f"    {key}: {original_data_length} -> {len(trimmed_data[key])} (-{original_data_length - len(trimmed_data[key])})"
                    )
                else:
                    trimmed_data[key] = []
                    print(f"    警告: {key} 裁剪后为空")
            else:
                # 非列表数据或空数据保持不变
                trimmed_data[key] = data_list

        return trimmed_data, trimmed_main_timestamps

    def _fix_multimodal_start_alignment(
        self, main_timestamps: np.ndarray, preprocessed_data: dict
    ) -> np.ndarray:
        """修正多模态开头时间戳偏差问题 - 使用与最终验证一致的逻辑"""
        if len(main_timestamps) == 0:
            return main_timestamps

        # 识别所有有效的数据模态（排除外参数据和空数据）
        valid_keys = []
        for key in preprocessed_data.keys():
            if (
                not key.endswith("_extrinsics")
                and not key.endswith("_camera_info")
                and len(preprocessed_data[key]) > 0
            ):
                valid_keys.append(key)

        if len(valid_keys) <= 1:
            print("  ✓ 只有一个或零个数据模态，无需修正")
            return main_timestamps

        print(f"  检查 {len(valid_keys)} 个数据模态的开头对齐情况")

        # 分析每个数据模态的开头时刻模态间最大最小差值
        alignment_info = []
        max_alignment_tolerance_ms = 20  # 20ms容差
        severe_stuck_threshold_ms = 1000  # 1秒阈值，认为是严重卡住

        # 先进行向量化对齐，获取对齐后的时间戳
        aligned_timestamps_by_key = {}
        for key in valid_keys:
            timestamps = np.array(
                [item["timestamp"] for item in preprocessed_data[key]]
            )

            # 检查前5帧的对齐情况
            check_frames = min(5, len(main_timestamps), len(timestamps))
            if check_frames == 0:
                continue

            main_subset = main_timestamps[:check_frames]

            # 找到最近的对齐索引
            closest_indices = self.find_closest_indices_vectorized(
                timestamps, main_subset
            )
            aligned_timestamps = timestamps[closest_indices]

            aligned_timestamps_by_key[key] = aligned_timestamps

        # 逐帧检查开头几帧的模态间时间戳差值
        check_frames = min(5, len(main_timestamps))
        frame_spreads = []
        severely_stuck_keys = []

        for frame_idx in range(check_frames):
            # 收集该帧所有模态的时间戳
            frame_timestamps = []
            frame_keys = []

            for key in valid_keys:
                if key in aligned_timestamps_by_key and frame_idx < len(
                    aligned_timestamps_by_key[key]
                ):
                    frame_timestamps.append(aligned_timestamps_by_key[key][frame_idx])
                    frame_keys.append(key)

            if len(frame_timestamps) > 1:
                # 计算该帧所有模态时间戳的最大最小差值
                frame_timestamps = np.array(frame_timestamps)
                min_ts = np.min(frame_timestamps)
                max_ts = np.max(frame_timestamps)
                spread_ms = (max_ts - min_ts) * 1000

                frame_spreads.append(
                    {
                        "frame_idx": frame_idx,
                        "spread_ms": spread_ms,
                        "timestamps": frame_timestamps,
                        "keys": frame_keys,
                        "main_timestamp": main_timestamps[frame_idx],
                    }
                )

        # 分析开头对齐质量
        if frame_spreads:
            max_spread = max(spread["spread_ms"] for spread in frame_spreads)
            avg_spread = np.mean([spread["spread_ms"] for spread in frame_spreads])

            print(f"    开头{len(frame_spreads)}帧模态间时间戳差值分析:")
            print(f"      最大差值: {max_spread:.1f}ms")
            print(f"      平均差值: {avg_spread:.1f}ms")

            # 显示每帧的详细情况
            for spread_info in frame_spreads:
                frame_idx = spread_info["frame_idx"]
                spread_ms = spread_info["spread_ms"]
                if spread_ms > max_alignment_tolerance_ms:
                    print(f"      帧{frame_idx}: 差值 {spread_ms:.1f}ms (超过阈值)")

                    # 检查是否有严重卡住的模态
                    timestamps = spread_info["timestamps"]
                    keys = spread_info["keys"]
                    main_ts = spread_info["main_timestamp"]

                    for i, (ts, key) in enumerate(zip(timestamps, keys)):
                        diff_ms = abs(ts - main_ts) * 1000
                        if diff_ms > severe_stuck_threshold_ms:
                            if key not in severely_stuck_keys:
                                severely_stuck_keys.append(key)
                                print(
                                    f"        {key}: 与主时间戳偏差 {diff_ms:.1f}ms (严重卡住)"
                                )
                else:
                    print(f"      帧{frame_idx}: 差值 {spread_ms:.1f}ms (正常)")

        # 识别不同类型的问题模态
        problematic_frames = [
            s for s in frame_spreads if s["spread_ms"] > max_alignment_tolerance_ms
        ]

        # 处理严重卡住的数据模态
        if severely_stuck_keys:
            print(
                f"  发现 {len(severely_stuck_keys)} 个严重卡住的数据模态，需要特殊处理:"
            )

            # 按模态类型分类显示
            severely_stuck_by_type = {}
            for key in severely_stuck_keys:
                if any(cam in key for cam in ["head_cam", "wrist_cam"]):
                    modality_type = "相机"
                elif "action." in key:
                    modality_type = "动作"
                elif "observation." in key:
                    modality_type = "观测"
                else:
                    modality_type = "其他"

                if modality_type not in severely_stuck_by_type:
                    severely_stuck_by_type[modality_type] = []
                severely_stuck_by_type[modality_type].append(key)

            print("  严重卡住模态分布:")
            for mod_type, keys in severely_stuck_by_type.items():
                print(f"    {mod_type}: {len(keys)} 个 - {keys}")

            # 对严重卡住的数据模态进行时间戳替换
            for key in severely_stuck_keys:
                self._fix_severely_stuck_timestamps(
                    preprocessed_data, key, main_timestamps, max_alignment_tolerance_ms
                )

        # 检查是否需要常规修正（排除已处理的严重卡住数据）
        if not problematic_frames or len(severely_stuck_keys) == len(valid_keys):
            if severely_stuck_keys:
                print("  ✓ 严重卡住数据已处理，其他模态开头对齐良好")
            else:
                print("  ✓ 所有模态开头对齐良好，无需修正")
            return main_timestamps

        print(
            f"  发现开头 {len(problematic_frames)} 帧存在模态间对齐偏差过大，开始修正..."
        )

        # 统计问题模态（排除严重卡住的）
        normal_problematic_keys = set()
        for spread_info in problematic_frames:
            for key in spread_info["keys"]:
                if key not in severely_stuck_keys:
                    normal_problematic_keys.add(key)

        if normal_problematic_keys:
            # 按类型分组显示
            problematic_by_type = {}
            for key in normal_problematic_keys:
                if any(cam in key for cam in ["head_cam", "wrist_cam"]):
                    modality_type = "相机"
                elif "action." in key:
                    modality_type = "动作"
                elif "observation." in key:
                    modality_type = "观测"
                else:
                    modality_type = "其他"

                if modality_type not in problematic_by_type:
                    problematic_by_type[modality_type] = []
                problematic_by_type[modality_type].append(key)

            print("  问题模态分布:")
            for mod_type, keys in problematic_by_type.items():
                print(f"    {mod_type}: {len(keys)} 个 - {keys}")

        # 策略：找到所有正常数据模态都能良好对齐的时间范围
        best_start_idx = 0
        min_max_spread = float("inf")

        # 在前20帧中寻找最佳起始点
        search_range = min(20, len(main_timestamps))

        # 只考虑非严重卡住的数据模态
        normal_valid_keys = [
            key for key in valid_keys if key not in severely_stuck_keys
        ]

        for start_candidate in range(search_range):
            if start_candidate >= len(main_timestamps):
                break

            candidate_timestamps = main_timestamps[start_candidate:]
            if len(candidate_timestamps) < 10:  # 至少保留10帧
                break

            # 检查从这个起始点开始的对齐情况
            check_frames_candidate = min(5, len(candidate_timestamps))
            candidate_subset = candidate_timestamps[:check_frames_candidate]

            # 计算这个起始点的模态间最大差值
            max_spread_at_this_start = 0
            valid_alignment = True

            for frame_idx in range(check_frames_candidate):
                # 收集该帧所有正常模态的时间戳
                frame_timestamps = []

                for key in normal_valid_keys:
                    timestamps = np.array(
                        [item["timestamp"] for item in preprocessed_data[key]]
                    )

                    # 找到能覆盖候选时间戳的数据
                    valid_indices = np.where(timestamps >= candidate_subset[0])[0]
                    if len(valid_indices) < len(candidate_subset):
                        valid_alignment = False
                        break

                    closest_indices = self.find_closest_indices_vectorized(
                        timestamps, candidate_subset
                    )
                    if frame_idx < len(closest_indices) and closest_indices[
                        frame_idx
                    ] < len(timestamps):
                        frame_timestamps.append(timestamps[closest_indices[frame_idx]])

                if not valid_alignment:
                    break

                if len(frame_timestamps) > 1:
                    frame_timestamps = np.array(frame_timestamps)
                    spread_ms = (
                        np.max(frame_timestamps) - np.min(frame_timestamps)
                    ) * 1000
                    max_spread_at_this_start = max(max_spread_at_this_start, spread_ms)

            if valid_alignment and max_spread_at_this_start < min_max_spread:
                min_max_spread = max_spread_at_this_start
                best_start_idx = start_candidate

            # 如果找到了很好的对齐点，提前退出
            if min_max_spread <= max_alignment_tolerance_ms:
                break

        # 应用修正
        if best_start_idx > 0:
            original_length = len(main_timestamps)
            main_timestamps = main_timestamps[best_start_idx:]
            removed_frames = original_length - len(main_timestamps)

            worst_before = max(spread["spread_ms"] for spread in problematic_frames)

            print(f"  ✓ 修正完成：移除开头 {removed_frames} 帧")
            print(f"    最大模态间差值: {worst_before:.1f}ms -> {min_max_spread:.1f}ms")
            print(f"    修正后主时间戳长度: {len(main_timestamps)}")

            # 重新验证修正效果
            print("  验证修正效果:")
            check_frames_verify = min(3, len(main_timestamps))

            for frame_idx in range(check_frames_verify):
                frame_timestamps = []
                frame_keys = []

                for key in normal_valid_keys[:5]:  # 只验证前5个模态
                    timestamps = np.array(
                        [item["timestamp"] for item in preprocessed_data[key]]
                    )
                    closest_indices = self.find_closest_indices_vectorized(
                        timestamps, main_timestamps[:check_frames_verify]
                    )
                    if frame_idx < len(closest_indices) and closest_indices[
                        frame_idx
                    ] < len(timestamps):
                        frame_timestamps.append(timestamps[closest_indices[frame_idx]])
                        frame_keys.append(key)

                if len(frame_timestamps) > 1:
                    frame_timestamps = np.array(frame_timestamps)
                    spread_ms = (
                        np.max(frame_timestamps) - np.min(frame_timestamps)
                    ) * 1000
                    print(f"    帧{frame_idx}: 修正后模态间差值 {spread_ms:.1f}ms")

            if len(normal_valid_keys) > 5:
                print(f"    ... 其余 {len(normal_valid_keys) - 5} 个正常模态也已修正")

        else:
            print(f"  ⚠️ 无法找到满意的修正方案，保持原始时间戳")
            print(f"  建议检查数据质量，当前最小模态间最大差值: {min_max_spread:.1f}ms")

        return main_timestamps

    def _fix_severely_stuck_timestamps(
        self,
        preprocessed_data: dict,
        key: str,
        main_timestamps: np.ndarray,
        tolerance_ms: float = 20,
    ):
        """修复严重卡住的数据模态的时间戳"""
        print(f"  开始修复严重卡住的数据模态: {key}")

        data_list = preprocessed_data[key]
        if len(data_list) == 0:
            return

        # 获取原始时间戳
        original_timestamps = np.array([item["timestamp"] for item in data_list])

        # 找到第一个正常时间戳的位置
        normal_start_index = None
        for i in range(len(original_timestamps)):
            if i < len(main_timestamps):
                main_ts = main_timestamps[i]
                data_ts = original_timestamps[i]
                diff_ms = abs(data_ts - main_ts) * 1000

                if diff_ms <= tolerance_ms:
                    normal_start_index = i
                    print(f"    在索引 {i} 处找到正常时间戳，偏差 {diff_ms:.1f}ms")
                    break

        if normal_start_index is None:
            # 如果没有找到正常的时间戳，寻找数据开始变化的位置
            print(f"    未找到正常时间戳，寻找数据开始变化的位置...")

            # 寻找时间戳开始明显变化的位置
            for i in range(1, min(len(original_timestamps), len(main_timestamps))):
                time_change = abs(original_timestamps[i] - original_timestamps[0])
                if time_change > 1.0:  # 时间戳变化超过1秒
                    # 检查这个位置是否能与主时间戳对齐
                    expected_main_ts = (
                        main_timestamps[i]
                        if i < len(main_timestamps)
                        else main_timestamps[-1]
                        + (i - len(main_timestamps) + 1) * 0.033
                    )
                    diff_ms = abs(original_timestamps[i] - expected_main_ts) * 1000

                    if diff_ms <= tolerance_ms * 5:  # 放宽5倍容差
                        normal_start_index = i
                        print(
                            f"    在索引 {i} 处找到数据变化点，开始正常同步，偏差 {diff_ms:.1f}ms"
                        )
                        break

        # 执行时间戳替换
        replaced_count = 0
        if normal_start_index is not None and normal_start_index > 0:
            # 从开头到normal_start_index，使用主时间戳替换
            for i in range(min(normal_start_index, len(main_timestamps))):
                if i < len(data_list):
                    old_timestamp = data_list[i]["timestamp"]
                    new_timestamp = main_timestamps[i]
                    data_list[i]["timestamp"] = new_timestamp
                    data_list[i]["timestamp_replaced"] = True
                    data_list[i]["original_timestamp"] = old_timestamp
                    replaced_count += 1

            print(f"    ✓ 替换了前 {replaced_count} 个时间戳")
            print(f"    从索引 {normal_start_index} 开始使用原始时间戳")

        else:
            # 如果始终无法同步，替换更多的开头时间戳
            max_replace_count = min(
                50, len(data_list), len(main_timestamps)
            )  # 最多替换50帧

            print(f"    无法找到同步点，强制替换前 {max_replace_count} 个时间戳")

            for i in range(max_replace_count):
                if i < len(data_list):
                    old_timestamp = data_list[i]["timestamp"]
                    new_timestamp = main_timestamps[i]
                    data_list[i]["timestamp"] = new_timestamp
                    data_list[i]["timestamp_replaced"] = True
                    data_list[i]["original_timestamp"] = old_timestamp
                    replaced_count += 1

            print(f"    ⚠️ 强制替换了前 {replaced_count} 个时间戳")

        # 验证修复效果
        print(f"  验证 {key} 修复效果:")
        new_timestamps = np.array([item["timestamp"] for item in data_list])
        check_frames = min(5, len(main_timestamps), len(new_timestamps))

        if check_frames > 0:
            main_subset = main_timestamps[:check_frames]
            data_subset = new_timestamps[:check_frames]
            time_diffs_ms = np.abs(data_subset - main_subset) * 1000
            max_diff = np.max(time_diffs_ms)
            avg_diff = np.mean(time_diffs_ms)

            print(
                f"    修复后开头{check_frames}帧: 最大偏差 {max_diff:.1f}ms, 平均偏差 {avg_diff:.1f}ms"
            )

            if max_diff <= tolerance_ms:
                print(f"    ✓ {key} 修复成功，开头偏差已控制在 {tolerance_ms}ms 内")
            else:
                print(f"    ⚠️ {key} 修复后仍有偏差，但已显著改善")

        # 更新预处理数据
        preprocessed_data[key] = data_list

    def _adjust_frame_rate_to_30fps(
        self, aligned_data: dict, main_timestamps: np.ndarray
    ):
        """
        调整帧率到30fps范围内（29.95-30.05Hz）
        通过插帧或抽帧来达到目标帧率
        """
        print("=" * 60)
        print("开始帧率调整到30fps...")

        if len(main_timestamps) < 2:
            print("  ⚠️ 时间戳数量不足，跳过帧率调整")
            return aligned_data, main_timestamps

        # 计算当前帧率
        time_span = main_timestamps[-1] - main_timestamps[0]
        current_fps = len(main_timestamps) / time_span
        target_fps_min = 29.905
        target_fps_max = 30.095

        print(f"  当前帧率: {current_fps:.2f}Hz")
        print(f"  目标范围: {target_fps_min:.2f}-{target_fps_max:.2f}Hz")
        print(f"  当前帧数: {len(main_timestamps)}")
        print(f"  时间跨度: {time_span:.3f}s")

        # 检查是否在目标范围内
        if target_fps_min <= current_fps <= target_fps_max:
            print(f"  ✓ 帧率已在目标范围内，无需调整")
            return aligned_data, main_timestamps

        # 转换为numpy数组便于操作
        main_timestamps = np.array(main_timestamps)

        # 收集所有有效的数据模态
        valid_modalities = {}
        for key, data_list in aligned_data.items():
            if len(data_list) > 0:
                valid_modalities[key] = list(data_list)  # 转换为列表便于插入/删除

        if current_fps < target_fps_min:
            # 帧率太低，需要插帧
            print(f"  帧率过低，开始插帧...")
            main_timestamps, valid_modalities = self._insert_frames_to_increase_fps(
                main_timestamps, valid_modalities, target_fps_min, time_span
            )
        elif current_fps > target_fps_max:
            # 帧率太高，需要抽帧
            print(f"  帧率过高，开始抽帧...")
            main_timestamps, valid_modalities = self._remove_frames_to_decrease_fps(
                main_timestamps, valid_modalities, target_fps_max, time_span
            )

        # 验证调整结果
        final_time_span = main_timestamps[-1] - main_timestamps[0]
        final_fps = len(main_timestamps) / final_time_span

        print(f"  调整后帧率: {final_fps:.2f}Hz")
        print(f"  调整后帧数: {len(main_timestamps)}")
        print(f"  调整后时间跨度: {final_time_span:.3f}s")

        if target_fps_min <= final_fps <= target_fps_max:
            print(f"  ✓ 帧率调整成功！")
        else:
            print(f"  ⚠️ 帧率调整后仍不在目标范围内")

        # 转换回原格式
        aligned_data_adjusted = {}
        for key, data_list in valid_modalities.items():
            aligned_data_adjusted[key] = data_list

        # 添加空的模态数据
        for key, data_list in aligned_data.items():
            if key not in aligned_data_adjusted:
                aligned_data_adjusted[key] = []

        print("=" * 60)
        return aligned_data_adjusted, main_timestamps

    def _insert_frames_to_increase_fps(
        self,
        main_timestamps: np.ndarray,
        valid_modalities: dict,
        target_fps: float,
        time_span: float,
    ):
        """
        通过插帧来提高帧率到目标值
        """
        target_frame_count = int(time_span * target_fps)
        current_frame_count = len(main_timestamps)
        frames_to_insert = target_frame_count - current_frame_count

        print(f"    需要插入 {frames_to_insert} 帧")

        if frames_to_insert <= 0:
            return main_timestamps, valid_modalities

        # 计算时间间隔
        time_intervals = np.diff(main_timestamps) * 1000  # 转换为毫秒
        insertion_threshold_ms = 33.0  # 33ms阈值

        inserted_count = 0
        max_iterations = frames_to_insert * 2  # 防止无限循环
        iteration = 0

        while inserted_count < frames_to_insert and iteration < max_iterations:
            iteration += 1

            # 重新计算间隔（因为插入会改变）
            time_intervals = np.diff(main_timestamps) * 1000

            # 找到最大的间隔
            max_interval_idx = np.argmax(time_intervals)
            max_interval_ms = time_intervals[max_interval_idx]

            if max_interval_ms <= insertion_threshold_ms:
                print(f"    无法找到超过{insertion_threshold_ms}ms的间隔进行插帧")
                break

            # 在最大间隔处插入一帧
            insert_pos = max_interval_idx + 1

            # 计算插入时间戳（两帧中间）
            prev_timestamp = main_timestamps[max_interval_idx]
            next_timestamp = main_timestamps[max_interval_idx + 1]
            new_timestamp = (prev_timestamp + next_timestamp) / 2

            # 插入主时间戳
            main_timestamps = np.insert(main_timestamps, insert_pos, new_timestamp)

            # 为所有模态插入数据（复制前一帧）
            for key, data_list in valid_modalities.items():
                if insert_pos <= len(data_list):
                    # 复制前一帧数据
                    reference_frame = data_list[max_interval_idx].copy()
                    reference_frame["timestamp"] = new_timestamp
                    reference_frame["frame_inserted"] = True  # 标记为插入帧
                    data_list.insert(insert_pos, reference_frame)

            inserted_count += 1

            if inserted_count % 10 == 0:  # 每插入10帧输出一次进度
                current_fps = len(main_timestamps) / (
                    main_timestamps[-1] - main_timestamps[0]
                )
                print(f"    已插入 {inserted_count} 帧，当前帧率: {current_fps:.2f}Hz")

        print(f"    实际插入了 {inserted_count} 帧")

        return main_timestamps, valid_modalities

    # def _remove_frames_to_decrease_fps(self, main_timestamps: np.ndarray, valid_modalities: dict,
    #                                 target_fps: float, time_span: float):
    #     """
    #     通过抽帧来降低帧率到目标值
    #     """
    #     target_frame_count = int(time_span * target_fps)
    #     current_frame_count = len(main_timestamps)
    #     frames_to_remove = current_frame_count - target_frame_count

    #     print(f"    需要删除 {frames_to_remove} 帧")

    #     if frames_to_remove <= 0:
    #         return main_timestamps, valid_modalities

    #     removal_threshold_ms = 40.0  # 40ms阈值
    #     removed_count = 0
    #     max_iterations = frames_to_remove * 2  # 防止无限循环
    #     iteration = 0

    #     # 创建可删除帧的候选列表（排除首尾帧）
    #     removable_indices = list(range(1, len(main_timestamps) - 1))

    #     while removed_count < frames_to_remove and iteration < max_iterations and removable_indices:
    #         iteration += 1

    #         # 寻找可以安全删除的帧
    #         best_remove_idx = None
    #         min_max_interval = float('inf')

    #         for candidate_idx in removable_indices[:]:  # 使用切片创建副本
    #             if candidate_idx <= 0 or candidate_idx >= len(main_timestamps) - 1:
    #                 removable_indices.remove(candidate_idx)
    #                 continue

    #             # 计算删除该帧后前后帧的间隔
    #             prev_timestamp = main_timestamps[candidate_idx - 1]
    #             next_timestamp = main_timestamps[candidate_idx + 1]
    #             resulting_interval_ms = (next_timestamp - prev_timestamp) * 1000

    #             # 检查是否满足删除条件
    #             if resulting_interval_ms < removal_threshold_ms:
    #                 # 选择删除后间隔最小的帧（更安全）
    #                 if resulting_interval_ms < min_max_interval:
    #                     min_max_interval = resulting_interval_ms
    #                     best_remove_idx = candidate_idx

    #         if best_remove_idx is None:
    #             print(f"    无法找到可以安全删除的帧（删除后间隔需小于{removal_threshold_ms}ms）")
    #             break

    #         # 删除选中的帧
    #         # 删除主时间戳
    #         main_timestamps = np.delete(main_timestamps, best_remove_idx)

    #         # 删除所有模态的对应数据
    #         for key, data_list in valid_modalities.items():
    #             if best_remove_idx < len(data_list):
    #                 del data_list[best_remove_idx]

    #         # 更新可删除索引列表（调整索引值）
    #         removable_indices = [idx - 1 if idx > best_remove_idx else idx for idx in removable_indices]
    #         removable_indices = [idx for idx in removable_indices if 0 < idx < len(main_timestamps) - 1]

    #         removed_count += 1

    #         if removed_count % 10 == 0:  # 每删除10帧输出一次进度
    #             current_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])
    #             print(f"    已删除 {removed_count} 帧，当前帧率: {current_fps:.2f}Hz")

    #     # 检查是否成功达到目标
    #     if removed_count < frames_to_remove:
    #         final_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])
    #         error_msg = (
    #             f"无法通过抽帧达到目标帧率。需要删除 {frames_to_remove} 帧，"
    #             f"但只能安全删除 {removed_count} 帧。当前帧率: {final_fps:.2f}Hz"
    #         )
    #         print(f"    ❌ {error_msg}")

    #         raise TimestampStuckError(
    #             message=f"帧率调整失败: {error_msg}",
    #             topic="frame_rate_adjustment",
    #             stuck_timestamp=None,
    #             stuck_duration=None,
    #             stuck_frame_count=frames_to_remove - removed_count,
    #             threshold=target_fps
    #         )

    #     print(f"    实际删除了 {removed_count} 帧")

    #     return main_timestamps, valid_modalities
    def _remove_frames_to_decrease_fps(
        self,
        main_timestamps: np.ndarray,
        valid_modalities: dict,
        target_fps: float,
        time_span: float,
    ):
        """
        通过滑动窗口删除+局部时间戳重新平均来降低帧率到目标值
        删除后对窗口内时间戳重新平均分布，并同步调整所有模态
        """
        target_frame_count = int(time_span * target_fps)
        current_frame_count = len(main_timestamps)
        frames_to_remove = current_frame_count - target_frame_count

        print(
            f"    需要删除 {frames_to_remove} 帧 (从 {current_frame_count} 帧降到 {target_frame_count} 帧)"
        )

        if frames_to_remove <= 0:
            return main_timestamps, valid_modalities

        removed_count = 0
        max_iterations = frames_to_remove * 3  # 防止无限循环
        iteration = 0

        # 滑动窗口参数
        window_size = 5  # 初始窗口大小（必须为奇数）
        max_window_size = 15  # 最大窗口大小
        max_interval_threshold_ms = 40.0  # 最大间隔阈值

        print(f"    使用滑动窗口删除+重新平均算法，初始窗口大小: {window_size}")

        while removed_count < frames_to_remove and iteration < max_iterations:
            iteration += 1

            if len(main_timestamps) <= window_size + 2:  # 保证至少有足够的帧数
                print(f"    剩余帧数过少({len(main_timestamps)})，无法继续删除")
                break

            # 寻找最佳删除候选
            best_candidate = None
            best_score = float("inf")
            candidates_found = 0

            # 滑动窗口遍历所有可能的删除位置
            for start_idx in range(len(main_timestamps) - window_size + 1):
                end_idx = start_idx + window_size
                center_idx = start_idx + window_size // 2  # 窗口中心索引

                # 跳过首尾帧附近的窗口
                if center_idx <= 1 or center_idx >= len(main_timestamps) - 2:
                    continue

                # 提取窗口时间戳
                window_timestamps = main_timestamps[start_idx:end_idx]

                # 模拟删除窗口中心帧
                timestamps_after_removal = np.concatenate(
                    [
                        window_timestamps[: window_size // 2],  # 中心帧之前
                        window_timestamps[window_size // 2 + 1 :],  # 中心帧之后
                    ]
                )

                # 对删除后的时间戳进行重新平均分布
                reaveraged_timestamps = self._reaverage_timestamps_in_window(
                    timestamps_after_removal,
                    window_timestamps[0],
                    window_timestamps[-1],
                )

                # 检查重新平均后的最大时间间隔
                if len(reaveraged_timestamps) > 1:
                    reaveraged_intervals_ms = np.diff(reaveraged_timestamps) * 1000
                    max_reaveraged_interval = np.max(reaveraged_intervals_ms)

                    # 检查是否满足40ms限制
                    if max_reaveraged_interval <= max_interval_threshold_ms:
                        candidates_found += 1

                        # 计算评分（优先选择重新平均后间隔最小且最均匀的）
                        interval_score = max_reaveraged_interval
                        uniformity_score = np.std(reaveraged_intervals_ms) * 2
                        density_score = -np.mean(
                            np.diff(window_timestamps) * 1000
                        )  # 优先删除密集区域

                        total_score = interval_score + uniformity_score + density_score

                        if total_score < best_score:
                            best_score = total_score
                            best_candidate = {
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "remove_idx": center_idx,
                                "window_size": window_size,
                                "original_timestamps": window_timestamps,
                                "reaveraged_timestamps": reaveraged_timestamps,
                                "max_interval_after": max_reaveraged_interval,
                                "score": total_score,
                            }

            # 如果找到了合适的删除候选
            if best_candidate is not None:
                # 执行删除和重新平均
                # 修改后的代码
                new_timestamps, success = self._execute_window_removal_and_reaverage(
                    main_timestamps, valid_modalities, best_candidate
                )

                if success:
                    main_timestamps = new_timestamps  # 更新时间戳数组
                    removed_count += 1

                    if removed_count % 10 == 0:
                        current_fps = len(main_timestamps) / (
                            main_timestamps[-1] - main_timestamps[0]
                        )
                        print(
                            f"      已删除 {removed_count} 帧，当前帧率: {current_fps:.2f}Hz"
                        )
                        print(
                            f"      最新删除: 窗口{best_candidate['start_idx']}-{best_candidate['end_idx']}, "
                            f"删除后最大间隔: {best_candidate['max_interval_after']:.1f}ms"
                        )
                else:
                    print(f"    执行删除失败，跳过此候选")

            else:
                # 没有找到合适的删除候选
                print(
                    f"    第{iteration}轮: 窗口大小{window_size}下找到 {candidates_found} 个候选"
                )

                # 扩大窗口大小再尝试
                if window_size < max_window_size:
                    window_size += 2  # 保持奇数
                    print(f"    扩大窗口大小到: {window_size}，继续尝试")
                    continue
                else:
                    # 窗口已经最大，无法继续
                    print(
                        f"    窗口大小已达到最大({window_size})，无法找到更多可删除位置"
                    )
                    break

        # 最终验证和统计
        final_fps = len(main_timestamps) / (main_timestamps[-1] - main_timestamps[0])

        print(f"    删除完成统计:")
        print(f"      目标删除: {frames_to_remove} 帧")
        print(f"      实际删除: {removed_count} 帧")
        print(f"      最终帧率: {final_fps:.3f}Hz")

        # 验证最终时间戳质量
        if len(main_timestamps) > 1:
            final_intervals_ms = np.diff(main_timestamps) * 1000
            max_final_interval = np.max(final_intervals_ms)
            avg_final_interval = np.mean(final_intervals_ms)
            std_final_interval = np.std(final_intervals_ms)

            print(f"      最终时间戳质量:")
            print(f"        最大间隔: {max_final_interval:.1f}ms")
            print(f"        平均间隔: {avg_final_interval:.1f}ms")
            print(f"        间隔标准差: {std_final_interval:.1f}ms")

            # 严格验证：检查是否仍有超过40ms的间隔
            large_intervals = final_intervals_ms > max_interval_threshold_ms
            if np.any(large_intervals):
                large_count = np.sum(large_intervals)
                worst_interval = np.max(final_intervals_ms)

                error_msg = (
                    f"删除后验证失败：仍有 {large_count} 个间隔超过{max_interval_threshold_ms}ms，"
                    f"最大间隔{worst_interval:.1f}ms"
                )
                print(f"        ❌ {error_msg}")

                # 显示具体问题间隔
                problem_indices = np.where(large_intervals)[0]
                for i, idx in enumerate(problem_indices[:3]):  # 只显示前3个
                    interval_value = final_intervals_ms[idx]
                    start_time = main_timestamps[idx]
                    end_time = main_timestamps[idx + 1]
                    print(
                        f"          问题间隔{i+1}: {start_time:.6f}s -> {end_time:.6f}s, 间隔={interval_value:.1f}ms"
                    )

                raise TimestampStuckError(
                    message=f"严格间隔验证失败: {error_msg}",
                    topic="strict_interval_validation",
                    stuck_timestamp=main_timestamps[problem_indices[0]],
                    stuck_duration=worst_interval / 1000,
                    stuck_frame_count=large_count,
                    threshold=max_interval_threshold_ms / 1000,
                )
            else:
                print(f"        ✓ 所有间隔都在{max_interval_threshold_ms}ms以内")

        # 严格检查：是否达到目标帧率
        if removed_count < frames_to_remove:
            shortfall = frames_to_remove - removed_count
            error_msg = (
                f"删除未完成：需要删除 {frames_to_remove} 帧，实际删除 {removed_count} 帧，"
                f"还差 {shortfall} 帧。当前帧率: {final_fps:.3f}Hz，目标: ≤{target_fps:.3f}Hz"
            )

            print(f"    ❌ {error_msg}")

            raise TimestampStuckError(
                message=f"严格帧率调整失败: {error_msg}",
                topic="strict_frame_rate_adjustment",
                stuck_timestamp=None,
                stuck_duration=None,
                stuck_frame_count=shortfall,
                threshold=target_fps,
            )

        # 最终检查：验证帧率是否达到目标
        if final_fps > target_fps:
            fps_excess = final_fps - target_fps
            error_msg = (
                f"最终帧率验证失败：当前帧率 {final_fps:.3f}Hz 仍然超过目标 {target_fps:.3f}Hz，"
                f"超出 {fps_excess:.3f}Hz"
            )

            print(f"    ❌ {error_msg}")

            raise TimestampStuckError(
                message=f"严格帧率目标未达成: {error_msg}",
                topic="strict_fps_target",
                stuck_timestamp=None,
                stuck_duration=None,
                stuck_frame_count=frames_to_remove - removed_count,
                threshold=target_fps,
            )

        print(f"    ✓ 滑动窗口删除+重新平均成功完成")
        print(f"      最终帧率: {final_fps:.3f}Hz ≤ {target_fps:.3f}Hz")
        print(
            f"      最大时间间隔: {max_final_interval:.1f}ms ≤ {max_interval_threshold_ms}ms"
        )

        return main_timestamps, valid_modalities

    def _reaverage_timestamps_in_window(
        self,
        timestamps_after_removal: np.ndarray,
        window_start_time: float,
        window_end_time: float,
    ) -> np.ndarray:
        """
        对删除帧后的窗口内时间戳进行重新平均分布
        修正版：只对内部时间戳重新平均，保持两端不变

        Args:
            timestamps_after_removal: 删除中心帧后的时间戳数组
            window_start_time: 窗口开始时间（保持不变）
            window_end_time: 窗口结束时间（保持不变）

        Returns:
            重新平均分布后的时间戳数组
        """
        if len(timestamps_after_removal) <= 2:
            # 如果只有2个或更少的点，无法进行内部重新平均
            return timestamps_after_removal

        # 使用传入的窗口起始和结束时间，而不是数组的首尾时间
        start_time = window_start_time
        end_time = window_end_time

        # 内部点数量
        num_internal_points = len(timestamps_after_removal) - 2

        if num_internal_points <= 0:
            # 没有内部点，只返回首尾时间戳
            return np.array([start_time, end_time])

        # 重新平均：在起始和结束时间之间均匀分布内部点
        internal_timestamps = np.linspace(
            start_time, end_time, num_internal_points + 2
        )[1:-1]

        # 构建完整的重新平均时间戳数组
        reaveraged_timestamps = np.concatenate(
            [
                [start_time],  # 起始点保持不变
                internal_timestamps,  # 内部点重新平均
                [end_time],  # 结束点保持不变
            ]
        )

        return reaveraged_timestamps

    def _execute_window_removal_and_reaverage(
        self, main_timestamps: np.ndarray, valid_modalities: dict, candidate: dict
    ) -> bool:
        """
        执行窗口删除和重新平均操作，同步更新所有模态
        修正版：只对窗口内部时间戳重新平均，两端保持不变；子时间戳使用变化量同步
        """
        try:
            start_idx = candidate["start_idx"]
            end_idx = candidate["end_idx"]
            remove_idx = candidate["remove_idx"]
            window_size = candidate["window_size"]

            # 确保窗口大小至少为5（删除后至少4个点，内部至少2个点可以平均）
            if window_size < 5:
                print(f"    窗口大小 {window_size} < 5，无法安全进行内部重新平均")
                return main_timestamps, False

            # 1. 删除主时间戳的中心帧
            main_timestamps_list = main_timestamps.tolist()
            del main_timestamps_list[remove_idx]

            # 2. 同步删除所有模态的对应帧
            for key, data_list in valid_modalities.items():
                if remove_idx < len(data_list):
                    del data_list[remove_idx]

            # 3. 更新主时间戳数组
            new_main_timestamps = np.array(main_timestamps_list)

            # 4. 重新计算窗口范围（删除后索引会变化）
            window_start_idx = start_idx
            window_end_idx = end_idx - 1  # 删除了一帧，所以end_idx要减1

            # 确保索引范围有效
            if window_start_idx >= len(new_main_timestamps) or window_end_idx > len(
                new_main_timestamps
            ):
                print(f"    删除后窗口索引超出范围，跳过此次操作")
                return main_timestamps, False

            # 5. 提取窗口内的时间戳进行重新平均（只平均内部点，保持两端不变）
            window_timestamps = new_main_timestamps[window_start_idx:window_end_idx]

            if len(window_timestamps) < 3:
                print(
                    f"    删除后窗口内时间戳过少({len(window_timestamps)})，无法重新平均"
                )
                return main_timestamps, False

            # 6. 重新平均：只平均内部点，保持首尾不变
            start_time = window_timestamps[0]  # 窗口起始时间（保持不变）
            end_time = window_timestamps[-1]  # 窗口结束时间（保持不变）

            # 计算内部点的新时间戳（均匀分布）
            num_internal_points = len(window_timestamps) - 2  # 内部点数量

            if num_internal_points > 0:
                # 在起始和结束时间之间均匀分布内部点
                internal_new_timestamps = np.linspace(
                    start_time, end_time, num_internal_points + 2
                )[1:-1]

                # 构建完整的重新平均时间戳数组
                reaveraged_timestamps = np.concatenate(
                    [
                        [start_time],  # 起始点保持不变
                        internal_new_timestamps,  # 内部点重新平均
                        [end_time],  # 结束点保持不变
                    ]
                )
            else:
                # 如果没有内部点，直接使用原时间戳
                reaveraged_timestamps = window_timestamps

            # 验证重新平均后的间隔
            if len(reaveraged_timestamps) > 1:
                reaveraged_intervals_ms = np.diff(reaveraged_timestamps) * 1000
                max_reaveraged_interval = np.max(reaveraged_intervals_ms)

                if max_reaveraged_interval > 40:
                    print(
                        f"    重新平均后最大间隔 {max_reaveraged_interval:.1f}ms 仍超过40ms"
                    )
                    return main_timestamps, False

            # 7. 更新窗口内的主时间戳
            for i, new_timestamp in enumerate(reaveraged_timestamps):
                global_idx = window_start_idx + i
                if global_idx < len(new_main_timestamps):
                    old_timestamp = new_main_timestamps[global_idx]
                    timestamp_delta = new_timestamp - old_timestamp

                    # 更新主时间戳
                    new_main_timestamps[global_idx] = new_timestamp

                    # 8. 同步更新所有模态对应帧的时间戳（使用变化量）
                    for key, data_list in valid_modalities.items():
                        if global_idx < len(data_list):
                            if "timestamp" in data_list[global_idx]:
                                # 保存原始时间戳
                                original_modality_timestamp = data_list[global_idx][
                                    "timestamp"
                                ]

                                # 应用相同的时间戳变化量（保持各模态间的相对关系）
                                new_modality_timestamp = (
                                    original_modality_timestamp + timestamp_delta
                                )

                                # 更新时间戳
                                data_list[global_idx][
                                    "timestamp"
                                ] = new_modality_timestamp

                                # 添加调试信息
                                data_list[global_idx]["timestamp_reaveraged"] = True
                                data_list[global_idx][
                                    "original_timestamp"
                                ] = original_modality_timestamp
                                data_list[global_idx][
                                    "timestamp_delta"
                                ] = timestamp_delta
                                data_list[global_idx][
                                    "main_timestamp_new"
                                ] = new_timestamp

            # 9. 将更新后的主时间戳数组复制回原数组
            # main_timestamps[:] = new_main_timestamps[:]

            # 10. 最终验证操作结果
            if len(new_main_timestamps) > 1:
                # 验证整个窗口的间隔
                window_intervals_ms = (
                    np.diff(new_main_timestamps[window_start_idx:window_end_idx]) * 1000
                )
                max_interval = (
                    np.max(window_intervals_ms) if len(window_intervals_ms) > 0 else 0
                )

                if max_interval > 40:
                    print(f"    ⚠️ 窗口重新平均后间隔仍然过大: {max_interval:.1f}ms")
                    return main_timestamps, False

                # 输出调试信息
                avg_interval = (
                    np.mean(window_intervals_ms) if len(window_intervals_ms) > 0 else 0
                )
                print(
                    f"    ✓ 窗口重新平均成功: 平均间隔 {avg_interval:.1f}ms, 最大间隔 {max_interval:.1f}ms"
                )

            return new_main_timestamps, True

        except Exception as e:
            print(f"    执行窗口删除和重新平均时出错: {e}")
            return main_timestamps, False

    def _final_alignment_validation(
        self, aligned_data: dict, main_timestamps: np.ndarray
    ):
        """最终验证对齐后的数据质量，不满足要求则抛出异常"""

        # === 新增验证：主时间戳长度和时间跨度检查 ===
        print("  验证0: 主时间戳基本要求检查")

        # 验证长度要求（至少300帧）
        min_required_frames = 300
        if len(main_timestamps) < min_required_frames:
            error_msg = f"主时间戳长度 {len(main_timestamps)} 小于最低要求 {min_required_frames} 帧"
            print(f"    ❌ {error_msg}")

            raise TimestampStuckError(
                message=f"数据长度不足: {error_msg}",
                topic="main_timeline_length",
                stuck_timestamp=(
                    main_timestamps[0] if len(main_timestamps) > 0 else None
                ),
                stuck_duration=None,
                stuck_frame_count=len(main_timestamps),
                threshold=min_required_frames,
            )
        else:
            print(
                f"    ✓ 主时间戳长度验证通过: {len(main_timestamps)} 帧 (>= {min_required_frames})"
            )

        # 验证时间跨度要求（至少10秒）
        min_required_duration = 10.0  # 秒
        if len(main_timestamps) > 1:
            time_span = main_timestamps[-1] - main_timestamps[0]

            if time_span < min_required_duration:
                error_msg = f"主时间戳时间跨度 {time_span:.3f}s 小于最低要求 {min_required_duration}s"
                print(f"    ❌ {error_msg}")

                raise TimestampStuckError(
                    message=f"数据时间跨度不足: {error_msg}",
                    topic="main_timeline_duration",
                    stuck_timestamp=main_timestamps[0],
                    stuck_duration=time_span,
                    stuck_frame_count=len(main_timestamps),
                    threshold=min_required_duration,
                )
            else:
                print(
                    f"    ✓ 主时间戳时间跨度验证通过: {time_span:.3f}s (>= {min_required_duration}s)"
                )

            # === 新增验证：主时间戳频率检查 ===
            # 计算实际频率：帧数 / 时间跨度
            actual_fps = len(main_timestamps) / time_span
            max_required_fps = 30.095  # Hz
            min_required_fps = 29.905

            if actual_fps > max_required_fps:
                error_msg = (
                    f"主时间戳频率 {actual_fps:.2f}Hz 大于最大要求 {max_required_fps}Hz"
                )
                print(f"    ❌ {error_msg}")

                raise TimestampStuckError(
                    message=f"数据频率过大: {error_msg}",
                    topic="main_timeline_fps",
                    stuck_timestamp=main_timestamps[0],
                    stuck_duration=time_span,
                    stuck_frame_count=len(main_timestamps),
                    threshold=max_required_fps,
                )
            elif actual_fps < min_required_fps:
                error_msg = (
                    f"主时间戳频率 {actual_fps:.2f}Hz 小于最低要求 {min_required_fps}Hz"
                )
                print(f"    ❌ {error_msg}")

                raise TimestampStuckError(
                    message=f"数据频率不足: {error_msg}",
                    topic="main_timeline_fps",
                    stuck_timestamp=main_timestamps[0],
                    stuck_duration=time_span,
                    stuck_frame_count=len(main_timestamps),
                    threshold=min_required_fps,
                )
            else:
                print(
                    f"    ✓ 主时间戳频率验证通过: {min_required_fps}Hz <={actual_fps:.2f}Hz (<= {max_required_fps}Hz)"
                )
        else:
            # 只有一个时间戳的情况
            error_msg = (
                f"主时间戳只有 {len(main_timestamps)} 个，无法计算时间跨度和频率"
            )
            print(f"    ❌ {error_msg}")

            raise TimestampStuckError(
                message=f"数据不足以计算时间跨度和频率: {error_msg}",
                topic="main_timeline_duration",
                stuck_timestamp=(
                    main_timestamps[0] if len(main_timestamps) > 0 else None
                ),
                stuck_duration=0,
                stuck_frame_count=len(main_timestamps),
                threshold=min_required_duration,
            )

        # 验证1: 检查主时间戳间隔是否超过40ms
        print("  验证1: 主时间戳间隔检查")
        if len(main_timestamps) > 1:
            main_timestamps_ns = (main_timestamps * 1e9).astype(np.int64)
            main_intervals_ns = np.diff(main_timestamps_ns)
            main_intervals_ms = main_intervals_ns / 1e6

            max_main_interval = np.max(main_intervals_ms)
            if max_main_interval > 40:
                error_msg = f"主时间戳最大间隔 {max_main_interval:.1f}ms 超过40ms阈值"
                print(f"    ❌ {error_msg}")

                # 显示具体的大间隔
                large_interval_indices = np.where(main_intervals_ms > 40)[0]
                print(f"    主时间戳大间隔详情:")
                for idx in large_interval_indices[:3]:  # 只显示前3个
                    print(
                        f"      间隔{idx}: {main_intervals_ms[idx]:.1f}ms "
                        f"({main_timestamps[idx]:.6f}s -> {main_timestamps[idx+1]:.6f}s)"
                    )

                raise TimestampStuckError(
                    message=f"时间戳间隔验证失败: {error_msg}",
                    topic="main_timeline",
                    stuck_timestamp=None,
                    stuck_duration=max_main_interval / 1000,
                    stuck_frame_count=None,
                    threshold=0.04,
                )
            else:
                print(f"    ✓ 主时间戳间隔验证通过: 最大间隔 {max_main_interval:.1f}ms")

        # 收集所有有效模态的时间戳
        valid_modalities = {}
        for key, data_list in aligned_data.items():
            # 跳过外参数据和camera_info数据的验证
            if (
                len(data_list) == 0
                or key.endswith("_extrinsics")
                or key.endswith("_camera_info")
                or key.startswith("end_")
            ):
                continue

            aligned_timestamps = np.array([item["timestamp"] for item in data_list])

            # 检查长度一致性
            if len(aligned_timestamps) != len(main_timestamps):
                print(
                    f"  ❌ {key}: 长度不匹配 ({len(aligned_timestamps)} vs {len(main_timestamps)})"
                )
                continue

            valid_modalities[key] = aligned_timestamps

        # 如果没有有效模态，跳过验证
        if not valid_modalities:
            print("  ⚠️ 没有找到有效的数据模态进行验证")
            return

        print(f"  开始验证 {len(valid_modalities)} 个数据模态的时间戳同步...")

        alignment_errors = []

        # 验证2: 每个时刻所有模态时间戳的最大最小差值不超过20ms
        print("  验证2: 每个时刻多模态间的时间戳差值")
        frame_sync_errors = []

        # 将所有模态的时间戳转换为纳秒精度
        all_timestamps_ns = {}
        for key, timestamps in valid_modalities.items():
            all_timestamps_ns[key] = (timestamps * 1e9).astype(np.int64)

        # 逐帧检查
        max_frame_spread = 0
        worst_frame_idx = -1

        for frame_idx in range(len(main_timestamps)):
            # 收集该帧所有模态的时间戳
            frame_timestamps_ns = []
            frame_keys = []

            for key, timestamps_ns in all_timestamps_ns.items():
                if frame_idx < len(timestamps_ns):
                    frame_timestamps_ns.append(timestamps_ns[frame_idx])
                    frame_keys.append(key)

            if len(frame_timestamps_ns) > 1:
                # 计算该帧所有模态时间戳的范围
                min_ts_ns = np.min(frame_timestamps_ns)
                max_ts_ns = np.max(frame_timestamps_ns)
                spread_ns = max_ts_ns - min_ts_ns
                spread_ms = spread_ns / 1e6

                if spread_ms > max_frame_spread:
                    max_frame_spread = spread_ms
                    worst_frame_idx = frame_idx

                if spread_ms > 20:
                    frame_sync_errors.append(
                        {
                            "frame_idx": frame_idx,
                            "spread_ms": spread_ms,
                            "timestamps": frame_timestamps_ns,
                            "keys": frame_keys,
                        }
                    )

        if frame_sync_errors:
            error_msg = (
                f"发现 {len(frame_sync_errors)} 个时刻的多模态时间戳差值超过20ms阈值"
            )
            print(f"    ❌ {error_msg}")

            # 显示最严重的几个时刻
            sorted_errors = sorted(
                frame_sync_errors, key=lambda x: x["spread_ms"], reverse=True
            )
            for i, error in enumerate(sorted_errors[:3]):  # 只显示前3个最严重的
                frame_idx = error["frame_idx"]
                spread_ms = error["spread_ms"]
                timestamps_s = [ts_ns / 1e9 for ts_ns in error["timestamps"]]
                keys = error["keys"]

                print(f"      时刻{frame_idx}: 时间戳差值 {spread_ms:.1f}ms")
                for key, ts_s in zip(keys, timestamps_s):
                    print(f"        {key}: {ts_s:.6f}s")

            alignment_errors.append(
                f"多模态同步: {error_msg}，最大差值 {max_frame_spread:.1f}ms"
            )
        else:
            print(f"    ✓ 多模态时间戳同步验证通过，最大差值 {max_frame_spread:.1f}ms")

        # 如果有验证错误，抛出异常
        if alignment_errors:
            error_summary = "; ".join(alignment_errors)
            detailed_msg = (
                f"严格对齐验证失败:\n"
                f"- 最大帧内时间戳差值: {max_frame_spread:.1f}ms\n"
                f"- 验证错误: {error_summary}\n"
                f"- 参与验证的数据模态数: {len(valid_modalities)}\n"
                f"- 主时间戳长度: {len(main_timestamps)}"
            )

            print(f"[ERROR] {detailed_msg}")
            raise TimestampStuckError(
                message=f"严格对齐验证失败: {error_summary}",
                topic="strict_alignment_validation",
                stuck_timestamp=(
                    main_timestamps[worst_frame_idx] if worst_frame_idx >= 0 else None
                ),
                stuck_duration=max_frame_spread / 1000,
                stuck_frame_count=len(alignment_errors),
                threshold=0.02,
            )

        # 验证通过，输出总结
        print("=" * 60)
        print("✓ 严格对齐验证通过!")

        # === 更新：输出基本要求验证结果（包含频率） ===
        time_span = (
            main_timestamps[-1] - main_timestamps[0] if len(main_timestamps) > 1 else 0
        )
        actual_fps = len(main_timestamps) / time_span if time_span > 0 else 0

        print(
            f"  - 主时间戳长度: {len(main_timestamps)} 帧 (要求 >= {min_required_frames} )"
        )
        print(
            f"  - 主时间戳时间跨度: {time_span:.3f}s (要求 >= {min_required_duration}s)"
        )
        print(f"  - 主时间戳频率: {actual_fps:.2f}Hz (要求 29.95~30.05Hz)")

        # 统计不同类型的模态（排除外参和camera_info）
        regular_modalities = [
            k
            for k in aligned_data.keys()
            if (
                not k.endswith("_extrinsics")
                and not k.endswith("_camera_info")
                and len(aligned_data[k]) > 0
            )
        ]
        extrinsics_modalities = [
            k
            for k in aligned_data.keys()
            if k.endswith("_extrinsics") and len(aligned_data[k]) > 0
        ]
        camera_info_modalities = [
            k
            for k in aligned_data.keys()
            if k.endswith("_camera_info") and len(aligned_data[k]) > 0
        ]

        print(f"  - 参与验证的数据模态数: {len(valid_modalities)}")
        print(f"  - 跳过验证的外参模态数: {len(extrinsics_modalities)}")
        print(f"  - 跳过验证的相机信息模态数: {len(camera_info_modalities)}")
        print(f"  - 最大帧内时间戳差值: {max_frame_spread:.1f}ms")

        if len(main_timestamps) > 1:
            print(f"  - 主时间戳平均间隔: {np.mean(main_intervals_ms):.1f}ms")
            print(f"  - 主时间戳最大间隔: {max_main_interval:.1f}ms")

        if extrinsics_modalities:
            print(f"  - 外参模态: {extrinsics_modalities}")
        if camera_info_modalities:
            print(f"  - 相机信息模态: {camera_info_modalities}")

        print("=" * 60)

    def align_frame_data(self, data: dict):
        aligned_data = defaultdict(list)
        main_timeline = max(
            self.DEFAULT_CAMERA_NAMES, key=lambda cam_k: len(data.get(cam_k, []))
        )
        jump = self.MAIN_TIMELINE_FPS // self.TRAIN_HZ
        main_img_timestamps = [t["timestamp"] for t in data[main_timeline]][
            self.SAMPLE_DROP : -self.SAMPLE_DROP
        ][::jump]
        min_end = min(
            [data[k][-1]["timestamp"] for k in data.keys() if len(data[k]) > 0]
        )
        main_img_timestamps = [t for t in main_img_timestamps if t < min_end]
        for stamp in main_img_timestamps:
            stamp_sec = stamp
            for key, v in data.items():
                if len(v) > 0:
                    this_obs_time_seq = [this_frame["timestamp"] for this_frame in v]
                    time_array = np.array([t for t in this_obs_time_seq])
                    idx = np.argmin(np.abs(time_array - stamp_sec))
                    aligned_data[key].append(v[idx])
                else:
                    aligned_data[key] = []
        print(
            f"Aligned {key}: {len((data[main_timeline]))} -> {len(next(iter(aligned_data.values())))}"
        )
        for k, v in aligned_data.items():
            if len(v) > 0:
                print(v[0]["timestamp"], v[1]["timestamp"], k)
        return aligned_data

    def list_bag_files(self, bag_dir: str):
        return sorted(glob.glob(os.path.join(bag_dir, "*.bag")))

    def process_rosbag_dir(self, bag_dir: str):
        all_data = []
        # 按照文件名排序，获取 bag 文件列表
        bag_files = self.list_bag_files(bag_dir)
        episode_id = 0
        for bf in bag_files:
            print(f"Processing bag file: {bf}")
            episode_data = self.process_rosbag(bf)
            all_data.append(episode_data)

        return all_data


class PostProcessorUtils:

    @staticmethod
    def torque_to_current_batch(
        torque_data: np.ndarray,
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
    ):
        """
        将扭矩数据批量转换为电流数据

        Args:
            torque_data: 扭矩数据数组(N, M)
            MOTOR_C2T: 电流转扭矩系数数组，默认值为 kuavo-ros-control 中定义的系数
        Returns:
            电流数据数组
        """
        if torque_data.shape[1] != len(MOTOR_C2T):
            print(
                f"警告: 扭矩数据长度({torque_data.shape[1]})与C2T系数数量({len(MOTOR_C2T)})不匹配"
            )
            return None

        # 复制数据避免修改原始数据
        current_data = torque_data.copy()

        from itertools import chain

        # 13~18 为左臂ruiwo电机数据, 20~27 为右臂ruiwo电机数据
        # 对于这些电机需要先除以MOTOR_C2T系数再乘以2.1
        for i in chain(range(13, 19), range(20, 28)):  # 修正为27+1=28
            current_data[:, i] = (torque_data[:, i] / MOTOR_C2T[i]) * 2.1

        # 1, 2, 7, 8, 12, 19 号电机需要特殊处理
        for i in [1, 2, 7, 8, 12, 19]:
            current_data[:, i] = (torque_data[:, i] / MOTOR_C2T[i]) * 1.2

        # 其他电机：EC电机，直接除以MOTOR_C2T系数
        other_indices = [
            i
            for i in range(len(MOTOR_C2T))
            if i not in chain(range(13, 19), range(20, 28), [1, 2, 7, 8, 12, 19])
        ]
        for i in other_indices:
            current_data[:, i] = torque_data[:, i] / MOTOR_C2T[i]

        return current_data

    @staticmethod
    def current_to_torque(
        current_data: np.ndarray,
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
    ):
        """
        将 sensors_data_raw 中的 joint_torque 电流数据转换为扭矩数据

        Args:
            current_data: 电流数据数组(N, 28)
            MOTOR_C2T: 电流转扭矩系数数组，默认值为 kuavo-ros-control 中定义的系数
        Returns:
            扭矩数据数组
        """
        if len(current_data) != len(MOTOR_C2T):
            print(
                f"警告: 电流数据长度({len(current_data)})与C2T系数数量({len(MOTOR_C2T)})不匹配"
            )
            # 扩展或截断系数数组
            return None

        torque_data = []
        # "MOTORS_TYPE":[
        # "PA100_18", "PA100", "PA100", "PA100_18", "CK", "CK",
        # "PA100_18", "PA100", "PA100", "PA100_18", "CK", "CK",
        # "PA100", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo",
        # "PA100", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo", "ruiwo"],

        for i, current in enumerate(current_data):
            # kuavo-ros-control/src/kuavo_common/include/kuavo_common/common/kuavo_settings.h
            # 中定义了 ruiwo 电机电流转扭矩系数 CK_C2T = 2.1，所以这里除以 2.1 转化回原始电流

            # 13~18 为左臂ruiwo电机数据, 20~25 为右臂ruiwo电机数据
            # 对于这些电机需要先除以2.1转换回原始电流
            if 13 <= i <= 18 or 20 <= i <= 27:
                torque = (current / 2.1) * MOTOR_C2T[i]
            elif i == 1 or i == 2 or i == 7 or i == 8 or i == 12 or i == 19:
                torque = (current / 1.2) * MOTOR_C2T[i]
            else:

                # EC 电机 sensors_data_raw 中已经是扭矩值
                torque = current
            torque_data.append(torque)

        return np.array(torque_data)

    @staticmethod
    def current_to_torque_batch(
        current_data: np.ndarray,
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
    ):
        """
        将 sensors_data_raw 中的 joint_torque 电流数据转换为扭矩数据

        Args:
            current_data: 电流数据数组(N, M)
            MOTOR_C2T: 电流转扭矩系数数组，默认值为 kuavo-ros-control 中定义的系数
        Returns:
            扭矩数据数组
        """
        if current_data.shape[1] != len(MOTOR_C2T):
            print(
                f"警告: 电流数据长度({current_data.shape[1]})与C2T系数数量({len(MOTOR_C2T)})不匹配"
            )
            # 扩展或截断系数数组
            return None

        from itertools import chain

        for i in chain(range(13, 19), range(20, 28)):
            current_data[:, i] = current_data[:, i] / 2.1 * MOTOR_C2T[i]
        for i in [1, 2, 7, 8, 12, 19]:
            current_data[:, i] = current_data[:, i] / 1.2 * MOTOR_C2T[i]
        # 对于其他电机直接使用原始电流
        # EC 电机 sensors_data_raw 中已经是扭矩值
        return current_data

    @staticmethod
    def save_to_hdf5(low_dim_data, file_path):
        """将数据保存为符合库帕思通用版数据格式的HDF5文件"""
        import h5py

        # 确保输出目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        def create_datasets_recursively(group, data_dict, current_path=""):
            """递归创建数据集和组"""
            for key, value in data_dict.items():
                full_path = f"{current_path}/{key}" if current_path else key

                if isinstance(value, dict):
                    # 如果是字典，创建子组并递归处理
                    subgroup = group.create_group(key)
                    create_datasets_recursively(subgroup, value, full_path)
                else:
                    # 如果是数据，创建数据集
                    try:
                        # 处理不同类型的数据
                        if isinstance(value, (list, tuple)):
                            value = np.array(value)

                        # 根据数据类型和路径进行特殊处理
                        processed_value = process_data_by_path(value, full_path)

                        # 创建数据集
                        group.create_dataset(key, data=processed_value)
                        print(
                            f"创建数据集: {full_path}, 形状: {processed_value.shape}, 类型: {processed_value.dtype}"
                        )

                    except Exception as e:
                        print(f"警告: 无法创建数据集 {full_path}: {e}")
                        # 创建空数据集作为占位符
                        try:
                            empty_data = np.array([])
                            group.create_dataset(key, data=empty_data)
                        except:
                            pass

        def process_data_by_path(value, path):
            """根据数据路径对数据进行特殊处理"""
            # 时间戳处理 - 扩展识别新的时间戳字段
            timestamp_fields = [
                "timestamps",
                "head_color_mp4_camera_timestamps",
                "hand_left_color_mp4_timestamps",
                "hand_right_color_mp4_timestamps",
                "head_depth_mkv_camera_timestamps",
                "hand_left_depth_mkv_timestamps",
                "hand_right_depth_mkv_timestamps",
                "camera_extrinsics_timestamps",
                "head_timestamps" "joint_timestamps",
                "effector_dexhand_timestamps",
                "effector_lejuclaw_timestamps",
            ]

            if any(ts_field in path for ts_field in timestamp_fields):
                if value.dtype != np.int64:
                    # 转换时间戳为纳秒级整数
                    if np.issubdtype(value.dtype, np.floating):
                        return (value * 1e9).astype(np.int64)
                    else:
                        return value.astype(np.int64)
                return value

            # 索引数据处理
            elif "index" in path:
                return value.astype(np.int64)

            # 其他数值数据处理
            elif np.issubdtype(value.dtype, np.number):
                # 根据数据类型决定精度
                if np.issubdtype(value.dtype, np.integer):
                    return value.astype(np.int32)
                else:
                    return value.astype(np.float32)

            # 保持原始数据类型
            return value

        def add_missing_required_fields(f, low_dim_data):
            """添加库帕思格式中必需但缺失的字段，使用null机制"""

            # 获取时间戳长度作为参考
            if "timestamps" in low_dim_data:
                N = len(low_dim_data["timestamps"])
            else:
                N = 1000  # 默认值
                for key, value in low_dim_data.items():
                    if hasattr(value, "__len__") and not isinstance(value, str):
                        N = len(value)
                        break

            # 创建控制索引
            control_indices = np.arange(N, dtype=np.int64)

            def create_null_dataset(group, name, shape, dtype):
                """创建一个表示缺失数据的数据集"""
                # 方法1: 使用NaN表示缺失数据（仅适用于浮点数）
                if dtype == np.float32 or dtype == np.float64:
                    data = np.full(shape, np.nan, dtype=dtype)
                    dataset = group.create_dataset(name, data=data)
                    # 添加属性标记这是缺失数据
                    dataset.attrs["missing_data"] = True
                    dataset.attrs["description"] = f"Missing data filled with NaN"
                    return dataset

                # 方法2: 创建空数据集（对于整数类型）
                elif np.issubdtype(dtype, np.integer):
                    # 对于整数，使用最小值表示缺失
                    if dtype == np.int32:
                        fill_value = np.iinfo(np.int32).min
                    elif dtype == np.int64:
                        fill_value = np.iinfo(np.int64).min
                    else:
                        fill_value = -999999  # 默认缺失值

                    data = np.full(shape, fill_value, dtype=dtype)
                    dataset = group.create_dataset(name, data=data)
                    dataset.attrs["missing_data"] = True
                    dataset.attrs["fill_value"] = fill_value
                    dataset.attrs["description"] = (
                        f"Missing data filled with {fill_value}"
                    )
                    return dataset

                # 方法3: 不创建数据集，仅添加占位符属性
                else:
                    # 创建一个只有属性的组来表示缺失
                    missing_group = group.create_group(name + "_missing")
                    missing_group.attrs["missing_data"] = True
                    missing_group.attrs["expected_shape"] = shape
                    missing_group.attrs["expected_dtype"] = str(dtype)
                    missing_group.attrs["description"] = (
                        "Data not available - missing field"
                    )
                    return missing_group

            def create_optional_dataset(
                group, name, shape, dtype, description="Optional field not available"
            ):
                """创建可选的数据集，明确标记为不可用"""
                # 方法4: 创建虚拟数据集，长度为0
                empty_data = np.array([], dtype=dtype)
                dataset = group.create_dataset(name, data=empty_data, maxshape=shape)
                dataset.attrs["data_available"] = False
                dataset.attrs["expected_shape"] = shape
                dataset.attrs["description"] = description
                return dataset

            # 检查并添加缺失的 action 组字段
            if "action" in f:
                action_group = f["action"]

                # # 添加缺失的 robot 组
                # if "robot" not in action_group:
                #     robot_group = action_group.create_group("robot")
                #     create_null_dataset(robot_group, "velocity", (N, 2), np.float32)
                #     create_null_dataset(robot_group, "index", (N,), np.float32)
                #     print(f"添加缺失字段: action/robot (使用NaN表示缺失)")

                # # 添加缺失的 waist 组
                # if "waist" not in action_group:
                #     waist_group = action_group.create_group("waist")
                #     create_null_dataset(waist_group, "position", (N, 2), np.float32)
                #     create_null_dataset(waist_group, "index", (N,), np.float32)
                #     print(f"添加缺失字段: action/waist (使用NaN表示缺失)")

                # # 添加缺失的 end 组
                # if "end" not in action_group:
                #     end_group = action_group.create_group("end")
                #     create_null_dataset(end_group, "orientation", (N, 2, 4), np.float32)
                #     create_null_dataset(end_group, "position", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "index", (N,), np.float32)
                #     print(f"添加缺失字段: action/end (使用NaN表示缺失)")

            # 检查并添加缺失的 state 组字段
            if "state" in f:
                state_group = f["state"]

                # # 添加缺失的 end 组
                # if "end" not in state_group:
                #     end_group = state_group.create_group("end")
                #     create_null_dataset(end_group, "angular", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "orientation", (N, 2, 4), np.float32)
                #     create_null_dataset(end_group, "position", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "velocity", (N, 2, 3), np.float32)
                #     create_null_dataset(end_group, "wrench", (N, 2, 6), np.float32)
                #     print(f"添加缺失字段: state/end (使用NaN表示缺失)")

                # 添加缺失的 robot 组
                if "robot" not in state_group:
                    robot_group = state_group.create_group("robot")

                    # 对于机器人姿态，如果没有IMU数据，明确标记为缺失
                    if "imu" in low_dim_data and "quat_xyzw" in low_dim_data["imu"]:
                        imu_data_quat_xyzw = low_dim_data["imu"]["quat_xyzw"]
                        if (
                            hasattr(imu_data_quat_xyzw, "shape")
                            and len(imu_data_quat_xyzw.shape) > 1
                            and imu_data_quat_xyzw.shape[1] >= 4
                        ):
                            # 有IMU数据，直接使用
                            orientation = np.zeros((N, 4), dtype=np.float32)
                            orientation[:, :] = imu_data_quat_xyzw
                            dataset = robot_group.create_dataset(
                                "orientation", data=orientation
                            )
                            dataset.attrs["data_source"] = "IMU sensor"
                            dataset.attrs["missing_data"] = False
                            print(f"从IMU数据提取机器人姿态")
                        else:
                            # IMU数据格式不对，标记为缺失
                            create_null_dataset(
                                robot_group, "orientation", (N, 4), np.float32
                            )
                            print(f"IMU数据格式异常，姿态数据标记为缺失")
                    else:
                        # 没有IMU数据，标记为缺失
                        create_null_dataset(
                            robot_group, "orientation", (N, 4), np.float32
                        )
                        print(f"无IMU数据，姿态数据标记为缺失")

                    # 其他机器人状态标记为缺失
                    # create_null_dataset(robot_group, "orientation_drift", (N, 4), np.float32)
                    # create_null_dataset(robot_group, "position", (N, 3), np.float32)
                    # create_null_dataset(robot_group, "position_drift", (N, 3), np.float32)
                    print(f"添加缺失字段: state/robot (使用NaN/缺失值表示)")

                # # 添加缺失的 waist 组
                # if "waist" not in state_group:
                #     waist_group = state_group.create_group("waist")
                #     create_null_dataset(waist_group, "effort", (N, 2), np.float32)
                #     create_null_dataset(waist_group, "position", (N, 2), np.float32)
                #     create_null_dataset(waist_group, "velocity", (N, 2), np.float32)
                #     print(f"添加缺失字段: state/waist (使用NaN表示缺失)")

                # 为现有组添加缺失的数据集
                # if "effector" in state_group:
                #     effector_group = state_group["effector"]
                #     if "force" not in effector_group:
                #         create_null_dataset(effector_group, "force", (N, 2), np.float32)
                #         print(f"添加缺失字段: state/effector/force (使用NaN表示缺失)")

                if "head" in state_group:
                    head_group = state_group["head"]
                    if "effort" not in head_group:
                        create_null_dataset(head_group, "effort", (N, 2), np.float32)
                        print(f"添加缺失字段: state/head/effort (使用NaN表示缺失)")

                if "joint" in state_group:
                    joint_group = state_group["joint"]
                    # 获取关节数量
                    joint_count = 14  # 默认值
                    if "position" in joint_group:
                        joint_count = joint_group["position"].shape[1]
                    elif "velocity" in joint_group:
                        joint_count = joint_group["velocity"].shape[1]

                    if "current_value" not in joint_group:
                        create_null_dataset(
                            joint_group, "current_value", (N, joint_count), np.float32
                        )
                        print(
                            f"添加缺失字段: state/joint/current_value (使用NaN表示缺失)"
                        )

                    if "effort" not in joint_group:
                        create_null_dataset(
                            joint_group, "effort", (N, joint_count), np.float32
                        )
                        print(f"添加缺失字段: state/joint/effort (使用NaN表示缺失)")

            # 添加 other_sensors 组（标记为可选）
            # if "other_sensors" not in f:
            #     other_group = f.create_group("other_sensors")
            #     other_group.attrs['description'] = 'Optional sensor data - currently empty'
            #     other_group.attrs['data_available'] = False
            #     print(f"添加缺失字段: other_sensors (标记为可选数据)")
            # 新增：在根级别添加时间戳字段的存在性信息

        # 创建 HDF5 文件
        with h5py.File(file_path, "w") as f:
            print(f"开始创建HDF5文件: {file_path}")

            # 递归创建所有数据集和组
            create_datasets_recursively(f, low_dim_data)

            # 添加库帕思格式要求的缺失字段填充为NaN或缺失值
            add_missing_required_fields(f, low_dim_data)

        print(f"数据已成功保存为HDF5格式: {file_path}")
        return file_path


if __name__ == "__main__":
    # 创建测试实例
    print("创建测试实例...")

    # 模拟配置
    class TestConfig:
        def __init__(self):
            self.default_camera_names = ["head_cam_h"]
            self.train_hz = 30
            self.main_timeline_fps = 30
            self.sample_drop = 0
            self.resize = type("obj", (object,), {"width": 640, "height": 480})()
            self.topics = []
            self.eef_type = "dexhand"

    config = TestConfig()
    reader = KuavoRosbagReader(config)

    # 创建测试数据：模拟前置步骤处理后的数据特征
    # - 所有间隔都小于40ms
    # - 总帧率约32Hz（需要删除帧降到30Hz）
    print("创建测试数据...")

    # 生成32Hz的基本时间戳序列
    base_interval = 1.0 / 32.0  # 32Hz = 31.25ms间隔
    total_frames = 800  # 足够长的数据
    total_duration = total_frames * base_interval  # 总时长

    # 创建均匀的32Hz时间戳作为基础
    uniform_timestamps = np.linspace(1.0, 1.0 + total_duration, total_frames)

    # 添加一些随机性，但确保间隔始终<40ms
    timestamps = []
    for i in range(total_frames):
        base_ts = uniform_timestamps[i]

        if i == 0:
            # 第一帧保持不变
            timestamps.append(base_ts)
        else:
            # 添加随机偏移，但确保与前一帧的间隔在15-38ms之间
            prev_ts = timestamps[-1]
            min_interval = 0.015  # 15ms
            max_interval = 0.038  # 38ms

            # 计算理想的下一个时间戳
            ideal_next = prev_ts + base_interval

            # 添加随机偏移，但限制在安全范围内
            random_offset = np.random.uniform(-0.008, 0.008)  # ±8ms随机偏移
            candidate_ts = ideal_next + random_offset

            # 确保间隔在安全范围内
            actual_interval = candidate_ts - prev_ts
            if actual_interval < min_interval:
                candidate_ts = prev_ts + min_interval
            elif actual_interval > max_interval:
                candidate_ts = prev_ts + max_interval

            timestamps.append(candidate_ts)

    main_timestamps = np.array(timestamps)

    # 验证生成的时间戳质量
    intervals_ms = np.diff(main_timestamps) * 1000
    max_interval_ms = np.max(intervals_ms)
    min_interval_ms = np.min(intervals_ms)
    avg_interval_ms = np.mean(intervals_ms)

    # 确保所有间隔都小于40ms
    assert (
        max_interval_ms < 40.0
    ), f"生成的最大间隔 {max_interval_ms:.1f}ms 超过40ms限制"

    # 子时间戳：比主时间戳晚2ms
    child_timestamps = main_timestamps + 0.002

    # 创建对应的数据
    valid_modalities = {
        "head_cam_h": [
            {"timestamp": ts, "data": f"main_frame_{i}", "frame_id": i}
            for i, ts in enumerate(main_timestamps)
        ],
        "child_sensor": [
            {"timestamp": ts, "data": f"child_data_{i}", "sensor_value": i * 10}
            for i, ts in enumerate(child_timestamps)
        ],
    }

    # 计算初始帧率
    time_span = main_timestamps[-1] - main_timestamps[0]
    initial_fps = len(main_timestamps) / time_span
    target_fps = 30.095

    print(f"测试数据创建完成:")
    print(f"  主时间戳长度: {len(main_timestamps)}")
    print(f"  时间跨度: {time_span:.3f}s")
    print(f"  初始帧率: {initial_fps:.2f}Hz")
    print(f"  目标帧率: {target_fps:.2f}Hz")
    print(f"  需要删除约 {len(main_timestamps) - int(time_span * target_fps)} 帧")

    print(f"  时间间隔统计:")
    print(f"    平均间隔: {avg_interval_ms:.1f}ms")
    print(f"    最大间隔: {max_interval_ms:.1f}ms")
    print(f"    最小间隔: {min_interval_ms:.1f}ms")
    print(f"    ✓ 所有间隔都在40ms以内（模拟前置处理完成）")

    # 验证帧率合理性
    if 31.5 <= initial_fps <= 33.0:
        print(f"    ✓ 初始帧率 {initial_fps:.2f}Hz 在期望范围内（31.5-33Hz）")
    else:
        print(f"    ⚠️ 初始帧率 {initial_fps:.2f}Hz 不在期望范围内")

    print("\n初始数据样本（前10帧）:")
    print("主时间戳:")
    for i in range(min(10, len(main_timestamps))):
        interval_ms = 0
        if i > 0:
            interval_ms = (main_timestamps[i] - main_timestamps[i - 1]) * 1000
        print(f"  帧{i}: {main_timestamps[i]:.6f}s (间隔: {interval_ms:.1f}ms)")

    print("\n子时间戳样本（前5帧）:")
    for i in range(min(5, len(valid_modalities["child_sensor"]))):
        item = valid_modalities["child_sensor"][i]
        main_ts = main_timestamps[i]
        diff_ms = (item["timestamp"] - main_ts) * 1000
        print(
            f"  帧{i}: {item['timestamp']:.6f}s (与主时间戳差: {diff_ms:.1f}ms, 数据: {item['data']})"
        )

    print("\n开始测试 _remove_frames_to_decrease_fps...")
    print("=" * 60)

    try:
        # 调用函数进行测试
        result_timestamps, result_modalities = reader._remove_frames_to_decrease_fps(
            main_timestamps.copy(),  # 使用副本避免修改原数据
            {k: list(v) for k, v in valid_modalities.items()},  # 深拷贝
            target_fps,
            time_span,
        )

        print("=" * 60)
        print("测试完成!")

        # 验证结果
        final_time_span = result_timestamps[-1] - result_timestamps[0]
        final_fps = len(result_timestamps) / final_time_span

        print(f"\n结果统计:")
        print(f"  最终时间戳长度: {len(result_timestamps)}")
        print(f"  最终时间跨度: {final_time_span:.3f}s")
        print(f"  最终帧率: {final_fps:.3f}Hz")
        print(f"  删除帧数: {len(main_timestamps) - len(result_timestamps)}")

        # 验证最终时间戳质量
        if len(result_timestamps) > 1:
            final_intervals_ms = np.diff(result_timestamps) * 1000
            max_final_interval = np.max(final_intervals_ms)
            avg_final_interval = np.mean(final_intervals_ms)
            std_final_interval = np.std(final_intervals_ms)

            print(f"\n最终时间戳质量:")
            print(f"  最大间隔: {max_final_interval:.1f}ms")
            print(f"  平均间隔: {avg_final_interval:.1f}ms")
            print(f"  间隔标准差: {std_final_interval:.1f}ms")

            if max_final_interval <= 40:
                print(f"  ✓ 所有间隔都在40ms以内")
            else:
                large_final_intervals = np.sum(final_intervals_ms > 40)
                print(f"  ❌ 仍有 {large_final_intervals} 个间隔超过40ms")

        # 验证子时间戳同步性（抽样检查）
        print(f"\n子时间戳同步性验证（抽样检查前20帧）:")
        sync_errors = 0
        check_frames = min(
            20, len(result_timestamps), len(result_modalities["child_sensor"])
        )

        for i in range(check_frames):
            main_ts = result_timestamps[i]
            child_item = result_modalities["child_sensor"][i]
            child_ts = child_item["timestamp"]
            expected_diff = 0.002  # 原始2ms差值
            actual_diff = child_ts - main_ts
            diff_error = abs(actual_diff - expected_diff) * 1000

            if diff_error > 0.1:  # 0.1ms容差
                sync_errors += 1
                print(f"  帧{i}: 同步偏差 {diff_error:.3f}ms")

                # 检查是否是重新平均过的帧
                reaveraged = child_item.get("timestamp_reaveraged", False)
                if reaveraged:
                    delta = child_item.get("timestamp_delta", 0)
                    print(f"        (该帧已重新平均, delta: {delta:.6f}s)")

        if sync_errors == 0:
            print(f"  ✓ 抽样检查的 {check_frames} 帧都保持了2ms的相对关系")
        else:
            print(f"  ❌ 在 {check_frames} 帧中发现 {sync_errors} 个同步偏差")

        # 显示处理前后的对比
        print(f"\n处理前后对比:")
        print(
            f"  长度: {len(main_timestamps)} -> {len(result_timestamps)} (-{len(main_timestamps) - len(result_timestamps)})"
        )
        print(f"  帧率: {initial_fps:.2f}Hz -> {final_fps:.2f}Hz")
        print(f"  最大间隔: {max_interval_ms:.1f}ms -> {max_final_interval:.1f}ms")

        # 测试结论
        length_ok = len(result_timestamps) >= 300  # 最终长度足够
        fps_ok = final_fps <= target_fps  # 帧率达标
        interval_ok = max_final_interval <= 40  # 间隔达标
        sync_ok = sync_errors == 0  # 同步达标

        print(f"\n测试结论:")
        print(
            f"  长度检查: {'✅' if length_ok else '❌'} ({len(result_timestamps)} >= 300)"
        )
        print(
            f"  帧率检查: {'✅' if fps_ok else '❌'} ({final_fps:.2f} <= {target_fps})"
        )
        print(
            f"  间隔检查: {'✅' if interval_ok else '❌'} ({max_final_interval:.1f} <= 40ms)"
        )
        print(f"  同步检查: {'✅' if sync_ok else '❌'}")

        if length_ok and fps_ok and interval_ok and sync_ok:
            print(f"  🎉 所有测试通过！滑动窗口删除+重新平均算法工作正常")
        else:
            print(f"  ⚠️ 部分测试未通过，需要进一步优化")

        # 额外检查：验证删除+重新平均的效果
        print(f"\n重新平均效果验证:")
        reaveraged_count = 0
        for item in result_modalities["child_sensor"]:
            if item.get("timestamp_reaveraged", False):
                reaveraged_count += 1

        if reaveraged_count > 0:
            print(f"  ✓ 共有 {reaveraged_count} 帧经过重新平均处理")
            print(
                f"  ✓ 重新平均比例: {reaveraged_count/len(result_modalities['child_sensor'])*100:.1f}%"
            )
        else:
            print(f"  ⚠️ 没有帧经过重新平均处理")

    except Exception as e:
        print(f"❌ 测试失败，出现异常:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {str(e)}")
        import traceback

        traceback.print_exc()
