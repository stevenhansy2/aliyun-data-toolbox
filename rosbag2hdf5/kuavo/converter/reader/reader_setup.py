from collections import defaultdict
from copy import deepcopy

import numpy as np
import rosbag
import rospy
import time

from converter.reader.msg_processor import KuavoMsgProcesser
from converter.kinematics.kuavo_pose_calculator import extract_camera_poses_from_bag_with_time


class ReaderSetupMixin:
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
        # 流式参数
        self.video_queue_limit = getattr(config, "video_queue_limit", 300)
        self.lowdim_batch_size = getattr(config, "lowdim_batch_size", 5000)
        self.video_workers = getattr(config, "video_workers", 6)

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
                log_print(f"[INFO] {camera}: 选择未压缩深度话题 {depth_topic_uncompressed}")
                self._topic_process_map[f"{camera}_depth"] = {
                    "topic": depth_topic_uncompressed,
                    "msg_process_fn": self._msg_processer.process_depth_image_16U,
                    "fallback_topic": depth_topic_compressed,
                    "fallback_fn": self._msg_processer.process_depth_image,
                }
            elif depth_topic_compressed in self.TOPICS:
                log_print(f"[INFO] {camera}: 仅找到压缩深度话题 {depth_topic_compressed}")
                self._topic_process_map[f"{camera}_depth"] = {
                    "topic": depth_topic_compressed,
                    "msg_process_fn": self._msg_processer.process_depth_image,
                }
            else:
                log_print(f"[WARN] {camera} 未找到深度话题（未压缩或压缩）")

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
            log_print(f"测试joint_current可用性时出错: {e}")
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
            urdf_path = "./assets/urdf/biped_s45.urdf"  # 默认路径

        # 检查话题
        bag = self.load_raw_rosbag(bag_file)
        bag_topics = set([t for t in bag.get_type_and_topic_info().topics])
        bag.close()
        if "/sensors_data_raw" not in bag_topics:
            return {}

        # 带时间裁剪的外参提取（外参为必需，异常直接抛出）
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
            log_print("使用 joint_current 话题")
        else:
            joint_current_processor = self._msg_processer.process_joint_torque_state
            log_print("使用 joint_torque 话题")

        # 自动适配手状态话题
        actual_hand_state_topic = self._find_actual_hand_state_topic(bag_file)
        if actual_hand_state_topic:
            log_print(f"使用手部状态话题: {actual_hand_state_topic}")
        else:
            log_print("[WARN] 未找到手部状态话题，将不会读取手部状态数据。")
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
        min_duration: float = 5.0,
        is_align: bool = True,
        return_raw: bool = False,
    ):
        t_start = time.time()
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
        fallback_data = {}

        # Get bag start time and duration
        bag_start = bag.get_start_time()
        bag_end = bag.get_end_time()
        bag_duration = bag_end - bag_start

        # Calculate absolute start/end times
        abs_start = bag_start + start_time * bag_duration
        abs_end = bag_start + end_time * bag_duration
        import rospy

        # 构建一次性遍历的 topic 列表
        topic_to_keys = defaultdict(list)
        fallback_topic_to_keys = defaultdict(list)
        all_topics = set()
        for key, topic_info in self._topic_process_map.items():
            main_topic = topic_info["topic"]
            all_topics.add(main_topic)
            topic_to_keys[main_topic].append((key, topic_info["msg_process_fn"]))
            data[key] = []
            if "fallback_topic" in topic_info and "fallback_fn" in topic_info:
                fb = topic_info["fallback_topic"]
                fallback_topic_to_keys[fb].append((key, topic_info["fallback_fn"]))
                all_topics.add(fb)

        log_print(f"[ROS] 单次遍历读取话题: {sorted(all_topics)}")

        for topic, msg, t in bag.read_messages(
            topics=list(all_topics),
            start_time=rospy.Time.from_sec(abs_start),
            end_time=rospy.Time.from_sec(abs_end),
        ):
            if topic in topic_to_keys:
                for key, fn in topic_to_keys[topic]:
                    msg_data = fn(msg)
                    msg_data["timestamp"] = t.to_sec()
                    data[key].append(msg_data)
            elif topic in fallback_topic_to_keys:
                for key, fn in fallback_topic_to_keys[topic]:
                    msg_data = fn(msg)
                    msg_data["timestamp"] = t.to_sec()
                    fallback_data.setdefault(key, []).append(msg_data)

        # 回退处理：主 topic 为空且有 fallback 数据时使用 fallback
        for key, topic_info in self._topic_process_map.items():
            if len(data.get(key, [])) == 0:
                fb_topic = topic_info.get("fallback_topic")
                if fb_topic and key in fallback_data:
                    log_print(
                        f"[WARN] {topic_info['topic']} 未读取到数据，使用回退 {fb_topic} 结果"
                    )
                    data[key] = fallback_data[key]
        # for key, topic_info in self._topic_process_map.items():
        #     topic = topic_info["topic"]
        #     msg_process_fn = topic_info["msg_process_fn"]
        #     data[key] = []
        #     for _, msg, t in bag.read_messages(
        #         topics=topic,
        #         start_time=rospy.Time.from_sec(abs_start),
        #         end_time=rospy.Time.from_sec(abs_end)
        #     ):
        #         msg_data = msg_process_fn(msg)
        #         # 如果没有 header.stamp或者时间戳是远古时间不合要求，使用bag的时间戳
        #         correct_timestamp = t.to_sec()
        #         msg_data["timestamp"] = correct_timestamp
        #         data[key].append(msg_data)
        extrinsics = self.extract_and_format_camera_extrinsics(
            bag_file, abs_start, abs_end
        )
        data.update(extrinsics)

        # 新增：末端执行器位姿计算
        joint_q_items = data.get("observation.sensorsData.joint_q", [])
        if joint_q_items:
            from converter.kinematics.endeffector_pose_from_bag import extract_and_format_eef_extrinsics

            joint_q_list = [item["data"] for item in joint_q_items]
            timestamps = [item["timestamp"] for item in joint_q_items]
            positions, quaternions = extract_and_format_eef_extrinsics(
                [{"joint_q": q} for q in joint_q_list],
                urdf_path="./assets/urdf/biped_s49.urdf",
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
        else:
            data["end.position"] = []
            data["end.orientation"] = []

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

        if is_align:
            raw_snapshot = deepcopy(data) if return_raw else None
            aligned = self.align_frame_data_optimized(
                data,
                drop_head,
                drop_tail,
                action_config=action_config,
                min_duration=min_duration,
            )
            if return_raw:
                log_print(
                    f"[ROS] 对齐完成，返回对齐+原始，耗时 {time.time()-t_start:.2f}s"
                )
                return aligned, raw_snapshot
            log_print(f"[ROS] 对齐完成，耗时 {time.time()-t_start:.2f}s")
            return aligned
        log_print(f"[ROS] 非对齐模式完成，耗时 {time.time()-t_start:.2f}s")
        return data
