#!/usr/bin/env python3
"""
优化了数据集对齐逻辑。

"""
import numpy as np
import rosbag
from pprint import pprint
import os
import glob
from collections import defaultdict
import time
from converter.configs import Config, load_config_from_json
from converter.configs.joint_names import (
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_DEXHAND_JOINT_NAMES,
    DEFAULT_HEAD_JOINT_NAMES,
    DEFAULT_JOINT_NAMES,
    DEFAULT_JOINT_NAMES_LIST,
    DEFAULT_LEG_JOINT_NAMES,
    DEFAULT_LEJUCLAW_JOINT_NAMES,
)
from converter.kinematics.kuavo_pose_calculator import (
    extract_camera_poses_from_bag,
    extract_camera_poses_from_bag_with_time,
)
from converter.reader.camera_topic_builder import build_camera_topic_process_map
from converter.reader.eef_builder import attach_eef_pose_from_joint_q
from converter.reader.source_topic_probe import (
    find_actual_hand_state_topic,
    test_joint_current_availability,
)
from converter.reader.topic_map_builder import build_main_topic_map
from converter.reader.message_processor import KuavoMsgProcesser
from converter.reader.postprocess_utils import PostProcessorUtils
from converter.reader.alignment_validation import final_alignment_validation
from converter.reader.alignment_postprocess import (
    detect_and_trim_aligned_data,
    fix_multimodal_start_alignment,
    fix_severely_stuck_timestamps,
    trim_aligned_data_by_frames,
)
from converter.reader.alignment_engine import align_frame_data_optimized as align_frame_data_optimized_impl
from converter.reader.frame_rate_adjust import (
    adjust_frame_rate_to_30fps,
    insert_frames_to_increase_fps,
    remove_frames_to_decrease_fps,
)
from converter.reader.on_demand_interpolation import (
    create_interpolated_data_point,
    interpolate_on_demand,
)
from converter.reader.timestamp_preprocess import (
    check_actual_time_gaps,
    find_closest_indices_vectorized,
    interpolate_timestamps_and_data,
    preprocess_timestamps_and_data,
    preprocess_timestamps_only_deduplicate,
    remove_duplicate_timestamps,
)
from converter.reader.timestamp_quality import validate_timestamp_quality
from converter.reader.timestamp_ops import (
    execute_window_removal_and_reaverage,
    reaverage_timestamps_in_window,
)
from converter.reader.timeline_prescan import (
    enforce_batch_continuity_main,
    prescan_main_timeline,
)
from converter.reader.streaming_pipeline import (
    compute_eef_poses_with_cached_calculator,
    prescan_and_prepare_batches,
    process_rosbag as process_rosbag_streaming,
    process_rosbag_parallel as process_rosbag_parallel_impl,
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


from dataclasses import dataclass, field


@dataclass
class StreamingAlignmentState:
    initialized: bool = False
    batch_index: int = 0
    last_main_timestamp: float | None = None
    per_key_last_timestamp: dict = field(default_factory=dict)

    def update(self, main_ts: np.ndarray, aligned: dict):
        if len(main_ts) > 0:
            self.last_main_timestamp = float(main_ts[-1])
        for k, v in aligned.items():
            if isinstance(v, list) and v:
                self.per_key_last_timestamp[k] = v[-1]["timestamp"]
        self.batch_index += 1


def _parallel_rosbag_worker(args: dict, result_queue, worker_id: int):
    """Compatibility wrapper; real implementation lives in parallel_worker.py."""
    from converter.reader.parallel_worker import parallel_rosbag_worker

    return parallel_rosbag_worker(args, result_queue, worker_id)


class KuavoRosbagReader:
    def __init__(self, config, use_depth=False):
        self.config = config
        self.DEFAULT_CAMERA_NAMES = config.default_camera_names
        self.TRAIN_HZ = config.train_hz
        self.MAIN_TIMELINE_FPS = config.main_timeline_fps
        self.SAMPLE_DROP = config.sample_drop
        self._msg_processer = KuavoMsgProcesser(
            config.resize.width, config.resize.height
        )
        self.TOPICS = config.topics
        self.EEF_TYPE = config.eef_type
        self.HAND_STATE_TOPICS = config.hand_state_topics
        self.MAIN_TIMESTAMP_TOPIC = getattr(config, "main_timeline_key", "camera_top")
        self.TIME_TOLERANCE = 180
        self.main_topic_map = None
        self._pose_calculator_cache = None
        self.USE_DEPTH = use_depth
        self.cam_map = config.default_cameras2topics
        self.camera_topic_specs = getattr(config, "camera_topic_specs", {}) or {}
        self.source_topics = getattr(config, "source_topics", {}) or {}
        self.urdf_path = getattr(
            config, "urdf_path", "./kuavo/assets/urdf/biped_s45.urdf"
        )

        # 动态构建topic处理映射
        self._topic_process_map = build_camera_topic_process_map(
            self.DEFAULT_CAMERA_NAMES,
            self.camera_topic_specs,
            self.cam_map,
            self.TOPICS,
            self.USE_DEPTH,
            self._msg_processer,
        )

    def _find_actual_hand_state_topic(self, bag_file):
        """自动检测实际存在的手状态话题"""
        return find_actual_hand_state_topic(bag_file, self.HAND_STATE_TOPICS)

    def _test_joint_current_availability(self, bag_file):
        """测试bag文件中的/sensors_data_raw消息是否有joint_current字段"""
        sensors_topic = self.source_topics.get("sensors_data_raw", "/sensors_data_raw")
        try:
            return test_joint_current_availability(bag_file, sensors_topic)
        except Exception as e:
            print(f"测试joint_current可用性时出错: {e}")
            return False

    def extract_and_format_camera_extrinsics(
        self, bag_file, abs_start=None, abs_end=None
    ):
        """提取并格式化相机外参，支持时间裁剪"""
        urdf_path = getattr(self, "urdf_path", "./kuavo/assets/urdf/biped_s45.urdf")

        # 检查话题
        bag = self.load_raw_rosbag(bag_file)
        bag_topics = set([t for t in bag.get_type_and_topic_info().topics])
        bag.close()
        sensors_topic = self.source_topics.get("sensors_data_raw", "/sensors_data_raw")
        if sensors_topic not in bag_topics:
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
        hand_state_candidates = self.source_topics.get("hand_state_candidates", [])
        default_primary_hand_state = (
            hand_state_candidates[0]
            if isinstance(hand_state_candidates, list) and hand_state_candidates
            else "/control_robot_hand_position_state"
        )

        use_joint_current = self._test_joint_current_availability(bag_file)
        if use_joint_current:
            joint_current_processor = self._msg_processer.process_joint_current_state
            print("使用 joint_current 话题")
        else:
            joint_current_processor = self._msg_processer.process_joint_torque_state
            print("使用 joint_torque 话题")

        actual_hand_state_topic = self._find_actual_hand_state_topic(bag_file)
        if actual_hand_state_topic:
            print(f"使用手部状态话题: {actual_hand_state_topic}")
        else:
            print("[WARN] 未找到手部状态话题，将不会读取手部状态数据。")

        return build_main_topic_map(
            self._msg_processer,
            self.source_topics,
            joint_current_processor=joint_current_processor,
            actual_hand_state_topic=actual_hand_state_topic,
            default_primary_hand_state=default_primary_hand_state,
        )

    def load_raw_rosbag(self, bag_file: str):
        return rosbag.Bag(bag_file)

    def print_bag_info(self, bag: rosbag.Bag):
        pprint(bag.get_type_and_topic_info().topics)

    def _enforce_batch_continuity_main(
        self,
        streaming_state: StreamingAlignmentState,
        main_timestamps: np.ndarray,
        target_interval: float = 0.033,
        tolerance: float = 0.008,
    ) -> np.ndarray:
        return enforce_batch_continuity_main(
            self,
            streaming_state,
            main_timestamps,
            target_interval=target_interval,
            tolerance=tolerance,
        )

    def _prescan_main_timeline(
        self,
        bag_file: str,
        abs_start: float,
        abs_end: float,
    ) -> np.ndarray:
        return prescan_main_timeline(self, bag_file, abs_start, abs_end)

    def _build_topic_handlers(self):
        """
        将 _topic_process_map 反向构建为 topic->handlers 的映射，支持fallback选择。
        返回:
          - topic_to_handlers: { topic: [ {key, fn, is_fallback} , ...] }
          - topics_to_read: 所有需要订阅的topic列表（含fallback）
          - key_channel_choice: { key: None/'primary'/'fallback' } 初始均为None，首帧决定通道
        """
        topic_to_handlers = {}
        topics_to_read = set()
        key_channel_choice = {}

        for key, info in self._topic_process_map.items():
            # primary
            primary_topic = info["topic"]
            primary_fn = info["msg_process_fn"]
            topics_to_read.add(primary_topic)
            topic_to_handlers.setdefault(primary_topic, []).append(
                {"key": key, "fn": primary_fn, "is_fallback": False}
            )
            key_channel_choice[key] = None

            # fallback（若有）
            if "fallback_topic" in info and "fallback_fn" in info:
                fb_topic = info["fallback_topic"]
                fb_fn = info["fallback_fn"]
                topics_to_read.add(fb_topic)
                topic_to_handlers.setdefault(fb_topic, []).append(
                    {"key": key, "fn": fb_fn, "is_fallback": True}
                )

        return topic_to_handlers, list(topics_to_read), key_channel_choice

    def get_or_create_pose_calculator(self, urdf_path=None):
        """获取或创建URDF位姿计算器（单例模式）

        Args:
            urdf_path: URDF文件路径

        Returns:
            KuavoPoseCalculator实例
        """
        if urdf_path is None:
            urdf_path = self.urdf_path
        if self._pose_calculator_cache is None:
            from converter.kinematics.endeffector_pose import KuavoPoseCalculator

            print(f"[CACHE] 首次创建URDF计算器: {urdf_path}")
            self._pose_calculator_cache = KuavoPoseCalculator(urdf_path)
        return self._pose_calculator_cache

    def _compute_eef_poses_with_cached_calculator(self, pose_calculator, joint_q_list):
        return compute_eef_poses_with_cached_calculator(
            self, pose_calculator, joint_q_list
        )

    def process_rosbag(
        self,
        bag_file: str,
        start_time: float = 0,
        end_time: float = 1,
        action_config=None,
        chunk_size: int = 200,
    ):
        yield from process_rosbag_streaming(
            self,
            bag_file,
            start_time,
            end_time,
            action_config,
            chunk_size,
            streaming_state_cls=StreamingAlignmentState,
            attach_eef_pose_fn=attach_eef_pose_from_joint_q,
        )

    def _prescan_and_prepare_batches(
        self,
        bag_file: str,
        start_time: float,
        end_time: float,
        chunk_size: int = 200,
    ):
        return prescan_and_prepare_batches(
            self, bag_file, start_time, end_time, chunk_size
        )

    def process_rosbag_parallel(
        self,
        bag_file: str,
        start_time: float = 0,
        end_time: float = 1,
        action_config=None,
        chunk_size: int = 200,
        num_workers: int = 2,
    ):
        yield from process_rosbag_parallel_impl(
            self,
            bag_file,
            start_time,
            end_time,
            action_config,
            chunk_size,
            num_workers,
            parallel_worker_fn=_parallel_rosbag_worker,
        )

    def _serialize_topic_process_map(self):
        """
        序列化 topic_process_map 为可 pickle 的格式。
        因为函数对象不能直接 pickle，我们传递函数名称。
        """
        result = {}
        for key, info in self._topic_process_map.items():
            result[key] = {
                "topic": info["topic"],
                "fn_name": info["msg_process_fn"].__name__,
            }
            if "fallback_topic" in info:
                result[key]["fallback_topic"] = info["fallback_topic"]
                result[key]["fallback_fn_name"] = info["fallback_fn"].__name__
        return result

    def find_closest_indices_vectorized(self, timestamps, target_timestamps):
        """向量化查找最近时间戳索引"""
        return find_closest_indices_vectorized(timestamps, target_timestamps)

    def _preprocess_timestamps_only_deduplicate(self, data: dict) -> dict:
        """预处理时间戳和数据：只去重和检测卡顿，不插值（按需插值策略）"""
        return preprocess_timestamps_only_deduplicate(
            data=data,
            time_tolerance=self.TIME_TOLERANCE,
            error_cls=TimestampStuckError,
        )

    def _preprocess_timestamps_and_data(self, data: dict) -> dict:
        """预处理时间戳和数据：去重、检测实际卡顿、插值"""
        return preprocess_timestamps_and_data(
            data=data,
            time_tolerance=self.TIME_TOLERANCE,
            create_interpolated_data_point=self._create_interpolated_data_point,
            error_cls=TimestampStuckError,
        )

    def _check_actual_time_gaps(
        self, data_list: list, key: str, max_gap_duration: float = 2.0
    ):
        """检测去重后数据的实际时间间隔卡顿"""
        return check_actual_time_gaps(
            data_list=data_list,
            key=key,
            max_gap_duration=max_gap_duration,
            error_cls=TimestampStuckError,
        )

    def _remove_duplicate_timestamps(self, data_list: list, key: str) -> list:
        """去除重复时间戳及对应数据（使用纳秒精度）"""
        return remove_duplicate_timestamps(data_list, key)

    def _interpolate_timestamps_and_data(self, data_list: list, key: str) -> list:
        """时间戳插值和数据填充（修复版本 - 严格控制间隔，超过2秒直接抛异常）"""
        return interpolate_timestamps_and_data(
            data_list=data_list,
            key=key,
            time_tolerance=self.TIME_TOLERANCE,
            create_interpolated_data_point=self._create_interpolated_data_point,
            error_cls=TimestampStuckError,
        )

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
        按需插值：只对误差 >10ms 的帧进行插值修正（向量化版本）

        策略：
        1. 如果原始数据中找不到 <10ms 的帧
        2. 尝试在相邻帧之间找更近的（复制最近帧）
        3. 如果仍然无法满足 <10ms，保持原选择并记录警告
        """
        return interpolate_on_demand(
            aligned_data=aligned_data,
            time_errors_ms=time_errors_ms,
            original_data_list=original_data_list,
            original_timestamps=original_timestamps,
            target_timestamps=target_timestamps,
            key=key,
        )

    def _create_interpolated_data_point(
        self, reference_item: dict, new_timestamp: float, data_type: str
    ) -> dict:
        """创建插值数据点"""
        return create_interpolated_data_point(reference_item, new_timestamp, data_type)

    def _validate_timestamp_quality(self, timestamps: np.ndarray, data_name: str):
        """验证时间戳质量（使用纳秒精度）- 增强版本"""
        return validate_timestamp_quality(
            timestamps=timestamps,
            data_name=data_name,
            error_cls=TimestampStuckError,
        )

    def align_frame_data_optimized(
        self,
        data: dict,
        drop_head: bool,
        drop_tail: bool,
        action_config=None,
        streaming_state: StreamingAlignmentState | None = None,
        external_main_timestamps: np.ndarray | None = None,
    ):
        return align_frame_data_optimized_impl(
            self,
            data,
            drop_head,
            drop_tail,
            action_config,
            streaming_state,
            external_main_timestamps,
        )

    def _detect_and_trim_aligned_data(
        self, aligned_data: dict, main_timestamps: np.ndarray, action_config=None
    ):
        """
        检测并裁剪对齐后数据中的静止区域，头尾裁剪上限由首尾动作持续帧数的一半决定
        """
        return detect_and_trim_aligned_data(
            aligned_data=aligned_data,
            main_timestamps=main_timestamps,
            action_config=action_config,
            default_camera_names=self.DEFAULT_CAMERA_NAMES,
            train_hz=self.TRAIN_HZ,
        )

    def _trim_aligned_data_by_frames(
        self,
        aligned_data: dict,
        main_timestamps: np.ndarray,
        head_trim_frames: int,
        tail_trim_frames: int,
    ):
        """按帧数裁剪对齐后的数据"""
        return trim_aligned_data_by_frames(
            aligned_data=aligned_data,
            main_timestamps=main_timestamps,
            head_trim_frames=head_trim_frames,
            tail_trim_frames=tail_trim_frames,
        )

    def _fix_multimodal_start_alignment(
        self, main_timestamps: np.ndarray, preprocessed_data: dict
    ) -> np.ndarray:
        """修正多模态开头时间戳偏差问题 - 使用与最终验证一致的逻辑"""
        return fix_multimodal_start_alignment(
            main_timestamps=main_timestamps,
            preprocessed_data=preprocessed_data,
            find_closest_indices_fn=self.find_closest_indices_vectorized,
            fix_severely_stuck_fn=self._fix_severely_stuck_timestamps,
        )

    def _fix_severely_stuck_timestamps(
        self,
        preprocessed_data: dict,
        key: str,
        main_timestamps: np.ndarray,
        tolerance_ms: float = 20,
    ):
        """修复严重卡住的数据模态的时间戳"""
        return fix_severely_stuck_timestamps(
            preprocessed_data=preprocessed_data,
            key=key,
            main_timestamps=main_timestamps,
            tolerance_ms=tolerance_ms,
        )

    def _adjust_frame_rate_to_30fps(
        self, aligned_data: dict, main_timestamps: np.ndarray
    ):
        """
        调整帧率到30fps范围内（29.95-30.05Hz）
        通过插帧或抽帧来达到目标帧率
        """
        return adjust_frame_rate_to_30fps(
            aligned_data=aligned_data,
            main_timestamps=main_timestamps,
            insert_fn=self._insert_frames_to_increase_fps,
            remove_fn=self._remove_frames_to_decrease_fps,
        )

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
        return insert_frames_to_increase_fps(
            main_timestamps=main_timestamps,
            valid_modalities=valid_modalities,
            target_fps=target_fps,
            time_span=time_span,
        )

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
        return remove_frames_to_decrease_fps(
            main_timestamps=main_timestamps,
            valid_modalities=valid_modalities,
            target_fps=target_fps,
            time_span=time_span,
            reaverage_fn=self._reaverage_timestamps_in_window,
            execute_window_fn=self._execute_window_removal_and_reaverage,
            error_cls=TimestampStuckError,
        )

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
        return reaverage_timestamps_in_window(
            timestamps_after_removal=timestamps_after_removal,
            window_start_time=window_start_time,
            window_end_time=window_end_time,
        )

    def _execute_window_removal_and_reaverage(
        self, main_timestamps: np.ndarray, valid_modalities: dict, candidate: dict
    ) -> tuple[np.ndarray, bool]:
        """
        执行窗口删除和重新平均操作，同步更新所有模态
        修正版：只对窗口内部时间戳重新平均，两端保持不变；子时间戳使用变化量同步
        """
        return execute_window_removal_and_reaverage(
            main_timestamps=main_timestamps,
            valid_modalities=valid_modalities,
            candidate=candidate,
            max_interval_ms=40.0,
        )

    def _final_alignment_validation(
        self, aligned_data: dict, main_timestamps: np.ndarray
    ):
        """最终验证对齐后的数据质量，不满足要求则抛出异常"""
        final_alignment_validation(
            aligned_data=aligned_data,
            main_timestamps=main_timestamps,
            error_cls=TimestampStuckError,
        )

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
