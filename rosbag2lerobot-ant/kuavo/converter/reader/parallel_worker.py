"""Parallel ROSbag worker process logic."""

from converter.reader.eef_builder import attach_eef_pose_from_joint_q


def parallel_rosbag_worker(args: dict, result_queue, worker_id: int):
    """
    并行 ROSbag 读取 worker 函数。
    在独立进程中运行，读取指定时间范围的数据并产出 batch。

    使用与主代码完全相同的对齐逻辑，确保数据一致性。

    Args:
        args: 包含以下字段的字典:
            - bag_file: ROSbag 文件路径
            - time_start: 读取开始时间
            - time_end: 读取结束时间
            - batch_timelines: 该 worker 负责的 batch 时间线列表
            - batch_start_idx: 该 worker 第一个 batch 的全局索引
            - topics_to_read: 需要读取的 topic 列表
            - topic_process_map: 序列化的 topic 处理映射
            - config_dict: 用于创建 KuavoRosbagReader 的配置字典
        result_queue: 用于发送结果的队列
        worker_id: worker 编号
    """
    import gc
    import time as _time
    import traceback

    import rosbag
    import rospy
    from converter.config import ResizeConfig
    from converter.reader.kuavo_dataset_slave_s import (
        KuavoRosbagReader,
        StreamingAlignmentState,
    )

    try:
        _t_start = _time.time()
        bag_file = args["bag_file"]
        time_start = args["time_start"]
        time_end = args["time_end"]
        batch_timelines = args["batch_timelines"]
        batch_start_idx = args["batch_start_idx"]
        topics_to_read = args["topics_to_read"]
        topic_process_map_ser = args["topic_process_map"]
        config_dict = args["config_dict"]

        print(f"[Worker {worker_id}] 启动，负责 {len(batch_timelines)} 个 batch")
        print(f"[Worker {worker_id}] 时间范围: [{time_start:.2f}, {time_end:.2f}]")

        # 重建配置对象（用于创建 KuavoRosbagReader）
        resize_config = ResizeConfig(
            width=config_dict["resize_width"],
            height=config_dict["resize_height"],
        )

        # 创建最小化的 Config 对象
        class MinimalConfig:
            def __init__(self, d):
                self.resize = resize_config
                self.train_hz = d["train_hz"]
                self.main_timeline_fps = d["main_timeline_fps"]
                self.sample_drop = d["sample_drop"]
                self.topics = d["topics"]
                self.eef_type = d["eef_type"]
                self.which_arm = d["which_arm"]
                self._default_camera_names = d["default_camera_names"]
                self._default_cameras2topics = d["default_cameras2topics"]
                self.urdf_path = d.get("urdf_path", "./kuavo/assets/urdf/biped_s45.urdf")
                self.main_timeline_key = d.get("main_timeline_key", "camera_top")
                self.camera_topic_specs = d.get("camera_topic_specs", {})
                self.source_topics = d.get("source_topics", {})
                self._hand_state_topics = d.get(
                    "hand_state_topics",
                    ["/control_robot_hand_position_state", "/dexhand/state"],
                )

            @property
            def default_camera_names(self):
                return self._default_camera_names

            @property
            def default_cameras2topics(self):
                return self._default_cameras2topics

            @property
            def hand_state_topics(self):
                return self._hand_state_topics

        minimal_config = MinimalConfig(config_dict)

        # 创建 KuavoRosbagReader 实例（用于调用原始对齐方法）
        reader = KuavoRosbagReader(
            minimal_config, use_depth=config_dict.get("use_depth", False)
        )

        # 重建消息处理器引用
        msg_processer = reader._msg_processer

        # 动态定义的处理函数（与 _build_main_topic_map 中一致）
        def process_dexhand_state(msg):
            """处理 /dexhand/state 话题"""
            return {
                "data": list(msg.position),
                "timestamp": msg.header.stamp.to_sec(),
            }

        # 重建 topic_process_map (从序列化格式恢复)
        fn_map = {
            # 图像处理
            "process_color_image": msg_processer.process_color_image,
            "process_depth_image": msg_processer.process_depth_image,
            "process_depth_image_16U": msg_processer.process_depth_image_16U,
            "process_camera_info": msg_processer.process_camera_info,
            "process_camera_metadata": msg_processer.process_camera_metadata,
            "process_depth": msg_processer.process_depth,
            # 关节状态
            "process_joint_q_state": msg_processer.process_joint_q_state,
            "process_joint_v_state": msg_processer.process_joint_v_state,
            "process_joint_vd_state": msg_processer.process_joint_vd_state,
            "process_joint_current_state": msg_processer.process_joint_current_state,
            "process_joint_torque_state": msg_processer.process_joint_torque_state,
            # IMU
            "process_sensors_data_raw_extract_imu": msg_processer.process_sensors_data_raw_extract_imu,
            # 关节指令
            "process_kuavo_arm_traj": msg_processer.process_kuavo_arm_traj,
            "process_joint_cmd_joint_q": msg_processer.process_joint_cmd_joint_q,
            "process_joint_cmd_joint_v": msg_processer.process_joint_cmd_joint_v,
            "process_joint_cmd_tau": msg_processer.process_joint_cmd_tau,
            "process_joint_cmd_tau_max": msg_processer.process_joint_cmd_tau_max,
            "process_joint_cmd_tau_ratio": msg_processer.process_joint_cmd_tau_ratio,
            "process_joint_cmd_joint_kp": msg_processer.process_joint_cmd_joint_kp,
            "process_joint_cmd_joint_kd": msg_processer.process_joint_cmd_joint_kd,
            "process_joint_cmd_control_modes": msg_processer.process_joint_cmd_control_modes,
            # 手部状态和指令
            "process_qiangnao_state": msg_processer.process_qiangnao_state,
            "process_qiangnao_cmd": msg_processer.process_qiangnao_cmd,
            "process_claw_state": msg_processer.process_claw_state,
            "process_claw_cmd": msg_processer.process_claw_cmd,
            # 动态定义的函数
            "process_dexhand_state": process_dexhand_state,
        }

        topic_to_handlers = {}
        key_channel_choice = {}

        for key, info in topic_process_map_ser.items():
            topic = info["topic"]
            fn_name = info["fn_name"]
            if fn_name not in fn_map:
                print(f"[Worker {worker_id}] 警告: 未知函数 {fn_name}")
                continue

            topic_to_handlers.setdefault(topic, []).append(
                {
                    "key": key,
                    "fn": fn_map[fn_name],
                    "is_fallback": False,
                }
            )
            key_channel_choice[key] = None

            if "fallback_topic" in info:
                fb_topic = info["fallback_topic"]
                fb_fn_name = info["fallback_fn_name"]
                if fb_fn_name in fn_map:
                    topic_to_handlers.setdefault(fb_topic, []).append(
                        {
                            "key": key,
                            "fn": fn_map[fb_fn_name],
                            "is_fallback": True,
                        }
                    )

        # 打开 bag 文件
        bag = rosbag.Bag(bag_file, "r")

        # 为每个 key 准备 buffer
        buffers = {key: [] for key in topic_process_map_ser.keys()}

        # 流对齐状态（使用原始类）
        stream_align_state = StreamingAlignmentState()

        # 读取消息
        processed_msgs = 0
        current_batch_local_idx = 0
        num_local_batches = len(batch_timelines)

        def make_batch(batch_timeline, batch_idx):
            """构建并返回一个 batch，使用与主代码相同的对齐逻辑"""
            nonlocal buffers

            if len(batch_timeline) == 0:
                return None

            first_ts = float(batch_timeline[0])
            last_ts = float(batch_timeline[-1])
            head_margin = 0.1
            tail_margin = 0.05

            # 截取时间窗数据
            data_window = {}
            for k, buf in buffers.items():
                if len(buf) == 0:
                    data_window[k] = []
                    continue
                lo = first_ts - head_margin
                hi = last_ts + tail_margin
                slice_items = [it for it in buf if (lo <= it["timestamp"] <= hi)]
                data_window[k] = slice_items

            # 末端执行器位姿计算
            try:
                attach_eef_pose_from_joint_q(data_window, reader.urdf_path)
            except Exception as e:
                print(f"[Worker {worker_id}] 末端位姿计算失败: {e}")
                data_window["end.position"] = []
                data_window["end.orientation"] = []

            # 使用原始的 align_frame_data_optimized 方法
            # 传入 external_main_timestamps 以使用全局预计算的时间线
            aligned_batch = reader.align_frame_data_optimized(
                data_window,
                drop_head=False,
                drop_tail=False,
                action_config=None,
                streaming_state=stream_align_state,
                external_main_timestamps=batch_timeline,
            )

            # 裁剪缓冲区
            for k in buffers.keys():
                if len(buffers[k]) == 0:
                    continue
                idx = 0
                while idx < len(buffers[k]) and buffers[k][idx]["timestamp"] <= last_ts:
                    idx += 1
                if idx > 0:
                    buffers[k] = buffers[k][idx:]

            return aligned_batch

        # 开始读取
        _t_read_start = _time.time()

        for topic, msg, t in bag.read_messages(
            topics=topics_to_read,
            start_time=rospy.Time.from_sec(time_start),
            end_time=rospy.Time.from_sec(time_end),
        ):
            processed_msgs += 1
            handlers = topic_to_handlers.get(topic, [])
            if not handlers:
                continue

            ts = t.to_sec()

            for h in handlers:
                key = h["key"]
                is_fb = h["is_fallback"]
                fn = h["fn"]

                # 通道选择
                choice = key_channel_choice.get(key)
                if choice is None:
                    key_channel_choice[key] = "fallback" if is_fb else "primary"
                else:
                    if (choice == "primary" and is_fb) or (
                        choice == "fallback" and not is_fb
                    ):
                        continue

                # 处理消息
                try:
                    item = fn(msg)
                    item["timestamp"] = ts
                    buffers[key].append(item)
                except Exception as e:
                    pass  # 静默处理错误

            # 检查是否可以产出 batch
            while current_batch_local_idx < num_local_batches:
                current_batch_timeline = batch_timelines[current_batch_local_idx]
                batch_end_ts = float(current_batch_timeline[-1])

                if ts > batch_end_ts + 0.1:
                    batch = make_batch(current_batch_timeline, current_batch_local_idx)
                    global_batch_idx = batch_start_idx + current_batch_local_idx
                    current_batch_local_idx += 1

                    if batch is not None:
                        result_queue.put(
                            {
                                "batch_idx": global_batch_idx,
                                "data": batch,
                            }
                        )
                        print(f"[Worker {worker_id}] 产出 Batch {global_batch_idx + 1}")
                else:
                    break

            # 定期回收
            if processed_msgs % 5000 == 0:
                gc.collect()

        # 处理剩余 batch
        while current_batch_local_idx < num_local_batches:
            current_batch_timeline = batch_timelines[current_batch_local_idx]
            batch = make_batch(current_batch_timeline, current_batch_local_idx)
            global_batch_idx = batch_start_idx + current_batch_local_idx
            current_batch_local_idx += 1

            if batch is not None:
                result_queue.put(
                    {
                        "batch_idx": global_batch_idx,
                        "data": batch,
                    }
                )
                print(f"[Worker {worker_id}] 产出 Batch {global_batch_idx + 1}")

        bag.close()
        del bag
        gc.collect()

        _t_total = _time.time() - _t_start
        _t_read = _time.time() - _t_read_start
        print(
            f"[Worker {worker_id}] 完成: 处理 {processed_msgs} 消息, 产出 {current_batch_local_idx} batch, 耗时 {_t_total:.2f}s (读取 {_t_read:.2f}s)"
        )

        # 发送完成信号
        result_queue.put(None)

    except Exception as e:
        print(f"[Worker {worker_id}] 错误: {e}")
        traceback.print_exc()
        result_queue.put({"error": str(e), "traceback": traceback.format_exc()})
