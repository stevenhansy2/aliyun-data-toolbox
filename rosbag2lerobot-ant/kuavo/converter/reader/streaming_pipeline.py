"""Streaming and parallel rosbag processing helpers for Kuavo reader."""

from __future__ import annotations

import gc
import os
import time as _time

import numpy as np
import rosbag


def compute_eef_poses_with_cached_calculator(self, pose_calculator, joint_q_list):
    """使用缓存的计算器计算末端执行器位姿"""
    positions = []
    quaternions = []

    for joint_q in joint_q_list:
        left_pose = pose_calculator.get_l_hand_camera_or_eef_pose(
            "zarm_l7_end_effector", joint_q[12:19]
        )
        left_pos = left_pose.translation()
        left_quat = left_pose.rotation().ToQuaternion()

        right_pose = pose_calculator.get_r_hand_camera_or_eef_pose(
            "zarm_r7_end_effector", joint_q[19:26]
        )
        right_pos = right_pose.translation()
        right_quat = right_pose.rotation().ToQuaternion()

        positions.append(np.concatenate([left_pos, right_pos]))
        quaternions.append(
            np.concatenate(
                [
                    [left_quat.x(), left_quat.y(), left_quat.z(), left_quat.w()],
                    [right_quat.x(), right_quat.y(), right_quat.z(), right_quat.w()],
                ]
            )
        )

    return np.array(positions), np.array(quaternions)


def process_rosbag(
    self,
    bag_file: str,
    start_time: float = 0,
    end_time: float = 1,
    action_config=None,
    chunk_size: int = 200,
    *,
    streaming_state_cls,
    attach_eef_pose_fn,
):
    """流式读取并对齐：一次只处理主时间线 chunk_size 帧。"""
    import rospy

    if not hasattr(self, "_stream_align_state"):
        self._stream_align_state = streaming_state_cls()

    if self.main_topic_map is None:
        self.main_topic_map = self._build_main_topic_map(bag_file)
        actual_hand_state_topic = None
        for t in self.HAND_STATE_TOPICS:
            if t in self.main_topic_map:
                actual_hand_state_topic = t
                break
        for topic in self.TOPICS:
            if topic in self.HAND_STATE_TOPICS:
                if actual_hand_state_topic and actual_hand_state_topic in self.main_topic_map:
                    for key, fn in self.main_topic_map[actual_hand_state_topic]:
                        self._topic_process_map[key] = {"topic": actual_hand_state_topic, "msg_process_fn": fn}
            elif topic in self.main_topic_map:
                for key, fn in self.main_topic_map[topic]:
                    self._topic_process_map[key] = {"topic": topic, "msg_process_fn": fn}

    bag = self.load_raw_rosbag(bag_file)
    bag_start = bag.get_start_time()
    bag_end = bag.get_end_time()
    bag_duration = bag_end - bag_start
    abs_start = bag_start + start_time * bag_duration
    abs_end = bag_start + end_time * bag_duration

    print(f"开始流式处理 bag: {bag_file}")
    print(f"时间窗: [{abs_start:.3f}, {abs_end:.3f}] (sec)")

    topic_to_handlers, topics_to_read, key_channel_choice = self._build_topic_handlers()

    main_key = getattr(self, "MAIN_TIMESTAMP_TOPIC", "head_cam_h")
    if main_key not in self._topic_process_map:
        cam_candidates = [c for c in self.DEFAULT_CAMERA_NAMES if c in self._topic_process_map]
        if cam_candidates:
            main_key = cam_candidates[0]
            print(f"[WARN] 主时间线 {self.MAIN_TIMESTAMP_TOPIC} 不存在，使用 {main_key}")
        else:
            if len(self._topic_process_map) == 0:
                print("[ERROR] 无任何可处理话题")
                bag.close()
                return
            main_key = max(self._topic_process_map.keys(), key=lambda k: k)

    global_main_timeline = self._prescan_main_timeline(bag_file, abs_start, abs_end)
    if len(global_main_timeline) == 0:
        print("[ERROR] 全局主时间线为空，无法继续处理")
        bag.close()
        return

    total_frames = len(global_main_timeline)
    num_batches = (total_frames + chunk_size - 1) // chunk_size
    batch_timelines = []
    for i in range(num_batches):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames)
        batch_timelines.append(global_main_timeline[start_idx:end_idx])

    print(f"[GLOBAL] 全局主时间线: {total_frames} 帧, 将分为 {num_batches} 个 batch")
    for i, bt in enumerate(batch_timelines):
        print(f"[GLOBAL] Batch {i+1}: {len(bt)} 帧, 时间范围 [{bt[0]:.3f}, {bt[-1]:.3f}]")

    buffers = {key: [] for key in self._topic_process_map.keys()}
    processed_msgs = 0
    produced_batches = 0
    current_batch_idx = 0

    def make_batch_and_yield(batch_main_timestamps: np.ndarray, batch_idx: int):
        nonlocal produced_batches
        if len(batch_main_timestamps) == 0:
            return None
        first_ts = float(batch_main_timestamps[0])
        last_ts = float(batch_main_timestamps[-1])
        print(f"[BATCH {batch_idx+1}] 使用全局时间线切片: {len(batch_main_timestamps)} 帧, 范围 [{first_ts:.3f}, {last_ts:.3f}]")

        data_window = {}
        head_margin = getattr(self, "WINDOW_HEAD_MARGIN", 0.1)
        tail_margin = 0.05
        for k, buf in buffers.items():
            if len(buf) == 0:
                data_window[k] = []
                continue
            lo = first_ts - head_margin
            hi = last_ts + tail_margin
            data_window[k] = [it for it in buf if (lo <= it["timestamp"] <= hi)]

        joint_q_items = data_window.get("observation.sensorsData.joint_q", [])
        if joint_q_items:
            attach_eef_pose_fn(data_window, self.urdf_path)
            del joint_q_items
            gc.collect()
            print("✅ 末端执行器位姿计算完成")
        else:
            data_window["end.position"] = []
            data_window["end.orientation"] = []
            del joint_q_items

        aligned_batch = self.align_frame_data_optimized(
            data_window,
            drop_head=False,
            drop_tail=False,
            action_config=None,
            streaming_state=self._stream_align_state,
            external_main_timestamps=batch_main_timestamps,
        )

        produced_batches += 1
        aligned_frame_count = len(aligned_batch.get(main_key, []))
        print(
            f"✅ 产出第 {produced_batches} 批: 主时间线 {aligned_frame_count} 帧 (预期 {len(batch_main_timestamps)} 帧), 时间窗 [{first_ts:.3f}, {last_ts:.3f}]"
        )
        if aligned_frame_count != len(batch_main_timestamps):
            print(f"[WARN] 帧数不匹配! 预期 {len(batch_main_timestamps)}, 实际 {aligned_frame_count}")

        for k in buffers.keys():
            if len(buffers[k]) == 0:
                continue
            idx = 0
            while idx < len(buffers[k]) and buffers[k][idx]["timestamp"] <= last_ts:
                idx += 1
            if idx > 0:
                buffers[k] = buffers[k][idx:]
        gc.collect()
        return aligned_batch

    _t_io_total = 0.0
    _t_process_total = 0.0
    _t_last_yield = _time.time()
    _msg_iter = bag.read_messages(
        topics=topics_to_read,
        start_time=rospy.Time.from_sec(abs_start),
        end_time=rospy.Time.from_sec(abs_end),
    )

    for topic, msg, t in _msg_iter:
        _t_io_end = _time.time()
        _t_io_total += (_t_io_end - _t_last_yield) if processed_msgs > 0 else 0
        processed_msgs += 1
        handlers = topic_to_handlers.get(topic, [])
        if not handlers:
            _t_last_yield = _time.time()
            continue

        ts = t.to_sec()
        _t_proc_start = _time.time()
        for h in handlers:
            key = h["key"]
            is_fb = h["is_fallback"]
            fn = h["fn"]
            choice = key_channel_choice.get(key)
            if choice is None:
                key_channel_choice[key] = "fallback" if is_fb else "primary"
            else:
                if (choice == "primary" and is_fb) or (choice == "fallback" and not is_fb):
                    continue
            try:
                item = fn(msg)
                item["timestamp"] = ts
                buffers[key].append(item)
            except Exception as e:
                print(f"[WARN] 处理 {topic} -> {key} 消息失败: {e}")

        _t_process_total += _time.time() - _t_proc_start

        while current_batch_idx < num_batches:
            current_batch_timeline = batch_timelines[current_batch_idx]
            batch_end_ts = float(current_batch_timeline[-1])
            if ts > batch_end_ts + 0.1:
                _t_align_start = _time.time()
                batch = make_batch_and_yield(current_batch_timeline, current_batch_idx)
                _t_align_end = _time.time()
                current_batch_idx += 1
                if batch is not None:
                    _io_pct = ((_t_io_total / (_t_io_total + _t_process_total) * 100) if (_t_io_total + _t_process_total) > 0 else 0)
                    print(
                        f"[I/O分析] Batch {current_batch_idx}: I/O={_t_io_total:.2f}s ({_io_pct:.1f}%), 消息处理={_t_process_total:.2f}s, 对齐={_t_align_end-_t_align_start:.2f}s"
                    )
                    _t_io_total = 0.0
                    _t_process_total = 0.0
                    yield batch
            else:
                break

        _t_last_yield = _time.time()
        if processed_msgs % 2000 == 0:
            gc.collect()

    while current_batch_idx < num_batches:
        current_batch_timeline = batch_timelines[current_batch_idx]
        _t_align_start = _time.time()
        batch = make_batch_and_yield(current_batch_timeline, current_batch_idx)
        _t_align_end = _time.time()
        current_batch_idx += 1
        if batch is not None:
            _io_pct = ((_t_io_total / (_t_io_total + _t_process_total) * 100) if (_t_io_total + _t_process_total) > 0 else 0)
            print(
                f"[I/O分析] Batch {current_batch_idx}: I/O={_t_io_total:.2f}s ({_io_pct:.1f}%), 消息处理={_t_process_total:.2f}s, 对齐={_t_align_end-_t_align_start:.2f}s"
            )
            yield batch

    bag.close()
    del bag
    gc.collect()
    print("✅ 流式处理完成，bag已关闭")


def prescan_and_prepare_batches(
    self,
    bag_file: str,
    start_time: float,
    end_time: float,
    chunk_size: int = 200,
):
    """预扫描并准备 batch 信息，用于并行读取。"""
    import rospy

    if self.main_topic_map is None:
        self.main_topic_map = self._build_main_topic_map(bag_file)
        actual_hand_state_topic = None
        for t in self.HAND_STATE_TOPICS:
            if t in self.main_topic_map:
                actual_hand_state_topic = t
                break
        for topic in self.TOPICS:
            if topic in self.HAND_STATE_TOPICS:
                if actual_hand_state_topic and actual_hand_state_topic in self.main_topic_map:
                    for key, fn in self.main_topic_map[actual_hand_state_topic]:
                        self._topic_process_map[key] = {"topic": actual_hand_state_topic, "msg_process_fn": fn}
            elif topic in self.main_topic_map:
                for key, fn in self.main_topic_map[topic]:
                    self._topic_process_map[key] = {"topic": topic, "msg_process_fn": fn}

    bag = rosbag.Bag(bag_file, "r")
    bag_start = bag.get_start_time()
    bag_end = bag.get_end_time()
    bag_duration = bag_end - bag_start
    abs_start = bag_start + start_time * bag_duration
    abs_end = bag_start + end_time * bag_duration

    joint_q_topic = None
    for k, info in self._topic_process_map.items():
        if k == "observation.sensorsData.joint_q":
            joint_q_topic = info["topic"]
            break

    if joint_q_topic is not None and abs_end > abs_start:
        joint_ts, joint_vals = [], []
        for _, msg, t in bag.read_messages(
            topics=[joint_q_topic],
            start_time=rospy.Time.from_sec(abs_start),
            end_time=rospy.Time.from_sec(abs_end),
        ):
            try:
                q = msg.joint_data.joint_q
                if hasattr(q, "__len__") and len(q) == 28:
                    joint_vals.append(np.array(q, dtype=np.float64))
                    joint_ts.append(t.to_sec())
            except Exception:
                item = self._msg_processer.process_joint_q_state(msg)
                if "data" in item and hasattr(item["data"], "__len__") and len(item["data"]) == 28:
                    joint_vals.append(np.array(item["data"], dtype=np.float64))
                    joint_ts.append(t.to_sec())

        if joint_ts:
            ts_arr = np.array(joint_ts, dtype=np.float64)
            vals_arr = np.vstack(joint_vals)
            head_static_end, tail_static_start = self._detect_static_by_sliding_2s(
                ts_arr,
                vals_arr,
                start=abs_start,
                end=abs_end,
                head_span_sec=10.0,
                tail_span_sec=10.0,
                win_sec=2.0,
                step_sec=1.0,
                tol_diff=0.1,
            )
            if head_static_end is not None and head_static_end > abs_start:
                abs_start = min(abs_end, head_static_end)
            if tail_static_start is not None and tail_static_start < abs_end:
                abs_end = max(abs_start, tail_static_start)
        del joint_ts, joint_vals

    bag.close()
    del bag
    gc.collect()
    if abs_end <= abs_start:
        return None

    topic_to_handlers, topics_to_read, key_channel_choice = self._build_topic_handlers()
    global_main_timeline = self._prescan_main_timeline(bag_file, abs_start, abs_end)
    if len(global_main_timeline) == 0:
        print("[ERROR] 全局主时间线为空，无法继续处理")
        return None

    total_frames = len(global_main_timeline)
    num_batches = (total_frames + chunk_size - 1) // chunk_size
    batch_timelines = []
    for i in range(num_batches):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames)
        batch_timelines.append(global_main_timeline[start_idx:end_idx])

    print(f"[PARALLEL] 预扫描完成，共 {total_frames} 帧, {num_batches} 个 batch")
    return {
        "abs_start": abs_start,
        "abs_end": abs_end,
        "global_main_timeline": global_main_timeline,
        "batch_timelines": batch_timelines,
        "topic_to_handlers": topic_to_handlers,
        "topics_to_read": topics_to_read,
        "key_channel_choice": key_channel_choice.copy(),
    }


def process_rosbag_parallel(
    self,
    bag_file: str,
    start_time: float = 0,
    end_time: float = 1,
    action_config=None,
    chunk_size: int = 200,
    num_workers: int = 2,
    *,
    parallel_worker_fn,
):
    """并行读取 ROSbag 文件，使用多进程分段读取。"""
    from multiprocessing import Process, Queue

    print(f"[PARALLEL] ========== 启动 {num_workers} 进程并行读取 ==========")
    _t_start = _time.time()

    prep = self._prescan_and_prepare_batches(bag_file, start_time, end_time, chunk_size)
    if prep is None:
        print("[PARALLEL] 预扫描失败，退出")
        return

    batch_timelines = prep["batch_timelines"]
    num_batches = len(batch_timelines)
    print(f"[PARALLEL] 预扫描耗时: {_time.time() - _t_start:.2f}s")
    print(f"[PARALLEL] 总 batch 数: {num_batches}, 将分配给 {num_workers} 个 worker")

    batches_per_worker = (num_batches + num_workers - 1) // num_workers
    worker_assignments = []
    for w in range(num_workers):
        start_batch = w * batches_per_worker
        end_batch = min((w + 1) * batches_per_worker, num_batches)
        if start_batch < end_batch:
            worker_assignments.append((start_batch, end_batch))
    print(f"[PARALLEL] Worker 分配: {worker_assignments}")

    result_queues = [Queue(maxsize=10) for _ in worker_assignments]
    workers = []
    for w_idx, (start_batch, end_batch) in enumerate(worker_assignments):
        first_timeline = batch_timelines[start_batch]
        last_timeline = batch_timelines[end_batch - 1]
        worker_time_start = float(first_timeline[0]) - 0.2
        worker_time_end = float(last_timeline[-1]) + 0.2
        config_dict = {
            "resize_width": self._msg_processer.RESIZE_W,
            "resize_height": self._msg_processer.RESIZE_H,
            "train_hz": self.TRAIN_HZ,
            "main_timeline_fps": self.MAIN_TIMELINE_FPS,
            "sample_drop": self.SAMPLE_DROP,
            "topics": self.TOPICS,
            "eef_type": self.EEF_TYPE,
            "which_arm": getattr(self, "WHICH_ARM", "both"),
            "default_camera_names": self.DEFAULT_CAMERA_NAMES,
            "default_cameras2topics": self.cam_map,
            "use_depth": self.USE_DEPTH,
            "urdf_path": self.urdf_path,
            "main_timeline_key": self.MAIN_TIMESTAMP_TOPIC,
            "camera_topic_specs": self.camera_topic_specs,
            "source_topics": self.source_topics,
            "hand_state_topics": self.HAND_STATE_TOPICS,
        }
        worker_args = {
            "bag_file": bag_file,
            "time_start": worker_time_start,
            "time_end": worker_time_end,
            "batch_timelines": batch_timelines[start_batch:end_batch],
            "batch_start_idx": start_batch,
            "topics_to_read": prep["topics_to_read"],
            "topic_process_map": self._serialize_topic_process_map(),
            "config_dict": config_dict,
        }
        p = Process(target=parallel_worker_fn, args=(worker_args, result_queues[w_idx], w_idx))
        p.start()
        workers.append(p)
        print(
            f"[PARALLEL] Worker {w_idx} 启动: batch {start_batch}-{end_batch-1}, 时间 [{worker_time_start:.2f}, {worker_time_end:.2f}]"
        )

    _t_read_start = _time.time()
    current_worker = 0
    batches_yielded = 0
    while current_worker < len(workers):
        try:
            result = result_queues[current_worker].get(timeout=300)
            if result is None:
                print(f"[PARALLEL] Worker {current_worker} 完成")
                current_worker += 1
                continue
            if "error" in result:
                print(f"[PARALLEL] Worker {current_worker} 错误: {result['error']}")
                current_worker += 1
                continue
            batch_idx = result["batch_idx"]
            batch_data = result["data"]
            batches_yielded += 1
            print(f"[PARALLEL] 收到 Batch {batch_idx + 1} (已产出 {batches_yielded}/{num_batches})")
            yield batch_data
        except Exception as e:
            print(f"[PARALLEL] 接收 Batch 出错: {e}")
            import traceback
            traceback.print_exc()
            break

    for w_idx, p in enumerate(workers):
        p.join(timeout=10)
        if p.is_alive():
            print(f"[PARALLEL] Worker {w_idx} 超时，强制终止")
            p.terminate()

    for q in result_queues:
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                pass

    _t_total = _time.time() - _t_start
    _t_read = _time.time() - _t_read_start
    print("[PARALLEL] ========== 并行读取完成 ==========")
    print(f"[PARALLEL] 总耗时: {_t_total:.2f}s, 读取阶段: {_t_read:.2f}s")
    print(f"[PARALLEL] 产出 {batches_yielded} 个 batch")
    gc.collect()

