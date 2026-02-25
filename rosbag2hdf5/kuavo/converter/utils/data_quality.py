import datetime
import os
from pathlib import Path

import cv2
import numpy as np

from converter.configs.runtime_config import Config


def validate_episode_data_consistency(episode_dir):
    """
    验证单个episode的HDF5数据和视频文件的帧数一致性
    """
    import h5py
    import cv2
    import os
    import numpy as np

    episode_path = Path(episode_dir)
    episode_name = episode_path.name

    # 查找HDF5文件
    hdf5_files = list(episode_path.glob("**/proprio_stats.hdf5"))
    if not hdf5_files:
        log_print(f"Episode {episode_name}: 未找到HDF5文件")
        return None

    hdf5_path = hdf5_files[0]

    # 获取HDF5数据长度
    hdf5_lengths = {}
    try:
        with h5py.File(hdf5_path, "r") as f:

            def check_dataset_length(name, obj):
                if isinstance(obj, h5py.Dataset) and len(obj.shape) > 0:
                    hdf5_lengths[name] = obj.shape[0]

            f.visititems(check_dataset_length)

        if not hdf5_lengths:
            log_print(f"Episode {episode_name}: HDF5文件中未找到有效数据")
            return None

        main_length = hdf5_lengths.get("timestamps", 0)
        if main_length == 0:
            # 如果没有timestamps，使用最常见的长度
            lengths = list(hdf5_lengths.values())
            main_length = max(set(lengths), key=lengths.count) if lengths else 0

        log_print(f"Episode {episode_name}: HDF5主时间戳长度: {main_length}")

    except Exception as e:
        log_print(f"Episode {episode_name}: 读取HDF5文件失败: {e}")
        return None

    # 检查视频文件的帧数
    video_dir = episode_path / "camera"
    color_videos = {}
    depth_videos = {}

    def get_video_frame_count(video_path):
        """获取视频文件的帧数"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count
        except Exception as e:
            log_print(f"获取视频帧数失败 {video_path}: {e}")
            return None

    # 彩色视频
    video_color_dir = video_dir / "video"
    if video_color_dir.exists():
        for video_file in video_color_dir.glob("*.mp4"):
            cam_name = video_file.stem
            frame_count = get_video_frame_count(video_file)
            if frame_count is not None:
                color_videos[cam_name] = frame_count
                log_print(f"彩色视频 {cam_name}: {frame_count} 帧")

    # 深度视频
    video_depth_dir = video_dir / "depth"
    if video_depth_dir.exists():
        for video_file in video_depth_dir.glob("*.mkv"):
            cam_name = video_file.stem
            frame_count = get_video_frame_count(video_file)
            if frame_count is not None:
                depth_videos[cam_name] = frame_count
                log_print(f"深度视频 {cam_name}: {frame_count} 帧")

    # 检查一致性
    inconsistencies = []

    # 检查彩色视频
    for cam_name, frame_count in color_videos.items():
        if frame_count != main_length:
            inconsistencies.append(
                {
                    "type": "color_video",
                    "camera": cam_name,
                    "expected": main_length,
                    "actual": frame_count,
                    "difference": frame_count - main_length,
                }
            )

    # 检查深度视频
    for cam_name, frame_count in depth_videos.items():
        if frame_count != main_length:
            inconsistencies.append(
                {
                    "type": "depth_video",
                    "camera": cam_name,
                    "expected": main_length,
                    "actual": frame_count,
                    "difference": frame_count - main_length,
                }
            )

    # 记录结果
    is_consistent = len(inconsistencies) == 0

    if is_consistent:
        log_print(f"✓ Episode {episode_name} 数据一致性检查通过")
    else:
        log_print(f"✗ Episode {episode_name} 数据不一致:")
        for inc in inconsistencies:
            log_print(
                f"  {inc['type']} {inc['camera']}: 期望 {inc['expected']} 帧, 实际 {inc['actual']} 帧, 差异 {inc['difference']:+d} 帧"
            )

    # 返回验证结果
    result = {
        "episode": episode_name,
        "hdf5_main_length": main_length,
        "hdf5_lengths": hdf5_lengths,
        "color_videos": color_videos,
        "depth_videos": depth_videos,
        "inconsistencies": inconsistencies,
        "is_consistent": is_consistent,
    }

    return result


def detect_and_trim_bag_data(bag_data: dict, raw_config: Config):
    """
    基于detect_stillness_periods逻辑检测并裁剪bag_data中的静止帧

    Args:
        bag_data: 从rosbag读取的所有数据
        raw_config: 配置对象

    Returns:
        裁剪后的bag_data
    """
    import cv2
    import numpy as np

    # 静止检测参数（与detect_stillness_periods保持一致）
    motion_threshold = 8.0
    stillness_ratio = 0.98
    check_duration = 12.0
    fps = raw_config.train_hz or 30

    # 获取需要检测的相机列表
    camera_keys = []
    for camera in raw_config.default_camera_names:
        if camera in bag_data and len(bag_data[camera]) > 0:
            camera_keys.append(camera)

    if not camera_keys:
        log_print("  未找到有效的相机数据，跳过静止检测")
        return bag_data

    log_print(f"  基于 {len(camera_keys)} 个相机检测静止区域: {camera_keys}")

    # 分析每个相机的静止情况
    all_stillness_results = {}

    for camera_key in camera_keys:
        frames_data = bag_data[camera_key]
        log_print(f"  分析 {camera_key}: 总帧数 {len(frames_data)}")

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

        log_print(
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

    log_print(f"  最终裁剪决定: 开头 {max_head_trim} 帧, 结尾 {max_tail_trim} 帧")

    # 应用裁剪到所有数据
    if max_head_trim > 0 or max_tail_trim > 0:
        trimmed_bag_data = trim_all_bag_data_by_frames(
            bag_data, max_head_trim, max_tail_trim
        )
        return trimmed_bag_data
    else:
        log_print("  无需裁剪")
        return bag_data


def detect_stillness_from_image_data(
    frames_data, camera_key, motion_threshold, stillness_ratio, check_duration, fps
):
    """
    从图像数据检测静止帧（基于detect_stillness_periods的逻辑）
    """
    import cv2
    import numpy as np

    total_frames = len(frames_data)
    check_frames = int(check_duration * fps)

    if total_frames < check_frames * 2:
        check_frames = max(0, int(total_frames / 2 - 30))
        log_print(f"    {camera_key}: 帧数不足，减少检测长度至{check_frames}帧")
        # return 0, 0

    # 解码开头帧
    start_frames = []
    for i in range(min(check_frames, total_frames)):
        try:
            frame_bytes = bytes(frames_data[i]["data"])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 降采样提高处理速度
                if gray.shape[0] > 240 or gray.shape[1] > 320:
                    gray = cv2.resize(gray, (320, 240))
                start_frames.append(gray)
        except Exception as e:
            log_print(f"    警告: {camera_key} 开头第{i}帧解码失败: {e}")
            continue

    # 解码结尾帧
    end_frames = []
    start_pos = max(0, total_frames - check_frames)
    for i in range(start_pos, total_frames):
        try:
            frame_bytes = bytes(frames_data[i]["data"])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 降采样提高处理速度
                if gray.shape[0] > 240 or gray.shape[1] > 320:
                    gray = cv2.resize(gray, (320, 240))
                end_frames.append(gray)
        except Exception as e:
            log_print(f"    警告: {camera_key} 结尾第{i}帧解码失败: {e}")
            continue

    # 分析开头静止情况
    head_stillness_frames = analyze_stillness_frames(
        start_frames, motion_threshold, stillness_ratio, fps, check_duration
    )

    # 分析结尾静止情况（需要反向分析）
    end_frames_reversed = list(reversed(end_frames))
    tail_stillness_frames = analyze_stillness_frames(
        end_frames_reversed, motion_threshold, stillness_ratio, fps, check_duration
    )

    return head_stillness_frames, tail_stillness_frames


def analyze_stillness_frames(
    frames, motion_threshold, stillness_ratio, fps, check_duration
):
    """
    分析帧序列的静止情况（与detect_stillness_periods中的analyze_stillness逻辑一致）
    """
    import cv2
    import numpy as np

    if len(frames) < 2:
        return 0

    def calculate_frame_difference(frame1, frame2):
        """计算两帧之间的平均像素差值"""
        diff = cv2.absdiff(frame1, frame2)
        return np.mean(diff)

    # 计算每帧与前一帧的差值
    motion_flags = []
    for i in range(1, len(frames)):
        diff = calculate_frame_difference(frames[i - 1], frames[i])
        is_motion = diff > motion_threshold
        motion_flags.append(is_motion)

    # 使用递增窗口检查：2s, 2.5s, 3s, 3.5s, 4s...
    max_stillness_duration = 0
    still_frame_count = 0

    # 从2秒开始，每次增加0.5秒
    current_duration = 2.0
    while current_duration <= check_duration:
        window_frames_count = int(current_duration * fps)

        # 确保不超过可用帧数
        if window_frames_count > len(motion_flags):
            break

        # 检查从开头开始的这个窗口
        window_motion_flags = motion_flags[:window_frames_count]

        # 计算该窗口内的静止帧比例
        still_count = sum(1 for flag in window_motion_flags if not flag)
        still_ratio = (
            still_count / len(window_motion_flags) if window_motion_flags else 0
        )

        if still_ratio >= stillness_ratio:
            # 该窗口被认为是静止的，更新最大静止时长
            max_stillness_duration = current_duration
            still_frame_count = window_frames_count
        else:
            # 该窗口不符合静止条件，停止检查
            break

        current_duration += 0.5

    return still_frame_count


def trim_all_bag_data_by_frames(
    bag_data: dict, head_trim_frames: int, tail_trim_frames: int
):
    """按帧数裁剪bag_data中的所有数据"""
    trimmed_data = {}

    for key, data_list in bag_data.items():
        if isinstance(data_list, list) and len(data_list) > 0:
            original_length = len(data_list)

            # 计算裁剪范围
            start_idx = head_trim_frames
            if tail_trim_frames > 0:
                end_idx = original_length - tail_trim_frames
            else:
                end_idx = original_length

            # 确保索引有效
            start_idx = max(0, start_idx)
            end_idx = min(original_length, end_idx)

            if start_idx < end_idx:
                trimmed_data[key] = data_list[start_idx:end_idx]
                log_print(
                    f"    {key}: {original_length} -> {len(trimmed_data[key])} (-{original_length - len(trimmed_data[key])})"
                )
            else:
                trimmed_data[key] = []
                log_print(f"    警告: {key} 裁剪后为空")
        else:
            # 非列表数据或空数据保持不变
            trimmed_data[key] = data_list

    return trimmed_data


def should_flip_camera(device_sn: str, start_timestamp: float) -> dict:
    """
    判断是否需要对左右手相机和深度数据进行翻转，返回 {'left': bool, 'right': bool}
    规则参考 fix_camera.py
    """
    from datetime import datetime

    # fix_camera.py 的设备翻转规则
    device_fix_map = {
        "P4-199": {"left": "正", "right": "反"},
        "P4-206": {"left": "正", "right": "反"},
        "P4-210": {"left": "正", "right": "反"},
        "P4-217": {"left": "正", "right": "反"},
        "P4-195": {"left": "正", "right": "反"},
        "P4-277": {"left": "反", "right": "正"},
        "P4-279": {"left": "反", "right": "正"},
        "P4-281": {"left": "反", "right": "正"},
        "P4-282": {"left": "正", "right": "正"},
        "P4-283": {"left": "反", "right": "正"},
        "P4-284": {"left": "反", "right": "正"},
        "P4-286": {"left": "反", "right": "正"},
        "P4-287": {"left": "反", "right": "正"},
        "P4-288": {"left": "反", "right": "正"},
        "P4-289": {"left": "反", "right": "正"},
        "P4-291": {"left": "反", "right": "正"},
        "P4-292": {"left": "反", "right": "正"},
        "P4-293": {"left": "反", "right": "正"},
        "P4-295": {"left": "正", "right": "反"},
        "P4-325": {"left": "正", "right": "反"},
        "P4-327": {"left": "反", "right": "反"},
        "P4-329": {"left": "反", "right": "正"},
        "P4-332": {"left": "反", "right": "正"},
        "P4-336": {"left": "反", "right": "正"},
        "P4-337": {"left": "反", "right": "正"},
        "P4-340": {"left": "正", "right": "正"},
        "P4-341": {"left": "反", "right": "正"},
        "P4-342": {"left": "反", "right": "正"},
        "P4-345": {"left": "反", "right": "正"},
        "P4-346": {"left": "反", "right": "正"},
        "P4-347": {"left": "反", "right": "正"},
        "P4-349": {"left": "反", "right": "正"},
        "P4-351": {"left": "反", "right": "正"},
        "P4-352": {"left": "反", "right": "正"},
        "P4-358": {"left": "反", "right": "正"},
        "P4-360": {"left": "反", "right": "正"},
        "P4-361": {"left": "反", "right": "正"},
        "P4-362": {"left": "反", "right": "正"},
        "P4-363": {"left": "正", "right": "正"},
        "P4-367": {"left": "反", "right": "正"},
        "P4-369": {"left": "反", "right": "正"},
        "P4-394": {"left": "正", "right": "反"},
        "P4-400": {"left": "反", "right": "正"},
        "P4-401": {"left": "反", "right": "正"},
        "P4-403": {"left": "正", "right": "正"},
        "P4-405": {"left": "反", "right": "正"},
        "P4-406": {"left": "反", "right": "正"},
        "P4-407": {"left": "反", "right": "正"},
        "P4-408": {"left": "反", "right": "正"},
        "P4-409": {"left": "反", "right": "反"},
    }
    # 截止日期
    cutoff_date = datetime(2025, 9, 11).date()
    # 特殊规则
    p4_217_cutoff = datetime(2024, 8, 28).date()

    # 设备号格式兼容
    device = device_sn
    if device not in device_fix_map:
        return {"left": False, "right": False}

    # 时间戳转日期
    dt = datetime.fromtimestamp(start_timestamp)
    bag_date = dt.date()

    # 日期规则
    if dt.year < 2024 or bag_date > cutoff_date:
        return {"left": False, "right": False}
    if device == "P4-217" and bag_date < p4_217_cutoff:
        return {"left": False, "right": False}

    cfg = device_fix_map[device]
    left_fix = cfg.get("left") == "反"
    right_fix = cfg.get("right") == "反"
    return {"left": left_fix, "right": right_fix}


def flip_camera_arrays_if_needed(
    imgs_per_cam, imgs_per_cam_depth, device_sn, start_timestamp
):
    """
    检查是否需要翻转，并对左右手相机和深度数据进行上下翻转（180度）。
    imgs_per_cam, imgs_per_cam_depth: dict[str, list[bytes]]
    device_sn: 设备序列号
    start_timestamp: 主时间戳开始时间（float, 秒）
    """
    flip_flags = should_flip_camera(device_sn, start_timestamp)
    left_fix = flip_flags["left"]
    right_fix = flip_flags["right"]

    if not (left_fix or right_fix):
        log_print("无需翻转相机数据")
        return imgs_per_cam, imgs_per_cam_depth

    log_print(f"⚡ 触发相机数据翻转: 左手={left_fix} 右手={right_fix}")

    def flip_image_bytes(img_bytes, is_depth=False):
        """对图像字节数据进行翻转"""
        try:
            # 对深度图像进行特殊处理
            if is_depth:
                # 查找PNG头（深度图像可能有偏移）
                png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
                if isinstance(img_bytes, bytes):
                    idx_png = img_bytes.find(png_magic)
                    if idx_png == -1:
                        log_print("    警告: 深度图像未找到PNG头，跳过翻转")
                        return img_bytes
                    # 提取有效的PNG数据
                    png_data = img_bytes[idx_png:]
                else:
                    log_print("    警告: 深度图像数据类型异常，跳过翻转")
                    return img_bytes

                # 解码PNG数据
                nparr = np.frombuffer(png_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            else:
                # 彩色图像直接解码
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if img is None:
                log_print("    警告: 图像解码失败，跳过翻转")
                return img_bytes

            # 确保深度图像是单通道
            if is_depth and img.ndim > 2:
                img = img[:, :, 0]

            # 进行翻转（180度旋转）
            flipped_img = np.flipud(np.fliplr(img))

            # 重新编码为字节
            if is_depth:
                # 深度图像使用PNG格式
                success, encoded_img = cv2.imencode(".png", flipped_img)
            elif img.ndim == 3:  # 彩色图像
                success, encoded_img = cv2.imencode(".jpg", flipped_img)
            else:  # 灰度图像
                success, encoded_img = cv2.imencode(".png", flipped_img)

            if success:
                if is_depth and isinstance(img_bytes, bytes) and idx_png > 0:
                    # 对于深度图像，保持原始的头部数据结构
                    original_header = img_bytes[:idx_png]
                    return original_header + encoded_img.tobytes()
                else:
                    return encoded_img.tobytes()
            else:
                log_print("    警告: 图像编码失败，返回原始数据")
                return img_bytes

        except Exception as e:
            log_print(f"    警告: 图像翻转失败: {e}，返回原始数据")
            return img_bytes

    # 翻转彩色相机数据
    for cam_key in imgs_per_cam:
        if left_fix and ("wrist_cam_l" in cam_key or "cam_l" in cam_key):
            log_print(f"    翻转左手相机数据: {cam_key}")
            imgs = imgs_per_cam[cam_key]
            if isinstance(imgs, list):
                # 处理字节数据列表
                imgs_per_cam[cam_key] = [
                    flip_image_bytes(img_bytes, is_depth=False) for img_bytes in imgs
                ]
            elif isinstance(imgs, np.ndarray):
                # 处理已解码的图像数组
                if imgs.ndim >= 2:
                    imgs_per_cam[cam_key] = (
                        np.flip(imgs, axis=(1, 2))
                        if imgs.ndim == 3
                        else np.flipud(np.fliplr(imgs))
                    )
                else:
                    log_print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

        if right_fix and ("wrist_cam_r" in cam_key or "cam_r" in cam_key):
            log_print(f"    翻转右手相机数据: {cam_key}")
            imgs = imgs_per_cam[cam_key]
            if isinstance(imgs, list):
                # 处理字节数据列表
                imgs_per_cam[cam_key] = [
                    flip_image_bytes(img_bytes, is_depth=False) for img_bytes in imgs
                ]
            elif isinstance(imgs, np.ndarray):
                # 处理已解码的图像数组
                if imgs.ndim >= 2:
                    imgs_per_cam[cam_key] = (
                        np.flip(imgs, axis=(1, 2))
                        if imgs.ndim == 3
                        else np.flipud(np.fliplr(imgs))
                    )
                else:
                    log_print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

    # 翻转深度相机数据 - 使用特殊的深度图像处理
    for cam_key in imgs_per_cam_depth:
        if left_fix and ("wrist_cam_l" in cam_key or "cam_l" in cam_key):
            log_print(f"    翻转左手深度数据: {cam_key}")
            imgs = imgs_per_cam_depth[cam_key]
            if isinstance(imgs, list):
                # 处理深度图像字节数据列表，标记为深度图像
                imgs_per_cam_depth[cam_key] = [
                    flip_image_bytes(img_bytes, is_depth=True) for img_bytes in imgs
                ]
            elif isinstance(imgs, np.ndarray):
                # 处理已解码的图像数组
                if imgs.ndim >= 2:
                    imgs_per_cam_depth[cam_key] = (
                        np.flip(imgs, axis=(1, 2))
                        if imgs.ndim == 3
                        else np.flipud(np.fliplr(imgs))
                    )
                else:
                    log_print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

        if right_fix and ("wrist_cam_r" in cam_key or "cam_r" in cam_key):
            log_print(f"    翻转右手深度数据: {cam_key}")
            imgs = imgs_per_cam_depth[cam_key]
            if isinstance(imgs, list):
                # 处理深度图像字节数据列表，标记为深度图像
                imgs_per_cam_depth[cam_key] = [
                    flip_image_bytes(img_bytes, is_depth=True) for img_bytes in imgs
                ]
            elif isinstance(imgs, np.ndarray):
                # 处理已解码的图像数组
                if imgs.ndim >= 2:
                    imgs_per_cam_depth[cam_key] = (
                        np.flip(imgs, axis=(1, 2))
                        if imgs.ndim == 3
                        else np.flipud(np.fliplr(imgs))
                    )
                else:
                    log_print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

    log_print("✅ 已完成左右手相机和深度数据的翻转")
    return imgs_per_cam, imgs_per_cam_depth


def swap_left_right_data_if_needed(bag_data, sn_code, main_time_line_timestamps):
    """
    对于指定型号和截止日期之前（含当天）的数据，将所有左右手相关数据调换
    - sn_code: 设备序列号（字符串，自动小写）
    - main_time_line_timestamps: 主时间戳数组（秒），用于判断日期
    """
    # 指定设备及各自截止日期
    swap_devices_cutoff = {
        "p4-327": datetime.date(2025, 9, 16),
        "lb-21": datetime.date(2025, 9, 17),
        "p4-195": datetime.date(2025, 9, 17),
    }

    sn_code_lower = str(sn_code).lower() if sn_code else ""
    cutoff_date = swap_devices_cutoff.get(sn_code_lower, None)

    # 用主时间戳第一个时刻作为日期判断
    if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
        first_ts = main_time_line_timestamps[0]
        bag_date = datetime.datetime.fromtimestamp(first_ts).date()
    else:
        bag_date = None

    if cutoff_date is not None and bag_date is not None and bag_date <= cutoff_date:
        log_print(
            f"⚡ 触发左右手数据调换: 型号={sn_code_lower} 日期={bag_date} <= {cutoff_date}"
        )
        swap_pairs = [
            ("wrist_cam_l", "wrist_cam_r"),
            ("wrist_cam_l_depth", "wrist_cam_r_depth"),
            ("left_hand_camera_extrinsics", "right_hand_camera_extrinsics"),
            ("hand_left_color_mp4_timestamps", "hand_right_color_mp4_timestamps"),
            ("hand_left_depth_mkv_timestamps", "hand_right_depth_mkv_timestamps"),
        ]
        for left_key, right_key in swap_pairs:
            if left_key in bag_data and right_key in bag_data:
                bag_data[left_key], bag_data[right_key] = (
                    bag_data[right_key],
                    bag_data[left_key],
                )
                log_print(f"已调换: {left_key} <-> {right_key}")
    else:
        log_print(f"无需调换左右手数据: 型号={sn_code_lower} 日期={bag_date}")
