import numpy as np
import tqdm
import json
from config_dataset_slave import Config, load_config_from_json
import argparse
from video_denoising import repair_depth_noise_focused
from kuavo_dataset_slave import KuavoRosbagReader, PostProcessorUtils
import zipfile
import datetime
import einops
from math import ceil
from copy import deepcopy
import rosbag
import cv2
import os
import concurrent.futures
import tempfile
from pathlib import Path
import subprocess
import shutil
import cv2
import numpy as np
from depthImage8to16BitConversion_optimized import process_depth_image_optimized


def get_mapped_camera_name(camera_key):
    """将相机键名映射为目标相机名称"""
    name_mapping = {
        "head_cam_h": "head",
        "wrist_cam_l": "hand_left",
        "wrist_cam_r": "hand_right",
    }
    return name_mapping.get(camera_key, camera_key)


def get_mapped_filename(camera_key, file_extension):
    """将相机键名映射为目标文件名"""
    name_mapping = {
        "head_cam_h": "head",
        "wrist_cam_l": "hand_left",
        "wrist_cam_r": "hand_right",
    }

    # 去掉可能的 _depth 后缀
    clean_key = camera_key.replace("_depth", "")
    mapped_name = name_mapping.get(clean_key, clean_key)

    if "depth" in camera_key:
        return f"{mapped_name}_depth{file_extension}"
    else:
        return f"{mapped_name}_color{file_extension}"


def save_camera_extrinsic_params(cameras, output_dir):
    """
    为每个相机生成外参文件

    Args:
        cameras: 相机名称列表，如 ['head_cam_h', 'wrist_cam_r', 'wrist_cam_l']
        output_dir: 输出目录路径
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)

    # 预定义的外参数据
    extrinsic_data = {
        "head_cam_h": {
            "rotation_matrix": [
                [0.8829475928589267, 0.0, 0.4694715627858914],
                [0.0, 1.0, 0.0],
                [-0.4694715627858914, 0.0, 0.8829475928589267],
            ],
            "translation_vector": [
                0.0967509784707853,
                0.0175003248712456,
                0.12595326511272098,
            ],
        },
        "wrist_cam_r": {
            "rotation_matrix": [
                [-0.7071096173630955, -0.7071039221275017, 0.0001798458248664092],
                [0.1830017433973951, -0.18275752957580388, 0.9659762146641413],
                [-0.6830127018922347, 0.6830839836325056, 0.25863085732104],
            ],
            "translation_vector": [
                0.115405591590931,
                0.015431235212043481,
                -0.10772412843089599,
            ],
        },
        "wrist_cam_l": {
            "rotation_matrix": [
                [-0.7076167402785503, 0.5411120873595152, 0.45439658646493686],
                [-0.17642866083771364, 0.48740501040759066, -0.855166231480516],
                [-0.6842159575109087, -0.685298522355772, -0.249428263724113],
            ],
            "translation_vector": [
                0.11540559159102,
                -0.014558611066996074,
                -0.110123491595499,
            ],
        },
    }

    for camera in cameras:
        if camera in extrinsic_data:
            extrinsic_json = {"extrinsic": extrinsic_data[camera]}

            # 使用映射后的相机名称
            mapped_camera_name = get_mapped_camera_name(camera)
            json_path = os.path.join(
                output_dir, f"{mapped_camera_name}_extrinsic_params.json"
            )
            with open(json_path, "w") as f:
                json.dump(extrinsic_json, f, indent=4)
            print(f"Saved {json_path}")
        else:
            print(f"Warning: No extrinsic data found for camera {camera}")


def load_raw_images_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict[str, list]:
    imgs_per_cam = {}
    for camera in default_camera_names:
        imgs_per_cam[camera] = [msg["data"] for msg in bag_data[camera]]
    return imgs_per_cam


def load_raw_depth_images_per_camera(bag_data: dict, default_camera_names: list[str]):
    imgs_per_cam = {}
    compressed_per_cam = {}
    for camera in default_camera_names:
        key = f"{camera}_depth"
        imgs_per_cam[key] = [msg["data"] for msg in bag_data[key]]
        # 只取第一帧的压缩状态（假设所有帧一致）
        if bag_data[key]:
            compressed_per_cam[key] = bag_data[key][0].get("compressed", None)
        else:
            compressed_per_cam[key] = None
    print("+" * 20, compressed_per_cam)
    return imgs_per_cam, compressed_per_cam


def load_camera_info_per_camera(
    bag_data: dict, default_camera_names: list[str]
) -> dict:
    info_per_cam = {}
    distortion_model = {}
    for camera in default_camera_names:
        info_per_cam[camera] = np.array(
            [msg["data"] for msg in bag_data[f"{camera}_camera_info"]], dtype=np.float32
        )
        distortion_model[camera] = [
            msg["distortion_model"] for msg in bag_data[f"{camera}_camera_info"]
        ]

    return info_per_cam, distortion_model


def save_camera_info_to_json(info_per_cam, output_dir):
    """
    将 info_per_cam 中的数据还原为每个摄像头的 json 文件，包含 D、K、R、P 四个参数
    """
    os.makedirs(output_dir, exist_ok=True)
    for camera, cam_infos in info_per_cam.items():
        # 取第0帧的参数（通常所有帧的内参都一样）
        camera_vec = cam_infos[0]
        # 假设 D5, K9, R9, P12 顺序拼接
        D = camera_vec[0:5].tolist()
        K = camera_vec[5:14].tolist()
        R = camera_vec[14:23].tolist()
        P = camera_vec[23:35].tolist()
        params = {"D": D, "K": K, "R": R, "P": P}
        # 生成文件名
        json_path = os.path.join(output_dir, f"{camera}_intrinsic_params.json")
        with open(json_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved {json_path}")


def save_camera_info_to_json_new(info_per_cam, distortion_model, output_dir):
    """
    将 info_per_cam 中的数据转换为 intrinsic 格式并保存为 json
    支持 D5/K9/R9/P12 或 D8/K9/R9/P12 等不同格式
    """
    os.makedirs(output_dir, exist_ok=True)
    for camera, cam_infos in info_per_cam.items():
        camera_vec = cam_infos[0]
        total_len = len(camera_vec)
        # 动态判断 D/K/R/P 长度
        # 常见有 D5/K9/R9/P12（总35），D8/K9/R9/P12（总38）
        if total_len == 35:
            D_len, K_len, R_len, P_len = 5, 9, 9, 12
        elif total_len == 38:
            D_len, K_len, R_len, P_len = 8, 9, 9, 12
        else:
            raise ValueError(f"未知的camera_vec长度: {total_len}，请检查相机参数格式")
        D = camera_vec[0:D_len].tolist()
        K = camera_vec[D_len : D_len + K_len].tolist()
        R = camera_vec[D_len + K_len : D_len + K_len + R_len].tolist()
        P = camera_vec[D_len + K_len + R_len : D_len + K_len + R_len + P_len].tolist()
        # distortion_model字段（优先取第一个，如有多个可自行调整）
        if camera in distortion_model and distortion_model[camera]:
            model = getattr(distortion_model[camera][0], "distortion_model", None)
            if model is None:
                model = (
                    distortion_model[camera][0]
                    if isinstance(distortion_model[camera][0], str)
                    else "unknown"
                )
        else:
            model = "unknown"
        # intrinsic格式
        intrinsic = {
            "fx": K[0],  # K[0]
            "fy": K[4],  # K[4]
            "ppx": K[2],  # K[2]
            "ppy": K[5],  # K[5]
            "distortion_model": model,
            "k1": D[0] if len(D) > 0 else None,
            "k2": D[1] if len(D) > 1 else None,
            "k3": D[4] if len(D) > 4 else None,
            "p1": D[2] if len(D) > 2 else None,
            "p2": D[3] if len(D) > 3 else None,
        }

        # 使用映射后的相机名称
        mapped_camera_name = get_mapped_camera_name(camera)
        json_path = os.path.join(
            output_dir, f"{mapped_camera_name}_intrinsic_params.json"
        )
        with open(json_path, "w") as f:
            json.dump({"intrinsic": intrinsic}, f, indent=2)
        print(f"Saved {json_path}")


def save_one_color_video_ffmpeg(cam_name, imgs, output_dir, raw_config):
    temp_img_dir = os.path.join(output_dir, f"temp_img_{cam_name}")
    os.makedirs(temp_img_dir, exist_ok=True)
    width, height = None, None

    # 保存所有帧为 PNG
    for idx, img_bytes in enumerate(imgs):
        img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_config is not None and hasattr(raw_config, "resize"):
            width, height = raw_config.resize.width, raw_config.resize.height
            img = cv2.resize(img, (width, height))
        else:
            if width is None or height is None:
                height, width = img.shape[:2]
        img_path = os.path.join(temp_img_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(img_path, img)

    # 用 ffmpeg 编码为 h264 mp4
    output_filename = get_mapped_filename(cam_name, ".mp4")
    video_path = os.path.join(output_dir, output_filename)
    # video_path = os.path.join(output_dir, f"{cam_name}.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        os.path.join(temp_img_dir, "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"已保存h264视频: {video_path}")
    except Exception as e:
        print(f"ffmpeg编码失败: {e}")

    # 删除临时图片和目录
    shutil.rmtree(temp_img_dir, ignore_errors=True)


def save_color_videos_ffmpeg_parallel(
    imgs_per_cam_color: dict, output_dir: str = "./", raw_config: Config = None
):
    """
    并行处理每个摄像头的视频生成，充分利用多核CPU
    """
    os.makedirs(output_dir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for cam_name, imgs in imgs_per_cam_color.items():
            futures.append(
                executor.submit(
                    save_one_color_video_ffmpeg, cam_name, imgs, output_dir, raw_config
                )
            )
        concurrent.futures.wait(futures)


def save_one_depth_video_ffmpeg(cam_name, imgs, output_dir, raw_config):
    temp_img_dir = os.path.join(output_dir, f"temp_img_{cam_name}")
    os.makedirs(temp_img_dir, exist_ok=True)
    width, height = None, None

    # 保存所有帧为 PNG（灰度8位）
    for idx, img_bytes in enumerate(imgs):
        img = cv2.imdecode(
            np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if img.ndim > 2:
            img = img[:, :, 0]
        if raw_config is not None and hasattr(raw_config, "resize"):
            width, height = raw_config.resize.width, raw_config.resize.height
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            if width is None or height is None:
                height, width = img.shape
        img_path = os.path.join(temp_img_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(img_path, img.astype(np.uint8))

    # 用 ffmpeg 编码为 FFV1 无损 8位灰度视频
    video_path = os.path.join(output_dir, f"{cam_name}.mkv")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        os.path.join(temp_img_dir, "frame_%05d.png"),
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gray",
        "-level",
        "3",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-slices",
        "16",
        "-an",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"已保存深度视频: {video_path}")
    except Exception as e:
        print(f"ffmpeg编码失败: {e}")

    # 删除临时图片和目录
    shutil.rmtree(temp_img_dir, ignore_errors=True)


def save_depth_videos_ffmpeg_parallel(
    imgs_per_cam_depth: dict, output_dir: str = "./", raw_config: Config = None
):
    """
    并行处理每个摄像头的深度视频生成，充分利用多核CPU，使用ffmpeg无损编码
    """
    os.makedirs(output_dir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for cam_name, imgs in imgs_per_cam_depth.items():
            futures.append(
                executor.submit(
                    save_one_depth_video_ffmpeg, cam_name, imgs, output_dir, raw_config
                )
            )
        concurrent.futures.wait(futures)


def save_color_videos_parallel(
    imgs_per_cam_color: dict, output_dir: str = "./", raw_config: Config = None
):
    import gc
    import concurrent.futures

    os.makedirs(output_dir, exist_ok=True)

    def save_one_color_video(cam_name, imgs):
        import tempfile
        import subprocess

        imgs_iter = iter(imgs)
        try:
            first_bytes = next(imgs_iter)
        except StopIteration:
            print(f"相机 {cam_name} 没有帧，跳过。")
            return
        first_img = cv2.imdecode(
            np.frombuffer(first_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        # === 新增：强制resize ===
        if raw_config is not None and hasattr(raw_config, "resize"):
            width, height = raw_config.resize.width, raw_config.resize.height
            first_img = cv2.resize(first_img, (width, height))
        else:
            height, width = first_img.shape[:2]

        # 临时保存为mp4v
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            tmp_video_path = tmpfile.name

        out = cv2.VideoWriter(
            tmp_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (width, height),
            isColor=True,
        )
        out.write(first_img)
        for img_bytes in imgs_iter:
            img = cv2.imdecode(
                np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if raw_config is not None and hasattr(raw_config, "resize"):
                img = cv2.resize(img, (width, height))
            out.write(img)
        out.release()
        del imgs_iter
        gc.collect()

        # 用ffmpeg转码为h264
        output_filename = get_mapped_filename(cam_name, ".mp4")
        video_path = os.path.join(output_dir, output_filename)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tmp_video_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            video_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"已保存h264视频: {video_path}")
        except Exception as e:
            print(f"ffmpeg转码失败: {e}")
        finally:
            os.remove(tmp_video_path)

    # 多线程并行处理每个相机
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = []
    #     for cam_name, imgs in imgs_per_cam_color.items():
    #         futures.append(executor.submit(save_one_color_video, cam_name, imgs))
    #     concurrent.futures.wait(futures)
    print("[DEBUG] save_color_videos_parallel start")
    for cam_name, imgs in imgs_per_cam_color.items():
        save_one_color_video(cam_name, imgs)
    print("[DEBUG] save_color_videos_parallel end")


def save_depth_videos_parallel(
    imgs_per_cam_depth: dict, output_dir: str = "./", raw_config: Config = None
):
    import gc
    import concurrent.futures

    os.makedirs(output_dir, exist_ok=True)

    def save_one_depth_video(cam_name, imgs):
        imgs_iter = iter(imgs)
        try:
            first_bytes = next(imgs_iter)
        except StopIteration:
            print(f"相机 {cam_name} 没有帧，跳过。")
            return
        first_img = cv2.imdecode(
            np.frombuffer(first_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if first_img.ndim > 2:
            first_img = first_img[:, :, 0]
        # === 新增：强制resize ===
        if raw_config is not None and hasattr(raw_config, "resize"):
            width, height = raw_config.resize.width, raw_config.resize.height
            first_img = cv2.resize(
                first_img, (width, height), interpolation=cv2.INTER_NEAREST
            )
        else:
            height, width = first_img.shape
        video_path = os.path.join(output_dir, f"{cam_name}.mkv")
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"FFV1"),
            30,
            (width, height),
            isColor=False,
        )
        out.write(first_img.astype(np.uint8))
        for img_bytes in imgs_iter:
            img = cv2.imdecode(
                np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            if img.ndim > 2:
                img = img[:, :, 0]
            # === 新增：强制resize ===
            if raw_config is not None and hasattr(raw_config, "resize"):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
            out.write(img.astype(np.uint8))
        out.release()
        del imgs_iter
        gc.collect()

    # 多线程并行处理每个相机
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for cam_name, imgs in imgs_per_cam_depth.items():
            futures.append(executor.submit(save_one_depth_video, cam_name, imgs))
        concurrent.futures.wait(futures)


def save_camera_info_to_json(info_per_cam, output_dir):
    """
    将 info_per_cam 中的数据还原为每个摄像头的 json 文件，包含 D、K、R、P 四个参数
    """
    os.makedirs(output_dir, exist_ok=True)
    for camera, cam_infos in info_per_cam.items():
        # 取第0帧的参数（通常所有帧的内参都一样）
        camera_vec = cam_infos[0]
        # 假设 D5, K9, R9, P12 顺序拼接
        D = camera_vec[0:5].tolist()
        K = camera_vec[5:14].tolist()
        R = camera_vec[14:23].tolist()
        P = camera_vec[23:35].tolist()
        params = {"D": D, "K": K, "R": R, "P": P}
        # 生成文件名
        json_path = os.path.join(output_dir, f"{camera}_intrinsic_params.json")
        with open(json_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved {json_path}")


def recursive_filter_and_position(data):
    """
    递归删除指定路径，并只保留有数据的 effector/position
    """

    # effector/position 只保留有数据的那一个
    def keep_only_valid_position(eff):
        pos_g = eff.get("position(gripper)")
        pos_d = eff.get("position(dexhand)")

        def is_valid(pos):
            arr = np.array(pos) if pos is not None else None
            return arr is not None and arr.size > 0 and np.any(arr != 0)

        if is_valid(pos_g) and is_valid(pos_d):
            # 保留 shape 更大者
            if np.array(pos_g).shape[1] >= np.array(pos_d).shape[1]:
                if "position(dexhand)" in eff:
                    del eff["position(dexhand)"]
            else:
                if "position(gripper)" in eff:
                    del eff["position(gripper)"]
        elif is_valid(pos_g):
            if "position(dexhand)" in eff:
                del eff["position(dexhand)"]
        elif is_valid(pos_d):
            if "position(gripper)" in eff:
                del eff["position(gripper)"]
        else:
            # 都无效，两个都留着
            pass

    for part in ["action", "state"]:
        if part in data and "effector" in data[part]:
            keep_only_valid_position(data[part]["effector"])


def save_one_depth_video_16U(cam_name, imgs, output_dir, raw_config):
    import shutil

    print(f"[DEBUG] save_one_depth_video_16U called for {cam_name}, 帧数: {len(imgs)}")

    temp_img_dir = os.path.join(output_dir, f"frames_{cam_name}")
    os.makedirs(temp_img_dir, exist_ok=True)
    width, height = None, None
    png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])

    # 检查是否为左右手相机，需要去噪
    is_hand_camera = "wrist_cam_l" in cam_name or "wrist_cam_r" in cam_name
    if is_hand_camera:
        print(f"[DENOISE] {cam_name} 检测为手部相机，将进行深度去噪处理")

    for idx, img_bytes in enumerate(imgs):
        # 查找PNG头
        if isinstance(img_bytes, bytes):
            idx_png = img_bytes.find(png_magic)
            if idx_png == -1:
                print(f"[{cam_name}] 第{idx}帧未找到PNG头，跳过")
                continue
            png_data = img_bytes[idx_png:]
        else:
            print(f"[{cam_name}] 第{idx}帧数据类型异常，跳过")
            continue

        img = cv2.imdecode(
            np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if img is None:
            print(f"[{cam_name}] 第{idx}帧解码失败，跳过")
            continue
        if img.ndim > 2:
            img = img[:, :, 0]

        # 确保是16位深度图
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)

        # 调整尺寸
        if raw_config is not None and hasattr(raw_config, "resize"):
            width, height = raw_config.resize.width, raw_config.resize.height
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            if width is None or height is None:
                height, width = img.shape

        # 如果是手部相机，进行去噪处理
        if is_hand_camera:
            try:
                img = repair_depth_noise_focused(
                    img,
                    max_valid_depth=10000,
                    median_kernel=5,
                    detect_white_spots=True,
                    spot_size_range=(10, 2000),  # 从(10, 1000)增强
                )
                # if idx % 50 == 0:  # 每50帧输出一次日志
                #     print(f"[DENOISE] {cam_name} 第{idx}帧去噪完成")
            except Exception as e:
                print(f"[DENOISE] {cam_name} 第{idx}帧去噪失败: {e}")
                # 去噪失败时继续使用原图

        img_path = os.path.join(temp_img_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(img_path, img.astype(np.uint16))

    # 用ffmpeg编码为16位ffv1无损mkv
    # video_path = os.path.join(output_dir, f"{cam_name}.mkv")
    output_filename = get_mapped_filename(cam_name, ".mkv")
    video_path = os.path.join(output_dir, output_filename)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        os.path.join(temp_img_dir, "frame_%05d.png"),
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gray16le",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"[{cam_name}] 已保存16位无损深度视频: {video_path}")
        if is_hand_camera:
            print(f"[DENOISE] {cam_name} 去噪处理完成并保存")
    except Exception as e:
        print(f"[{cam_name}] ffmpeg编码失败: {e}")

    shutil.rmtree(temp_img_dir, ignore_errors=True)


def save_depth_videos_16U_parallel(
    imgs_per_cam_depth: dict, output_dir: str = "./", raw_config: Config = None
):
    import concurrent.futures
    import os

    os.makedirs(output_dir, exist_ok=True)
    print("[DEBUG] save_depth_videos_16U_parallel start")
    print("imgs_per_cam_depth keys:", imgs_per_cam_depth.keys())
    for k, v in imgs_per_cam_depth.items():
        print(f"{k}: {len(v)} 帧")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for cam_name, imgs in imgs_per_cam_depth.items():
            print(f"[DEBUG] 提交 {cam_name} 到线程池, 帧数: {len(imgs)}")
            futures.append(
                executor.submit(
                    save_one_depth_video_16U, cam_name, imgs, output_dir, raw_config
                )
            )
        concurrent.futures.wait(futures)
    print("[DEBUG] save_depth_videos_16U_parallel end")

    # import os
    # os.makedirs(output_dir, exist_ok=True)
    # print("[DEBUG] save_depth_videos_16U_parallel start")
    # for cam_name, imgs in imgs_per_cam_depth.items():
    #     save_one_depth_video_16U(cam_name, imgs, output_dir, raw_config)
    # print("[DEBUG] save_depth_videos_16U_parallel end")


# ...existing code...


def save_depth_videos_enhanced_parallel(
    compressed_group: dict,
    imgs_per_cam_color: dict,
    output_dir: str = "./",
    raw_config: Config = None,
):
    """
    处理压缩深度数据，使用彩色图像引导进行8位到16位的深度增强，并保存为16位无损视频

    Args:
        compressed_group: 压缩深度图像数据 {cam_name: [img_bytes, ...]}
        imgs_per_cam_color: 彩色图像数据 {cam_name: [img_bytes, ...]}
        output_dir: 输出目录
        raw_config: 配置对象
    """
    import gc
    import concurrent.futures
    import shutil
    import subprocess

    os.makedirs(output_dir, exist_ok=True)

    def save_one_enhanced_depth_video(cam_name, depth_imgs):
        """
        为单个相机保存增强深度视频
        """
        print(f"[ENHANCED] 开始处理相机 {cam_name}, 深度帧数: {len(depth_imgs)}")

        # 检查是否为左右手相机，需要去噪
        is_hand_camera = "wrist_cam_l" in cam_name or "wrist_cam_r" in cam_name
        if is_hand_camera:
            print(f"[DENOISE] {cam_name} 检测为手部相机，将进行深度去噪处理")

        # 获取对应的彩色相机名（去掉_depth后缀）
        color_cam_name = cam_name.replace("_depth", "")
        color_imgs = imgs_per_cam_color.get(color_cam_name, [])

        if not color_imgs:
            print(
                f"[ENHANCED] 警告: 未找到相机 {cam_name} 对应的彩色图像 {color_cam_name}，将使用黑色图像"
            )

        # 创建临时目录
        temp_img_dir = os.path.join(output_dir, f"enhanced_frames_{cam_name}")
        os.makedirs(temp_img_dir, exist_ok=True)

        try:
            # 确保深度和彩色帧数匹配
            min_frames = (
                min(len(depth_imgs), len(color_imgs)) if color_imgs else len(depth_imgs)
            )
            if min_frames == 0:
                print(f"[ENHANCED] 相机 {cam_name} 没有有效帧，跳过")
                return

            width, height = None, None

            # 处理每一帧
            for idx in range(min_frames):
                try:
                    # 解码深度图像（8位压缩）
                    depth_bytes = depth_imgs[idx]
                    depth_img = cv2.imdecode(
                        np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                    )
                    if depth_img is None:
                        print(f"[ENHANCED] {cam_name} 第{idx}帧深度解码失败，跳过")
                        continue

                    # 确保是单通道
                    if depth_img.ndim > 2:
                        depth_img = depth_img[:, :, 0]

                    # 解码彩色图像
                    if color_imgs and idx < len(color_imgs):
                        color_bytes = color_imgs[idx]
                        color_img = cv2.imdecode(
                            np.frombuffer(color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if color_img is None:
                            print(
                                f"[ENHANCED] {cam_name} 第{idx}帧彩色解码失败，使用黑色图像"
                            )
                            color_img = np.zeros(
                                (depth_img.shape[0], depth_img.shape[1], 3),
                                dtype=np.uint8,
                            )
                    else:
                        # 创建与深度图同尺寸的黑色彩色图像
                        color_img = np.zeros(
                            (depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8
                        )

                    # 设置和调整尺寸
                    if width is None or height is None:
                        if raw_config is not None and hasattr(raw_config, "resize"):
                            width, height = (
                                raw_config.resize.width,
                                raw_config.resize.height,
                            )
                        else:
                            width, height = 848, 480  # 默认尺寸

                    # 调整尺寸
                    depth_img = cv2.resize(
                        depth_img, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                    color_img = cv2.resize(color_img, (width, height))

                    # 使用深度增强算法（8位转16位）
                    try:
                        enhanced_depth = process_depth_image_optimized(
                            color_img, depth_img
                        )
                        # 确保输出是16位
                        if enhanced_depth.dtype != np.uint16:
                            enhanced_depth = enhanced_depth.astype(np.uint16)
                    except Exception as e:
                        print(f"[ENHANCED] {cam_name} 第{idx}帧深度增强失败: {e}")
                        # 降级为简单的8到16位转换
                        enhanced_depth = depth_img.astype(np.uint16) * 256

                    # 如果是手部相机，进行去噪处理
                    if is_hand_camera:
                        try:
                            enhanced_depth = repair_depth_noise_focused(
                                enhanced_depth,
                                max_valid_depth=10000,
                                median_kernel=5,
                                detect_white_spots=True,
                                spot_size_range=(10, 500),
                            )
                            # if idx % 50 == 0:  # 每50帧输出一次日志
                            #     print(f"[DENOISE] {cam_name} 第{idx}帧去噪完成")
                        except Exception as e:
                            print(f"[DENOISE] {cam_name} 第{idx}帧去噪失败: {e}")
                            # 去噪失败时继续使用增强后的图像

                    # 保存为16位PNG
                    img_path = os.path.join(temp_img_dir, f"frame_{idx:05d}.png")
                    success = cv2.imwrite(img_path, enhanced_depth)
                    if not success:
                        print(f"[ENHANCED] {cam_name} 第{idx}帧保存失败")

                except Exception as e:
                    print(f"[ENHANCED] {cam_name} 第{idx}帧处理异常: {e}")
                    continue

            # 检查是否有有效帧
            frame_files = [f for f in os.listdir(temp_img_dir) if f.endswith(".png")]
            if not frame_files:
                print(f"[ENHANCED] {cam_name} 没有生成有效帧")
                return

            # 使用FFmpeg编码为16位无损视频
            output_filename = get_mapped_filename(cam_name, ".mkv")
            video_path = os.path.join(output_dir, output_filename)
            # video_path = os.path.join(output_dir, f"{cam_name}.mkv")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                "30",
                "-i",
                os.path.join(temp_img_dir, "frame_%05d.png"),
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "gray16le",  # 16位小端灰度
                video_path,
            ]

            try:
                result = subprocess.run(
                    ffmpeg_cmd, check=True, capture_output=True, text=True
                )
                print(f"[ENHANCED] 已保存增强16位深度视频: {video_path}")
                if is_hand_camera:
                    print(f"[DENOISE] {cam_name} 去噪处理完成并保存")
            except subprocess.CalledProcessError as e:
                print(f"[ENHANCED] {cam_name} FFmpeg编码失败: {e.stderr}")
            except Exception as e:
                print(f"[ENHANCED] {cam_name} FFmpeg执行异常: {e}")
        except Exception as e:
            print(f"[ENHANCED] {cam_name} 处理异常: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # 清理临时文件
            if os.path.exists(temp_img_dir):
                shutil.rmtree(temp_img_dir, ignore_errors=True)
            gc.collect()

    # 并行处理每个相机
    # print(f"[ENHANCED] 开始并行处理 {len(compressed_group)} 个相机的增强深度视频")
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = []
    #     for cam_name, depth_imgs in compressed_group.items():
    #         futures.append(executor.submit(save_one_enhanced_depth_video, cam_name, depth_imgs))

    #     # 等待所有任务完成
    #     concurrent.futures.wait(futures)

    # print(f"[ENHANCED] 所有增强深度视频处理完成")
    print(f"[ENHANCED] 开始串行处理 {len(compressed_group)} 个相机的增强深度视频")
    for cam_name, depth_imgs in compressed_group.items():
        save_one_enhanced_depth_video(cam_name, depth_imgs)
    print(f"[ENHANCED] 所有增强深度视频处理完成")


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
        print(f"Episode {episode_name}: 未找到HDF5文件")
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
            print(f"Episode {episode_name}: HDF5文件中未找到有效数据")
            return None

        main_length = hdf5_lengths.get("timestamps", 0)
        if main_length == 0:
            # 如果没有timestamps，使用最常见的长度
            lengths = list(hdf5_lengths.values())
            main_length = max(set(lengths), key=lengths.count) if lengths else 0

        print(f"Episode {episode_name}: HDF5主时间戳长度: {main_length}")

    except Exception as e:
        print(f"Episode {episode_name}: 读取HDF5文件失败: {e}")
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
            print(f"获取视频帧数失败 {video_path}: {e}")
            return None

    # 彩色视频
    video_color_dir = video_dir / "video"
    if video_color_dir.exists():
        for video_file in video_color_dir.glob("*.mp4"):
            cam_name = video_file.stem
            frame_count = get_video_frame_count(video_file)
            if frame_count is not None:
                color_videos[cam_name] = frame_count
                print(f"彩色视频 {cam_name}: {frame_count} 帧")

    # 深度视频
    video_depth_dir = video_dir / "depth"
    if video_depth_dir.exists():
        for video_file in video_depth_dir.glob("*.mkv"):
            cam_name = video_file.stem
            frame_count = get_video_frame_count(video_file)
            if frame_count is not None:
                depth_videos[cam_name] = frame_count
                print(f"深度视频 {cam_name}: {frame_count} 帧")

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
        print(f"✓ Episode {episode_name} 数据一致性检查通过")
    else:
        print(f"✗ Episode {episode_name} 数据不一致:")
        for inc in inconsistencies:
            print(
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
        print("  未找到有效的相机数据，跳过静止检测")
        return bag_data

    print(f"  基于 {len(camera_keys)} 个相机检测静止区域: {camera_keys}")

    # 分析每个相机的静止情况
    all_stillness_results = {}

    for camera_key in camera_keys:
        frames_data = bag_data[camera_key]
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

    print(f"  最终裁剪决定: 开头 {max_head_trim} 帧, 结尾 {max_tail_trim} 帧")

    # 应用裁剪到所有数据
    if max_head_trim > 0 or max_tail_trim > 0:
        trimmed_bag_data = trim_all_bag_data_by_frames(
            bag_data, max_head_trim, max_tail_trim
        )
        return trimmed_bag_data
    else:
        print("  无需裁剪")
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
        print(f"    {camera_key}: 帧数不足，减少检测长度至{check_frames}帧")
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
            print(f"    警告: {camera_key} 开头第{i}帧解码失败: {e}")
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
            print(f"    警告: {camera_key} 结尾第{i}帧解码失败: {e}")
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
                print(
                    f"    {key}: {original_length} -> {len(trimmed_data[key])} (-{original_length - len(trimmed_data[key])})"
                )
            else:
                trimmed_data[key] = []
                print(f"    警告: {key} 裁剪后为空")
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
        print("无需翻转相机数据")
        return imgs_per_cam, imgs_per_cam_depth

    print(f"⚡ 触发相机数据翻转: 左手={left_fix} 右手={right_fix}")

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
                        print("    警告: 深度图像未找到PNG头，跳过翻转")
                        return img_bytes
                    # 提取有效的PNG数据
                    png_data = img_bytes[idx_png:]
                else:
                    print("    警告: 深度图像数据类型异常，跳过翻转")
                    return img_bytes

                # 解码PNG数据
                nparr = np.frombuffer(png_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            else:
                # 彩色图像直接解码
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if img is None:
                print("    警告: 图像解码失败，跳过翻转")
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
                print("    警告: 图像编码失败，返回原始数据")
                return img_bytes

        except Exception as e:
            print(f"    警告: 图像翻转失败: {e}，返回原始数据")
            return img_bytes

    # 翻转彩色相机数据
    for cam_key in imgs_per_cam:
        if left_fix and ("wrist_cam_l" in cam_key or "cam_l" in cam_key):
            print(f"    翻转左手相机数据: {cam_key}")
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
                    print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

        if right_fix and ("wrist_cam_r" in cam_key or "cam_r" in cam_key):
            print(f"    翻转右手相机数据: {cam_key}")
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
                    print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

    # 翻转深度相机数据 - 使用特殊的深度图像处理
    for cam_key in imgs_per_cam_depth:
        if left_fix and ("wrist_cam_l" in cam_key or "cam_l" in cam_key):
            print(f"    翻转左手深度数据: {cam_key}")
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
                    print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

        if right_fix and ("wrist_cam_r" in cam_key or "cam_r" in cam_key):
            print(f"    翻转右手深度数据: {cam_key}")
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
                    print(f"    警告: {cam_key} 数据维度不足，跳过翻转")

    print("✅ 已完成左右手相机和深度数据的翻转")
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
        print(
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
                print(f"已调换: {left_key} <-> {right_key}")
    else:
        print(f"无需调换左右手数据: 型号={sn_code_lower} 日期={bag_date}")
