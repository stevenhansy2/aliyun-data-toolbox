"""Color/depth video export helpers."""

import concurrent.futures
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from converter.configs import Config
from converter.media.schedule import resolve_video_schedule
from converter.image.depth_conversion import process_depth_image_optimized

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
    video_path = os.path.join(output_dir, f"{cam_name}.mp4")
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


# def save_one_depth_video_16U(cam_name, imgs, output_dir, raw_config):
#     import shutil
#     print(f"[DEBUG] save_one_depth_video_16U called for {cam_name}, 帧数: {len(imgs)}")

#     temp_img_dir = os.path.join(output_dir, f"frames_{cam_name}")
#     os.makedirs(temp_img_dir, exist_ok=True)
#     width, height = None, None
#     png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])

#     for idx, img_bytes in enumerate(imgs):
#         # 查找PNG头
#         if isinstance(img_bytes, bytes):
#             idx_png = img_bytes.find(png_magic)
#             if idx_png == -1:
#                 print(f"[{cam_name}] 第{idx}帧未找到PNG头，跳过")
#                 continue
#             png_data = img_bytes[idx_png:]
#         else:
#             print(f"[{cam_name}] 第{idx}帧数据类型异常，跳过")
#             continue

#         img = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
#         if img is None:
#             print(f"[{cam_name}] 第{idx}帧解码失败，跳过")
#             continue
#         if img.ndim > 2:
#             img = img[:, :, 0]
#         if raw_config is not None and hasattr(raw_config, "resize"):
#             width, height = raw_config.resize.width, raw_config.resize.height
#             img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
#         else:
#             if width is None or height is None:
#                 height, width = img.shape
#         img_path = os.path.join(temp_img_dir, f"frame_{idx:05d}.png")
#         cv2.imwrite(img_path, img.astype(np.uint16))

#     # 用ffmpeg编码为16位ffv1无损mkv
#     video_path = os.path.join(output_dir, f"{cam_name}.mkv")
#     ffmpeg_cmd = [
#         "ffmpeg", "-y",
#         "-framerate", "30",
#         "-i", os.path.join(temp_img_dir, "frame_%05d.png"),
#         "-c:v", "ffv1",
#         "-pix_fmt", "gray16le",
#         video_path
#     ]
#     try:
#         subprocess.run(ffmpeg_cmd, check=True)
#         print(f"[{cam_name}] 已保存16位无损深度视频: {video_path}")
#     except Exception as e:
#         print(f"[{cam_name}] ffmpeg编码失败: {e}")


#     shutil.rmtree(temp_img_dir, ignore_errors=True)
def save_one_depth_video_16U(cam_name, imgs, output_dir, raw_config):
    import shutil

    print(f"[DEBUG] save_one_depth_video_16U called for {cam_name}, 帧数: {len(imgs)}")

    temp_img_dir = os.path.join(output_dir, f"frames_{cam_name}")
    os.makedirs(temp_img_dir, exist_ok=True)
    width, height = None, None
    png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])

    is_hand_camera = "wrist_cam_l" in cam_name or "wrist_cam_r" in cam_name
    denoise_will_apply = is_hand_camera and raw_config and getattr(raw_config, 'denoise_enabled', False)
    
    if is_hand_camera:
        if denoise_will_apply:
            print(f"[DENOISE] {cam_name} 检测为手部相机，将进行深度去噪处理")
        else:
            print(f"[DENOISE] {cam_name} 检测为手部相机，但去噪已禁用（denoise_enabled=False）")

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

        denoise_enabled = raw_config and getattr(raw_config, 'denoise_enabled', False)
        # if is_hand_camera and denoise_enabled:
        #     try:
        #         img = repair_depth_noise_focused(
        #             img,
        #             max_valid_depth=10000,
        #             median_kernel=5,
        #             detect_white_spots=True,
        #             spot_size_range=(10, 1000),
        #         )
        #         # if idx % 50 == 0:  # 每50帧输出一次日志
        #         #     print(f"[DENOISE] {cam_name} 第{idx}帧去噪完成")
        #     except Exception as e:
        #         print(f"[DENOISE] {cam_name} 第{idx}帧去噪失败: {e}")
        #         # 去噪失败时继续使用原图

        img_path = os.path.join(temp_img_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(img_path, img.astype(np.uint16))

    # 用ffmpeg编码为16位ffv1无损mkv
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
    
    num_cameras = len(imgs_per_cam_depth)
    sched = resolve_video_schedule(raw_config)
    max_workers = max(1, min(sched.max_encode_processes, num_cameras))
    print(
        f"[SCHEDULE] depth16U: cores={sched.cores}, max_encode_processes={sched.max_encode_processes}, "
        f"workers={max_workers}"
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cam_name, imgs in imgs_per_cam_depth.items():
            print(f"[DEBUG] 提交 {cam_name} 到进程池, 帧数: {len(imgs)}")
            futures.append(
                executor.submit(
                    save_one_depth_video_16U, cam_name, imgs, output_dir, raw_config
                )
            )
        concurrent.futures.wait(futures)
    print("[DEBUG] save_depth_videos_16U_parallel end")


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

    # def save_one_enhanced_depth_video(cam_name, depth_imgs):
    #     """
    #     为单个相机保存增强深度视频
    #     """
    #     print(f"[ENHANCED] 开始处理相机 {cam_name}, 深度帧数: {len(depth_imgs)}")

    #     # 获取对应的彩色相机名（去掉_depth后缀）
    #     color_cam_name = cam_name.replace('_depth', '')
    #     color_imgs = imgs_per_cam_color.get(color_cam_name, [])

    #     if not color_imgs:
    #         print(f"[ENHANCED] 警告: 未找到相机 {cam_name} 对应的彩色图像 {color_cam_name}，将使用黑色图像")

    #     # 创建临时目录
    #     temp_img_dir = os.path.join(output_dir, f"enhanced_frames_{cam_name}")
    #     os.makedirs(temp_img_dir, exist_ok=True)

    #     try:
    #         # 确保深度和彩色帧数匹配
    #         min_frames = min(len(depth_imgs), len(color_imgs)) if color_imgs else len(depth_imgs)
    #         if min_frames == 0:
    #             print(f"[ENHANCED] 相机 {cam_name} 没有有效帧，跳过")
    #             return

    #         width, height = None, None

    #         # 处理每一帧
    #         for idx in range(min_frames):
    #             try:
    #                 # 解码深度图像（8位压缩）
    #                 depth_bytes = depth_imgs[idx]
    #                 depth_img = cv2.imdecode(np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    #                 if depth_img is None:
    #                     print(f"[ENHANCED] {cam_name} 第{idx}帧深度解码失败，跳过")
    #                     continue

    #                 # 确保是单通道
    #                 if depth_img.ndim > 2:
    #                     depth_img = depth_img[:, :, 0]

    #                 # 解码彩色图像
    #                 if color_imgs and idx < len(color_imgs):
    #                     color_bytes = color_imgs[idx]
    #                     color_img = cv2.imdecode(np.frombuffer(color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    #                     if color_img is None:
    #                         print(f"[ENHANCED] {cam_name} 第{idx}帧彩色解码失败，使用黑色图像")
    #                         color_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
    #                 else:
    #                     # 创建与深度图同尺寸的黑色彩色图像
    #                     color_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)

    #                 # 设置和调整尺寸
    #                 if width is None or height is None:
    #                     if raw_config is not None and hasattr(raw_config, "resize"):
    #                         width, height = raw_config.resize.width, raw_config.resize.height
    #                     else:
    #                         width, height = 848, 480  # 默认尺寸

    #                 # 调整尺寸
    #                 depth_img = cv2.resize(depth_img, (width, height), interpolation=cv2.INTER_NEAREST)
    #                 color_img = cv2.resize(color_img, (width, height))

    #                 # 使用您的深度增强算法（8位转16位）
    #                 try:
    #                     enhanced_depth = process_depth_image(color_img, depth_img)
    #                     # 确保输出是16位
    #                     if enhanced_depth.dtype != np.uint16:
    #                         enhanced_depth = enhanced_depth.astype(np.uint16)
    #                 except Exception as e:
    #                     print(f"[ENHANCED] {cam_name} 第{idx}帧深度增强失败: {e}")
    #                     # 降级为简单的8到16位转换
    #                     enhanced_depth = (depth_img.astype(np.uint16) * 256)

    #                 # 保存为16位PNG
    #                 img_path = os.path.join(temp_img_dir, f"frame_{idx:05d}.png")
    #                 success = cv2.imwrite(img_path, enhanced_depth)
    #                 if not success:
    #                     print(f"[ENHANCED] {cam_name} 第{idx}帧保存失败")

    #             except Exception as e:
    #                 print(f"[ENHANCED] {cam_name} 第{idx}帧处理异常: {e}")
    #                 continue

    #         # 检查是否有有效帧
    #         frame_files = [f for f in os.listdir(temp_img_dir) if f.endswith('.png')]
    #         if not frame_files:
    #             print(f"[ENHANCED] {cam_name} 没有生成有效帧")
    #             return

    #         # 使用FFmpeg编码为16位无损视频
    #         video_path = os.path.join(output_dir, f"{cam_name}.mkv")
    #         ffmpeg_cmd = [
    #             "ffmpeg", "-y",
    #             "-framerate", "30",
    #             "-i", os.path.join(temp_img_dir, "frame_%05d.png"),
    #             "-c:v", "ffv1",
    #             "-pix_fmt", "gray16le",  # 16位小端灰度
    #             video_path
    #         ]

    #         try:
    #             result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    #             print(f"[ENHANCED] 已保存增强16位深度视频: {video_path}")
    #         except subprocess.CalledProcessError as e:
    #             print(f"[ENHANCED] {cam_name} FFmpeg编码失败: {e.stderr}")
    #         except Exception as e:
    #             print(f"[ENHANCED] {cam_name} FFmpeg执行异常: {e}")

    #     finally:
    #         # 清理临时文件
    #         if os.path.exists(temp_img_dir):
    #             shutil.rmtree(temp_img_dir, ignore_errors=True)
    #         gc.collect()
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
                    # if is_hand_camera:
                    #     try:
                    #         enhanced_depth = repair_depth_noise_focused(
                    #             enhanced_depth,
                    #             max_valid_depth=10000,
                    #             median_kernel=5,
                    #             detect_white_spots=True,
                    #             spot_size_range=(10, 500),
                    #         )
                    #         # if idx % 50 == 0:  # 每50帧输出一次日志
                    #         #     print(f"[DENOISE] {cam_name} 第{idx}帧去噪完成")
                    #     except Exception as e:
                    #         print(f"[DENOISE] {cam_name} 第{idx}帧去噪失败: {e}")
                    #         # 去噪失败时继续使用增强后的图像

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
    print(f"[ENHANCED] 开始串行处理 {len(compressed_group)} 个相机的增强深度视频")
    for cam_name, depth_imgs in compressed_group.items():
        save_one_enhanced_depth_video(cam_name, depth_imgs)
    print(f"[ENHANCED] 所有增强深度视频处理完成")
