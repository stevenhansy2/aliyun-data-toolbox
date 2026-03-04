"""Low-level video worker functions."""

import gc
import glob
import os
import shutil
import subprocess
import tempfile

import cv2
import einops
import numpy as np
from lerobot.datasets.compute_stats import get_feature_stats

def save_image_bytes_to_temp(
    imgs_per_cam: dict, imgs_per_cam_depth: dict, temp_base_dir: str, batch_id: int
):
    """
    直接保存图像字节流到临时目录（不解码、不缩放，传入什么尺寸就保存什么尺寸）

    Args:
        imgs_per_cam: 彩色图像字节流 {camera: [bytes, ...]}
        imgs_per_cam_depth: 深度图像字节流 {camera: [bytes, ...]}
        temp_base_dir: 临时目录基路径
        batch_id: 批次ID
    """
    import os

    cam_stats = {}
    # 保存彩色图像
    for camera, jpeg_list in imgs_per_cam.items():
        camera_dir = os.path.join(temp_base_dir, "color", camera)
        os.makedirs(camera_dir, exist_ok=True)

        for i, jpeg_bytes in enumerate(jpeg_list):
            frame_path = os.path.join(
                camera_dir, f"batch_{batch_id:04d}_frame_{i:06d}.jpg"
            )
            with open(frame_path, "wb") as f:
                f.write(jpeg_bytes)
            if i == 0:
                img_np_bgr = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if img_np_bgr is None:
                    raise ValueError(
                        f"Failed to decode color image for camera {camera} at frame {i}"
                    )
                h0, w0 = img_np_bgr.shape[:2]
                img_np = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                img_np = einops.rearrange(img_np, "h w c -> c h w")
                key = f"observation.images.{camera}"
                cam_stats[key] = get_feature_stats(
                    [img_np], axis=(0, 2, 3), keepdims=True
                )
                cam_stats[key] = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                    for k, v in cam_stats[key].items()
                }
                # 额外记录实际图像高宽，供 info.json 使用
                cam_stats[key]["height"] = int(h0)
                cam_stats[key]["width"] = int(w0)

    # 保存深度图像（PNG字节流）
    if imgs_per_cam_depth is not None:
        for camera, png_list in imgs_per_cam_depth.items():
            camera_dir = os.path.join(temp_base_dir, "depth", camera)
            os.makedirs(camera_dir, exist_ok=True)

            for i, png_bytes in enumerate(png_list):
                frame_path = os.path.join(
                    camera_dir, f"batch_{batch_id:04d}_frame_{i:06d}.png"
                )
                png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
                if isinstance(png_bytes, bytes):
                    idx_png = png_bytes.find(png_magic)
                    if idx_png != -1:
                        png_data = png_bytes[idx_png:]
                        with open(frame_path, "wb") as f:
                            f.write(png_data)

    print(f"[TEMP] 批次{batch_id} 图像字节流已保存到临时目录")
    return cam_stats


def _encode_color_camera_worker(
    camera_dir: str,
    camera: str,
    out_path: str,
    train_hz: int,
    stats_output_dir: str,
    codec_threads: int = 1,
):
    """
    子进程：编码彩色视频（PyAV）
    """
    import av
    from PIL import Image
    import glob
    import shutil
    import gc

    try:
        frame_files = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
        print(f"[VIDEO][COLOR] {camera}: 帧数 {len(frame_files)}")
        if len(frame_files) == 0:
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            print(f"[VIDEO][COLOR] {camera}: 无帧，已清理")
            return

        video_options = {
            "g": "2",
            "crf": "30",
            "threads": str(max(1, codec_threads)),
        }
        first_img = Image.open(frame_files[0])
        width, height = first_img.size

        with av.open(str(out_path), "w") as output:
            stream = output.add_stream("libx264", train_hz, options=video_options)
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height

            for frame_file in frame_files:
                img = Image.open(frame_file).convert("RGB")
                frame = av.VideoFrame.from_image(img)
                packet = stream.encode(frame)
                if packet:
                    output.mux(packet)
            packet = stream.encode()
            if packet:
                output.mux(packet)
        print(f"[VIDEO][COLOR] ✅ {camera} 完成: {out_path}")
    except Exception as e:
        print(f"[VIDEO][COLOR] ❌ {camera} 失败: {e}")
        raise
    finally:
        shutil.rmtree(camera_dir, ignore_errors=True)
        gc.collect()
        print(f"[VIDEO][COLOR] 🗑️  {camera} 临时文件已清理")


def _encode_depth_camera_worker(
    camera_dir: str,
    camera: str,
    out_path: str,
    train_hz: int,
    apply_denoise: bool,
    ffmpeg_threads: int = 1,
):
    """
    子进程：编码深度视频（ffmpeg + 可选去噪）
    """
    import glob
    import shutil
    import tempfile
    import gc
    import numpy as np
    import cv2
    import subprocess

    try:
        frame_files = sorted(glob.glob(os.path.join(camera_dir, "*.png")))
        print(f"[VIDEO][DEPTH] {camera}: 帧数 {len(frame_files)}")
        if len(frame_files) == 0:
            shutil.rmtree(camera_dir, ignore_errors=True)
            gc.collect()
            print(f"[VIDEO][DEPTH] {camera}: 无帧，已清理")
            return

        is_hand_camera = "wrist_cam" in camera
        if is_hand_camera and apply_denoise:
            print(f"[VIDEO][DEPTH] {camera}: 应用去噪")

        with tempfile.TemporaryDirectory() as processed_dir:
            for idx, frame_file in enumerate(frame_files):
                img = cv2.imread(frame_file, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim > 2:
                    img = img[:, :, 0]
                if img.dtype != np.uint16:
                    img = img.astype(np.uint16)

                if is_hand_camera and apply_denoise:
                    try:
                        from converter.image.video_denoising import repair_depth_noise_focused

                        img = repair_depth_noise_focused(
                            img,
                            max_valid_depth=10000,
                            median_kernel=5,
                            detect_white_spots=True,
                            spot_size_range=(10, 1000),
                        )
                    except Exception:
                        pass

                processed_path = os.path.join(processed_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(processed_path, img)

            cmd = [
                "ffmpeg",
                "-y",
                "-threads",
                str(max(1, ffmpeg_threads)),
                "-framerate",
                str(train_hz),
                "-i",
                os.path.join(processed_dir, "frame_%06d.png"),
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "gray16le",
                out_path,
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[VIDEO][DEPTH] ✅ {camera} 完成: {out_path}")
    except Exception as e:
        print(f"[VIDEO][DEPTH] ❌ {camera} 失败: {e}")
        raise
    finally:
        shutil.rmtree(camera_dir, ignore_errors=True)
        gc.collect()
        print(f"[VIDEO][DEPTH] 🗑️  {camera} 临时文件已清理")


# ==================== 流式视频编码器 ====================
