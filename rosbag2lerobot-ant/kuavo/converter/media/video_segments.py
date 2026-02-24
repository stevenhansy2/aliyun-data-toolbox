"""Batch segment encoding and concatenation helpers."""

import glob
import os
import shutil
import subprocess

import av
from PIL import Image

def _encode_batch_segment_color(
    batch_id: int,
    camera: str,
    temp_dir: str,
    segment_dir: str,
    train_hz: int,
    chunk_size: int = 800,
) -> str:
    """
    编码单个 batch 的彩色视频片段

    Args:
        batch_id: 批次ID
        camera: 相机名称
        temp_dir: 临时帧目录 (包含 batch_XXXX_frame_XXXXXX.jpg)
        segment_dir: 片段输出目录
        train_hz: 视频帧率
        chunk_size: 每批次帧数，用于计算全局帧起始位置

    Returns:
        片段文件路径，失败返回 None
    """
    import av
    from PIL import Image
    import glob
    import time as time_module

    start_time = time_module.time()

    # 查找该 batch 的帧
    pattern = os.path.join(temp_dir, f"batch_{batch_id:04d}_frame_*.jpg")
    frame_files = sorted(glob.glob(pattern))

    if len(frame_files) == 0:
        print(f"[PIPELINE][{camera}] Batch {batch_id}: 无帧，跳过")
        return None

    # 输出片段路径
    os.makedirs(segment_dir, exist_ok=True)
    segment_path = os.path.join(segment_dir, f"segment_{batch_id:04d}.mp4")

    # 计算该 batch 的全局帧起始位置 (batch_id 从 1 开始)
    global_frame_start = (batch_id - 1) * chunk_size

    try:
        video_options = {
            "g": "2",
            "crf": "30",
        }
        first_img = Image.open(frame_files[0])
        width, height = first_img.size

        with av.open(str(segment_path), "w") as output:
            stream = output.add_stream("libx264", train_hz, options=video_options)
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height

            for local_idx, frame_file in enumerate(frame_files):
                img = Image.open(frame_file).convert("RGB")
                frame = av.VideoFrame.from_image(img)
                # 设置全局 PTS，确保拼接后时间戳连续
                frame.pts = global_frame_start + local_idx
                packet = stream.encode(frame)
                if packet:
                    output.mux(packet)
            packet = stream.encode()
            if packet:
                output.mux(packet)

        elapsed_ms = (time_module.time() - start_time) * 1000
        print(
            f"[PIPELINE][{camera}] Batch {batch_id}: {len(frame_files)} 帧 (起始帧={global_frame_start}) → {segment_path} ({elapsed_ms:.0f}ms)"
        )
        return segment_path

    except Exception as e:
        print(f"[PIPELINE][{camera}] Batch {batch_id} 编码失败: {e}")
        return None


def _concat_segments_ffmpeg(
    segment_paths: list, output_path: str, train_hz: int = 30
) -> bool:
    """
    使用 ffmpeg 拼接视频片段（重新编码以确保时间戳正确）

    Args:
        segment_paths: 片段文件路径列表（已排序）
        output_path: 最终输出路径
        train_hz: 视频帧率

    Returns:
        成功返回 True
    """
    import tempfile
    import time as time_module

    if not segment_paths:
        print(f"[PIPELINE] 拼接失败: 无片段")
        return False

    start_time = time_module.time()

    # 创建 filelist.txt
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg in segment_paths:
            f.write(f"file '{seg}'\n")
        filelist_path = f.name

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 使用重新编码确保时间戳严格连续
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            filelist_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "30",
            "-g",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(train_hz),
            "-vsync",
            "cfr",  # 强制恒定帧率，确保时间戳精确
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[PIPELINE] ffmpeg 拼接失败: {result.stderr}")
            return False

        elapsed_ms = (time_module.time() - start_time) * 1000
        print(
            f"[PIPELINE] 拼接完成: {len(segment_paths)} 片段 → {output_path} ({elapsed_ms:.0f}ms)"
        )
        return True

    finally:
        os.unlink(filelist_path)


