#!/usr/bin/env python3
"""
统计testoutput目录下所有文件夹的大小和视频时长，并按三级目录结构重新组织
"""

import os
import shutil
import argparse
from pathlib import Path
import subprocess
import json


def collect_and_save_metadata_json_from_data(
    metadata_list, scene, sub_scene, continuous_action, output_root
):
    """
    直接用已收集的metadata数据生成json
    """
    task_info_dir = Path(output_root) / "task_info"
    task_info_dir.mkdir(parents=True, exist_ok=True)
    json_name = f"{scene}-{sub_scene}-{continuous_action}.json"
    json_path = task_info_dir / json_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    print(f"\n已生成 task_info: {json_path}")


def get_folder_size(folder_path):
    """获取文件夹大小（字节）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, FileNotFoundError):
                pass
    return total_size


def format_size_gb(size_bytes):
    """将字节转换为GB格式，用p表示小数点"""
    size_gb = size_bytes / (1024**3)
    int_part = int(size_gb)
    frac_part = int(round((size_gb - int_part) * 100))
    return f"{int_part}p{frac_part:02d}"


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    if not os.path.exists(video_path):
        return 0

    try:
        # 使用ffprobe获取视频信息
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            return duration
        else:
            print(f"警告: 无法获取视频时长 {video_path}")
            return 0

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"错误: 获取视频时长失败 {video_path}: {e}")
        return 0


def format_duration_hours(total_seconds):
    """将秒数转换为小时格式，用p表示小数点"""
    hours = total_seconds / 3600
    int_part = int(hours)
    frac_part = int(round((hours - int_part) * 100))
    return f"{int_part}p{frac_part:02d}"


def count_folders(base_path):
    """统计文件夹数量"""
    count = 0
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            count += 1
    return count


def collect_and_save_metadata_json(
    folders, scene, sub_scene, continuous_action, output_root
):
    """
    收集每个数据文件夹下的metadata.json，合并后保存到task_info目录下
    """
    task_info_dir = Path(output_root) / "task_info"
    task_info_dir.mkdir(parents=True, exist_ok=True)
    # 文件名格式：场景名称-子场景名称-连续动作名称.json
    json_name = f"{scene}-{sub_scene}-{continuous_action}.json"
    json_path = task_info_dir / json_name

    merged_metadata = []
    for folder in folders:
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    merged_metadata.append(data)
            except Exception as e:
                print(f"  警告: 读取 {metadata_path} 失败: {e}")
        else:
            print(f"  警告: 未找到 {metadata_path}")

    # 保存合并后的json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, ensure_ascii=False, indent=2)
    print(f"\n已生成 task_info: {json_path}")


def reorganize_folders(testoutput_dir, scene, sub_scene, continuous_action):
    """重新组织文件夹结构"""
    testoutput_path = Path(testoutput_dir)

    if not testoutput_path.exists():
        print(f"错误: 目录 {testoutput_dir} 不存在")
        return

    folders = [f for f in testoutput_path.iterdir() if f.is_dir()]
    # 新增：先收集metadata
    merged_metadata = []
    for folder in folders:
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    merged_metadata.append(data)
            except Exception as e:
                print(f"  警告: 读取 {metadata_path} 失败: {e}")
        else:
            print(f"  警告: 未找到 {metadata_path}")
    # ...existing code...
    # 移动所有文件夹到新的目录结构
    # ...existing code...
    # 新增：生成 task_info 目录及合并后的 json
    collect_and_save_metadata_json_from_data(
        merged_metadata, scene, sub_scene, continuous_action, testoutput_path
    )

    if not folders:
        print(f"警告: 在 {testoutput_dir} 中没有找到任何文件夹")
        return

    print(f"找到 {len(folders)} 个文件夹需要处理")

    # 统计总大小和总时长
    total_size = 0
    total_duration = 0

    print("正在统计文件夹大小和视频时长...")

    for folder in folders:
        # 统计文件夹大小
        folder_size = get_folder_size(folder)
        total_size += folder_size

        # 统计视频时长
        video_path = folder / "camera" / "video" / "head_cam_h.mp4"
        if video_path.exists():
            duration = get_video_duration(str(video_path))
            total_duration += duration
            print(f"  {folder.name}: {format_size_gb(folder_size)}GB, {duration:.2f}秒")
        else:
            print(f"  {folder.name}: {format_size_gb(folder_size)}GB, 未找到视频文件")

    # 格式化统计结果
    size_str = format_size_gb(total_size)
    duration_str = format_duration_hours(total_duration)
    count = len(folders)

    print(f"\n统计结果:")
    print(f"  总大小: {total_size / (1024**3):.2f} GB")
    print(f"  总时长: {total_duration / 3600:.2f} 小时")
    print(f"  文件夹数: {count}")

    # 创建三级目录结构
    scene_dir = f"{scene}-{size_str}GB_{count}counts_{duration_str}h"
    sub_scene_dir = f"{sub_scene}-{size_str}GB_{count}counts_{duration_str}h"
    continuous_action_dir = (
        f"{continuous_action}-{size_str}GB_{count}counts_{duration_str}h"
    )

    target_path = testoutput_path / scene_dir / sub_scene_dir / continuous_action_dir
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"\n正在移动文件夹到新的目录结构...")
    print(f"目标路径: {target_path}")

    # 移动所有文件夹到新的目录结构
    moved_count = 0
    for folder in folders:
        try:
            target_folder = target_path / folder.name
            if target_folder.exists():
                print(f"  跳过 {folder.name} (目标已存在)")
                continue

            shutil.move(str(folder), str(target_folder))
            moved_count += 1
            print(
                f"  移动 {folder.name} -> {target_folder.relative_to(testoutput_path)}"
            )

        except Exception as e:
            print(f"  错误: 移动 {folder.name} 失败: {e}")

    print(f"\n重组完成!")
    print(f"成功移动 {moved_count} 个文件夹")
    print(f"新的目录结构:")
    print(f"└── {scene_dir}/")
    print(f"    └── {sub_scene_dir}/")
    print(f"        └── {continuous_action_dir}/")
    print(f"            ├── [包含 {moved_count} 个数据文件夹]")

    return {
        "total_size_gb": total_size / (1024**3),
        "total_duration_hours": total_duration / 3600,
        "folder_count": count,
        "moved_count": moved_count,
        "scene_dir": scene_dir,
        "sub_scene_dir": sub_scene_dir,
        "continuous_action_dir": continuous_action_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="统计testoutput目录并重新组织文件夹结构"
    )
    parser.add_argument(
        "--testoutput_dir", default="hdf5_0717/", type=str, help="testoutput目录路径"
    )
    parser.add_argument("--scene", default="Factory", type=str, help="场景名称")
    parser.add_argument(
        "--sub_scene", default="Material_sorting", type=str, help="子场景名称"
    )
    parser.add_argument(
        "--continuous_action",
        default="Place_the_coil_in_the_corresponding_box",
        type=str,
        help="连续动作名称",
    )
    parser.add_argument("--dry_run", action="store_true", help="仅统计不移动文件")

    args = parser.parse_args()

    print("=" * 60)
    print("文件夹统计和重组工具")
    print("=" * 60)
    print(f"输入目录: {args.testoutput_dir}")
    print(f"场景: {args.scene}")
    print(f"子场景: {args.sub_scene}")
    print(f"连续动作: {args.continuous_action}")
    print(f"仅统计模式: {'是' if args.dry_run else '否'}")
    print("=" * 60)

    if args.dry_run:
        # 仅统计模式
        testoutput_path = Path(args.testoutput_dir)
        if not testoutput_path.exists():
            print(f"错误: 目录 {args.testoutput_dir} 不存在")
            return

        folders = [f for f in testoutput_path.iterdir() if f.is_dir()]
        total_size = 0
        total_duration = 0

        print("正在统计...")
        for folder in folders:
            folder_size = get_folder_size(folder)
            total_size += folder_size

            video_path = folder / "camera" / "video" / "head_cam_h.mp4"
            if video_path.exists():
                duration = get_video_duration(str(video_path))
                total_duration += duration

        size_str = format_size_gb(total_size)
        duration_str = format_duration_hours(total_duration)
        count = len(folders)

        print(f"\n统计结果:")
        print(f"  总大小: {total_size / (1024**3):.2f} GB")
        print(f"  总时长: {total_duration / 3600:.2f} 小时")
        print(f"  文件夹数: {count}")

        print(f"\n预期的目录结构:")
        scene_dir = f"{args.scene}-{size_str}GB_{count}counts_{duration_str}h"
        sub_scene_dir = f"{args.sub_scene}-{size_str}GB_{count}counts_{duration_str}h"
        continuous_action_dir = (
            f"{args.continuous_action}-{size_str}GB_{count}counts_{duration_str}h"
        )

        print(f"└── {scene_dir}/")
        print(f"    └── {sub_scene_dir}/")
        print(f"        └── {continuous_action_dir}/")
        print(f"            ├── [将包含 {count} 个数据文件夹]")

    else:
        # 执行重组
        result = reorganize_folders(
            args.testoutput_dir, args.scene, args.sub_scene, args.continuous_action
        )

        if result:
            print(f"\n最终统计:")
            print(f"  总大小: {result['total_size_gb']:.2f} GB")
            print(f"  总时长: {result['total_duration_hours']:.2f} 小时")
            print(f"  文件夹总数: {result['folder_count']}")
            print(f"  成功移动: {result['moved_count']}")


if __name__ == "__main__":
    main()
