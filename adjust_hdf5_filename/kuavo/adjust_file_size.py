import os
import re
import json
import shutil
import sys


def format_suffix(file_size, file_duration, number_of_records):
    size_gb = round(file_size, 2)  # 已是GB，无需转换
    duration_h = round(file_duration / 3600, 2)
    size_str = f"{str(size_gb).replace('.', 'p')}GB"
    duration_str = f"{str(duration_h).replace('.', 'p')}h"
    return f"-{size_str}_{number_of_records}counts_{duration_str}"


def remove_suffix(name):
    # 移除后缀: -xxGB_xxcounts_xxh
    return re.sub(r"-\d+p\d+GB_\d+counts_\d+p\d+h$", "", name)


def is_uuid_folder_complete(uuid_path):
    """检查uuid目录结构完整性和metadata有效性，不完整或无效则删除该目录并返回False"""
    required_files = [
        os.path.join(uuid_path, "metadata.json"),
        os.path.join(uuid_path, "camera/depth/hand_left_depth.mkv"),
        os.path.join(uuid_path, "camera/depth/hand_right_depth.mkv"),
        os.path.join(uuid_path, "camera/depth/head_depth.mkv"),
        os.path.join(uuid_path, "camera/video/hand_left_color.mp4"),
        os.path.join(uuid_path, "camera/video/hand_right_color.mp4"),
        os.path.join(uuid_path, "camera/video/head_color.mp4"),
        os.path.join(uuid_path, "parameters/hand_left_extrinsic_params.json"),
        os.path.join(uuid_path, "parameters/hand_left_intrinsic_params.json"),
        os.path.join(uuid_path, "parameters/hand_right_extrinsic_params.json"),
        os.path.join(uuid_path, "parameters/hand_right_intrinsic_params.json"),
        os.path.join(uuid_path, "parameters/head_extrinsic_params.json"),
        os.path.join(uuid_path, "parameters/head_intrinsic_params.json"),
        os.path.join(uuid_path, "proprio_stats/proprio_stats.hdf5"),
        os.path.join(uuid_path, "proprio_stats/proprio_stats_original.hdf5"),
    ]
    for f in required_files:
        if not os.path.isfile(f):
            print(f"❌ 缺失文件: {f}，删除uuid目录: {uuid_path}")
            shutil.rmtree(uuid_path, ignore_errors=True)
            return False
    meta_path = os.path.join(uuid_path, "metadata.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not meta or not isinstance(meta, dict) or len(meta) == 0:
            print(f"❌ metadata.json为空或无效，删除uuid目录: {uuid_path}")
            shutil.rmtree(uuid_path, ignore_errors=True)
            return False
    except Exception:
        print(f"❌ metadata.json解析失败，删除uuid目录: {uuid_path}")
        shutil.rmtree(uuid_path, ignore_errors=True)
        return False
    return True


def get_folder_stats(folder_path):
    """统计某文件夹下所有uuid的总大小、总时长、数量"""
    total_size = 0
    total_duration = 0
    total_count = 0
    for action in os.listdir(folder_path):
        action_path = os.path.join(folder_path, action)
        if not os.path.isdir(action_path):
            continue
        for uuid_folder in os.listdir(action_path):
            uuid_path = os.path.join(action_path, uuid_folder)
            if os.path.isdir(uuid_path) and is_uuid_folder_complete(uuid_path):
                meta_path = os.path.join(uuid_path, "metadata.json")
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    total_size += meta.get("file_size", 0)
                    total_duration += meta.get("file_duration", 0)
                    total_count += 1
                except Exception:
                    continue
    return total_size, total_duration, total_count


def get_action_stats(action_path):
    total_size = 0
    total_duration = 0
    total_count = 0
    for uuid_folder in os.listdir(action_path):
        uuid_path = os.path.join(action_path, uuid_folder)
        if os.path.isdir(uuid_path) and is_uuid_folder_complete(uuid_path):
            meta_path = os.path.join(uuid_path, "metadata.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                total_size += meta.get("file_size", 0)
                total_duration += meta.get("file_duration", 0)
                total_count += 1
            except Exception:
                continue
    return total_size, total_duration, total_count


def get_scene_stats(scene_path):
    """递归统计主场景下所有子场景、动作、uuid的总大小、总时长、数量"""
    total_size = 0
    total_duration = 0
    total_count = 0
    for sub_scene in os.listdir(scene_path):
        sub_scene_path = os.path.join(scene_path, sub_scene)
        if not os.path.isdir(sub_scene_path):
            continue
        sub_size, sub_duration, sub_count = get_folder_stats(sub_scene_path)
        total_size += sub_size
        total_duration += sub_duration
        total_count += sub_count
    return total_size, total_duration, total_count


def adjust_scene_names_local(root_path, scene_name):
    # 支持带后缀的主场景名
    scene_base = remove_suffix(scene_name)
    # 查找实际主场景目录（可能已带后缀）
    scene_dir_candidates = [
        d
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d)) and remove_suffix(d) == scene_base
    ]
    if not scene_dir_candidates:
        print(f"主场景 {scene_name} 不存在，操作中止。")
        return
    scene_dir = scene_dir_candidates[0]
    scene_path = os.path.join(root_path, scene_dir)

    # 1. 主场景后缀（递归统计所有子场景下所有动作下所有uuid）
    scene_size, scene_duration, scene_records = get_scene_stats(scene_path)
    scene_suffix = format_suffix(scene_size, scene_duration, scene_records)
    new_scene_name = scene_base + scene_suffix
    new_scene_path = os.path.join(root_path, new_scene_name)
    if scene_path != new_scene_path:
        os.rename(scene_path, new_scene_path)
        scene_path = new_scene_path

    # 2. 子场景
    for sub_scene in os.listdir(scene_path):
        sub_scene_path = os.path.join(scene_path, sub_scene)
        if not os.path.isdir(sub_scene_path):
            continue
        sub_base = remove_suffix(sub_scene)
        sub_size, sub_duration, sub_records = get_folder_stats(sub_scene_path)
        sub_suffix = format_suffix(sub_size, sub_duration, sub_records)
        new_sub_scene = sub_base + sub_suffix
        new_sub_scene_path = os.path.join(scene_path, new_sub_scene)
        if sub_scene_path != new_sub_scene_path:
            os.rename(sub_scene_path, new_sub_scene_path)
            sub_scene_path = new_sub_scene_path

        # 3. 连续动作
        for action in os.listdir(sub_scene_path):
            action_path = os.path.join(sub_scene_path, action)
            if not os.path.isdir(action_path):
                continue
            act_base = remove_suffix(action)
            act_size, act_duration, act_records = get_action_stats(action_path)
            act_suffix = format_suffix(act_size, act_duration, act_records)
            new_action = act_base + act_suffix
            new_action_path = os.path.join(sub_scene_path, new_action)
            if action_path != new_action_path:
                os.rename(action_path, new_action_path)


def get_unique_folder_name(parent_path, base_name, suffix, split_idx=1):
    # 先尝试 base_name+suffix
    candidate = base_name + suffix
    candidate_path = os.path.join(parent_path, candidate)
    while os.path.exists(candidate_path):
        candidate = f"{base_name}_split{split_idx}{suffix}"
        candidate_path = os.path.join(parent_path, candidate)
        split_idx += 1
    return candidate, candidate_path


def get_action_stats_from_uuids(uuid_paths):
    total_size = 0
    total_duration = 0
    total_count = 0
    for uuid_path in uuid_paths:
        if os.path.isdir(uuid_path) and is_uuid_folder_complete(uuid_path):
            meta_path = os.path.join(uuid_path, "metadata.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                total_size += meta.get("file_size", 0)
                total_duration += meta.get("file_duration", 0)
                total_count += 1
            except Exception:
                continue
    return total_size, total_duration, total_count


def split_sub_scene_by_size_limit(root_path, scene_name, size_limit_gb):
    scene_base = remove_suffix(scene_name)
    # 查找主场景目录
    scene_dir_candidates = [
        d
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d)) and remove_suffix(d) == scene_base
    ]
    if not scene_dir_candidates:
        print(f"主场景 {scene_name} 不存在，操作中止。")
        return
    scene_dir = scene_dir_candidates[0]
    scene_path = os.path.join(root_path, scene_dir)

    for sub_scene in os.listdir(scene_path):
        sub_scene_path = os.path.join(scene_path, sub_scene)
        if not os.path.isdir(sub_scene_path):
            continue

        # 统计子场景总大小
        sub_size, _, _ = get_folder_stats(sub_scene_path)
        if sub_size < size_limit_gb:
            continue  # 小于限额跳过

        split_idx = 1
        curr_size = 0
        curr_duration = 0
        curr_count = 0
        curr_actions = []  # 整个动作目录（完全纳入当前块）
        # 关键修复：按动作名分组收集uuid，避免不同动作的uuid混到一个动作目录里
        curr_uuids_by_action = {}

        actions = sorted(os.listdir(sub_scene_path))
        for action in actions:
            action_path = os.path.join(sub_scene_path, action)
            if not os.path.isdir(action_path):
                continue
            act_size, act_duration, act_count = get_action_stats(action_path)
            if curr_size + act_size <= size_limit_gb:
                curr_size += act_size
                curr_duration += act_duration
                curr_count += act_count
                curr_actions.append(action_path)  # 整动作先累积
                continue

            # 当前动作文件夹需要拆分
            for uuid_folder in sorted(os.listdir(action_path)):
                uuid_path = os.path.join(action_path, uuid_folder)
                meta_path = os.path.join(uuid_path, "metadata.json")
                if not os.path.isdir(uuid_path) or not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    uuid_size = meta.get("file_size", 0)
                    uuid_duration = meta.get("file_duration", 0)
                except Exception:
                    continue

                # 达到限额则先落盘当前分块（仅在原子场景内新建动作目录，不新建子场景目录）
                if curr_size + uuid_size > size_limit_gb and curr_size > 0:
                    sub_size_blk = curr_size
                    sub_duration_blk = curr_duration
                    sub_count_blk = curr_count
                    sub_base = remove_suffix(sub_scene)
                    sub_suffix = format_suffix(
                        sub_size_blk, sub_duration_blk, sub_count_blk
                    )

                    # 1) 移动整动作：重命名为带后缀的新动作目录，仍放在 sub_scene_path 下
                    for act_path in curr_actions:
                        act_name = os.path.basename(act_path)
                        act_base = remove_suffix(act_name)
                        # 重新统计该整动作，避免复用过期统计
                        a_size, a_dur, a_cnt = get_action_stats(act_path)
                        act_suffix = format_suffix(a_size, a_dur, a_cnt)
                        new_action_name, new_action_path = get_unique_folder_name(
                            sub_scene_path, act_base, act_suffix, split_idx
                        )
                        os.rename(act_path, new_action_path)

                    # 2) 分动作：为每个动作单独建目录并移动其 uuid（仍在 sub_scene_path 下）
                    for action_name, uuids_list in curr_uuids_by_action.items():
                        act_base = remove_suffix(action_name)
                        act_size_blk, act_duration_blk, act_count_blk = (
                            get_action_stats_from_uuids(uuids_list)
                        )
                        act_suffix = format_suffix(
                            act_size_blk, act_duration_blk, act_count_blk
                        )
                        new_action_name, new_action_path = get_unique_folder_name(
                            sub_scene_path, act_base, act_suffix, split_idx
                        )
                        os.makedirs(new_action_path, exist_ok=True)
                        for uuid_p in uuids_list:
                            shutil.move(uuid_p, new_action_path)

                    # 准备下一个分块
                    split_idx += 1
                    curr_size = 0
                    curr_duration = 0
                    curr_count = 0
                    curr_actions = []
                    curr_uuids_by_action = {}

                # 把当前 uuid 纳入正在累积的分块
                curr_size += uuid_size
                curr_duration += uuid_duration
                curr_count += 1
                curr_uuids_by_action.setdefault(action, []).append(uuid_path)

            # 拆分动作剩余 uuid 不立即分块，留到下一个分块累加

        # 最后剩余部分也要分块（仍在原子场景内创建动作目录，不新建子场景目录）
        if curr_actions or curr_uuids_by_action:
            sub_size_blk = curr_size
            sub_duration_blk = curr_duration
            sub_count_blk = curr_count
            sub_base = remove_suffix(sub_scene)
            sub_suffix = format_suffix(sub_size_blk, sub_duration_blk, sub_count_blk)

            # 移动整动作
            for act_path in curr_actions:
                act_name = os.path.basename(act_path)
                act_base = remove_suffix(act_name)
                a_size, a_dur, a_cnt = get_action_stats(act_path)
                act_suffix = format_suffix(a_size, a_dur, a_cnt)
                new_action_name, new_action_path = get_unique_folder_name(
                    sub_scene_path, act_base, act_suffix, split_idx
                )
                os.rename(act_path, new_action_path)

            # 分动作：为每个动作单独建目录并移动其 uuid
            for action_name, uuids_list in curr_uuids_by_action.items():
                act_base = remove_suffix(action_name)
                act_size_blk, act_duration_blk, act_count_blk = (
                    get_action_stats_from_uuids(uuids_list)
                )
                act_suffix = format_suffix(
                    act_size_blk, act_duration_blk, act_count_blk
                )
                new_action_name, new_action_path = get_unique_folder_name(
                    sub_scene_path, act_base, act_suffix, split_idx
                )
                os.makedirs(new_action_path, exist_ok=True)
                for uuid_p in uuids_list:
                    shutil.move(uuid_p, new_action_path)

            split_idx += 1

        # 清理空动作文件夹（若部分 uuid 被移走导致原动作目录为空）
        for action in os.listdir(sub_scene_path):
            action_path = os.path.join(sub_scene_path, action)
            if os.path.isdir(action_path) and not os.listdir(action_path):
                shutil.rmtree(action_path)

    print("分块完成。")


if __name__ == "__main__":
    # 示例用法
    # adjust_scene_names_local(
    #     "/home/c/桌面/kuavo-data-toolbox/adjust_hdf5_filename/", "manufacturing_plant"
    # )
    root_path = sys.argv[1]
    scene_name = sys.argv[2]
    # 先分块和清理
    split_sub_scene_by_size_limit(
        root_path,
        scene_name,
        1500,
    )
    # 再刷新目录名（统计后再重命名）
    adjust_scene_names_local(root_path, scene_name)

