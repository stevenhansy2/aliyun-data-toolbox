import os
import re
import json

def format_suffix(file_size, file_duration, number_of_records):
    size_gb = round(file_size, 2)  # 已是GB，无需转换
    duration_h = round(file_duration / 3600, 2)
    # "p" 代表小数点
    size_str = f"{str(size_gb).replace('.', 'p')}GB"
    duration_str = f"{str(duration_h).replace('.', 'p')}h"
    return f"-{size_str}_{number_of_records}counts_{duration_str}"

def remove_suffix(name):
    # 移除后缀: -xxGB_xxcounts_xxh
    return re.sub(r'-\d+p\d+GB_\d+counts_\d+p\d+h$', '', name)

def get_json_stats(json_paths):
    total_size = 0
    total_duration = 0
    total_records = 0
    for jp in json_paths:
        try:
            with open(jp, 'r') as f:
                data = json.load(f)
            total_size += data.get('total_size', 0)
            total_duration += data.get('total_duration', 0)
            total_records += data.get('record_count', 0)
        except Exception:
            continue
    return total_size, total_duration, total_records

def adjust_scene_names(root_path, scene_name):
    # 支持带后缀的主场景名
    scene_base = remove_suffix(scene_name)
    # 查找实际主场景目录（可能已带后缀）
    scene_dir_candidates = [d for d in os.listdir(root_path)
                           if os.path.isdir(os.path.join(root_path, d)) and remove_suffix(d) == scene_base]
    if not scene_dir_candidates:
        print(f"主场景 {scene_name} 不存在，操作中止。")
        return
    scene_dir = scene_dir_candidates[0]
    scene_path = os.path.join(root_path, scene_dir)

    stats_dir = os.path.join(root_path, 'task_stats')
    if not os.path.isdir(stats_dir):
        print(f"统计目录不存在: {stats_dir}")
        return

    # 1. 主场景后缀
    scene_jsons = [os.path.join(stats_dir, f) for f in os.listdir(stats_dir)
                   if f.startswith(scene_base + '-') and f.endswith('.json')]
    scene_size, scene_duration, scene_records = get_json_stats(scene_jsons)
    scene_suffix = format_suffix(scene_size, scene_duration, scene_records)

    # 重命名主场景目录
    new_scene_name = scene_base #+ scene_suffix
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
        # 查找所有相关 json
        sub_jsons = [os.path.join(stats_dir, f) for f in os.listdir(stats_dir)
                     if f.startswith(scene_base + '-' + sub_base + '-') and f.endswith('.json')]
        sub_size, sub_duration, sub_records = get_json_stats(sub_jsons)
        sub_suffix = format_suffix(sub_size, sub_duration, sub_records)
        new_sub_scene = sub_base #+ sub_suffix
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
            # 查找对应 json
            json_name = f"{scene_base}-{sub_base}-{act_base}.json"
            json_path = os.path.join(stats_dir, json_name)
            if not os.path.isfile(json_path):
                continue
            act_size, act_duration, act_records = get_json_stats([json_path])
            act_suffix = format_suffix(act_size, act_duration, act_records)
            new_action = act_base + act_suffix
            new_action_path = os.path.join(sub_scene_path, new_action)
            if action_path != new_action_path:
                os.rename(action_path, new_action_path)

if __name__ == "__main__":
    # 示例用法
    adjust_scene_names('/home/c/桌面/刻行action/adjust-file-name-local/', 'manufacturing_plant')