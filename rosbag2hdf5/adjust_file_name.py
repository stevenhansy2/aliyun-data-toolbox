import os
import re
import json
import sys
import argparse

def format_suffix(file_size, file_duration, number_of_records):
    size_gb = round(file_size, 2)  # 已是GB，无需转换
    duration_h = round(file_duration / 3600, 2) if file_duration > 0 else 0
    # "p" 代表小数点
    size_str = f"{str(size_gb).replace('.', 'p')}GB"
    duration_str = f"{str(duration_h).replace('.', 'p')}h"
    # 对于整数值，确保格式正确（例如 0h -> 0p0h）
    if 'p' not in size_str:
        size_str = size_str.replace('GB', 'p0GB')
    if 'p' not in duration_str:
        duration_str = duration_str.replace('h', 'p0h')
    return f"-{size_str}_{int(number_of_records)}counts_{duration_str}"

def remove_suffix(name):
    # 移除后缀: -xxGB_xxcounts_xxh 或 -xxpxxGB_xxcounts_xxpxxh
    return re.sub(r'-\d+p?\d*GB_\d+counts_\d+p?\d*h$', '', name)

import subprocess

def oss_mv(src, dst):
    print(f"[DEBUG] 开始移动: {src} -> {dst}")
    
    # 首先检查源目录是否存在
    check_src_cmd = ["ossutil", "ls", src]
    check_result = subprocess.run(check_src_cmd, capture_output=True, text=True)
    if check_result.returncode != 0 or "Object Number is: 0" in check_result.stdout:
        print(f"[ERROR] 源目录不存在或为空: {src}")
        return
    
    # 确保源路径和目标路径对目录有正确的格式
    if not src.endswith('/'):
        src_for_cp = src + '/'
    else:
        src_for_cp = src
    
    cp_cmd = ["ossutil", "cp", "-r", src_for_cp, dst]
    cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)
    if cp_result.returncode != 0:
        print(f"OSS复制失败: {src} -> {dst}\n{cp_result.stderr}")
        return
    
    # 检查新目录是否有内容
    ls_cmd = ["ossutil", "ls", dst]
    ls_result = subprocess.run(ls_cmd, capture_output=True, text=True)
    
    if "Object Number is: 0" in ls_result.stdout:
        print(f"OSS复制失败：新目录无内容，保留原目录！{dst}")
        return
    
    # 删除原目录
    rm_cmd = ["ossutil", "rm", "-r", src, "-f"]
    rm_result = subprocess.run(rm_cmd, capture_output=True, text=True)
    if rm_result.returncode != 0:
        print(f"OSS删除失败: {src}\n{rm_result.stderr}")
    else:
        print(f"OSS重命名成功: {src} -> {dst}")

def adjust_scene_names_oss(root_path, scene_name):
    # 1. 主场景后缀
    stats_dir = f"{root_path}/task_stats"
    print("stats_dir",stats_dir)
    # 获取所有主场景相关json
    ls_cmd = ["ossutil", "ls", stats_dir]
    result = subprocess.run(ls_cmd, capture_output=True, text=True)
    
    scene_jsons = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or 'ObjectName' in line or 'LastModifiedTime' in line:
            continue  # 跳过标题行
        
        # ossutil ls 输出格式：时间 大小 存储类型 ETAG 路径
        # 路径在最后一列
        parts = line.split()
        if len(parts) >= 5:
            oss_path = parts[-1]
            if oss_path.endswith('.json') and oss_path.startswith('oss://'):
                filename = os.path.basename(oss_path)
                if filename.startswith(f"{scene_name}-"):
                    scene_jsons.append(oss_path)

    # 计算主场景的后缀
    scene_size, scene_duration, scene_records = get_json_stats_oss(scene_jsons)
    scene_suffix = format_suffix(scene_size, scene_duration, scene_records)
    new_scene_name = remove_suffix(scene_name) + scene_suffix
    
    # 主场景路径
    scene_path = f"{root_path}/{scene_name}"
    new_scene_path = f"{root_path}/{new_scene_name}"
    print(f"scene_path: {scene_path}")
    print(f"new_scene_path: {new_scene_path}")
    
    # 2. 遍历子场景
    result = subprocess.run(["ossutil", "ls", scene_path], capture_output=True, text=True)
    
    # 收集唯一的子场景
    sub_scenes = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or 'ObjectName' in line or 'LastModifiedTime' in line:
            continue  # 跳过标题行
        
        # 解析ossutil ls输出，路径在最后一列
        parts = line.split()
        if len(parts) >= 5:
            oss_path = parts[-1]
            if oss_path.startswith('oss://') and oss_path.endswith('/'):
                # 这是一个目录
                # 提取子场景名称（第一级子目录）
                relative_path = oss_path.replace(scene_path + '/', '', 1).rstrip('/')
                if '/' in relative_path:
                    sub_scene = relative_path.split('/')[0]
                else:
                    sub_scene = relative_path
                
                if sub_scene:
                    sub_scenes.add(sub_scene)

    # 处理每个唯一的子场景
    for sub_scene in sorted(sub_scenes):
        # 计算子场景的后缀
        sub_jsons = [j for j in scene_jsons if f"-{remove_suffix(sub_scene)}-" in j]
        sub_size, sub_duration, sub_records = get_json_stats_oss(sub_jsons)
        sub_suffix = format_suffix(sub_size, sub_duration, sub_records)
        new_sub_scene = remove_suffix(sub_scene) + sub_suffix
        
        sub_scene_path = f"{scene_path}/{sub_scene}"
        
        # 3. 遍历连续动作
        result2 = subprocess.run(["ossutil", "ls", sub_scene_path], capture_output=True, text=True)
        
        # 收集唯一的动作
        actions = set()
        for line2 in result2.stdout.splitlines():
            line2 = line2.strip()
            if not line2 or 'ObjectName' in line2 or 'LastModifiedTime' in line2:
                continue
            
            parts2 = line2.split()
            if len(parts2) >= 5:
                oss_path2 = parts2[-1]
                if oss_path2.startswith('oss://') and oss_path2.endswith('/'):
                    # 提取动作名称（第二级子目录）
                    relative_path2 = oss_path2.replace(sub_scene_path + '/', '', 1).rstrip('/')
                    if '/' in relative_path2:
                        action = relative_path2.split('/')[0]
                    else:
                        action = relative_path2
                    
                    if action:
                        actions.add(action)
        
        # 处理每个唯一的动作
        for action in sorted(actions):
            action_path = f"{sub_scene_path}/{action}"
            
            # 查找对应的JSON统计文件
            json_name = f"{remove_suffix(scene_name)}-{remove_suffix(sub_scene)}-{remove_suffix(action)}.json"
            json_path = f"{stats_dir}/{json_name}"
            result3 = subprocess.run(["ossutil", "ls", json_path], capture_output=True, text=True)
            if result3.returncode != 0:
                continue
            
            # 计算动作的后缀
            act_size, act_duration, act_records = get_json_stats_oss([json_path])
            act_suffix = format_suffix(act_size, act_duration, act_records)
            new_action = remove_suffix(action) + act_suffix
            
            # 构建完整的新路径（包含所有层级的后缀）
            new_action_path = f"{new_scene_path}/{new_sub_scene}/{new_action}"
            
            # 一次性重命名整个路径
            if action_path != new_action_path:
                print(f"[INFO] 重命名: {action_path} -> {new_action_path}")
                oss_mv(action_path, new_action_path)

def get_json_stats_oss(json_paths):
    total_size = 0
    total_duration = 0
    total_records = 0
    for jp in json_paths:
        # 使用 ossutil cat 直接读取 JSON 内容
        cat_cmd = ["ossutil", "cat", jp]
        result = subprocess.run(cat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] 无法读取 JSON: {jp}")
            continue
        try:
            # 过滤掉最后的时间统计行
            lines = result.stdout.strip().split('\n')
            json_lines = []
            for line in lines:
                if 'elapsed' in line and line.strip().endswith('elapsed'):
                    break  # 跳过时间统计行
                json_lines.append(line)
            json_content = '\n'.join(json_lines)
            
            data = json.loads(json_content)
            total_size += data.get('total_size', 0)
            total_duration += data.get('total_duration', 0)
            total_records += data.get('record_count', 0)
        except Exception as e:
            print(f"[ERROR] JSON 解析失败 {jp}: {e}")
            continue
    print(f"[统计] 汇总: total_size={total_size}, total_duration={total_duration}, total_records={total_records}")
    return total_size, total_duration, total_records

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OSS场景批量重命名工具")
    parser.add_argument("--root_path", type=str, required=True, help="OSS根路径，如 oss://leju-delivery-test/")
    parser.add_argument("--scene_name", type=str, required=True, help="主场景名称，如 hotel_services")
    args = parser.parse_args()

    print(f"Root path: {args.root_path}, Scene name: {args.scene_name}")
    adjust_scene_names_oss(args.root_path, args.scene_name)