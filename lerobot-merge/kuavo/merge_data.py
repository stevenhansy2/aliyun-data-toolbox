import os
import shutil
import json
from util import *
from pathlib import Path
import subprocess
from datetime import datetime
from uuid import uuid4

PARAMETERS = "parameters"
META = "meta"
MASK = "mask/chunk-000"


def deepest_dirs(root, absolute=False):
    root = os.path.abspath(root)
    deepest = []
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        if not dirnames:
            if absolute:
                deepest.append(dirpath)
            else:
                rel = os.path.relpath(dirpath, root)
                deepest.append(rel)
    return [
        dirname for dirname in deepest
        if dirname not in [PARAMETERS, META] and MASK + "/episode" not in dirname
    ]


class LeRobotMetadata:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root)
        self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self.episodes_stats = load_episodes_stats(self.root)

    def get_info(self):
        return self.info

    def get_tasks(self):
        return self.tasks

    def get_episodes(self):
        return self.episodes

    def get_episodes_stats(self):
        return self.episodes_stats


def _human_readable_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {units[i]}"


def validate_episode_structure(root: Path) -> None:
    """
    验证 root 目录是否符合标准 LeRobot episode 结构。
    若不符合，抛出 ValueError。
    """
    # 1. 必须存在 meta/
    meta_dir = root / "meta"
    if not meta_dir.exists() or not meta_dir.is_dir():
        raise ValueError("缺失 meta/ 目录")


    required_meta_files = ["info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]
    missing_meta = [f for f in required_meta_files if not (meta_dir / f).exists()]
    if missing_meta:
        raise ValueError(f"meta/ 目录缺失必要文件: {missing_meta}")

    # 检查 jsonl 文件是否为空
    jsonl_files = ["episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]
    empty_jsonl = []
    for jf in jsonl_files:
        file_path = meta_dir / jf
        if file_path.exists():
            try:
                if file_path.stat().st_size == 0:
                    empty_jsonl.append(jf)
                else:
                    # 进一步检查是否内容全为空行
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines()]
                        if not any(lines):
                            empty_jsonl.append(jf)
            except Exception as e:
                raise ValueError(f"检查 {jf} 是否为空时出错: {e}")
    if empty_jsonl:
        raise ValueError(f"meta/ 目录下存在空的 jsonl 文件: {empty_jsonl}")

    # 2. 必须存在 data/chunk-000/ 且包含至少一个 episode_*.parquet
    data_chunk = root / "data" / "chunk-000"
    if not data_chunk.exists() or not data_chunk.is_dir():
        raise ValueError("缺失 data/chunk-000/ 目录")
    parquet_files = list(data_chunk.glob("episode_*.parquet"))
    if not parquet_files:
        raise ValueError("data/chunk-000/ 中无 episode_*.parquet 文件")

    # 3. 必须存在 videos/chunk-000/observation.images.camera_*/*.mp4
    videos_base = root / "videos" / "chunk-000"
    expected_cameras = [
        "observation.images.camera_top",
        "observation.images.camera_wrist_left",
        "observation.images.camera_wrist_right"
    ]
    for cam in expected_cameras:
        cam_dir = videos_base / cam
        if not cam_dir.exists() or not cam_dir.is_dir():
            raise ValueError(f"缺失视频目录: {cam_dir}")
        mp4_files = list(cam_dir.glob("episode_*.mp4"))
        if not mp4_files:
            raise ValueError(f"视频目录 {cam_dir} 中无 episode_*.mp4 文件")

    # 4. parameters/ 必须包含所有相机参数文件
    param_dir = root / "parameters"
    if not param_dir.exists() or not param_dir.is_dir():
        raise ValueError("缺失 parameters/ 目录")
    required_param_files = [
        "camera_top_extrinsic.json",
        "camera_top_intrinsic.json",
        "camera_wrist_left_extrinsic.json",
        "camera_wrist_left_intrinsic.json",
        "camera_wrist_right_extrinsic.json",
        "camera_wrist_right_intrinsic.json"
    ]
    missing_params = [f for f in required_param_files if not (param_dir / f).exists()]
    if missing_params:
        raise ValueError(f"parameters/ 缺失必要文件: {missing_params}")

    #print(f"[validate] ✅ 结构验证通过: {root}")


def merge_meta(src_path, tgt_path, generate_summary: bool = False):
    src_path = Path(src_path)
    if not src_path.exists():
        raise ValueError(f"源目录不存在: {src_path}")

    # 获取所有子目录（仅目录）
    srcs = [p for p in src_path.iterdir() if p.is_dir()]
    #print(f"[merge_meta] 遍历源目录: {src_path}, 发现 {len(srcs)} 个子目录")

    new_info = None
    total_size_bytes = 0
    success_count = 0
    skipped_dirs = []
    merged_metadata = {}  # 用于合并 metadata.json

    tgt_path = Path(tgt_path)

    # 按名称排序，确保顺序一致
    for local_path in sorted(srcs):
        try:
            #print(f"[merge_meta] 处理子目录: {local_path}")

            # === 【增强】结构完整性检查 ===
            validate_episode_structure(local_path)

            # === 【新增】跳过总帧数 < 30 的数据集 ===
            info_json_path = local_path / "meta" / "info.json"
            with open(info_json_path, 'r', encoding='utf-8') as f:
                tmp_info_early = json.load(f)

            if tmp_info_early.get("total_frames", 0) < 95:
                #print(f"[merge_meta] 跳过 {local_path}：总帧数 {tmp_info_early['total_frames']} < 95")
                skipped_dirs.append((str(local_path), "total_frames < 95"))
                continue

            # 计算大小
            size_bytes = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
            total_size_bytes += size_bytes

            # 加载元数据（此时可安全调用）
            metas = LeRobotMetadata(local_path)
            tmp_info = metas.get_info()
            tmp_episodes = metas.get_episodes()
            tmp_episodes_stats = metas.get_episodes_stats()

            #print(f"[merge_meta] 读取 meta 成功: {local_path}")

            if success_count == 0:
                # 初始化 new_info（仅第一个成功子集）
                new_info = tmp_info.copy()
                new_tasks = metas.get_tasks()
                #print(f"[merge_meta] 初始化元数据，写入 idx=0 到 {tgt_path}")
                append_ep_idx(tmp_episodes, 0, tgt_path)
                append_ep_sts_idx(tmp_episodes_stats, 0, tgt_path)
                append_tasks_idx(new_tasks, tgt_path)

                # 检查并合并 metadata.json
                metadata_path = local_path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        # 更新 episode_id 为数字索引
                        metadata['episode_id'] = 0
                        merged_metadata["0"] = metadata
                    except Exception as e:
                        print(f"[WARNING] 读取 metadata.json 失败 ({local_path}): {e}")

                dirs = deepest_dirs(local_path)
                #print(f"[merge_meta] 最深层数据目录: {dirs}")
                if (local_path / PARAMETERS).exists():
                    #print(f"[merge_meta] 拷贝 PARAMETERS: {local_path / PARAMETERS} -> {tgt_path / PARAMETERS}")
                    mv_data(local_path / PARAMETERS, tgt_path / PARAMETERS)
                if (local_path / MASK).exists():
                    #print(f"[merge_meta] 拷贝 MASK: {local_path / MASK} -> {tgt_path / MASK}")
                    mv_data_dir_idx(local_path / MASK, tgt_path / MASK, 0)
                for path in dirs:
                    #print(f"[merge_meta] 拷贝数据目录: {local_path / path} -> {tgt_path / path}")
                    mv_data_idx(local_path / path, tgt_path / path, 0)
            else:
                assert new_info is not None, "new_info 未初始化"
                ep_idx_delta = new_info['total_episodes']
                #print(f"[merge_meta] 合并第 {success_count} 个子集，ep_idx_delta={ep_idx_delta}")
                append_ep_idx(tmp_episodes, ep_idx_delta, tgt_path)
                append_ep_sts_idx(tmp_episodes_stats, ep_idx_delta, tgt_path)

                # 检查并合并 metadata.json
                metadata_path = local_path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        # 更新 episode_id 为数字索引
                        metadata['episode_id'] = ep_idx_delta
                        merged_metadata[str(ep_idx_delta)] = metadata
                    except Exception as e:
                        print(f"[WARNING] 读取 metadata.json 失败 ({local_path}): {e}")

                # 累加统计信息
                new_info['total_episodes'] += tmp_info['total_episodes']
                new_info['total_frames'] += tmp_info['total_frames']
                new_info['total_videos'] += tmp_info['total_videos']

                if (local_path / MASK).exists():
                    #print(f"[merge_meta] 拷贝 MASK: {local_path / MASK} -> {tgt_path / MASK}")
                    mv_data_dir_idx(local_path / MASK, tgt_path / MASK, success_count)
                dirs = deepest_dirs(local_path)
                for path in dirs:
                    #print(f"[merge_meta] 拷贝数据目录: {local_path / path} -> {tgt_path / path}")
                    mv_data_idx(local_path / path, tgt_path / path, ep_idx_delta)

            success_count += 1

        except Exception as e:
            error_msg = str(e)
            #print(f"[ERROR] 跳过子目录 {local_path}: {error_msg}")
            skipped_dirs.append((str(local_path), error_msg))
            continue

    # === 校验是否至少有一个成功 ===
    if new_info is None:
        raise RuntimeError("❌ 所有子目录均合并失败！无法生成有效 LeRobot 数据集。")

    # === 自动计算 total_chunks 和 chunks_size ===
    import math
    chunk_size = 1000
    total_episodes = new_info.get("total_episodes", 0)
    total_chunks = math.ceil(total_episodes / chunk_size) if total_episodes > 0 else 1
    new_info["chunks_size"] = chunk_size
    new_info["total_chunks"] = total_chunks

    # 写入最终 info.json
    write_info(new_info, tgt_path)

    # 写入合并后的 metadata.json（如果存在）
    if merged_metadata:
        metadata_output_path = tgt_path / "metadata.json"
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_metadata, f, indent=4, ensure_ascii=False)
        #print(f"[merge_meta] ✅ 已合并并写入 metadata.json: {metadata_output_path}")

    # === 计算总时长 ===
    if 'fps' in new_info and new_info['fps'] is not None and new_info['fps'] > 0:
        total_duration_sec_val = new_info['total_frames'] / new_info['fps']
        total_duration_sec_str = f"{total_duration_sec_val:.3f} sec"
        total_duration_hour_str = f"{total_duration_sec_val / 3600:.3f} hour"
    else:
        total_duration_sec_str = None
        total_duration_hour_str = None
        #print(f"[警告] 合并后的 info.json 缺少有效 'fps'（当前值: {new_info.get('fps')}），无法计算总时长")

    # === 构建最终 summary ===
    total_size_mb = total_size_bytes / (1024 ** 2)
    total_size_tb = total_size_bytes / (1024 ** 4)

    summary = {
        "input_subdirs_total": len(srcs),
        "successful_merges": success_count,
        "skipped_subdirs_count": len(skipped_dirs),
        # "skipped_details": skipped_dirs,
        "skipped_details": str(src_path),
        "final_total_episodes": new_info["total_episodes"],
        "final_total_frames": new_info["total_frames"],
        "final_total_videos": new_info["total_videos"],
        "total_duration_sec": total_duration_sec_str,
        "total_duration_hour": total_duration_hour_str,
        "total_size_bytes": total_size_bytes,
        "total_size_mb": round(total_size_mb, 2),
        "total_size_tb": round(total_size_tb, 4),
    }

    if generate_summary:
        summary_path = tgt_path / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        #print(f"\n[merge_meta] ✅ 已生成数据集摘要: {summary_path}")
    #print(f"[merge_meta] 📊 成功合并 {success_count} / {len(srcs)} 个子目录")
    if skipped_dirs:
        #print("[merge_meta] ⚠️ 跳过的目录（前5个）:")
        for path, reason in skipped_dirs[:5]:
            print(f"  - {path}: {reason}")
        if len(skipped_dirs) > 5:
            print(f"  ... 还有 {len(skipped_dirs) - 5} 个未列出")

    #print(f"[merge_meta] 🔚 合并完成。")


def parallel_tar_compress(src_dir, out_path):
    src_dir = Path(src_dir)
    out_path = Path(out_path)
    items = [f.name for f in src_dir.iterdir() if f.name != out_path.name]
    if not items:
        raise RuntimeError("没有可压缩的内容")

    cmd = ["tar", "-cf", str(out_path), "-C", str(src_dir)] + items
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"tar 打包失败: {result.stderr}")


def get_tar(src_dir, out_path, max_tar=2):
    for tar_attempt in range(1, max_tar + 1):
        try:
            parallel_tar_compress(src_dir, out_path)
            #print(f"📦 数据集已打包至: {out_path}")
            break
        except Exception as e:
            #print(f"❌ 第{tar_attempt}次打包失败: {e}")
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                    print(f"⚠️ 已删除残余压缩包: {out_path}")
                except Exception as del_e:
                    print(f"⚠️ 删除残余压缩包失败: {del_e}")
            if tar_attempt == max_tar:
                print(f"💥 打包最终失败: {e}")


def reorganize_chunks(tgt_path):
    """
    只将超过1000的部分移动到下一个chunk文件夹，chunk-000保留前1000条，chunk-001放1001~2000条，以此类推。
    文件命名为 episode_000000, episode_000001, ...，编号全局递增。
    videos 同步移动，chunk-000及其子目录保留。
    """
    tgt_path = Path(tgt_path)
    data_dir = tgt_path / "data"
    chunk0_dir = data_dir / "chunk-000"
    if not chunk0_dir.exists():
        print(f"❌ 未找到 {chunk0_dir}")
        return

    # 1. 统计所有 episode_*.parquet
    parquet_files = sorted(chunk0_dir.glob("episode_*.parquet"))
    total = len(parquet_files)
    chunk_size = 1000
    num_chunks = (total + chunk_size - 1) // chunk_size

    # 2. 创建 chunk-xxx 文件夹（除chunk-000外）
    for i in range(1, num_chunks):
        chunk_name = f"chunk-{i:03d}"
        chunk_dir = data_dir / chunk_name
        chunk_dir.mkdir(parents=True, exist_ok=True)

    # 3. 只移动 chunk-000 目录下超过1000的 parquet 文件到下一个chunk
    for idx, file in enumerate(parquet_files):
        if idx < chunk_size:
            # 前1000条留在chunk-000
            new_name = f"episode_{idx:06d}.parquet"
            target = chunk0_dir / new_name
            if file.name != new_name:
                shutil.move(str(file), target)
        else:
            chunk_idx = idx // chunk_size
            chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
            new_name = f"episode_{idx:06d}.parquet"
            target = chunk_dir / new_name
            shutil.move(str(file), target)

    # 4. videos 只移动 chunk-000/observation.images.camera_* 下超过1000的 mp4 到下一个chunk
    videos_dir = tgt_path / "videos"
    cam_base_dir = videos_dir / "chunk-000"
    for cam_dir in cam_base_dir.glob("observation.images.camera_*"):
        if not cam_dir.is_dir():
            continue
        mp4_files = sorted(cam_dir.glob("episode_*.mp4"))
        for i in range(1, num_chunks):
            chunk_cam_dir = videos_dir / f"chunk-{i:03d}" / cam_dir.name
            chunk_cam_dir.mkdir(parents=True, exist_ok=True)
        for idx, mp4 in enumerate(mp4_files):
            if idx < chunk_size:
                # 前1000条留在chunk-000
                new_name = f"episode_{idx:06d}.mp4"
                target = cam_dir / new_name
                if mp4.name != new_name:
                    shutil.move(str(mp4), target)
            else:
                chunk_idx = idx // chunk_size
                chunk_cam_dir = videos_dir / f"chunk-{chunk_idx:03d}" / cam_dir.name
                new_name = f"episode_{idx:06d}.mp4"
                target = chunk_cam_dir / new_name
                shutil.move(str(mp4), target)

    # 5. 不清理 chunk-000 及其 observation.images.camera_* 目录，只移动多余部分

    # 6. 更新 info.json 的 total_chunks
    info_path = tgt_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        info["total_chunks"] = num_chunks
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"✅ 已更新 info.json total_chunks={num_chunks}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重组 testoutput 目录下的 LeRobot 数据集")
    parser.add_argument("--src_dir", type=str, required=True, help="待重组的源父目录（包含多个子数据集）")
    parser.add_argument("--tgt_dir", type=str, required=True, help="合并后的目标目录")
    parser.add_argument("--summary", action="store_true",
                        help="是否生成 dataset_summary.json（输出到 tgt_dir）")
    parser.add_argument("--max_tar", default=2, type=int, help="最大重试打包次数（当前未使用）")
    args = parser.parse_args()

    merge_meta(args.src_dir, args.tgt_dir, generate_summary=args.summary)

    # 新增：自动分块
    reorganize_chunks(args.tgt_dir)


if __name__ == '__main__':
    main()
