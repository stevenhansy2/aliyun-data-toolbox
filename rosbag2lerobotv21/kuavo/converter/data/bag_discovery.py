"""Bag discovery and sidecar resolution helpers."""

import datetime
import os

import rosbag


def _is_valid_bag_file(file_name: str) -> bool:
    return file_name.endswith(".bag") and not file_name.endswith(".c.bag")


def get_bag_time_info(bag_path: str) -> dict:
    """
    获取 rosbag 包的时间信息

    Args:
        bag_path: rosbag 文件路径

    Returns:
        dict: 包含时间信息的字典，包括：
            - unix_timestamp: Unix时间戳（秒）
            - iso_format: ISO格式时间字符串（东八区）
            - nanoseconds: 纳秒格式时间戳
            - duration: bag持续时间（秒）
            - end_time: 结束时间Unix时间戳
    """
    try:
        bag = rosbag.Bag(bag_path, "r")
        bag_start_time = bag.get_start_time()
        bag_end_time = bag.get_end_time()
        bag_duration = bag_end_time - bag_start_time
        bag.close()

        # 转换为带时区的ISO格式（东八区）
        start_datetime = datetime.datetime.fromtimestamp(
            bag_start_time, tz=datetime.timezone(datetime.timedelta(hours=8))
        )
        start_iso = start_datetime.isoformat()

        # 转换为纳秒
        start_nanoseconds = int(bag_start_time * 1e9)

        return {
            "unix_timestamp": bag_start_time,
            "iso_format": start_iso,
            "nanoseconds": start_nanoseconds,
            "duration": bag_duration,
            "end_time": bag_end_time,
        }

    except Exception as e:
        print(f"获取bag时间信息失败: {e}")
        return {
            "unix_timestamp": None,
            "iso_format": None,
            "nanoseconds": None,
            "duration": None,
            "end_time": None,
        }


def list_bag_files_auto(raw_dir):
    bag_files = []
    for i, fname in enumerate(sorted(os.listdir(raw_dir))):
        if _is_valid_bag_file(fname):
            bag_files.append(
                {
                    "link": "",  # 保持为空
                    "start": 0,  # 批量设置为0
                    "end": 1,  # 批量设置为1
                    "local_path": os.path.join(raw_dir, fname),
                }
            )
    return bag_files


def _sanitize_dataset_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "bag"


def _resolve_sidecar_path(
    override_path: str | None, bag_path: str, file_name: str, root_dir: str
) -> str | None:
    if override_path:
        if os.path.isfile(override_path):
            return override_path
        if os.path.isdir(override_path):
            bag_parent = os.path.basename(os.path.dirname(bag_path))
            bag_stem = os.path.splitext(os.path.basename(bag_path))[0]
            candidates = [
                os.path.join(override_path, bag_parent, file_name),
                os.path.join(override_path, bag_stem, file_name),
                os.path.join(override_path, file_name),
            ]
            for p in candidates:
                if os.path.exists(p):
                    return p
        return None

    local = os.path.join(os.path.dirname(bag_path), file_name)
    if os.path.exists(local):
        return local
    root_level = os.path.join(root_dir, file_name)
    if os.path.exists(root_level):
        return root_level
    return None


def discover_bag_tasks_auto(
    raw_dir: str,
    metadata_json_path: str | None = None,
    moment_json_path: str | None = None,
) -> list[dict]:
    """
    扫描输入目录，返回按 bag 维度的任务列表。
    支持：
      1) 父目录下直接多个 .bag
      2) 父目录下多个子目录，每个子目录含一个或多个 .bag
      3) 直接传入单个 .bag 文件
    """
    tasks: list[dict] = []

    bag_paths: list[str] = []
    if os.path.isfile(raw_dir) and _is_valid_bag_file(raw_dir):
        bag_paths = [os.path.abspath(raw_dir)]
        root_dir = os.path.dirname(os.path.abspath(raw_dir))
    else:
        root_dir = os.path.abspath(raw_dir)
        # 顶层 .bag
        for fname in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, fname)
            if os.path.isfile(p) and _is_valid_bag_file(fname):
                bag_paths.append(p)
        # 顶层子目录内 .bag（只下探一层）
        for entry in sorted(os.listdir(root_dir)):
            subdir = os.path.join(root_dir, entry)
            if not os.path.isdir(subdir):
                continue
            for fname in sorted(os.listdir(subdir)):
                p = os.path.join(subdir, fname)
                if os.path.isfile(p) and _is_valid_bag_file(fname):
                    bag_paths.append(p)

    name_count: dict[str, int] = {}
    for bag_path in bag_paths:
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        dataset_name = _sanitize_dataset_name(bag_name)
        idx = name_count.get(dataset_name, 0)
        name_count[dataset_name] = idx + 1
        if idx > 0:
            dataset_name = f"{dataset_name}_{idx+1}"

        tasks.append(
            {
                "local_path": bag_path,
                "start": 0,
                "end": 1,
                "bag_name": bag_name,
                "dataset_name": dataset_name,
                "metadata_json_path": _resolve_sidecar_path(
                    metadata_json_path, bag_path, "metadata.json", root_dir
                ),
                "moment_json_path": _resolve_sidecar_path(
                    moment_json_path, bag_path, "moments.json", root_dir
                ),
            }
        )

    return tasks
