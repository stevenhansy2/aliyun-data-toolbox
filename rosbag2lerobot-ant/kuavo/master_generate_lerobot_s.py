"""CLI entrypoint for Kuavo ROSbag -> LeRobot conversion."""

import argparse
import copy
import fnmatch
import json
import logging
import os
import time

import converter.runtime.lerobot_compat_patch  # noqa: F401
from converter.configs import load_config_from_json
from converter.data.bag_discovery import discover_bag_tasks_auto
from converter.pipeline.conversion_orchestrator import port_kuavo_rosbag

logger = logging.getLogger(__name__)


def _merge_robot_profile_to_config(config, robot_profile: dict):
    if not robot_profile:
        return
    if "model" in robot_profile:
        config.robot_model = robot_profile["model"]
    if "urdf_path" in robot_profile:
        config.urdf_path = robot_profile["urdf_path"]
    if "topics" in robot_profile and isinstance(robot_profile["topics"], dict):
        cur = dict(config.source_topics or {})
        cur.update(robot_profile["topics"])
        config.source_topics = cur
    if "cameras" in robot_profile and isinstance(robot_profile["cameras"], dict):
        cur = dict(config.camera_topic_specs or {})
        cur.update(robot_profile["cameras"])
        config.camera_topic_specs = cur


def _apply_override_fields(config, override: dict):
    if not override:
        return
    _merge_robot_profile_to_config(config, override.get("robot_profile", {}))
    for key in [
        "robot_model",
        "urdf_path",
        "which_arm",
        "eef_type",
        "dex_dof_needed",
        "main_timeline",
        "main_timeline_fps",
        "main_timeline_key",
        "only_arm",
    ]:
        if key in override:
            setattr(config, key, override[key])
    if "source_topics" in override and isinstance(override["source_topics"], dict):
        cur = dict(config.source_topics or {})
        cur.update(override["source_topics"])
        config.source_topics = cur
    if (
        "camera_topic_specs" in override
        and isinstance(override["camera_topic_specs"], dict)
    ):
        cur = dict(config.camera_topic_specs or {})
        cur.update(override["camera_topic_specs"])
        config.camera_topic_specs = cur
    if "topics" in override and isinstance(override["topics"], list):
        config.topics = override["topics"]


def _extract_model_from_metadata(metadata_json_path: str | None) -> str | None:
    if not metadata_json_path or not os.path.exists(metadata_json_path):
        return None
    try:
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.debug("解析 metadata 失败: %s (%s)", metadata_json_path, exc)
        return None
    for k in ["robot_model", "robotModel", "deviceModel", "device_model", "model"]:
        v = data.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _build_config_for_bag(base_config, bag_task: dict):
    cfg = copy.deepcopy(base_config)
    bag_path = bag_task["local_path"]
    bag_name = bag_task.get("bag_name", os.path.basename(bag_path))
    rel_key = bag_name

    # 1) 先按 metadata 中机型命中 model_profiles
    model_id = _extract_model_from_metadata(bag_task.get("metadata_json_path"))
    if model_id and getattr(base_config, "model_profiles", None):
        prof = base_config.model_profiles.get(model_id)
        if isinstance(prof, dict):
            logger.info("[CONFIG] 命中 model_profiles: %s", model_id)
            _apply_override_fields(cfg, prof)

    # 2) 再按 bag_overrides（文件名/路径匹配）覆盖
    for rule in getattr(base_config, "bag_overrides", []) or []:
        patt = rule.get("match")
        if not patt:
            continue
        if fnmatch.fnmatch(bag_name, patt) or fnmatch.fnmatch(rel_key, patt):
            logger.info("[CONFIG] 命中 bag_overrides: match=%s", patt)
            _apply_override_fields(cfg, rule)
    return cfg


def main():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    start = time.time()

    parser = argparse.ArgumentParser(description="Kuavo ROSbag to Lerobot Converter")
    parser.add_argument("--bag_dir", default="/home/leju_kuavo/tmp/123/", type=str, help="Path to ROS bag")
    parser.add_argument("--moment_json_dir", type=str, required=False, help="Path to moment.json")
    parser.add_argument("--metadata_json_dir", type=str, required=False, help="Path to metadata.json")
    parser.add_argument("--output_dir", default="testoutput/", type=str, required=False, help="Path to output")
    parser.add_argument("--train_frequency", type=int, help="Training frequency (Hz), overrides config")
    parser.add_argument("--only_arm", type=str, choices=["true", "false"], help="Use only arm data")
    parser.add_argument("--which_arm", type=str, choices=["left", "right", "both"], help="Which arm to use")
    parser.add_argument("--dex_dof_needed", type=int, help="Degrees of freedom for dex hand")
    parser.add_argument("--config", type=str, default="./kuavo/configs/request.json", help="Path to config JSON file")
    parser.add_argument(
        "--use_depth",
        action="store_true",
        help="如果指定，忽略所有与 metadata.json / moments.json 相关的输入与输出（不读取也不写入）",
    )
    args = parser.parse_args()

    config = load_config_from_json(args.config)
    if args.train_frequency is not None:
        config.train_hz = args.train_frequency
        logger.info("覆盖配置: train_hz = %s", args.train_frequency)
    if args.only_arm is not None:
        config.only_arm = args.only_arm.lower() == "true"
        logger.info("覆盖配置: only_arm = %s", config.only_arm)
    if args.which_arm is not None:
        config.which_arm = args.which_arm
        logger.info("覆盖配置: which_arm = %s", args.which_arm)
    if args.dex_dof_needed is not None:
        config.dex_dof_needed = args.dex_dof_needed
        logger.info("覆盖配置: dex_dof_needed = %s", args.dex_dof_needed)

    bag_dir = args.bag_dir
    logger.info("Bag directory: %s", bag_dir)

    output_dir = args.output_dir

    bag_tasks = discover_bag_tasks_auto(
        bag_dir,
        metadata_json_path=args.metadata_json_dir,
        moment_json_path=args.moment_json_dir,
    )
    if not bag_tasks:
        raise RuntimeError(f"未发现可转换的 .bag 文件: {bag_dir}")
    logger.info("发现 %s 个 bag 任务", len(bag_tasks))

    for idx, task in enumerate(bag_tasks, 1):
        bag_path = task["local_path"]
        dataset_name = task["dataset_name"]
        logger.info("(%s/%s) 开始转换: %s", idx, len(bag_tasks), bag_path)
        logger.info("输出目录名: %s", dataset_name)
        cfg = _build_config_for_bag(config, task)
        port_kuavo_rosbag(
            raw_config=cfg,
            processed_files=[
                {
                    "local_path": bag_path,
                    "start": task.get("start", 0),
                    "end": task.get("end", 1),
                }
            ],
            moment_json_DIR=task.get("moment_json_path"),
            metadata_json_DIR=task.get("metadata_json_path"),
            lerobot_dir=output_dir,
            use_depth=args.use_depth,
            dataset_name=dataset_name,
        )

    end = time.time()
    logger.info("总用时: %.2f 秒", end - start)


if __name__ == "__main__":
    main()
