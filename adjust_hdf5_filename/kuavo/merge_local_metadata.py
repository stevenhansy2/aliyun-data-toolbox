#!/usr/bin/env python3
"""
Local Metadata Merge Script

Usage:
    python merge_local_metadata.py <root_path> <scene_name> [options]

Arguments:
    root_path  - Root directory containing scene data
    scene_name - Scene name to process

Options:
    --workers N         Max concurrent file read workers (default: 32)

Examples:
    # Default processing
    python merge_local_metadata.py /data hotel_services
    
    # Custom worker count for systems with many cores
    python merge_local_metadata.py /data hotel_services --workers 64

This script processes metadata.json files for a specific scene and generates:
- task_info/{scene}-{sub_scene}-{task}.json: Full metadata records
- task_stats/{scene}-{sub_scene}-{task}.json: Summary statistics
"""

import os
import sys
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LocalMetadataMerger:
    def __init__(self, root_path: str, scene_name: str):
        self.root_path = os.path.abspath(root_path)
        self.scene_name = scene_name

        self.max_file_read_workers = 32  # Max concurrent threads for reading files

        if not os.path.exists(self.root_path):
            logger.error(f"Root path does not exist: {self.root_path}")
            sys.exit(1)

        # Create output directories
        self.task_info_dir = os.path.join(self.root_path, "task_info")
        self.task_stats_dir = os.path.join(self.root_path, "task_stats")
        os.makedirs(self.task_info_dir, exist_ok=True)
        os.makedirs(self.task_stats_dir, exist_ok=True)

    def remove_old_files(self):
        """Remove old summary files for the scene"""
        logger.info("========== Removing old summary files ==========")
        logger.info(f"Removing old summaries for scene: {self.scene_name}")

        # Remove from task_info
        removed_count = 0
        for filename in os.listdir(self.task_info_dir):
            if filename.startswith(f"{self.scene_name}-") and filename.endswith(
                ".json"
            ):
                os.remove(os.path.join(self.task_info_dir, filename))
                removed_count += 1
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old summary files from task_info")
        else:
            logger.info("No existing summary files found in task_info")

        # Remove from task_stats
        removed_count = 0
        for filename in os.listdir(self.task_stats_dir):
            if filename.startswith(f"{self.scene_name}-") and filename.endswith(
                ".json"
            ):
                os.remove(os.path.join(self.task_stats_dir, filename))
                removed_count += 1
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old statistics files from task_stats")
        else:
            logger.info("No existing statistics files found in task_stats")

    def scan_metadata_files(self) -> List[str]:
        """Scan for metadata.json files in the scene"""
        logger.info("========== Scanning metadata.json files ==========")
        logger.info(f"Scanning directory for scene: {self.scene_name}")

        scene_path = os.path.join(self.root_path, self.scene_name)
        if not os.path.exists(scene_path):
            logger.error(f"Scene directory not found: {scene_path}")
            sys.exit(1)

        metadata_files = []

        for sub_scene in os.listdir(scene_path):
            sub_scene_path = os.path.join(scene_path, sub_scene)
            if not os.path.isdir(sub_scene_path):
                continue

            for task in os.listdir(sub_scene_path):
                task_path = os.path.join(sub_scene_path, task)
                if not os.path.isdir(task_path):
                    continue

                for uuid_folder in os.listdir(task_path):
                    uuid_path = os.path.join(task_path, uuid_folder)
                    if not os.path.isdir(uuid_path):
                        continue

                    metadata_path = os.path.join(uuid_path, "metadata.json")
                    if os.path.isfile(metadata_path):
                        metadata_files.append(metadata_path)

        if not metadata_files:
            logger.error(f"No metadata.json files found for scene: {self.scene_name}")
            sys.exit(1)

        logger.info(f"Found {len(metadata_files)} metadata.json files")
        return metadata_files

    def group_files_by_unit(self, metadata_files: List[str]) -> Dict[str, List[str]]:
        """Group metadata files by unit (scene-sub_scene-task)"""
        logger.info("========== Processing and grouping metadata ==========")

        unit_files = defaultdict(list)
        processed = 0

        for metadata_path in metadata_files:
            # Extract components from path
            # Format: root_path/scene_name/sub_scene_name/english_task_name/uuid/metadata.json
            rel_path = os.path.relpath(metadata_path, self.root_path)
            components = rel_path.split(os.sep)

            if len(components) != 5 or components[4] != "metadata.json":
                logger.warning(f"Skipping invalid path structure: {metadata_path}")
                continue

            scene_name = components[0]
            sub_scene_name = components[1]
            english_task_name = components[2]
            unit_key = f"{scene_name}-{sub_scene_name}-{english_task_name}"

            unit_files[unit_key].append(metadata_path)
            processed += 1

            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(metadata_files)} files...")

        logger.info(f"Grouped into {len(unit_files)} units")
        return dict(unit_files)

    def _process_metadata_batch(self, tasks: List[Tuple[str, str]]) -> Dict[str, dict]:
        """Process a batch of metadata files and return aggregated results per unit

        This implements a pipeline approach where each worker:
        1. Reads files from assigned units sequentially (I/O phase)
        2. Processes JSON and aggregates locally (compute phase)
        3. Returns complete results (no shared state)

        Files are grouped by unit for better cache locality."""
        unit_results = defaultdict(
            lambda: {"records": [], "total_size": 0.0, "total_duration": 0.0}
        )

        for metadata_path, unit_key in tasks:
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                file_size = float(metadata.get("file_size", 0))
                file_duration = float(metadata.get("file_duration", 0))

                unit_results[unit_key]["records"].append(metadata)
                unit_results[unit_key]["total_size"] += file_size
                unit_results[unit_key]["total_duration"] += file_duration

                logger.debug(
                    f"Successfully processed {metadata_path} (size: {file_size}GB, duration: {file_duration}s)"
                )
            except Exception as e:
                logger.warning(f"Failed to process {metadata_path}: {e}")

        return dict(unit_results)

    def merge_metadata(self):
        """Main merge process with lock-free batch processing"""
        logger.info("========== Local Metadata Merge Script ==========")
        logger.info(f"Root path: {self.root_path}")
        logger.info(f"Scene filter: {self.scene_name}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.remove_old_files()
        metadata_files = self.scan_metadata_files()
        unit_files = self.group_files_by_unit(metadata_files)

        logger.info("========== Creating unit summaries ==========")

        total_files = sum(len(paths) for paths in unit_files.values())
        logger.info(f"Processing {total_files} files across {len(unit_files)} units")

        num_workers = min(self.max_file_read_workers, total_files)

        task_batches = []
        unit_keys = list(unit_files.keys())
        units_per_worker = max(1, len(unit_keys) // num_workers)

        for i in range(0, len(unit_keys), units_per_worker):
            batch = []
            for unit_key in unit_keys[i : i + units_per_worker]:
                for path in unit_files[unit_key]:
                    batch.append((path, unit_key))
            if batch:
                task_batches.append(batch)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._process_metadata_batch, batch)
                for batch in task_batches
            ]

            all_results = []
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                all_results.append(result)
                logger.info(f"  Completed batch {i+1}/{len(task_batches)}")

        unit_data = defaultdict(
            lambda: {"records": [], "total_size": 0.0, "total_duration": 0.0}
        )

        for worker_results in all_results:
            for unit_key, data in worker_results.items():
                unit_data[unit_key]["records"].extend(data["records"])
                unit_data[unit_key]["total_size"] += data["total_size"]
                unit_data[unit_key]["total_duration"] += data["total_duration"]

        for idx, (unit_key, data) in enumerate(unit_data.items(), 1):
            logger.info(f"\n[{idx}/{len(unit_data)}] Saving unit: {unit_key}")

            summary_path = os.path.join(self.task_info_dir, f"{unit_key}.json")
            with open(summary_path, "w",encoding="utf-8") as f:
                json.dump(data["records"], f, indent=2, ensure_ascii=False)
            logger.info(f"  Summary saved: {summary_path}")

            statistics = {
                "record_count": len(data["records"]),
                "total_size": round(data["total_size"], 2),
                "total_duration": round(data["total_duration"], 2),
            }
            stats_path = os.path.join(self.task_stats_dir, f"{unit_key}.json")
            with open(stats_path, "w",encoding="utf-8") as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            logger.info(f"  Statistics saved: {stats_path}")

        logger.info("\n========== Summary ==========")
        logger.info(f"Processed {len(unit_files)} units")
        logger.info(f"Full metadata saved to: {self.task_info_dir}")
        logger.info(f"Statistics saved to: {self.task_stats_dir}")
        logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python merge_local_metadata.py <root_path> <scene_name> [options]"
        )
        print("Options:")
        print("  --workers N         Max concurrent file read workers (default: 32)")
        sys.exit(1)

    root_path = sys.argv[1]
    scene_name = sys.argv[2]

    merger = LocalMetadataMerger(root_path, scene_name)

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--workers" and i + 1 < len(sys.argv):
            merger.max_file_read_workers = int(sys.argv[i + 1])

    merger.merge_metadata()

    # Optionally adjust file names after merging
    from adjust_file_name_local import adjust_scene_names

    print(f"========== Adjusting local scene names ==========")
    print(f"Root path: {root_path}, Scene name: {scene_name}")
    adjust_scene_names(root_path, scene_name)


if __name__ == "__main__":
    main()
