#!/usr/bin/env python3
"""
OSS Metadata Merge Script

Usage:
    python merge_oss_metadata.py

Required environment variables:
    SCENE_NAME        - Scene name to process (required)
    ACCESS_KEY_ID     - OSS access key ID
    ACCESS_KEY_SECRET - OSS access key secret  
    ENDPOINT          - OSS endpoint
    OSS_BUCKET        - OSS bucket path

This script processes metadata.json files for a specific scene and generates:
- task_info/{scene}-{sub_scene}-{task}.json: Full metadata records
- task_stats/{scene}-{sub_scene}-{task}.json: Summary statistics
"""

import os
import sys
import json
import subprocess
import tempfile
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OSSMetadataMerger:
    def __init__(self):
        self.scene_name = os.environ.get('SCENE_NAME')
        if not self.scene_name:
            logger.error("SCENE_NAME environment variable is required")
            sys.exit(1)

        # Check other required environment variables
        required_vars = ['ACCESS_KEY_ID', 'ACCESS_KEY_SECRET', 'ENDPOINT', 'OSS_BUCKET']
        for var in required_vars:
            if not os.environ.get(var):
                logger.error(f"Missing required environment variable: {var}")
                sys.exit(1)

        self.oss_bucket = os.environ['OSS_BUCKET'].rstrip('/')
        self.temp_dir = tempfile.mkdtemp(prefix='oss_metadata_')

        # Generate ossutil config
        self._generate_ossutil_config()

    def _generate_ossutil_config(self):
        """Generate ossutil configuration file"""
        config_path = os.path.expanduser('~/.ossutilconfig')
        with open(config_path, 'w') as f:
            f.write(f"""[default]
accessKeyId={os.environ['ACCESS_KEY_ID']}
accessKeySecret={os.environ['ACCESS_KEY_SECRET']}
region=cn-hangzhou
endpoint={os.environ['ENDPOINT']}
""")
        logger.info("ossutil config generated")

    def _run_ossutil(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run ossutil command"""
        full_cmd = ['ossutil'] + cmd
        try:
            result = subprocess.run(full_cmd, capture_output=True, text=True, check=False)
            if check and result.returncode != 0:
                logger.error(f"ossutil command failed: {' '.join(full_cmd)}")
                if result.stderr:
                    logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"ossutil command failed with code {result.returncode}")
            return result
        except FileNotFoundError:
            logger.error("ossutil command not found. Please ensure ossutil is installed and in PATH")
            sys.exit(1)

    def remove_old_files(self):
        """Remove old summary files for the scene"""
        logger.info("========== Removing old summary files ==========")
        logger.info(f"Removing old summaries for scene: {self.scene_name}")

        # Remove from task_info
        prefix = f"{self.oss_bucket}/task_info/{self.scene_name}-"
        result = self._run_ossutil(['ls', f"{self.oss_bucket}/task_info/"], check=False)
        if result.returncode == 0:
            files_to_remove = [line.strip() for line in result.stdout.splitlines() 
                             if line.strip().startswith(prefix)]
            for file in files_to_remove:
                self._run_ossutil(['rm', file, '-f'], check=False)
            if files_to_remove:
                logger.info(f"Removed {len(files_to_remove)} old summary files")
            else:
                logger.info("No existing summary files found")

        # Remove from task_stats  
        prefix = f"{self.oss_bucket}/task_stats/{self.scene_name}-"
        result = self._run_ossutil(['ls', f"{self.oss_bucket}/task_stats/"], check=False)
        if result.returncode == 0:
            files_to_remove = [line.strip() for line in result.stdout.splitlines()
                             if line.strip().startswith(prefix)]
            for file in files_to_remove:
                self._run_ossutil(['rm', file, '-f'], check=False)
            if files_to_remove:
                logger.info(f"Removed {len(files_to_remove)} old statistics files")
            else:
                logger.info("No existing statistics files found")

    def scan_metadata_files(self) -> List[str]:
        """Scan for metadata.json files in the scene"""
        logger.info("========== Scanning metadata.json files ==========")
        logger.info(f"Scanning OSS bucket for scene: {self.scene_name}")

        result = self._run_ossutil(['ls', '-r', f"{self.oss_bucket}/{self.scene_name}/"], check=False)
        if result.returncode != 0:
            logger.error(f"Scene '{self.scene_name}' not found or contains no files")
            sys.exit(1)

        metadata_files = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if 'metadata.json' in line and 'oss://' in line:
                oss_index = line.find('oss://')
                if oss_index != -1:
                    oss_path = line[oss_index:].strip()
                    if oss_path.endswith('metadata.json'):
                        metadata_files.append(oss_path)

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

        for oss_path in metadata_files:
            # Extract components from path
            # Format: oss://bucket/scene_name/sub_scene_name/english_task_name/uuid/metadata.json
            path_without_bucket = oss_path.replace(f"{self.oss_bucket}/", "")
            components = path_without_bucket.split('/')

            if len(components) != 5 or components[4] != 'metadata.json':
                logger.warning(f"Skipping invalid path structure: {oss_path}")
                continue

            scene_name = components[0]
            sub_scene_name = components[1] 
            english_task_name = components[2]
            unit_key = f"{scene_name}-{sub_scene_name}-{english_task_name}"

            unit_files[unit_key].append(oss_path)
            processed += 1

            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(metadata_files)} files...")

        logger.info(f"Grouped into {len(unit_files)} units")
        return dict(unit_files)

    def process_unit(self, unit_key: str, metadata_paths: List[str]) -> Tuple[List[dict], dict]:
        """Process metadata files for a single unit"""
        logger.info(f"Processing unit: {unit_key}")

        records = []
        total_size = 0.0
        total_duration = 0.0

        for metadata_path in metadata_paths:
            logger.debug(f"  Downloading: {metadata_path}")

            # Download metadata file
            temp_file = os.path.join(self.temp_dir, f"metadata_{len(records)}.json")
            result = self._run_ossutil(['cp', metadata_path, temp_file, '-f'], check=False)

            if result.returncode == 0 and os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r') as f:
                        metadata = json.load(f)

                    # Extract file size and duration
                    file_size = float(metadata.get('file_size', 0))
                    file_duration = float(metadata.get('file_duration', 0))

                    total_size += file_size
                    total_duration += file_duration

                    records.append(metadata)
                    logger.debug(f"    Successfully processed (size: {file_size}GB, duration: {file_duration}s)")

                except Exception as e:
                    logger.warning(f"    Failed to process {metadata_path}: {e}")
            else:
                logger.warning(f"    Failed to download: {metadata_path}")

        statistics = {
            'record_count': len(records),
            'total_size': round(total_size, 2),
            'total_duration': round(total_duration, 2),
        }

        logger.info(f"  Records: {len(records)}, Total size: {total_size:.2f}GB, Total duration: {total_duration:.2f}s")
        return records, statistics

    def merge_metadata(self):
        """Main merge process"""
        try:
            logger.info("========== OSS Metadata Merge Script ==========")
            logger.info(f"Scene filter: {self.scene_name}")
            logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Step 1: Remove old files
            self.remove_old_files()

            # Step 2: Scan metadata files
            metadata_files = self.scan_metadata_files()

            # Step 3: Group files by unit
            unit_files = self.group_files_by_unit(metadata_files)

            # Step 4: Process each unit
            logger.info("========== Creating unit summaries ==========")
            unit_count = 0

            for unit_key, metadata_paths in unit_files.items():
                unit_count += 1
                logger.info(f"\n[{unit_count}/{len(unit_files)}] Processing unit: {unit_key}")

                # Process metadata for this unit
                records, statistics = self.process_unit(unit_key, metadata_paths)

                # Upload full metadata array
                summary_path = os.path.join(self.temp_dir, f"{unit_key}_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(records, f, indent=2)

                summary_oss_path = f"{self.oss_bucket}/task_info/{unit_key}.json"
                self._run_ossutil(['cp', summary_path, summary_oss_path, '-f'])
                logger.info(f"  Summary uploaded: {summary_oss_path}")

                # Upload statistics
                stats_path = os.path.join(self.temp_dir, f"{unit_key}_stats.json")
                with open(stats_path, 'w') as f:
                    json.dump(statistics, f, indent=2)

                stats_oss_path = f"{self.oss_bucket}/task_stats/{unit_key}.json"
                self._run_ossutil(['cp', stats_path, stats_oss_path, '-f'])
                logger.info(f"  Statistics uploaded: {stats_oss_path}")

            # Final summary
            logger.info("\n========== Summary ==========")
            logger.info(f"Processed {len(unit_files)} units")
            logger.info(f"Full metadata saved to: {self.oss_bucket}/task_info/")
            logger.info(f"Statistics saved to: {self.oss_bucket}/task_stats/")
            logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        finally:
            # Always cleanup
            self._cleanup()

    def _cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")


def main():
    merger = OSSMetadataMerger()
    merger.merge_metadata()




if __name__ == '__main__':
    main()
