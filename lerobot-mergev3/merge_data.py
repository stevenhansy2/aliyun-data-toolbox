import os
import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

def find_lerobot_dirs(root_dir):
    lerobot_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "lerobot" in dirnames:
            lerobot_dirs.append(os.path.join(dirpath, "lerobot"))
    return lerobot_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/home/leju_kuavo/temp",
        type=str,
        required=False,
        help="Path to the root directory containing lerobot datasets",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/leju_kuavo/temp/kuavo-merged-dataset",
        type=str,
        required=False,
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--output_repo_id",
        default="lerobot/kuavo-merged-dataset",
        type=str,
        required=False,
        help="Output repo id for merged dataset",
    )
    args = parser.parse_args()

    lerobot_dirs = find_lerobot_dirs(args.input_dir)
    datasets = [LeRobotDataset(d) for d in lerobot_dirs]
    merged = merge_datasets(
        datasets,
        output_repo_id=args.output_repo_id,
        output_dir=args.output_dir
    )


