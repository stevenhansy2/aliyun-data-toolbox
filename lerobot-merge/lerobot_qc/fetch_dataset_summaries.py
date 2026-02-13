#!/usr/bin/env python3
import os
import json
import csv
import argparse

def find_lerobot_dirs(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == 'lerobot':
            yield dirpath

def process(root, output):
    rows = []
    for lerobot_dir in find_lerobot_dirs(root):
        # print(lerobot_dir)
        summary_path = os.path.join(lerobot_dir, '../report', 'dataset_summary.json')
        # print(summary_path)
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Failed to load {summary_path}: {e}")
                continue
            parts = summary_path.split('/')
            # print(parts)
            nth_part = parts[-6]  # n为你想要的下标
            row = {
                # '任务唯一编码': os.path.abspath(summary_path),
                '任务唯一编码': nth_part,
                '总条数': data.get('final_total_episodes', ''),
                '总时长(秒)': data.get('total_duration_sec', ''),
                '总时长(小时)': data.get('total_duration_hour', ''),
                '总大小(MB)': data.get('total_size_mb', ''),
                '总大小(TB)': data.get('total_size_tb', ''),
            }
            rows.append(row)

    fieldnames = ['任务唯一编码', '总条数', '总时长(秒)', '总时长(小时)', '总大小(MB)', '总大小(TB)']
    with open(output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Found {len(rows)} summaries. Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description='Fetch summary values from dataset_summary.json under lerobot/report')
    parser.add_argument('--root', '-r', default=os.getcwd(), help='Root directory to search')
    parser.add_argument('--output', '-o', default=os.path.join(os.getcwd(), 'fetch_all.csv'), help='Output CSV path')
    args = parser.parse_args()
    process(args.root, args.output)


if __name__ == '__main__':
    main()
