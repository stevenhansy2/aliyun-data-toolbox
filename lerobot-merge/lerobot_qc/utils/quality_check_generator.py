import json
import os
from pathlib import Path
from typing import Dict, List, Any
from utils.report_loader import ReportLoader

discard_code = 2

class QualityGenerator:
    """质量结果生成器"""
    def __init__(self, report: ReportLoader, oss_client, output_dir):
        self.task_result = report
        self.oss_prefix = report.task_name if report.from_oss else ''
        self.to_oss = self.oss_prefix != ''
        self.folder_name = 'meta'
        self.file_name = 'quality_check.jsonl'
        self.output_dir = output_dir
        self.oss_client = oss_client

    def generate(self, discard_list):
        output_file = Path(os.path.join(self.output_dir, self.file_name))
        if output_file.exists():
            output_file.unlink()
        with open(output_file, "w", encoding="utf-8") as f:
            for idx in discard_list:
                item = {"episode_index": idx, "quality_check": discard_code}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if self.to_oss:
            print(f"from : {self.oss_prefix}{self.folder_name}/{self.file_name}")
            print(f"to : {output_file}")
            self.oss_client.upload_file(f"{self.oss_prefix}{self.folder_name}/{self.file_name}", output_file)
            


