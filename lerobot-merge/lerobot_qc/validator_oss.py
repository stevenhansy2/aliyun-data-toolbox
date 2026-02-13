#!/usr/bin/env python3
"""
LeRobot 数据集质量检测工具 - OSS 流式检测入口

检测 OSS 上的单个任务，无需下载完整数据集

用法:
    python validator_oss.py --oss-config config/oss_config.yaml
    python validator_oss.py --oss-config config/oss_config.yaml --config config/detection_config.yaml
"""

import argparse
import logging
import sys
import yaml
import json
import os
import io
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import concurrent.futures

from validators.lerobot_format_validator import LeRobotFormatValidator
from validators.dataset_validator import DatasetValidator
from validators.episode_validator import EpisodeValidator
from utils.report_generator import ReportGenerator
from utils.oss_client import OSSClient
from utils.report_loader import ReportLoader
from validators.report_validator import ReportValidator
from utils.quality_check_generator import QualityGenerator


def setup_logging(log_level: str = 'INFO'):
    """配置日志"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_single_episode(args):
    episode_index, oss_client, episode_validator, _local_dataset_path, logger = args
    chunk_index = episode_index // 1000
    file_name = f"episode_{episode_index:06d}.parquet"
    oss_key = oss_client.get_task_path(f'data/chunk-{chunk_index:03d}/{file_name}')
    tmp_file = f"/root/{file_name}"
    try:
        tmp_path = Path(tmp_file)
        content = oss_client.get_object(oss_key)
        if content:
            with open(tmp_path, 'wb') as f:
                f.write(content)
            result = episode_validator.validate_episode(tmp_path, episode_index)
            tmp_path.unlink()
        else:
            logger.error(f"无法从OSS获取文件: {oss_key}")
            result = {
                'episode_index': episode_index,
                'passed': False,
                'error': f'File not found in OSS: {oss_key}'
            }
        # with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        #     tmp_path = Path(tmp_file.name)
        #     content = oss_client.get_object(oss_key)
        #     if content:
        #         with open(tmp_path, 'wb') as f:
        #             f.write(content)
        #         result = episode_validator.validate_episode(tmp_path, episode_index)
        #         tmp_path.unlink()
        #     else:
        #         logger.error(f"无法从OSS获取文件: {oss_key}")
        #         result = {
        #             'episode_index': episode_index,
        #             'passed': False,
        #             'error': f'File not found in OSS: {oss_key}'
        #         }
    except Exception as e:
        result = {
            'episode_index': episode_index,
            'passed': False,
            'error': str(e)
        }
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='LeRobot 数据集质量检测工具 - OSS流式检测',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--oss-config', type=str, default='config/oss_config.yaml', help='OSS配置文件路径')
    parser.add_argument('--config', type=str, default='config/detection_config.yaml', help='检测配置文件路径')
    parser.add_argument('--output', type=str, default='reports', help='报告输出目录')
    parser.add_argument('--tolerance', type=float, default=0.3, help='gripper检测边界扩充范围')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='日志级别')
    parser.add_argument('--full-validation', action='store_true', help='启用完整验证（访问所有帧，较慢但更彻底）')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # 加载OSS配置
    oss_config_path = Path(args.oss_config)
    logger.info(f"加载OSS配置: {oss_config_path}")
    oss_config = load_config(oss_config_path)
    num_workers = oss_config['oss'].get('num_workers', 4)  # 默认4线程
    # 从OSS配置中获取任务名称
    if oss_config['oss']['prefix'] == '':
        oss_config['oss']['prefix'] = os.environ.get('OSS_PREFIX')
    task_name = oss_config['oss']['prefix']
    logger.info(f"任务名称: {task_name}")

    # 加载检测配置
    config_path = Path(args.config)
    logger.info(f"加载检测配置: {config_path}")
    config = load_config(config_path)

    # 初始化OSS客户端
    oss_client = OSSClient(oss_config['oss'])

    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp(prefix='lerobot_check_'))
    local_dataset_path = temp_dir / 'dataset'
    local_dataset_path.mkdir(parents=True, exist_ok=True)

    # 初始化结果字典
    results = {
        'task_name': task_name,
        'from_oss': True,
        'config_path': str(config_path),
        'level0_validation': {},
        'dataset_validation': {},
        'episode_validations': [],
        'overall_passed': True
    }

    try:
        logger.info("=" * 60)
        logger.info("使用 OSS 流式检测模式")
        logger.info("=" * 60)

        # 下载元数据文件
        logger.info("下载元数据文件...")
        meta_dir = local_dataset_path / 'meta'
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        meta_files = ['info.json', 'episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl']
        
        for file_name in meta_files:
            oss_key = oss_client.get_task_path(f'meta/{file_name}')
            local_file = meta_dir / file_name
            
            content = oss_client.get_object(oss_key)
            if content:
                with open(local_file, 'wb') as f:
                    f.write(content)
                logger.info(f"✅ 下载: {file_name}")
            else:
                logger.warning(f"⚠️  文件不存在: {file_name}")
        
        # 下载标注文件
        logger.info("下载标注文件...")
        annotation = {}
        qwen_file_name = "episodes_qwen3vl_woquan_cn.jsonl"
        oss_key = oss_client.get_task_path(f'meta/{qwen_file_name}')
        local_file = meta_dir / qwen_file_name
        content = oss_client.get_object(oss_key)
        if content:
            logger.info(f"✅ 下载: {qwen_file_name}")
            bio = io.BytesIO(content)
            with io.TextIOWrapper(bio, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    episode_idx = data['episode_index']
                    action_cfg = data['action_config']
                    annotation[episode_idx] = (action_cfg[0]['start_frame'], action_cfg[-1]['end_frame'])
        else:
            logger.warning(f"⚠️  标注文件不存在: {oss_key} 检测完整数据")

        # hard code!
        if len(annotation) > 0:
            config['static_detection']['max_static_ratio'] = 0.10
            config['annotation'] = annotation
        else:
            config['static_detection']['max_static_ratio'] = 0.30

        # ========================================
        # Level 0 检查：LeRobot 格式验证（一票否决）
        # ========================================
        logger.info("=" * 60)
        logger.info("开始 Level 0 检查（LeRobot 格式验证）...")
        logger.info("=" * 60)

        format_validator = LeRobotFormatValidator(config)

        if args.full_validation:
            level0_results = format_validator.validate_full(local_dataset_path)
        else:
            level0_results = format_validator.validate(local_dataset_path)

        results['level0_validation'] = level0_results

        if not level0_results['passed']:
            results['overall_passed'] = False
            logger.error("=" * 60)
            logger.error("❌ Level 0 检查失败：数据集不符合 LeRobot 格式")
            logger.error("❌ 一票否决：后续检查将不会执行")
            logger.error("=" * 60)

            # 生成报告
            output_dir = Path(args.output)
            report_generator = ReportGenerator(output_dir)
            output_formats = config.get('global', {}).get('output_formats', ['json', 'markdown'])
            report_generator.generate_report(results, formats=output_formats)

            prefix_folder = os.environ.get('OSS_PREFIX_FOLDER', 'ori_raw_data/quality_check/')
            if not prefix_folder.endswith('/'):
                prefix_folder += '/'
            task_name_clean = os.path.basename(os.path.normpath(oss_client.prefix))
            oss_client.upload_folder_non_recursive(f"{prefix_folder}{task_name_clean}", output_dir)

            return 1

        logger.info("=" * 60)
        logger.info("✅ Level 0 检查通过，继续执行后续检查")
        logger.info("=" * 60)

        # ========================================
        # A 级检查：数据集级别（元数据检查）
        # ========================================
        logger.info("=" * 60)
        logger.info("开始 A 级检查（数据集级别）...")
        logger.info("=" * 60)

        dataset_validator = DatasetValidator(config)
        dataset_results = dataset_validator.validate(local_dataset_path)
        results['dataset_validation'] = dataset_results

        if not dataset_results['passed']:
            results['overall_passed'] = False
            logger.error("=" * 60)
            logger.error("❌ Level A 检查失败：数据集不符合要求")
            logger.error("❌ 一票否决：后续检查将不会执行")
            logger.error("=" * 60)

            # 生成报告
            output_dir = Path(args.output)
            report_generator = ReportGenerator(output_dir)
            output_formats = config.get('global', {}).get('output_formats', ['json', 'markdown'])
            report_generator.generate_report(results, formats=output_formats)

            prefix_folder = os.environ.get('OSS_PREFIX_FOLDER', 'ori_raw_data/quality_check/')
            if not prefix_folder.endswith('/'):
                prefix_folder += '/'
            task_name_clean = os.path.basename(os.path.normpath(oss_client.prefix))
            oss_client.upload_folder_non_recursive(f"{prefix_folder}{task_name_clean}", output_dir)

            return 1

        logger.info("=" * 60)
        logger.info("✅ Level A 检查通过，继续执行后续检查")
        logger.info("=" * 60)

        # ========================================
        # B 级检查：Episode 级别（流式检测）
        # ========================================
        logger.info("=" * 60)
        logger.info("开始 B 级检查（Episode 级别）...")
        logger.info("=" * 60)

        # 从info.json读取总episode数和fps
        with open(local_dataset_path / 'meta' / 'info.json', 'r') as f:
            info = json.load(f)
            total_episodes = info['total_episodes']
            fps = info.get('fps', None)

        episode_validator = EpisodeValidator(config, args.tolerance, fps=fps)

        logger.info(f"将检查 {total_episodes} 个 episodes (FPS={fps})")

        # 并行 episode 检查
        episode_args = [
            (i, oss_client, episode_validator, local_dataset_path, logger)
            for i in range(total_episodes)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for result in executor.map(validate_single_episode, episode_args):
                results['episode_validations'].append(result)
                if not result['passed']:
                    results['overall_passed'] = False

        # 统计
        passed = sum(1 for ep in results['episode_validations'] if ep.get('passed', False))
        logger.info(f"Episode 检查完成: {passed}/{total_episodes} 通过")

        # ========================================
        # 生成报告
        # ========================================
        logger.info("=" * 60)
        logger.info("生成检测报告...")
        logger.info("=" * 60)

        output_dir = Path(args.output)
        report_generator = ReportGenerator(output_dir)

        output_formats = config.get('global', {}).get('output_formats', ['json', 'markdown'])
        report_generator.generate_report(results, formats=output_formats)

        prefix_folder = os.environ.get('OSS_PREFIX_FOLDER', 'ori_raw_data/quality_check/')
        if not prefix_folder.endswith('/'):
            prefix_folder += '/'
        task_name_clean = os.path.basename(os.path.normpath(oss_client.prefix))
        oss_client.upload_folder_non_recursive(f"{prefix_folder}{task_name_clean}", output_dir)

        # 生成质量检查结果
        report = ReportLoader(results)
        validator = ReportValidator(report)
        status = validator.episode_final_status()
        if status is not None:
            logger.info(f"废弃 episodes: {validator.discard}")
            q_c_generator = QualityGenerator(report, oss_client, output_dir)
            q_c_generator.generate(validator.discard)

        # 输出总结
        logger.info("=" * 60)

        # 最终判定
        level0_passed = report.get_level0_validation_result()
        levelA_passed = report.get_dataset_validation_result()

        # Level B: 检查是否有 episode 失败
        if results['episode_validations']:
            passed_episodes = sum(1 for ep in results['episode_validations'] if ep.get('passed', False))
            total_ep = len(results['episode_validations'])
            levelB_passed = (passed_episodes == total_ep)
        else:
            levelB_passed = True

        final_passed = level0_passed and levelA_passed and levelB_passed

        if final_passed:
            logger.info("✅ 数据集质量检测通过")
            logger.info("   - Level 0 (LeRobot 格式验证): ✅ 通过")
            logger.info("   - Level A (数据集级别检查): ✅ 通过")
            if results['episode_validations']:
                logger.info(f"   - Level B (Episode 级别检查): ✅ {passed_episodes}/{total_ep} episodes 通过")
        else:
            logger.warning("❌ 数据集质量检测未通过")
            if not level0_passed:
                logger.warning("   - Level 0 (LeRobot 格式验证): ❌ 失败")
            else:
                logger.info("   - Level 0 (LeRobot 格式验证): ✅ 通过")
            if not levelA_passed:
                logger.warning("   - Level A (数据集级别检查): ❌ 失败")
            else:
                logger.info("   - Level A (数据集级别检查): ✅ 通过")
            if results['episode_validations']:
                if not levelB_passed:
                    logger.warning(f"   - Level B (Episode 级别检查): ❌ {passed_episodes}/{total_ep} episodes 通过")
                else:
                    logger.info(f"   - Level B (Episode 级别检查): ✅ {passed_episodes}/{total_ep} episodes 通过")

        logger.info("=" * 60)

    finally:
        # 清理临时文件
        logger.info("清理临时文件...")
        shutil.rmtree(temp_dir)
        logger.info(f"✅ 已删除临时目录: {temp_dir}")

    return 0 if level0_passed and levelA_passed else 1


if __name__ == '__main__':
    sys.exit(main())