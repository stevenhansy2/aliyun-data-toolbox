#!/usr/bin/env python3
"""
LeRobot 数据集本地质量检测工具

用法:
    python validator_local.py --dataset /root/data/task
    python validator_local.py --dataset /home/zhouzhuang/projects/VLADataProcess/robbylerobot_utils/test_data/bodeng_test
    python validator_local.py --dataset /root/data/task --config config/detection_config.yaml
"""

import argparse
import logging
import sys
import os
import yaml
import subprocess
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

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


def find_episode_files(dataset_path: Path) -> List[Path]:
    """
    查找所有 episode parquet 文件

    Args:
        dataset_path: 数据集根目录

    Returns:
        episode 文件路径列表
    """
    data_dir = dataset_path / 'data'
    
    if not data_dir.exists():
        logging.warning(f"数据目录不存在: {data_dir}")
        return []
    
    episode_files = []
    
    # 查找所有 chunk-* 目录
    chunk_dirs = sorted(data_dir.glob('chunk-*'))
    
    if not chunk_dirs:
        logging.warning(f"在 {data_dir} 中未找到 chunk-* 目录")
        return []
    
    # 在每个 chunk 目录中查找 episode_*.parquet 文件
    for chunk_dir in chunk_dirs:
        if chunk_dir.is_dir():
            chunk_episodes = sorted(chunk_dir.glob('episode_*.parquet'))
            episode_files.extend(chunk_episodes)
    
    if not episode_files:
        logging.warning(f"在 chunk 目录中未找到 episode_*.parquet 文件")
    
    return episode_files


def parse_episode_indices(episodes_str: str, total_episodes: int) -> List[int]:
    """
    解析 episode 索引字符串

    Args:
        episodes_str: 索引字符串，如 "0,1,2" 或 "0-10"
        total_episodes: 总 episode 数

    Returns:
        索引列表
    """
    if not episodes_str:
        # 默认检查所有 episodes
        return list(range(total_episodes))

    indices = []

    for part in episodes_str.split(','):
        part = part.strip()
        if '-' in part:
            # 范围
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            indices.extend(range(start, end + 1))
        else:
            # 单个索引
            indices.append(int(part))

    return sorted(set(indices))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='LeRobot 数据集本地质量检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='/home/leju_kuavo/temp/test',
        help='数据集根目录路径（默认: /root/data/task）'
    )
    parser.add_argument('--oss-config', type=str, default='/home/leju_kuavo/lerobot-merge/lerobot_qc/config/oss_config.yaml', help='OSS配置文件路径')
    parser.add_argument('--from-oss', action="store_true", help='数据来自oss,会下载到本地dataset中')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/leju_kuavo/lerobot-merge/lerobot_qc/config/custom_leju_kuavo4pro_claw.yaml',
        help='配置文件路径（默认: config/detection_config.yaml）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='报告输出目录（默认: reports）'
    )
    parser.add_argument(
        '--episodes',
        type=str,
        help='指定要检查的 episode 索引，支持范围（例如: 0,1,2 或 0-10）'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别（默认: INFO）'
    )
    parser.add_argument(
        '--full-validation',
        action='store_true',
        help='启用完整验证（访问所有帧，较慢但更彻底）'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.3,
        help='gripper检测边界扩充范围（默认: 0.3）'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

     # 加载OSS配置
    oss_config_path = Path(args.oss_config)
    logger.info(f"加载OSS配置: {oss_config_path}")
    oss_config = load_config(oss_config_path)

    # 获取任务名称
    dataset_path = Path(args.dataset)
    if args.from_oss:
        if oss_config['oss']['prefix'] == '':
            oss_config['oss']['prefix'] = os.environ.get('OSS_PREFIX')
        task_name = oss_config['oss']['prefix']
        oss_config['oss']['bucket'] = os.environ.get('BUCKET_NAME')
        bucket_name = oss_config['oss']['bucket']
        # Download data from oss to dataset
        oss_full_path = f"oss://{bucket_name}/{task_name}"
        if dataset_path.exists() and dataset_path.is_dir():
                shutil.rmtree(dataset_path)
        result = subprocess.run(["ossutil64", "cp", "-r", oss_full_path, args.dataset, "-j", "10"], capture_output=True, text=True)
        result = subprocess.run(["ossutil64", "cp", "-r", oss_full_path, args.dataset, "-u", "-j", "10"], capture_output=True, text=True)
        if result.returncode != 0:
            print("下载数据失败！")
            print("错误信息:", result.stderr)
        else:
            print("下载数据成功！")
            res = subprocess.run(["ls", "/root/data/task"], capture_output=True, text=True)
            print(res.stdout)
    else:
        task_name = dataset_path
    # 初始化OSS客户端
    oss_client = OSSClient(oss_config['oss'])
    logger.info(f"任务名称: {task_name}")

    # 验证路径
    if not dataset_path.exists():
        logger.error(f"数据集路径不存在: {dataset_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    # 初始化结果字典
    results = {
        'task_name': str(task_name),
        'from_oss': args.from_oss,
        'config_path': str(config_path),
        'level0_validation': {},
        'dataset_validation': {},
        'episode_validations': [],
        'overall_passed': True
    }

    # ========================================
    # Level 0 检查：LeRobot 格式验证（一票否决）
    # ========================================
    logger.info("=" * 60)
    logger.info("开始 Level 0 检查（LeRobot 格式验证）...")
    logger.info("=" * 60)

    format_validator = LeRobotFormatValidator(config)
    
    if args.full_validation:
        level0_results = format_validator.validate_full(dataset_path)
    else:
        level0_results = format_validator.validate(dataset_path)
    
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
        
        # prefix_folder = os.environ.get('OSS_PREFIX_FOLDER', 'ori_raw_data/quality_check/')
        # if not prefix_folder.endswith('/'):
        #     prefix_folder += '/'
        # task_name = os.path.basename(os.path.normpath(oss_client.prefix))
        # oss_client.upload_folder_non_recursive(f"{prefix_folder}{task_name}",output_dir)
        
        return 1

    logger.info("=" * 60)
    logger.info("✅ Level 0 检查通过，继续执行后续检查")
    logger.info("=" * 60)

    # ========================================
    # A 级检查：数据集级别
    # ========================================
    logger.info("=" * 60)
    logger.info("开始 A 级检查（数据集级别）...")
    logger.info("=" * 60)

    dataset_validator = DatasetValidator(config)
    dataset_results = dataset_validator.validate(dataset_path)
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
        # prefix_folder = os.environ.get('OSS_PREFIX_FOLDER', 'ori_raw_data/quality_check/')
        # if not prefix_folder.endswith('/'):
        #     prefix_folder += '/'
        # task_name = os.path.basename(os.path.normpath(oss_client.prefix))
        # oss_client.upload_folder_non_recursive(f"{prefix_folder}{task_name}",output_dir)

        return 1

    logger.info("=" * 60)
    logger.info("✅ Level A 检查通过，继续执行后续检查")
    logger.info("=" * 60)

    # ========================================
    # B 级检查：Episode 级别
    # ========================================
    logger.info("=" * 60)
    logger.info("开始 B 级检查（Episode 级别）...")
    logger.info("=" * 60)

    # 查找 episode 文件
    episode_files = find_episode_files(dataset_path)

    if not episode_files:
        logger.error("未找到 episode 文件，跳过 B 级检查")
    else:
        # 解析要检查的 episode 索引
        episode_indices = parse_episode_indices(args.episodes, len(episode_files))

        logger.info(f"找到 {len(episode_files)} 个 episode 文件")
        logger.info(f"将检查 {len(episode_indices)} 个 episodes: {episode_indices[:10]}{'...' if len(episode_indices) > 10 else ''}")

        # 从 Level 0 结果中获取 fps
        fps = level0_results.get('fps', None)
        episode_validator = EpisodeValidator(config, tolerance=args.tolerance, fps=fps)

        for idx in episode_indices:
            if idx >= len(episode_files):
                logger.warning(f"Episode 索引 {idx} 超出范围，跳过")
                continue

            episode_file = episode_files[idx]
            logger.info(f"检查 Episode {idx}: {episode_file.name}")

            episode_result = episode_validator.validate_episode(episode_file, idx)
            results['episode_validations'].append(episode_result)

            if not episode_result['passed']:
                results['overall_passed'] = False

        # 统计
        total_episodes = len(results['episode_validations'])
        passed_episodes = sum(1 for ep in results['episode_validations'] if ep.get('passed', False))
        logger.info(f"Episode 检查完成: {passed_episodes}/{total_episodes} 通过")

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
    # prefix_folder = os.environ.get('OSS_PREFIX_FOLDER', 'ori_raw_data/quality_check/')
    # if not prefix_folder.endswith('/'):
    #     prefix_folder += '/'
    # task_name = os.path.basename(os.path.normpath(oss_client.prefix))
    # oss_client.upload_folder_non_recursive(f"{prefix_folder}{task_name}",output_dir)

    report = ReportLoader(results)
    validator = ReportValidator(report)
    status = validator.episode_final_status()
    if status is not None:
        print(f"discard: {validator.discard}")
        q_c_generator = QualityGenerator(report, oss_client, output_dir)
        q_c_generator.generate(validator.discard)

    # 输出总结
    logger.info("=" * 60)

    # 最终判定：Level 0、Level A 和 Level B 都必须通过
    level0_passed = report.get_level0_validation_result()
    levelA_passed = report.get_dataset_validation_result()

    # Level B: 检查是否有 episode 失败
    if results['episode_validations']:
        passed_episodes = sum(1 for ep in results['episode_validations'] if ep.get('passed', False))
        total_episodes = len(results['episode_validations'])
        levelB_passed = (passed_episodes == total_episodes)
    else:
        levelB_passed = True  # 没有 episode 检查，视为通过

    final_passed = level0_passed and levelA_passed and levelB_passed

    if final_passed:
        logger.info("✅ 数据集质量检测通过")
        logger.info("   - Level 0 (LeRobot 格式验证): ✅ 通过")
        logger.info("   - Level A (数据集级别检查): ✅ 通过")
        if results['episode_validations']:
            logger.info(f"   - Level B (Episode 级别检查): ✅ {passed_episodes}/{total_episodes} episodes 通过")
    else:
        logger.warning("❌ 数据集质量检测未通过")

        # Level 0 状态
        if not level0_passed:
            logger.warning("   - Level 0 (LeRobot 格式验证): ❌ 失败")
        else:
            logger.info("   - Level 0 (LeRobot 格式验证): ✅ 通过")

        # Level A 状态
        if not levelA_passed:
            logger.warning("   - Level A (数据集级别检查): ❌ 失败")
        else:
            logger.info("   - Level A (数据集级别检查): ✅ 通过")

        # Level B 状态
        if results['episode_validations']:
            if not levelB_passed:
                logger.warning(f"   - Level B (Episode 级别检查): ❌ {passed_episodes}/{total_episodes} episodes 通过")
            else:
                logger.info(f"   - Level B (Episode 级别检查): ✅ {passed_episodes}/{total_episodes} episodes 通过")

    logger.info("=" * 60)

    return 0 if level0_passed and levelA_passed else 1


if __name__ == '__main__':
    sys.exit(main())

