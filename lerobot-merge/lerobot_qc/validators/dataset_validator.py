"""
A 级检查：数据集级别的 LeRobot 格式合规性检查
- A1: 数据结构与维度一致性
- A2: 统计信息完整性检查
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class DatasetValidator:
    """数据集级别验证器"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.expected_features = config.get('expected_features', {})
        self.allow_zero_fields = config.get('allow_zero_fields', [])

    def validate(self, dataset_path: Path) -> Dict[str, Any]:
        """
        执行完整的 A 级检查

        Args:
            dataset_path: 数据集根目录路径

        Returns:
            检查结果字典
        """
        results = {
            'A1_structure_check': {},
            'A2_stats_check': {},
            'passed': True,
            'errors': []
        }

        # A1: 数据结构与维度一致性
        logger.info("执行 A1: 数据结构与维度一致性检查...")
        a1_result = self.check_data_structure(dataset_path)
        results['A1_structure_check'] = a1_result

        if not a1_result['passed']:
            results['passed'] = False
            results['errors'].append("A1 检查失败")

        # A2: 统计信息完整性检查
        logger.info("执行 A2: 统计信息完整性检查...")
        a2_result = self.check_stats_integrity(dataset_path)
        results['A2_stats_check'] = a2_result

        if not a2_result['passed']:
            results['passed'] = False
            results['errors'].append("A2 检查失败")

        return results

    def check_data_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """
        A1: 检查 info.json 中 features 的 keys 和类型

        Args:
            dataset_path: 数据集根目录

        Returns:
            检查结果
        """
        result = {
            'passed': True,
            'info_json_exists': False,
            'missing_keys': [],
            'unexpected_keys': [],
            'type_mismatches': [],
            'dim_mismatches': []
        }

        info_path = dataset_path / 'meta' / 'info.json'

        # 检查 info.json 是否存在
        if not info_path.exists():
            result['passed'] = False
            result['error'] = f"info.json 不存在: {info_path}"
            logger.error(result['error'])
            return result

        result['info_json_exists'] = True

        # 读取 info.json
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
        except Exception as e:
            result['passed'] = False
            result['error'] = f"读取 info.json 失败: {e}"
            logger.error(result['error'])
            return result

        # 检查 features 字段
        if 'features' not in info_data:
            result['passed'] = False
            result['error'] = "info.json 中缺少 'features' 字段"
            logger.error(result['error'])
            return result

        actual_features = info_data['features']

        # 检查期望的 keys 是否存在
        expected_keys = set(self.expected_features.keys())
        actual_keys = set(actual_features.keys())

        # 找出缺失的 keys
        missing_keys = expected_keys - actual_keys
        if missing_keys:
            result['missing_keys'] = list(missing_keys)
            logger.warning(f"缺失的 keys: {missing_keys}")

        # 找出多余的 keys（实际存在但未在配置中定义）
        unexpected_keys = actual_keys - expected_keys
        if unexpected_keys:
            result['unexpected_keys'] = list(unexpected_keys)
            logger.info(f"未在配置中定义的 keys（可能是额外字段）: {unexpected_keys}")

        # 检查类型和维度
        for key in actual_keys & expected_keys:
            expected_spec = self.expected_features[key]
            actual_spec = actual_features[key]

            # 检查数据类型
            expected_dtype = expected_spec.get('dtype')
            actual_dtype = actual_spec.get('dtype')

            if expected_dtype and actual_dtype:
                # 标准化类型名称（处理 float32/float/Float 等变体）
                expected_dtype_norm = self._normalize_dtype(expected_dtype)
                actual_dtype_norm = self._normalize_dtype(actual_dtype)

                if expected_dtype_norm != actual_dtype_norm:
                    mismatch = {
                        'key': key,
                        'expected_dtype': expected_dtype,
                        'actual_dtype': actual_dtype
                    }
                    result['type_mismatches'].append(mismatch)
                    logger.warning(f"类型不匹配: {key} - 期望 {expected_dtype}, 实际 {actual_dtype}")

            # 检查维度（如果配置中指定了）
            expected_dim = expected_spec.get('dim')
            actual_shape = actual_spec.get('shape')

            if expected_dim is not None and actual_shape is not None:
                # shape 可能是列表，如 [None, 14] 表示 (batch, 14)
                if isinstance(actual_shape, list) and len(actual_shape) > 0:
                    actual_dim = actual_shape[-1]  # 取最后一个维度
                    if actual_dim != expected_dim:
                        mismatch = {
                            'key': key,
                            'expected_dim': expected_dim,
                            'actual_dim': actual_dim,
                            'actual_shape': actual_shape
                        }
                        result['dim_mismatches'].append(mismatch)
                        logger.warning(f"维度不匹配: {key} - 期望 {expected_dim}, 实际 {actual_dim}")

        # 判定是否通过
        if result['missing_keys'] or result['type_mismatches'] or result['dim_mismatches']:
            result['passed'] = False

        return result

    def check_stats_integrity(self, dataset_path: Path) -> Dict[str, Any]:
        """
        A2: 检查 episodes_stats.jsonl 中统计信息的完整性

        兼容LeRobot v2.1格式：每个episode一行，每行包含episode_index和stats字段

        Args:
            dataset_path: 数据集根目录

        Returns:
            检查结果
        """
        result = {
            'passed': True,
            'stats_file_exists': False,
            'anomalies': [],
            'total_episodes_checked': 0
        }

        stats_path = dataset_path / 'meta' / 'episodes_stats.jsonl'

        # 检查文件是否存在
        if not stats_path.exists():
            result['passed'] = False
            result['error'] = f"episodes_stats.jsonl 不存在: {stats_path}"
            logger.error(result['error'])
            return result

        result['stats_file_exists'] = True

        # 读取 JSONL 文件（LeRobot格式：每个episode一行）
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            logger.debug(f"读取了 {len(lines)} 行episode统计数据")

            if not lines:
                result['passed'] = False
                result['error'] = "episodes_stats.jsonl 为空"
                logger.error(result['error'])
                return result

            # 解析每个episode的统计数据
            all_stats_fields = {}
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue

                try:
                    episode_stats = json.loads(line)

                    # 检查是否有episode_index
                    if 'episode_index' not in episode_stats:
                        result['anomalies'].append({
                            'line': line_num,
                            'issues': ['缺少episode_index字段']
                        })
                        result['passed'] = False
                        continue

                    # 提取stats字段（可能在顶层或嵌套在'stats'字段下）
                    if 'stats' in episode_stats:
                        stats_fields = episode_stats['stats']
                    else:
                        # 如果没有stats字段，假设统计数据在顶层
                        stats_fields = {k: v for k, v in episode_stats.items()
                                       if k not in ['episode_index', 'frame_index', 'timestamp']}

                    # 合并所有episode的统计字段（用于后续检查）
                    for field_name, field_stats in stats_fields.items():
                        if field_name not in all_stats_fields:
                            all_stats_fields[field_name] = []
                        all_stats_fields[field_name].append({
                            'episode_index': episode_stats.get('episode_index'),
                            'stats': field_stats
                        })

                    result['total_episodes_checked'] += 1

                except json.JSONDecodeError as e:
                    result['anomalies'].append({
                        'line': line_num,
                        'issues': [f'JSON格式错误: {str(e)}']
                    })
                    result['passed'] = False

        except Exception as e:
            result['passed'] = False
            result['error'] = f"读取 episodes_stats.jsonl 失败: {e}"
            logger.error(result['error'])
            return result

        # 检查统计字段的完整性和合理性
        # 对每个字段的所有episode统计数据进行检查
        for field_name, episode_stats_list in all_stats_fields.items():
            for episode_stat_info in episode_stats_list:
                episode_idx = episode_stat_info['episode_index']
                stats = episode_stat_info['stats']

                if not isinstance(stats, dict):
                    continue

                anomaly_info = {
                    'field': field_name,
                    'episode_index': episode_idx,
                    'issues': []
                }

                # 检查必要的统计量是否存在
                required_stats = ['mean', 'min', 'max']
                for stat_name in required_stats:
                    if stat_name not in stats:
                        anomaly_info['issues'].append(f"{stat_name} 字段缺失")
                    elif stats[stat_name] is None:
                        anomaly_info['issues'].append(f"{stat_name} 为空值")

                # # 检查 mean 和 max 是否为 0（如果字段不在白名单中）
                # if field_name not in self.allow_zero_fields:
                #     mean_val = stats.get('mean')
                #     max_val = stats.get('max')

                #     # 处理标量和数组的情况
                #     if mean_val is not None:
                #         if isinstance(mean_val, (list, tuple)):
                #             # 数组：检查是否所有元素都为 0
                #             if all(v == 0 for v in mean_val):
                #                 anomaly_info['issues'].append("mean 所有元素均为 0")
                #         else:
                #             # 标量
                #             if mean_val == 0:
                #                 anomaly_info['issues'].append("mean 为 0")

                #     if max_val is not None:
                #         if isinstance(max_val, (list, tuple)):
                #             if all(v == 0 for v in max_val):
                #                 anomaly_info['issues'].append("max 所有元素均为 0")
                #         else:
                #             if max_val == 0:
                #                 anomaly_info['issues'].append("max 为 0")

                # 检查 min <= max
                min_val = stats.get('min')
                max_val = stats.get('max')
                if min_val is not None and max_val is not None:
                    try:
                        if isinstance(min_val, (list, tuple)) and isinstance(max_val, (list, tuple)):
                            for i, (mn, mx) in enumerate(zip(min_val, max_val)):
                                if mn > mx:
                                    anomaly_info['issues'].append(f"min > max at index {i}: {mn} > {mx}")
                        else:
                            if min_val > max_val:
                                anomaly_info['issues'].append(f"min > max: {min_val} > {max_val}")
                    except Exception as e:
                        logger.warning(f"比较 min/max 时出错: {field_name}, episode {episode_idx}, {e}")

                # 记录异常
                if anomaly_info['issues']:
                    result['anomalies'].append(anomaly_info)
                    logger.warning(f"Episode {episode_idx} 字段 {field_name} 存在异常: {anomaly_info['issues']}")

        # 判定是否通过
        if result['anomalies']:
            result['passed'] = False

        return result

    @staticmethod
    def _normalize_dtype(dtype: str) -> str:
        """标准化数据类型名称"""
        dtype_lower = dtype.lower()

        # uint8 类型
        if 'uint8' in dtype_lower or 'byte' in dtype_lower:
            return 'uint8'

        # float 类型
        if 'float32' in dtype_lower or dtype_lower == 'float':
            return 'float32'

        if 'float64' in dtype_lower or dtype_lower == 'double':
            return 'float64'

        # int 类型
        if 'int32' in dtype_lower or dtype_lower == 'int':
            return 'int32'

        if 'int64' in dtype_lower or dtype_lower == 'long':
            return 'int64'

        return dtype_lower
