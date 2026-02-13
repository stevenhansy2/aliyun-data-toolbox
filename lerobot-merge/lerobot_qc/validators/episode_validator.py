"""
B 级检查：Episode 级别的动作/状态数据质量检查
- B1: 异常静止检测
- B2: 角度域与 Gripper 检查
- B3: 原始信号异常检测
- B4: 时间戳一致性检查
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.signal_processing import (
    detect_smoothness_issues,
    detect_spikes_zscore,
    detect_static_windows,
    detect_angle_domain,
    detect_outliers_mad,
    compute_derivatives,
    detect_spikes,
    compute_total_variation,
    check_physical_limits,
    get_top_k_anomalies
)

logger = logging.getLogger(__name__)


class EpisodeValidator:
    """Episode 级别验证器"""

    def __init__(self, config: Dict[str, Any], tolerance: float = 0.3, fps: float = None):
        """
        Args:
            config: 配置字典
            tolerance: 容错范围
            fps: 数据集的 FPS（来自 meta/info.json）
        """
        self.config = config
        self.static_config = config.get('static_detection', {})
        self.angle_gripper_config = config.get('angle_gripper_check', {})
        self.anomaly_config = config.get('anomaly_detection', {})
        self.global_config = config.get('global', {})
        self.timestamp_config = config.get('timestamp_check', {})
        self.annotation = config.get('annotation', None)
        self.tolerance = tolerance
        # fps 优先使用传入的值（来自 info.json），其次使用配置文件，最后默认 30
        self.fps = fps if fps is not None else self.anomaly_config.get('fps', 30)


    def validate_episode(self, episode_path: Path, episode_idx: int) -> Dict[str, Any]:
        """
        执行完整的 B 级检查

        Args:
            episode_path: Episode parquet 文件路径
            episode_idx: Episode 索引

        Returns:
            检查结果字典
        """
        results = {
            'episode_idx': episode_idx,
            'episode_path': str(episode_path),
            'range': None,
            'B1_static_check': {},
            'B2_angle_gripper_check': {},
            'B3_anomaly_check': {},
            'B4_timestamp_check': {},
            'passed': True,
            'errors': []
        }

        # 读取 episode 数据
        try:
            table = pq.read_table(episode_path)
            range = self.annotation.get(episode_path) if self.annotation else None
            if range:
                x, y = range
                table = table.slice(x, y - x + 1)
                results['range'] = range
            data = table.to_pydict()
        except Exception as e:
            results['passed'] = False
            results['errors'].append(f"读取 episode 文件失败: {e}")
            logger.error(f"读取 {episode_path} 失败: {e}")
            return results

        # 提取时间戳
        timestamps = self._extract_timestamps(data)
        if timestamps is None:
            results['passed'] = False
            results['errors'].append("无法提取时间戳")
            logger.error(f"Episode {episode_idx}: 无法提取时间戳")
            return results
        
        # 检查episode持续时间
        duration = timestamps[-1] - timestamps[0]
        if duration < 3.0:
            results['passed'] = False
            results['errors'].append(f"Episode持续时间过短: {duration:.2f}秒 (最小要求: 3.0秒)")
            logger.error(f"Episode {episode_idx}: 持续时间过短 {duration:.2f}秒")
            return results
        
        logger.info(f"Episode {episode_idx}: 持续时间 {duration:.2f}秒, 共 {len(timestamps)} 帧")
        

        # B1: 异常静止检测
        logger.info(f"Episode {episode_idx}: 执行 B1 - 异常静止检测...")
        b1_result = self.check_static_motion(data, timestamps)
        results['B1_static_check'] = b1_result

        if not b1_result['passed']:
            results['passed'] = False
            results['errors'].append("B1 检查失败")

        # B2: 角度域与 Gripper 检查
        logger.info(f"Episode {episode_idx}: 执行 B2 - 角度域与 Gripper 检查...")
        b2_result = self.check_angle_gripper(data)
        results['B2_angle_gripper_check'] = b2_result

        if not b2_result['passed']:
            results['passed'] = False
            results['errors'].append("B2 检查失败")

        # B3: 原始信号异常检测
        logger.info(f"Episode {episode_idx}: 执行 B3 - 原始信号异常检测...")
        b3_result = self.check_signal_anomalies(data, timestamps)
        results['B3_anomaly_check'] = b3_result

        # B3 仅在极端情况下才算失败
        if not b3_result.get('passed', True):
            results['passed'] = False
            results['errors'].append("B3 检查失败")
            # 记录极端问题
            for issue in b3_result.get('critical_issues', []):
                logger.error(f"Episode {episode_idx}: B3 极端异常 - {issue}")

        # 记录警告信息
        if b3_result.get('warnings'):
            for warning in b3_result['warnings']:
                logger.warning(f"Episode {episode_idx}: B3 警告 - {warning}")

        # B4: 时间戳一致性检查
        logger.info(f"Episode {episode_idx}: 执行 B4 - 时间戳一致性检查...")
        b4_result = self.check_timestamp_consistency(timestamps)
        results['B4_timestamp_check'] = b4_result

        if not b4_result['passed']:
            results['passed'] = False
            results['errors'].append("B4 检查失败")
            for issue in b4_result.get('issues', []):
                logger.error(f"Episode {episode_idx}: B4 问题 - {issue}")

        return results

    def check_static_motion(self, data: Dict[str, Any], timestamps: np.ndarray) -> Dict[str, Any]:
        """
        B1: 检测异常静止（多维度综合检测）

        检测逻辑：
        1. 收集所有可能的运动维度数据（arm、base、waist、head等）
        2. 对每个维度单独检测静止窗口
        3. 合并所有维度的静止窗口，只有所有维度都静止才算真正静止
        4. 这样可以避免"关节不动但底盘在移动"等误报

        Args:
            data: Episode 数据字典
            timestamps: 时间戳数组

        Returns:
            检查结果
        """
        result = {
            'passed': True,
            'static_intervals': [],
            'static_ratio': 0.0,
            'threshold': self.static_config.get('max_static_ratio', 0.01),
            'dimension_details': {}  # 记录各维度的检测详情
        }

        # 获取配置参数
        window_sec = self.static_config.get('window_sec', 3.0)
        step_sec = self.static_config.get('step_sec', 0.5)
        threshold = self.static_config.get('angle_range_threshold', 0.01)
        max_ratio = self.static_config.get('max_static_ratio', 0.01)
        unwrap = self.global_config.get('unwrap_angles', True)

        # 定义所有可能的运动维度键（按优先级）
        motion_dimension_groups = {
            'arm': self.static_config.get('arm_joint_keys', []),
            'base': self.static_config.get('base_keys', [
                'action.base.position', 'action.base.velocity',
                'observation.state.base.position', 'observation.state.base.velocity'
            ]),
            'waist': self.static_config.get('waist_keys', [
                'action.waist.position', 'observation.state.waist.position'
            ]),
            'head': self.static_config.get('head_keys', [
                'action.head.position', 'observation.state.head.position'
            ]),
            'effector': self.static_config.get('effector_keys', [
                'action.effector.position', 'observation.state.effector.position'
            ])
        }

        # 收集所有有效的运动数据
        all_dimension_data = {}
        for dim_name, key_list in motion_dimension_groups.items():
            if not key_list:
                continue

            dim_data = []
            for key in key_list:
                data_array = self._extract_field(data, key)
                if data_array is not None:
                    dim_data.append(data_array)

            if dim_data:
                # 合并同一维度的多个数据源
                merged_data = np.concatenate(dim_data, axis=1) if len(dim_data) > 1 else dim_data[0]
                all_dimension_data[dim_name] = merged_data
                logger.info(f"检测到 {dim_name} 维度数据: shape={merged_data.shape}")

        if not all_dimension_data:
            result['warning'] = "未找到任何运动维度数据，跳过静止检测"
            logger.warning(result['warning'])
            return result

        # 帧数裁剪配置（避免开头结尾的静止）
        config = {
            'value': 1800,
            'l_threshold': 90,
            'g_threshold': 150
        }

        # 获取总帧数（从第一个维度）
        n_frames = len(next(iter(all_dimension_data.values())))

        # 计算裁剪范围
        if n_frames <= config['value']:
            if n_frames <= config['l_threshold'] * 3:
                trim_start = 0
                trim_end = n_frames
            else:
                trim_start = config['l_threshold']
                trim_end = n_frames - config['l_threshold']
        else:
            trim_start = config['g_threshold']
            trim_end = n_frames - config['g_threshold']

        # 对每个维度单独检测静止窗口
        dimension_static_masks = {}
        trimmed_length = trim_end - trim_start

        if trimmed_length <= 0:
            result['warning'] = f"裁剪后数据长度为0 (trim_start={trim_start}, trim_end={trim_end})"
            logger.warning(result['warning'])
            return result

        for dim_name, dim_data in all_dimension_data.items():
            # 裁剪数据
            trimmed_data = dim_data[trim_start:trim_end]
            trimmed_timestamps = timestamps[trim_start:trim_end]

            try:
                # 检测该维度的静止窗口
                static_intervals, static_ratio = detect_static_windows(
                    trimmed_data,
                    trimmed_timestamps,
                    window_sec=window_sec,
                    step_sec=step_sec,
                    threshold=threshold,
                    unwrap=unwrap
                )

                # 创建静止掩码（相对于裁剪后的数据）
                static_mask = np.zeros(len(trimmed_data), dtype=bool)
                for start, end in static_intervals:
                    static_mask[start:end+1] = True

                dimension_static_masks[dim_name] = static_mask

                # 记录详情
                result['dimension_details'][dim_name] = {
                    'static_ratio': float(static_ratio),
                    'static_intervals': [(int(s), int(e)) for s, e in static_intervals]
                }

                logger.info(f"{dim_name} 维度静止占比: {static_ratio:.2%}")

            except Exception as e:
                logger.warning(f"{dim_name} 维度静止检测失败: {e}")
                continue

        if not dimension_static_masks:
            result['warning'] = "所有维度的静止检测都失败"
            logger.warning(result['warning'])
            return result

        # 合并逻辑：只有所有维度都静止的帧才算真正静止
        combined_static_mask = np.ones(len(trimmed_data), dtype=bool)
        for dim_name, mask in dimension_static_masks.items():
            combined_static_mask &= mask  # 逻辑与：所有维度都静止才为True

        # 计算综合静止占比
        combined_static_ratio = np.sum(combined_static_mask) / len(combined_static_mask) if len(combined_static_mask) > 0 else 0.0

        # 找出综合静止区间
        combined_intervals = []
        in_static = False
        start_idx = 0

        for i in range(len(combined_static_mask)):
            if combined_static_mask[i] and not in_static:
                start_idx = i
                in_static = True
            elif not combined_static_mask[i] and in_static:
                combined_intervals.append((start_idx, i - 1))
                in_static = False

        if in_static:
            combined_intervals.append((start_idx, len(combined_static_mask) - 1))

        result['static_intervals'] = [(int(s), int(e)) for s, e in combined_intervals]
        result['static_ratio'] = float(combined_static_ratio)
        result['dimensions_checked'] = list(dimension_static_masks.keys())

        # 判定是否通过
        if combined_static_ratio > max_ratio:
            result['passed'] = False
            logger.warning(f"综合静止帧占比 {combined_static_ratio:.2%} 超过阈值 {max_ratio:.2%}")
            logger.warning(f"检测维度: {', '.join(dimension_static_masks.keys())}")
        else:
            logger.info(f"✅ 静止检测通过: {combined_static_ratio:.2%} <= {max_ratio:.2%}")

        return result

    def check_angle_gripper(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        B2: 角度域与 Gripper 检查

        Args:
            data: Episode 数据字典

        Returns:
            检查结果
        """
        result = {
            'passed': True,
            'angle_domain_results': {},
            'gripper_check_results': {},
            'angle_limit_check': {}
        }

        # 1. 角度域检测
        angle_keys = self.angle_gripper_config.get('angle_keys', [])
        for key in angle_keys:
            angles = self._extract_field(data, key)
            if angles is None:
                continue

            try:
                domain_type, out_of_range_indices = detect_angle_domain(angles)
                result['angle_domain_results'][key] = {
                    'domain_type': domain_type,
                    'out_of_range_count': len(out_of_range_indices),
                    'out_of_range_indices': out_of_range_indices
                }

                if domain_type == "other":
                    logger.warning(f"{key} 的角度域不在标准范围内，有 {len(out_of_range_indices)} 个异常值")
                    sample_size = min(10, len(out_of_range_indices))
                    for idx in out_of_range_indices[:sample_size]:
                        logger.warning(f"idx {idx}: value={angles[idx]}")

            except Exception as e:
                logger.error(f"角度域检测失败 ({key}): {e}")

        # 1.5. 角度物理限制检查（-2π ~ 2π）
        import numpy as np
        angle_min = -2 * np.pi
        angle_max = 2 * np.pi

        for key in angle_keys:
            angles = self._extract_field(data, key)
            if angles is None:
                continue

            try:
                # 展平数组以便检查
                angles_flat = angles.flatten() if angles.ndim > 1 else angles

                # 检查是否有超出范围的值
                exceed_min = angles_flat < angle_min
                exceed_max = angles_flat > angle_max
                exceed_mask = exceed_min | exceed_max
                exceed_count = np.sum(exceed_mask)
                exceed_ratio = exceed_count / len(angles_flat) if len(angles_flat) > 0 else 0.0

                result['angle_limit_check'][key] = {
                    'angle_min': float(angle_min),
                    'angle_max': float(angle_max),
                    'exceed_count': int(exceed_count),
                    'exceed_ratio': float(exceed_ratio),
                    'total_values': int(len(angles_flat))
                }

                if exceed_count > 0:
                    result['passed'] = False
                    # 找出超限的索引和值
                    exceed_indices = np.where(exceed_mask)[0]
                    exceed_values = angles_flat[exceed_indices]

                    logger.error(f"❌ {key} 角度超出物理限制 [-2π, 2π]")
                    logger.error(f"   超限数量: {exceed_count}/{len(angles_flat)} ({exceed_ratio*100:.2f}%)")
                    logger.error(f"   角度范围: [{angle_min:.4f}, {angle_max:.4f}] rad")

                    # 显示前10个超限值
                    sample_size = min(10, len(exceed_indices))
                    logger.error(f"   前{sample_size}个超限值:")
                    for i in range(sample_size):
                        idx = exceed_indices[i]
                        val = exceed_values[i]
                        logger.error(f"     索引 {idx}: {val:.4f} rad")

                    result['angle_limit_check'][key]['exceed_indices'] = exceed_indices[:100].tolist()  # 最多保存100个
                    result['angle_limit_check'][key]['exceed_values'] = exceed_values[:100].tolist()
                else:
                    logger.debug(f"✅ {key} 角度在物理限制范围内")

            except Exception as e:
                logger.error(f"角度物理限制检查失败 ({key}): {e}")

        # 2. Gripper 检
        gripper_min = self.angle_gripper_config.get('gripper_min', 0.0) 
        gripper_max = self.angle_gripper_config.get('gripper_max', 1.0)
        
        # 获取 Gripper 数据(支持两种模式)
        gripper_data = self._extract_gripper_data(data)
        
        length = gripper_max - gripper_min
        EPS = 1e-6
        toleranced_gripper_min = gripper_min
        toleranced_gripper_max = gripper_max
        if abs(self.tolerance) > EPS: 
            if abs(gripper_min) < EPS:
                toleranced_gripper_min = -(abs(length) * self.tolerance)
            elif gripper_min > 0:
                toleranced_gripper_min = gripper_min * (1.0 - self.tolerance)
            else:
                toleranced_gripper_min = gripper_min * (1.0 + self.tolerance)
            
            if abs(gripper_max) < EPS:
                toleranced_gripper_max = abs(length) * self.tolerance
            elif gripper_max > 0:
                toleranced_gripper_max = gripper_max * (1.0 + self.tolerance)
            else:
                toleranced_gripper_max = gripper_max * (1.0 - self.tolerance)
        logger.info(f"tolerance gripper min: {toleranced_gripper_min}, tolerance gripper max: {toleranced_gripper_max}")

        for key, gripper_vals in gripper_data.items():
            if gripper_vals is None:
                continue
            try:
                out_of_range_mask = (gripper_vals < toleranced_gripper_min) | (gripper_vals > toleranced_gripper_max)
                out_of_range_indices = np.where(out_of_range_mask.any(axis=1))[0].tolist()


                pending_low_mask = (gripper_vals < gripper_min) & (gripper_vals > toleranced_gripper_min)
                pending_high_mask = (gripper_vals < toleranced_gripper_max) & (gripper_vals > gripper_max)
                pending_range_mask = pending_low_mask | pending_high_mask

                pending_range_indices = np.where(pending_range_mask.any(axis=1))[0].tolist()

                result['gripper_check_results'][key] = {
                    'out_of_range_count': len(out_of_range_indices),
                    'out_of_range_indices': out_of_range_indices,
                    'tolerance_range_count': len(pending_range_indices),
                    'tolerance_range_indices': pending_range_indices,
                    'min_value': float(np.min(gripper_vals)),
                    'max_value': float(np.max(gripper_vals)),
                    'gripper_min': gripper_min,
                    'gripper_max': gripper_max,
                    'toleranced_gripper_min': toleranced_gripper_min,
                    'toleranced_gripper_max': toleranced_gripper_max
                }

                if out_of_range_indices:
                    result['passed'] = False
                    logger.warning(f"{key} 有 {len(out_of_range_indices)} 个值超出 [{gripper_min}, {gripper_max}] 范围")
                    sample_size = min(10, len(out_of_range_indices))
                    for idx in out_of_range_indices[:sample_size]:
                        logger.warning(f"idx {idx}: value={gripper_vals[idx]} (范围: [{gripper_min}, {gripper_max}])")
            except Exception as e:
                logger.error(f"Gripper 检查失败 ({key}): {e}")

        return result



    def check_signal_anomalies(self, data: Dict[str, Any], timestamps: np.ndarray) -> Dict[str, Any]:
        """
        B3: 原始信号异常检测

        Args:
            data: Episode 数据字典
            timestamps: 时间戳数组

        Returns:
            检查结果（仅在极端情况下失败）
        """
        result = {
            'passed': True,              # 默认通过
            'warnings': [],              # 警告列表
            'critical_issues': [],       # 极端问题列表
            'outlier_detection': {},
            'spike_detection': {},
            'smoothness_check': {},
            'physical_limits_check': {}
        }

        # 获取配置参数
        fps = self.anomaly_config.get('fps', 30)
        dt = 1.0 / fps

        angle_mad_k = self.anomaly_config.get('angle_mad_k', 3.0)
        vel_mad_k = self.anomaly_config.get('vel_mad_k', 3.0)
        acc_mad_k = self.anomaly_config.get('acc_mad_k', 3.0)

        dtheta_thr = self.anomaly_config.get('dtheta_thr', 0.5)
        d2theta_thr = self.anomaly_config.get('d2theta_thr', 0.3)
        min_consecutive = self.anomaly_config.get('min_consecutive', 3)

        tv_per_sec_thr = self.anomaly_config.get('tv_per_sec_thr', 10.0)
        top_k = self.anomaly_config.get('top_k', 10)

        motor_limits = self.anomaly_config.get('motor_limits', {})

        # 获取需要检测的字段（angle 和 state 字段）
        angle_keys = self.angle_gripper_config.get('angle_keys', [])

        for key in angle_keys:
            angles = self._extract_field(data, key)
            if angles is None:
                continue

            field_result = {}

            try:
                # 1. 离群值检测（角度）
                outlier_mask, outlier_ratio = detect_outliers_mad(angles, k=angle_mad_k)
                field_result['outlier_ratio'] = float(outlier_ratio)
                field_result['outlier_top_k'] = get_top_k_anomalies(outlier_mask, k=top_k)

                # 2. 计算速度和加速度
                velocity = compute_derivatives(angles, dt, order=1)
                acceleration = compute_derivatives(angles, dt, order=2)

                # 速度离群值
                vel_outlier_mask, vel_outlier_ratio = detect_outliers_mad(velocity, k=vel_mad_k)
                field_result['velocity_outlier_ratio'] = float(vel_outlier_ratio)

                # 加速度离群值
                acc_outlier_mask, acc_outlier_ratio = detect_outliers_mad(acceleration, k=acc_mad_k)
                field_result['acceleration_outlier_ratio'] = float(acc_outlier_ratio)

                # # 3. 突变/尖峰检测
                # spike_intervals_1st, spike_ratio_1st = detect_spikes(velocity, dtheta_thr, min_consecutive)
                # field_result['first_derivative_spikes'] = {
                #     'spike_ratio': float(spike_ratio_1st),
                #     'spike_intervals': [(int(s), int(e)) for s, e in spike_intervals_1st[:top_k]]
                # }

                # spike_intervals_2nd, spike_ratio_2nd = detect_spikes(acceleration, d2theta_thr, min_consecutive)
                # field_result['second_derivative_spikes'] = {
                #     'spike_ratio': float(spike_ratio_2nd),
                #     'spike_intervals': [(int(s), int(e)) for s, e in spike_intervals_2nd[:top_k]]
                # }
                # 3. 突变/尖峰检测（改用 Z-score 自适应方法）
                spike_intervals_1st, spike_ratio_1st = detect_spikes_zscore(
                    velocity, 
                    z_threshold=3.0,  # 自适应阈值
                    min_consecutive=min_consecutive
                )
                field_result['first_derivative_spikes'] = {
                    'spike_ratio': float(spike_ratio_1st),
                    'spike_intervals': [(int(s), int(e)) for s, e in spike_intervals_1st[:top_k]],
                    'method': 'zscore'  # 标记方法
                }

                spike_intervals_2nd, spike_ratio_2nd = detect_spikes_zscore(
                    acceleration, 
                    z_threshold=3.0,
                    min_consecutive=min_consecutive
                )
                field_result['second_derivative_spikes'] = {
                    'spike_ratio': float(spike_ratio_2nd),
                    'spike_intervals': [(int(s), int(e)) for s, e in spike_intervals_2nd[:top_k]],
                    'method': 'zscore'
                }

                # 新增：基于 Jerk 的平顺度检测（黄金指标）
                smoothness_result = detect_smoothness_issues(
                    angles,
                    dt,
                    jerk_z_threshold=3.0,
                    min_consecutive=2
                )
                field_result['jerk_smoothness'] = smoothness_result
                field_result['is_smooth_by_jerk'] = smoothness_result['jerk_ratio'] < 0.05  # 阈值可调
                

                # 4. 抖动/不顺滑检测
                tv_per_sec = compute_total_variation(angles, dt)
                field_result['total_variation_per_sec'] = float(tv_per_sec)
                field_result['tv_threshold'] = float(tv_per_sec_thr)
                field_result['is_smooth'] = bool(tv_per_sec <= tv_per_sec_thr)

                

                # 5. 物理限制检查
                if motor_limits and key in motor_limits:
                    
                    limits = motor_limits[key]

                    # 角度限制
                    angle_min = limits.get('angle_min')
                    angle_max = limits.get('angle_max')
                    if angle_min is not None or angle_max is not None:
                        exceed_mask, exceed_ratio = check_physical_limits(angles, angle_min, angle_max)
                        field_result['angle_limit_exceed_ratio'] = float(exceed_ratio)

                    # 速度限制
                    vel_max = limits.get('velocity_max')
                    if vel_max is not None:
                        exceed_mask, exceed_ratio = check_physical_limits(np.abs(velocity), None, vel_max)
                        field_result['velocity_limit_exceed_ratio'] = float(exceed_ratio)

                    # 加速度限制
                    acc_max = limits.get('acceleration_max')
                    if acc_max is not None:
                        exceed_mask, exceed_ratio = check_physical_limits(np.abs(acceleration), None, acc_max)
                        field_result['acceleration_limit_exceed_ratio'] = float(exceed_ratio)
                else:
                    # 未提供物理限制，报告分位数
                    field_result['angle_percentiles'] = {
                        'p1': float(np.percentile(angles, 1)),
                        'p99': float(np.percentile(angles, 99)),
                        'min': float(np.min(angles)),
                        'max': float(np.max(angles))
                    }
                    field_result['velocity_percentiles'] = {
                        'p1': float(np.percentile(velocity, 1)),
                        'p99': float(np.percentile(velocity, 99)),
                        'min': float(np.min(velocity)),
                        'max': float(np.max(velocity))
                    }

                # 统计各关节的异常分布
                if angles.ndim == 2:
                    joint_anomaly_counts = np.sum(outlier_mask, axis=0).tolist()
                    field_result['joint_anomaly_distribution'] = joint_anomaly_counts

                result['outlier_detection'][key] = field_result

            except Exception as e:
                logger.error(f"信号异常检测失败 ({key}): {e}")
                result['outlier_detection'][key] = {'error': str(e)}

        # 判断是否存在极端异常（仅极端情况才算失败）
        # 获取极端情况阈值配置
        critical_thresholds = self.anomaly_config.get('b3_critical_thresholds', {})
        outlier_critical = critical_thresholds.get('outlier_ratio_critical', 0.50)
        spike_1st_critical = critical_thresholds.get('spike_ratio_1st_critical', 0.30)
        spike_2nd_critical = critical_thresholds.get('spike_ratio_2nd_critical', 0.30)
        jerk_critical = critical_thresholds.get('jerk_ratio_critical', 0.20)
        tv_critical = critical_thresholds.get('tv_per_sec_critical', 30.0)

        for key, field_result in result['outlier_detection'].items():
            if 'error' in field_result:
                continue

            # 1. 检查离群值是否过多
            outlier_ratio = field_result.get('outlier_ratio', 0)
            vel_outlier_ratio = field_result.get('velocity_outlier_ratio', 0)
            acc_outlier_ratio = field_result.get('acceleration_outlier_ratio', 0)

            if outlier_ratio > outlier_critical:
                result['passed'] = False
                result['critical_issues'].append(
                    f"{key}: 角度离群值过多 ({outlier_ratio:.1%} > {outlier_critical:.0%})"
                )
            elif outlier_ratio > 0.30:
                result['warnings'].append(
                    f"{key}: 角度离群值较多 ({outlier_ratio:.1%})"
                )

            if vel_outlier_ratio > outlier_critical:
                result['passed'] = False
                result['critical_issues'].append(
                    f"{key}: 速度离群值过多 ({vel_outlier_ratio:.1%} > {outlier_critical:.0%})"
                )
            elif vel_outlier_ratio > 0.30:
                result['warnings'].append(
                    f"{key}: 速度离群值较多 ({vel_outlier_ratio:.1%})"
                )

            if acc_outlier_ratio > outlier_critical:
                result['passed'] = False
                result['critical_issues'].append(
                    f"{key}: 加速度离群值过多 ({acc_outlier_ratio:.1%} > {outlier_critical:.0%})"
                )
            elif acc_outlier_ratio > 0.30:
                result['warnings'].append(
                    f"{key}: 加速度离群值较多 ({acc_outlier_ratio:.1%})"
                )

            # 2. 检查一阶导数尖峰是否过多
            spike_ratio_1st = field_result.get('first_derivative_spikes', {}).get('spike_ratio', 0)
            if spike_ratio_1st > spike_1st_critical:
                result['passed'] = False
                result['critical_issues'].append(
                    f"{key}: 一阶导数尖峰过多 ({spike_ratio_1st:.1%} > {spike_1st_critical:.0%})"
                )
            elif spike_ratio_1st > 0.10:
                result['warnings'].append(
                    f"{key}: 一阶导数尖峰较多 ({spike_ratio_1st:.1%})"
                )

            # 3. 检查二阶导数尖峰是否过多
            spike_ratio_2nd = field_result.get('second_derivative_spikes', {}).get('spike_ratio', 0)
            if spike_ratio_2nd > spike_2nd_critical:
                result['passed'] = False
                result['critical_issues'].append(
                    f"{key}: 二阶导数尖峰过多 ({spike_ratio_2nd:.1%} > {spike_2nd_critical:.0%})"
                )
            elif spike_ratio_2nd > 0.10:
                result['warnings'].append(
                    f"{key}: 二阶导数尖峰较多 ({spike_ratio_2nd:.1%})"
                )

            # 4. 检查Jerk平滑度（黄金指标）
            jerk_result = field_result.get('jerk_smoothness', {})
            jerk_ratio = jerk_result.get('jerk_ratio', 0)
            if jerk_ratio > jerk_critical:
                result['passed'] = False
                result['critical_issues'].append(
                    f"{key}: 数据极不平滑 (jerk异常 {jerk_ratio:.1%} > {jerk_critical:.0%})"
                )
            elif jerk_ratio > 0.10:
                result['warnings'].append(
                    f"{key}: 数据不够平滑 (jerk异常 {jerk_ratio:.1%})"
                )

            # 5. 检查总变差
            tv_per_sec = field_result.get('total_variation_per_sec', 0)
            if tv_per_sec > tv_critical:
                result['passed'] = False
                result['critical_issues'].append(
                    f"{key}: 信号抖动严重 (TV={tv_per_sec:.1f} > {tv_critical:.0f} rad/s)"
                )
            elif tv_per_sec > tv_per_sec_thr:
                result['warnings'].append(
                    f"{key}: 信号有抖动 (TV={tv_per_sec:.1f} rad/s)"
                )

        return result

    def check_timestamp_consistency(self, timestamps: np.ndarray) -> Dict[str, Any]:
        """
        B4: 时间戳一致性检查

        检查内容：
        1. 时间戳必须单调递增
        2. 时间戳间隔需要稳定
        3. 时间戳间隔需要与 fps 保持一致，误差在 0.1ms 内

        Args:
            timestamps: 时间戳数组（秒）

        Returns:
            检查结果字典
        """
        result = {
            'passed': True,
            'issues': [],
            'expected_fps': self.fps,
            'expected_interval_sec': 1.0 / self.fps,
            'monotonic': True,
            'interval_stats': {},
            'fps_consistency': {}
        }

        # 获取配置参数
        tolerance_ms = self.timestamp_config.get('tolerance_ms', 0.1)  # 默认 0.1ms
        tolerance_sec = tolerance_ms / 1000.0

        expected_interval = 1.0 / self.fps

        n_timestamps = len(timestamps)
        if n_timestamps < 2:
            result['passed'] = True
            result['warning'] = "时间戳数量不足，跳过检查"
            logger.warning(result['warning'])
            return result

        # 计算时间戳间隔
        intervals = np.diff(timestamps)

        # ========================================
        # 检查 1: 时间戳单调递增
        # ========================================
        non_increasing_indices = np.where(intervals <= 0)[0]
        if len(non_increasing_indices) > 0:
            result['passed'] = False
            result['monotonic'] = False
            result['non_increasing_count'] = int(len(non_increasing_indices))
            result['non_increasing_indices'] = non_increasing_indices[:20].tolist()  # 最多显示 20 个

            issue_msg = f"时间戳非单调递增: {len(non_increasing_indices)} 个位置"
            result['issues'].append(issue_msg)
            logger.error(f"❌ {issue_msg}")

            # 显示前 5 个问题位置
            for i in non_increasing_indices[:5]:
                logger.error(f"   索引 {i}: t[{i}]={timestamps[i]:.6f}s -> t[{i+1}]={timestamps[i+1]:.6f}s, diff={intervals[i]:.6f}s")

        # ========================================
        # 检查 2: 时间戳间隔稳定性和 FPS 一致性
        # ========================================
        # 计算间隔统计信息
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals))
        interval_min = float(np.min(intervals))
        interval_max = float(np.max(intervals))
        interval_median = float(np.median(intervals))

        result['interval_stats'] = {
            'mean_sec': interval_mean,
            'std_sec': interval_std,
            'min_sec': interval_min,
            'max_sec': interval_max,
            'median_sec': interval_median,
            'mean_ms': interval_mean * 1000,
            'std_ms': interval_std * 1000
        }

        # 检查与期望 FPS 的一致性
        deviation_from_expected = np.abs(intervals - expected_interval)
        max_deviation = float(np.max(deviation_from_expected))
        mean_deviation = float(np.mean(deviation_from_expected))

        # 统计超出容差的帧数
        exceeds_tolerance_mask = deviation_from_expected > tolerance_sec
        exceeds_count = int(np.sum(exceeds_tolerance_mask))
        exceeds_ratio = exceeds_count / len(intervals)

        result['fps_consistency'] = {
            'expected_interval_sec': expected_interval,
            'expected_interval_ms': expected_interval * 1000,
            'tolerance_ms': tolerance_ms,
            'max_deviation_ms': max_deviation * 1000,
            'mean_deviation_ms': mean_deviation * 1000,
            'exceeds_tolerance_count': exceeds_count,
            'exceeds_tolerance_ratio': float(exceeds_ratio),
            'total_intervals': len(intervals)
        }

        # 计算实际 FPS
        actual_fps = 1.0 / interval_mean if interval_mean > 0 else 0
        result['fps_consistency']['actual_fps'] = actual_fps
        result['fps_consistency']['fps_error'] = abs(actual_fps - self.fps)

        # 判断是否通过 FPS 一致性检查
        if exceeds_count > 0:
            result['passed'] = False
            issue_msg = f"时间戳间隔与 FPS 不一致: {exceeds_count}/{len(intervals)} ({exceeds_ratio*100:.2f}%) 帧超出 {tolerance_ms}ms 容差"
            result['issues'].append(issue_msg)
            logger.error(f"❌ {issue_msg}")
            logger.error(f"   期望间隔: {expected_interval*1000:.3f}ms (FPS={self.fps})")
            logger.error(f"   实际平均间隔: {interval_mean*1000:.3f}ms")
            logger.error(f"   最大偏差: {max_deviation*1000:.3f}ms")

            # 显示超出容差的前 10 个位置
            exceed_indices = np.where(exceeds_tolerance_mask)[0]
            for i in exceed_indices[:10]:
                deviation_ms = deviation_from_expected[i] * 1000
                logger.error(f"   索引 {i}: 间隔={intervals[i]*1000:.3f}ms, 偏差={deviation_ms:.3f}ms")

            result['exceeds_tolerance_indices'] = exceed_indices[:50].tolist()  # 最多保存 50 个
        else:
            logger.info(f"✅ 时间戳一致性检查通过")
            logger.info(f"   FPS: 期望={self.fps}, 实际={actual_fps:.2f}")
            logger.info(f"   间隔: 期望={expected_interval*1000:.3f}ms, 实际平均={interval_mean*1000:.3f}ms")
            logger.info(f"   最大偏差: {max_deviation*1000:.4f}ms (容差: {tolerance_ms}ms)")

        return result

    def _extract_gripper_data(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        提取 Gripper 数据(支持独立字段和嵌入模式)
        
        Returns:
            {字段路径: gripper数据数组}
        """
        gripper_data = {}
        extraction_config = self.angle_gripper_config.get('gripper_extraction', {})
        # 模式1: 尝试独立字段
        standalone_keys = extraction_config.get('standalone_keys', [])
        for key in standalone_keys:
            vals = self._extract_field(data, key)
            if vals is not None:
                gripper_data[key] = vals
        
        # 模式2: 从 arm.position 中提取
        embedded_rules = extraction_config.get('embedded_rules', {})
        for source_key, rule in embedded_rules.items():
            source_data = self._extract_field(data, source_key)
            print(f"source_key {source_key}")
            print(f"source_data {source_data.shape}")
            if source_data is None:
                continue
            
            try:
                indices = rule.get('indices', [])
                print(indices)
                if not indices:
                    continue
                
                # 提取指定维度
                gripper_vals = source_data[:, indices]
                
                # 生成描述性键名
                desc = rule.get('description', 'embedded')
                result_key = f"{source_key}[{','.join(map(str, indices))}]"
                
                gripper_data[result_key] = gripper_vals
                logger.info(f"从 {source_key} 提取 Gripper 数据: {desc}")
                
            except Exception as e:
                logger.error(f"从 {source_key} 提取 Gripper 失败: {e}")
        
        # 兼容旧配置(如果没有 gripper_extraction 字段)
        if not gripper_data:
            old_keys = self.angle_gripper_config.get('gripper_keys', [])
            for key in old_keys:
                vals = self._extract_field(data, key)
                if vals is not None:
                    gripper_data[key] = vals
        
        return gripper_data
    def _extract_field(self, data: Dict[str, Any], key: str) -> Optional[np.ndarray]:
        """
        从数据字典中提取字段

        Args:
            data: 数据字典
            key: 字段路径，如 "observation.state.arm.position"

        Returns:
            数据数组或 None
        """
        parts = key.split('.')
        current = data
        if key in current:
            current = current[key]
        else:
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    logger.debug(f"字段 {key} 不存在")
                    return None

        # 转换为 numpy 数组
        if current is not None:
            try:
                arr = np.array(current)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                return arr
            except Exception as e:
                logger.error(f"转换字段 {key} 为 numpy 数组失败: {e}")
                return None

        return None

    def _extract_timestamps(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        提取时间戳

        Args:
            data: 数据字典

        Returns:
            时间戳数组或 None
        """
        # 尝试常见的时间戳字段名
        timestamp_keys = ['timestamp', 'timestamps', 'time', 'index']

        for key in timestamp_keys:
            if key in data:
                try:
                    timestamps = np.array(data[key])
                    if len(timestamps) > 0:
                        return timestamps
                except Exception as e:
                    logger.debug(f"提取时间戳字段 {key} 失败: {e}")

        # 如果没有时间戳，使用帧索引生成（假设 30fps）
        logger.warning("未找到时间戳字段，使用帧索引生成（假设 30fps）")
        fps = self.anomaly_config.get('fps', 30)

        # 找一个有效字段来确定帧数
        for key, value in data.items():
            try:
                arr = np.array(value)
                n_frames = len(arr)
                timestamps = np.arange(n_frames) / fps
                return timestamps
            except:
                continue

        return None
    

    def visualize_signal_anomalies(self, data: Dict[str, Any], timestamps: np.ndarray, 
                                episode_idx: int, output_dir: Path) -> None:
            """
            可视化信号异常检测结果
            
            Args:
                data: Episode 数据字典
                timestamps: 时间戳数组 (未使用,改用帧索引计算)
                episode_idx: Episode 索引
                output_dir: 输出目录
            """
            angle_keys = self.angle_gripper_config.get('angle_keys', [])
            fps = self.anomaly_config.get('fps', 30)
            dt = 1.0 / fps
            
            for key in angle_keys:
                angles = self._extract_field(data, key)
                if angles is None:
                    continue
                    
                try:
                    # 使用帧索引计算时间轴
                    n_frames = len(angles)
                    time_axis = np.arange(n_frames) / fps  # 时间 = 帧索引 / 30fps
                    
                    # 计算导数
                    velocity = compute_derivatives(angles, dt, order=1)
                    acceleration = compute_derivatives(angles, dt, order=2)
                    
                    # 检测离群值
                    outlier_mask, _ = detect_outliers_mad(angles, k=self.anomaly_config.get('angle_mad_k', 3.0))
                    vel_outlier_mask, _ = detect_outliers_mad(velocity, k=self.anomaly_config.get('vel_mad_k', 3.0))
                    acc_outlier_mask, _ = detect_outliers_mad(acceleration, k=self.anomaly_config.get('acc_mad_k', 3.0))
                    
                    # 检测尖峰区间
                    dtheta_thr = self.anomaly_config.get('dtheta_thr', 0.5)
                    d2theta_thr = self.anomaly_config.get('d2theta_thr', 0.3)
                    min_consecutive = self.anomaly_config.get('min_consecutive', 3)
                    
                    vel_spike_intervals, _ = detect_spikes(velocity, dtheta_thr, min_consecutive)
                    acc_spike_intervals, _ = detect_spikes(acceleration, d2theta_thr, min_consecutive)
                    
                    # 创建图形
                    n_joints = angles.shape[1]
                    fig = plt.figure(figsize=(18, 5 * n_joints))
                    gs = GridSpec(n_joints, 3, figure=fig, hspace=0.4, wspace=0.3)
                    
                    for joint_idx in range(n_joints):
                        # ==================== 角度信号 ====================
                        ax1 = fig.add_subplot(gs[joint_idx, 0])
                        ax1.plot(time_axis, angles[:, joint_idx], 'b-', linewidth=1.2, 
                                label='Angle', alpha=0.8)
                        
                        # 标记离群点
                        outlier_indices = np.where(outlier_mask[:, joint_idx])[0]
                        if len(outlier_indices) > 0:
                            ax1.scatter(time_axis[outlier_indices], angles[outlier_indices, joint_idx], 
                                    c='red', s=30, marker='x', zorder=5, 
                                    label=f'Outliers ({len(outlier_indices)})', alpha=0.7)
                        
                        ax1.set_xlabel('Time (s)', fontsize=10)
                        ax1.set_ylabel('Angle (rad)', fontsize=10)
                        ax1.set_title(f'Joint {joint_idx} - Position', fontsize=11, fontweight='bold')
                        ax1.legend(loc='upper right', fontsize=9)
                        ax1.grid(True, alpha=0.3, linestyle='--')
                        
                        # 添加统计信息
                        angle_mean = np.mean(angles[:, joint_idx])
                        angle_std = np.std(angles[:, joint_idx])
                        ax1.text(0.02, 0.98, f'μ={angle_mean:.3f}\nσ={angle_std:.3f}', 
                                transform=ax1.transAxes, fontsize=9,
                                verticalalignment='top', bbox=dict(boxstyle='round', 
                                facecolor='wheat', alpha=0.5))
                        
                        # ==================== 速度信号 ====================
                        ax2 = fig.add_subplot(gs[joint_idx, 1])
                        ax2.plot(time_axis, velocity[:, joint_idx], 'g-', linewidth=1.2, 
                                label='Velocity', alpha=0.8)
                        
                        # 添加阈值线
                        ax2.axhline(y=dtheta_thr, color='orange', linestyle='--', 
                                linewidth=1.5, label=f'Threshold (±{dtheta_thr})', alpha=0.7)
                        ax2.axhline(y=-dtheta_thr, color='orange', linestyle='--', 
                                linewidth=1.5, alpha=0.7)
                        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
                        
                        # 标记尖峰区间（半透明填充）
                        spike_labeled = False
                        for start, end in vel_spike_intervals:
                            label = 'Spike Region' if not spike_labeled else ''
                            ax2.axvspan(time_axis[start], time_axis[min(end, n_frames-1)], 
                                    color='red', alpha=0.2, label=label)
                            spike_labeled = True
                        
                        # # 标记离群点
                        # vel_outlier_indices = np.where(vel_outlier_mask[:, joint_idx])[0]
                        # if len(vel_outlier_indices) > 0:
                        #     ax2.scatter(time_axis[vel_outlier_indices], 
                        #             velocity[vel_outlier_indices, joint_idx],
                        #             c='red', s=30, marker='x', zorder=5, 
                        #             label=f'Outliers ({len(vel_outlier_indices)})', alpha=0.7)
                        
                        ax2.set_xlabel('Time (s)', fontsize=10)
                        ax2.set_ylabel('Velocity (rad/s)', fontsize=10)
                        ax2.set_title(f'Joint {joint_idx} - Velocity (1st Derivative)', 
                                    fontsize=11, fontweight='bold')
                        ax2.legend(loc='upper right', fontsize=9)
                        ax2.grid(True, alpha=0.3, linestyle='--')
                        
                        # 添加统计信息
                        vel_mean = np.mean(velocity[:, joint_idx])
                        vel_std = np.std(velocity[:, joint_idx])
                        vel_max = np.max(np.abs(velocity[:, joint_idx]))
                        ax2.text(0.02, 0.98, 
                                f'μ={vel_mean:.3f}\nσ={vel_std:.3f}\nmax={vel_max:.3f}', 
                                transform=ax2.transAxes, fontsize=9,
                                verticalalignment='top', bbox=dict(boxstyle='round', 
                                facecolor='lightgreen', alpha=0.5))
                        
                        # ==================== 加速度信号 ====================
                        ax3 = fig.add_subplot(gs[joint_idx, 2])
                        ax3.plot(time_axis, acceleration[:, joint_idx], 'r-', linewidth=1.2, 
                                label='Acceleration', alpha=0.8)
                        
                        # 添加阈值线
                        ax3.axhline(y=d2theta_thr, color='orange', linestyle='--', 
                                linewidth=1.5, label=f'Threshold (±{d2theta_thr})', alpha=0.7)
                        ax3.axhline(y=-d2theta_thr, color='orange', linestyle='--', 
                                linewidth=1.5, alpha=0.7)
                        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
                        
                        # 标记尖峰区间
                        spike_labeled = False
                        for start, end in acc_spike_intervals:
                            label = 'Spike Region' if not spike_labeled else ''
                            ax3.axvspan(time_axis[start], time_axis[min(end, n_frames-1)], 
                                    color='red', alpha=0.2, label=label)
                            spike_labeled = True
                        
                        # # 标记离群点
                        # acc_outlier_indices = np.where(acc_outlier_mask[:, joint_idx])[0]
                        # if len(acc_outlier_indices) > 0:
                        #     ax3.scatter(time_axis[acc_outlier_indices], 
                        #             acceleration[acc_outlier_indices, joint_idx],
                        #             c='red', s=30, marker='x', zorder=5, 
                        #             label=f'Outliers ({len(acc_outlier_indices)})', alpha=0.7)
                        
                        ax3.set_xlabel('Time (s)', fontsize=10)
                        ax3.set_ylabel('Acceleration (rad/s²)', fontsize=10)
                        ax3.set_title(f'Joint {joint_idx} - Acceleration (2nd Derivative)', 
                                    fontsize=11, fontweight='bold')
                        ax3.legend(loc='upper right', fontsize=9)
                        ax3.grid(True, alpha=0.3, linestyle='--')
                        
                        # 添加统计信息
                        acc_mean = np.mean(acceleration[:, joint_idx])
                        acc_std = np.std(acceleration[:, joint_idx])
                        acc_max = np.max(np.abs(acceleration[:, joint_idx]))
                        ax3.text(0.02, 0.98, 
                                f'μ={acc_mean:.3f}\nσ={acc_std:.3f}\nmax={acc_max:.3f}', 
                                transform=ax3.transAxes, fontsize=9,
                                verticalalignment='top', bbox=dict(boxstyle='round', 
                                facecolor='lightcoral', alpha=0.5))
                    
                    # 添加总标题
                    total_time = n_frames / fps
                    fig.suptitle(f'Episode {episode_idx} - Signal Analysis: {key} | Duration: {total_time:.2f}s ({n_frames} frames @ {fps}fps)', 
                                fontsize=14, fontweight='bold', y=0.995)
                    
                    # 保存图形
                    output_dir.mkdir(parents=True, exist_ok=True)
                    safe_key = key.replace('.', '_')
                    output_path = output_dir / f'episode_{episode_idx:06d}_{safe_key}_anomalies.png'
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"✅ 已保存可视化结果: {output_path}")
                    
                except Exception as e:
                    logger.error(f"❌ 可视化失败 ({key}): {e}")
                    import traceback
                    logger.error(traceback.format_exc())