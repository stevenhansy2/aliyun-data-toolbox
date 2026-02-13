"""
信号处理工具模块
提供各种信号分析和异常检测函数
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """
    对角度序列进行解缠操作，避免 ±π 边界跳变

    Args:
        angles: 角度数组，shape (n_frames, n_joints) 或 (n_frames,)

    Returns:
        解缠后的角度数组
    """
    if angles.ndim == 1:
        return np.unwrap(angles)
    else:
        # 对每个关节单独解缠
        unwrapped = np.zeros_like(angles)
        for i in range(angles.shape[1]):
            unwrapped[:, i] = np.unwrap(angles[:, i])
        return unwrapped


def detect_angle_domain(angles: np.ndarray, tolerance: float = 0.1) -> Tuple[str, List[int]]:
    """
    检测角度数据属于哪个域

    Args:
        angles: 角度数组
        tolerance: 容差值（用于判断是否超出范围）

    Returns:
        domain_type: 角度域类型 ("(-pi, pi)", "(0, 2pi)", "other")
        out_of_range_indices: 超出范围的索引列表（仅当 domain_type="other" 时有意义）
    """
    angles_flat = angles.flatten()

    # 移除 NaN 值
    valid_angles = angles_flat[~np.isnan(angles_flat)]

    if len(valid_angles) == 0:
        return "unknown", []

    min_val = np.min(valid_angles)
    max_val = np.max(valid_angles)

    # 检查是否在 (-π, π) 范围内
    if min_val >= (-np.pi - tolerance) and max_val <= (np.pi + tolerance):
        return "(-pi, pi)", []

    # 检查是否在 (0, 2π) 范围内
    if min_val >= (0 - tolerance) and max_val <= (2 * np.pi + tolerance):
        return "(0, 2pi)", []

    # 不在标准范围内，找出超出范围的索引
    out_of_range_mask = (valid_angles < (-np.pi - tolerance)) | \
                        ((valid_angles > (np.pi + tolerance)) & (valid_angles < (0 - tolerance))) | \
                        (valid_angles > (2 * np.pi + tolerance))

    # 转换回原始索引
    valid_indices = np.where(~np.isnan(angles_flat))[0]
    out_of_range_indices = valid_indices[out_of_range_mask].tolist()

    return "other", out_of_range_indices


def compute_mad(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    计算中位数绝对偏差 (Median Absolute Deviation)

    Args:
        data: 输入数据
        axis: 计算轴

    Returns:
        MAD 值
    """
    median = np.nanmedian(data, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(data - median), axis=axis)
    return mad


def detect_outliers_mad(data: np.ndarray, k: float = 3.0) -> Tuple[np.ndarray, float]:
    """
    使用 MAD 方法检测离群值

    Args:
        data: 输入数据，shape (n_frames, n_joints) 或 (n_frames,)
        k: MAD 倍数阈值

    Returns:
        outlier_mask: 布尔数组，标记离群值位置
        outlier_ratio: 离群值占比
    """
    median = np.nanmedian(data, axis=0, keepdims=True)
    mad = compute_mad(data, axis=0)

    # 避免 MAD 为 0 的情况
    mad = np.where(mad == 0, 1e-6, mad)

    # 标记离群值
    outlier_mask = np.abs(data - median) > k * mad
    outlier_ratio = np.sum(outlier_mask) / outlier_mask.size

    return outlier_mask, outlier_ratio


def compute_derivatives(signal: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    """
    计算信号的导数（使用中心差分）

    Args:
        signal: 输入信号，shape (n_frames, n_joints) 或 (n_frames,)
        dt: 时间步长
        order: 导数阶数（1 或 2）

    Returns:
        导数数组
    """
    if order == 1:
        # 一阶导数 - 中心差分
        derivative = np.gradient(signal, dt, axis=0)
    elif order == 2:
        # 二阶导数
        first_deriv = np.gradient(signal, dt, axis=0)
        derivative = np.gradient(first_deriv, dt, axis=0)
    else:
        raise ValueError(f"Unsupported derivative order: {order}")

    return derivative


def detect_spikes(signal: np.ndarray, threshold: float, min_consecutive: int = 3) -> Tuple[List[Tuple[int, int]], float]:
    """
    检测信号中的尖峰/突变

    Args:
        signal: 输入信号，shape (n_frames, n_joints) 或 (n_frames,)
        threshold: 阈值
        min_consecutive: 最小连续帧数

    Returns:
        spike_intervals: 尖峰区间列表 [(start, end), ...]
        spike_ratio: 尖峰帧占比
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    # 检测超过阈值的点
    exceed_mask = np.abs(signal) > threshold

    # 任意关节超过阈值则该帧标记为异常
    exceed_any = np.any(exceed_mask, axis=1)

    # 找连续区间
    spike_intervals = []
    in_spike = False
    start_idx = 0

    for i in range(len(exceed_any)):
        if exceed_any[i] and not in_spike:
            # 进入尖峰区间
            start_idx = i
            in_spike = True
        elif not exceed_any[i] and in_spike:
            # 离开尖峰区间
            if i - start_idx >= min_consecutive:
                spike_intervals.append((start_idx, i - 1))
            in_spike = False

    # 处理末尾情况
    if in_spike and len(exceed_any) - start_idx >= min_consecutive:
        spike_intervals.append((start_idx, len(exceed_any) - 1))

    # 计算占比
    total_spike_frames = sum(end - start + 1 for start, end in spike_intervals)
    spike_ratio = total_spike_frames / len(exceed_any) if len(exceed_any) > 0 else 0.0

    return spike_intervals, spike_ratio


def compute_total_variation(signal: np.ndarray, dt: float) -> float:
    """
    计算单位时间总变差 (Total Variation per second)

    Args:
        signal: 输入信号，shape (n_frames, n_joints) 或 (n_frames,)
        dt: 时间步长（秒）

    Returns:
        tv_per_sec: 单位时间总变差
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    # 计算逐帧差分的绝对值之和
    diff = np.diff(signal, axis=0)
    tv = np.sum(np.abs(diff))

    # 归一化到每秒
    total_time = len(signal) * dt
    tv_per_sec = tv / total_time if total_time > 0 else 0.0

    return tv_per_sec


def detect_static_windows(angles: np.ndarray,
                         timestamps: np.ndarray,
                         window_sec: float = 3.0,
                         step_sec: float = 0.5,
                         threshold: float = 0.01,
                         unwrap: bool = True,
                         fps: float = 30.0) -> Tuple[List[Tuple[int, int]], float]:
    """
    检测静止窗口

    Args:
        angles: 角度数据，shape (n_frames, n_joints)
        timestamps: 时间戳数组，shape (n_frames,)
        window_sec: 窗口长度（秒）
        step_sec: 滑动步长（秒）
        threshold: 角度范围阈值（弧度）
        unwrap: 是否对角度进行解缠
        fps: 帧率（默认30fps）

    Returns:
        static_intervals: 静止区间列表 [(start, end), ...]
        static_ratio: 静止帧占比
    """
    if angles.ndim == 1:
        angles = angles.reshape(-1, 1)

    n_frames, n_joints = angles.shape

    # 解缠角度
    if unwrap:
        angles = unwrap_angles(angles)

    # 标记每帧是否在静止窗口内
    static_mask = np.zeros(n_frames, dtype=bool)

    # 转换秒为帧数
    window_frames = int(window_sec * fps)
    step_frames = int(step_sec * fps)

    # 滑动窗口（基于帧索引）
    current_idx = 0

    while current_idx + window_frames <= n_frames:
        # 窗口内的帧索引
        window_indices = np.arange(current_idx, current_idx + window_frames)

        if len(window_indices) > 1:
            # 计算窗口内所有关节的角度范围
            window_angles = angles[window_indices, :]
            angle_ranges = np.ptp(window_angles, axis=0)  # max - min for each joint

            # 检查是否所有关节的范围都小于阈值
            if np.all(angle_ranges < threshold):
                # 标记窗口内所有帧为静止
                static_mask[window_indices] = True

        current_idx += step_frames

    # 找连续的静止区间
    static_intervals = []
    in_static = False
    start_idx = 0

    for i in range(n_frames):
        if static_mask[i] and not in_static:
            start_idx = i
            in_static = True
        elif not static_mask[i] and in_static:
            static_intervals.append((start_idx, i - 1))
            in_static = False

    if in_static:
        static_intervals.append((start_idx, n_frames - 1))

    # 计算占比
    static_ratio = np.sum(static_mask) / n_frames if n_frames > 0 else 0.0

    return static_intervals, static_ratio


def check_physical_limits(data: np.ndarray,
                         min_limit: Optional[float] = None,
                         max_limit: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    检查数据是否超出物理限制

    Args:
        data: 输入数据
        min_limit: 最小值限制
        max_limit: 最大值限制

    Returns:
        exceed_mask: 布尔数组，标记超限位置
        exceed_ratio: 超限占比
    """
    exceed_mask = np.zeros_like(data, dtype=bool)

    if min_limit is not None:
        exceed_mask |= (data < min_limit)

    if max_limit is not None:
        exceed_mask |= (data > max_limit)

    exceed_ratio = np.sum(exceed_mask) / exceed_mask.size if exceed_mask.size > 0 else 0.0

    return exceed_mask, exceed_ratio


def get_top_k_anomalies(anomaly_mask: np.ndarray, k: int = 10) -> List[Tuple[int, int, int]]:
    """
    获取 Top-K 异常片段

    Args:
        anomaly_mask: 异常掩码，shape (n_frames, n_joints)
        k: 返回前 K 个片段

    Returns:
        top_k_segments: [(start, end, joint_idx), ...] 按异常严重程度排序
    """
    if anomaly_mask.ndim == 1:
        anomaly_mask = anomaly_mask.reshape(-1, 1)

    n_frames, n_joints = anomaly_mask.shape
    segments = []

    # 对每个关节单独处理
    for joint_idx in range(n_joints):
        joint_mask = anomaly_mask[:, joint_idx]

        # 找连续区间
        in_anomaly = False
        start_idx = 0

        for i in range(n_frames):
            if joint_mask[i] and not in_anomaly:
                start_idx = i
                in_anomaly = True
            elif not joint_mask[i] and in_anomaly:
                segment_length = i - start_idx
                segments.append((start_idx, i - 1, joint_idx, segment_length))
                in_anomaly = False

        if in_anomaly:
            segment_length = n_frames - start_idx
            segments.append((start_idx, n_frames - 1, joint_idx, segment_length))

    # 按片段长度排序，取 Top-K
    segments.sort(key=lambda x: x[3], reverse=True)
    top_k = [(s[0], s[1], s[2]) for s in segments[:k]]

    return top_k




def detect_spikes_zscore(signal: np.ndarray, 
                        z_threshold: float = 3.0, 
                        min_consecutive: int = 3) -> Tuple[List[Tuple[int, int]], float]:
    """
    基于 Z-score 的自适应尖峰检测（统计方法）
    
    Args:
        signal: 输入信号，shape (n_frames, n_joints) 或 (n_frames,)
        z_threshold: Z-score 阈值（通常 3.0 或 4.0），超过则为异常
        min_consecutive: 最小连续帧数
    
    Returns:
        spike_intervals: 尖峰区间列表 [(start, end), ...]
        spike_ratio: 尖峰帧占比
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    n_frames, n_joints = signal.shape
    
    # 计算每个关节的Z-score
    mean = np.nanmean(signal, axis=0, keepdims=True)
    std = np.nanstd(signal, axis=0, keepdims=True)
    
    # 避免除以零
    std = np.where(std == 0, 1e-6, std)
    
    z_scores = np.abs((signal - mean) / std)
    
    # 任意关节超过阈值则该帧标记为异常
    exceed_mask = z_scores > z_threshold
    exceed_any = np.any(exceed_mask, axis=1)
    
    # 找连续区间
    spike_intervals = []
    in_spike = False
    start_idx = 0
    
    for i in range(len(exceed_any)):
        if exceed_any[i] and not in_spike:
            start_idx = i
            in_spike = True
        elif not exceed_any[i] and in_spike:
            if i - start_idx >= min_consecutive:
                spike_intervals.append((start_idx, i - 1))
            in_spike = False
    
    # 处理末尾情况
    if in_spike and len(exceed_any) - start_idx >= min_consecutive:
        spike_intervals.append((start_idx, len(exceed_any) - 1))
    
    # 计算占比
    total_spike_frames = sum(end - start + 1 for start, end in spike_intervals)
    spike_ratio = total_spike_frames / len(exceed_any) if len(exceed_any) > 0 else 0.0
    
    return spike_intervals, spike_ratio


def compute_jerk(signal: np.ndarray, dt: float) -> np.ndarray:
    """
    计算加加速度 (Jerk = 三阶导数)
    
    Args:
        signal: 输入信号，shape (n_frames, n_joints) 或 (n_frames,)
        dt: 时间步长
    
    Returns:
        jerk 数组
    """
    # 一阶导数（速度）
    first_deriv = np.gradient(signal, dt, axis=0)
    
    # 二阶导数（加速度）
    second_deriv = np.gradient(first_deriv, dt, axis=0)
    
    # 三阶导数（加加速度/Jerk）
    jerk = np.gradient(second_deriv, dt, axis=0)
    
    return jerk


def detect_smoothness_issues(signal: np.ndarray, 
                            dt: float,
                            jerk_z_threshold: float = 3.0,
                            min_consecutive: int = 2) -> Dict[str, any]:
    """
    基于 Jerk 的平顺度检测（黄金指标）
    
    Args:
        signal: 输入信号，shape (n_frames, n_joints) 或 (n_frames,)
        dt: 时间步长
        jerk_z_threshold: Jerk Z-score 阈值
        min_consecutive: 最小连续帧数
    
    Returns:
        {
            'jerk_intervals': [(start, end), ...],
            'jerk_ratio': float,
            'jerk_stats': {
                'mean': float,
                'std': float,
                'max': float,
                'p95': float
            }
        }
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    # 计算 Jerk
    jerk = compute_jerk(signal, dt)
    jerk_abs = np.abs(jerk)
    
    # 统计信息
    jerk_mean = np.nanmean(jerk_abs)
    jerk_std = np.nanstd(jerk_abs)
    jerk_max = np.nanmax(jerk_abs)
    jerk_p95 = np.nanpercentile(jerk_abs, 95)
    
    # 基于Z-score检测异常Jerk
    mean_per_joint = np.nanmean(jerk_abs, axis=0, keepdims=True)
    std_per_joint = np.nanstd(jerk_abs, axis=0, keepdims=True)
    std_per_joint = np.where(std_per_joint == 0, 1e-6, std_per_joint)
    
    z_scores = (jerk_abs - mean_per_joint) / std_per_joint
    exceed_mask = z_scores > jerk_z_threshold
    exceed_any = np.any(exceed_mask, axis=1)
    
    # 找连续区间
    jerk_intervals = []
    in_spike = False
    start_idx = 0
    
    for i in range(len(exceed_any)):
        if exceed_any[i] and not in_spike:
            start_idx = i
            in_spike = True
        elif not exceed_any[i] and in_spike:
            if i - start_idx >= min_consecutive:
                jerk_intervals.append((start_idx, i - 1))
            in_spike = False
    
    if in_spike and len(exceed_any) - start_idx >= min_consecutive:
        jerk_intervals.append((start_idx, len(exceed_any) - 1))
    
    # 计算占比
    total_jerk_frames = sum(end - start + 1 for start, end in jerk_intervals)
    jerk_ratio = total_jerk_frames / len(exceed_any) if len(exceed_any) > 0 else 0.0
    
    return {
        'jerk_intervals': jerk_intervals,
        'jerk_ratio': float(jerk_ratio),
        'jerk_stats': {
            'mean': float(jerk_mean),
            'std': float(jerk_std),
            'max': float(jerk_max),
            'p95': float(jerk_p95)
        }
    }


def estimate_spike_thresholds(signals: List[np.ndarray], 
                             dt: float,
                             method: str = 'percentile',
                             percentile: float = 95,
                             safety_margin: float = 1.2,
                             mad_k: float = 3.0) -> Dict[str, float]:
    """
    估计尖峰检测阈值（保持原有接口兼容）
    
    Args:
        signals: 多个信号数组列表 (来自不同关节或episode)
        dt: 时间步长
        method: 估计方法 ('percentile' 或 'mad')
        percentile: 百分位数 (仅用于 percentile 方法)
        safety_margin: 安全系数
        mad_k: MAD 倍数 (仅用于 mad 方法)
    
    Returns:
        {'dtheta_thr': float, 'd2theta_thr': float}
    """
    velocities = []
    accelerations = []
    
    for signal in signals:
        if signal is None or len(signal) == 0:
            continue
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        vel = compute_derivatives(signal, dt, order=1)
        acc = compute_derivatives(signal, dt, order=2)
        
        velocities.append(np.abs(vel))
        accelerations.append(np.abs(acc))
    
    if not velocities:
        return {'dtheta_thr': 0.5, 'd2theta_thr': 0.3}
    
    all_vel = np.concatenate(velocities)
    all_acc = np.concatenate(accelerations)
    
    if method == 'percentile':
        vel_thr = np.percentile(all_vel, percentile) * safety_margin
        acc_thr = np.percentile(all_acc, percentile) * safety_margin
    elif method == 'mad':
        vel_median = np.median(all_vel)
        vel_mad = np.median(np.abs(all_vel - vel_median))
        vel_thr = vel_median + mad_k * vel_mad
        
        acc_median = np.median(all_acc)
        acc_mad = np.median(np.abs(all_acc - acc_median))
        acc_thr = acc_median + mad_k * acc_mad
    else:
        return {'dtheta_thr': 0.5, 'd2theta_thr': 0.3}
    
    return {
        'dtheta_thr': float(vel_thr),
        'd2theta_thr': float(acc_thr)
    }
