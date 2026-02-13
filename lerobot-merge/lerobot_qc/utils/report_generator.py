"""
报告生成器
支持 JSON 和 Markdown 格式的报告输出
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, results: Dict[str, Any], formats: List[str] = ['json', 'markdown']):
        """
        生成检测报告

        Args:
            results: 检测结果字典
            formats: 输出格式列表，可包含 'json' 和/或 'markdown'
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if 'json' in formats:
            json_path = self.output_dir / f'report_{timestamp}.json'
            self._generate_json_report(results, json_path)

        if 'markdown' in formats:
            md_path = self.output_dir / f'report_{timestamp}.md'
            self._generate_markdown_report(results, md_path)

    def _generate_json_report(self, results: Dict[str, Any], output_path: Path):
        """生成 JSON 格式报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSON 报告已保存到: {output_path}")

    def _generate_markdown_report(self, results: Dict[str, Any], output_path: Path):
        """生成 Markdown 格式报告"""
        lines = []

        # 标题
        lines.append("# LeRobot 数据集质量检测报告")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**数据集路径**: {results.get('dataset_path', 'N/A')}")
        lines.append("")

        # 总体结果
        overall_passed = results.get('overall_passed', False)
        status_emoji = "✅" if overall_passed else "❌"
        lines.append(f"## 总体结果: {status_emoji} {'通过' if overall_passed else '未通过'}")
        lines.append("")

        # Level 0 检查结果（如果存在）
        if 'level0_validation' in results:
            lines.append("---")
            lines.append("## Level 0 检查：LeRobot 格式验证")
            lines.append("")

            level0_results = results['level0_validation']
            level0_passed = level0_results.get('passed', False)

            lines.append(f"**状态**: {'✅ 通过' if level0_passed else '❌ 未通过（一票否决）'}")
            lines.append("")

            if level0_results.get('can_load'):
                lines.append("**数据集信息**:")
                lines.append(f"  - 总帧数: {level0_results.get('total_frames', 0)}")
                lines.append(f"  - FPS: {level0_results.get('fps', 0)}")
                lines.append(f"  - 总 Episodes: {level0_results.get('total_episodes', 0)}")
                lines.append("")

            if level0_results.get('errors'):
                lines.append("**错误**:")
                for error in level0_results['errors']:
                    lines.append(f"  - {error}")
                lines.append("")

            if level0_results.get('warnings'):
                lines.append("**警告**:")
                for warning in level0_results['warnings']:
                    lines.append(f"  - {warning}")
                lines.append("")

        # A 级检查结果
        lines.append("---")
        lines.append("## A 级检查：数据集级别")
        lines.append("")

        dataset_results = results.get('dataset_validation', {})

        # A1
        lines.append("### A1. 数据结构与维度一致性")
        a1_result = dataset_results.get('A1_structure_check', {})
        a1_passed = a1_result.get('passed', False)
        lines.append(f"**状态**: {'✅ 通过' if a1_passed else '❌ 未通过'}")
        lines.append("")

        if not a1_passed:
            if 'error' in a1_result:
                lines.append(f"**错误**: {a1_result['error']}")
            else:
                if a1_result.get('missing_keys'):
                    lines.append("**缺失的 keys**:")
                    for key in a1_result['missing_keys']:
                        lines.append(f"  - `{key}`")
                    lines.append("")

                if a1_result.get('type_mismatches'):
                    lines.append("**类型不匹配**:")
                    lines.append("| Key | 期望类型 | 实际类型 |")
                    lines.append("|-----|----------|----------|")
                    for mismatch in a1_result['type_mismatches']:
                        lines.append(f"| `{mismatch['key']}` | {mismatch['expected_dtype']} | {mismatch['actual_dtype']} |")
                    lines.append("")

                if a1_result.get('dim_mismatches'):
                    lines.append("**维度不匹配**:")
                    lines.append("| Key | 期望维度 | 实际维度 |")
                    lines.append("|-----|----------|----------|")
                    for mismatch in a1_result['dim_mismatches']:
                        lines.append(f"| `{mismatch['key']}` | {mismatch['expected_dim']} | {mismatch['actual_dim']} |")
                    lines.append("")

        # A2
        lines.append("### A2. 统计信息完整性检查")
        a2_result = dataset_results.get('A2_stats_check', {})
        a2_passed = a2_result.get('passed', False)
        lines.append(f"**状态**: {'✅ 通过' if a2_passed else '❌ 未通过'}")
        lines.append("")

        if not a2_passed:
            if 'error' in a2_result:
                lines.append(f"**错误**: {a2_result['error']}")
            else:
                anomalies = a2_result.get('anomalies', [])
                if anomalies:
                    lines.append(f"**发现 {len(anomalies)} 个字段存在异常**:")
                    lines.append("")
                    for anomaly in anomalies:
                        lines.append(f"- **{anomaly['field']}**:")
                        for issue in anomaly['issues']:
                            lines.append(f"  - {issue}")
                    lines.append("")

        # B 级检查结果
        lines.append("---")
        lines.append("## B 级检查：Episode 级别")
        lines.append("")

        episode_results = results.get('episode_validations', [])
        total_episodes = len(episode_results)
        passed_episodes = sum(1 for ep in episode_results if ep.get('passed', False))

        lines.append(f"**总 Episodes 数**: {total_episodes}")
        lines.append(f"**通过的 Episodes**: {passed_episodes}")
        lines.append(f"**未通过的 Episodes**: {total_episodes - passed_episodes}")
        lines.append("")

        # 汇总统计
        lines.append("### 汇总统计")
        lines.append("")

        # B1 汇总
        b1_stats = self._aggregate_b1_stats(episode_results)
        lines.append("#### B1. 异常静止检测")
        lines.append(f"- 平均静止帧占比: {b1_stats['avg_static_ratio']:.2%}")
        lines.append(f"- 最大静止帧占比: {b1_stats['max_static_ratio']:.2%}")
        lines.append(f"- 超过阈值的 Episodes: {b1_stats['episodes_exceeding_threshold']}/{total_episodes}")
        lines.append("")

        # B2 汇总
        b2_stats = self._aggregate_b2_stats(episode_results)
        lines.append("#### B2. 角度域与 Gripper 检查")
        if b2_stats['angle_domain_summary']:
            lines.append("**角度域分布**:")
            for domain, count in b2_stats['angle_domain_summary'].items():
                lines.append(f"  - {domain}: {count} 次")
        if b2_stats['gripper_issues']:
            lines.append(f"**Gripper 超出范围**: {b2_stats['gripper_issues']} 个 Episodes")
        lines.append("")

        # B3 汇总
        b3_stats = self._aggregate_b3_stats(episode_results)
        lines.append("#### B3. 原始信号异常检测")
        lines.append(f"- **失败的 Episodes**: {b3_stats['b3_failed_episodes']}/{total_episodes}")
        lines.append(f"- **有警告的 Episodes**: {b3_stats['b3_warning_episodes']}/{total_episodes}")
        lines.append("")
        lines.append("**离群值检测**:")
        lines.append(f"  - 平均角度离群值占比: {b3_stats['avg_outlier_ratio']:.2%}")
        lines.append(f"  - 平均速度离群值占比: {b3_stats['avg_vel_outlier_ratio']:.2%}")
        lines.append(f"  - 平均加速度离群值占比: {b3_stats['avg_acc_outlier_ratio']:.2%}")
        lines.append("")
        lines.append("**尖峰检测**:")
        lines.append(f"  - 平均一阶尖峰占比: {b3_stats['avg_spike_1st_ratio']:.2%}")
        lines.append(f"  - 平均二阶尖峰占比: {b3_stats['avg_spike_2nd_ratio']:.2%}")
        lines.append("")
        lines.append("**平滑度检测**:")
        lines.append(f"  - 平均Jerk异常占比: {b3_stats['avg_jerk_ratio']:.2%}")
        lines.append(f"  - 平均总变差: {b3_stats['avg_tv']:.2f} rad/s")
        lines.append(f"  - 最大总变差: {b3_stats['max_tv']:.2f} rad/s")
        lines.append(f"  - 不顺滑的 Episodes: {b3_stats['non_smooth_episodes']}/{total_episodes}")
        lines.append("")

        # B4 汇总
        b4_stats = self._aggregate_b4_stats(episode_results)
        lines.append("#### B4. 时间戳一致性检查")
        lines.append(f"- **失败的 Episodes**: {b4_stats['b4_failed_episodes']}/{total_episodes}")
        lines.append(f"- **非单调递增的 Episodes**: {b4_stats['non_monotonic_episodes']}/{total_episodes}")
        lines.append(f"- **FPS 不一致的 Episodes**: {b4_stats['fps_inconsistent_episodes']}/{total_episodes}")
        lines.append("")
        lines.append("**时间戳间隔统计**:")
        lines.append(f"  - 平均间隔: {b4_stats['avg_interval_ms']:.3f} ms")
        lines.append(f"  - 平均 FPS: {b4_stats['avg_fps']:.2f}")
        lines.append(f"  - 平均最大偏差: {b4_stats['avg_max_deviation_ms']:.4f} ms")
        lines.append(f"  - 平均超出容差帧占比: {b4_stats['avg_exceeds_ratio']:.2%}")
        lines.append("")

        # 详细的 Episode 报告（仅显示未通过的）
        failed_episodes = [ep for ep in episode_results if not ep.get('passed', False)]
        if failed_episodes:
            lines.append("---")
            lines.append("## 未通过的 Episodes 详情")
            lines.append("")

            for ep in failed_episodes[:10]:  # 最多显示 10 个
                ep_idx = ep.get('episode_idx', 'N/A')
                lines.append(f"### Episode {ep_idx}")
                lines.append("")

                errors = ep.get('errors', [])
                if errors:
                    lines.append("**错误**:")
                    for error in errors:
                        lines.append(f"  - {error}")
                    lines.append("")

                # B1 详情
                b1 = ep.get('B1_static_check', {})
                if not b1.get('passed', True):
                    lines.append(f"- **B1 静止检测**: 静止帧占比 {b1.get('static_ratio', 0):.2%}")

                # B2 详情
                b2 = ep.get('B2_angle_gripper_check', {})
                if not b2.get('passed', True):
                    gripper_results = b2.get('gripper_check_results', {})
                    for key, result in gripper_results.items():
                        if result.get('out_of_range_count', 0) > 0:
                            lines.append(f"- **B2 Gripper ({key})**: {result['out_of_range_count']} 个值超出范围")

                # B3 详情
                b3 = ep.get('B3_anomaly_check', {})
                if not b3.get('passed', True):
                    critical_issues = b3.get('critical_issues', [])
                    if critical_issues:
                        lines.append("- **B3 极端异常**:")
                        for issue in critical_issues:
                            lines.append(f"  - {issue}")

                # B3 警告（对于失败的episodes，显示前3个警告）
                warnings = b3.get('warnings', [])
                if warnings:
                    lines.append(f"- **B3 警告** ({len(warnings)} 个):")
                    for warning in warnings[:3]:
                        lines.append(f"  - {warning}")
                    if len(warnings) > 3:
                        lines.append(f"  - ... 还有 {len(warnings) - 3} 个警告")

                # B4 详情
                b4 = ep.get('B4_timestamp_check', {})
                if not b4.get('passed', True):
                    issues = b4.get('issues', [])
                    if issues:
                        lines.append("- **B4 时间戳问题**:")
                        for issue in issues:
                            lines.append(f"  - {issue}")

                    # 显示详细统计
                    fps_consistency = b4.get('fps_consistency', {})
                    if fps_consistency:
                        lines.append(f"  - 期望 FPS: {b4.get('expected_fps', 'N/A')}, 实际 FPS: {fps_consistency.get('actual_fps', 'N/A'):.2f}")
                        lines.append(f"  - 最大偏差: {fps_consistency.get('max_deviation_ms', 'N/A'):.4f} ms")
                        lines.append(f"  - 超出容差帧数: {fps_consistency.get('exceeds_tolerance_count', 0)}/{fps_consistency.get('total_intervals', 0)}")

                lines.append("")

            if len(failed_episodes) > 10:
                lines.append(f"*（还有 {len(failed_episodes) - 10} 个未通过的 Episodes 未显示）*")
                lines.append("")

        # B3 警告示例（显示一些有代表性的警告）
        warning_episodes = [ep for ep in episode_results if ep.get('passed', False) and ep.get('B3_anomaly_check', {}).get('warnings')]
        if warning_episodes:
            lines.append("---")
            lines.append("## B3 警告示例（通过但有警告的 Episodes）")
            lines.append("")
            lines.append(f"**说明**: 共有 {len(warning_episodes)} 个 Episodes 通过检测但存在 B3 警告")
            lines.append("")

            # 显示前3个有警告的episodes作为示例
            for ep in warning_episodes[:3]:
                ep_idx = ep.get('episode_idx', 'N/A')
                b3 = ep.get('B3_anomaly_check', {})
                warnings = b3.get('warnings', [])

                lines.append(f"### Episode {ep_idx}")
                lines.append(f"**警告数量**: {len(warnings)}")
                lines.append("")
                lines.append("**警告详情** (前5个):")
                for warning in warnings[:5]:
                    lines.append(f"  - {warning}")
                if len(warnings) > 5:
                    lines.append(f"  - ... 还有 {len(warnings) - 5} 个警告")
                lines.append("")

            if len(warning_episodes) > 3:
                lines.append(f"*（还有 {len(warning_episodes) - 3} 个有警告的 Episodes 未显示）*")
                lines.append("")

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Markdown 报告已保存到: {output_path}")

    def _aggregate_b1_stats(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总 B1 检查统计"""
        static_ratios = []
        exceeding_count = 0

        for ep in episode_results:
            b1 = ep.get('B1_static_check', {})
            ratio = b1.get('static_ratio', 0.0)
            static_ratios.append(ratio)

            threshold = b1.get('threshold', 0.01)
            if ratio > threshold:
                exceeding_count += 1

        return {
            'avg_static_ratio': sum(static_ratios) / len(static_ratios) if static_ratios else 0.0,
            'max_static_ratio': max(static_ratios) if static_ratios else 0.0,
            'episodes_exceeding_threshold': exceeding_count
        }

    def _aggregate_b2_stats(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总 B2 检查统计"""
        angle_domain_counts = {}
        gripper_issues_count = 0

        for ep in episode_results:
            b2 = ep.get('B2_angle_gripper_check', {})

            # 角度域统计
            angle_results = b2.get('angle_domain_results', {})
            for key, result in angle_results.items():
                domain = result.get('domain_type', 'unknown')
                angle_domain_counts[domain] = angle_domain_counts.get(domain, 0) + 1

            # Gripper 问题统计
            gripper_results = b2.get('gripper_check_results', {})
            for key, result in gripper_results.items():
                if result.get('out_of_range_count', 0) > 0:
                    gripper_issues_count += 1
                    break

        return {
            'angle_domain_summary': angle_domain_counts,
            'gripper_issues': gripper_issues_count
        }

    def _aggregate_b3_stats(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总 B3 检查统计"""
        outlier_ratios = []
        vel_outlier_ratios = []
        acc_outlier_ratios = []
        spike_1st_ratios = []
        spike_2nd_ratios = []
        jerk_ratios = []
        tv_values = []
        non_smooth_count = 0
        b3_failed_count = 0
        b3_warning_count = 0

        for ep in episode_results:
            b3 = ep.get('B3_anomaly_check', {})
            outlier_detection = b3.get('outlier_detection', {})

            # 统计失败和警告
            if not b3.get('passed', True):
                b3_failed_count += 1
            if b3.get('warnings'):
                b3_warning_count += 1

            for key, result in outlier_detection.items():
                if 'error' not in result:
                    outlier_ratios.append(result.get('outlier_ratio', 0.0))
                    vel_outlier_ratios.append(result.get('velocity_outlier_ratio', 0.0))
                    acc_outlier_ratios.append(result.get('acceleration_outlier_ratio', 0.0))

                    # 尖峰统计
                    spike_1st = result.get('first_derivative_spikes', {}).get('spike_ratio', 0.0)
                    spike_2nd = result.get('second_derivative_spikes', {}).get('spike_ratio', 0.0)
                    spike_1st_ratios.append(spike_1st)
                    spike_2nd_ratios.append(spike_2nd)

                    # Jerk统计
                    jerk_ratio = result.get('jerk_smoothness', {}).get('jerk_ratio', 0.0)
                    jerk_ratios.append(jerk_ratio)

                    # 总变差统计
                    tv = result.get('total_variation_per_sec', 0.0)
                    tv_values.append(tv)

                    if not result.get('is_smooth', True):
                        non_smooth_count += 1
                        break

        return {
            'avg_outlier_ratio': sum(outlier_ratios) / len(outlier_ratios) if outlier_ratios else 0.0,
            'avg_vel_outlier_ratio': sum(vel_outlier_ratios) / len(vel_outlier_ratios) if vel_outlier_ratios else 0.0,
            'avg_acc_outlier_ratio': sum(acc_outlier_ratios) / len(acc_outlier_ratios) if acc_outlier_ratios else 0.0,
            'avg_spike_1st_ratio': sum(spike_1st_ratios) / len(spike_1st_ratios) if spike_1st_ratios else 0.0,
            'avg_spike_2nd_ratio': sum(spike_2nd_ratios) / len(spike_2nd_ratios) if spike_2nd_ratios else 0.0,
            'avg_jerk_ratio': sum(jerk_ratios) / len(jerk_ratios) if jerk_ratios else 0.0,
            'avg_tv': sum(tv_values) / len(tv_values) if tv_values else 0.0,
            'max_tv': max(tv_values) if tv_values else 0.0,
            'non_smooth_episodes': non_smooth_count,
            'b3_failed_episodes': b3_failed_count,
            'b3_warning_episodes': b3_warning_count
        }

    def _aggregate_b4_stats(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总 B4 时间戳检查统计"""
        b4_failed_count = 0
        non_monotonic_count = 0
        fps_inconsistent_count = 0

        interval_means = []
        actual_fps_values = []
        max_deviations = []
        exceeds_ratios = []

        for ep in episode_results:
            b4 = ep.get('B4_timestamp_check', {})

            # 统计失败
            if not b4.get('passed', True):
                b4_failed_count += 1

            # 统计非单调递增
            if not b4.get('monotonic', True):
                non_monotonic_count += 1

            # 统计 FPS 不一致
            fps_consistency = b4.get('fps_consistency', {})
            if fps_consistency.get('exceeds_tolerance_count', 0) > 0:
                fps_inconsistent_count += 1

            # 收集统计数据
            interval_stats = b4.get('interval_stats', {})
            if interval_stats:
                interval_means.append(interval_stats.get('mean_ms', 0.0))

            if fps_consistency:
                actual_fps = fps_consistency.get('actual_fps', 0.0)
                if actual_fps > 0:
                    actual_fps_values.append(actual_fps)

                max_deviations.append(fps_consistency.get('max_deviation_ms', 0.0))

                total_intervals = fps_consistency.get('total_intervals', 0)
                if total_intervals > 0:
                    exceeds_count = fps_consistency.get('exceeds_tolerance_count', 0)
                    exceeds_ratios.append(exceeds_count / total_intervals)

        return {
            'b4_failed_episodes': b4_failed_count,
            'non_monotonic_episodes': non_monotonic_count,
            'fps_inconsistent_episodes': fps_inconsistent_count,
            'avg_interval_ms': sum(interval_means) / len(interval_means) if interval_means else 0.0,
            'avg_fps': sum(actual_fps_values) / len(actual_fps_values) if actual_fps_values else 0.0,
            'avg_max_deviation_ms': sum(max_deviations) / len(max_deviations) if max_deviations else 0.0,
            'avg_exceeds_ratio': sum(exceeds_ratios) / len(exceeds_ratios) if exceeds_ratios else 0.0
        }
