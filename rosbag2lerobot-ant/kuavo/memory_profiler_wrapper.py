#!/usr/bin/env python3
"""
内存分析包装器 - 使用Python内置tracemalloc精确追踪内存使用

使用方法:
    python memory_profiler_wrapper.py master_generate_lerobot.py --bag_dir ... --output_dir ...
"""

import sys
import os
import time
import tracemalloc
import psutil
import threading
import json
from datetime import datetime


class MemoryProfiler:
    """实时内存监控器"""

    def __init__(self, output_file="memory_profile.json", interval=1.0):
        self.output_file = output_file
        self.interval = interval
        self.running = False
        self.samples = []
        self.start_time = None
        self.process = psutil.Process()

    def start(self):
        """启动内存监控"""
        print(f"[MemProfiler] 启动内存监控，采样间隔: {self.interval}秒")
        print(f"[MemProfiler] 输出文件: {self.output_file}")

        # 启动tracemalloc
        tracemalloc.start()

        self.start_time = time.time()
        self.running = True

        # 在后台线程中持续采样
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        """监控循环"""
        peak_memory = 0

        while self.running:
            try:
                # 获取tracemalloc数据
                current, peak = tracemalloc.get_traced_memory()

                # 获取进程级内存数据
                mem_info = self.process.memory_info()

                # 获取CPU使用率
                cpu_percent = self.process.cpu_percent(interval=0.1)

                sample = {
                    "timestamp": time.time() - self.start_time,
                    "tracemalloc_current_mb": current / 1024 / 1024,
                    "tracemalloc_peak_mb": peak / 1024 / 1024,
                    "rss_mb": mem_info.rss / 1024 / 1024,
                    "vms_mb": mem_info.vms / 1024 / 1024,
                    "cpu_percent": cpu_percent,
                }

                self.samples.append(sample)

                # 更新峰值
                if mem_info.rss > peak_memory:
                    peak_memory = mem_info.rss

                # 实时输出
                print(
                    f"\r[MemProfiler] 时间:{sample['timestamp']:.1f}s "
                    f"RSS:{sample['rss_mb']:.1f}MB "
                    f"Peak:{sample['tracemalloc_peak_mb']:.1f}MB "
                    f"CPU:{cpu_percent:.1f}%",
                    end="",
                    flush=True,
                )

            except Exception as e:
                print(f"\n[MemProfiler] 监控错误: {e}")

            time.sleep(self.interval)

    def stop(self):
        """停止监控并保存数据"""
        print("\n[MemProfiler] 停止内存监控...")
        self.running = False

        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=2)

        # 获取最终快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # 停止tracemalloc
        tracemalloc.stop()

        # 生成报告
        self._generate_report(top_stats)

    def _generate_report(self, top_stats):
        """生成内存分析报告"""
        if not self.samples:
            print("[MemProfiler] 无采样数据")
            return

        # 计算统计信息
        rss_values = [s["rss_mb"] for s in self.samples]
        tracemalloc_values = [s["tracemalloc_current_mb"] for s in self.samples]

        report = {
            "summary": {
                "total_duration_seconds": self.samples[-1]["timestamp"],
                "sample_count": len(self.samples),
                "rss_memory": {
                    "min_mb": min(rss_values),
                    "max_mb": max(rss_values),
                    "avg_mb": sum(rss_values) / len(rss_values),
                },
                "tracemalloc_memory": {
                    "min_mb": min(tracemalloc_values),
                    "max_mb": max(tracemalloc_values),
                    "avg_mb": sum(tracemalloc_values) / len(tracemalloc_values),
                },
            },
            "top_memory_allocations": [
                {
                    "file": str(stat.traceback[0].filename),
                    "line": stat.traceback[0].lineno,
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
                for stat in top_stats[:20]  # 前20个最大分配
            ],
            "samples": self.samples,
        }

        # 保存JSON报告
        with open(self.output_file, "w") as f:
            json.dump(report, f, indent=2)

        # 保存CSV格式（兼容原有脚本）
        csv_file = self.output_file.replace(".json", ".csv")
        with open(csv_file, "w") as f:
            f.write("timestamp,rss_mb,vms_mb,cpu_percent,tracemalloc_mb\n")
            for s in self.samples:
                f.write(
                    f"{s['timestamp']:.1f},{s['rss_mb']:.1f},"
                    f"{s['vms_mb']:.1f},{s['cpu_percent']:.1f},"
                    f"{s['tracemalloc_current_mb']:.1f}\n"
                )

        # 打印摘要
        print(f"\n[MemProfiler] ===== 内存分析报告 =====")
        print(f"持续时间: {report['summary']['total_duration_seconds']:.1f}秒")
        print(f"采样次数: {report['summary']['sample_count']}")
        print(f"\nRSS内存 (进程实际物理内存):")
        print(f"  最小值: {report['summary']['rss_memory']['min_mb']:.1f} MB")
        print(f"  最大值: {report['summary']['rss_memory']['max_mb']:.1f} MB (峰值)")
        print(f"  平均值: {report['summary']['rss_memory']['avg_mb']:.1f} MB")
        print(f"\nTracemalloc (Python对象内存):")
        print(f"  最小值: {report['summary']['tracemalloc_memory']['min_mb']:.1f} MB")
        print(f"  最大值: {report['summary']['tracemalloc_memory']['max_mb']:.1f} MB")
        print(f"  平均值: {report['summary']['tracemalloc_memory']['avg_mb']:.1f} MB")
        print(f"\n前5个最大内存分配:")
        for i, alloc in enumerate(report["top_memory_allocations"][:5], 1):
            print(
                f"  {i}. {alloc['file']}:{alloc['line']} - "
                f"{alloc['size_mb']:.1f} MB ({alloc['count']} 个对象)"
            )
        print(f"\n详细报告已保存: {self.output_file}")
        print(f"CSV数据已保存: {csv_file}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python memory_profiler_wrapper.py <script.py> [args...]")
        print(
            "示例: python memory_profiler_wrapper.py master_generate_lerobot.py --bag_dir /data/..."
        )
        sys.exit(1)

    script_path = sys.argv[1]
    script_args = sys.argv[2:]

    if not os.path.exists(script_path):
        print(f"错误: 找不到脚本 {script_path}")
        sys.exit(1)

    # 创建输出文件名
    output_file = f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 创建内存分析器
    profiler = MemoryProfiler(output_file=output_file, interval=1.0)

    print(f"[MemProfiler] 准备运行: {script_path}")
    print(f"[MemProfiler] 参数: {' '.join(script_args)}")
    print("")

    # 启动监控
    profiler.start()

    # 运行目标脚本
    exit_code = 0
    try:
        # 修改sys.argv让目标脚本看到正确的参数
        sys.argv = [script_path] + script_args

        # 执行脚本
        with open(script_path) as f:
            code = compile(f.read(), script_path, "exec")
            exec(code, {"__name__": "__main__", "__file__": script_path})

    except SystemExit as e:
        exit_code = e.code if e.code else 0
    except Exception as e:
        print(f"\n[MemProfiler] 脚本执行错误: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # 停止监控
        profiler.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
