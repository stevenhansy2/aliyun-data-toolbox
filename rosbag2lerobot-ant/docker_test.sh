#!/bin/bash
# Docker环境中的性能测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 参数处理
MODE=${1:-all} # all, baseline, optimized
MEMORY_MONITOR=${2:-true}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ROSBag转LeRobot性能测试${NC}"
echo -e "${GREEN}========================================${NC}"

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate kuavo_il

echo -e "${BLUE}环境信息:${NC}"
python --version
echo "NumPy版本: $(python -c 'import numpy; print(numpy.__version__)')"
echo "Drake版本: $(python -c 'import pydrake; print(pydrake.__version__)' 2>/dev/null || echo '未安装')"
echo "工作目录: $(pwd)"
echo ""

# 测试参数
BAG_DIR="/data/bagdir/长三角一体化示范区智能机器人训练中心-A02-A04-yangli-2-112-yangli2-dex_hand-20251112170337-v1.bag"
MOMENT_JSON="$BAG_DIR/moments.json"
METADATA_JSON="$BAG_DIR/metadata.json"

# 清理输出
rm -rf /test_output/*

# 改进的内存监控函数 - 监控Python进程
monitor_memory_python() {
	local log_file=$1
	local python_script=$2

	echo "timestamp,rss_mb,vsz_mb,cpu_percent,python_pid" >"$log_file"

	# 等待Python进程启动
	sleep 2

	while true; do
		# 找到正在运行的Python进程（排除监控脚本自己）
		python_pid=$(ps aux | grep "python.*master_generate_lerobot.py" | grep -v grep | awk '{print $2}' | head -1)

		if [ -z "$python_pid" ]; then
			# Python进程未启动或已结束
			sleep 0.5
			continue
		fi

		# 检查进程是否还存在
		if ! kill -0 $python_pid 2>/dev/null; then
			break
		fi

		# 获取内存信息
		mem_info=$(ps -p $python_pid -o rss=,vsz=,pcpu= 2>/dev/null || echo "")

		if [ -n "$mem_info" ]; then
			rss=$(echo $mem_info | awk '{print $1/1024}') # KB to MB
			vsz=$(echo $mem_info | awk '{print $2/1024}') # KB to MB
			cpu=$(echo $mem_info | awk '{print $3}')
			echo "$(date +%s),$rss,$vsz,$cpu,$python_pid" >>"$log_file"

			# 实时显示（覆盖同一行）
			printf "\r${BLUE}[监控]${NC} PID:$python_pid  内存:${GREEN}%.0f MB${NC}  CPU:${YELLOW}%.1f%%${NC}  " "$rss" "$cpu"
		fi

		sleep 1
	done

	echo "" # 换行
	echo -e "${GREEN}内存监控结束${NC}"
}

# 生成内存使用报告
generate_memory_report() {
	local csv_file=$1
	local output_file=$2

	if [ ! -f "$csv_file" ] || [ $(wc -l <"$csv_file") -le 1 ]; then
		echo "无有效内存数据" >"$output_file"
		return
	fi

	# 使用awk处理CSV并生成报告
	awk -F',' '
    NR>1 {
        if (NR==2) {
            min_rss=max_rss=$2
            min_vsz=max_vsz=$3
            sum_rss=$2
            sum_vsz=$3
            sum_cpu=$4
            count=1
            max_rss_time=$1
        } else {
            if ($2 < min_rss) min_rss=$2
            if ($2 > max_rss) {
                max_rss=$2
                max_rss_time=$1
            }
            if ($3 < min_vsz) min_vsz=$3
            if ($3 > max_vsz) max_vsz=$3
            sum_rss+=$2
            sum_vsz+=$3
            sum_cpu+=$4
            count++
        }
    }
    END {
        if (count > 0) {
            print "=== 内存使用统计 ==="
            printf "采样次数: %d\n", count
            printf "持续时间: %d 秒\n\n", count
            
            printf "RSS (物理内存):\n"
            printf "  最小值: %.1f MB\n", min_rss
            printf "  最大值: %.1f MB (峰值)\n", max_rss
            printf "  平均值: %.1f MB\n", sum_rss/count
            
            printf "\nVSZ (虚拟内存):\n"
            printf "  最小值: %.1f MB\n", min_vsz
            printf "  最大值: %.1f MB\n", max_vsz
            printf "  平均值: %.1f MB\n", sum_vsz/count
            
            printf "\nCPU使用:\n"
            printf "  平均值: %.1f%%\n", sum_cpu/count
            
            printf "\n峰值内存占用: %.1f MB\n", max_rss
            printf "峰值出现时间: %d\n", max_rss_time
        }
    }' "$csv_file" >"$output_file"
}

#==============================================================================
# Baseline测试（如果需要）
#==============================================================================
if [ "$MODE" = "all" ] || [ "$MODE" = "baseline" ]; then
	echo -e "\n${MAGENTA}========================================${NC}"
	echo -e "${MAGENTA}[Baseline] 当前版本测试${NC}"
	echo -e "${MAGENTA}========================================${NC}"

	START_TIME=$(date +%s)

	# 启动内存监控（后台）
	if [ "$MEMORY_MONITOR" = "true" ]; then
		echo -e "${BLUE}启动内存监控...${NC}"
		monitor_memory_python /test_output/baseline_memory.csv "master_generate_lerobot.py" &
		MONITOR_PID=$!
	fi

	# 启动转换进程
	echo -e "${YELLOW}开始转换...${NC}"
	python kuavo/master_generate_lerobot.py \
		--bag_dir "$BAG_DIR" \
		--moment_json_dir "$MOMENT_JSON" \
		--metadata_json_dir "$METADATA_JSON" \
		--output_dir "/test_output/baseline" \
		--train_frequency 30 \
		--only_arm false \
		--which_arm both 2>&1 | tee /test_output/baseline.log

	EXIT_CODE=$?

	# 等待内存监控结束
	if [ "$MEMORY_MONITOR" = "true" ] && [ -n "$MONITOR_PID" ]; then
		sleep 2 # 确保最后的数据被写入
		kill $MONITOR_PID 2>/dev/null || true
		wait $MONITOR_PID 2>/dev/null || true
	fi

	END_TIME=$(date +%s)
	DURATION_BASELINE=$((END_TIME - START_TIME))

	if [ $EXIT_CODE -eq 0 ]; then
		echo -e "\n${GREEN}✓ Baseline测试完成 (耗时: ${DURATION_BASELINE}秒)${NC}"
	else
		echo -e "\n${RED}✗ Baseline测试失败 (退出码: $EXIT_CODE)${NC}"
	fi

	# 生成内存报告
	if [ "$MEMORY_MONITOR" = "true" ]; then
		echo -e "${BLUE}生成内存报告...${NC}"
		generate_memory_report /test_output/baseline_memory.csv /test_output/baseline_memory_report.txt
		echo ""
		cat /test_output/baseline_memory_report.txt
	fi
fi

#==============================================================================
# 优化版本测试
#==============================================================================
if [ "$MODE" = "all" ] || [ "$MODE" = "optimized" ]; then
	echo -e "\n${YELLOW}========================================${NC}"
	echo -e "${YELLOW}[优化版] 流式处理测试${NC}"
	echo -e "${YELLOW}========================================${NC}"

	START_TIME=$(date +%s)

	# 启动内存监控（后台）
	if [ "$MEMORY_MONITOR" = "true" ]; then
		echo -e "${BLUE}启动内存监控...${NC}"
		monitor_memory_python /test_output/optimized_memory.csv "master_generate_lerobot.py" &
		MONITOR_PID=$!
	fi

	# 启动转换进程
	echo -e "${YELLOW}开始转换...${NC}"
	python kuavo/master_generate_lerobot.py \
		--bag_dir "$BAG_DIR" \
		--moment_json_dir "$MOMENT_JSON" \
		--metadata_json_dir "$METADATA_JSON" \
		--output_dir "/test_output/optimized" \
		--train_frequency 30 \
		--only_arm false \
		--which_arm both 2>&1 | tee /test_output/optimized.log

	EXIT_CODE=$?

	# 等待内存监控结束
	if [ "$MEMORY_MONITOR" = "true" ] && [ -n "$MONITOR_PID" ]; then
		sleep 2
		kill $MONITOR_PID 2>/dev/null || true
		wait $MONITOR_PID 2>/dev/null || true
	fi

	END_TIME=$(date +%s)
	DURATION_OPTIMIZED=$((END_TIME - START_TIME))

	if [ $EXIT_CODE -eq 0 ]; then
		echo -e "\n${GREEN}✓ 优化版测试完成 (耗时: ${DURATION_OPTIMIZED}秒)${NC}"
	else
		echo -e "\n${RED}✗ 优化版测试失败 (退出码: $EXIT_CODE)${NC}"
	fi

	# 生成内存报告
	if [ "$MEMORY_MONITOR" = "true" ]; then
		echo -e "${BLUE}生成内存报告...${NC}"
		generate_memory_report /test_output/optimized_memory.csv /test_output/optimized_memory_report.txt
		echo ""
		cat /test_output/optimized_memory_report.txt
	fi
fi

#==============================================================================
# 输出统计
#==============================================================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  测试结果统计${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查输出文件
for mode in baseline optimized; do
	dir="/test_output/$mode"
	if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
		# 统计文件
		file_count=$(find "$dir" -type f | wc -l | tr -d ' ')
		total_size=$(du -sh "$dir" 2>/dev/null | cut -f1)

		# 统计特定类型文件
		hdf5_count=$(find "$dir" -name "*.hdf5" 2>/dev/null | wc -l | tr -d ' ')
		json_count=$(find "$dir" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
		video_count=$(find "$dir" -name "*.mp4" -o -name "*.mkv" 2>/dev/null | wc -l | tr -d ' ')
		parquet_count=$(find "$dir" -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')

		echo -e "\n${MAGENTA}$mode:${NC}"
		echo -e "  文件总数: ${GREEN}$file_count${NC}"
		echo -e "  总大小: ${GREEN}$total_size${NC}"
		echo -e "  详细统计:"
		echo -e "    - HDF5文件: ${GREEN}$hdf5_count${NC}"
		echo -e "    - JSON文件: ${GREEN}$json_count${NC}"
		echo -e "    - 视频文件: ${GREEN}$video_count${NC}"
		echo -e "    - Parquet文件: ${GREEN}$parquet_count${NC}"

		# 显示内存统计
		if [ -f "/test_output/${mode}_memory_report.txt" ]; then
			echo -e "  ${BLUE}内存统计:${NC}"
			grep "峰值内存占用" "/test_output/${mode}_memory_report.txt" | sed 's/^/    /'
			grep "平均值.*MB$" "/test_output/${mode}_memory_report.txt" | head -1 | sed 's/^/    /'
		fi
	fi
done

# 处理时间
if [ "$MODE" = "all" ] || [ "$MODE" = "baseline" ]; then
	echo -e "\n${BLUE}处理时间:${NC}"
	if [ -n "$DURATION_BASELINE" ] && ((DURATION_BASELINE > 0)); then
		echo -e "  Baseline: ${GREEN}${DURATION_BASELINE}秒${NC}"
	fi
fi

# 错误检查
echo -e "\n${BLUE}错误检查:${NC}"
for log in baseline.log optimized.log; do
	if [ -f "/test_output/$log" ]; then
		name=$(echo $log | sed 's/.log//')
		error_count=$(grep -c "Error\|Exception\|Traceback" "/test_output/$log" 2>/dev/null || echo "0")
		if ((error_count > 0)); then
			echo -e "  $name: ${RED}发现 $error_count 个错误${NC}"
			# 显示第一个错误
			grep -m1 -A3 "Error\|Exception" "/test_output/$log" | sed 's/^/    /'
		else
			echo -e "  $name: ${GREEN}无错误${NC}"
		fi
	fi
done

echo -e "\n${GREEN}测试完成！${NC}"
echo -e "详细日志: /test_output/*.log"
echo -e "内存数据: /test_output/*_memory.csv"
echo -e "内存报告: /test_output/*_memory_report.txt"
