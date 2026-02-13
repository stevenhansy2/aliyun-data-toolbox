#!/bin/bash
# 监控Docker容器内存使用的脚本

CONTAINER_NAME=${1:-rosbag2lerobot_test}
OUTPUT_FILE=${2:-memory_usage.log}
INTERVAL=${3:-1} # 默认1秒采样一次

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}开始监控容器 $CONTAINER_NAME 的内存使用${NC}"
echo "输出文件: $OUTPUT_FILE"
echo "采样间隔: ${INTERVAL}秒"
echo ""

# 检查容器是否运行
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
	echo -e "${RED}错误: 容器 $CONTAINER_NAME 未运行${NC}"
	exit 1
fi

# 写入CSV头
echo "timestamp,container,mem_usage_mb,mem_limit_mb,mem_percent,cpu_percent" >"$OUTPUT_FILE"

# 获取系统信息
echo -e "${BLUE}系统信息:${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
	echo "操作系统: macOS"
	echo "总内存: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}') GB"
else
	echo "操作系统: Linux"
	echo "总内存: $(free -h | grep Mem | awk '{print $2}')"
fi
echo ""

echo -e "${YELLOW}开始监控... (按 Ctrl+C 停止)${NC}"

# 监控循环
while true; do
	# 获取容器统计信息
	stats=$(docker stats --no-stream --format "{{.Container}},{{.MemUsage}},{{.MemPerc}},{{.CPUPerc}}" "$CONTAINER_NAME" 2>/dev/null)

	if [ -z "$stats" ]; then
		echo -e "${RED}容器已停止运行${NC}"
		break
	fi

	# 解析内存使用
	container=$(echo "$stats" | cut -d',' -f1)
	mem_usage_raw=$(echo "$stats" | cut -d',' -f2)
	mem_percent=$(echo "$stats" | cut -d',' -f3 | tr -d '%')
	cpu_percent=$(echo "$stats" | cut -d',' -f4 | tr -d '%')

	# 提取内存数值（处理 "1.5GiB / 2GiB" 格式）
	mem_usage=$(echo "$mem_usage_raw" | awk '{print $1}' | sed 's/GiB/*1024/; s/MiB//; s/KiB/\/1024/' | bc 2>/dev/null || echo "0")
	mem_limit=$(echo "$mem_usage_raw" | awk '{print $3}' | sed 's/GiB/*1024/; s/MiB//; s/KiB/\/1024/' | bc 2>/dev/null || echo "0")

	# 写入数据
	timestamp=$(date +%s)
	echo "$timestamp,$container,$mem_usage,$mem_limit,$mem_percent,$cpu_percent" >>"$OUTPUT_FILE"

	# 显示当前状态
	printf "\r内存: ${GREEN}%.1f MB${NC} / %.1f MB (${YELLOW}%.1f%%${NC})  CPU: ${BLUE}%.1f%%${NC}  " \
		"$mem_usage" "$mem_limit" "$mem_percent" "$cpu_percent"

	sleep "$INTERVAL"
done

echo -e "\n\n${GREEN}监控结束${NC}"

# 生成统计报告
if [ -f "$OUTPUT_FILE" ] && [ $(wc -l <"$OUTPUT_FILE") -gt 1 ]; then
	echo -e "\n${BLUE}=== 内存使用统计报告 ===${NC}"

	# 使用awk计算统计数据
	awk -F',' '
    NR>1 {
        if (NR==2) {
            min_mem=max_mem=$3
            sum_mem=$3
            sum_cpu=$6
            count=1
        } else {
            if ($3 < min_mem) min_mem=$3
            if ($3 > max_mem) max_mem=$3
            sum_mem+=$3
            sum_cpu+=$6
            count++
        }
    }
    END {
        if (count > 0) {
            printf "采样次数: %d\n", count
            printf "持续时间: %d 秒\n", count * '"$INTERVAL"'
            printf "\n内存使用:\n"
            printf "  最小值: %.1f MB\n", min_mem
            printf "  最大值: %.1f MB (峰值)\n", max_mem
            printf "  平均值: %.1f MB\n", sum_mem/count
            printf "\nCPU使用:\n"
            printf "  平均值: %.1f%%\n", sum_cpu/count
        }
    }' "$OUTPUT_FILE"

	echo -e "\n详细数据已保存到: ${GREEN}$OUTPUT_FILE${NC}"
fi
