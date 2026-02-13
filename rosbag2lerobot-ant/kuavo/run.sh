#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
START=$(date +%s)
# SETUP
OUTPUT_DIR="temp"

##############################################
# 内存监控配置
##############################################
# 通过环境变量启用内存监控: export ENABLE_MEMORY_MONITOR=true
ENABLE_MEMORY_MONITOR=${ENABLE_MEMORY_MONITOR:-false}
MEMORY_LOG_CSV="${OUTPUT_DIR}/memory_usage.csv"
MEMORY_LOG_REPORT="${OUTPUT_DIR}/memory_report.txt"

##############################################
# 脚本版本选择
##############################################
# 通过环境变量选择脚本版本: export USE_STREAMING_VERSION=false
# 默认使用流式版本（_s），内存优化更好
USE_STREAMING_VERSION=${USE_STREAMING_VERSION:-true}

##############################################
# 强制重新转换配置
##############################################
# 通过环境变量强制重新转换已转换的记录: export FORCE_RECONVERT=true
# 默认为false，检测到lerobot_success标签会跳过转换
FORCE_RECONVERT=${FORCE_RECONVERT:-false}

##############################################
# 流水线视频编码配置
##############################################
# 通过环境变量启用流水线视频编码: export USE_PIPELINE_ENCODING=true
# 启用后，视频编码与批处理并行执行，（但是无法优化性能，因为需要拼接）
USE_PIPELINE_ENCODING=${USE_PIPELINE_ENCODING:-false}
export USE_PIPELINE_ENCODING

##############################################
# 流式视频编码配置
##############################################
# 通过环境变量启用流式视频编码: export USE_STREAMING_VIDEO=true
# 启用后，视频编码在批处理过程中实时进行，无需临时文件
USE_STREAMING_VIDEO=${USE_STREAMING_VIDEO:-false}
export USE_STREAMING_VIDEO

# 流式编码队列上限（控制内存使用）
VIDEO_QUEUE_LIMIT=${VIDEO_QUEUE_LIMIT:-100}
export VIDEO_QUEUE_LIMIT

##############################################
# 并行 ROSbag 读取配置
##############################################
# 通过环境变量启用并行读取: export USE_PARALLEL_ROSBAG_READ=true
# 启用后，使用多进程并行读取 ROSbag，可获得约 2x 加速
USE_PARALLEL_ROSBAG_READ=${USE_PARALLEL_ROSBAG_READ:-false}
export USE_PARALLEL_ROSBAG_READ

# 并行读取的 worker 数量（建议 2，更多收益递减）
PARALLEL_ROSBAG_WORKERS=${PARALLEL_ROSBAG_WORKERS:-2}
export PARALLEL_ROSBAG_WORKERS

if [ "$USE_STREAMING_VERSION" = "true" ]; then
    MASTER_SCRIPT="kuavo/master_generate_lerobot_s.py"
    echo "使用流式版本脚本 (master_generate_lerobot_s.py)"
else
    MASTER_SCRIPT="kuavo/master_generate_lerobot.py"
    echo "使用标准版本脚本 (master_generate_lerobot.py)"
fi

##############################################
# 颜色定义
##############################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'


echo "========== Step 1: 生成 ossutil 配置文件 =========="
cat > ~/.ossutilconfig <<EOF
[default]
accessKeyId=$accessKeyID
accessKeySecret=$accessKeySecret
region=cn-hangzhou
endpoint=$endpoint
EOF

##############################################
# 内存监控函数（监控本地Python进程）
##############################################
monitor_memory_python() {
	local log_file=$1
	local python_script=$2

	# 参数应该已经是绝对路径
	echo "[DEBUG] 监控函数接收路径: $log_file" >&2
	echo "[DEBUG] 监控函数工作目录: $(pwd)" >&2

	# 确保目录存在
	local log_dir=$(dirname "$log_file")
	mkdir -p "$log_dir" 2>/dev/null || true

	# 写入CSV头
	echo "timestamp,rss_mb,vsz_mb,cpu_percent,python_pid" >"$log_file" 2>&1

	if [ ! -f "$log_file" ]; then
		echo "[ERROR] 无法创建日志文件: $log_file" >&2
		echo "[ERROR] 目录是否存在: $(ls -ld "$log_dir" 2>&1)" >&2
		return 1
	fi

	echo "[DEBUG] CSV头已写入: $log_file" >&2

	# 等待Python进程启动
	sleep 2

	local first_log=true

	while true; do
		# 找到正在运行的Python进程（精确匹配 python3，排除 timeout）
		# 匹配 "python3 " 或 "python " 开头的，后面跟着脚本名
		PYTHON_PID=$(ps aux | grep -E "python3? .*$python_script" | grep -v grep | grep -v timeout | awk '{print $2}' | head -1)

		if [ -z "$PYTHON_PID" ]; then
			sleep 1
			continue
		fi

		# 第一次找到进程时，输出进程信息
		if [ "$first_log" = true ]; then
			echo "[DEBUG] 找到Python进程 PID: $PYTHON_PID" >&2
			PROCESS_CMD=$(ps -p "$PYTHON_PID" -o args= 2>/dev/null)
			echo "[DEBUG] 进程命令: ${PROCESS_CMD:0:120}" >&2

			# 验证这是正确的进程
			if echo "$PROCESS_CMD" | grep -q "python"; then
				echo "[DEBUG] ✅ 确认是 Python 进程" >&2
			else
				echo "[WARN] ⚠️ 这可能不是 Python 进程！" >&2
			fi

			first_log=false
		fi

	# 获取内存和CPU使用情况（包括子进程）
	# 主进程
	STATS=$(ps -p "$PYTHON_PID" -o rss=,vsz=,%cpu= 2>/dev/null || echo "")

	if [ -z "$STATS" ]; then
		# 进程已结束
		break
	fi

	RSS_KB=$(echo "$STATS" | awk '{print $1}')
	VSZ_KB=$(echo "$STATS" | awk '{print $2}')
	CPU_PERCENT=$(echo "$STATS" | awk '{print $3}')
	
	# 统计所有子进程的CPU（ProcessPool等）
	CHILD_PIDS=$(pgrep -P "$PYTHON_PID" 2>/dev/null || echo "")
	CHILD_CPU=0
	if [ -n "$CHILD_PIDS" ]; then
		for CHILD_PID in $CHILD_PIDS; do
			CHILD_STAT=$(ps -p "$CHILD_PID" -o %cpu= 2>/dev/null || echo "0")
			CHILD_CPU=$(awk "BEGIN {printf \"%.1f\", $CHILD_CPU + $CHILD_STAT}")
		done
	fi
	
	# 总CPU = 主进程 + 所有子进程
	TOTAL_CPU=$(awk "BEGIN {printf \"%.1f\", $CPU_PERCENT + $CHILD_CPU}")

	RSS_MB=$(awk "BEGIN {printf \"%.2f\", $RSS_KB/1024}")
	VSZ_MB=$(awk "BEGIN {printf \"%.2f\", $VSZ_KB/1024}")

	TIMESTAMP=$(date +%s)

	# 写入CSV（确保目录存在）
	if [ ! -d "$(dirname "$log_file")" ]; then
		mkdir -p "$(dirname "$log_file")" 2>/dev/null || true
	fi

	# 写入总CPU（包括子进程）
	echo "$TIMESTAMP,$RSS_MB,$VSZ_MB,$TOTAL_CPU,$PYTHON_PID" >>"$log_file" 2>&1 || {
		echo "[ERROR] 写入失败: $log_file, 工作目录: $(pwd)" >&2
	}

	# 控制台输出（不换行，使用\r覆盖）
	# 显示总CPU（包括子进程）
	if [ "$CHILD_CPU" != "0" ] && [ "$CHILD_CPU" != "0.0" ]; then
		printf "\r[监控] PID:%-6s 内存:%-8s CPU:%-8s (主:%.1f%% 子:%.1f%%)" \
		       "$PYTHON_PID" "${RSS_MB} MB" "${TOTAL_CPU}%" "$CPU_PERCENT" "$CHILD_CPU"
	else
		printf "\r[监控] PID:%-6s 内存:%-8s CPU:%-8s" "$PYTHON_PID" "${RSS_MB} MB" "${CPU_PERCENT}%"
	fi

		sleep 1
	done

	echo "" # 换行
	echo -e "${GREEN}内存监控结束${NC}"
}

##############################################
# 生成内存使用报告
##############################################
generate_memory_report() {
	local csv_file=$1
	local output_file=$2

	if [ ! -f "$csv_file" ] || [ $(wc -l <"$csv_file") -le 1 ]; then
		echo "无有效内存数据" >"$output_file"
		return
	fi

	# 使用awk计算统计数据
	awk -F',' '
    BEGIN {
        min_rss = 999999
        max_rss = 0
        sum_rss = 0
        min_vsz = 999999
        max_vsz = 0
        sum_vsz = 0
        sum_cpu = 0
        count = 0
        max_rss_time = 0
    }
    NR > 1 {
        rss = $2 + 0
        vsz = $3 + 0
        cpu = $4 + 0
        timestamp = $1

        if (rss > 0) {
            if (rss < min_rss) min_rss = rss
            if (rss > max_rss) {
                max_rss = rss
                max_rss_time = timestamp
            }
            sum_rss += rss
        }

        if (vsz > 0) {
            if (vsz < min_vsz) min_vsz = vsz
            if (vsz > max_vsz) max_vsz = vsz
            sum_vsz += vsz
        }

        sum_cpu += cpu
        count++
    }
    END {
        if (count > 0) {
            printf "=== 内存使用统计 ===\n"
            printf "采样次数: %d\n", count
            printf "持续时间: %d 秒\n\n", count

            printf "RSS (物理内存):\n"
            printf "  最小值: %.1f MB\n", min_rss
            printf "  最大值: %.1f MB (峰值)\n", max_rss
            printf "  平均值: %.1f MB\n\n", sum_rss/count

            printf "VSZ (虚拟内存):\n"
            printf "  最小值: %.1f MB\n", min_vsz
            printf "  最大值: %.1f MB\n", max_vsz
            printf "  平均值: %.1f MB\n\n", sum_vsz/count

            printf "CPU使用:\n"
            printf "  平均值: %.1f%%\n\n", sum_cpu/count

            printf "峰值内存占用: %.1f MB\n", max_rss
            printf "峰值出现时间: %d\n", max_rss_time
        }
    }
    ' "$csv_file" >"$output_file"
}

##############################################
# 组装成功标签
##############################################
SUCCESS_LABELS="ant:success"

if [[ -n "${SUCCESS_ADDITIONAL_LABELS:-}" ]]; then
	# 1) 把 “逗号前后出现的空白” 统一删掉，再去掉头尾空白
	_trimmed=$(echo "${SUCCESS_ADDITIONAL_LABELS}" |
		sed -E 's/[[:space:]]*,[[:space:]]*/,/g' |
		sed -E 's/^[[:space:]]+|[[:space:]]+$//g')

	# 2) 过滤掉因输入形如 ",foo,," 产生的空标签
	IFS=',' read -r -a _parts <<<"${_trimmed}"
	_cleaned=()
	for p in "${_parts[@]}"; do
		[[ -n "$p" ]] && _cleaned+=("$p") # 忽略空串
	done

	# 3) 追加到 LABELS
	if ((${#_cleaned[@]})); then
		SUCCESS_LABELS+=","$(
			IFS=','
			echo "${_cleaned[*]}"
		)
	fi
fi

########################################
# defer-like 收尾：统一失败处理
########################################
cleanup() {
	local status=$? # 捕获最后一次命令的退出码
	set +e          # 关闭 -e，避免 cocli 失败递归触发

	if [[ "${status}" -ne 0 ]]; then
		echo "⚠️  脚本异常退出（exit code=${status}），给记录打失败标签..."
		cocli record update "$COS_RECORDID" --append-labels ant:failed || true
	fi

	# 无论成功或失败都清理临时目录
	rm -rf "$OUTPUT_DIR"
}
trap cleanup EXIT
trap 'echo "❌ 发生错误，行号: $LINENO";' ERR # 可选：行号提示

##################################
# Step 0: 判断当前记录有无执行过转换
echo "========== Step 0: 获取 记录 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "项目ID: $COS_PROJECTID"
echo "目标记录ID: $recordName"

# 检查是否强制重转（环境变量控制）
FORCE_RECONVERT=${FORCE_RECONVERT:-false}

# 检查是否已成功转换
HAS_ROSBAG_SUCCESS=$(cocli record describe "$COS_RECORDID" -o json | jq -r '.labels[]?.display_name' | grep -w lerobot_success || true)

if [[ "$FORCE_RECONVERT" == "true" ]]; then
	if [[ -n "$HAS_ROSBAG_SUCCESS" ]]; then
		echo "⚠️  检测到 lerobot_success 标签，但 FORCE_RECONVERT=true，强制重新转换"
		echo "🗑️  清理旧的输出目录..."
		rm -rf "$COS_FILE_VOLUME/export/lerobot"
	fi
	# 继续执行转换
elif [[ -n "$HAS_ROSBAG_SUCCESS" ]]; then
	echo "✅ 当前记录已存在 lerobot_success 标签，跳过 Step1~Step4"
	echo "💡 提示：如需重新转换，请设置环境变量 FORCE_RECONVERT=true"
	echo "🎉 所有处理完成！（无需重复转换）"
	# Step 5: 复制 lerobot 文件夹到指定记录的 /result 目录
	echo "========== Step 5: 复制到指定记录 =========="
	# echo "目标项目ID: $projectName"
	echo "目标记录ID: $recordName"

	echo "上传整个 lerobot 文件夹"
	# 找出 export/lerobot 下的唯一文件夹（uuid文件夹）
	UUID_FOLDER=$(ls -d "$COS_FILE_VOLUME/export/lerobot"/*/ 2>/dev/null | head -n1)
	if [[ -z "$UUID_FOLDER" ]]; then
		echo "❌ 未找到 export/lerobot 下的文件夹"
		exit 1
	fi
	UUID_NAME=$(basename "$UUID_FOLDER")
	echo "找到 UUID 文件夹: $UUID_NAME"

	# 复制到根目录（如果已存在则先删除）
	if [[ -d "$COS_FILE_VOLUME/$UUID_NAME" ]]; then
		echo "⚠️ 目标UUID文件夹已存在，正在删除..."
		rm -rf "$COS_FILE_VOLUME/$UUID_NAME"
	fi
	cp -r "$UUID_FOLDER" "$COS_FILE_VOLUME/"
	echo "已复制到 $COS_FILE_VOLUME/$UUID_NAME"

	# # 列出所有文件的相对路径
	# cd "$COS_FILE_VOLUME"
	# file_to_copy=$(find "$UUID_NAME" -type f | tr '\n' ',' | sed 's/,$//')
	# echo "要上传的文件: $file_to_copy"

	# 上传文件
	#if cocli record file copy "$COS_RECORDID" "$recordName" --files "$file_to_copy" -f ; then
	# if cocli record file copy "$COS_RECORDID" "$recordName" --files "$UUID_NAME/" -f; then
	# 	echo "✅ 已上传文件到 $recordName"
	# else
	# 	echo "❌ 上传失败"
	# 	rm -rf "$COS_FILE_VOLUME/$UUID_NAME"
	# 	exit 1
	# fi
	# 删除临时复制的文件夹
	rm -rf "$COS_FILE_VOLUME/$UUID_NAME"
	echo "已删除临时文件夹"

	# 给记录打标签（允许没有 customer 时不打标签）
	if [[ -n "${customer:-}" ]]; then
		if cocli record update "$COS_RECORDID" --append-labels "$customer"; then
			echo "✅ 已成功添加标签 customer"
		else
			echo "❌ 添加标签失败"
			exit 1
		fi
	fi
fi
cho "========== Step 2: 获取 标注 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "文件存储路径: $COS_FILE_VOLUME"
echo "项目ID: $COS_PROJECTID"
echo ""
if [[ -f "$COS_FILE_VOLUME/moments.json" ]]; then
  echo "⚠️ 已存在 moments.json，删除后生成"
  rm -f "$COS_FILE_VOLUME/moments.json"
fi
if [[ -f "$COS_FILE_VOLUME/metadata.json" ]]; then
  echo "⚠️ 已存在 metadata.json，删除"
  rm -f "$COS_FILE_VOLUME/metadata.json"
fi
# 只获取 data_id 字段并保存到变量
data_id=$(curl --location --request GET "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
  --header "Authorization: Basic ${basicAuth}" \
  --header 'Accept: */*' \
  --header 'Host: openapi.coscene.cn' \
  --header 'Connection: keep-alive' \
  | jq -r '.customFieldValues[] | select(.property.name=="data_id") | .text.value // empty')

if [[ -z "$kuavoAuth" ]]; then
  echo "❌ 未获取到 kuavoAuth"
  exit 1
fi

if [[ -z "$data_id" ]]; then
  echo "❌ 未获取到 data_id"
  exit 1
fi

echo "data_id: $data_id"

# 使用curl下载 metadata.json
if curl -sS -L -H "kuavo-auth: bearer $kuavoAuth" \
  "http://www.lejugym.com/api/kuavo-task/data/detail-with-mark?id=$data_id" \
  | jq '.data' > "$COS_FILE_VOLUME/metadata_raw.json"; then
  echo "✅ metadata_raw.json 下载成功"
else
  echo "❌ metadata_raw.json 下载失败"
  rm -f "$COS_FILE_VOLUME/metadata_raw.json"
  exit 1
fi

# 用 jq 生成精简 metadata.json
if jq '{
  topScene: (.topScene // ""),
  topSceneCode: (.topSceneCode // ""),
  deviceSn: (.deviceSn // ""),
  scene: (.scene // ""),
  sceneCode: (.sceneCode // ""),
  initSceneText: (.initSceneText // ""),
  subSceneCode: (.subSceneCode // ""),
  taskGroupCode: (.taskGroupCode // ""),
  taskCode: (.taskCode // ""),
  subScene: (.subScene // ""),
  taskGroupName: (.taskGroupName // ""),
  englishInitSceneText: (.englishInitSceneText // ""),
  taskName: (.taskName // ""),
  eefType: (.eefType // ""),
  marks: (
    (.marks // []) | map({
      markStart: (.markStart // ""),
      markEnd: (.markEnd // ""),
      duration: (.duration // ""),
      skillDetail: (.skillDetail // ""),
      startPosition: ((.startPosition|tostring) // ""),
      markType: (.markType // ""),
      endPosition: ((.endPosition|tostring) // ""),
      skillAtomic: (.skillAtomic // ""),
      enSkillDetail: (.enSkillDetail // "")
    })
  )
}' "$COS_FILE_VOLUME/metadata_raw.json" > "$COS_FILE_VOLUME/metadata.json"; then
  echo "✅ metadata.json 生成成功"
  rm -f "$COS_FILE_VOLUME/metadata_raw.json"
else
  echo "❌ metadata.json 生成失败"
  rm -f "$COS_FILE_VOLUME/metadata_raw.json" "$COS_FILE_VOLUME/metadata.json"
  exit 1
fi

# Step 2: 执行 ROSbag 到 Lerobot 转换
echo "========== Step 2: ROSbag 转换处理 =========="
echo "转换脚本: $MASTER_SCRIPT"
echo "输入目录: $COS_FILE_VOLUME"
echo "输出目录: $COS_FILE_VOLUME/export/lerobot"
echo ""

# 检查必要的输入文件
# echo "检查输入文件..."
# if [[ ! -f "$COS_FILE_VOLUME/moments.json" ]]; then
# 	echo "❌ 缺少 moments.json 文件"
# 	exit 1
# fi

# if [[ ! -f "$COS_FILE_VOLUME/metadata.json" ]]; then
# 	echo "❌ 缺少 metadata.json 文件"
# 	exit 1
# fi
# 查找 rosbag 文件
ROSBAG_FILES=$(find "${COS_FILE_VOLUME}" -name "*.bag")
ROSBAG_COUNT=$(echo "${ROSBAG_FILES}" | grep -v "^$" | wc -l)
if [[ $ROSBAG_COUNT -eq 0 ]]; then
	echo "❌ 未找到 .bag 文件"
	cocli record update "$COS_RECORDID" --append-labels not bag
	exit 1
fi

echo "✅ 输入文件检查通过"
echo "📊 发现 $ROSBAG_COUNT 个 rosbag 文件"
echo ""

echo "开始执行 ROSbag 转换..."

# 显示内存监控状态
if [ "$ENABLE_MEMORY_MONITOR" = "true" ]; then
	echo -e "${BLUE}✅ 内存监控已启用${NC}"
	echo "  CSV日志: $MEMORY_LOG_CSV"
	echo "  报告文件: $MEMORY_LOG_REPORT"
else
	echo -e "${YELLOW}ℹ️  内存监控未启用（设置 ENABLE_MEMORY_MONITOR=true 启用）${NC}"
fi

# 显示流水线编码状态
if [ "$USE_PIPELINE_ENCODING" = "true" ]; then
	echo -e "${BLUE}✅ 流水线视频编码已启用${NC}"
else
	echo -e "${YELLOW}ℹ️  流水线视频编码未启用（设置 USE_PIPELINE_ENCODING=true 启用）${NC}"
fi

# 显示流式视频编码状态
if [ "$USE_STREAMING_VIDEO" = "true" ]; then
	echo -e "${BLUE}✅ 流式视频编码已启用 (队列上限=${VIDEO_QUEUE_LIMIT})${NC}"
else
	echo -e "${YELLOW}ℹ️  流式视频编码未启用（设置 USE_STREAMING_VIDEO=true 启用）${NC}"
fi

# 显示并行 ROSbag 读取状态
if [ "$USE_PARALLEL_ROSBAG_READ" = "true" ]; then
	echo -e "${BLUE}✅ 并行 ROSbag 读取已启用 (workers=${PARALLEL_ROSBAG_WORKERS})${NC}"
else
	echo -e "${YELLOW}ℹ️  并行 ROSbag 读取未启用（设置 USE_PARALLEL_ROSBAG_READ=true 启用）${NC}"
fi

START_TIME=$(date +%s)

# 先创建输出目录（监控需要）
mkdir -p $OUTPUT_DIR

# 使用数组构建参数（更安全的方法）
ARGS=(
  "python3" "$MASTER_SCRIPT"
  "--bag_dir" "$COS_FILE_VOLUME"
  "--moment_json_dir" "$COS_FILE_VOLUME/moments.json" 
  "--metadata_json_dir" "$COS_FILE_VOLUME/metadata.json"
  "--output_dir" "$OUTPUT_DIR"
)

if [[ -n "${train_frequency:-}" ]]; then
	ARGS+=("--train_frequency" "$train_frequency")
	echo "✅ 设置 --train_frequency $train_frequency"
fi

if [[ -n "${only_arm:-}" ]]; then
	ARGS+=("--only_arm" "$only_arm")
	echo "✅ 设置 --only_arm $only_arm"
fi

if [[ -n "${which_arm:-}" ]]; then
	ARGS+=("--which_arm" "$which_arm")
	echo "✅ 设置 --which_arm $which_arm"
fi

if [[ -n "${dex_dof_needed:-}" ]]; then
	ARGS+=("--dex_dof_needed" "$dex_dof_needed")
	echo "✅ 设置 --dex_dof_needed $dex_dof_needed"
fi

if [[ -n "${use_depth:-}" ]]; then
    # 接受 true/false/1/0（大小写不敏感）
    ARGS+=("--use_depth")
    echo "✅ 设置 --use_depth"
fi


# 显示最终执行的命令（用于调试）
echo "📝 执行命令: ${ARGS[*]}"
echo ""

# 启动内存监控（后台）- 在创建目录之后
MONITOR_PID=""
if [ "$ENABLE_MEMORY_MONITOR" = "true" ]; then
	echo -e "${BLUE}🚀 启动内存监控进程...${NC}"
	echo "  日志文件（相对）: $MEMORY_LOG_CSV"
	echo "  工作目录: $(pwd)"

	# 转换为绝对路径（在启动监控前）
	if [[ "$MEMORY_LOG_CSV" != /* ]]; then
		MEMORY_LOG_CSV_ABS="$(pwd)/$MEMORY_LOG_CSV"
	else
		MEMORY_LOG_CSV_ABS="$MEMORY_LOG_CSV"
	fi

	if [[ "$MEMORY_LOG_REPORT" != /* ]]; then
		MEMORY_LOG_REPORT_ABS="$(pwd)/$MEMORY_LOG_REPORT"
	else
		MEMORY_LOG_REPORT_ABS="$MEMORY_LOG_REPORT"
	fi

	echo "  日志文件（绝对）: $MEMORY_LOG_CSV_ABS"

	# 提取脚本文件名（去掉路径）
	SCRIPT_BASENAME=$(basename "$MASTER_SCRIPT")
	echo "  监控脚本: $SCRIPT_BASENAME"

	# 启动监控（传入绝对路径和实际脚本名）
	monitor_memory_python "$MEMORY_LOG_CSV_ABS" "$SCRIPT_BASENAME" &
	MONITOR_PID=$!

	# 等待1秒，检查监控进程是否启动成功
	sleep 1
	if kill -0 $MONITOR_PID 2>/dev/null; then
		echo -e "${GREEN}✅ 监控进程启动成功 (PID: $MONITOR_PID)${NC}"

		# 验证日志文件是否创建
		if [ -f "$MEMORY_LOG_CSV_ABS" ]; then
			echo -e "${GREEN}✅ 日志文件已创建${NC}"
		else
			echo -e "${YELLOW}⚠️  日志文件尚未创建，等待中...${NC}"
		fi
	else
		echo -e "${RED}❌ 监控进程启动失败${NC}"
		MONITOR_PID=""
	fi
	echo ""
fi

# 执行命令
if timeout 36000 "${ARGS[@]}"; then

	END_TIME=$(date +%s)
	DURATION=$((END_TIME - START_TIME))

	# 等待内存监控结束
	if [ "$ENABLE_MEMORY_MONITOR" = "true" ] && [ -n "$MONITOR_PID" ]; then
		echo -e "${BLUE}⏳ 等待内存监控进程结束...${NC}"
		sleep 2 # 确保最后的数据被写入
		kill $MONITOR_PID 2>/dev/null || true
		wait $MONITOR_PID 2>/dev/null || true

		# 生成内存报告
		echo -e "${BLUE}📊 生成内存使用报告...${NC}"
		generate_memory_report "$MEMORY_LOG_CSV_ABS" "$MEMORY_LOG_REPORT_ABS"

		if [ -f "$MEMORY_LOG_REPORT_ABS" ]; then
			echo ""
			echo -e "${GREEN}=== 内存使用报告 ===${NC}"
			cat "$MEMORY_LOG_REPORT_ABS"
			echo ""
		fi
	fi

	echo ""
	echo "✅ ROSbag 转换成功完成！"
	echo "⏱️  转换耗时: ${DURATION} 秒"

	# 显示输出文件统计
	if [[ -d $OUTPUT_DIR ]]; then
		OUTPUT_SIZE=$(du -sh $OUTPUT_DIR | cut -f1)
		OUTPUT_FILES=$(find $OUTPUT_DIR -type f | wc -l)
		echo "📊 输出文件大小: $OUTPUT_SIZE"
		echo "📊 输出文件数量: $OUTPUT_FILES 个"
	fi

else
	# 转换失败，停止内存监控
	if [ "$ENABLE_MEMORY_MONITOR" = "true" ] && [ -n "$MONITOR_PID" ]; then
		echo -e "${BLUE}⏳ 停止内存监控进程...${NC}"
		sleep 1
		kill $MONITOR_PID 2>/dev/null || true
		wait $MONITOR_PID 2>/dev/null || true

		# 尝试生成报告
		if [ -f "$MEMORY_LOG_CSV_ABS" ] && [ $(wc -l <"$MEMORY_LOG_CSV_ABS") -gt 1 ]; then
			generate_memory_report "$MEMORY_LOG_CSV_ABS" "$MEMORY_LOG_REPORT_ABS"
			echo -e "${YELLOW}⚠️  内存报告已生成（尽管转换失败）${NC}"
		fi
	fi

	echo "❌ ROSbag 转换失败"
	exit 1
fi
echo ""

# Step 3: 处理转换结果
echo "========== Step 3: 处理转换结果 =========="
echo "开始处理输出文件..."

if [[ -n "${TARGET_PROJECT_SLUG:-}" ]]; then
	# 情况1: 上传到指定项目的新记录
	echo "🎯 上传模式: 上传到目标项目 (Project SLUG: $TARGET_PROJECT_SLUG)"

	# 获取当前记录名字
	CURRENT_RECORD_TITLE=$(cocli record describe $COS_RECORDID -o json | jq -r '.title')
	# 创建新记录到指定项目
	NEW_RECORD_NAME=$(cocli record create -p "${TARGET_PROJECT_SLUG}" --title "${CURRENT_RECORD_TITLE} 转换结果-$(date +%Y%m%d_%H%M%S)" -o json | jq -r '.name')
	echo "✅ 已创建新记录: $NEW_RECORD_NAME"

	if cocli record upload -R "$NEW_RECORD_NAME" $OUTPUT_DIR -p "$TARGET_PROJECT_SLUG"; then
		cocli record update $NEW_RECORD_NAME -p $TARGET_PROJECT_SLUG --append-labels $SUCCESS_LABELS
		echo "✅ 文件已上传到新记录: $NEW_RECORD_NAME"

		# 清理临时文件
		echo "🗑️ 清理临时文件..."
		rm -rf $OUTPUT_DIR
		echo "✅ 临时文件清理完成"
	else
		echo "❌ 上传到目标项目失败"
		exit 1
	fi

elif [[ "${CREATE_NEW_RECORD:-false}" == "true" ]]; then
	# 情况2: 在当前项目创建新记录
	echo "📝 上传模式: 在当前项目中创建新记录"

	# 获取当前记录名字
	CURRENT_RECORD_TITLE=$(cocli record describe $COS_RECORDID -o json | jq -r '.title')
	# 在当前项目创建新记录
	NEW_RECORD_NAME=$(cocli record create --title "${CURRENT_RECORD_TITLE} 转换结果-$(date +%Y%m%d_%H%M%S)" -o json | jq -r '.name')
	echo "✅ 已创建新记录: $NEW_RECORD_NAME"

	# 上传文件到新记录
	if cocli record upload -R "$NEW_RECORD_NAME" $OUTPUT_DIR; then
		cocli record update $NEW_RECORD_NAME --append-labels $SUCCESS_LABELS
		echo "✅ 文件已上传到新记录: $NEW_RECORD_NAME"
		# 清理临时文件
		echo "🗑️ 清理临时文件..."
		rm -rf $OUTPUT_DIR
		echo "✅ 临时文件清理完成"
	else
		echo "❌ 在当前项目中创建新记录失败"
		exit 1
	fi

else
	# 情况3: 保存到当前记录的output目录
	echo "📂 保存模式: 保存到当前记录的output目录"

	# 创建output目录并移动文件
	echo "📁 移动文件到输出目录..."
	if [[ -d "$COS_FILE_VOLUME/export/lerobot" ]]; then
		echo "⚠️ 发现已存在 lerobot 文件夹，正在删除..."
		rm -rf "$COS_FILE_VOLUME/export/lerobot"
	fi
	mkdir -p "$COS_FILE_VOLUME/export/lerobot"

	if mv $OUTPUT_DIR/* "$COS_FILE_VOLUME/export/lerobot/"; then
		echo "✅ 成功移动文件到 $COS_FILE_VOLUME/export/lerobot"
		# 清理空的临时目录
		rm -rf $OUTPUT_DIR
		echo "✅ 临时目录清理完成"
	else
		echo "❌ 移动文件失败"
		exit 1
	fi
fi

# ----------- mask图片与动作帧处理 -----------
UUID_FOLDER=$(ls -d "$COS_FILE_VOLUME/export/lerobot"/*/ 2>/dev/null | head -n1)
UUID_NAME=$(basename "$UUID_FOLDER")

# 检查 mask 或 masks 文件夹
MASK_DIR=""
if [[ -d "$COS_FILE_VOLUME/mask" ]]; then
	MASK_DIR="$COS_FILE_VOLUME/mask"
	echo "📷 发现 mask 文件夹: $MASK_DIR"
elif [[ -d "$COS_FILE_VOLUME/masks" ]]; then
	MASK_DIR="$COS_FILE_VOLUME/masks"
	echo "📷 发现 masks 文件夹: $MASK_DIR"
fi

METADATA_JSON="$COS_FILE_VOLUME/export/lerobot/$UUID_NAME/metadata.json"
TARGET_MASK_DIR="$COS_FILE_VOLUME/export/lerobot/$UUID_NAME/mask/chunk-000/episode_000000"

# 检查mask文件夹和metadata.json是否存在
if [[ -n "$MASK_DIR" && -f "$METADATA_JSON" ]]; then
	echo "🔍 开始处理 mask 图片..."
	ACTION_COUNT=$(jq '.label_info.action_config | length' "$METADATA_JSON" 2>/dev/null || echo "0")
	MASK_FILES=($(ls "$MASK_DIR"/mask_*.jpg 2>/dev/null))
	MASK_COUNT=${#MASK_FILES[@]}

	echo "📊 动作数量: $ACTION_COUNT, mask图片数量: $MASK_COUNT"

	if [[ "$ACTION_COUNT" -eq 0 ]]; then
		echo "⚠️ metadata.json中无动作，跳过mask处理"
	elif [[ "$MASK_COUNT" -eq 0 ]]; then
		echo "⚠️ mask文件夹中无图片，跳过mask处理"
	elif [[ "$ACTION_COUNT" -ne "$MASK_COUNT" ]]; then
		echo "⚠️ 动作数量($ACTION_COUNT)与mask图片数量($MASK_COUNT)不一致，跳过mask处理"
	else
		echo "✅ 开始复制 mask 图片..."
		mkdir -p "$TARGET_MASK_DIR"
		for ((i = 0; i < "$ACTION_COUNT"; i++)); do
			# 获取动作帧
			START_FRAME=$(jq ".label_info.action_config[$i].start_frame" "$METADATA_JSON")
			END_FRAME=$(jq ".label_info.action_config[$i].end_frame" "$METADATA_JSON")
			MASK_SRC="${MASK_FILES[$i]}"
			MASK_BASENAME=$(basename "$MASK_SRC" .jpg)
			MASK_TARGET="$TARGET_MASK_DIR/${MASK_BASENAME}_${START_FRAME}-${END_FRAME}.jpg"

			if cp "$MASK_SRC" "$MASK_TARGET"; then
				echo "✅ 已保存mask图片: $(basename "$MASK_TARGET")"
			else
				echo "❌ 复制失败: $MASK_SRC -> $MASK_TARGET"
			fi
		done
		echo "🎯 mask 图片处理完成，共处理 $ACTION_COUNT 张图片"
	fi
else
	if [[ -z "$MASK_DIR" ]]; then
		echo "⚠️ 未找到 mask 或 masks 文件夹，跳过mask处理"
	elif [[ ! -f "$METADATA_JSON" ]]; then
		echo "⚠️ 未找到 metadata.json，跳过mask处理"
	fi
fi
# ----------- mask图片与动作帧处理结束 -----------


# === 获取 EEF_TYPE ===
echo "🔍 获取 eef_type 字段..."
EEF_TYPE=$(curl --silent --location \
  "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
  --header "Authorization: Basic ${basicAuth}" \
  --header 'Accept: */*' \
  | jq -r '
    .customFieldValues 
    | select(. != null) 
    | .[] 
    | select(.property.name == "eef_type") 
    | .text.value
  ')

# 去除首尾空格（防御性处理）
EEF_TYPE=$(echo "$EEF_TYPE" | xargs)

# === 动态选择配置 ===
case "${EEF_TYPE}" in
  "dex_hand")
    CONFIG_FILE="kuavo/lerobot_qc/config/custom_leju_kuavo4pro.yaml"
    ;;
  "leju_claw")
    CONFIG_FILE="kuavo/lerobot_qc/config/custom_leju_kuavo4pro_claw.yaml"
    ;;
  "")
    echo -e "${RED}❌ eef_type 为空，请检查 COS 记录字段！${NC}"
    exit 1
    ;;
  *)
    echo -e "${RED}❌ 未知 eef_type: '$EEF_TYPE'${NC}"
    echo "   仅支持: dex_hand, leju_claw"
    exit 1
    ;;
esac

if [ ! -f "$CONFIG_FILE" ]; then
  echo -e "${RED}❌ 配置文件缺失: $CONFIG_FILE${NC}"
  exit 1
fi

echo "✅ 使用配置: $CONFIG_FILE (eef_type=$EEF_TYPE)"

# === 获取 dataset 路径（第三层）===
shopt -s nullglob
DATASET_PATHS=( "$COS_FILE_VOLUME/export/lerobot"/*/*/* )
shopt -u nullglob

if [ ${#DATASET_PATHS[@]} -eq 0 ]; then
  echo -e "${RED}❌ 未找到 dataset 目录${NC}"
  exit 1
fi
DATASET="${DATASET_PATHS[0]}"

# === 执行验证 ===
VALIDATE_ARGS=(
  python
  kuavo/lerobot_qc/validator_local.py
  "--dataset" "$DATASET"
  "--config" "$CONFIG_FILE"
  "--oss-config" "kuavo/lerobot_qc/config/oss_config.yaml"
)

echo "📝 执行: ${VALIDATE_ARGS[*]}"
timeout 3600 "${VALIDATE_ARGS[@]}" || { echo -e "${RED}❌ 验证失败${NC}";}
echo -e "${GREEN}✅ 验证成功完成！${NC}"



# 启用 nullglob 防止无匹配时报错

shopt -s nullglob
files=(reports/report*.json)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "❌ 未找到 reports/report*.json 文件"
    exit 1
fi

json_file="${files[0]}"
echo "📄 使用报告文件: $json_file"

# Step 0: 验证 JSON 是否可解析
if ! jq empty "$json_file" >/dev/null 2>&1; then
    echo "❌ JSON 文件无法解析"
    exit 1
fi

# 初始化 summary
summary=""

# Step 1: 判断整体是否通过
if jq -e '.overall_passed == true' "$json_file" >/dev/null 2>&1; then
    summary="整体验证通过"
else
    # 整体失败，逐级提取具体错误
    if [[ -z "$summary" ]] && jq -e '.level0_validation.passed == false' "$json_file" >/dev/null 2>&1; then
        summary=$(jq -r '
            .level0_validation.errors[0] // ""
            | split("\n")[0]
            | split(":")[0]
            | gsub("^ +"; "")
        ' "$json_file")
    fi

    if [[ -z "$summary" ]] && jq -e '.dataset_validation.passed == false' "$json_file" >/dev/null 2>&1; then
        summary=$(jq -r '
            .dataset_validation.A1_structure_check.errors[0] //
            .dataset_validation.A2_stats_check.errors[0] //
            .dataset_validation.errors[0] //
            ""
            | split("\n")[0]
            | split(":")[0]
            | gsub("^ +"; "")
        ' "$json_file")
    fi

    if [[ -z "$summary" ]]; then
        summary=$(jq -r '
            .episode_validations[]
            | select(.passed == false)
            | .B1_static_check.errors[0] //
              .B2_angle_gripper_check.errors[0] //
              .B3_anomaly_check.errors[0] //
              .B4_timestamp_check.errors[0] //
              .errors[0] //
              ("Episode \(.episode_idx) 验证失败")
            | split("\n")[0]
            | split(":")[0]
            | gsub("^ +"; "")
            | select(length > 0)
        ' "$json_file" | head -n1)
    fi

    if [[ -z "$summary" ]]; then
        summary="整体验证失败，但未找到具体错误信息"
    fi
fi

# 统一输出
if [[ "$summary" == "整体验证通过" ]]; then
    echo "✅ $summary"
else
    echo "❌ 检测到错误：$summary"
	if cocli record update "$COS_RECORDID" --append-labels "$summary"; then
		echo "✅ 已成功添加标签: $summary"
		exit 1
	else
		echo "❌ 添加标签失败"
		exit 1
	fi
fi



# Step 4: 给当前记录打 lerobot_success 标签
echo "========== Step 4: 打标签 =========="
echo "为当前记录 $COS_RECORDID 添加标签 lerobot_success ..."
if cocli record update "$COS_RECORDID" --append-labels lerobot_success; then
	echo "✅ 已成功添加标签 lerobot_success"
else
	echo "❌ 添加标签失败"
	exit 1
fi

# ========== Step 5: 复制到指定记录 ==========
echo "========== Step 5: 复制到指定记录 =========="
echo "目标记录ID: $recordName"
echo "硬盘ID: $DISK_ID"
echo "FILE_ID: $FILE_ID"

# 1. 找到顶层 UUID 文件夹（export/lerobot 下的第一个目录）
UUID_FOLDER=$(ls -d "$COS_FILE_VOLUME/export/lerobot"/*/ 2>/dev/null | head -n1)
if [[ -z "$UUID_FOLDER" ]]; then
    echo "❌ 未找到 export/lerobot 下的文件夹"
    exit 1
fi
UUID_NAME=$(basename "$UUID_FOLDER")
echo "找到顶层 UUID 文件夹: $UUID_NAME"

# 2. 构建真实数据集路径（三层嵌套：uuid/uuid/uuid/）
REAL_SOURCE="$UUID_FOLDER/$UUID_NAME"
echo "✅ 真实数据集路径: $REAL_SOURCE"

# 3. 获取 device_sn
DEVICE_SN=$(curl --silent --location \
  "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
  --header "Authorization: Basic ${basicAuth}" \
  --header 'Accept: */*' \
  | jq -r '
    .customFieldValues 
    | select(. != null) 
    | .[] 
    | select(.property.name == "device_sn") 
    | .text.value
  ')

if [[ -z "$DEVICE_SN" ]]; then
    echo "❌ 未能获取 device_sn 字段"
    exit 1
fi
echo "Device SN: $DEVICE_SN"


# 4. 根据 device_sn 设置 ROBOT_VERSION
if [[ "$DEVICE_SN" == P4-* ]]; then
    ROBOT_VERSION="Kuavo_4Pro"
elif [[ "$DEVICE_SN" == LB-* ]]; then
    ROBOT_VERSION="Kuavo_LB"
else
    echo "⚠️ 未知设备前缀: $DEVICE_SN，使用默认 Kuavo_4Pro"
    ROBOT_VERSION="Kuavo_4Pro"
fi


TASK_ID=$(curl --silent --location \
  "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
  --header "Authorization: Basic ${basicAuth}" \
  --header 'Accept: */*' \
  | jq -r '
    .customFieldValues 
    | select(. != null) 
    | .[] 
    | select(.property.name == "task_id") 
    | .text.value
  ')

if [[ -z "$TASK_ID" ]]; then
    echo "❌ 未能获取 task_id 字段"
    exit 1
fi
echo "Task ID: $TASK_ID"

#!/bin/bash

CSV_FILE="kuavo/任务ID与采集时间对照表.csv"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "❌ 错误：文件 $CSV_FILE 不存在！" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -z "${TASK_ID+x}" ]]; then
    echo "❌ 错误：变量 TASK_ID 未定义！" >&2
    return 1 2>/dev/null || exit 1
fi

# 查找日期
task_date=$(awk -F',' -v id="$TASK_ID" '
{
    for (i = 1; i <= NF; i += 2) {
        if ($i == id) {
            split($(i+1), dt_arr, " ")
            date_part = dt_arr[1]
            gsub(/-/, "", date_part)
            print date_part
            exit
        }
    }
}
' "$CSV_FILE")

if [[ -n "$task_date" ]]; then
    # 成功找到，task_date 已被赋值
    echo "✅ 找到任务ID $TASK_ID 对应的日期: $task_date" >&2
else
    echo "🔍 未找到任务ID: $TASK_ID" >&2
    unset task_date  # 确保未找到时变量为空或未定义
fi


CSV_FILE="kuavo/任务ID对照表 .csv"


# 检查文件是否存在

if [[ ! -f "$CSV_FILE" ]]; then

    echo "❌ 错误：文件 $CSV_FILE 不存在！" >&2

    return 1 2>/dev/null || exit 1

fi


# 检查 task_id 是否已定义

if [[ -z "${TASK_ID+x}" ]]; then

    echo "❌ 错误：变量 TASK_ID 未定义！" >&2

    return 1 2>/dev/null || exit 1

fi


# 使用 awk 查找：遍历每行，按逗号分割，两两一组匹配

task_code=$(awk -F',' -v id="$TASK_ID" '

{

    # 遍历字段，步长为2：0,2,4,... 是ID；1,3,5,... 是Code（awk 从1开始）

    for (i = 1; i <= NF; i += 2) {

        if ($i == id) {

            code = $(i + 1)

            # 跳过空或纯空白编码

            if (code != "" && code !~ /^[[:space:]]*$/) {

                print code

                exit

            }

        }

    }

}

' "$CSV_FILE")


# 输出结果到 stderr（不影响变量捕获）

if [[ -n "$task_code" ]]; then

    echo "✅ 找到 task_id=$TASK_ID 的编码: $task_code" >&2

else

    echo "🔍 未找到 task_id: $TASK_ID" >&2

    unset task_code

fi


# 5. 构建目标路径


OSS_BUCKET_CLEAN="${OSS_BUCKET%/}"

OSS_TARGET_DIR="${OSS_BUCKET_CLEAN}/${DISK_ID}/${ROBOT_VERSION}/${DEVICE_SN}/${task_date}/${task_code}/${FILE_ID}/"

echo "OSS 目标目录: $OSS_TARGET_DIR"

# 上传到 OSS 四级目录
ossutil cp -r -u "$REAL_SOURCE" "$OSS_TARGET_DIR"


# 删除临时复制的文件夹
echo "已删除临时文件夹"
END=$(date +%s)
DURATION_TIME=$((END - START))
echo "⏱️  转换耗时: ${DURATION_TIME} 秒"
echo "🎉 所有处理完成！"

exit 0
