#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# SETUP
OUTPUT_DIR="temp"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
##############################################
# 组装成功标签
##############################################
SUCCESS_LABELS="r25:success"

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
    cocli record update "$COS_RECORDID" --append-labels lerobotv3:failed || true
  fi

  # 无论成功或失败都清理临时目录
  rm -rf "$OUTPUT_DIR"
}
trap cleanup EXIT
trap 'echo "❌ 发生错误，行号: $LINENO";' ERR # 可选：行号提示

##################################
# Step 0: 判断当前记录有无执行过转换
echo "========== Step 1: 获取 记录 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "项目ID: $COS_PROJECTID"
echo "目标记录ID: $COS_RECORDID"


# 给记录打标签（允许没有 customer 时不打标签）
if [[ -n "${customer:-}" ]]; then
  if cocli record update "$COS_RECORDID" --append-labels "$customer"; then
    echo "✅ 已成功添加标签 customer"
  else
    echo "❌ 添加标签失败"
    exit 1
  fi
fi

echo "========== Step 2: ROSbag 转换处理 =========="
echo "转换脚本: CvtRosbag2Lerobot.py"
echo "输入目录: $COS_FILE_VOLUME"
echo "输出目录: $COS_FILE_VOLUME/export/lerobot"
echo ""

# 查找 rosbag 文件
ROSBAG_FILES=$(find "${COS_FILE_VOLUME}" -name "*.bag")
ROSBAG_COUNT=$(echo "${ROSBAG_FILES}" | grep -v "^$" | wc -l)
if [[ $ROSBAG_COUNT -eq 0 ]]; then
  echo "❌ 未找到 .bag 文件"
  exit 1
fi

echo "✅ 输入文件检查通过"
echo "📊 发现 $ROSBAG_COUNT 个 rosbag 文件"
echo ""

echo "开始执行 ROSbag 转换..."
START_TIME=$(date +%s)

mkdir -p $OUTPUT_DIR
echo "🔍 获取 eef_type 字段..."

EEF_TYPE=""
for bag in $ROSBAG_FILES; do
  fname=$(basename "$bag")
  if [[ "$fname" == *dex_hand* ]]; then
    EEF_TYPE="dex_hand"
    break
  elif [[ "$fname" == *leju_claw* ]]; then
    EEF_TYPE="leju_claw"
    break
  fi
done

# 防御性处理
if [[ -z "$EEF_TYPE" ]]; then
  echo -e "${RED}❌ 未能从 bag 文件名自动识别 eef_type${NC}"
  echo "   文件名需包含 dex_hand 或 leju_claw"
  exit 1
fi

# 去除首尾空格（防御性处理）
EEF_TYPE=$(echo "$EEF_TYPE" | xargs)

# === 动态选择配置 ===
case "${EEF_TYPE}" in
  "dex_hand")
    CONFIG_FILE="KuavoRosbag2Lerobot.yaml"
    ;;
  "leju_claw")
    CONFIG_FILE="KuavoRosbag2Lerobot_claw.yaml"
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

echo "✅ 使用配置: $CONFIG_FILE (eef_type=$EEF_TYPE)"
# 使用数组构建参数（更安全的方法）
ARGS=(
  "python3" "kuavo_data/CvtRosbag2Lerobot.py"
  "--config-path" "configs/data/"
  "--config-name" "$CONFIG_FILE"
  "rosbag.rosbag_dir=$COS_FILE_VOLUME"
  "rosbag.lerobot_dir=$COS_FILE_VOLUME/export"
)

# 显示最终执行的命令（用于调试）
echo "📝 执行命令: ${ARGS[*]}"

# 执行命令
if timeout 7200 "${ARGS[@]}"; then

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  echo ""
  echo "✅ ROSbag 转换成功完成！"
  echo "⏱️  转换耗时: ${DURATION} 秒"
  
else
  echo "❌ ROSbag 转换失败"
  exit 1
fi


# Step 4: 给当前记录打 lerobot_success 标签
echo "========== Step 4: 打标签 =========="
echo "为当前记录 $COS_RECORDID 添加标签 lerobotv3_success ..."
if cocli record update "$COS_RECORDID" --append-labels lerobotv3_success; then
  echo "✅ 已成功添加标签 lerobot_success"
else
  echo "❌ 添加标签失败"
  exit 1
fi

echo "🎉 所有处理完成！"

exit 0

