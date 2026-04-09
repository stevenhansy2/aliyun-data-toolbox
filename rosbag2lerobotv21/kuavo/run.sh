#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

START=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT_SRC="$PROJECT_ROOT/lerobot/src"

# 本地直跑时，确保能导入仓库内 lerobot 源码
if [[ -d "$LEROBOT_SRC" ]]; then
  export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi

# 输入相关环境变量
OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"
INPUT_DIR="${INPUT_DIR:-/inputs}"
MASTER_TIMEOUT_SEC="${MASTER_TIMEOUT_SEC:-36000}"

# 转换参数：当前仓库仅保留 _s 逻辑
MASTER_SCRIPT="$SCRIPT_DIR/master_generate_lerobot_s.py"
echo "使用流式版本脚本 (master_generate_lerobot_s.py)"

# Step 1: 执行 ROSbag 到 LeRobot 转换
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "❌ INPUT_DIR 不是目录或不存在: $INPUT_DIR"
  exit 1
fi
if [[ ! -f "$MASTER_SCRIPT" ]]; then
  echo "❌ 转换脚本不存在: $MASTER_SCRIPT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========== Step 2: ROSbag 转换处理 =========="
echo "转换脚本: $MASTER_SCRIPT"
echo "输入根目录: $INPUT_DIR"
echo "输出根目录: $OUTPUT_DIR"
echo ""

OUTPUT_DIR_DATA="$OUTPUT_DIR/export/lerobot"
mkdir -p "$OUTPUT_DIR_DATA"

mapfile -t DATA_DIRS < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#DATA_DIRS[@]} -eq 0 ]]; then
  if find "$INPUT_DIR" -maxdepth 1 -type f -name "*.bag" ! -name "*.c.bag" -print -quit | grep -q .; then
    DATA_DIRS=("$INPUT_DIR")
    echo "ℹ️ INPUT_DIR 顶层检测到 bag 文件，按单目录批量模式处理: $INPUT_DIR"
  else
    echo "❌ INPUT_DIR 下没有可处理的 data_id 子目录或 bag 文件: $INPUT_DIR"
    exit 1
  fi
fi

for DATA_DIR in "${DATA_DIRS[@]}"; do
  data_id="$(basename "$DATA_DIR")"

  echo "========== 处理 data_id: $data_id =========="

  # 检测 metadata.json（默认放在输入目录）
  METADATA_JSON_PATH="$DATA_DIR/metadata.json"
  if [[ ! -f "$METADATA_JSON_PATH" ]]; then
    echo "⚠️ 缺少 metadata.json，继续转换（将不合并 metadata）: $METADATA_JSON_PATH"
    METADATA_JSON_PATH=""
  else
    echo "✅ 检测到 metadata.json: $METADATA_JSON_PATH"
  fi

  if ! find "$DATA_DIR" -maxdepth 1 -type f -name "*.bag" ! -name "*.c.bag" -print -quit | grep -q .; then
    echo "❌ 未找到 .bag 文件: $DATA_DIR"
    exit 1
  fi
  ROSBAG_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.bag" ! -name "*.c.bag" | wc -l)

  echo "✅ 输入文件检查通过"
  echo "📊 发现 $ROSBAG_COUNT 个 rosbag 文件"

  ARGS=(
    python3 "$MASTER_SCRIPT"
    --bag_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR_DATA"
  )
  if [[ -n "$METADATA_JSON_PATH" ]]; then
    ARGS+=(--metadata_json_dir "$METADATA_JSON_PATH")
  fi
  if [[ -f "$SCRIPT_DIR/configs/request.json" ]]; then
    ARGS+=(--config "$SCRIPT_DIR/configs/request.json")
  fi

  if [[ -n "${train_frequency:-}" ]]; then
    ARGS+=(--train_frequency "$train_frequency")
    echo "✅ 设置 --train_frequency $train_frequency"
  fi
  if [[ -n "${only_arm:-}" ]]; then
    ARGS+=(--only_arm "$only_arm")
    echo "✅ 设置 --only_arm $only_arm"
  fi
  if [[ -n "${which_arm:-}" ]]; then
    ARGS+=(--which_arm "$which_arm")
    echo "✅ 设置 --which_arm $which_arm"
  fi
  if [[ -n "${dex_dof_needed:-}" ]]; then
    ARGS+=(--dex_dof_needed "$dex_dof_needed")
    echo "✅ 设置 --dex_dof_needed $dex_dof_needed"
  fi
  if [[ "${use_depth:-false}" == "true" ]]; then
    ARGS+=(--use_depth)
    echo "✅ 设置 --use_depth"
  fi

  echo "📝 执行命令: ${ARGS[*]}"

  START_TIME=$(date +%s)
  if (cd "$PROJECT_ROOT" && timeout "$MASTER_TIMEOUT_SEC" "${ARGS[@]}"); then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "✅ data_id=$data_id 转换成功完成！"
    echo "⏱️  转换耗时: ${DURATION} 秒"
  else
    echo "❌ ROSbag 转换失败 (data_id: $data_id)"
    exit 1
  fi

  echo ""
done

if [[ ! -d "$OUTPUT_DIR_DATA" ]]; then
  echo "❌ 转换输出目录不存在: $OUTPUT_DIR_DATA"
  exit 1
fi

OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR_DATA" | cut -f1)
OUTPUT_FILES=$(find "$OUTPUT_DIR_DATA" -type f | wc -l)
echo "📊 总输出文件大小: $OUTPUT_SIZE"
echo "📊 总输出文件数量: $OUTPUT_FILES 个"

END=$(date +%s)
DURATION_TIME=$((END - START))
echo "⏱️  总耗时: ${DURATION_TIME} 秒"
echo "🎉 所有处理完成！"

exit 0
