#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
START=$(date +%s)
# SETUP
SCRIPT_DIR="/app/kuavo"
MASTER_SCRIPT="convert_rosbag_to_hdf5.py"
CONFIG_FILE="configs/request.json"

# 输入/上传相关环境变量
OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"                # 输出目录
INPUT_DIR="${INPUT_DIR:-/inputs}"                  # 输入 bag 目录
OSS_BUCKET="${OSS_BUCKET:-}"                # OSS 桶名称，测试用 oss://leju-delivery-mayi-03/
FOLDER_ID="${FOLDER_ID:-}"                  # OSS 目标目录名，例如 testv3
ACCESS_KEY_ID="${ACCESS_KEY_ID:-}"
ACCESS_KEY_SECRET="${ACCESS_KEY_SECRET:-}"
ENDPOINT="${ENDPOINT:-oss-cn-hangzhou.aliyuncs.com}"

# 平台有时会传入相对路径（如 inputs/outputs），这里自动标准化
normalize_dir() {
  local p="${1:-}"
  if [[ -z "$p" ]]; then
    echo ""
    return
  fi
  if [[ "$p" == /* ]]; then
    echo "$p"
    return
  fi
  if [[ -d "/$p" ]]; then
    echo "/$p"
  else
    echo "$p"
  fi
}

INPUT_DIR="$(normalize_dir "$INPUT_DIR")"
OUTPUT_DIR="$(normalize_dir "$OUTPUT_DIR")"
# Step 1: 通过环境变量生成 ossutil 配置文件（仅在需要上传时）
if [[ -n "$OSS_BUCKET" || -n "$FOLDER_ID" ]]; then

  if [[ -z "$ACCESS_KEY_ID" || -z "$ACCESS_KEY_SECRET" || -z "$ENDPOINT" ]]; then
    echo "❌ 缺少 OSS 配置：需要 ACCESS_KEY_ID / ACCESS_KEY_SECRET / ENDPOINT"
    exit 1
  fi

  echo "========== Step 1: 生成 ossutil 配置文件 =========="
  cat > ~/.ossutilconfig <<EOF
[default]
accessKeyId=${ACCESS_KEY_ID}
accessKeySecret=${ACCESS_KEY_SECRET}
region=cn-hangzhou
endpoint=${ENDPOINT}
EOF
else
  echo "========== Step 1: 跳过 ossutil 配置（未配置 OSS_BUCKET/FOLDER_ID） =========="
fi

# Step 2: 执行 ROSbag 到 HDF5 转换
echo "========== Step 2: ROSbag 转换处理 =========="
if [[ -z "$INPUT_DIR" ]]; then
  echo "❌ 缺少 INPUT_DIR 环境变量"
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "❌ INPUT_DIR 不是目录或不存在: $INPUT_DIR"
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "❌ 缺少 OUTPUT_DIR 环境变量"
  exit 1
fi
echo "转换脚本: $MASTER_SCRIPT"
echo "输入根目录: $INPUT_DIR"
echo "输出根目录: $OUTPUT_DIR"
echo ""

# 进入脚本目录，确保相对路径资源可用（如 biped_s45.urdf）
cd "$SCRIPT_DIR"

# 检查必要的配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ 缺少配置文件: $CONFIG_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

OUTPUT_DIR_DATA="$OUTPUT_DIR/export/hdf5"
mkdir -p "$OUTPUT_DIR_DATA"

mapfile -t DATA_DIRS < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#DATA_DIRS[@]} -eq 0 ]]; then
  if find "$INPUT_DIR" -maxdepth 1 -type f -name "*.bag" ! -name "*.c.bag" -print -quit | grep -q .; then
    DATA_DIRS=("$INPUT_DIR")
    echo "ℹ️ INPUT_DIR 顶层检测到 bag 文件，按单目录模式处理: $INPUT_DIR"
  else
    echo "❌ INPUT_DIR 下没有可处理的 data_id 子目录或 bag 文件: $INPUT_DIR"
    exit 1
  fi
fi

for DATA_DIR in "${DATA_DIRS[@]}"; do
  data_id="$(basename "$DATA_DIR")"
  echo "========== 处理 data_id: $data_id =========="

  # metadata.json 约定与 bag 同级目录
  METADATA_JSON_PATH="$DATA_DIR/metadata.json"
  if [[ ! -f "$METADATA_JSON_PATH" ]]; then
    echo "❌ 缺少 metadata.json 文件: $METADATA_JSON_PATH"
    exit 1
  fi
  echo "========== metadata.json 内容预览 =========="
  echo "📄 metadata.json 内容预览: $(head -c 300 "$METADATA_JSON_PATH")"
  

  # 查找 rosbag 文件
  ROSBAG_FILES=$(find "${DATA_DIR}" -name "*.bag" ! -name "*.c.bag")
  ROSBAG_COUNT=$(echo "${ROSBAG_FILES}" | grep -v "^$" | wc -l)
  if [[ $ROSBAG_COUNT -eq 0 ]]; then
    echo "❌ 未找到 .bag 文件: $DATA_DIR"
    exit 1
  fi
  echo "✅ 输入文件检查通过"
  echo "📊 发现 $ROSBAG_COUNT 个 rosbag 文件"
  echo ""

  echo "开始执行 ROSbag 转换..."
  START_TIME=$(date +%s)

  OUTPUT_DIR_ONE="$OUTPUT_DIR_DATA/$data_id"
  mkdir -p "$OUTPUT_DIR_ONE"
  # 如果没有传入 MIN_DURATION，则自动设置为 5
  if [[ -z "${MIN_DURATION:-}" ]]; then
    MIN_DURATION=5                                      #######测试用1 秒##########
    echo "⚠️ 未检测到 MIN_DURATION，自动设置为默认值 5 秒"   #######测试用1 秒##########
  else
    echo "ℹ️ 检测到 MIN_DURATION，设置为 ${MIN_DURATION} 秒"
  fi

  if python3 "$MASTER_SCRIPT" \
    --config "$CONFIG_FILE" \
    --bag_dir "$DATA_DIR" \
    --metadata_json_dir "$METADATA_JSON_PATH" \
    --output_dir "$OUTPUT_DIR_ONE" \
    --scene "test_scene" \
    --sub_scene "test_sub_scene" \
    --continuous_action "test_continuous_action" \
    --min_duration "$MIN_DURATION" \
    --mode "simplified"; then

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "✅ ROSbag 转换成功完成！"
    echo "⏱️  转换耗时: ${DURATION} 秒"

    # 显示输出文件统计
    if [[ -d "$OUTPUT_DIR_DATA" ]]; then
      OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR_ONE" | cut -f1)
      OUTPUT_FILES=$(find "$OUTPUT_DIR_ONE" -type f | wc -l)
      echo "📊 输出文件大小: $OUTPUT_SIZE"
      echo "📊 输出文件数量: $OUTPUT_FILES 个"
    fi

  else
    echo "❌ ROSbag 转换失败 (data_id: $data_id)"
    exit 1
  fi
  echo ""
done

# Step 3: 上传 $OUTPUT_DIR 到 oss（可选）
if [[ -n "$OSS_BUCKET" || -n "$FOLDER_ID" ]]; then

  if [[ -z "$OSS_BUCKET" || -z "$FOLDER_ID" ]]; then
    echo "❌ 上传需要同时配置 OSS_BUCKET 和 FOLDER_ID"
    exit 1
  fi
  
  echo "========== Step 3: 上传到 oss =========="
  if [[ ! -d "$OUTPUT_DIR_DATA" ]]; then
    echo "❌ 转换输出目录不存在，无法上传: $OUTPUT_DIR_DATA"
    exit 1
  fi
  echo "✅ 真实数据集路径: $OUTPUT_DIR_DATA"
  echo "上传目录名称: $FOLDER_ID"

  OSS_BUCKET_CLEAN="${OSS_BUCKET%/}"
  OSS_TARGET_DIR="${OSS_BUCKET_CLEAN}/${FOLDER_ID}/"

  echo "OSS 目标目录: $OSS_TARGET_DIR"
  ossutil cp -r -u "$OUTPUT_DIR_DATA" "$OSS_TARGET_DIR"
else
  echo "========== Step 3: 跳过上传（未配置 OSS_BUCKET/FOLDER_ID） =========="
fi

echo "🎉 所有处理完成！"
END=$(date +%s)
DURATION_TIME=$((END - START))
echo "⏱️  总耗时: ${DURATION_TIME} 秒"
exit 0
