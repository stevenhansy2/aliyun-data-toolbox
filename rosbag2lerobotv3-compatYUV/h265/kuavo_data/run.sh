#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

START=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEROBOT_SRC="$PROJECT_ROOT/lerobot/src"
LOCAL_CONDA_PYTHON_DEFAULT="/tmp/r2l050_yuv_env/bin/python"
LOCAL_ROS_PYTHONPATH_DEFAULT="/opt/ros/noetic/lib/python3/dist-packages"

# 本地直跑时，确保能导入仓库内 lerobot 源码，并尽量补齐 ROS Python 路径
if [[ -d "$LEROBOT_SRC" ]]; then
  if [[ -d "$LOCAL_ROS_PYTHONPATH_DEFAULT" ]]; then
    export PYTHONPATH="$LOCAL_ROS_PYTHONPATH_DEFAULT:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
  else
    export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
  fi
elif [[ -d "$LOCAL_ROS_PYTHONPATH_DEFAULT" ]]; then
  export PYTHONPATH="$LOCAL_ROS_PYTHONPATH_DEFAULT${PYTHONPATH:+:$PYTHONPATH}"
fi

# 输入/上传相关环境变量（对齐 rosbag2hdf5/run.sh）
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/rosbag2lerobotv3_compat_local_test}"
INPUT_DIR="${INPUT_DIR:-/home/leju_kuavo/hsy-temp/file/data/rosbag/no_color_raw}"
OSS_BUCKET="${OSS_BUCKET:-}"
FOLDER_ID="${FOLDER_ID:-}"
ACCESS_KEY_ID="${ACCESS_KEY_ID:-LTAI5tEs3xD65oJHSAF8S7fJ}"
ACCESS_KEY_SECRET="${ACCESS_KEY_SECRET:-gpcIcxhVUT0ybGqlvNoNrNkb13suIs}"
ENDPOINT="${ENDPOINT:-oss-cn-hangzhou.aliyuncs.com}"
MASTER_TIMEOUT_SEC="${MASTER_TIMEOUT_SEC:-36000}"

### 转换参数：当前仓库仅保留 _s 逻辑
if [[ -z "${PYTHON:-}" && -x "$LOCAL_CONDA_PYTHON_DEFAULT" ]]; then
  PYTHON="$LOCAL_CONDA_PYTHON_DEFAULT"
else
  PYTHON="${PYTHON:-python}"
fi
MASTER_SCRIPT="$SCRIPT_DIR/CvtRosbag2Lerobot.py"
echo "使用流式版本脚本 (CvtRosbag2Lerobot.py)"
echo "Python 解释器: $PYTHON"
echo "PYTHONPATH: ${PYTHONPATH:-<empty>}"

detect_eef_type_by_topic() {
  local bag_path="$1"
  "$PYTHON" - "$bag_path" <<'PY'
import sys
try:
    import rosbag
except Exception:
    print("")
    raise SystemExit(0)

bag_path = sys.argv[1]
try:
    bag = rosbag.Bag(bag_path)
    topics = set(bag.get_type_and_topic_info().topics.keys())
    bag.close()
except Exception:
    print("")
    raise SystemExit(0)

if "/dexhand/state" in topics or "/control_robot_hand_position" in topics:
    print("qiangnao")
elif "/leju_claw_state" in topics or "/leju_claw_command" in topics:
    print("leju_claw")
elif "/gripper/state" in topics or "/gripper/command" in topics:
    print("rq2f85")
else:
    print("")
PY
}

# Step 1: 生成 ossutil 配置（仅上传时需要）
if [[ -n "$OSS_BUCKET" || -n "$FOLDER_ID" ]]; then
  if [[ -z "$ACCESS_KEY_ID" || -z "$ACCESS_KEY_SECRET" || -z "$ENDPOINT" ]]; then
    echo "❌ 缺少 OSS 配置：需要 ACCESS_KEY_ID / ACCESS_KEY_SECRET / ENDPOINT"
    exit 1
  fi

  echo "========== Step 1: 生成 ossutil 配置文件 =========="
  cat > ~/.ossutilconfig <<EOC
[default]
accessKeyId=${ACCESS_KEY_ID}
accessKeySecret=${ACCESS_KEY_SECRET}
region=cn-hangzhou
endpoint=${ENDPOINT}
EOC
else
  echo "========== Step 1: 跳过 ossutil 配置（未配置 OSS_BUCKET/FOLDER_ID） =========="
fi

# Step 2: 执行 ROSbag 到 LeRobot 转换
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

  METADATA_JSON_PATH="$DATA_DIR/metadata.json"
  if [[ ! -f "$METADATA_JSON_PATH" ]]; then
    echo "⚠️ 缺少 metadata.json，继续转换（将不合并 metadata）: $METADATA_JSON_PATH"
    METADATA_JSON_PATH=""
  else
    echo "✅ 检测到 metadata.json: $METADATA_JSON_PATH"
  fi

  if ! find "$DATA_DIR" -maxdepth 1 -type f -name "*.bag" ! -name "*.c.bag" -print -quit | grep -q .; then
    echo -e "${YELLOW}⚠️ 未找到有效 .bag 文件（仅含 .c.bag），跳过: $DATA_DIR${NC}"
    continue
  fi
  ROSBAG_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.bag" ! -name "*.c.bag" | wc -l)

  echo "✅ 输入文件检查通过"
  echo "📊 发现 $ROSBAG_COUNT 个 rosbag 文件"

  EEF_TYPE=""
  for bag in "$DATA_DIR"/*.bag; do
    fname=$(basename "$bag")
    if [[ "$fname" == *dex_hand* ]]; then
      EEF_TYPE="qiangnao"
      break
    elif [[ "$fname" == *leju_claw* ]]; then
      EEF_TYPE="leju_claw"
      break
    elif [[ "$fname" == *rq2f85* ]]; then
      EEF_TYPE="rq2f85"
      break
    else
      echo "⚠️ 无法从文件名识别 eef_type: $fname"
    fi
  done

  if [[ -z "$EEF_TYPE" ]]; then
    for bag in "$DATA_DIR"/*.bag; do
      PROBED_EEF_TYPE="$(detect_eef_type_by_topic "$bag" | tr -d '[:space:]')"
      if [[ -n "$PROBED_EEF_TYPE" ]]; then
        EEF_TYPE="$PROBED_EEF_TYPE"
        echo "✅ 通过 bag topic 自动识别 eef_type: $EEF_TYPE ($bag)"
        break
      fi
    done
  fi

  if [[ -z "$EEF_TYPE" ]]; then
    echo -e "${RED}❌ 未能自动识别 eef_type${NC}"
    echo "   文件名或 bag topic 需能识别为: qiangnao, leju_claw, rq2f85"
    exit 1
  fi

  EEF_TYPE=$(echo "$EEF_TYPE" | xargs)
  CONFIG_OVERRIDES=()

  case "$EEF_TYPE" in
    "qiangnao")
      CONFIG_FILE="KuavoRosbag2Lerobot.yaml"
      CONFIG_OVERRIDES+=("dataset.eef_type=qiangnao")
      ;;
    "leju_claw")
      CONFIG_FILE="KuavoRosbag2Lerobot_claw.yaml"
      CONFIG_OVERRIDES+=("dataset.eef_type=leju_claw")
      ;;
    "rq2f85")
      CONFIG_FILE="KuavoRosbag2Lerobot.yaml"
      CONFIG_OVERRIDES+=("dataset.eef_type=rq2f85")
      ;;
    *)
      echo -e "${RED}❌ 未知 eef_type: '$EEF_TYPE'${NC}"
      echo "   仅支持: qiangnao, leju_claw, rq2f85"
      exit 1
      ;;
  esac

  echo "✅ 使用配置: $CONFIG_FILE (eef_type=$EEF_TYPE)"

  ARGS=(
    "$PYTHON" "$MASTER_SCRIPT"
    "--config-path" "configs/"
    "--config-name" "$CONFIG_FILE"
    "rosbag.rosbag_dir=$DATA_DIR"
    "rosbag.lerobot_dir=$OUTPUT_DIR_DATA"
  )
  ARGS+=("${CONFIG_OVERRIDES[@]}")

  if [[ -n "$METADATA_JSON_PATH" ]]; then
    ARGS+=("rosbag.metadata_json=$METADATA_JSON_PATH")
    echo "✅ 传递 metadata.json 到脚本: $METADATA_JSON_PATH"
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
  echo "❌ 转换输出目录不存在，无法上传: $OUTPUT_DIR_DATA"
  exit 1
fi

OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR_DATA" | cut -f1)
OUTPUT_FILES=$(find "$OUTPUT_DIR_DATA" -type f | wc -l)
echo "📊 总输出文件大小: $OUTPUT_SIZE"
echo "📊 总输出文件数量: $OUTPUT_FILES 个"

# Step 3: 上传到 OSS（可选）
if [[ -n "$OSS_BUCKET" || -n "$FOLDER_ID" ]]; then
  if [[ -z "$OSS_BUCKET" || -z "$FOLDER_ID" ]]; then
    echo "❌ 上传需要同时配置 OSS_BUCKET 和 FOLDER_ID"
    exit 1
  fi

  echo "========== Step 3: 上传到 oss =========="
  OSS_BUCKET_CLEAN="${OSS_BUCKET%/}"
  OSS_TARGET_ROOT="${OSS_BUCKET_CLEAN}/${FOLDER_ID}"
  OSS_TARGET_DIR="${OSS_TARGET_ROOT}/"
  echo "✅ 上传目录: $OUTPUT_DIR_DATA"
  echo "OSS 目标目录: $OSS_TARGET_DIR"
  ossutil cp -r -u "$OUTPUT_DIR_DATA" "$OSS_TARGET_DIR"
else
  echo "========== Step 3: 跳过上传（未配置 OSS_BUCKET/FOLDER_ID） =========="
fi

END=$(date +%s)
DURATION=$((END - START))

echo "========== 全部流程完成 =========="
echo "总耗时: ${DURATION} 秒"
