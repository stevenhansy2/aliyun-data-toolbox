#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ANSI 颜色（可选）
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "========== Step 1: 生成 ossutil 配置文件 =========="
cat > ~/.ossutilconfig <<EOF
[default]
accessKeyId=$accessKeyID
accessKeySecret=$accessKeySecret
region=cn-hangzhou
endpoint=$endpoint
EOF

# Step 0: 打印关键信息
echo "========== Step 0: 获取 记录 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "文件存储路径: $COS_FILE_VOLUME"
echo "项目ID: $COS_PROJECTID"
echo ""

# Step 1: 设置 MERGE_DIR
MERGE_DIR="$COS_FILE_VOLUME/result"

# 获取所有第8层目录（用于后续上传目标位置）
all_dirs=$(find "$COS_FILE_VOLUME" -mindepth 9 -maxdepth 9 -type d | sort)
echo "目标目录: $all_dirs"
if [ -z "$all_dirs" ]; then
  echo -e "${RED}❌ 未找到任何第8层目录${NC}"
  exit 1
fi

output_report=$(find "$COS_FILE_VOLUME" -mindepth 8 -maxdepth 8 -type d | sort)
echo "目标目录: $all_dirs"

# Step 2: 运行转换脚本
echo "🔄 运行 merge_data.py..."
python3 lerobot_qc/merge_data.py --src_dir "$COS_FILE_VOLUME"/*/*/*/*/*/*/*/*/*/ --tgt_dir "$MERGE_DIR" --summary_dir "$output_report/report" --save
echo "🎉 汇总与转换完成！"

# === 获取 EEF_TYPE ===
EEF_TYPE=$(echo "$EEF_TYPE" | xargs)
echo "🔍 eef_type = '$EEF_TYPE'"

# === 动态选择配置 ===
case "${EEF_TYPE}" in
  "dex_hand")
    CONFIG_FILE="lerobot_qc/config/custom_leju_kuavo4pro.yaml"
    ;;
  "leju_claw")
    CONFIG_FILE="lerobot_qc/config/custom_leju_kuavo4pro_claw.yaml"
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

# === 获取 dataset 路径（MERGE_DIR 下的第一个子目录）===
# ❗❗ 修复：必须用 /*/ 获取子目录，而不是字符串
shopt -s nullglob
DATASET_SUBDIRS=("$MERGE_DIR")
shopt -u nullglob

if [ ${#DATASET_SUBDIRS[@]} -eq 0 ] || [ ! -d "${DATASET_SUBDIRS[0]}" ]; then
  echo -e "${RED}❌ $MERGE_DIR 下没有子目录！请检查 merge_data.py。${NC}"
  ls -la "$MERGE_DIR"
  exit 1
fi

DATASET="${DATASET_SUBDIRS[0]}"
echo "📂 使用 dataset: $DATASET"

# === 执行验证 ===
VALIDATE_ARGS=(
  python
  lerobot_qc/validator_local.py
  "--dataset" "$DATASET"
  "--config" "$CONFIG_FILE"
  "--output" "$output_report/report"
  "--oss-config" "lerobot_qc/config/oss_config.yaml"
)

echo "📝 执行: ${VALIDATE_ARGS[*]}"
timeout 3600 "${VALIDATE_ARGS[@]}" || { echo -e "${RED}❌ 验证失败${NC}"; exit 1; }
echo -e "${GREEN}✅ 验证成功完成！${NC}"

# Step 4: 打标签
echo "========== Step 4: 打标签 =========="
if cocli record update "$COS_RECORDID" --append-labels merge_success; then
  echo "✅ 已成功添加标签 merge_success"
else
  echo "❌ 添加标签失败"
  exit 1
fi


echo "========== Step 5: 构建相对路径结构并上传 =========="

# 清理临时变量
rm -rf "$all_dirs"

# 标准化 COS_FILE_VOLUME（去掉末尾 /）
BASE="${COS_FILE_VOLUME%/}"

mkdir -p "$(dirname "$all_dirs")"

# 复制整个 result 目录（包括其子目录）到临时位置
cp -r "$MERGE_DIR" "$all_dirs"
echo "✅ 已复制到临时上传目录: $all_dirs"

# 删除 MERGE_DIR
rm -rf "$MERGE_DIR"

# 清理 OSS_BUCKET 变量
OSS_BUCKET_CLEAN="${OSS_BUCKET%/}"

# 确定 OSS_TARGET_DIR
OSS_TARGET_DIR="${OSS_BUCKET_CLEAN}/${dir}/"

echo "OSS 目标目录: $OSS_TARGET_DIR"

# 获取所有非隐藏的顶层子目录（即不上传 .开头的文件夹）
shopt -s nullglob
NON_HIDDEN_DIRS=("$BASE"/[^.]*)
shopt -u nullglob

if [ ${#NON_HIDDEN_DIRS[@]} -eq 0 ]; then
  echo -e "${RED}❌ 没有找到任何非隐藏的顶层目录（如 example/）${NC}"
  exit 1
fi

echo "✅ 将上传以下目录: ${NON_HIDDEN_DIRS[*]}"

# 逐个上传这些目录，而不是上传整个父目录
for dir in "${NON_HIDDEN_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    dir_name=$(basename "$dir")
    target="$OSS_TARGET_DIR/$dir_name/"
    echo "📤 上传 $dir 到 $target"
    ossutil cp -r -u "$dir" "$target" --exclude ".*"  # 排除内部的隐藏文件
  fi
done

echo -e "${GREEN}✅ 已上传到 OSS: $OSS_TARGET_DIR${NC}"
echo "🧹 临时文件已清理"

echo -e "${GREEN}🎉 所有处理完成！${NC}"
exit 0