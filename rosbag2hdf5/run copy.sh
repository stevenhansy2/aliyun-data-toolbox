#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# SETUP
OUTPUT_DIR="temp"

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
    cocli record update "$COS_RECORDID" --append-labels r25:failed || true
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

#记录时间
SCRIPT_START_TIME=$(date +%s)

print_step_info() {
  local step="$1"
  local now=$(date "+%Y-%m-%d %H:%M:%S")
  local now_sec=$(date +%s)
  local duration=$((now_sec - SCRIPT_START_TIME))
  echo "当前时间：${now}，脚本总运行时间：${duration}秒，当前步骤：${step}"
}
print_step_info "1.action开始执行"
# 检查是否已成功转换
HAS_ROSBAG_SUCCESS=$(cocli record describe "$COS_RECORDID" -o json | jq -r '.labels[]?.display_name' | grep -w hdf5_success || true)
if [[ -n "$HAS_ROSBAG_SUCCESS" ]]; then
  echo "✅ 当前记录已存在 hdf5_success 标签，跳过 Step1~Step4"
  echo "🎉 所有处理完成！（无需重复转换）"
  # Step 5: 复制 hdf5 文件夹到指定记录的 /result 目录
  echo "========== Step 5: 复制到指定记录 =========="
  # echo "目标项目ID: $projectName"
  echo "目标记录ID: $recordName"

  echo "上传整个 hdf5 文件夹"
  # 找出 export/hdf5 下的唯一文件夹（uuid文件夹）
  UUID_FOLDER=$(ls -d "$COS_FILE_VOLUME/export/hdf5"/*/ 2>/dev/null | head -n1)
  if [[ -z "$UUID_FOLDER" ]]; then
    echo "❌ 未找到 export/hdf5 下的文件夹"
    exit 1
  fi
  UUID_NAME=$(basename "$UUID_FOLDER")
  echo "找到 UUID 文件夹: $UUID_NAME"

  # 复制到根目录
  cp -r "$UUID_FOLDER" "$COS_FILE_VOLUME/"
  echo "已复制到 $COS_FILE_VOLUME/$UUID_NAME"

  # # 列出所有文件的相对路径
  # cd "$COS_FILE_VOLUME"
  # file_to_copy=$(find "$UUID_NAME" -type f | tr '\n' ',' | sed 's/,$//')
  # echo "要上传的文件: $file_to_copy"

  # 上传文件
  #if cocli record file copy "$COS_RECORDID" "$recordName" --files "$file_to_copy" -f ; then
  if cocli record file copy "$COS_RECORDID" "$recordName" --files "$UUID_NAME/" -f ; then
    echo "✅ 已上传文件到 $recordName"
  else
    echo "❌ 上传失败"
    rm -rf "$COS_FILE_VOLUME/$UUID_NAME"
    exit 1
  fi
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
  exit 0
fi
# 给记录打标签

# 给记录打标签（允许没有 customer 时不打标签）
if [[ -n "${customer:-}" ]]; then
  if cocli record update "$COS_RECORDID" --append-labels "$customer"; then
    echo "✅ 已成功添加标签 customer"
  else
    echo "❌ 添加标签失败"
    exit 1
  fi
fi

# Step 1: 获取当前记录的 moments.json 文件
echo "========== Step 1: 获取 Moments 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "文件存储路径: $COS_FILE_VOLUME"
echo "项目ID: $COS_PROJECTID"
echo ""

# 获取 moments.json（如果已存在则跳过下载）
if [[ -f "$COS_FILE_VOLUME/moments.json" ]]; then
  echo "⚠️ 已存在 moments.json，跳过下载"
else
  echo "正在获取记录$COS_RECORDID的 moments 数据..."
  if cocli record list-moments "$COS_RECORDID" -o json | jq '{
    moments: [
      .events[] | {
        triggerTime: .triggerTime,
        duration: .duration,
        customFieldValues: (
          .customFieldValues | map({
            (.property.name): .text.value
          }) | add
        )
      }
    ]
  }' > "$COS_FILE_VOLUME/moments.json"; then
    # 判断 moments 是否有有效内容
  MOMENTS_COUNT=$(jq -r '.moments | length' "$COS_FILE_VOLUME/moments.json" 2>/dev/null || echo "0")
  if [[ "$MOMENTS_COUNT" -eq 0 ]]; then
    echo "❌ moments.json 中没有任何有效记录，请检查数据源！"
    rm -f "$COS_FILE_VOLUME/moments.json"
    exit 1
  fi
    echo "✅ Moments 数据已成功保存到: $COS_FILE_VOLUME/moments.json"
    echo "📊 发现 $MOMENTS_COUNT 个 moment 事件"
  else
    echo "❌ 获取 moments 数据失败"
    rm -f "$COS_FILE_VOLUME/moments.json"
    exit 1
  fi
fi

# 获取 metadata.json（如果已存在则跳过下载）

if [[ -f "$COS_FILE_VOLUME/metadata.json" ]]; then
  echo "⚠️ 已存在 metadata.json，跳过下载"
else
curl --location --request GET "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
  --header "Authorization: Basic ${basicAuth}" \
  --header 'Accept: */*' \
  --header 'Host: openapi.coscene.cn' \
  --header 'Connection: keep-alive' | jq 'if .customFieldValues == null then {} else .customFieldValues | map({(.property.name): .text.value}) | add end' >  $COS_FILE_VOLUME/metadata.json
fi

# Step 2: 执行 ROSbag 到 HDF5 转换
echo "========== Step 2: ROSbag 转换处理 =========="
echo "转换脚本: cvt_rosbag2hdf5.py"
echo "输入目录: $COS_FILE_VOLUME"
echo "输出目录: $COS_FILE_VOLUME/export/hdf5"
echo ""
print_step_info "2.python转化脚本开始执行"
# 检查必要的输入文件
echo "检查输入文件..."
if [[ ! -f "$COS_FILE_VOLUME/moments.json" ]]; then
  echo "❌ 缺少 moments.json 文件"
  exit 1
fi

if [[ ! -f "$COS_FILE_VOLUME/metadata.json" ]]; then
  echo "❌ 缺少 metadata.json 文件"
  exit 1
fi

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

if timeout 36000 python3 cvt_rosbag2hdf5.py \
  --bag_dir "$COS_FILE_VOLUME" \
  --moment_json_dir "$COS_FILE_VOLUME/moments.json" \
  --metadata_json_dir "$COS_FILE_VOLUME/metadata.json" \
  --output_dir $OUTPUT_DIR \
  --scene "test_scene" \
  --sub_scene "test_sub_scene" \
  --continuous_action "test_continuous_action" \
  --mode "simplified"; then

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
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
  echo "❌ ROSbag 转换失败"
  exit 1
fi
echo ""

# Step 3: 处理转换结果
echo "========== Step 3: 处理转换结果 =========="
echo "开始处理输出文件..."
print_step_info "3.在当前记录中保存结果"
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
  if [[ -d "$COS_FILE_VOLUME/export/hdf5" ]]; then
    echo "⚠️ 发现已存在 hdf5 文件夹，正在删除..."
    rm -rf "$COS_FILE_VOLUME/export/hdf5"
  fi
  mkdir -p "$COS_FILE_VOLUME/export/hdf5"
  if mv $OUTPUT_DIR/* "$COS_FILE_VOLUME/export/hdf5/"; then
    echo "✅ 成功移动文件到 $COS_FILE_VOLUME/export/hdf5"
    # 清理空的临时目录
    rm -rf $OUTPUT_DIR
    echo "✅ 临时目录清理完成"
  else
    echo "❌ 移动文件失败"
    exit 1
  fi
fi

# Step 4: 给当前记录打 hdf5_success 标签
echo "========== Step 4: 打标签 =========="
echo "为当前记录 $COS_RECORDID 添加标签 hdf5_success ..."
if cocli record update "$COS_RECORDID" --append-labels hdf5_success; then
  echo "✅ 已成功添加标签 hdf5_success"
else
  echo "❌ 添加标签失败"
  exit 1
fi
print_step_info "4.上传结果到指定记录"
# Step 5: 复制 hdf5 文件夹到指定记录的 /result 目录
echo "========== Step 5: 复制到指定记录 =========="
# echo "目标项目ID: $projectName"
echo "目标记录ID: $recordName"

echo "上传整个 hdf5 文件夹"

# 找出 export/hdf5 下的唯一文件夹（uuid文件夹）
UUID_FOLDER=$(ls -d "$COS_FILE_VOLUME/export/hdf5"/*/ 2>/dev/null | head -n1)
if [[ -z "$UUID_FOLDER" ]]; then
  echo "❌ 未找到 export/hdf5 下的文件夹"
  exit 1
fi
UUID_NAME=$(basename "$UUID_FOLDER")
echo "找到 UUID 文件夹: $UUID_NAME"

# 复制到根目录
cp -r "$UUID_FOLDER" "$COS_FILE_VOLUME/"
echo "已复制到 $COS_FILE_VOLUME/$UUID_NAME"

# # 列出所有文件的相对路径
# cd "$COS_FILE_VOLUME"
# file_to_copy=$(find "$UUID_NAME" -type f | tr '\n' ',' | sed 's/,$//')
# echo "要上传的文件: $file_to_copy"

# 上传文件
#if cocli record file copy "$COS_RECORDID" "$recordName" --files "$file_to_copy" -f; then
if cocli record file copy "$COS_RECORDID" "$recordName" --files "$UUID_NAME/" -f ; then
  echo "✅ 已上传文件到 $recordName"
else
  echo "❌ 上传失败"
  rm -rf "$COS_FILE_VOLUME/$UUID_NAME"
  exit 1
fi

# 删除临时复制的文件夹
rm -rf "$COS_FILE_VOLUME/$UUID_NAME"
echo "已删除临时文件夹"


echo "🎉 所有处理完成！"
print_step_info "5.全部完成"
exit 0
