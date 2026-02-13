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

#记录时间
SCRIPT_START_TIME=$(date +%s)

print_step_info() {
  local step="$1"
  local now=$(date "+%Y-%m-%d %H:%M:%S")
  local now_sec=$(date +%s)
  local duration=$((now_sec - SCRIPT_START_TIME))
  echo "当前时间：${now}，脚本总运行时间：${duration}秒，当前步骤：${step}"
}

# Step 1: 通过环境变量生成 ossutil 配置文件
echo "========== Step 1: 生成 ossutil 配置文件 =========="
# cat > ~/.ossutilconfig <<EOF
# [default]
# accessKeyId=${ACCESS_KEY_ID}
# accessKeySecret=${ACCESS_KEY_SECRET}
# region=cn-hangzhou
# endpoint=${ENDPOINT}
# EOF

print_step_info "1.action开始执行"

# 检查是否已成功转换
HAS_ROSBAG_SUCCESS=$(cocli record describe "$COS_RECORDID" -o json | jq -r '.labels[]?.display_name' | grep -w hdf5_success || true)
# 在 Step 0 的跳过转换部分（大约第85行附近）
if [[ -n "$HAS_ROSBAG_SUCCESS" ]]; then
  echo "✅ 当前记录已存在 hdf5_success 标签，跳过 Step1~Step4"
  echo "🎉 所有处理完成！（无需重复转换）"
  # Step 6: 上传 $OUTPUT_DIR 到 oss
  # echo "========== Step 6: 上传到 oss =========="
  # print_step_info "6.上传到oss"
  # OSS_BUCKET="${OSS_BUCKET}"
  # # 获取唯一 uuid 文件夹
  # UUID_DIR=$(find "$COS_FILE_VOLUME/export/hdf5/" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  # METADATA_PATH="$UUID_DIR/metadata.json"

  # # 检查 metadata.json 是否存在
  # if [[ ! -f "$METADATA_PATH" ]]; then
  #   echo "❌ 未找到 metadata.json: $METADATA_PATH"
  #   exit 1
  # fi

  # # 提取字段并处理空格
  # scene_name=$(jq -r '.scene_name' "$METADATA_PATH" | sed 's/ /_/g')
  # sub_scene_name=$(jq -r '.sub_scene_name' "$METADATA_PATH" | sed 's/ /_/g')
  # english_task_name=$(jq -r '.english_task_name' "$METADATA_PATH" | sed 's/ /_/g')
  # EEF_METADATA_PATH="$COS_FILE_VOLUME/metadata.json"
  # eef_type=$(jq -r '.eef_type' "$EEF_METADATA_PATH" | sed 's/ /_/g')

  # # 检查字段有效性
  # if [[ -z "$scene_name" || -z "$sub_scene_name" || -z "$english_task_name" || -z "$eef_type" ]]; then
  #   echo "❌ metadata.json 缺少必要字段或字段为空"
  #   exit 1
  # fi
  # # 去掉 OSS_BUCKET 结尾的斜杠
  # OSS_BUCKET_CLEAN="${OSS_BUCKET%/}"

  # OSS_TARGET_DIR="${OSS_BUCKET_CLEAN}/${eef_type}/${scene_name}/${sub_scene_name}/${english_task_name}/"

  # echo "OSS 目标目录: $OSS_TARGET_DIR"

  # # 上传到 OSS 四级目录
  # ossutil cp -r -u "$COS_FILE_VOLUME/export/hdf5/" "$OSS_TARGET_DIR"

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

# Step 2: 获取当前记录的 moments.json 文件
echo "========== Step 2: 获取 Moments 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "文件存储路径: $COS_FILE_VOLUME"
echo "项目ID: $COS_PROJECTID"
echo ""

# 获取 moments.json（如果已存在则删除）
if [[ -f "$COS_FILE_VOLUME/moments.json" ]]; then
  echo "⚠️ 已存在 moments.json，删除后生成"
  rm -f "$COS_FILE_VOLUME/moments.json"
fi
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


# 获取 metadata.json（如果已存在则删除）

if [[ -f "$COS_FILE_VOLUME/metadata.json" ]]; then
  echo "⚠️ 已存在 metadata.json，删除"
  rm -f "$COS_FILE_VOLUME/metadata.json"
fi
curl --location --request GET "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
  --header "Authorization: Basic YXBpa2V5OllUVTVZVFpqTURZeU9EVmtOall6TVRNM1pHWTVaRFk1Wm1Zek16WTBPRGxrTnpOaFpqRTNNelF6WVRNME5EWmtPRGhqWlRObVpXTXhPV1ZqTUdJeU53PT0=" \
  --header 'Accept: */*' \
  --header 'Host: openapi.coscene.cn' \
  --header 'Connection: keep-alive' | jq 'if .customFieldValues == null then {} else .customFieldValues | map({(.property.name): .text.value}) | add end' >  $COS_FILE_VOLUME/metadata.json


# Step 3: 执行 ROSbag 到 HDF5 转换
echo "========== Step 3: ROSbag 转换处理 =========="
echo "转换脚本: cvt_rosbag2hdf5.py"
echo "输入目录: $COS_FILE_VOLUME"
echo "输出目录: $COS_FILE_VOLUME/export/hdf5"
echo ""
print_step_info "3.python转化脚本开始执行"
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
# 如果没有传入 MIN_DURATION，则自动设置为 5
if [[ -z "${MIN_DURATION:-}" ]]; then
  MIN_DURATION=5
  echo "⚠️ 未检测到 MIN_DURATION，自动设置为默认值 5 秒"
else
  echo "ℹ️ 检测到 MIN_DURATION，设置为 ${MIN_DURATION} 秒"
fi
if timeout 36000 python3 cvt_rosbag2hdf5.py \
  --bag_dir "$COS_FILE_VOLUME" \
  --moment_json_dir "$COS_FILE_VOLUME/moments.json" \
  --metadata_json_dir "$COS_FILE_VOLUME/metadata.json" \
  --output_dir $OUTPUT_DIR \
  --scene "test_scene" \
  --sub_scene "test_sub_scene" \
  --continuous_action "test_continuous_action" \
  --min_duration $MIN_DURATION\
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

# Step 4: 处理转换结果
echo "========== Step 4: 处理转换结果 =========="
echo "开始处理输出文件..."
print_step_info "4.在当前记录中保存结果"
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
  # step4只走这个情况3: 保存到当前记录的output目录
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

# Step 5: 给当前记录打 hdf5_success 标签
echo "========== Step 5: 打标签 =========="
print_step_info "5.打标签"
echo "为当前记录 $COS_RECORDID 添加标签 hdf5_success ..."
if cocli record update "$COS_RECORDID" --append-labels hdf5_success; then
  echo "✅ 已成功添加标签 hdf5_success"
else
  echo "❌ 添加标签失败"
  exit 1
fi


# # Step 6: 上传 $OUTPUT_DIR 到 oss
# echo "========== Step 6: 上传到 oss =========="
# print_step_info "6.上传到oss"
# OSS_BUCKET="${OSS_BUCKET}"
# # 获取唯一 uuid 文件夹
# UUID_DIR=$(find "$COS_FILE_VOLUME/export/hdf5/" -mindepth 1 -maxdepth 1 -type d | head -n 1)
# METADATA_PATH="$UUID_DIR/metadata.json"

# # 检查 metadata.json 是否存在
# if [[ ! -f "$METADATA_PATH" ]]; then
#   echo "❌ 未找到 metadata.json: $METADATA_PATH"
#   exit 1
# fi
# # 在 Step 6 的主要上传逻辑部分（大约第367行附近）
# # 提取字段并处理空格
# scene_name=$(jq -r '.scene_name' "$METADATA_PATH" | sed 's/ /_/g')
# sub_scene_name=$(jq -r '.sub_scene_name' "$METADATA_PATH" | sed 's/ /_/g')
# english_task_name=$(jq -r '.english_task_name' "$METADATA_PATH" | sed 's/ /_/g')
# EEF_METADATA_PATH="$COS_FILE_VOLUME/metadata.json"
# eef_type=$(jq -r '.eef_type' "$EEF_METADATA_PATH" | sed 's/ /_/g')

# # 检查字段有效性
# if [[ -z "$scene_name" || -z "$sub_scene_name" || -z "$english_task_name" || -z "$eef_type" ]]; then
#   echo "❌ metadata.json 缺少必要字段或字段为空"
#   exit 1
# fi

# # 去掉 OSS_BUCKET 结尾的斜杠
# OSS_BUCKET_CLEAN="${OSS_BUCKET%/}"

# OSS_TARGET_DIR="${OSS_BUCKET_CLEAN}/${eef_type}/${scene_name}/${sub_scene_name}/${english_task_name}/"

# echo "OSS 目标目录: $OSS_TARGET_DIR"

# # 上传到 OSS 四级目录
# ossutil cp -r -u "$COS_FILE_VOLUME/export/hdf5/" "$OSS_TARGET_DIR"


echo "🎉 所有处理完成！"
print_step_info "7.全部完成"
exit 0