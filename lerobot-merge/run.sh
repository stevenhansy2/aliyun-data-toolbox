#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# === 配置区 ===
# ROOT_DIR="/home/leju_kuavo/1234/"
ROOT_DIR="/media/leju_kuavo/My Passport/batch11/batch11/default/Kuavo_4Pro/04"
MERGE_SCRIPT="lerobot_qc/merge_data.py"
VALIDATE_SCRIPT="lerobot_qc/validator_local.py"
FETCH_SCRIPT="lerobot_qc/fetch_dataset_summaries.py"
CONFIG_CLAW="lerobot_qc/config/custom_leju_kuavo4pro_claw.yaml"
CONFIG_DEX="lerobot_qc/config/custom_leju_kuavo4pro.yaml"

# 检查脚本是否存在
for f in "$MERGE_SCRIPT" "$VALIDATE_SCRIPT" "$CONFIG_CLAW"; do
    if [ ! -f "$f" ]; then
        echo "❌ 缺失文件: $f"
        exit 1
    fi
done

# === 主流程 ===
echo "🔍 查找 $ROOT_DIR 下两层内的 lerobot 目录..."
mapfile -t lerobot_dirs < <(find "$ROOT_DIR" -mindepth 3 -maxdepth 8 -type d -name 'lerobot' 2>/dev/null | sort)

if [ ${#lerobot_dirs[@]} -eq 0 ]; then
    echo "⚠️ 未找到任何 lerobot 目录"
    exit 0
fi

echo "✅ 找到 ${#lerobot_dirs[@]} 个 lerobot 目录，开始处理..."

for lerobot in "${lerobot_dirs[@]}"; do
    echo "==============================================="
    echo "📦 处理: $lerobot"

    # 跳过已成功合并的
    if [ -f "$lerobot/.merged_success" ]; then
        echo "⏭️  已标记为已合并，跳过: $lerobot"
        continue
    fi

    # 跳过空目录
    if ! ls "$lerobot"/*/ &>/dev/null 2>&1; then
        echo "⚠️ 跳过空目录: $lerobot"
        continue
    fi
    
    # echo '$CONFIG_YAML'

    success_dir=$(dirname "$lerobot")
    report_dir="$success_dir/report"
    mkdir -p "$report_dir"

    merged_dir="$success_dir/lerobot_merged"
    rm -rf "$merged_dir"

    echo "🔄 合并数据到临时目录: $merged_dir"
    if ! python3 "$MERGE_SCRIPT" \
        --src_dir "$lerobot" \
        --tgt_dir "$merged_dir" \
        --summary_dir "$report_dir" \
        --save; then
        echo "❌ 合并失败，跳过质检: $lerobot"
        continue
    fi
    info_json="$merged_dir/meta/info.json"
    echo "$info_json"
    if [ -f "$info_json" ]; then
        if grep 'qiangnao' "$info_json"; then
            # echo "$CONFIG_DEX"
            CONFIG_YAML=$CONFIG_DEX
        else
            # echo "$CONFIG_CLAW"
            CONFIG_YAML=$CONFIG_CLAW
        fi
    else
        echo "⚠️ 未找到meta/info.json 跳过该目录: $lerobot"
        continue
    fi
    echo "🔍 质检中..."
    if timeout 3600 python3 "$VALIDATE_SCRIPT" \
        --dataset "$merged_dir" \
        --config "$CONFIG_YAML" \
        --output "$report_dir" \
        --oss-config "lerobot_qc/config/oss_config.yaml"; then

        echo "✅ 质检通过！替换原始 lerobot（无备份）"
        rm -rf "$lerobot"
        mv "$merged_dir" "$lerobot"
        touch "$lerobot/.merged_success"
        echo "🎉 替换完成！"
    else
        echo "❌ 质检未通过！保留原始数据，不作替换"
        echo "🗑️  清理临时合并目录: $merged_dir"
        rm -rf "$merged_dir"
        # 注意：这里不创建任何标记，所以下次运行会重试（如果你希望重试）
        # 如果你希望跳过失败项，可创建 .merged_failed 标记并检查它
    fi

    echo "📄 报告位置: $report_dir"
done

if timeout 3600 python3 "$FETCH_SCRIPT" \
    --root "$ROOT_DIR" \
    --output "$ROOT_DIR/fetch_all.csv" ; then
    echo "🎉 所有任务处理完毕！"
fi
