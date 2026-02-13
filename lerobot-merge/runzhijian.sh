#!/usr/bin/env bash
set -uo pipefail  # 保留 -u 和 -o pipefail，但移除 -e
IFS=$'\n\t'

# === 配置区 ===
ROOT_DIR="/home/leju_kuavo/temp-copy/lerobot"
VALIDATE_SCRIPT="lerobot_qc/validator_local.py"
CONFIG_CLAW="lerobot_qc/config/custom_leju_kuavo4pro_claw.yaml"
OSS_CONFIG="lerobot_qc/config/oss_config.yaml"

# 检查必要文件
for f in "$VALIDATE_SCRIPT" "$CONFIG_CLAW"; do
    if [ ! -f "$f" ]; then
        echo "❌ 缺失文件: $f" >&2
        exit 1
    fi
done

if [ ! -d "$ROOT_DIR" ]; then
    echo "❌ ROOT_DIR 不存在: $ROOT_DIR" >&2
    exit 1
fi

# === 获取所有非隐藏子目录（健壮方式）===
echo "🔍 查找 $ROOT_DIR 下的所有 episode 目录..."
mapfile -t episode_dirs < <(find "$ROOT_DIR" -maxdepth 1 -mindepth 1 -type d ! -name ".*" | sort)

if [ ${#episode_dirs[@]} -eq 0 ]; then
    echo "⚠️ 未找到任何非隐藏子目录"
    exit 0
fi

echo "✅ 找到 ${#episode_dirs[@]} 个 episode 目录，开始质检..."

deleted_count=0
success_count=0
total=${#episode_dirs[@]}

for idx in "${!episode_dirs[@]}"; do
    episode="${episode_dirs[$idx]}"
    echo "==============================================="
    echo "📦 [$((idx + 1))/$total] 质检子集: $episode"

    report_base="/home/leju_kuavo/report"
    uuid_name=$(basename "$episode")
    report_dir="$report_base/$uuid_name"
    mkdir -p "$report_dir"

    echo "🔍 质检中: $episode → 报告将保存至 $report_dir"

    # === 关键：显式运行并捕获退出码，不依赖 set -e ===
    if timeout 3600 python3 "$VALIDATE_SCRIPT" \
        --dataset "$episode" \
        --config "$CONFIG_CLAW" \
        --output "$report_dir" \
        --oss-config "$OSS_CONFIG"; then
        echo "✅ 质检成功: $episode"
        ((success_count++))
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "⚠️ 质检超时（>3600秒）: $episode"
        else
            echo "⚠️ 质检失败（退出码: $exit_code）: $episode"
        fi
        echo "🗑️ 正在删除无效 episode 目录: $episode"
        rm -rf "$episode"
        ((deleted_count++))
    fi
done

echo "================================================"
echo "🎉 质检完成！"
echo "   ✅ 成功保留: $success_count 个"
echo "   🗑️ 已删除失败: $deleted_count 个"
echo "   💾 剩余有效数据位于: $ROOT_DIR"