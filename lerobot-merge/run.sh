#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# === 配置区 ===
INPUT_DIR="${INPUT_DIR:-/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"
MERGE_SCRIPT="${MERGE_SCRIPT:-}"
DEBUG_SLEEP_SECONDS="${DEBUG_SLEEP_SECONDS:-0}"
TARGET_SCRIPT_NAME="${TARGET_SCRIPT_NAME:-}"
TARGET_SCRIPT_NAMES="${TARGET_SCRIPT_NAMES:-}"
STAGING_MODE="${STAGING_MODE:-copy}"  # copy|symlink ; readonly input should use copy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_merge_script() {
    # 优先使用显式传入路径
    if [[ -n "$MERGE_SCRIPT" ]]; then
        if [[ -f "$MERGE_SCRIPT" ]]; then
            return 0
        fi
        if [[ -f "$SCRIPT_DIR/$MERGE_SCRIPT" ]]; then
            MERGE_SCRIPT="$SCRIPT_DIR/$MERGE_SCRIPT"
            return 0
        fi
    fi

    # 自动探测常见位置（当前仓库是 kuavo 目录）
    if [[ -f "$SCRIPT_DIR/kuavo/merge_data.py" ]]; then
        MERGE_SCRIPT="$SCRIPT_DIR/kuavo/merge_data.py"
        return 0
    fi
    if [[ -f "$SCRIPT_DIR/lerobot_qc/merge_data.py" ]]; then
        MERGE_SCRIPT="$SCRIPT_DIR/lerobot_qc/merge_data.py"
        return 0
    fi

    echo "❌ 未找到 merge_data.py，请设置 MERGE_SCRIPT。"
    return 1
}

validate_merged_output() {
    local merged_dir="$1"
    if [[ ! -d "$merged_dir/meta" ]]; then
        echo "❌ 合并结果缺失目录: $merged_dir/meta"
        return 1
    fi
    if [[ ! -f "$merged_dir/meta/info.json" ]]; then
        echo "❌ 合并结果缺失文件: $merged_dir/meta/info.json"
        return 1
    fi
    if [[ ! -d "$merged_dir/data/chunk-000" ]]; then
        echo "❌ 合并结果缺失目录: $merged_dir/data/chunk-000"
        return 1
    fi
    if ! compgen -G "$merged_dir/data/chunk-000/episode_*.parquet" > /dev/null; then
        echo "❌ 合并结果缺少文件: $merged_dir/data/chunk-000/episode_*.parquet"
        return 1
    fi
    return 0
}

# 判断一个目录是否为 LeRobot 数据集目录（最小结构校验）
is_lerobot_dataset_dir() {
    local dir="$1"
    [[ -d "$dir/meta" ]] \
        && [[ -f "$dir/meta/info.json" ]] \
        && [[ -f "$dir/meta/episodes.jsonl" ]] \
        && [[ -f "$dir/meta/episodes_stats.jsonl" ]] \
        && [[ -f "$dir/meta/tasks.jsonl" ]] \
        && [[ -d "$dir/data/chunk-000" ]] \
        && [[ -d "$dir/videos/chunk-000" ]] \
        && [[ -d "$dir/parameters" ]]
}

# 递归查找“可作为 merge_data.py --src_dir 的目录”：
# 即该目录的一级子目录中，至少有一个是有效 LeRobot 数据集
find_merge_sources() {
    local root="$1"
    local candidate_dir child_dir found_valid_child

    while IFS= read -r -d '' candidate_dir; do
        found_valid_child=0
        while IFS= read -r -d '' child_dir; do
            if is_lerobot_dataset_dir "$child_dir"; then
                found_valid_child=1
                break
            fi
        done < <(find "$candidate_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

        if [[ $found_valid_child -eq 1 ]]; then
            printf '%s\0' "$candidate_dir"
        fi
    done < <(find "$root" -type d -print0 2>/dev/null)
}

# 仅保留最深层可处理目录，避免同时处理父/子目录导致结果异常
filter_deepest_sources() {
    local candidates=("$@")
    local i j is_parent
    local deepest=()

    for ((i = 0; i < ${#candidates[@]}; i++)); do
        is_parent=0
        for ((j = 0; j < ${#candidates[@]}; j++)); do
            if [[ $i -ne $j && "${candidates[j]}" == "${candidates[i]}"/* ]]; then
                is_parent=1
                break
            fi
        done
        if [[ $is_parent -eq 0 ]]; then
            deepest+=("${candidates[i]}")
        fi
    done

    if [[ ${#deepest[@]} -gt 0 ]]; then
        printf '%s\0' "${deepest[@]}"
    fi
}

# 查找 source_dir 下可参与合并的数据集子目录数量（一级）
count_dataset_children() {
    local source_dir="$1"
    local child_dir count=0
    while IFS= read -r -d '' child_dir; do
        if is_lerobot_dataset_dir "$child_dir"; then
            count=$((count + 1))
        fi
    done < <(find "$source_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    echo "$count"
}

print_dataset_issues() {
    local dir="$1"
    local issues=()
    local f

    [[ -d "$dir/meta" ]] || issues+=("缺失目录: meta")
    for f in info.json episodes.jsonl episodes_stats.jsonl tasks.jsonl; do
        [[ -f "$dir/meta/$f" ]] || issues+=("缺失文件: meta/$f")
    done
    for f in episodes.jsonl episodes_stats.jsonl tasks.jsonl; do
        if [[ -f "$dir/meta/$f" ]] && [[ ! -s "$dir/meta/$f" ]]; then
            issues+=("空文件: meta/$f")
        fi
    done

    [[ -d "$dir/data/chunk-000" ]] || issues+=("缺失目录: data/chunk-000")
    compgen -G "$dir/data/chunk-000/episode_*.parquet" > /dev/null || issues+=("缺少文件: data/chunk-000/episode_*.parquet")

    for cam in camera_top camera_wrist_left camera_wrist_right; do
        [[ -d "$dir/videos/chunk-000/observation.images.${cam}" ]] || issues+=("缺失目录: videos/chunk-000/observation.images.${cam}")
        compgen -G "$dir/videos/chunk-000/observation.images.${cam}/episode_*.mp4" > /dev/null || issues+=("缺少文件: videos/chunk-000/observation.images.${cam}/episode_*.mp4")
    done

    [[ -d "$dir/parameters" ]] || issues+=("缺失目录: parameters")
    for f in \
        camera_top_extrinsic.json \
        camera_top_intrinsic.json \
        camera_wrist_left_extrinsic.json \
        camera_wrist_left_intrinsic.json \
        camera_wrist_right_extrinsic.json \
        camera_wrist_right_intrinsic.json; do
        [[ -f "$dir/parameters/$f" ]] || issues+=("缺失文件: parameters/$f")
    done

    if [[ ${#issues[@]} -eq 0 ]]; then
        echo "    ✅ 结构完整"
    else
        echo "    ❌ 结构不完整，问题数=${#issues[@]}"
        for issue in "${issues[@]}"; do
            echo "      - $issue"
        done
    fi
}

diagnose_source_dir() {
    local source_dir="$1"
    local child_dir
    local child_count=0

    echo "🔬 诊断 source_dir: $source_dir"
    while IFS= read -r -d '' child_dir; do
        child_count=$((child_count + 1))
        echo "  [子目录 $child_count] $child_dir"
        print_dataset_issues "$child_dir"
    done < <(find "$source_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

    if [[ $child_count -eq 0 ]]; then
        echo "  ⚠️ 该目录下没有一级子目录"
    fi
}

# 将脚本名配置解析为数组，支持逗号或空格分隔
parse_target_scripts() {
    local raw="$1"
    local normalized
    normalized="$(echo "$raw" | tr ',' ' ')"
    read -r -a TARGET_SCRIPTS <<< "$normalized"
}

# 在 /inputs/<data_id>/<script_name> 下收集需要检索的根目录
collect_search_roots() {
    local input_root="$1"
    shift
    local script_names=("$@")
    local data_id_dir script_dir script_name

    while IFS= read -r -d '' data_id_dir; do
        for script_name in "${script_names[@]}"; do
            script_dir="$data_id_dir/$script_name"
            if [[ -d "$script_dir" ]]; then
                printf '%s\0' "$script_dir"
            fi
        done
    done < <(find "$input_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
}

# === 主流程 ===
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "❌ INPUT_DIR 不存在或不是目录: $INPUT_DIR"
    exit 1
fi

# TARGET_SCRIPT_NAME 是平台强约束，优先单值变量
if [[ -n "$TARGET_SCRIPT_NAME" ]]; then
    TARGET_SCRIPT_NAMES="$TARGET_SCRIPT_NAME"
fi
if [[ -z "${TARGET_SCRIPT_NAMES// }" ]]; then
    echo "❌ 缺失 TARGET_SCRIPT_NAME（或 TARGET_SCRIPT_NAMES）。"
    exit 1
fi

if ! resolve_merge_script; then
    exit 1
fi
echo "🧩 使用合并脚本: $MERGE_SCRIPT"
echo "🧩 staging 模式: $STAGING_MODE"

mkdir -p "$OUTPUT_DIR"
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "❌ OUTPUT_DIR 不可写: $OUTPUT_DIR"
    exit 1
fi

echo "🔍 递归查找 $INPUT_DIR 下可合并的源目录..."
echo "📁 $INPUT_DIR 下的全部目录列表："
find "$INPUT_DIR" -type d | sort

if [[ "$DEBUG_SLEEP_SECONDS" =~ ^[0-9]+$ ]] && [[ "$DEBUG_SLEEP_SECONDS" -gt 0 ]]; then
    echo "⏸️ 调试暂停 ${DEBUG_SLEEP_SECONDS} 秒（约 $((DEBUG_SLEEP_SECONDS / 60)) 分钟），可进入容器检查目录结构..."
    sleep "$DEBUG_SLEEP_SECONDS"
fi

TARGET_SCRIPTS=()
parse_target_scripts "$TARGET_SCRIPT_NAMES"

search_roots=()
if [[ ${#TARGET_SCRIPTS[@]} -gt 0 ]]; then
    echo "🎯 限定脚本目录: ${TARGET_SCRIPTS[*]}"
    mapfile -d '' -t search_roots < <(collect_search_roots "$INPUT_DIR" "${TARGET_SCRIPTS[@]}" | sort -z)
    if [[ ${#search_roots[@]} -eq 0 ]]; then
        echo "❌ 未找到指定脚本目录: ${TARGET_SCRIPTS[*]}"
        exit 1
    fi
else
    search_roots=("$INPUT_DIR")
fi

echo "🔎 实际检索根目录数量: ${#search_roots[@]}"
for r in "${search_roots[@]}"; do
    echo "  - $r"
done

all_source_dirs=()
for root in "${search_roots[@]}"; do
    mapfile -d '' -t root_sources < <(find_merge_sources "$root" | sort -z)
    all_source_dirs+=("${root_sources[@]}")
done
if [[ ${#all_source_dirs[@]} -gt 0 ]]; then
    mapfile -d '' -t all_source_dirs < <(printf '%s\0' "${all_source_dirs[@]}" | sort -zu)
else
    all_source_dirs=()
fi

mapfile -d '' -t source_dirs < <(filter_deepest_sources "${all_source_dirs[@]}" | sort -z)

# 兜底移除空路径，避免空元素被当成一个目录
valid_source_dirs=()
for source_dir in "${source_dirs[@]}"; do
    if [[ -n "$source_dir" ]]; then
        valid_source_dirs+=("$source_dir")
    fi
done
source_dirs=("${valid_source_dirs[@]}")

if [ ${#source_dirs[@]} -eq 0 ]; then
    echo "❌ 未找到可处理目录。请确认目录中存在标准 LeRobot 数据集结构。"
    exit 1
fi

echo "✅ 找到 ${#source_dirs[@]} 个可处理目录。"
for s in "${source_dirs[@]}"; do
    echo "  - $s"
done

all_dataset_dirs=()
for source_dir in "${source_dirs[@]}"; do
    dataset_count="$(count_dataset_children "$source_dir")"
    diagnose_source_dir "$source_dir"
    if [[ "$dataset_count" -eq 0 ]]; then
        echo "⚠️ 跳过目录（无有效数据集子目录）: $source_dir"
        continue
    fi
    while IFS= read -r -d '' child_dir; do
        if is_lerobot_dataset_dir "$child_dir"; then
            all_dataset_dirs+=("$child_dir")
        fi
    done < <(find "$source_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
done

if [[ ${#all_dataset_dirs[@]} -eq 0 ]]; then
    echo "❌ 没有可用于全局合并的 LeRobot 数据集子目录。"
    exit 1
fi

mapfile -d '' -t all_dataset_dirs < <(printf '%s\0' "${all_dataset_dirs[@]}" | sort -zu)
echo "📦 收集到 ${#all_dataset_dirs[@]} 个数据集子目录，将执行一次全局合并。"
for d in "${all_dataset_dirs[@]}"; do
    echo "  - $d"
done

staging_dir="$(mktemp -d /tmp/lerobot_merge_staging.XXXXXX)"
cleanup_staging() {
    chmod -R u+w "$staging_dir" 2>/dev/null || true
    rm -rf "$staging_dir"
}
trap cleanup_staging EXIT

idx=1
for d in "${all_dataset_dirs[@]}"; do
    target="$staging_dir/dataset_$(printf '%04d' "$idx")"
    if [[ "$STAGING_MODE" == "symlink" ]]; then
        ln -s "$d" "$target"
    else
        cp -a "$d" "$target"
        chmod -R u+w "$target" 2>/dev/null || true
    fi
    idx=$((idx + 1))
done

merged_dir="$OUTPUT_DIR/lerobot_merged"
echo "========== 全局合并 =========="
echo "汇总目录: $staging_dir"
echo "输出目录: $merged_dir"

rm -rf "$merged_dir"
mkdir -p "$(dirname "$merged_dir")"

if python3 "$MERGE_SCRIPT" \
    --src_dir "$staging_dir" \
    --tgt_dir "$merged_dir" \
    --save; then
    if validate_merged_output "$merged_dir"; then
        echo "✅ 全局合并完成: $merged_dir"
    else
        echo "❌ 合并结果校验失败。"
        exit 1
    fi
else
    echo "❌ 全局合并失败。"
    exit 1
fi

echo "🎉 全部处理完成，输出目录: $merged_dir"
