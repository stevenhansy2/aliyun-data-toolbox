#!/usr/bin/env bash

set -euo pipefail
IFS="$(printf "\n\t")"

INPUT_DIR="${INPUT_DIR:-/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"
STAGING_MODE="${STAGING_MODE:-cp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT1="${SCRIPT_DIR}/merge_local_metadata.py"
MERGE_SCRIPT2="${SCRIPT_DIR}/adjust_file_size.py"

usage() {
  cat <<USAGE
用法:
  ./run.sh [原始数据根目录] [分类输出根目录]

环境变量:
  INPUT_DIR      默认 /inputs
  OUTPUT_DIR     默认 /outputs
  STAGING_MODE   默认 cp，可选 cp 或 mv

说明:
  1. 递归查找类似下面结构中的 uuid 目录:
     <bag目录>/.../export/hdf5/<data_id>/<uuid>/metadata.json
  2. 按 metadata.json 中的字段归类到:
     <分类输出根目录>/<eef_type>/<scene_name>/<sub_scene_name>/<english_task_name>/<uuid>
  3. 第一阶段按 STAGING_MODE=cp 复制，或 STAGING_MODE=mv 剪切
  4. 归类完成后，对每个 eef_type 目录继续执行目录规范化和统计重命名

示例:
  INPUT_DIR=/data/source OUTPUT_DIR=/data/classified_hdf5 STAGING_MODE=cp ./run.sh
  STAGING_MODE=mv ./run.sh /data/source /data/classified_hdf5
  ./run.sh /data/source /data/classified_hdf5
USAGE
}

if [[ $# -gt 2 ]]; then
  usage
  exit 1
fi

SOURCE_ROOT="${1:-$INPUT_DIR}"
OUTPUT_ROOT="${2:-$OUTPUT_DIR}"
SOURCE_ROOT="${SOURCE_ROOT%/}"
OUTPUT_ROOT="${OUTPUT_ROOT%/}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "❌ 原始数据根目录不存在: $SOURCE_ROOT"
  exit 1
fi

if [[ "$STAGING_MODE" != "cp" && "$STAGING_MODE" != "mv" ]]; then
  echo "❌ STAGING_MODE 仅支持 cp 或 mv，当前值: $STAGING_MODE"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "❌ 未找到 jq，请先安装 jq"
  exit 1
fi

if [[ ! -f "$MERGE_SCRIPT1" || ! -f "$MERGE_SCRIPT2" ]]; then
  echo "❌ 缺少依赖脚本，请确认以下文件存在:"
  echo "   $MERGE_SCRIPT1"
  echo "   $MERGE_SCRIPT2"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

SOURCE_ROOT_ABS="$(cd "$SOURCE_ROOT" && pwd)"
OUTPUT_ROOT_ABS="$(cd "$OUTPUT_ROOT" && pwd)"

sanitize_field() {
  printf "%s" "${1:-}" | tr " " "_" | tr -d "\r\n"
}

ensure_unique_path() {
  local target="$1"
  if [[ ! -e "$target" ]]; then
    printf "%s\n" "$target"
    return 0
  fi

  local parent base candidate idx
  parent="$(dirname "$target")"
  base="$(basename "$target")"
  idx=1

  while true; do
    candidate="${parent}/${base}_dup${idx}"
    if [[ ! -e "$candidate" ]]; then
      printf "%s\n" "$candidate"
      return 0
    fi
    ((idx++))
  done
}

capitalize_dir() {
  local dir="$1"
  dir="${dir%/}"
  [[ -d "$dir" ]] || return 0

  local name cap parent
  name="$(basename "$dir")"
  cap="${name^}"

  [[ "$name" == "$cap" ]] && return 0

  parent="$(dirname "$dir")"
  if [[ -e "$parent/$cap" && "$parent/$cap" != "$dir" ]]; then
    echo "⚠️ 目标已存在，跳过重命名: $parent/$cap"
    return 0
  fi

  mv "$dir" "$parent/$cap"
}

post_process_eef_root() {
  local target_dir="$1"
  local base
  local target
  local prefix
  local local_base_name
  local basename_dir
  local field
  local p
  local dir

  [[ -d "$target_dir" ]] || return 0

  for depth in 3 2 1; do
    while IFS= read -r p; do
      [[ -n "$p" ]] || continue
      capitalize_dir "$p"
    done < <(find "$target_dir" -mindepth "$depth" -maxdepth "$depth" -type d | sort)
  done

  echo "后处理目录: $target_dir"
  base="$(basename "$target_dir")"

  if [[ "$base" == *dex_hand* ]]; then
    prefix="Kuavo4Pro-Dexhand"
  elif [[ "$base" == *leju_claw* ]]; then
    prefix="Kuavo4Pro-Lejuclaw"
  else
    echo "⚠️ 目录 $base 不包含 dex_hand 或 leju_claw，跳过后处理"
    return 0
  fi

  (
    cd "$target_dir" || exit 1
    for d in */; do
      [[ -d "$d" ]] || continue
      local_base_name="${d%/}"
      target="${prefix}-${local_base_name}"
      mkdir -p "$target"
      mv -n "$d" "$target/"
    done
  )

  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    basename_dir="$(basename "$dir")"
    field="$(printf "%s" "$basename_dir" | cut -d- -f3-)"

    if [[ -z "$field" ]]; then
      echo "跳过 $dir（名字里不足 2 个 - ）"
      continue
    fi

    echo "处理 $dir >>> 字段: $field"
    python "$MERGE_SCRIPT1" "$dir" "$field"
    python "$MERGE_SCRIPT2" "$dir" "$field"
  done < <(find "$target_dir" -mindepth 1 -maxdepth 1 -type d | sort)
}

moved_count=0
skipped_count=0
already_classified_count=0
declare -A seen_eef_types

while IFS= read -r metadata_path; do
  [[ -n "$metadata_path" ]] || continue

  case "$metadata_path" in
    "$OUTPUT_ROOT_ABS"/*)
      continue
      ;;
  esac

  uuid_dir="$(dirname "$metadata_path")"
  uuid_name="$(basename "$uuid_dir")"

  scene_name="$(jq -r ".scene_name // empty" "$metadata_path")"
  sub_scene_name="$(jq -r ".sub_scene_name // empty" "$metadata_path")"
  english_task_name="$(jq -r ".english_task_name // empty" "$metadata_path")"
  eef_type="$(jq -r ".eefType // empty" "$metadata_path")"

  scene_name="$(sanitize_field "$scene_name")"
  sub_scene_name="$(sanitize_field "$sub_scene_name")"
  english_task_name="$(sanitize_field "$english_task_name")"
  eef_type="$(sanitize_field "$eef_type")"

  if [[ -z "$scene_name" || -z "$sub_scene_name" || -z "$english_task_name" || -z "$eef_type" ]]; then
    echo "⚠️ 跳过 $uuid_name，metadata.json 缺少必要字段"
    ((skipped_count += 1))
    continue
  fi

  target_parent="${OUTPUT_ROOT_ABS}/${eef_type}/${scene_name}/${sub_scene_name}/${english_task_name}"
  mkdir -p "$target_parent"

  target_uuid_path="${target_parent}/${uuid_name}"
  if [[ "$uuid_dir" == "$target_uuid_path" ]]; then
    echo "ℹ️ 已在目标位置，跳过: $uuid_dir"
    ((already_classified_count += 1))
    continue
  fi

  target_uuid_path="$(ensure_unique_path "$target_uuid_path")"
  if [[ "$STAGING_MODE" == "mv" ]]; then
    echo "剪切归类 $uuid_name -> $target_uuid_path"
    mv "$uuid_dir" "$target_uuid_path"
  else
    echo "复制归类 $uuid_name -> $target_uuid_path"
    cp -a "$uuid_dir" "$target_uuid_path"
  fi

  seen_eef_types["$eef_type"]=1
  ((moved_count += 1))
done < <(find "$SOURCE_ROOT_ABS" -type f -name metadata.json | sort)

echo "✅ 第一阶段处理完成（模式: ${STAGING_MODE}）"
echo "   已处理: ${moved_count}"
echo "   已跳过: ${skipped_count}"
echo "   已在目标位置: ${already_classified_count}"

for eef_type in "${!seen_eef_types[@]}"; do
  post_process_eef_root "${OUTPUT_ROOT_ABS}/${eef_type}"
done

echo "✅ 全部处理完成"
