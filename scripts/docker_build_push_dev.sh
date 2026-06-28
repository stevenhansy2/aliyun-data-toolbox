#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_DB="$REPO_ROOT/.docker_dev_versions.json"
OLD_VERSION_DB="$REPO_ROOT/.docker_dev_versions.tsv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

read_required() {
  local __var_name="$1"
  local prompt="$2"
  local value=""
  while [[ -z "$value" ]]; do
    read -r -p "$prompt" value
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
  done
  printf -v "$__var_name" '%s' "$value"
}

read_secret() {
  local __var_name="$1"
  local prompt="$2"
  local value=""
  while [[ -z "$value" ]]; do
    read -r -s -p "$prompt" value
    printf '\n'
  done
  printf -v "$__var_name" '%s' "$value"
}

read_project_dir() {
  local __var_name="$1"
  local prompt="$2"
  local value=""
  local resolved=""

  while true; do
    read -r -p "$prompt" value
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ -n "$value" ]] || continue

    resolved="$(resolve_project_dir "$value")"
    if [[ -d "$resolved" ]]; then
      printf -v "$__var_name" '%s' "$value"
      return
    fi

    echo "❌ 项目目录不存在: $resolved"
    echo "   可用的一级目录:"
    find "$REPO_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "   - %f\n" | sort
    echo "请重新输入。"
    value=""
  done
}

resolve_project_dir() {
  local input="$1"
  if [[ "$input" = /* ]]; then
    printf '%s' "$input"
  else
    printf '%s/%s' "$REPO_ROOT" "$input"
  fi
}

next_version_number() {
  local image_name="$1"
  "$PYTHON_BIN" - "$VERSION_DB" "$image_name" <<'PY'
import json
import re
import sys
from pathlib import Path

db_path = Path(sys.argv[1])
image_name = sys.argv[2]
max_version = 0

if db_path.exists():
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    for item in data.get("images", {}).get(image_name, []):
        match = re.fullmatch(r"dev-v(\d+)", str(item.get("version", "")))
        if match:
            max_version = max(max_version, int(match.group(1)))

print(max_version + 1)
PY
}

show_history() {
  local image_name="$1"

  if [[ ! -f "$VERSION_DB" ]]; then
    echo "版本记录文件: $VERSION_DB（尚未创建）"
    if [[ -f "$OLD_VERSION_DB" ]]; then
      echo "检测到旧 TSV 记录文件: $OLD_VERSION_DB（新记录将写入 JSON）"
    fi
    return
  fi

  echo "版本记录文件: $VERSION_DB"
  "$PYTHON_BIN" - "$VERSION_DB" "$image_name" <<'PY'
import json
import sys
from pathlib import Path

db_path = Path(sys.argv[1])
image_name = sys.argv[2]

try:
    data = json.loads(db_path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    data = {}

items = data.get("images", {}).get(image_name, [])
print(f"当前镜像 {image_name} 的历史版本:")
if not items:
    print("  - 暂无")
else:
    for item in items:
        version = item.get("version", "")
        remote = item.get("remote_image", "")
        built_at = item.get("built_at", "")
        project_dir = item.get("project_dir", "")
        print(f"  - {version} -> {remote} ({built_at}, {project_dir})")
PY
}

record_version() {
  local image_name="$1"
  local version_tag="$2"
  local local_image="$3"
  local remote_image="$4"
  local project_dir="$5"
  local tar_path="$6"
  local build_time="$7"

  "$PYTHON_BIN" - "$VERSION_DB" "$image_name" "$version_tag" "$local_image" "$remote_image" "$project_dir" "$tar_path" "$build_time" <<'PY'
import json
import sys
from pathlib import Path

db_path = Path(sys.argv[1])
image_name = sys.argv[2]
entry = {
    "version": sys.argv[3],
    "local_image": sys.argv[4],
    "remote_image": sys.argv[5],
    "project_dir": sys.argv[6],
    "tar_path": sys.argv[7],
    "built_at": sys.argv[8],
}

if db_path.exists():
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
else:
    data = {}

data.setdefault("images", {}).setdefault(image_name, []).append(entry)
db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

echo "========== Docker Dev 镜像构建与推送 =========="
echo "请一次依次输入以下参数；输入完成后脚本才会开始执行 Docker 命令。"
echo ""

read_project_dir PROJECT_DIR_INPUT "1) 需要进入哪个文件夹进行打包生成（可填相对仓库路径或绝对路径）: "
read_required DOCKER_USERNAME "2) Docker 登录用户名: "
read_required DOCKER_REGISTRY "3) Docker 登录地址，例如 cr.lejugym.com: "
read_secret DOCKER_PASSWORD "4) Docker 登录密码/token: "
read_required REMOTE_NAMESPACE "5) 远端仓库前缀，例如 cr.lejugym.com/a03: "

PROJECT_DIR="$(resolve_project_dir "$PROJECT_DIR_INPUT")"
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "❌ 项目目录不存在: $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/Dockerfile" ]]; then
  echo "❌ 项目目录下没有 Dockerfile: $PROJECT_DIR/Dockerfile" >&2
  exit 1
fi

IMAGE_NAME="$(basename "$PROJECT_DIR")"
VERSION_NO="$(next_version_number "$IMAGE_NAME")"
VERSION_TAG="dev-v${VERSION_NO}"
LOCAL_IMAGE="${IMAGE_NAME}:${VERSION_TAG}"
REMOTE_NAMESPACE="${REMOTE_NAMESPACE%/}"
REMOTE_IMAGE="${REMOTE_NAMESPACE}/${IMAGE_NAME}:${VERSION_TAG}"
TAR_PATH="$PROJECT_DIR/${IMAGE_NAME}.tar"
BUILD_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

echo ""
show_history "$IMAGE_NAME"
echo ""
echo "即将执行:"
echo "  cd $PROJECT_DIR"
echo "  docker build -t $LOCAL_IMAGE ."
echo "  docker save -o $TAR_PATH $LOCAL_IMAGE"
echo "  docker login -u $DOCKER_USERNAME $DOCKER_REGISTRY --password-stdin"
echo "  docker tag $LOCAL_IMAGE $REMOTE_IMAGE"
echo "  docker push $REMOTE_IMAGE"
echo ""

read -r -p "确认执行？[y/N]: " CONFIRM
case "$CONFIRM" in
  y|Y|yes|YES) ;;
  *)
    echo "已取消。"
    exit 0
    ;;
esac

cd "$PROJECT_DIR"

docker build -t "$LOCAL_IMAGE" .
docker save -o "$TAR_PATH" "$LOCAL_IMAGE"
printf '%s' "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" "$DOCKER_REGISTRY" --password-stdin
docker tag "$LOCAL_IMAGE" "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE"

record_version "$IMAGE_NAME" "$VERSION_TAG" "$LOCAL_IMAGE" "$REMOTE_IMAGE" "$PROJECT_DIR" "$TAR_PATH" "$BUILD_TIME"

echo ""
echo "✅ 完成"
echo "本地镜像: $LOCAL_IMAGE"
echo "远端镜像: $REMOTE_IMAGE"
echo "镜像 tar: $TAR_PATH"
echo "版本记录: $VERSION_DB"
show_history "$IMAGE_NAME"
