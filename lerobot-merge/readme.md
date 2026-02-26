# LeRobot Merge 数据合并说明

本文档面向维护者，覆盖当前代码结构、端到端流程、核心脚本职责、运行方式、参数说明和排障建议。

## 1. 目标与入口

目标：
- 在平台输入目录下，按 `TARGET_SCRIPT_NAME` 精确定位待处理目录。
- 收集所有符合 LeRobot 结构的数据集子目录，执行一次全局合并。
- 将结果写入可写输出目录（不修改输入目录）。

主入口：
- Shell 入口：`run.sh`
- Python 合并脚本：`kuavo/merge_data.py`
- 元数据工具：`kuavo/util.py`

## 2. 端到端流程

```text
run.sh
  -> 读取环境变量配置（INPUT_DIR / OUTPUT_DIR / TARGET_SCRIPT_NAME）
  -> collect_search_roots：定位 /inputs/<data_id>/<TARGET_SCRIPT_NAME>/
  -> discover：递归识别可合并源目录
  -> gather：收集所有有效 LeRobot 子目录
  -> merge：通过临时 staging 全局合并到 /outputs/lerobot_merged
```

### 2.1 流程分层（run.sh）

1. 参数解析：`parse_args`
- 支持 CLI 参数和环境变量双通道配置。

2. 输入定位：`collect_search_roots`
- 强制基于 `TARGET_SCRIPT_NAME` 在每个 data_id 下定位脚本目录。

3. 数据发现：`discover_lerobot_dirs`
- 使用 `find` 按深度扫描目录名（默认 `lerobot`）。

4. 全局收集与合并
- 收集所有有效数据集子目录，建立 staging 软链接目录。
- 对 staging 目录执行一次 `merge_data.py` 全局合并。

5. 输出
- 固定输出到 `OUTPUT_DIR/lerobot_merged`。
- 输入目录只读，不执行任何替换/重命名/标记写入。

## 3. 当前目录结构与职责

```text
lerobot-merge/
  run.sh                      # 合并主入口（只读输入 + 可写输出）
  Dockerfile                  # 轻量运行镜像
  readme.md                   # 本文档
  kuavo/
    merge_data.py             # LeRobot 数据集合并与分块重排
    util.py                   # meta/info/tasks/episodes/stats 读写工具
```

## 4. 常用运行方式

### 4.1 本地直接运行

```bash
cd lerobot-merge
chmod +x run.sh

# 最小用法（平台变量方式）
export INPUT_DIR=/inputs
export OUTPUT_DIR=/outputs
export TARGET_SCRIPT_NAME=org_rosbag2lerobotv21_d
./run.sh
```

### 4.2 带完整参数运行

```bash
export INPUT_DIR=/path/to/input
export OUTPUT_DIR=/path/to/output
export TARGET_SCRIPT_NAMES="org_rosbag2lerobotv21_d,org_rosbag2lerobotv22_d"
export MERGE_SCRIPT=kuavo/merge_data.py
./run.sh
```

### 4.3 环境变量方式

```bash
export INPUT_DIR=/path/to/input
export OUTPUT_DIR=/path/to/output
export TARGET_SCRIPT_NAME=org_rosbag2lerobotv21_d
./run.sh
```

### 4.4 Docker 运行（推荐）

构建镜像：

```bash
docker build -t lerobot-merge:lite lerobot-merge
```

容器内执行（输入只读挂载）：

```bash
docker run --rm \
  -v /path/to/input:/work/input:ro \
  -v /path/to/output:/work/output \
  lerobot-merge:lite \
  -e TARGET_SCRIPT_NAME=org_rosbag2lerobotv21_d \
  bash -lc 'cd /app && ./run.sh'
```

## 5. 参数与环境变量

环境变量（平台模式）：
- `INPUT_DIR`：输入根目录，默认 `/inputs`。
- `OUTPUT_DIR`：输出根目录，默认 `/outputs`。
- `TARGET_SCRIPT_NAME`：必填，平台脚本目录名（单值）。
- `TARGET_SCRIPT_NAMES`：可选，多值（逗号或空格分隔）；若同时设置，`TARGET_SCRIPT_NAME` 优先。
- `MERGE_SCRIPT`
- `STAGING_MODE`：`copy|symlink`，默认 `copy`。只读输入必须用 `copy`。
- `DEBUG_SLEEP_SECONDS`

## 6. 输出目录（典型）

输入目录（示例）：

```text
/inputs/<data_id>/<TARGET_SCRIPT_NAME>/...
  <source_dir_1>/<dataset_1>
  <source_dir_1>/<dataset_2>
  <source_dir_2>/<dataset_3>
```

输出目录（示例）：

```text
<OUTPUT_DIR>/lerobot_merged/
  data/chunk-000/*.parquet
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    metadata.json (可选)
  videos/chunk-000/...
  parameters/...

<OUTPUT_DIR>/lerobot_merged/meta/...
```

## 7. 故障排查

1. 未找到指定脚本目录
- 检查 `TARGET_SCRIPT_NAME` 是否与平台目录名完全一致。
- 确认输入结构为 `/inputs/<data_id>/<TARGET_SCRIPT_NAME>/`。

2. 合并失败
- 使用 `DEBUG_SLEEP_SECONDS` 暂停容器后手工检查目录结构。
- 单独执行：
  `python3 kuavo/merge_data.py --src_dir <src> --tgt_dir <tgt> --summary_dir <report> --save`
- 检查子目录是否满足 LeRobot 结构约束（`meta/`, `data/`, `videos/`, `parameters/`）。

4. 容器内权限问题
- 输入目录建议只读挂载。
- 输出目录必须可写。
- 容器产物可能是 root 用户，需要按部署规范处理目录权限。
- 若输入只读，`STAGING_MODE` 需保持默认 `copy`，不要使用 `symlink`。

## 8. 当前约束

- 当前仓库是 merge-only 版本，不包含质检流程。
- 合并逻辑默认会对 episode 进行重编号并重组 chunk。
- 若上游数据结构不标准，可能被 `merge_data.py` 跳过并写入对应 `report/dataset_summary.json`。
