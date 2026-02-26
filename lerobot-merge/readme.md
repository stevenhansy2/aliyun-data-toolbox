# LeRobot Merge 数据合并说明

本文档面向维护者，覆盖当前代码结构、端到端流程、核心脚本职责、参数说明和排障建议。

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

1. 输入定位：`collect_search_roots`
- 基于 `TARGET_SCRIPT_NAME` 在每个 data_id 下定位脚本目录。

2. 目录识别：`find_merge_sources`
- 识别“一级子目录中包含有效 LeRobot 数据集”的源目录。

3. 过滤策略：`filter_deepest_sources`
- 仅保留最深层可处理目录，避免父子目录重复处理。

4. 全局收集与合并
- 收集所有有效数据集子目录，建立 staging 目录。
- 对 staging 执行一次 `merge_data.py` 全局合并。

5. 输出
- 固定输出到 `OUTPUT_DIR/lerobot_merged`。
- 输入目录只读，不执行任何替换/重命名/标记写入。

## 3. 当前目录结构与职责

```text
lerobot-merge/
  run.sh                      # 合并主入口（平台流程）
  Dockerfile                  # 轻量运行镜像
  readme.md                   # 本文档
  kuavo/
    merge_data.py             # LeRobot 数据集合并与分块重排
    util.py                   # meta/info/tasks/episodes/stats 读写工具
```

## 4. 本地运行

在 `lerobot-merge` 目录下直接执行：

```bash
python3 kuavo/merge_data.py \
  --src_dir /path/to/src_dir \
  --tgt_dir /path/to/lerobot_merged \
  --summary_dir /path/to/report \
  --save
```

参数说明：
- `--src_dir`：待合并的源父目录（其一级子目录应为可合并数据集目录）。
- `--tgt_dir`：合并结果输出目录。
- `--summary_dir`：汇总信息输出目录（会生成 `dataset_summary.json`）。
- `--save`：保留 `tgt_dir`（建议开启）。

最小示例：

```bash
cd lerobot-merge
python3 kuavo/merge_data.py \
  --src_dir /data/input \
  --tgt_dir /data/output/lerobot_merged \
  --summary_dir /data/output/report \
  --save
```

## 5. 参数与环境变量（run.sh）

环境变量（平台模式）：
- `INPUT_DIR`：输入根目录，默认 `/inputs`。
- `OUTPUT_DIR`：输出根目录，默认 `/outputs`。
- `TARGET_SCRIPT_NAME`：必填，平台脚本目录名（单值）。
- `TARGET_SCRIPT_NAMES`：可选，多值（逗号或空格分隔）；若同时设置，`TARGET_SCRIPT_NAME` 优先。
- `MERGE_SCRIPT`：可选，默认自动探测 `kuavo/merge_data.py`。
- `STAGING_MODE`：`copy|symlink`，默认 `copy`。只读输入必须用 `copy`。
- `DEBUG_SLEEP_SECONDS`：调试暂停秒数，默认 `0`。

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

3. 容器内权限问题
- 输入目录建议只读挂载。
- 输出目录必须可写。
- 若输入只读，`STAGING_MODE` 需保持默认 `copy`，不要使用 `symlink`。

## 8. 当前约束

- 当前仓库是 merge-only 版本，不包含质检流程。
- 合并逻辑默认会对 episode 进行重编号并重组 chunk。
- 若上游数据结构不标准，可能被 `merge_data.py` 跳过并写入对应汇总结果。
