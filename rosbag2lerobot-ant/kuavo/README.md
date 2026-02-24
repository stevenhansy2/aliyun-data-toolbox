# Kuavo ROSBag -> LeRobotV21 转换说明

本文档面向维护者，覆盖当前代码结构、端到端流程、核心脚本职责、运行方式、扩展新话题的方法和排障建议。

## 1. 目标与入口

目标：将 Kuavo ROS bag 数据转换为 LeRobot 数据集（含 parquet、meta、视频）。

主入口：
- Python 入口：`kuavo/master_generate_lerobot_s.py`
- Shell 入口：`kuavo/run.sh`

核心编排函数：
- `converter/port_pipeline.py:port_kuavo_rosbag`

## 2. 端到端流程

```text
master_generate_lerobot_s.py
  -> 读取配置 + 扫描 bag 任务
  -> 对每个 bag 调用 port_kuavo_rosbag

port_pipeline.py
  -> 探测手型/配置上下文
  -> stream_population.populate_dataset_stream
      -> KuavoRosbagReader 流式(或并行)读 bag
      -> 对齐多模态 + 组装 low-dim
      -> 按 batch 写 parquet / 临时图像
  -> merge_batches 合并 batch
  -> video_pipeline 完成视频编码
  -> 输出最终 LeRobot 目录
```

## 3. 当前目录结构与职责

### 3.1 顶层脚本
- `kuavo/master_generate_lerobot_s.py`
  - CLI 参数解析。
  - 配置加载和覆盖（`--train_frequency` / `--which_arm` 等）。
  - 自动发现 bag 任务（单 bag 或目录批处理）。

- `kuavo/run.sh`
  - 容器/平台统一入口。
  - 可选下载 metadata、批处理多个输入目录、可选上传 OSS。

### 3.2 编排与写出
- `converter/port_pipeline.py`
  - 转换总流程控制。
  - 选择串行/并行读 bag，选择视频策略（同步/异步/流式/分段）。
  - 批次合并与收尾。

- `converter/pipeline/stream_population.py`
  - 单 bag 主循环（batch 处理）。
  - 从 `aligned_batch` 提取视觉和 low-dim，调用 frame 写入。

- `converter/pipeline/frame_builder.py`
  - 逐帧写入 dataset 结构。

- `converter/pipeline/stream_finalize.py`
  - 持久化媒体、写 batch metadata、保存参数。

### 3.3 Reader（读包 + 对齐）
- `converter/reader/kuavo_dataset_slave_s.py`
  - 兼容入口（re-export）。

- `converter/reader/kuavo_dataset_reader_impl.py`
  - Reader 主实现（topic 映射、预处理、对齐入口）。

- `converter/reader/streaming_pipeline.py`
  - 流式读取与并行读取核心。

- `converter/reader/alignment_engine.py`
  - 主对齐逻辑（外部主时间线/内部主时间线、按需插值）。

- `converter/reader/timeline_prescan.py`
  - 主时间线预扫描、跨 batch 连续性控制。

- 相关辅助模块：
  - `converter/reader/message_processor.py`：ROS msg -> 统一 `{"data": ...}`。
  - `converter/reader/topic_map_builder.py`：非相机话题映射。
  - `converter/reader/camera_topic_builder.py`：相机话题映射。
  - `converter/reader/timestamp_*.py` / `alignment_*.py`：时间戳质量与修复。

### 3.4 数据与媒体工具
- `converter/data_utils.py`
  - 兼容导出层（老调用入口，实际实现在 `converter/data/*`）。

- `converter/data/metadata_merge.py`
  - metadata/moments 合并、动作帧区间计算。

- `converter/data/bag_discovery.py`
  - 扫描 bag 与 sidecar（metadata/moments）解析。

- `converter/data/episode_loader.py`
  - 单 episode 抽取与 low-dim 组织。

- `converter/slave_utils.py`
  - 兼容导出层（老调用入口，实际实现在 `converter/media/*`）。

- `converter/media/camera_params.py`
  - 相机参数和原始图像提取工具。

- `converter/media/camera_flip.py`
  - 静止帧分析与左右互换/翻转策略。

- `converter/media/depth_video_export.py`
  - 深度视频导出函数。

- `converter/video_pipeline.py`
  - 视频编码主入口与编码器类。

- `converter/media/video_workers.py`
  - 单相机编码 worker。

- `converter/media/video_segments.py`
  - 分段编码与拼接。

- `converter/media/video_finalize.py`
  - 从临时帧目录完成最终视频编码。

### 3.5 配置与常量
- `converter/config.py`
  - 配置对象与默认值（默认 topic、相机 spec、切片规则）。

- `converter/configs/joint_names.py`
  - 关节名统一配置（如 `DEFAULT_LEG_JOINT_NAMES`）。

## 4. 常用运行方式

### 4.1 Python 直接运行

```bash
cd rosbag2lerobot-ant
python kuavo/master_generate_lerobot_s.py \
  --bag_dir /path/to/bags \
  --output_dir /path/to/output \
  --config ./kuavo/configs/request.json
```

可选参数：
- `--metadata_json_dir /path/to/metadata.json`
- `--moment_json_dir /path/to/moments.json`
- `--train_frequency 30`
- `--which_arm left|right|both`
- `--only_arm true|false`
- `--dex_dof_needed 1`
- `--use_depth`

### 4.2 Shell 封装运行

```bash
cd rosbag2lerobot-ant/kuavo
bash run.sh
```

常用环境变量：
- `INPUT_DIR` / `OUTPUT_DIR`
- `LOG_LEVEL=INFO|DEBUG`
- `USE_PARALLEL_ROSBAG_READ=true|false`
- `PARALLEL_ROSBAG_WORKERS=2`
- `USE_STREAMING_VIDEO=true|false`
- `USE_PIPELINE_ENCODING=true|false`

## 5. 关键配置字段（request.json）

高频字段说明：
- `robot_profile.model`：机器人型号标识。
- `robot_profile.urdf_path`：URDF 路径（影响末端位姿计算）。
- `robot_profile.topics`：非相机话题配置。
- `robot_profile.cameras`：相机 topic 规范（color/depth/camera_info）。
- `main_timeline.topic`：主时间线来源话题。
- `main_timeline.frequency`：主时间线频率（常见 30）。
- `train_frequency`：最终训练帧率。
- `only_arm` / `which_arm`：上肢/左右臂输出控制。
- `bag_overrides`：按 bag 名匹配覆盖配置。
- `model_profiles`：按型号覆盖配置。

## 6. 输出目录（典型）

```text
<output>/<dataset_name>/
  data/chunk-000/*.parquet
  meta/
    episodes.jsonl
    info.json
    tasks.jsonl
    episodes_stats.jsonl
    metadata.json
  videos/chunk-000/...
  depth/chunk-000/... (可选)
```

## 7. 新增话题接入指南（推荐流程）

目标：新增传感器/控制话题，并稳定写入最终数据集。

### 步骤 1：在配置中声明话题
1. 在 `kuavo/configs/request.json` 中加入：
   - 非相机：`robot_profile.topics`
   - 相机：`robot_profile.cameras`（含 color/depth/camera_info）
2. 若需全局默认，更新 `converter/config.py` 的：
   - `DEFAULT_SOURCE_TOPICS`
   - `DEFAULT_CAMERA_TOPIC_SPECS`

### 步骤 2：补充消息解析
在 `converter/reader/message_processor.py` 增加 `process_xxx()`：
- 返回格式保持统一：`{"data": ...}`。
- 时间戳不要在这里写，reader 会统一注入 `timestamp`。

### 步骤 3：挂接 topic 映射
- 相机类：`converter/reader/camera_topic_builder.py`
- 非相机类：`converter/reader/topic_map_builder.py`

建议命名规范：
- 观测：`observation.xxx`
- 动作：`action.xxx`
- 避免同义 key 重复。

### 步骤 4：接入批处理组织
在 `converter/pipeline/stream_population.py`：
1. 从 `aligned_batch` 读取新 key。
2. 需要入 low-dim 时加入 `all_low_dim_data`。
3. 需要帧级写入时接入 `write_batch_frames` 参数。

### 步骤 5：验证
1. 小 bag 回归，观察日志是否有 KeyError/shape 错误。
2. `LOG_LEVEL=DEBUG` 检查：
   - 话题命中
   - 对齐误差分布
   - batch 帧数一致性
3. 验证输出目录中的 parquet/meta/video 是否齐全。

## 8. 新增相机接入指南

1. 在 `robot_profile.cameras` 新增相机 spec。
2. 确认 `camera_topic_builder.py` 能生成该相机 key。
3. 在 `stream_population.py` 的 `cameras = raw_config.default_camera_names` 流程下自动纳入（若配置正确通常无需改代码）。
4. 检查相机内参/外参保存逻辑是否覆盖该相机。

## 9. 新机型适配指南

推荐优先用配置，不改代码：
1. 在 `request.json` 的 `model_profiles` 增加新机型（URDF、topics、camera specs）。
2. 如需按 bag 名自动命中，在 `bag_overrides` 增加 `match` 规则。
3. 必要时在 `master_generate_lerobot_s.py` 的配置覆盖逻辑中补字段映射。

## 10. 性能与稳定性建议

- 优先开启流式处理，避免一次性持有整包数据。
- 大数据量场景开启并行读包：
  - `USE_PARALLEL_ROSBAG_READ=true`
  - `PARALLEL_ROSBAG_WORKERS=2`（先从 2 开始）
- 视频策略建议：
  - 机器内存紧张：优先 `USE_STREAMING_VIDEO=true`
  - CPU 较强可尝试 `USE_PIPELINE_ENCODING=true`
- 对齐异常优先检查主时间线 topic 是否稳定、帧率是否合理。

## 11. 常见问题排查

- 症状：某些模态全空
  - 检查 `request.json` 话题名是否与 bag 内实际一致。
  - 检查 topic map 是否包含该 key。

- 症状：对齐误差过大
  - 检查主时间线来源与频率。
  - 查看 `alignment_engine` 日志中的误差分布。

- 症状：视频缺失/帧数不对
  - 检查 `separate_video_storage` 与视频策略开关。
  - 检查临时目录是否提前清理。

- 症状：末端位姿为空
  - 检查 `urdf_path` 是否正确。
  - 检查 `observation.sensorsData.joint_q` 是否存在且维度正确。

## 12. 维护约定

- 新能力优先加到子模块（`reader/`、`pipeline/`、`data/`、`media/`），避免重回超大单文件。
- 常量统一放 `converter/config.py` 或 `converter/configs/*`。
- 兼容层（如 `data_utils.py` / `slave_utils.py`）仅做导出，不放业务逻辑。

