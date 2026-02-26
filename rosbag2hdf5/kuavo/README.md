# Kuavo ROSBag -> HDF5 转换说明

本文档面向维护者，覆盖当前代码结构、转换流程、运行方式和扩展建议。

## 1. 目标与入口

目标：将 Kuavo ROS bag 数据转换为 HDF5 数据集，并产出视频、深度视频、相机参数和 metadata。

主入口：
- Python 入口：`kuavo/convert_rosbag_to_hdf5.py`
- Shell 入口：`kuavo/run.sh`

## 2. 当前目录结构

```text
kuavo/
  assets/
    urdf/
      biped_s45.urdf
      biped_s49.urdf
  configs/
    request.json
    requirements.txt
  converter/
    data/                 # 数据发现/加载/metadata 处理
    reader/               # Reader 拆分模块（mixin + postprocess）
    utils/                # 视频/HDF5/数据质量工具拆分模块
  docs/
    rosbag2HDF5_readme.md
    metadata和moment使用字段.md
  convert_rosbag_to_hdf5.py
  run.sh
```

说明：
- `converter/` 下模块已按职责拆分，单文件控制在 1000 行以内。

## 3. 转换流程

1. 读取 `configs/request.json`。
2. 扫描 `--bag_dir` 下 `.bag` 文件，自动过滤 `.c.bag`。
3. 默认读取 `--bag_dir/metadata.json`（可通过 `--metadata_json_dir` 覆盖）。
4. 对每个 episode：
   - 读取并对齐多模态 rosbag 数据。
   - 生成 `proprio_stats.hdf5`（以及 `proprio_stats_original.hdf5`）。
   - 导出彩色视频与深度视频。
   - 写入相机内外参与 metadata。
   - 执行数据一致性校验。

## 4. 运行方式

### 4.1 Python 直接运行

```bash
cd rosbag2hdf5/kuavo
python3 convert_rosbag_to_hdf5.py \
  --config configs/request.json \
  --bag_dir /path/to/bag_dir \
  --output_dir /path/to/output \
  --scene test_scene \
  --sub_scene test_sub_scene \
  --continuous_action test_continuous_action \
  --mode simplified
```

常用参数：
- `--metadata_json_dir`: 可选，不传时默认 `--bag_dir/metadata.json`
- `--scene --sub_scene --continuous_action`
- `--min_duration`

### 4.2 Shell 封装运行

```bash
cd rosbag2hdf5/kuavo
bash run.sh
```

容器/平台命令（已移除 conda）：

```bash
/bin/bash -lc "bash /app/kuavo/run.sh"
```

常用环境变量：
- `INPUT_DIR`, `OUTPUT_DIR`
- `MIN_DURATION`
- `OSS_BUCKET`, `FOLDER_ID`（可选上传）
- 性能调优（可选）：
  - `KUAVO_VALIDATE_OUTPUT=false` 跳过最终一致性校验（更快）
  - `KUAVO_ENHANCE_ENABLED=false` 关闭压缩深度增强（更快）
  - `KUAVO_ENABLE_DEPTH_DENOISE=false` 关闭手部深度去噪（更快）
  - `KUAVO_COLOR_PRESET=veryfast` 或 `ultrafast`（更快，体积更大）
  - `KUAVO_COLOR_CRF=22`（更快，画质略降）
  - `KUAVO_SCHED_CORES=1|2|4|8` 视频调度档位（彩色+深度，解码喂帧/编码并发）
    - `1`: decode_feed=1, encode=1, queue=96
    - `2`: decode_feed=2, encode=1, queue=160
    - `4`: decode_feed=3, encode=2, queue=260
    - `8`: decode_feed=4, encode=3, queue=400

## 5. 开发约定

- 新增逻辑优先放到 `converter/` 分层模块，避免再向入口脚本堆积。
- 与视频编码相关逻辑统一走 `converter/utils/`。
- 目录扫描逻辑统一走 `converter/data/bag_discovery.py`。

## 6. 脚本功能速查（逐文件）

### 6.1 顶层入口
- `kuavo/convert_rosbag_to_hdf5.py`
  - Python 主入口。
  - 调用 `converter/pipeline/conversion_orchestrator.py` 中的 `main()`。

- `kuavo/run.sh`
  - 容器/平台批处理入口。
  - 按子目录遍历数据，检查 `metadata.json` 与 `.bag`，执行转换并输出耗时。

### 6.2 Pipeline
- `converter/pipeline/conversion_orchestrator.py`
  - 转换总编排（参数解析、循环处理 bag、写 hdf5/视频/metadata、一致性校验）。

### 6.3 Data
- `converter/data/bag_discovery.py`
  - 自动发现 bag 文件，过滤 `.c.bag`。

- `converter/data/episode_loader.py`
  - 读取单 episode 的多模态数据，组织 low-dim 结构。

- `converter/data/metadata_ops.py`
  - `metadata.json` 字段校验与转换、动作帧区间计算、目录结构生成。

### 6.4 Reader
- `converter/reader/reader_entry.py`
  - Reader 兼容导出入口（对外统一导出 `KuavoRosbagReader`、`PostProcessorUtils` 等）。

- `converter/reader/rosbag_reader.py`
  - 聚合各 mixin 的 Reader 主类。

- `converter/reader/reader_setup.py`
  - 话题映射、bag 读取初始化、外参提取入口。

- `converter/reader/reader_timestamp.py`
  - 时间戳预处理、质量检查、插值/去重辅助。

- `converter/reader/reader_alignment_core.py`
  - 多模态主对齐流程（裁剪、对齐、修正）。

- `converter/reader/reader_alignment_fps.py`
  - 帧率调整（插帧/删帧）与时间轴微调。

- `converter/reader/reader_alignment_validation.py`
  - 最终对齐质量验证。

- `converter/reader/postprocess_utils.py`
  - 后处理工具（电流/力矩转换、hdf5 保存等）。

### 6.5 Utils / Image / Kinematics
- `converter/utils/facade.py`
  - 工具函数兼容导出层（对外统一入口）。

- `converter/utils/video_streaming.py`
  - 流式视频写入与编码。

- `converter/utils/video_parallel.py`
  - 并行视频处理与编码。

- `converter/utils/hdf5_utils.py`
  - HDF5 增量/批量写入工具。

- `converter/utils/data_quality.py`
  - 数据一致性校验、静止帧检测、翻转/左右交换逻辑。

- `converter/utils/camera_utils.py`
  - 相机命名映射、内外参导出、图像结构化处理。

- `converter/image/depth_conversion.py`
  - 深度图处理与转换。

- `converter/image/video_denoising.py`
  - 深度图去噪与修复。

- `converter/kinematics/kuavo_pose_calculator.py`
  - 机器人位姿/相机位姿计算。

- `converter/kinematics/endeffector_pose_from_bag.py`
  - 末端位姿提取。

### 6.6 配置与资源
- `configs/request.json`
  - 运行时配置（话题、帧率、模型、导出策略等）。

- `configs/requirements.txt`
  - Python 依赖清单。

- `assets/urdf/*.urdf`
  - 机器人 URDF 模型文件。
