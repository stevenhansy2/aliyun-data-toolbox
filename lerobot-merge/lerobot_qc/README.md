# LeRobot 数据集质量检测工具

用于验证 LeRobot 格式数据集质量的工具，支持本地和 OSS 两种检测模式。

## 功能

- **Level 0 检查**：验证数据集是否符合 LeRobot 格式（一票否决）
- **Level A 检查**：数据集级别元数据验证
- **Level B 检查**：Episode 级别数据质量检测
  - B1: 异常静止检测
  - B2: 角度域与 Gripper 检查
  - B3: 原始信号异常检测
  - B4: 时间戳一致性检查（单调递增、间隔稳定、与 FPS 一致）
- 自动生成 JSON/Markdown 格式检测报告
- 支持上传检测结果到 OSS

## 安装

```bash
# 使用 aistudio conda 环境
conda create -n aistudio python=3.9
conda activate aistudio

# 安装依赖
pip install --index-url='https://pypi.antfin-inc.com/simple' -r requirements_aistudio.txt
pip install -r requirements.txt
```

## 环境配置

OSS 访问需要配置 `.env` 文件或设置环境变量：
```bash
export OSS_PREFIX="your_oss_prefix/"
export OSS_PREFIX_FOLDER="ori_raw_data/quality_check/"
```

## 使用方法

### 本地检测

```bash
# 基本用法
python validator_local.py --dataset /path/to/dataset

# 指定配置文件
python validator_local.py --dataset /path/to/dataset --config config/detection_config.yaml

# 完整验证（访问所有帧）
python validator_local.py --dataset /path/to/dataset --full-validation

# 从 OSS 下载数据后检测
python validator_local.py --dataset /root/data/task --from-oss
```

### OSS 流式检测

```bash
# 直接检测 OSS 上的数据（无需下载完整数据集）
python validator_oss.py --oss-config config/oss_config.yaml

# 自定义检测配置
python validator_oss.py --oss-config config/oss_config.yaml --config config/detection_config.yaml
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集路径 | `/root/data/task` |
| `--config` | 检测配置文件 | `config/detection_config.yaml` |
| `--oss-config` | OSS 配置文件 | `config/oss_config.yaml` |
| `--output` | 报告输出目录 | `reports` |
| `--tolerance` | 夹爪检测边界扩充范围 | `0.3` |
| `--full-validation` | 启用完整验证 | `False` |
| `--log-level` | 日志级别 | `INFO` |

## 配置文件

在 `config/` 目录下提供多种机器人配置：

- `detection_config.yaml` - 默认检测参数配置
- `oss_config.yaml` - OSS 连接配置
- 各机器人专用配置（如 `robby_franka.yaml`、`kupasi_tienkung.yaml` 等）

## 输出

检测完成后在 `reports/` 目录生成：
- `report.json` - 详细检测结果
- `report.md` - 可读性报告
- `quality_check.jsonl` - 废弃 episode 列表（上传到 OSS）
