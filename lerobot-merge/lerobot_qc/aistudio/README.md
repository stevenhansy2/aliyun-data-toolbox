# AIStudio 数据质量检测批量提交工具

本目录包含用于在 AIStudio 集群上批量执行 LeRobot 数据集质量检测的脚本和配置文件。

## 📁 目录结构

```
aistudio/
├── data_checker.py          # 主脚本：批量提交质量检测任务
├── files/                   # 任务列表文件目录
│   ├── example_tasks.txt    # 示例任务列表
│   └── *.txt                # 其他任务列表文件
├── local/                   # 本地配置目录
│   └── user_config.yml      # 用户配置（OSS 认证信息等）
└── remote/                  # 远程执行脚本目录
    └── data_checker.sh      # 在集群上执行的检测脚本
```

## 🚀 快速开始

### 1. 配置 OSS 认证信息

复制配置模板并填入您的 OSS 配置：

```bash
cd aistudio/local
cp user_config.yml.example user_config.yml
```

编辑 `local/user_config.yml`，填入您的真实 OSS 配置：

```yaml
oss_domain: your-oss-domain.aliyuncs.com
oss_key_id: YOUR_OSS_KEY_ID
oss_key_secret: YOUR_OSS_KEY_SECRET
```

**注意**：`user_config.yml` 已在 `.gitignore` 中，不会被提交到代码仓库。

### 2. 准备任务列表

在 `files/` 目录下创建任务列表文件（或使用现有文件），格式如下：

```
# 格式：config_file oss_path
custom_bjrenxing_agilex_splitAloha buy_customized/bj_renxing/20251119/data02/anti_lerobot_1030_1/agilex_splitAloha_dualArm-gripper-3cameras_1/task_name
robby_franka buy/task_001/episode_001
```

**格式说明：**
- `config_file`: 配置文件名（不含 `.yaml` 后缀），对应项目根目录 `config/` 下的配置文件
- `oss_path`: OSS 数据路径，相对于 `OSS_DATA_PREFIX` 的路径
- 以 `#` 开头的行为注释
- 空行会被忽略

参考 `files/example_tasks.txt` 查看完整示例。

### 3. 修改脚本配置

编辑 `data_checker.py`，根据需要修改以下配置：

```python
# 资源配置
RESOURCE_CONFIG = {
    'cpu': 16,
    'memory_gb': 32,
    'gpu_num': 0,
    'disk_gb': 256,
}

# OSS 路径配置
OSS_DATA_PREFIX = "ori_raw_data/"
OSS_OUTPUT_PREFIX = "ori_raw_data/quality_check/20251126_rest/"

# 任务配置文件路径
TASK_LIST_FILE = os.path.join('files', 'your_task_list.txt')
```

### 4. 运行脚本

从项目根目录或任意位置运行：

```bash
python aistudio/data_checker.py
```

脚本会自动：
1. 读取任务列表
2. 为每个任务创建 AIStudio 作业
3. 提交到集群执行

## 📋 配置说明

### 资源配置 (RESOURCE_CONFIG)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cpu` | CPU 核心数 | 16 |
| `memory_gb` | 内存大小（GB） | 32 |
| `gpu_num` | GPU 数量 | 0 |
| `disk_gb` | 磁盘大小（GB） | 256 |

### 代码仓库配置

```python
CODE_REPO_URL = "https://code.alipay.com/renyiyu.ryy/lerobot_qc.git"
CODE_REPO_BRANCH = "master"
```

### Docker 镜像

```python
DOCKER_IMAGE = "reg.docker.alibaba-inc.com/aii/aistudio:13800121-20251127130629"
```

### OSS 路径配置

- `OSS_DATA_PREFIX`: OSS 数据根目录前缀
- `OSS_OUTPUT_PREFIX`: 质量检测报告输出目录

## 🔧 工作原理

1. **读取配置**：加载用户配置和任务列表
2. **创建作业**：为每个任务创建 PythonJobBuilder 作业
3. **提交执行**：
   - 将代码仓库克隆到集群
   - 设置环境变量（OSS 路径、配置文件路径等）
   - 执行 `remote/data_checker.sh` 脚本
4. **质量检测**：在集群上运行数据质量检测
5. **结果输出**：检测报告保存到 OSS 的 `OSS_OUTPUT_PREFIX` 目录


