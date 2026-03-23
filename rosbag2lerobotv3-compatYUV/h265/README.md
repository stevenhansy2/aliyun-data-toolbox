# rosbag2lerobotv3-compatYUV/h265

这是一条独立长期维护的数转版本线，目标是：

- LeRobot `0.5.0`
- LeRobot dataset `v3.0`
- 兼容 ROS bag 视频话题中的 `h265` 和 YUV 格式
- 优先支持本地 `conda` 测试，不依赖先打 Docker

## 目录说明

- `kuavo_data/`: 当前数转脚本和 Kuavo 侧逻辑
- `lerobot/`: 仓库内置的 LeRobot 源码副本，当前作为 `0.5.0` 适配基线
- `Dockerfile`: 后续容器化时使用的依赖声明

## 本地运行前提

本地 `conda` 环境只负责 Python 依赖。
如果要真的读取 `.bag`，宿主机仍然需要已有 ROS Python 包可用，例如：

- `<ros_python_dist_packages>`

当前仓库代码会直接导入这些模块：

- `rosbag`
- `rospy`
- `sensor_msgs`
- `std_msgs`

如果宿主机没有这些 ROS Python 包，本地环境仍然可以做以下工作：

- 安装 LeRobot `0.5.0` 相关依赖
- 跑纯 Python 单测
- 验证 metadata / merge / 非 ROS 侧逻辑
- 验证 ffmpeg 对 `h265` 的可用性

## 创建本地 conda 环境

推荐使用命名环境，便于长期维护：

```bash
conda create -y -n rosbag2lerobotv3_compat python=3.12 pip
conda activate rosbag2lerobotv3_compat
```

创建完成后建议先确认解释器版本：

```bash
python --version
```

## 安装 Python 依赖

先进入本目录：

```bash
cd rosbag2lerobotv3-compatYUV/h265
```

安装步骤：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./lerobot --no-deps
python -m pip install \
  'torch==2.7.1' \
  'torchvision==0.22.1' \
  'datasets>=4.0.0,<5.0.0' \
  'diffusers>=0.27.2,<0.36.0' \
  'huggingface-hub>=1.0.0,<2.0.0' \
  'accelerate>=1.10.0,<2.0.0' \
  'numpy>=2.0.0,<2.3.0' \
  'opencv-python-headless>=4.9.0,<4.13.0' \
  'av>=15.0.0,<16.0.0' \
  'jsonlines>=4.0.0,<5.0.0' \
  'pynput>=1.7.8,<1.9.0' \
  'pyserial>=3.5,<4.0' \
  'wandb>=0.24.0,<0.25.0' \
  'draccus==0.10.0' \
  'gymnasium>=1.1.1,<2.0.0' \
  'rerun-sdk>=0.24.0,<0.27.0' \
  'deepdiff>=7.0.1,<9.0.0' \
  'imageio[ffmpeg]>=2.34.0,<3.0.0' \
  'termcolor>=2.4.0,<4.0.0' \
  hydra-core omegaconf rich joblib h5py scikit-learn oss2 requests tqdm psutil pyarrow pyyaml Pillow einops \
  pycryptodomex python-gnupg rospkg catkin-pkg \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## ROS Python 路径

如果宿主机 ROS 已安装，运行前建议把 ROS Python 路径加入环境：

```bash
export PYTHONPATH=<ros_python_dist_packages>:$PYTHONPATH
```

如果你直接跑 `kuavo_data/run.sh`，脚本会自动把仓库内 `lerobot/src` 加入 `PYTHONPATH`。

## 本地运行方法

### 1. 直接跑批处理入口

```bash
cd rosbag2lerobotv3-compatYUV/h265/kuavo_data
export PYTHONPATH=<ros_python_dist_packages>:$PYTHONPATH
export INPUT_DIR=<input_root>
export OUTPUT_DIR=<output_root>
export PYTHON=$(which python)
bash run.sh
```

默认输出会落到：

```text
$OUTPUT_DIR/export/lerobot
```

### 2. 直接跑 Python 主脚本

```bash
cd rosbag2lerobotv3-compatYUV/h265/kuavo_data
export PYTHONPATH=<ros_python_dist_packages>:../lerobot/src:$PYTHONPATH
python -u CvtRosbag2Lerobot.py \
  --config-path configs \
  --config-name KuavoRosbag2Lerobot \
  rosbag.rosbag_dir=<bag_dir> \
  rosbag.lerobot_dir=<output_dir> \
  rosbag.metadata_json=<metadata_json>
```

## 快速自检

### ffmpeg 是否支持 h265

宿主机当前可用 `ffmpeg`，并且已经编译了 `libx265`。可自检：

```bash
ffmpeg -version | head -n 3
ffmpeg -codecs | grep -i hevc
```

### Python 依赖自检

```bash
export PYTHONPATH=<ros_python_dist_packages>:$PYTHONPATH
python -c "import yaml, hydra, torch, cv2, av, lerobot; print('ok')"
```

## 当前状态

当前目录已经完成：

- 新版本线隔离
- 本地 conda 测试说明
- `lerobot/pyproject.toml` 向 `0.5.0` 对齐
- `Dockerfile` 依赖向 `0.5.0` 对齐
- `h265_stream` 的 Python 版参考码流处理链路
- 对真实 ROS bag 的单包端到端验证

当前目录还在持续完善：

- YUV 话题的高性能直写路径
- 全量多 bag 回归与进一步提速
- README 与 run.sh 的进一步收口
