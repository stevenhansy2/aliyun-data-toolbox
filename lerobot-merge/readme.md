
# lerobot-merge 使用说明

## 1. 拉取 OSS 上的 lerobot 数据集

（请根据实际情况补充数据集拉取命令）

---

## 2. 安装 Conda 并创建环境

```bash
# 下载 Miniconda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 创建并激活新环境
conda create -y -n kuavo_il python=3.10.16
conda activate kuavo_il
```

---

## 3. 拉取 lerobot-merge 代码

```bash
git clone https://github.com/Huangri-believe/test.git
cd lerobot-merge
```

---

## 4. 安装依赖

```bash
pip install -e lerobot -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install oss2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install 'numpy==1.26.4' --force-reinstall --no-deps \
        -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 5. 修改 `run.sh` 操作路径

编辑 `run.sh` 文件，将如下内容：

```bash
ROOT_DIR="/home/leju_kuavo/1234/"
```

修改为你实际的数据存储硬盘路径。

---

## 6. 运行脚本

```bash
sudo chmod +x run.sh
./run.sh
```