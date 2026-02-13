FROM osrf/ros:noetic-desktop-full-focal

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 设置时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

# 安装额外依赖
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    jq \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libsm6 \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# 接受 Conda ToS（新版 Miniconda 需要）
RUN conda config --set auto_activate_base false && \
    conda init bash

# 创建 conda 环境并安装 Python 包
WORKDIR /app
# 复制代码
COPY lerobot /app/lerobot
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n kuavo_il python=3.10.16 

# 安装额外 pip 包（使用指定环境的 pip）

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate kuavo_il && \
    pip install -e ./lerobot -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install rospkg==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install gnupg==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install pycryptodomex==3.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install pympler -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install rich==14.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install hydra-core==1.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple "

# 安装 cocli
RUN curl -fL https://download.coscene.cn/cocli/install.sh | sh -s -- v1.3.0-rc3

# 复制应用文件
COPY kuavo_data /app/kuavo_data
RUN chmod +x /app/kuavo_data/*.sh
# 默认命令
# 激活 conda 环境并设置 PATH
ENV PATH=/opt/conda/envs/kuavo_il/bin:/opt/conda/bin:$PATH

# 确保 ROS 环境已 source（可选，但推荐）
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]