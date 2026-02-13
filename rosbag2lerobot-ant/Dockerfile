FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 设置时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ENV TZ=Asia/Shanghai

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
    gnupg \
    lsb-release \
    golang \
    && rm -rf /var/lib/apt/lists/*

# 安装 cocli
# RUN curl -fL https://download.coscene.cn/cocli/install.sh | sh

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /app
COPY lerobot /app/lerobot

RUN conda create -y -n kuavo_il python=3.10.16
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate kuavo_il && \
    pip install -e lerobot -i https://pypi.tuna.tsinghua.edu.cn/simple"

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate kuavo_il && \
    pip install --no-cache-dir --upgrade pip && \
    pip install \
        joblib \
        h5py \
        bagpy \
        scikit-learn \
        oss2 \
        opencv-python-headless \          
        drake==1.19.0 \
        'pyarrow>=10.0.0' \
        'pyyaml>=6.0' \
        -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install 'numpy==1.26.4' --force-reinstall --no-deps \
        -i https://pypi.tuna.tsinghua.edu.cn/simple"
RUN curl -fL https://download.coscene.cn/cocli/install.sh | sh -s -- v1.3.0-rc3

RUN apt-get update && \
    apt-get install -y unzip && \
    apt-get install -y bc && \
    rm -rf /var/lib/apt/lists/* && \
    curl -o ossutil-2.1.2-linux-amd64.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip && \
    unzip ossutil-2.1.2-linux-amd64.zip && \
    cd ossutil-2.1.2-linux-amd64 && \
    chmod 755 ossutil && \
    mv ossutil /usr/local/bin/ && \
    ln -s /usr/local/bin/ossutil /usr/bin/ossutil && \
    cd .. && \
    rm -rf ossutil-2.1.2-linux-amd64 ossutil-2.1.2-linux-amd64.zip

# 复制应用文件
COPY kuavo /app/kuavo

# 设置脚本可执行权限
RUN chmod +x /app/kuavo/*.sh

# 设置环境变量
ENV PATH=/opt/conda/envs/kuavo_il/bin:$PATH

CMD ["/bin/bash"]
