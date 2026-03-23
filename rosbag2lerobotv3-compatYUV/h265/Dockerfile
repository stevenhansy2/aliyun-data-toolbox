FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-rosbag \
    python3-roslib \
    python3-std-msgs \
    python3-sensor-msgs \
    python3-yaml \
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
    unzip \
    bc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH=/opt/venv/bin:$PATH

RUN mkdir -p /app/lerobot
COPY lerobot/ /app/lerobot/

RUN pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -e /app/lerobot --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install \
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
    hydra-core \
    omegaconf \
    rich \
    joblib \
    h5py \
    scikit-learn \
    oss2 \
    opencv-python-headless \
    requests \
    tqdm \
    psutil \
    'pyarrow>=10.0.0' \
    'pyyaml>=6.0' \
    Pillow \
    einops \
    pycryptodomex \
    python-gnupg \
    rospkg \
    catkin-pkg \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN curl -o ossutil-2.1.2-linux-amd64.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip && \
    unzip ossutil-2.1.2-linux-amd64.zip && \
    cd ossutil-2.1.2-linux-amd64 && \
    chmod 755 ossutil && \
    mv ossutil /usr/local/bin/ && \
    ln -s /usr/local/bin/ossutil /usr/bin/ossutil && \
    cd .. && \
    rm -rf ossutil-2.1.2-linux-amd64 ossutil-2.1.2-linux-amd64.zip

COPY kuavo_data /app/kuavo_data
RUN chmod +x /app/kuavo_data/*.sh

CMD ["/bin/bash"]
