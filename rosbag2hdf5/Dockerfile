FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

RUN ln -sf /usr/share/zoneinfo/${TZ} /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-rosbag \
    python3-roslib \
    python3-std-msgs \
    python3-sensor-msgs \
    ca-certificates \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH=/opt/venv/bin:$PATH

COPY kuavo/configs/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Fail fast: extrinsics relies on pydrake; image build must guarantee import works.
RUN python3 -c "import pydrake; print('pydrake import ok')"

# Install ossutil binary
RUN curl -fsSL -o /tmp/ossutil.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip && \
    cd /tmp && unzip /tmp/ossutil.zip && \
    chmod 755 /tmp/ossutil-2.1.2-linux-amd64/ossutil && \
    mv /tmp/ossutil-2.1.2-linux-amd64/ossutil /usr/local/bin/ossutil && \
    ln -s /usr/local/bin/ossutil /usr/bin/ossutil && \
    rm -rf /tmp/ossutil-2.1.2-linux-amd64 /tmp/ossutil.zip

COPY kuavo /app/kuavo
RUN chmod +x /app/kuavo/run.sh

CMD ["/bin/bash"]
