FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    unzip \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

# Optional OSS utility for upload/download workflows.
RUN curl -fsSLo /tmp/ossutil.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip \
    && unzip /tmp/ossutil.zip -d /tmp \
    && mv /tmp/ossutil-2.1.2-linux-amd64/ossutil /usr/local/bin/ossutil \
    && chmod 755 /usr/local/bin/ossutil \
    && ln -sf /usr/local/bin/ossutil /usr/local/bin/ossutil64 \
    && rm -rf /tmp/ossutil.zip /tmp/ossutil-2.1.2-linux-amd64

# Minimal dependencies for merge_data.py.
RUN pip install --upgrade pip \
    && pip install \
      jsonlines==4.0.0 \
      numpy==1.26.4 \
      pandas==2.2.3 \
      pyarrow==20.0.0

COPY run.sh /app/run.sh
COPY kuavo/merge_data.py /app/kuavo/merge_data.py
COPY kuavo/util.py /app/kuavo/util.py

RUN chmod +x /app/run.sh

CMD ["/bin/bash"]
