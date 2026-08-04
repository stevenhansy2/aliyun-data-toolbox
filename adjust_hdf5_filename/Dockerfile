FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai \
    INPUT_DIR=/inputs \
    OUTPUT_DIR=/outputs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    unzip \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/ossutil.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip \
    && unzip /tmp/ossutil.zip -d /tmp \
    && mv /tmp/ossutil-2.1.2-linux-amd64/ossutil /usr/local/bin/ossutil \
    && chmod 755 /usr/local/bin/ossutil \
    && ln -sf /usr/local/bin/ossutil /usr/local/bin/ossutil64 \
    && rm -rf /tmp/ossutil.zip /tmp/ossutil-2.1.2-linux-amd64

RUN pip install --upgrade pip

COPY run.sh /app/run.sh
COPY kuavo/merge_local_metadata.py /app/merge_local_metadata.py
COPY kuavo/adjust_file_size.py /app/adjust_file_size.py
COPY kuavo/adjust_file_name_local.py /app/adjust_file_name_local.py

RUN chmod +x /app/run.sh

CMD ["/bin/bash", "/app/run.sh"]
