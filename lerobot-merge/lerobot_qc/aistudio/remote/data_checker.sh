#!/bin/bash

export ACCESS_KEY_ID=
export ACCESS_KEY_SECRET=
ossutil64 config -e cn-shanghai-ant-internal.oss-alipay.aliyuncs.com -i -k 
cd /workspace/bin/lerobot_qc
python3 validator_local.py --config "$CONFIG_PATH" --tolerance 0.1 --from-oss