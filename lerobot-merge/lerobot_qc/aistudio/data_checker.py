#!/usr/bin/env python3
"""
AIStudio 数据质量检测任务提交脚本

用于批量提交 LeRobot 数据集质量检测任务到 AIStudio 平台
"""

import os
import yaml
from easydict import EasyDict
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf, CodeRepoConf
from pypai.conf.retry_strategy import RetryStrategy


os.environ["ENV_ENCRYPTED_SECRET"] = "w1ODdlZjFlZTRlYzU4YThi"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESOURCE_CONFIG = {
    'cpu': 16,
    'memory_gb': 32,
    'gpu_num': 0,
    'disk_gb': 256,
}

CODE_REPO_URL = "https://code.alipay.com/renyiyu.ryy/lerobot_qc.git"
CODE_REPO_BRANCH = "master"

DOCKER_IMAGE = "reg.docker.alibaba-inc.com/aii/aistudio:13800121-20251127130629"

OSS_DATA_PREFIX = "ori_raw_data/"
OSS_OUTPUT_PREFIX = "ori_raw_data/quality_check/20251126_rest/"

TASK_LIST_FILE = os.path.join(SCRIPT_DIR, 'files', '20251126_rest_robot.txt')
USER_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'local', 'user_config.yml')


def load_user_config():
    with open(USER_CONFIG_FILE, 'r') as f:
        return EasyDict(yaml.load(f.read(), Loader=yaml.FullLoader))


def load_task_list():
    config_files = {}
    with open(TASK_LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 2:
                config_file, oss_path = parts
                config_files.setdefault(config_file, []).append(oss_path)
            else:
                print(f"⚠️  行格式异常，已跳过: {line}")

    return config_files


def create_exec_conf():
    return ExecConf(
        num=1,
        cpu=RESOURCE_CONFIG['cpu'],
        memory=1024 * RESOURCE_CONFIG['memory_gb'],
        gpu_num=RESOURCE_CONFIG['gpu_num'],
        disk_m=1024 * RESOURCE_CONFIG['disk_gb']
    )


def create_km_conf():
    return KMConf(
        image=DOCKER_IMAGE,
        cluster="auto",
        retry_strategy=RetryStrategy(retry_policy="never", max_attempt=3)
    )


def create_code_repo_conf():
    return CodeRepoConf(repo_url=CODE_REPO_URL, branch=CODE_REPO_BRANCH)


def submit_job(config_file, oss_path, user_config, master, km_conf, code_repo):
    config_path = f'./config/{config_file}.yaml'
    oss_prefix = f'{OSS_DATA_PREFIX}{oss_path}'
    if not oss_prefix.endswith('/'):
        oss_prefix += '/'

    command = (
        f'cd /workspace/bin && '
        f'export OSS_PREFIX_FOLDER={OSS_OUTPUT_PREFIX} && '
        f'export OSS_PREFIX="{oss_prefix}" && '
        f'export CONFIG_PATH="{config_path}" && '
        f'chmod 777 data_checker.sh && '
        f'bash data_checker.sh'
    )

    launch_args = {
        'oss_domain': user_config.oss_domain,
        'oss_key_id': user_config.oss_key_id,
        'oss_key_secret': user_config.oss_key_secret
    }

    print(f"📋 提交任务: config={config_file}, oss={oss_prefix}")

    job = PythonJobBuilder(
        source_root=os.path.join('aistudio', 'remote', ''),
        command=command,
        main_file='',
        master=master,
        km_conf=km_conf,
        tag=f"mytype=data-checker,type=OTHER,basemodel=AntGLM-5B",
        runtime='pytorch',
        k8s_priority="high",
        data_stores=[],
        k8s_app_name="vilab",
        code_repo_configs=[code_repo],
        envs={},
        global_params=launch_args,
    )

    job.run(enable_wait=False)


def main():
    print("=" * 60)
    print("AIStudio 数据质量检测任务提交")
    print("=" * 60)

    user_config = load_user_config()
    task_list = load_task_list()

    master = create_exec_conf()
    km_conf = create_km_conf()
    code_repo = create_code_repo_conf()

    total_tasks = sum(len(paths) for paths in task_list.values())
    print(f"\n📊 共 {len(task_list)} 个配置文件，{total_tasks} 个任务\n")

    task_count = 0
    for config_file, oss_path_list in task_list.items():
        for oss_path in oss_path_list:
            task_count += 1
            print(f"[{task_count}/{total_tasks}] ", end="")
            submit_job(config_file, oss_path, user_config, master, km_conf, code_repo)

    print("\n" + "=" * 60)
    print(f"✅ 所有任务已提交完成！共 {total_tasks} 个任务")
    print("=" * 60)


if __name__ == '__main__':
    main()
