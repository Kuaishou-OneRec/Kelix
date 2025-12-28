#!/bin/bash



# 检查当前的 shell 是否为 bash
if [ -z "$BASH_VERSION" ]; then
    echo "此脚本必须使用 bash 启动，请使用 'bash script.bash' 来运行它。" >&2
    exit 1
fi



# Get the directory of the current script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${SCRIPT_DIR}/.env"

# Check if .deepspeed_env exists
if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: ${ENV_FILE} not found"
    exit 1
fi

# Load environment variables from .env file
set -a  # automatically export all variables
source "${ENV_FILE}"
set +a  # disable auto-export

# Print loaded variables (optional)
echo "Loaded environment variables from ${ENV_FILE}:"
cat "${ENV_FILE}"

nohup rm -rf hs_err_pid*.log &
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "/opt/conda/envs/py312/bin/pip3 install keye_vl_utils"
# registry.corp.kuaishou.com/kml-supercomputing-project/v1.6_vllm0.7.3_ray_torch2.5.1_cuda11.8.0_python3.10.12_hadoop_xray:v1
#mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile -x http_proxy=http://oversea-squid1.jp.txyun:11080 -x https_proxy=http://oversea-squid1.jp.txyun:11080 --pernode bash -c  "apt-get install numactl"
#mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install easydict"
# mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install transformers==4.49"
# mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install --upgrade torchao"
# mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install timm==1.0.15"
# mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install fastparquet==2024.2.0"
# mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "wget https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects/user-cloud-storage/xray%2Finstall_xray.sh -O install_xray.sh && bash install_xray.sh"
