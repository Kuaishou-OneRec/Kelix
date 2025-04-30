#!/bin/bash



# 检查当前的 shell 是否为 bash
if [ -z "$BASH_VERSION" ]; then
    echo "此脚本必须使用 bash 启动，请使用 'bash script.bash' 来运行它。" >&2
    exit 1
fi



# Get the directory of the current script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${SCRIPT_DIR}/.deepspeed_env"

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
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install timm==1.0.15"
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install fastparquet==2024.2.0"
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "wget https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects/user-cloud-storage/xray%2Finstall_xray.sh -O install_xray.sh && bash install_xray.sh"

