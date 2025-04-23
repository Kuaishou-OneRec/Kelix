#!/bin/bash

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


mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install timm==1.0.15"
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install fastparquet==2024.2.0"
