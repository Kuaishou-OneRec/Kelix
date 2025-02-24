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