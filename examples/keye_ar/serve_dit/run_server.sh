#!/usr/bin/env bash
set -euo pipefail

# 启动 DiT/AR2Image 本地 HTTP 服务
#
# 用法：
#   bash examples/keye_ar/serve_dit/run_server.sh [config.json]
#
# 默认使用同目录下的 config_local_ar2image.json

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
CONFIG_PATH=${1:-"$ROOT_DIR/examples/keye_ar/serve_dit/config_local_ar2image.json"}

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo "Using config: $CONFIG_PATH"
python3 -u "$ROOT_DIR/examples/keye_ar/serve_dit/serve_visualize_reconstruction.py" \
  --config "$CONFIG_PATH" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-1}"

