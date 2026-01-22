#!/usr/bin/env bash
set -euo pipefail

# 启动 DiT/AR2Image 多 GPU HTTP 服务
#
# 用法：
#   bash examples/keye_ar/serve_dit/run_server_multigpu.sh [config.json]
#
# 说明：
# - 服务端脚本：tests/models/keye_ar/demo_local_infer_visualize_reconstruction_multigpus.py
# - 通过 --config 指定 LocalAR2ImageConfig json
# - 通过 CUDA_VISIBLE_DEVICES 控制可见 GPU（建议与 config 里的 service_gpu_ids 对齐）

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
CONFIG_PATH=${1:-"$ROOT_DIR/examples/keye_ar/serve_dit/config_local_ar2image.json"}

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo "Using config: $CONFIG_PATH"
python3 -u "$ROOT_DIR/tests/models/keye_ar/demo_local_infer_visualize_reconstruction_multigpus.py" \
  --config "$CONFIG_PATH" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0,1}"
