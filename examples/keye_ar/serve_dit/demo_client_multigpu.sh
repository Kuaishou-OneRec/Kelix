#!/usr/bin/env bash
set -euo pipefail

# client demo (multi-gpu)：向服务端发送 prompt，可选指定 output_path 和 gpu_id。
#
# 用法：
#   bash examples/keye_ar/serve_dit/demo_client_multigpu.sh "a cat" [/path/to/save.jpg] [gpu_id]
#
# 环境变量：
#   HOST / PORT

PROMPT=${1:-"a black cat."}
OUT_PATH=${2:-""}
GPU_ID=${3:-""}
HOST=${HOST:-"10.48.50.167"}
PORT=${PORT:-"18080"}

PAYLOAD=$(python3 - <<'PY'
import json
import os

prompt = os.environ.get('PROMPT', 'a black cat.')
out_path = os.environ.get('OUT_PATH', '')
gpu_id = os.environ.get('GPU_ID', '')

req = {"prompt": prompt}
if out_path:
    req["output_path"] = out_path
if gpu_id:
    req["gpu_id"] = int(gpu_id)

print(json.dumps(req, ensure_ascii=False))
PY
)

echo "POST http://$HOST:$PORT/generate"
echo "payload=$PAYLOAD"

echo "response:"
curl -sS -X POST "http://$HOST:$PORT/generate" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD" | python3 -m json.tool
