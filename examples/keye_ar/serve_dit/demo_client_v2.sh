#!/usr/bin/env bash
set -euo pipefail

# client demo：向服务端发送 prompt，拿到 output_path
#
# 用法：
#   bash examples/keye_ar/serve_dit/demo_client.sh "a cat" [/path/to/save.jpg]

PROMPT=${1:-"a black cat and an apple."}
OUT_PATH=${2:-"vis_output_local/service_outputs/tmp.jpg"}
HOST=${HOST:-"10.48.50.167"}
PORT=${PORT:-"18080"}

if [[ -n "$OUT_PATH" ]]; then
  PAYLOAD=$(python3 - <<PY
import json
print(json.dumps({"prompt": "$PROMPT", "output_path": "$OUT_PATH"}, ensure_ascii=False))
PY
)
else
  PAYLOAD=$(python3 - <<PY
import json
print(json.dumps({"prompt": "$PROMPT"}, ensure_ascii=False))
PY
)
fi

echo "POST http://$HOST:$PORT/generate"
echo "payload=$PAYLOAD"

echo "response:"
curl -sS -X POST "http://$HOST:$PORT/generate" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD" | python3 -m json.tool

