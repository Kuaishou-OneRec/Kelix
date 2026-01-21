#!/usr/bin/env bash
set -euo pipefail

# while client demo：循环调用 demo_client_v2.sh
#
# 用法：
#   bash examples/keye_ar/serve_dit/while_demo_client.sh "a black cat." 10 2
#
# 参数：
#   $1 prompt（可选，默认 "a black cat."）
#   $2 循环次数（可选，默认无限循环）
#   $3 每次间隔秒数（可选，默认 1 秒）
#
# 环境变量：
#   HOST / PORT：透传给 demo_client_v2.sh

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
PROMPT=${1:-"a black cat."}
LOOPS=${2:-""}
SLEEP_SECS=${3:-1}

CLIENT_SH="$ROOT_DIR/examples/keye_ar/serve_dit/demo_client_v2.sh"

if [[ ! -f "$CLIENT_SH" ]]; then
  echo "client script not found: $CLIENT_SH" >&2
  exit 1
fi

n=0
while true; do
  n=$((n+1))
  echo "\n===== [$n] $(date '+%Y-%m-%d %H:%M:%S') ====="
  bash "$CLIENT_SH" "$PROMPT"

  if [[ -n "$LOOPS" && "$n" -ge "$LOOPS" ]]; then
    echo "Reached loops=$LOOPS, exiting."
    break
  fi

  sleep "$SLEEP_SECS"
done
