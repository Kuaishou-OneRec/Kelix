#!/bin/bash
set -e

PID_LIST_FILE=${1:-"/llm_reco_ssd/zhouyang12/data/creator/photo_ids.txt"}
# WARN: 请替换成自己的cache_dir，否则可能被其他人删除。
CACHE_DIR=${2:-"/llm_reco/zhouyang12/.cache"}

PHOTO_DIR=${CACHE_DIR}/Photo

# 创建必要的目录
mkdir -p "${PHOTO_DIR}"

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Downloading PID information..."
python3 tools/kwai_video/download_kwai_video.py \
    "${PID_LIST_FILE}" \
    --output-dir "${PHOTO_DIR}"