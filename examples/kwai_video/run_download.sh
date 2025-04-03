#!/bin/bash
set -e

PID_LIST_FILE=$1
# WARN: 请替换成自己的cache_dir，否则可能被其他人删除。
CACHE_DIR=${2:-"/llm_reco/zhouyang12/.cache"}

PHOTO_DIR=${CACHE_DIR}/Photo
DATASET_DIR=${CACHE_DIR}/Dataset

# 创建必要的目录
mkdir -p "${PHOTO_DIR}"
mkdir -p "${DATASET_DIR}"


echo "Downloading PID information..."
python3 examples/kwai_video/download.py \
    "${PID_LIST_FILE}" \
    --output-dir "${PHOTO_DIR}"