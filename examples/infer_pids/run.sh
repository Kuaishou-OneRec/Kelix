#!/bin/bash
set -e

# 获取当前脚本所在目录的父目录的父目录的绝对路径
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 将父目录添加到PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查输入参数
if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <pid_list_file> [dataset_name] [prompt_name]"
    echo "Example: $0 pid_list.txt my_dataset describe_video"
    exit 1
fi

CACHE_DIR="/llm_reco/zhouyang12/.cache"
PHOTO_DIR="${CACHE_DIR}/Photo"
DATASET_DIR="${CACHE_DIR}/Dataset"

PID_LIST_FILE=$1
DATASET_NAME=${2:-"dataset"}  # 如果没有提供dataset_name，默认使用"dataset"
PROMPT_NAME=${3:-"describe_video"}  # 如果没有提供prompt_name，默认使用"describe_video"

# 创建必要的目录
mkdir -p "${PHOTO_DIR}"
mkdir -p "${DATASET_DIR}"

# Step 1: 下载PID信息
echo "Step 1: Downloading PID information..."
python3 examples/infer_pids/download.py \
    "${PID_LIST_FILE}" \
    --output-dir "${PHOTO_DIR}"

# Step 2: 准备数据集
echo "Step 2: Preparing dataset..."
python3 examples/infer_pids/prepare_dataset.py \
    "${PID_LIST_FILE}" \
    --output-path "${DATASET_DIR}/${DATASET_NAME}" \
    --photo-dir "${PHOTO_DIR}" \
    --prompt-name "${PROMPT_NAME}" \
    --num-shards 4

# # 运行批量推理
# echo "Running batch inference..."
# python -m recovlm.recipes.offline_batch_inference \
#     --input "${DATASET_DIR}/${DATASET_NAME}.*.parquet" \
#     --output "${OUTPUT_DIR}/${DATASET_NAME}_results.jsonl" \
#     --batch_size 4

# echo "All steps completed successfully!"