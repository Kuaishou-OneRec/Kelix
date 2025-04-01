#!/bin/bash
set -e

# 检查输入参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pid_list_file>"
    exit 1
fi

PID_LIST_FILE=$1
OUTPUT_DIR="./output"
DATASET_DIR="${OUTPUT_DIR}/dataset"

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATASET_DIR}"

# Step 1: 下载PID信息
echo "Step 1: Downloading PID information..."
python3 download.py "${PID_LIST_FILE}" --output-dir "${OUTPUT_DIR}"

# Step 2: 准备数据集
echo "Step 2: Preparing dataset..."
python3 prepare_dataset.py \
    --input-dir "${OUTPUT_DIR}" \
    --output-path "${DATASET_DIR}/dataset" \
    --prompt-name "describe_video" \
    --num-shards 4

# # Step 3: 运行批量推理
# echo "Step 3: Running batch inference..."
# python -m recovlm.recipes.offline_batch_inference \
#     --input "${DATASET_DIR}/dataset.*.parquet" \
#     --output "${OUTPUT_DIR}/results.jsonl" \
#     --batch_size 4

# echo "All steps completed successfully!"