#!/bin/bash
set -e

# 获取当前脚本所在目录的父目录的父目录的绝对路径
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 将父目录添加到PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# 检查输入参数
if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <pid_list_file> [dataset_name] [prompt_name] [model_dir]"
    echo "Example: $0 pid_list.txt my_dataset describe_video /path/to/model"
    exit 1
fi

CACHE_DIR="/llm_reco/zhouyang12/.cache"
PHOTO_DIR="${CACHE_DIR}/Photo"
DATASET_DIR="${CACHE_DIR}/Dataset"

PID_LIST_FILE=$1
DATASET_NAME=${2:-"dataset"}  # 如果没有提供dataset_name，默认使用"dataset"
PROMPT_NAME=${3:-"describe_video"}  # 如果没有提供prompt_name，默认使用"describe_video"
MODEL_DIR=${4:-"/llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-72B-Instruct"}  # 如果没有提供model_dir，使用默认值

# 创建必要的目录
mkdir -p "${PHOTO_DIR}"
mkdir -p "${DATASET_DIR}"

# 为当前数据集创建专门的目录
CURRENT_DATASET_DIR="${DATASET_DIR}/${DATASET_NAME}"
mkdir -p "${CURRENT_DATASET_DIR}"

# Step 1: 下载PID信息
echo "Step 1: Downloading PID information..."
python3 examples/infer_pids/download.py \
    "${PID_LIST_FILE}" \
    --output-dir "${PHOTO_DIR}"

# Step 2: 准备数据集
echo "Step 2: Preparing dataset..."
python3 examples/infer_pids/prepare_dataset.py \
    "${PID_LIST_FILE}" \
    --output-path "${CURRENT_DATASET_DIR}" \
    --photo-dir "${PHOTO_DIR}" \
    --prompt-name "${PROMPT_NAME}" \
    --num-shards 4 \
    --model-path "${MODEL_DIR}"

# Step 3: 运行批量推理
# echo "Step 3: Running batch inference..."
# # 获取dataset_config.json的路径
# DATASET_CONFIG="${CURRENT_DATASET_DIR}/dataset_config.json"
# # 设置输出目录
# OUTPUT_DIR="${MODEL_DIR}/${DATASET_NAME}"

# mkdir -p "${OUTPUT_DIR}"

# # 调用run_inference.sh进行推理
# bash examples/infer_pids/run_inference.sh \
#     "${MODEL_DIR}" \
#     "${DATASET_CONFIG}" \
#     "${OUTPUT_DIR}"

# echo "All steps completed successfully!"