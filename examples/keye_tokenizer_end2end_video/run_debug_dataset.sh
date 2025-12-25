#!/bin/bash

# Debug script for testing Dataset packing and sample_idx
# Usage: bash run_debug_dataset.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Configuration
DATASET_CONFIG="examples/keye_tokenizer_end2end_video/debug_config.json"
MODEL_DIR="/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"
NUM_BATCHES=10
NUM_GPUS=8  # Number of GPUs (processes) to simulate

# Export Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Source environment if available
if [ -f "set_env.sh" ]; then
    source set_env.sh
fi

echo "=========================================="
echo "Dataset Debug Script (Multi-GPU)"
echo "=========================================="
echo "Dataset config: $DATASET_CONFIG"
echo "Model dir: $MODEL_DIR"
echo "Number of batches: $NUM_BATCHES"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# Run the debug script with torchrun for proper distributed initialization
# Using multiple GPUs (nproc_per_node=NUM_GPUS)
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    examples/keye_tokenizer_end2end_video/debug_dataset.py \
    --dataset-config "$DATASET_CONFIG" \
    --model-dir "$MODEL_DIR" \
    --num-batches "$NUM_BATCHES"

echo ""
echo "=========================================="
echo "Debug completed!"
echo "=========================================="

