#!/bin/bash

# Debug script for testing training loop with fixed data
# Usage: bash run_debug_training.sh
# For single GPU: torchrun --nproc_per_node=1 examples/keye_tokenizer_end2end_video/debug_training.py ...

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Configuration
DATASET_CONFIG="examples/keye_tokenizer_end2end_video/debug_config.json"
MODEL_DIR="/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"
NUM_TRAINING_STEPS=100
OVERFIT_BATCHES=1
LOGGING_PER_STEP=1

# Export Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Source environment if available
if [ -f "set_env.sh" ]; then
    source set_env.sh
fi

echo "=========================================="
echo "Training Debug Script (Single GPU)"
echo "=========================================="
echo "Dataset config: $DATASET_CONFIG"
echo "Model dir: $MODEL_DIR"
echo "Training steps: $NUM_TRAINING_STEPS"
echo "Overfit batches: $OVERFIT_BATCHES"
echo "=========================================="
echo ""

# Run with torchrun for single GPU
torchrun --nproc_per_node=1 \
    examples/keye_tokenizer_end2end_video/debug_training.py \
    --model-dir "$MODEL_DIR" \
    --dataset-config "$DATASET_CONFIG" \
    --num-training-steps "$NUM_TRAINING_STEPS" \
    --overfit-batches "$OVERFIT_BATCHES" \
    --logging-per-step "$LOGGING_PER_STEP" \
    --lr 2e-4 \
    --vision_lr 2e-5 \
    --min_lr 1e-7 \
    --weight-decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --codebook-loss-weight 1.0 \
    --commitment-loss-weight 0.25 \
    --seed 19260817 \
    --use-flash-attention-2 \
    --enable-gradient-checkpointing \
    --freeze-navit \
    --freeze-llm \
    --freeze-navit-mlp-ar \
    --context-parallel-size 1

echo ""
echo "=========================================="
echo "Training debug completed!"
echo "=========================================="

