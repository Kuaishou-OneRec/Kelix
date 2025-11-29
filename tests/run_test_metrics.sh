#!/bin/bash
# Quick test script for Metrics and StepScheduler

set -e

echo "=================================================="
echo "Metrics and StepScheduler Distributed Test"
echo "=================================================="
echo ""

# Check if running on a GPU machine
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "  Available GPUs: $NUM_GPUS"
    echo ""
else
    echo "✗ No NVIDIA GPU detected, will use CPU"
    NUM_GPUS=0
    echo ""
fi

# Test parameters
GRAD_ACC_STEPS=4
LOGGING_PER_STEP=5
SAVE_CKPT_PER_STEP=10
NUM_ITERATIONS=50

echo "Test Configuration:"
echo "  Gradient Accumulation Steps: $GRAD_ACC_STEPS"
echo "  Logging Per Step: $LOGGING_PER_STEP"
echo "  Save Checkpoint Per Step: $SAVE_CKPT_PER_STEP"
echo "  Total Iterations: $NUM_ITERATIONS"
echo ""

# Run test based on available GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running multi-GPU test with $NUM_GPUS GPUs..."
    echo "=================================================="
    torchrun --nproc_per_node=$NUM_GPUS test_metrics_distributed.py \
        --gradient-accumulation-steps $GRAD_ACC_STEPS \
        --logging-per-step $LOGGING_PER_STEP \
        --save-checkpoint-per-step $SAVE_CKPT_PER_STEP \
        --num-iterations $NUM_ITERATIONS
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single-GPU test..."
    echo "=================================================="
    python test_metrics_distributed.py \
        --gradient-accumulation-steps $GRAD_ACC_STEPS \
        --logging-per-step $LOGGING_PER_STEP \
        --save-checkpoint-per-step $SAVE_CKPT_PER_STEP \
        --num-iterations $NUM_ITERATIONS
else
    echo "Running CPU test..."
    echo "=================================================="
    python test_metrics_distributed.py \
        --gradient-accumulation-steps $GRAD_ACC_STEPS \
        --logging-per-step $LOGGING_PER_STEP \
        --save-checkpoint-per-step $SAVE_CKPT_PER_STEP \
        --num-iterations $NUM_ITERATIONS \
        --backend gloo
fi

echo ""
echo "=================================================="
echo "Test completed! Check test_output/ for results."
echo "=================================================="
