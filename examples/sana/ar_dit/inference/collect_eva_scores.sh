#!/usr/bin/env bash
# Script to collect evaluation scores from exp18_ar_dit_multiscale_324tokens_2e-5
# Usage: bash examples/sana/ar_dit/inference/collect_eva_scores.sh

set -euo pipefail

# Set the DCP checkpoint directory
DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5/"

# Set model tag (default is BLIP3OTransformersSFT)
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="exp18_eval"

# Ensure Python path is set
export PYTHONPATH="${PYTHONPATH:-.}"

echo "Starting evaluation score collection for exp18_ar_dit_multiscale_324tokens_2e-5"
echo "DCP checkpoint directory: $DCP_CKPT_DIR"

# Run the visualize mode to collect scores (only need dcp-ckpt-dir for visualize mode)
python recipes/sana/inference_ar2image.py \
    --mode visualize \
    --dcp-ckpt-dir "$DCP_CKPT_DIR" \
    --model-tag "$MODEL_TAG" \
    --tb-log-name "$TB_LOG_NAME"

echo "Score collection completed!"
echo "Results saved to:"
echo "- TensorBoard logs: $DCP_CKPT_DIR/tf_eval_log"
echo "- CSV file: $DCP_CKPT_DIR/gen_eval_scores.csv"

# Optional: Start TensorBoard if needed
echo
echo "To view TensorBoard results, run:"
echo "tensorboard --logdir $DCP_CKPT_DIR/tf_eval_log"