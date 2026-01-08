#!/usr/bin/env bash
# Run the inference/visualization demo for Keye AR + DiT (单机版本)
# Usage: modify the defaults below or call with env overrides, then run:
#   bash examples/sana/ar_dit/inference/run_local_infer_visualize_reconstruction.sh
# Example:
#   MODEL_DIR=/path/to/model VAEDIR=/path/to/vae KEYE_AR_DIR=/path/to/keye bash $0
# For DCP checkpoint:
#   MODEL_DIR=/path/to/dcp/checkpoint DCP_CKPT_DIR=/path/to/source/dir DCP_TAG=global_step8000 bash $0

set -euo pipefail

# Ensure project root is on PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-.}"

# ---- Defaults (单机版本，移除分布式相关配置) ----
MODEL_DIR="/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/DiTSFT/batch6_324_1024_more_data/global_step80000/muse_converted/"
VAE_DIR="/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"
DATASET_CONFIG="examples/sana/ar_dit/exp21_ar_dit_324tokens_1e-4_reproduce_inf.json"
PARQUET_PATH="/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet"
OUTPUT_DIR="./vis_output_local"
RESULTS_DIR="./vis_output_local/results"
NUM_IMAGES=1
DEVICE="cuda"           # set to "cuda" if running on GPU
DTYPE="bfloat16"        # float32 for CPU runs; bfloat16/float16 for GPU runs
NUM_SAMPLING_STEPS=20
FLOW_SHIFT=3.0
CFG_SCALE=1.0
MAX_CONDITION_LENGTH=324
IMAGE_SIZE=1024
SEED=42
MODEL_CONFIG_OVERRIDES=""  # Model config overrides, e.g., "caption_channels=4096 model_max_length=324"
DCP_CKPT_DIR=""
DCP_TAG=""


# Allow overrides from environment
MODEL_DIR=${MODEL_DIR:-$MODEL_DIR}
VAE_DIR=${VAE_DIR:-$VAE_DIR}
KEYE_AR_DIR=${KEYE_AR_DIR:-$KEYE_AR_DIR}
DATASET_CONFIG=${DATASET_CONFIG:-$DATASET_CONFIG}
PARQUET_PATH=${PARQUET_PATH:-$PARQUET_PATH}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_DIR}
RESULTS_DIR=${RESULTS_DIR:-$RESULTS_DIR}
NUM_IMAGES=${NUM_IMAGES:-$NUM_IMAGES}
DEVICE=${DEVICE:-$DEVICE}
DTYPE=${DTYPE:-$DTYPE}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-$NUM_SAMPLING_STEPS}
FLOW_SHIFT=${FLOW_SHIFT:-$FLOW_SHIFT}
CFG_SCALE=${CFG_SCALE:-$CFG_SCALE}
MAX_CONDITION_LENGTH=${MAX_CONDITION_LENGTH:-$MAX_CONDITION_LENGTH}
IMAGE_SIZE=${IMAGE_SIZE:-$IMAGE_SIZE}
SEED=${SEED:-$SEED}
MODEL_CONFIG_OVERRIDES=${MODEL_CONFIG_OVERRIDES:-$MODEL_CONFIG_OVERRIDES}
DCP_CKPT_DIR=${DCP_CKPT_DIR:-$DCP_CKPT_DIR}
DCP_TAG=${DCP_TAG:-$DCP_TAG}

# ---- Prepare flags ----

# Prepare model config overrides flag
MODEL_CONFIG_OVERRIDES_FLAG=""
if [ -n "$MODEL_CONFIG_OVERRIDES" ]; then
  MODEL_CONFIG_OVERRIDES_FLAG="--model-config-overrides $MODEL_CONFIG_OVERRIDES"
fi

# Prepare DCP flags
DCP_FLAGS=""
if [ -n "$DCP_TAG" ]; then
  DCP_FLAGS="--dcp-tag $DCP_TAG"
  if [ -n "$DCP_CKPT_DIR" ]; then
    DCP_FLAGS="$DCP_FLAGS --dcp-ckpt-dir $DCP_CKPT_DIR"
  fi
fi


mkdir -p "${OUTPUT_DIR}"
mkdir -p "${RESULTS_DIR}"

# Run the local inference script
PYTHONPATH=. python tests/models/keye_ar/local_infer_visualize_reconstruction.py \
  --model-dir "${MODEL_DIR}" \
  --vae-dir "${VAE_DIR}" \
  --keye-ar-dir "${KEYE_AR_DIR}" \
  --dataset-config "${DATASET_CONFIG}" \
  --parquet-path "${PARQUET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-images ${NUM_IMAGES} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --num-sampling-steps ${NUM_SAMPLING_STEPS} \
  --flow-shift ${FLOW_SHIFT} \
  --cfg-scale ${CFG_SCALE} \
  --max-condition-length ${MAX_CONDITION_LENGTH} \
  --image-size ${IMAGE_SIZE} \
  --seed ${SEED} \
  --results-dir "${RESULTS_DIR}" \
  ${MODEL_CONFIG_OVERRIDES_FLAG} \
  ${DCP_FLAGS}
