#!/usr/bin/env bash
# Wrapper script to run the Python auto monitoring tool
# run mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pkill -9 python3; pkill -9 python"

export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
set -euo pipefail

# Default parameters (can be overridden)
MONITOR_INTERVAL=30
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="auto_eval"

KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9.1/v8_stage3_0.29/step18000/global_step18000/muse_converted"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp10/step7000/global_step7000/muse_converted/"

DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp15x/exp154_0116sftv1_1e-4lr_from152_sft_from6k"
DATASET_CONFIG="examples/sana/ar_dit/exp154_eval_different_prompt/exp154_prompt_step3000_v1.json"
MAX_CONDITION_LENGTH=720
INFERENCE_SCRIPT="examples/sana/ar_dit/exp42_eval_different_prompt/mpi_infer_custom_cond_spe_overwrite.sh"
MODEL_CONFIG_OVERRIDES="model_max_length=720"
log_file=${DCP_CKPT_DIR}/auto_monitor1.log
echo "log_file=${log_file}"

# Run the Python script with all parameters
PYTHONPATH=. \
INFERENCE_SCRIPT=${INFERENCE_SCRIPT} \
MODEL_CONFIG_OVERRIDES=${MODEL_CONFIG_OVERRIDES} \
MAX_CONDITION_LENGTH=${MAX_CONDITION_LENGTH} \
nohup python3 -u examples/keye_ar/auto_infer_eval.py \
    --dcp-ckpt-dir "$DCP_CKPT_DIR" \
    --monitor-interval "$MONITOR_INTERVAL" \
    --model-tag "$MODEL_TAG" \
    --tb-log-name "$TB_LOG_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --keye-ar-dir "$KEYE_AR_DIR" \
    --inference-script "$INFERENCE_SCRIPT" \
    --eval-id "prompt1" \
    --good-steps "3000" \
    > ${log_file} &
