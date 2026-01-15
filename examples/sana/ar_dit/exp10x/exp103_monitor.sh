#!/usr/bin/env bash
# Wrapper script to run the Python auto monitoring tool
# run mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pkill -9 python3; pkill -9 python"

export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
set -euo pipefail

# Default parameters (can be overridden)
MONITOR_INTERVAL=30
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="auto_eval"

MODEL_DIR="/mmu_mllm_hdd_2/yangyiping/models/SANA1.5_4.8B_1024px_diffusers_muse_converted-0105-advanced-conf/"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp9/step5500/global_step5500/muse_converted/"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp9/step5500/global_step5500/muse_converted/"
DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp10x/exp103_ar_dit_324tokens_1e-4_baseline/"
DATASET_CONFIG="examples/sana/ar_dit/exp10x/exp100_ar_dit_324tokens_1e-4_sft_from52k.json"
MAX_CONDITION_LENGTH=720
INFERENCE_SCRIPT="examples/sana/ar_dit/inference/mpi_infer_custom_cond_spe.sh"
MODEL_CONFIG_OVERRIDES="model_max_length=720"
log_file=${DCP_CKPT_DIR}/auto_monitor.log
echo "log_file=${log_file}"
# Run the Python script with all parameters
PYTHONPATH=. \
INFERENCE_SCRIPT=${INFERENCE_SCRIPT} \
MODEL_CONFIG_OVERRIDES=${MODEL_CONFIG_OVERRIDES} \
MAX_CONDITION_LENGTH=${MAX_CONDITION_LENGTH} \
MODEL_DIR=${MODEL_DIR} \
nohup python3 -u examples/keye_ar/auto_infer_eval.py \
    --dcp-ckpt-dir "$DCP_CKPT_DIR" \
    --monitor-interval "$MONITOR_INTERVAL" \
    --model-tag "$MODEL_TAG" \
    --tb-log-name "$TB_LOG_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --keye-ar-dir "$KEYE_AR_DIR" \
    --inference-script "$INFERENCE_SCRIPT" \
    --good-steps "200" \
    > ${log_file} &
