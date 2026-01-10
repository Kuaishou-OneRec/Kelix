#!/usr/bin/env bash
# Wrapper script to run the Python auto monitoring tool
# run mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pkill -9 python3; pkill -9 python"

export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
set -euo pipefail

# Default parameters (can be overridden)
DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5"
MONITOR_INTERVAL=30
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="auto_eval"
DATASET_CONFIG="examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"
INFERENCE_SCRIPT="examples/sana/ar_dit/inference/mpi_infer_custom.sh"


DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp22_ar_dit_324tokens_1e-4_reproduce"
#DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp25_ar_dit_324tokens_1e-4_reproduce_lbs_v2/"
#DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp30_ar_dit_324tokens_1e-4_reproduce_lbs/"

DATASET_CONFIG="examples/sana/ar_dit/exp21_ar_dit_324tokens_1e-4_reproduce.json"
DATASET_CONFIG="examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json"
INFERENCE_SCRIPT="examples/sana/ar_dit/inference/mpi_infer_custom_v2.sh"


#KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9.1/v8_stage3_0.29/step18000/global_step18000/muse_converted"
#DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp30_ar_dit_324tokens_1e-4_reproduce_lbs/"
#DATASET_CONFIG="examples/sana/ar_dit/exp30_ar_dit_324tokens_1e-4_reproduce_lbs_inf.jsonn"

DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp31_ar_dit_324tokens_1e-4_cond_special/"
#DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp36_ar_dit_324tokens_1e-3_cond_special_8gpu/"
KEYE_AR_DIR="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9.1/v8_stage3_0.29/step18000/global_step18000/muse_converted"


#DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp1031_ar_dit_324tokens_1e-4_cond_special_16gpu"
#DATASET_CONFIG="examples/sana/ar_dit/exp1031_ar_dit_324tokens_1e-4_cond_special_16gpu_inf.json"

MAX_CONDITION_LENGTH=658
INFERENCE_SCRIPT="examples/sana/ar_dit/inference/mpi_infer_custom_cond_spe.sh"
MODEL_CONFIG_OVERRIDES="model_max_length=658"
log_file=${DCP_CKPT_DIR}/auto_monitor.log
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
    > ${log_file} &
