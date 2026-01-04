#!/usr/bin/env bash
# Auto monitoring script for DCP checkpoint inference and evaluation

set -euo pipefail

# Configuration
MONITOR_INTERVAL=30
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="auto_eval"
# Write DCP_CKPT_DIR directly in the script
DCP_CKPT_DIR="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5"

# Helper functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_inference() {
    local step_name="$1"
    log "Starting inference for $step_name"
    
    export DCP_TAG="$step_name"
    export OUTPUT_DIR="$DCP_CKPT_DIR/$step_name/inference/GenEval/outputs"
    export DATASET_CONFIG="${DATASET_CONFIG:-examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json}"
    export KEYE_AR_DIR="${KEYE_AR_DIR:-/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted}"
    
    mkdir -p "$OUTPUT_DIR"
    
    bash examples/sana/ar_dit/inference/mpi_infer_custom.sh > "$OUTPUT_DIR/inference.log" 2>&1
    return $?
}

run_evaluation() {
    local step_name="$1"
    log "Starting evaluation for $step_name"
    
    cd "/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit" || return 1
    source "/mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate" >/dev/null 2>&1
    conda activate ulmevalkit2 >/dev/null 2>&1
    
    local work_dir="$DCP_CKPT_DIR/$step_name/inference/GenEval/outputs/ulmeval/aggresults/"
    mkdir -p "$work_dir"
    
    max_infer_items=300000 PYTHONPATH=. torchrun --nproc_per_node=8 \
        run_eval_only.py --config config/blip3o_sft_step800.json \
        --eval-id default --work-dir "$work_dir" > "$work_dir/eval.log" 2>&1
    return $?
}

collect_scores() {
    local step_name="$1"
    log "Collecting scores for $step_name"
    
    python recipes/sana/inference_ar2image.py \
        --mode visualize \
        --dcp-ckpt-dir "$DCP_CKPT_DIR" \
        --model-tag "$MODEL_TAG" \
        --tb-log-name "${TB_LOG_NAME}_$(echo "$step_name" | sed 's/global_step//')"
    return $?
}

# Main monitoring function
monitor() {
    log "Starting monitoring for $DCP_CKPT_DIR"
    declare -A existing_steps
    
    while true; do
        while IFS= read -r -d '' step_dir; do
            if [[ "$step_dir" =~ global_step[0-9]+ ]]; then
                step_name=$(basename "$step_dir")
                if [[ -z "${existing_steps[$step_name]:-}" ]] && [ -f "$step_dir/.metadata" ]; then
                    log "Found new step: $step_name"
                    existing_steps["$step_name"]=1
                    
                    if run_inference "$step_name"; then
                        if run_evaluation "$step_name"; then
                            collect_scores "$step_name"
                        fi
                    fi
                fi
            fi
        done < <(find "$DCP_CKPT_DIR" -maxdepth 1 -type d -name "global_step*" -print0 2>/dev/null)
        
        sleep "$MONITOR_INTERVAL"
    done
}

LOG_FILE="$DCP_CKPT_DIR/auto_monitor.log"

echo "Starting auto monitoring..."
echo "DCP checkpoint directory: $DCP_CKPT_DIR"
echo "Log file: $LOG_FILE"
echo

mkdir -p "$DCP_CKPT_DIR"
monitor > "$LOG_FILE" 2>&1 &

echo "Background process started with PID: $!"
echo "You can stop it with: kill $!"
echo "To check logs: tail -f '$LOG_FILE'"