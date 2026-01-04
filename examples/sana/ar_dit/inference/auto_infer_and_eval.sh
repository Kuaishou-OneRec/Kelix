#!/usr/bin/env bash
# Auto monitoring script for DCP checkpoint inference and evaluation
# Usage: bash examples/sana/ar_dit/inference/auto_infer_and_eval.sh DCP_CKPT_DIR
#
# This script monitors a DCP checkpoint directory for new global_step folders
# with .metadata files, and automatically runs inference -> evaluation -> score collection

set -euo pipefail

# Script configuration
MONITOR_INTERVAL=30  # Check every 5 minutes
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="auto_eval"
LOG_DIR="/tmp/auto_infer_eval_logs"
mkdir -p "$LOG_DIR"

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/auto_infer_eval.log"
}

# Function to run inference
run_inference() {
    local step_name="$1"
    log "Starting inference for $step_name"
    
    # Set environment variables including commonly used parameters
    export DCP_CKPT_DIR="$DCP_CKPT_DIR"
    export DCP_TAG="$step_name"
    export OUTPUT_DIR="$DCP_CKPT_DIR/$step_name/inference/GenEval/outputs"
    export KEYE_AR_DIR="${KEYE_AR_DIR:-/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted}"
    export DATASET_CONFIG="${DATASET_CONFIG:-examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json}"
    
    # Create output and log directories
    mkdir -p "$OUTPUT_DIR"
    local auto_log_dir="$DCP_CKPT_DIR/auto_infer_logs"
    mkdir -p "$auto_log_dir"
    
    # Run inference script with proper log redirection
    local inference_log="$auto_log_dir/inference_${step_name}_$(date +%Y%m%d_%H%M%S).log"
    log "Running inference - logs: $inference_log"
    log "Using DCP_CKPT_DIR: $DCP_CKPT_DIR"
    log "Using DCP_TAG: $step_name"
    
    # Capture both stdout and stderr to the log file
    bash examples/sana/ar_dit/inference/mpi_run_infer_visualize_reconstruction_notf_324.sh > "$inference_log" 2>&1
    
    local inference_status=$?
    
    if [ $inference_status -eq 0 ]; then
        log "Inference completed successfully for $step_name"
        return 0
    else
        log "Error: Inference failed for $step_name - check $inference_log"
        # Also log the last few lines of the error for quick debugging
        tail -20 "$inference_log" | while read -r line; do
            log "INFERENCE_ERROR: $line"
        done
        return 1
    fi
}

# Function to run evaluation
run_evaluation() {
    local step_name="$1"
    log "Starting evaluation for $step_name"
    
    # Check if ULMEvalKit directory exists
    if [ ! -d "/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit" ]; then
        log "Error: ULMEvalKit directory not found at /llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit"
        return 1
    fi
    
    cd "/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit"
    
    # Setup environment
    source "/mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate" >/dev/null 2>&1
    conda activate ulmevalkit2 >/dev/null 2>&1
    
    local cf="blip3o_sft_step800"
    local config="config/${cf}.json"
    local work_dir="$DCP_CKPT_DIR/$step_name/inference/GenEval/outputs/ulmeval/aggresults/"
    
    mkdir -p "$work_dir"
    local eval_log="$work_dir/eval_${cf}_$(date +%Y%m%d_%H%M%S).out"
    
    log "Running evaluation - config: $config, logs: $eval_log"
    
    # Run evaluation
    max_infer_items=300000 \
    PYTHONPATH=. \
    torchrun \
        --nproc_per_node=8 \
        run_eval_only.py --config "$config" \
        --eval-id default \
        --work-dir "$work_dir" \
        > "$eval_log" 2>&1
    
    local eval_status=$?
    cd - > /dev/null
    
    if [ $eval_status -eq 0 ]; then
        log "Evaluation completed successfully for $step_name"
        return 0
    else
        log "Error: Evaluation failed for $step_name - check $eval_log"
        return 1
    fi
}

# Function to collect scores
collect_scores() {
    local step_name="$1"
    local step_number="$2"
    log "Collecting evaluation scores for $step_name"
    
    python recipes/sana/inference_ar2image.py \
        --mode visualize \
        --dcp-ckpt-dir "$DCP_CKPT_DIR" \
        --model-tag "$MODEL_TAG" \
        --tb-log-name "${TB_LOG_NAME}_${step_number}"
    
    if [ $? -eq 0 ]; then
        log "Score collection completed for $step_name"
        log "Results available at:"
        log "- TensorBoard logs: $DCP_CKPT_DIR/tf_eval_log"
        log "- CSV file: $DCP_CKPT_DIR/gen_eval_scores.csv"
        return 0
    else
        log "Error: Score collection failed for $step_name"
        return 1
    fi
}

# Main function
main() {
    # Check if DCP_CKPT_DIR is provided
    if [ $# -eq 0 ]; then
        echo "Usage: $0 DCP_CKPT_DIR"
        echo "Example: $0 /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5/"
        exit 1
    fi

    DCP_CKPT_DIR="$1"
    log "Starting auto monitoring for DCP checkpoint directory: $DCP_CKPT_DIR"

    # Ensure the directory exists
    if [ ! -d "$DCP_CKPT_DIR" ]; then
        log "Error: DCP checkpoint directory $DCP_CKPT_DIR does not exist!"
        exit 1
    fi

    # Get list of existing global_step directories to exclude from initial monitoring
    declare -A existing_steps
    if [ -d "$DCP_CKPT_DIR" ]; then
        while IFS= read -r -d '' step_dir; do
            if [[ "$step_dir" =~ global_step[0-9]+ ]]; then
                step_name=$(basename "$step_dir")
                existing_steps["$step_name"]=1
            fi
        done < <(find "$DCP_CKPT_DIR" -maxdepth 1 -type d -name "global_step*" -print0 2>/dev/null)
    fi

    log "Existing steps (excluded from monitoring): ${!existing_steps[*]}"
    log "Starting monitoring for new global_step directories..."

    # Main monitoring loop
    while true; do
        # Find all global_step directories
        new_steps_found=()
        while IFS= read -r -d '' step_dir; do
            if [[ "$step_dir" =~ global_step[0-9]+ ]]; then
                step_name=$(basename "$step_dir")
                # Check if this is a new step
                if [[ -z "${existing_steps[$step_name]:-}" ]]; then
                    # Check if .metadata file exists (indicating step is ready)
                    if [ -f "$step_dir/.metadata" ]; then
                        log "Found new ready step: $step_name"
                        new_steps_found+=("$step_name")
                        existing_steps["$step_name"]=1
                    else
                        log "Found new step (not ready): $step_name - waiting for .metadata file"
                    fi
                fi
            fi
        done < <(find "$DCP_CKPT_DIR" -maxdepth 1 -type d -name "global_step*" -print0 2>/dev/null)

        # Process new steps
        for step_name in "${new_steps_found[@]}"; do
            log "=== Processing step: $step_name ==="
            
            # Extract step number
            step_number=$(echo "$step_name" | sed 's/global_step//')
            log "Step number: $step_number"
            
            # Run the full pipeline: Inference -> Evaluation -> Score Collection
            if run_inference "$step_name"; then
                if run_evaluation "$step_name"; then
                    collect_scores "$step_name" "$step_number"
                fi
            fi
            
            log "=== Completed processing step: $step_name ==="
        done
        
        # Sleep if no new steps found
        if [ ${#new_steps_found[@]} -eq 0 ]; then
            sleep $MONITOR_INTERVAL
        fi
    done
}

# Run main function with all arguments
main "$@"