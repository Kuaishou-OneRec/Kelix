#!/usr/bin/env bash
# Auto monitoring script for DCP checkpoint inference and evaluation
# Usage: bash examples/sana/ar_dit/inference/auto_infer_and_eval.sh DCP_CKPT_DIR
#
# This script monitors a DCP checkpoint directory for new global_step folders
# with .metadata files, and automatically runs inference -> evaluation -> score collection

set -euo pipefail

# Script configuration
MONITOR_INTERVAL=30  # Check every 30 seconds
MODEL_TAG="BLIP3OTransformersSFT"
TB_LOG_NAME="auto_eval"

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run inference
run_inference() {
    local step_name="$1"
    log "Starting inference for $step_name"
    
    # Set environment variables
    export DCP_CKPT_DIR="$DCP_CKPT_DIR"
    export DCP_TAG="$step_name"
    export OUTPUT_DIR="$DCP_CKPT_DIR/$step_name/inference/GenEval/outputs"
    export KEYE_AR_DIR="${KEYE_AR_DIR:-/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted}"
    export DATASET_CONFIG="${DATASET_CONFIG:-examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json}"
    
    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    
    # Run inference script
    local inference_log="$DCP_CKPT_DIR/auto_infer_logs/inference_${step_name}_$(date +%Y%m%d_%H%M%S).log"
    log "Running inference - logs: $inference_log"
    
    bash examples/sana/ar_dit/inference/mpi_infer_custom.sh > "$inference_log" 2>&1
    
    if [ $? -eq 0 ]; then
        log "Inference completed successfully for $step_name"
        return 0
    else
        log "Error: Inference failed for $step_name - check $inference_log"
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
    
    if [ ! -d "/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit" ]; then
        log "Error: ULMEvalKit directory not found"
        return 1
    fi
    
    cd "/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit"
    source "/mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate" >/dev/null 2>&1
    conda activate ulmevalkit2 >/dev/null 2>&1
    
    local cf="blip3o_sft_step800"
    local work_dir="$DCP_CKPT_DIR/$step_name/inference/GenEval/outputs/ulmeval/aggresults/"
    mkdir -p "$work_dir"
    local eval_log="$work_dir/eval_${cf}_$(date +%Y%m%d_%H%M%S).out"
    
    log "Running evaluation - logs: $eval_log"
    
    max_infer_items=300000 \
    PYTHONPATH=. \
    torchrun \
        --nproc_per_node=8 \
        run_eval_only.py --config "config/${cf}.json" \
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

# Background monitoring function
run_background_monitoring() {
    local dcp_dir="$1"
    
    # Create log directory within DCP directory
    local log_dir="$dcp_dir/auto_monitor_logs"
    mkdir -p "$log_dir"
    LOG_FILE="$log_dir/auto_monitor_$(date +%Y%m%d_%H%M%S).log"
    
    # Redirect all output to log file
    exec > >(tee -a "$LOG_FILE") 2>&1
    
    log "Auto monitoring started for: $dcp_dir"
    log "Monitoring interval: $MONITOR_INTERVAL seconds"
    log "Log file: $LOG_FILE"
    
    # Ensure the directory exists
    if [ ! -d "$dcp_dir" ]; then
        log "Error: DCP checkpoint directory $dcp_dir does not exist!"
        exit 1
    fi
    
    # Get list of existing global_step directories to exclude from initial monitoring
    declare -A existing_steps
    if [ -d "$dcp_dir" ]; then
        while IFS= read -r -d '' step_dir; do
            if [[ "$step_dir" =~ global_step[0-9]+ ]]; then
                step_name=$(basename "$step_dir")
                existing_steps["$step_name"]=1
            fi
        done < <(find "$dcp_dir" -maxdepth 1 -type d -name "global_step*" -print0 2>/dev/null)
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
        done < <(find "$dcp_dir" -maxdepth 1 -type d -name "global_step*" -print0 2>/dev/null)
        
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

# Main function
main() {
    # Check if DCP_CKPT_DIR is provided
    if [ $# -eq 0 ]; then
        echo "Usage: $0 DCP_CKPT_DIR"
        echo "Example: $0 /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp18_ar_dit_multiscale_324tokens_2e-5/"
        echo
        echo "The script will run in background mode and output log file path."
        exit 1
    fi
    
    DCP_CKPT_DIR="$1"
    
    # Create log directory for main process output
    mkdir -p "$DCP_CKPT_DIR/auto_infer_logs"
    
    echo "Starting auto monitoring in background mode..."
    echo "DCP checkpoint directory: $DCP_CKPT_DIR"
    
    # Start background process
    nohup bash -c "
        export DCP_CKPT_DIR='$DCP_CKPT_DIR'
        export MONITOR_INTERVAL='$MONITOR_INTERVAL'
        export MODEL_TAG='$MODEL_TAG'
        export TB_LOG_NAME='$TB_LOG_NAME'
        
        # Set PATH and PYTHONPATH for background process
        export PYTHONPATH=\${PYTHONPATH:-.}
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
        
        # Call the background monitoring function
        $(declare -f run_background_monitoring)
        $(declare -f run_inference)
        $(declare -f run_evaluation)
        $(declare -f collect_scores)
        $(declare -f log)
        
        run_background_monitoring \"$DCP_CKPT_DIR\"
    " > /dev/null 2>&1 &
    
    local pid=$!
    local log_file="$DCP_CKPT_DIR/auto_monitor_logs/auto_monitor_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Background process started with PID: $pid"
    echo "Log file will be: $log_file"
    echo "You can stop the process with: kill $pid"
    echo
    echo "Monitoring is now running in the background."
    echo "To monitor the progress, run: tail -f '$log_file'"
}

# Run main function with all arguments
main "$@"