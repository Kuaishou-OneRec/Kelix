#!/usr/bin/env python3
"""
Auto monitoring script for DCP checkpoint inference and evaluation
"""

import os
import sys
import time
import subprocess
import signal
import re
from pathlib import Path
import argparse
from typing import List, Dict, Set

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Auto monitoring script for DCP checkpoint inference and evaluation')
    parser.add_argument('--dcp-ckpt-dir', required=True,
                       help='Path to DCP checkpoint directory')
    parser.add_argument('--monitor-interval', type=int, default=30,
                       help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--model-tag', default="BLIP3OTransformersSFT",
                       help='Model tag (default: BLIP3OTransformersSFT)')
    parser.add_argument('--tb-log-name', default="auto_eval",
                       help='TensorBoard log name (default: auto_eval)')
    parser.add_argument('--dataset-config',
                       default="examples/sana/ar_dit/inference/run_ar_dit_lzx_4096_v2_1024im_multiscale_inf.json",
                       help='Dataset config path')
    parser.add_argument('--keye-ar-dir',
                       default="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted",
                       help='Keye AR directory path')
    parser.add_argument('--inference-script',
                       default="examples/sana/ar_dit/inference/mpi_infer_custom.sh",
                       help='Path to inference script (default: examples/sana/ar_dit/inference/mpi_infer_custom.sh)')
    return parser.parse_args()

# Set environment variables for subprocesses
ENV_SETTINGS = {}


def log(message: str):
    """Log message with timestamp"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')


def run_command(cmd: List[str], env: Dict = None, log_output: bool = True) -> bool:
    """Run a command and log its output"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    log(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, env=full_env, capture_output=True, text=True, check=False)
        duration = int(time.time() - start_time)
        
        if result.returncode == 0:
            log(f"Command completed successfully (took {duration}s)")
            if log_output and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    log(f"  OUTPUT: {line}")
            return True
        else:
            log(f"Command failed with code {result.returncode} (took {duration}s)")
            if result.stderr.strip():
                for line in result.stderr.strip().split('\n'):
                    log(f"  ERROR: {line}")
            return False
    except Exception as e:
        log(f"Command failed with exception: {e}")
        return False


def run_inference(step_name: str) -> bool:
    """Run inference for a step"""
    log(f"Starting inference for {step_name}")
    
    env_vars = ENV_SETTINGS.copy()
    env_vars.update({
        "DCP_CKPT_DIR": DCP_CKPT_DIR,
        "DCP_TAG": step_name,
        "OUTPUT_DIR": f"{DCP_CKPT_DIR}/{step_name}/inference/GenEval/outputs"
    })
    
    # Create output directory
    output_dir = Path(env_vars["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference script using the provided path
    log(f"Using inference script: {INFERENCE_SCRIPT}")
    cmd = ["bash", INFERENCE_SCRIPT]
    return run_command(cmd, env_vars)


def run_evaluation(step_name: str) -> bool:
    """Run evaluation for a step"""
    log(f"Starting evaluation for {step_name}")
    
    work_dir = f"{DCP_CKPT_DIR}/{step_name}/inference/GenEval/outputs/ulmeval/aggresults/"
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    
    eval_cmd = [
        "torchrun", "--nproc_per_node=8", "run_eval_only.py",
        "--config", "config/blip3o_sft_step800.json",
        "--eval-id", "default",
        "--work-dir", work_dir
    ]
    
    # Run in ULMEvalKit directory with conda environment
    ulm_dir = "/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit"
    if not os.path.exists(ulm_dir):
        log(f"ULMEvalKit directory not found: {ulm_dir}")
        return False
    
    activation_cmd = [
        "bash", "-c",
        f"source /mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate && "
        f"conda activate ulmevalkit2 && "
        f"cd {ulm_dir} && "
        f"max_infer_items=300000 PYTHONPATH=. {' '.join(eval_cmd)} > {work_dir}/eval_${cf}.out 2>&1 &"
    ]
    print(f"Activation command: {' '.join(activation_cmd)}")
    return run_command(activation_cmd)


def collect_scores(step_name: str) -> bool:
    """Collect evaluation scores"""
    log(f"Collecting scores for {step_name}")
    
    step_number = step_name.replace("global_step", "")
    cmd = [
        "python", "recipes/sana/inference_ar2image.py",
        "--mode", "visualize",
        "--dcp-ckpt-dir", DCP_CKPT_DIR,
        "--model-tag", MODEL_TAG,
        "--tb-log-name", f"{TB_LOG_NAME}_{step_number}"
    ]
    
    return run_command(cmd)


def find_available_steps() -> List[str]:
    """Find and sort all available global_step directories"""
    steps = []
    
    for item in Path(DCP_CKPT_DIR).iterdir():
        if item.is_dir() and item.name.startswith("global_step"):
            # Extract step number
            match = re.match(r"global_step(\d+)", item.name)
            if match:
                step_number = int(match.group(1))
                metadata_file = item / ".metadata"
                if metadata_file.exists():
                    steps.append((step_number, item.name))
    
    # Sort by step number descending
    steps.sort(key=lambda x: x[0], reverse=True)
    return [step_name for _, step_name in steps]


def monitor():
    """Main monitoring loop"""
    log(f"Starting monitoring for {DCP_CKPT_DIR}")
    processed_steps: Set[str] = set()
    
    while True:
        log("Checking for new global_step directories...")
        
        available_steps = find_available_steps()
        new_steps = [step for step in available_steps if step not in processed_steps]
        
        for step_name in new_steps:
            # if int(step_name.split('step')[-1]) % 4000 != 0: continue
            log(f"Found new step: {step_name}")
            processed_steps.add(step_name)
            
            if run_inference(step_name):
                if run_evaluation(step_name):
                    collect_scores(step_name)
            else:
                log(f"Failed to process {step_name}, skipping further steps")
            break
            
        if not new_steps:
            time.sleep(MONITOR_INTERVAL)


def main():
    """Main function"""
    args = parse_args()
    
    # Set global variables from arguments
    global DCP_CKPT_DIR, MONITOR_INTERVAL, MODEL_TAG, TB_LOG_NAME, ENV_SETTINGS, INFERENCE_SCRIPT
    DCP_CKPT_DIR = args.dcp_ckpt_dir
    MONITOR_INTERVAL = args.monitor_interval
    MODEL_TAG = args.model_tag
    TB_LOG_NAME = args.tb_log_name
    INFERENCE_SCRIPT = args.inference_script
    ENV_SETTINGS = {
        "DATASET_CONFIG": args.dataset_config,
        "KEYE_AR_DIR": args.keye_ar_dir
    }
    
    # Ensure directory exists
    Path(DCP_CKPT_DIR).mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = Path(DCP_CKPT_DIR) / "auto_monitor.log"
    
    print(f"Starting auto monitoring...")
    print(f"DCP checkpoint directory: {DCP_CKPT_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Monitor interval: {MONITOR_INTERVAL} seconds")
    print()
    
    # Write initial log entry
    log("Script started")
    
    try:
        monitor()
    except KeyboardInterrupt:
        log("Monitoring interrupted by user")
    except Exception as e:
        log(f"Monitoring failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
