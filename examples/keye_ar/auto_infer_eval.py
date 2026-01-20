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
                       help='Path to inference script (由外部wrapper通过 INFERENCE_SCRIPT env + --inference-script 传入)')
    parser.add_argument('--eval-id', default="default",
                       help='Evaluation ID (default: default)')
    parser.add_argument('--good-steps', default='',
                       help='List of good steps to monitor, split by comma')
    parser.add_argument('--benchnames', default='GenEval',
                       help='Comma separated benchmark names to run (default: GenEval,DPGBench)')
    parser.add_argument('--ulmeval-dir', default="/llm_reco/lingzhixin/dit_eval_lzx/ULMEvalKit",
                       help='ULMEvalKit directory path')
    parser.add_argument('--ulmeval-config', default="config/blip3o_sft_step800.json",
                       help='ULMEvalKit config json path (single value, legacy)')
    parser.add_argument('--ulmeval-configs', default='',
                       help='Comma separated ULMEvalKit configs aligned with --benchnames. '
                            'Example: GenEval,DPGBench with configs: config/blip3o_sft_step800.json,config/dpg_blip3o_sft.json. '
                            'If empty, fallback to --ulmeval-config for all benches.')
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


def run_inference(step_name: str, args, benchname: str) -> bool:
    """Run inference for a step for a specific benchmark."""
    log(f"Starting inference for {step_name}, bench={benchname}")

    env_vars = ENV_SETTINGS.copy()
    env_vars.update({
        "DCP_CKPT_DIR": DCP_CKPT_DIR,
        "DCP_TAG": step_name,
        # 注意：inference 脚本内部默认会用 GenEval 组织 OUTPUT_DIR，
        # 我们通过 env 显式传入 OUTPUT_DIR 覆盖它。
        "OUTPUT_DIR": f"{DCP_CKPT_DIR}/{step_name}/inference/{benchname}/outputs",
        "EVAL_ID": args.eval_id,
        # twobenches 脚本会把 BENCHNAME 透传给 recipes/sana/inference_ar2image.py --benchname
        # 旧脚本即便不使用该 env，也不会受影响（兼容性）
        "BENCHNAME": benchname,
    })

    # Create output directory
    output_dir = Path(env_vars["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference script using the provided path
    log(f"Using inference script: {INFERENCE_SCRIPT}")
    cmd = ["bash", INFERENCE_SCRIPT]
    return run_command(cmd, env_vars)


def _parse_bench_to_config(args) -> Dict[str, str]:
    """Parse benchmark -> ULMEval config mapping.

    兼容性要求：
    - 如果未传 --ulmeval-configs，则所有 bench 都使用 --ulmeval-config（旧行为）
    - 如果传了 --ulmeval-configs：
        * 支持两种格式：
            1) 按 benchnames 顺序对齐的列表：cfg1,cfg2,...
            2) 显式键值对：GenEval=config/xxx.json,DPGBench=config/yyy.json
    """

    benchnames = [b.strip() for b in args.benchnames.split(",") if b.strip()]
    raw = (args.ulmeval_configs or "").strip()

    bench_to_cfg: Dict[str, str] = {}

    if not raw:
        for b in benchnames:
            bench_to_cfg[b] = args.ulmeval_config
        return bench_to_cfg

    # key=value 格式
    if "=" in raw:
        items = [x.strip() for x in raw.split(",") if x.strip()]
        for it in items:
            if "=" not in it:
                continue
            k, v = it.split("=", 1)
            bench_to_cfg[k.strip()] = v.strip()
        # fallback to legacy
        for b in benchnames:
            bench_to_cfg.setdefault(b, args.ulmeval_config)
        return bench_to_cfg

    # 列表对齐格式
    cfgs = [c.strip() for c in raw.split(",") if c.strip()]
    if len(cfgs) == 1:
        for b in benchnames:
            bench_to_cfg[b] = cfgs[0]
        return bench_to_cfg

    if len(cfgs) != len(benchnames):
        # 不抛异常，回退旧行为，保证兼容性
        for b in benchnames:
            bench_to_cfg[b] = args.ulmeval_config
        return bench_to_cfg

    for b, c in zip(benchnames, cfgs):
        bench_to_cfg[b] = c
    return bench_to_cfg


def run_evaluation(step_name: str, args, benchname: str) -> bool:
    """Run evaluation for a step for a specific benchmark."""
    log(f"Starting evaluation for {step_name}, bench={benchname}")

    work_dir = f"{DCP_CKPT_DIR}/{step_name}/inference/{benchname}/outputs/ulmeval/aggresults/"
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    bench_to_cfg = _parse_bench_to_config(args)
    ulm_cfg = bench_to_cfg.get(benchname, args.ulmeval_config)

    eval_cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "run_eval_only.py",
        "--config",
        ulm_cfg,
        "--eval-id",
        args.eval_id,
        "--work-dir",
        work_dir,
    ]

    ulm_dir = args.ulmeval_dir
    if not os.path.exists(ulm_dir):
        log(f"ULMEvalKit directory not found: {ulm_dir}")
        return False

    activation_cmd = [
        "bash",
        "-c",
        f"source /mmu_mllm_hdd_2/chuchenglong/miniconda3/bin/activate && "
        f"conda activate ulmevalkit2 && "
        f"cd {ulm_dir} && "
        f"max_infer_items=300000 PYTHONPATH=. {' '.join(eval_cmd)}",
    ]
    print(f"Activation command: {' '.join(activation_cmd)}")
    return run_command(activation_cmd)


def collect_scores(step_name: str, benchname: str) -> bool:
    """Collect evaluation scores for a benchmark.

    注意：exp163_monitor_twobenches.sh 调用的 inference 脚本是 mpi_infer_custom_cond_spe.sh，
    它只负责生成 outputs 目录下的图片与 messages json；
    聚合/打分依然复用 `recipes/sana/inference_ar2image.py --mode visualize`。

    同时为了对齐 mpi 脚本的目录组织，这里会显式传入 output-dir。
    """

    log(f"Collecting scores for {step_name}, bench={benchname}")

    step_number = step_name.replace("global_step", "")
    cmd = [
        "python",
        "recipes/sana/inference_ar2image.py",
        "--mode",
        "visualize",
        "--dcp-ckpt-dir",
        DCP_CKPT_DIR,
        "--dcp-tag",
        step_name,
        "--model-tag",
        MODEL_TAG,
        "--tb-log-name",
        f"{TB_LOG_NAME}_{step_number}",
        "--benchname",
        benchname,
    ]

    return run_command(cmd)


def find_available_steps(args: argparse.Namespace) -> List[str]:
    """Find and sort all available global_step directories"""
    steps = []
    good_steps = args.good_steps.split(',')
    good_steps = [int(step) for step in good_steps if step != '']
    
    for item in Path(DCP_CKPT_DIR).iterdir():
        if item.is_dir() and item.name.startswith("global_step"):
            # Extract step number
            match = re.match(r"global_step(\d+)", item.name)
            if match:
                step_number = int(match.group(1))
                if good_steps and step_number not in good_steps:
                    continue
                metadata_file = item / ".metadata"
                if metadata_file.exists():
                    steps.append((step_number, item.name))
    
    # Sort by step number descending
    steps.sort(key=lambda x: x[0], reverse=True)
    return [step_name for _, step_name in steps]


def monitor(args):
    """Main monitoring loop"""
    log(f"Starting monitoring for {DCP_CKPT_DIR}")
    processed_steps: Set[str] = set()

    benchnames = [b.strip() for b in args.benchnames.split(",") if b.strip()]

    while True:
        log("Checking for new global_step directories...")

        available_steps = find_available_steps(args)

        new_steps = [step for step in available_steps if step not in processed_steps]

        for step_name in new_steps:
            log(f"Found new step: {step_name}")

            all_ok = True
            for benchname in benchnames:
                ok = run_inference(step_name, args, benchname)
                if not ok:
                    all_ok = False
                    log(f"Inference failed for {step_name}, bench={benchname}")
                    break

                ok = run_evaluation(step_name, args, benchname)
                if not ok:
                    all_ok = False
                    log(f"Evaluation failed for {step_name}, bench={benchname}")
                    break

                ok = collect_scores(step_name, benchname)
                if not ok:
                    all_ok = False
                    log(f"Collect scores failed for {step_name}, bench={benchname}")
                    break

            if all_ok:
                processed_steps.add(step_name)
            else:
                log(f"Failed to process {step_name}, skipping further steps")

            # 每轮只处理一个最新 step（保持原行为）
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
        monitor(args=args)
    except KeyboardInterrupt:
        log("Monitoring interrupted by user")
    except Exception as e:
        log(f"Monitoring failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
