#!/usr/bin/env python3
"""
Test script for distributed training with fake data.

This script tests the Metrics and StepScheduler in a distributed training setting
without requiring actual models or datasets. It generates fake loss, token counts,
and other metrics to verify the distributed reduction and logging behavior.

Correct calling order:
1. scheduler.step() - advance micro_step and global_step
2. Forward/Backward pass - compute loss and gradients
3. metrics.loss.append(local_value) - append local values
4. metrics.step() - perform distributed reduction and sync
"""

import argparse
import random
import time
import os
import torch
import torch.distributed as dist

# Import Muse modules
from muse.training.common import define_metrics, StepScheduler
from muse.utils.metrics import Logger, StdoutBackend, CSVBackend


def init_distributed():
    """
    Initialize distributed training if multi-process.
    
    Returns:
        tuple: (rank, world_size, local_rank)
    """
    if 'RANK' in os.environ:
        # Distributed mode (launched with torchrun)
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        print(f"[Rank {rank}/{world_size}] Initialized with NCCL backend")
    else:
        # Single process mode (for testing without torchrun)
        rank = 0
        world_size = 1
        local_rank = 0
        
        print(f"[Single Process Mode] Running without distributed training")
    
    return rank, world_size, local_rank


def generate_fake_batch(rank: int, step: int):
    """
    Generate fake batch data with rank-specific values.
    This simulates different loss values on different ranks to test distributed reduction.
    
    Args:
        rank: Process rank
        step: Current training step
        
    Returns:
        dict: Fake batch data with loss, tokens, samples, grad_norm
    """
    # Each rank gets slightly different values to test distributed reduction
    base_loss = 2.5 + rank * 0.1 + random.uniform(-0.2, 0.2)
    tokens = random.randint(1000, 2000)
    samples = random.randint(1, 4)
    grad_norm = random.uniform(0.5, 2.0)
    
    return {
        'loss': base_loss,
        'tokens': tokens,
        'samples': samples,
        'grad_norm': grad_norm
    }


def calculate_ground_truth(all_ranks_data, acc_steps, logging_per_step, world_size):
    """
    Calculate ground truth with REAL distributed reduction.
    
    Steps:
    1. Manually compute reduction from all ranks' local data:
       - loss: MEAN across ranks
       - tokens: SUM across ranks
       - samples: SUM across ranks
       - grad_norm: MEAN across ranks
    2. Use reduced values with define_metrics() logic
    3. Return expected logged values
    
    Args:
        all_ranks_data: List of raw_data dicts from all ranks
        acc_steps: Gradient accumulation steps
        logging_per_step: Logging interval in global steps
        world_size: Number of processes
    
    Returns:
        dict: {global_step: {metric_name: expected_value}}
    """
    import numpy as np
    
    num_steps = len(all_ranks_data[0]['loss'])
    
    print(f"  Received data from {len(all_ranks_data)} ranks, {num_steps} steps each")
    
    # Step 1: Manually compute distributed reduction
    loss_reduced = []
    tokens_reduced = []
    samples_reduced = []
    
    for step_idx in range(num_steps):
        # Gather this step's values from all ranks
        loss_vals = [rank_data['loss'][step_idx] for rank_data in all_ranks_data]
        tokens_vals = [rank_data['tokens'][step_idx] for rank_data in all_ranks_data]
        samples_vals = [rank_data['samples'][step_idx] for rank_data in all_ranks_data]
        
        # Apply reduction: MEAN for loss, SUM for tokens/samples
        loss_reduced.append(np.mean(loss_vals))
        tokens_reduced.append(np.sum(tokens_vals))
        samples_reduced.append(np.sum(samples_vals))
    
    # grad_norm reduction (at accumulation boundaries only)
    grad_norm_reduced = []
    if len(all_ranks_data[0]['grad_norm']) > 0:
        for idx in range(len(all_ranks_data[0]['grad_norm'])):
            grad_vals = [rank_data['grad_norm'][idx] for rank_data in all_ranks_data]
            grad_norm_reduced.append(np.mean(grad_vals))
    
    # Use rank 0's step_time and learning_rate (no reduction needed)
    step_time = np.array(all_ranks_data[0]['step_time'])
    learning_rate = all_ranks_data[0]['learning_rate']
    
    # Convert to numpy arrays
    loss = np.array(loss_reduced)
    tokens = np.array(tokens_reduced)
    samples = np.array(samples_reduced)
    
    print(f"  After manual reduction:")
    print(f"    Rank 0 local loss[0]={all_ranks_data[0]['loss'][0]:.4f}, reduced loss[0]={loss[0]:.4f}")
    print(f"    Rank 0 local tokens[0]={all_ranks_data[0]['tokens'][0]}, reduced tokens[0]={tokens[0]}")
    if world_size > 1:
        print(f"    Rank 1 local tokens[0]={all_ranks_data[1]['tokens'][0]}, verifying sum")
    
    # Step 2: Apply define_metrics() logic with reduced values
    
    # Cumulative sums
    total_tokens = np.cumsum(tokens)
    total_samples = np.cumsum(samples)
    
    # Helper functions
    def window_avg(arr, window):
        """Sliding window average matching Series.avg() behavior."""
        if window is None:
            return np.array([np.mean(arr[:i+1]) for i in range(len(arr))])
        result = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result.append(np.mean(arr[start:i+1]))
        return np.array(result)
    
    def diff(arr):
        """Calculate difference, first element is None."""
        result = [None]
        for i in range(1, len(arr)):
            result.append(arr[i] - arr[i-1])
        return np.array(result)
    
    # Calculate metrics following define_metrics() logic:
    
    # training/loss = loss.avg(window=acc_steps)[::acc_steps][1:].avg(window=logging_per_step)[::logging_per_step]
    loss_avg = window_avg(loss, acc_steps)
    loss_at_global = loss_avg[::acc_steps][1:]  # Skip sentinel
    loss_logged_avg = window_avg(loss_at_global, logging_per_step)
    logged_loss = loss_logged_avg[::logging_per_step]
    
    print(f"  Processed loss: loss_avg len={len(loss_avg)}, at_global len={len(loss_at_global)}, logged len={len(logged_loss)}")
    
    # perf/total_tokens = total_tokens[::acc_steps][::logging_per_step]
    logged_total_tokens = total_tokens[::acc_steps][::logging_per_step]
    
    # perf/total_samples = total_samples[::acc_steps][::logging_per_step]
    logged_total_samples = total_samples[::acc_steps][::logging_per_step]
    
    # perf/tokens_per_sec_per_gpu = (total_tokens.diff() / step_time.diff())[::acc_steps][1:].avg(window=logging_per_step)[::logging_per_step] / world_size
    tokens_diff = diff(total_tokens)
    time_diff = diff(step_time)
    
    # Element-wise division (handle None values)
    tokens_per_sec = []
    for td, tmd in zip(tokens_diff, time_diff):
        if td is None or tmd is None or tmd == 0:
            tokens_per_sec.append(None)
        else:
            tokens_per_sec.append(td / tmd)
    tokens_per_sec = np.array(tokens_per_sec, dtype=object)
    
    # Sample at global steps, skip first (sentinel)
    tokens_per_sec_global = tokens_per_sec[::acc_steps][1:]
    
    # Average over logging window (skip None values)
    valid_indices = [i for i, v in enumerate(tokens_per_sec_global) if v is not None]
    if len(valid_indices) > 0:
        tokens_per_sec_valid = np.array([tokens_per_sec_global[i] for i in valid_indices], dtype=float)
        tokens_per_sec_avg = window_avg(tokens_per_sec_valid, logging_per_step)
        logged_tokens_per_sec_gpu = tokens_per_sec_avg[::logging_per_step] / world_size
    else:
        logged_tokens_per_sec_gpu = np.array([])
    
    # perf/samples_per_sec_per_gpu (same logic)
    samples_diff = diff(total_samples)
    samples_per_sec = []
    for sd, tmd in zip(samples_diff, time_diff):
        if sd is None or tmd is None or tmd == 0:
            samples_per_sec.append(None)
        else:
            samples_per_sec.append(sd / tmd)
    samples_per_sec = np.array(samples_per_sec, dtype=object)
    
    samples_per_sec_global = samples_per_sec[::acc_steps][1:]
    valid_indices_s = [i for i, v in enumerate(samples_per_sec_global) if v is not None]
    if len(valid_indices_s) > 0:
        samples_per_sec_valid = np.array([samples_per_sec_global[i] for i in valid_indices_s], dtype=float)
        samples_per_sec_avg = window_avg(samples_per_sec_valid, logging_per_step)
        logged_samples_per_sec_gpu = samples_per_sec_avg[::logging_per_step] / world_size
    else:
        logged_samples_per_sec_gpu = np.array([])
    
    # perf/tokens_per_day = tokens_per_sec_per_gpu * 86400 * world_size
    logged_tokens_per_day = logged_tokens_per_sec_gpu * 86400 * world_size
    
    # Build ground truth: step -> metrics
    # Steps are logged at: global_step % logging_per_step == 0
    num_global_steps = len(loss) // acc_steps
    logged_steps = [gs for gs in range(1, num_global_steps + 1) if gs % logging_per_step == 0]
    
    print(f"  Expected logged steps: {logged_steps[:5]}... (total {len(logged_steps)})")
    
    ground_truth = {}
    for i, step in enumerate(logged_steps):
        if i < len(logged_loss):
            gt = {
                'training/loss': float(logged_loss[i]),
                'perf/total_tokens': float(logged_total_tokens[i]),
                'perf/total_samples': float(logged_total_samples[i]),
            }
            if i < len(logged_tokens_per_sec_gpu):
                gt['perf/tokens_per_sec_per_gpu'] = float(logged_tokens_per_sec_gpu[i])
            if i < len(logged_samples_per_sec_gpu):
                gt['perf/samples_per_sec_per_gpu'] = float(logged_samples_per_sec_gpu[i])
            if i < len(logged_tokens_per_day):
                gt['perf/tokens_per_day'] = float(logged_tokens_per_day[i])
            
            ground_truth[step] = gt
    
    print(f"  Ground truth computed for {len(ground_truth)} steps")
    return ground_truth


def validate_csv_against_ground_truth(csv_path, ground_truth, rank, tolerance=0.01):
    """
    Validate CSV matches ground truth within tolerance.
    
    Args:
        csv_path: CSV file path
        ground_truth: Expected values
        rank: Process rank
        tolerance: Relative tolerance (1% = 0.01)
    
    Returns:
        bool: True if all values match
    """
    if rank != 0:
        return True
    
    import csv
    
    print("\n" + "=" * 70)
    print("GROUND TRUTH VALIDATION")
    print("=" * 70)
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    try:
        with open(csv_path, 'r') as f:
            csv_data = {int(row['step']): row for row in csv.DictReader(f)}
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    print(f"CSV file: {csv_path}")
    print(f"CSV rows: {len(csv_data)}, Ground truth: {len(ground_truth)}")
    print()
    
    all_pass = True
    
    for step in sorted(ground_truth.keys()):
        if step not in csv_data:
            print(f"❌ Step {step}: Not in CSV")
            all_pass = False
            continue
        
        print(f"Step {step}:")
        csv_row = csv_data[step]
        gt = ground_truth[step]
        
        for metric, expected in gt.items():
            try:
                actual = float(csv_row[metric])
                
                # Calculate relative error
                if abs(expected) > 1e-9:
                    rel_err = abs(actual - expected) / abs(expected)
                else:
                    rel_err = abs(actual - expected)
                
                if rel_err <= tolerance:
                    print(f"  ✓ {metric}: {actual:.6f} ≈ {expected:.6f} (err: {rel_err*100:.2f}%)")
                else:
                    print(f"  ❌ {metric}: {actual:.6f} ≠ {expected:.6f} (err: {rel_err*100:.2f}%)")
                    all_pass = False
            except (KeyError, ValueError) as e:
                print(f"  ❌ {metric}: Error - {e}")
                all_pass = False
        print()
    
    print("=" * 70)
    if all_pass:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")
    print("=" * 70)
    
    return all_pass


def run_training(args, rank, world_size) -> bool:
    """
    Run the training loop with fake data.
    
    This tests the correct integration of Metrics and StepScheduler:
    - Metrics tracks all values and performs distributed reduction
    - StepScheduler manages micro_step, global_step, and determines when to log/checkpoint
    
    Args:
        args: Command line arguments
        rank: Process rank
        world_size: Total number of processes
    """
    
    loggers = []
    # Setup logger (only on rank 0 to avoid duplicate output)
    if rank == 0:
        # Add stdout logger for immediate feedback
        stdout_logger = Logger("stdout", [StdoutBackend()])
        
        # Add CSV logger to save metrics to file
        csv_logger = Logger("csv", [CSVBackend("/tmp/metrics_test.csv")])
        loggers.append(stdout_logger)
        loggers.append(csv_logger)

    # Initialize metrics (using define_metrics from common.py)
    metrics = define_metrics(
        acc_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step,
        loggers=loggers
    )
        
    
    # Initialize step scheduler
    scheduler = StepScheduler(args)
    
    # Record REDUCED data for ground truth calculation
    # These are recorded AFTER metrics.step(), so they already include
    # distributed reduction (mean for loss/grad_norm, sum for tokens/samples)
    raw_data = {
        'loss': [],           # Per micro-step, after reduce mean
        'tokens': [],         # Per micro-step, after reduce sum
        'samples': [],        # Per micro-step, after reduce sum
        'step_time': [],      # Per micro-step timestamps
        'grad_norm': [],      # Per global-step, after reduce mean
        'learning_rate': []   # Per global-step (no reduction, only on rank 0)
    }
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Number of training steps: {args.num_training_steps}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Logging per step: {args.logging_per_step}")
        print(f"Save checkpoint per step: {args.save_checkpoint_per_step}")
        print(f"World size: {world_size}")
        print("=" * 60)
        print()
    
    print(f"[Rank {rank}] Starting training loop...")
    
    for step in range(args.num_training_steps):
        # 1. Advance scheduler (manages micro_step and global_step)
        scheduler.step()
        
        # 2. Generate fake batch data (simulates data loading)
        batch = generate_fake_batch(rank, step)
        
        # Record LOCAL batch values (ALL ranks record their own data)
        raw_data['loss'].append(batch['loss'])
        raw_data['tokens'].append(batch['tokens'])
        raw_data['samples'].append(batch['samples'])
        raw_data['step_time'].append(time.time())
        
        if step < 3 and rank == 0:
            print(f"[Rank {rank}, Step {step}] Generated batch: loss={batch['loss']:.4f}, tokens={batch['tokens']}")
        
        # 3. Simulate forward pass - append loss and token metrics
        #    Each rank appends its LOCAL value
        metrics.loss.append(batch['loss'])
        metrics.tokens.append(batch['tokens'])
        metrics.samples.append(batch['samples'])
        
        # 4. Simulate backward pass and optimizer step at gradient accumulation boundary
        if scheduler.is_gradient_accumulation_boundary():
            # Record local values
            raw_data['grad_norm'].append(batch['grad_norm'])
            lr = 0.0001 * (1 - scheduler.global_step / (args.num_training_steps / args.gradient_accumulation_steps))
            raw_data['learning_rate'].append(lr)
            
            # Append to metrics
            metrics.grad_norm.append(batch['grad_norm'])
            # Simulate learning rate decay
            metrics.learning_rate.append(lr)
        
        # 5. Record step time (tick marks current time)
        metrics.step_time.tick()
        
        # 6. Call metrics.step() to perform distributed reduction and sync
        #    This is where the magic happens - reduces loss/tokens across ranks
        metrics.step()
        
        # Small delay to simulate computation
        time.sleep(0.005)
        
        # 7. Logging at specified intervals
        if scheduler.should_logging():
            if rank == 0:
                print(f"\n{'=' * 60}")
                print(f"[Global Step {scheduler.global_step}] LOGGING METRICS")
                print(f"{'=' * 60}")
                
                # Debug: print metrics summary
                metrics.print_summary(last_n=3, rank=rank, show_derived=False)
            
            # Write to logger backends
            metrics.write_logs(scheduler.global_step)
        
        # 8. Checkpoint saving (just print, no actual save)
        if scheduler.should_save_checkpoint():
            if rank == 0:
                print(f"\n[Global Step {scheduler.global_step}] Would save checkpoint here")
    
    # Gather all ranks' local data to rank 0 for validation
    if world_size > 1:
        all_ranks_data = [None] * world_size
        dist.all_gather_object(all_ranks_data, raw_data)
    else:
        # Single GPU: just use local data
        all_ranks_data = [raw_data]
    
    validation_passed = True
    if rank == 0:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total micro steps: {scheduler.micro_step}")
        print(f"Total global steps: {scheduler.global_step}")
        print(f"Metrics index length: {len(metrics._index)}")
        print(f"Series tracked: {list(metrics._series.keys())}")
        print("=" * 60)
        
        # Final summary with more values
        metrics.print_summary(last_n=5, rank=rank, show_derived=False)
        
        # Calculate ground truth with manual distributed reduction
        print("\nCalculating ground truth with manual distributed reduction...")
        ground_truth = calculate_ground_truth(
            all_ranks_data,
            acc_steps=args.gradient_accumulation_steps,
            logging_per_step=args.logging_per_step,
            world_size=world_size
        )
        
        # Validate CSV against ground truth
        validation_passed = validate_csv_against_ground_truth(
            "/tmp/metrics_test.csv",
            ground_truth,
            rank,
            tolerance=0.01  # 1% tolerance
        )
    
    return validation_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test distributed training with fake data"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", 
        type=int, 
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--logging-per-step", 
        type=int, 
        default=10,
        help="Log every N global steps"
    )
    parser.add_argument(
        "--save-checkpoint-per-step", 
        type=int, 
        default=50,
        help="Save checkpoint every N global steps"
    )
    parser.add_argument(
        "--num-training-steps", 
        type=int, 
        default=100,
        help="Total number of micro steps to run"
    )
    
    args = parser.parse_args()
    
    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()
    
    # Run training loop
    validation_passed = run_training(args, rank, world_size)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0:
        if validation_passed:
            print("\n✅ TEST PASSED")
        else:
            print("\n❌ TEST FAILED")
            exit(1)


if __name__ == "__main__":
    main()
