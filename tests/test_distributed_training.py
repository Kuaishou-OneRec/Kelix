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


def run_training(args, rank, world_size):
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
        
        if step < 3 and rank == 0:
            print(f"[Rank {rank}, Step {step}] Generated batch: loss={batch['loss']:.4f}, tokens={batch['tokens']}")
        
        # 3. Simulate forward pass - append loss and token metrics
        #    Each rank appends its LOCAL value
        metrics.loss.append(batch['loss'])
        metrics.tokens.append(batch['tokens'])
        metrics.samples.append(batch['samples'])
        
        # 4. Simulate backward pass and optimizer step at gradient accumulation boundary
        if scheduler.is_gradient_accumulation_boundary():
            metrics.grad_norm.append(batch['grad_norm'])
            # Simulate learning rate decay
            lr = 0.0001 * (1 - scheduler.global_step / (args.num_training_steps / args.gradient_accumulation_steps))
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
        
        # Show output locations
        print("CSV output saved to: /tmp/metrics_test.csv")
        print()


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
    run_training(args, rank, world_size)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"[Rank {rank}] Cleanup complete")


if __name__ == "__main__":
    main()
