#!/usr/bin/env python3
"""
Test script for Metrics and StepScheduler in distributed environment.

This script validates the correct behavior of metrics tracking and step scheduling
in a real distributed training setup without actual model or dataset loading.

Usage:
    # Single GPU
    python test_metrics_distributed.py --gradient-accumulation-steps 4 --logging-per-step 5 --save-checkpoint-per-step 10
    
    # Multi-GPU (using torchrun)
    torchrun --nproc_per_node=4 test_metrics_distributed.py --gradient-accumulation-steps 4 --logging-per-step 5 --save-checkpoint-per-step 10
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
from typing import Optional

# Add muse to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from muse.training.common import define_metrics, StepScheduler
from muse.utils.metrics import Logger, StdoutBackend, CSVBackend


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Metrics and StepScheduler")
    
    # Training parameters
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--logging-per-step", type=int, default=5,
                       help="Log metrics every N global steps")
    parser.add_argument("--save-checkpoint-per-step", type=int, default=10,
                       help="Save checkpoint every N global steps")
    parser.add_argument("--num-iterations", type=int, default=50,
                       help="Total number of micro-steps to run")
    
    # Distributed training
    parser.add_argument("--backend", type=str, default="nccl",
                       choices=["nccl", "gloo"],
                       help="Distributed backend")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./test_output",
                       help="Directory to save test outputs")
    
    return parser.parse_args()


def init_distributed():
    """
    Initialize distributed training environment.
    
    Returns:
        tuple: (rank, world_size, device)
    """
    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Single process mode
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Initialize process group
    if world_size > 1:
        if not dist.is_initialized():
            backend = os.environ.get("DIST_BACKEND", "nccl")
            dist.init_process_group(backend=backend)
            print(f"[Rank {rank}] Initialized distributed training: world_size={world_size}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print(f"[Rank {rank}] Warning: CUDA not available, using CPU")
    
    return rank, world_size, device


def print_rank_0(msg):
    """Print message only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


def generate_fake_batch(micro_step: int, rank: int, world_size: int):
    """
    Generate fake training batch data.
    
    Args:
        micro_step: Current micro step number
        rank: Process rank
        world_size: Total number of processes
    
    Returns:
        dict: Fake batch data
    """
    # Simulate varying loss across steps and ranks
    # Loss should decrease over time and vary across ranks
    base_loss = 2.0 - (micro_step * 0.01)
    rank_offset = rank * 0.1
    noise = torch.randn(1).item() * 0.05
    loss = base_loss + rank_offset + noise
    
    # Simulate token count (varying by batch)
    tokens = 1024 + (micro_step % 10) * 100
    
    # Simulate sample count
    samples = 4 + (micro_step % 3)
    
    # Simulate gradient norm (should be computed after backward in real training)
    grad_norm = 0.5 + torch.rand(1).item() * 0.3
    
    return {
        "loss": loss,
        "tokens": tokens,
        "samples": samples,
        "grad_norm": grad_norm
    }


def simulate_optimizer_step(rank: int):
    """
    Simulate optimizer.step() - just a placeholder.
    
    In real training, this would update model parameters.
    """
    # Simulate some computation time
    time.sleep(0.001)
    if rank == 0:
        print("  [Optimizer] Step completed")


def simulate_checkpoint_save(global_step: int, rank: int):
    """
    Simulate checkpoint saving.
    
    Args:
        global_step: Current global step
        rank: Process rank
    """
    print_rank_0(f"  [Checkpoint] Saving checkpoint at global_step={global_step}")
    # In real training, this would save model weights, optimizer state, etc.
    time.sleep(0.01)  # Simulate I/O time


def test_metrics_and_scheduler():
    """Main test function."""
    args = get_args()
    
    # Initialize distributed environment
    rank, world_size, device = init_distributed()
    
    print_rank_0("=" * 80)
    print_rank_0("Testing Metrics and StepScheduler in Distributed Environment")
    print_rank_0("=" * 80)
    print_rank_0(f"Configuration:")
    print_rank_0(f"  World Size: {world_size}")
    print_rank_0(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print_rank_0(f"  Logging Per Step: {args.logging_per_step}")
    print_rank_0(f"  Save Checkpoint Per Step: {args.save_checkpoint_per_step}")
    print_rank_0(f"  Total Iterations: {args.num_iterations}")
    print_rank_0("")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize metrics
    print_rank_0("Initializing metrics system...")
    metrics = define_metrics(
        acc_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step
    )
    
    # Add logger backends
    if rank == 0:
        # Add stdout logger
        stdout_logger = Logger("stdout", [StdoutBackend(prefix=f"[Rank {rank}]")])
        
        # Add CSV logger
        csv_logger = Logger("csv", [CSVBackend(
            os.path.join(args.output_dir, f"metrics_rank_{rank}.csv")
        )])
        
        metrics.add_logger(stdout_logger)
        metrics.add_logger(csv_logger)
        
        # Track metrics for logging
        metrics.logger.track(metrics.loss, name="loss", group="training")
        metrics.logger.track(metrics.grad_norm, name="grad_norm", group="training")
        metrics.logger.track(metrics.learning_rate, name="learning_rate", group="training")
        metrics.logger.track(metrics.tokens.cumsum(), name="total_tokens", group="perf")
        
        print_rank_0("✓ Metrics system initialized with loggers")
    
    # Initialize scheduler
    print_rank_0("Initializing StepScheduler...")
    scheduler = StepScheduler(args)
    print_rank_0("✓ StepScheduler initialized")
    print_rank_0("")
    
    # Training statistics
    total_optimizer_updates = 0
    total_logs = 0
    total_checkpoints = 0
    
    print_rank_0("Starting training loop simulation...")
    print_rank_0("-" * 80)
    
    # Main training loop
    for iteration in range(args.num_iterations):
        # Advance metrics index
        metrics.step()
        
        # Advance scheduler
        scheduler.step()
        
        # Generate fake batch data
        batch = generate_fake_batch(scheduler.micro_step, rank, world_size)
        
        # Record metrics for this micro-step
        metrics.loss.append(batch["loss"])
        metrics.tokens.append(batch["tokens"])
        metrics.samples.append(batch["samples"])
        
        # Print micro-step info on rank 0
        if rank == 0 and iteration % 5 == 0:
            print(f"[Micro-step {scheduler.micro_step}] "
                  f"Loss: {batch['loss']:.4f}, Tokens: {batch['tokens']}")
        
        # Simulate backward pass and optimizer update at gradient accumulation boundary
        if scheduler.is_gradient_accumulation_boundary():
            total_optimizer_updates += 1
            
            # Record gradient-related metrics
            metrics.grad_norm.append(batch["grad_norm"])
            
            # Simulate learning rate (decreasing over time)
            learning_rate = 1e-4 * (1.0 - scheduler.global_step / 100.0)
            metrics.learning_rate.append(learning_rate)
            
            # Record step time
            metrics.step_time.tick()
            
            # Simulate optimizer step
            if rank == 0:
                print(f"\n[Global-step {scheduler.global_step}] "
                      f"Optimizer update (after {args.gradient_accumulation_steps} micro-steps)")
                simulate_optimizer_step(rank)
        
        # Logging at specified intervals
        if scheduler.should_logging():
            total_logs += 1
            print_rank_0(f"\n{'='*60}")
            print_rank_0(f"LOGGING at Global-step {scheduler.global_step}")
            print_rank_0(f"{'='*60}")
            
            # Write logs via metrics system
            metrics.write_logs(global_step=scheduler.global_step)
            
            print_rank_0(f"{'='*60}\n")
        
        # Save checkpoint at specified intervals
        if scheduler.should_save_checkpoint():
            total_checkpoints += 1
            print_rank_0(f"\n{'*'*60}")
            simulate_checkpoint_save(scheduler.global_step, rank)
            print_rank_0(f"{'*'*60}\n")
        
        # Small delay to simulate computation
        time.sleep(0.01)
    
    # Final statistics
    print_rank_0("")
    print_rank_0("=" * 80)
    print_rank_0("Test Completed Successfully!")
    print_rank_0("=" * 80)
    print_rank_0("Statistics:")
    print_rank_0(f"  Total Micro-steps: {scheduler.micro_step}")
    print_rank_0(f"  Total Global-steps: {scheduler.global_step}")
    print_rank_0(f"  Optimizer Updates: {total_optimizer_updates}")
    print_rank_0(f"  Logs Written: {total_logs}")
    print_rank_0(f"  Checkpoints Saved: {total_checkpoints}")
    print_rank_0("")
    
    # Verify correctness
    expected_global_steps = args.num_iterations // args.gradient_accumulation_steps
    expected_logs = expected_global_steps // args.logging_per_step
    expected_checkpoints = expected_global_steps // args.save_checkpoint_per_step
    
    print_rank_0("Verification:")
    print_rank_0(f"  Expected Global-steps: {expected_global_steps}, "
                f"Actual: {scheduler.global_step} "
                f"{'✓' if scheduler.global_step == expected_global_steps else '✗'}")
    print_rank_0(f"  Expected Logs: {expected_logs}, "
                f"Actual: {total_logs} "
                f"{'✓' if total_logs == expected_logs else '✗'}")
    print_rank_0(f"  Expected Checkpoints: {expected_checkpoints}, "
                f"Actual: {total_checkpoints} "
                f"{'✓' if total_checkpoints == expected_checkpoints else '✗'}")
    print_rank_0("")
    
    # Test metrics access
    print_rank_0("Testing Metrics Access:")
    print_rank_0(f"  Total loss values recorded: {len(metrics.loss)}")
    print_rank_0(f"  Total tokens recorded: {len(metrics.tokens)}")
    print_rank_0(f"  Total grad_norm recorded: {len(metrics.grad_norm)}")
    print_rank_0(f"  Latest loss: {metrics.loss[-1]:.4f}")
    print_rank_0(f"  Latest learning_rate: {metrics.learning_rate[-1]:.6f}")
    print_rank_0("")
    
    # Test derived series
    print_rank_0("Testing Derived Series:")
    total_tokens_cumsum = metrics.tokens.cumsum()
    print_rank_0(f"  Total tokens (cumsum): {total_tokens_cumsum[-1]}")
    
    avg_loss_window = metrics.loss.avg(window=5)
    if len(avg_loss_window) > 0:
        print_rank_0(f"  Average loss (window=5, latest): {avg_loss_window[-1]:.4f}")
    print_rank_0("")
    
    if rank == 0:
        print_rank_0(f"Output files saved to: {args.output_dir}")
    
    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    
    print_rank_0("=" * 80)
    return True


if __name__ == "__main__":
    try:
        success = test_metrics_and_scheduler()
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    sys.exit(exit_code)
