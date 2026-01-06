#!/usr/bin/env python3
"""
Debug script for testing Dataset packing and sample_idx logic.

This script loads the video dataset and prints detailed information about
each batch's packing structure, sample_idx distribution, and related metrics.
"""

import argparse
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set environment variable to disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from muse.data.datasets import ChatCompletionVisionDataset_keye_vitrope_slowfast_video
from muse.utils.common import print_rank_0
from muse.training.parallel import initialize_model_parallel


def print_with_rank(msg, rank=None):
    """Print message with rank prefix."""
    if rank is None:
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            rank = 0
    print(f"[Rank {rank}] {msg}", flush=True)


def analyze_batch(batch, batch_idx, accumulated_samples, rank=0):
    """Analyze and print detailed information about a packed batch.
    
    Args:
        batch: The batch data
        batch_idx: Index of current batch
        accumulated_samples: Total samples accumulated so far (before this batch)
    
    Returns:
        num_samples_in_batch: Number of logical samples in this batch
    """
    def log(msg):
        print_with_rank(msg, rank)
    
    log(f"\n{'='*80}")
    log(f"=== Batch {batch_idx} ===")
    log(f"{'='*80}")
    
    # Basic shape information
    input_ids = batch.get("input_ids")
    sample_idx = batch.get("sample_idx")
    cu_seqlens = batch.get("cu_seqlens")
    loss_mask = batch.get("loss_mask")
    
    num_samples_in_batch = 0
    
    if input_ids is not None:
        log(f"input_ids shape: {input_ids.shape}")
        log(f"input_ids dtype: {input_ids.dtype}")
        log(f"Total tokens: {input_ids.numel()}")
    
    if sample_idx is not None:
        sample_idx_flat = sample_idx.flatten()
        unique_samples = torch.unique(sample_idx_flat)
        
        # Print raw sample_idx tensor (first 100 values if too long)
        log(f"\nsample_idx tensor:")
        log(f"  shape: {sample_idx.shape}")
        if sample_idx_flat.numel() <= 100:
            log(f"  values: {sample_idx_flat.tolist()}")
        else:
            log(f"  first 50: {sample_idx_flat[:50].tolist()}")
            log(f"  last 50: {sample_idx_flat[-50:].tolist()}")
        
        log(f"  unique values: {unique_samples.tolist()}")
        
        # Count samples (excluding padding with -1)
        valid_samples = unique_samples[unique_samples >= 0]
        num_valid_samples = len(valid_samples)
        log(f"  Number of valid samples (excluding padding): {num_valid_samples}")
        
        # Calculate logical samples as done in training code
        max_sample_idx = sample_idx.max().item()
        logical_samples = max_sample_idx + 1
        num_samples_in_batch = logical_samples
        log(f"  Logical samples (max_idx + 1): {logical_samples}")
        
        # Sample distribution
        log(f"\nSample distribution:")
        for sample_id in valid_samples:
            mask = (sample_idx_flat == sample_id)
            token_count = mask.sum().item()
            loss_mask_count = (loss_mask.flatten()[mask] > 0).sum().item() if loss_mask is not None else token_count
            log(f"  Sample {sample_id.item()}: {token_count} tokens (loss_mask: {loss_mask_count})")
        
        # Padding tokens
        padding_mask = (sample_idx_flat == -1)
        padding_count = padding_mask.sum().item()
        if padding_count > 0:
            log(f"  Padding (-1): {padding_count} tokens")
        
        # Verify continuity
        if len(valid_samples) > 0:
            valid_samples_sorted = sorted(valid_samples.tolist())
            expected = list(range(valid_samples_sorted[0], valid_samples_sorted[-1] + 1))
            is_continuous = valid_samples_sorted == expected
            log(f"\nSample indices continuous: {is_continuous}")
            if not is_continuous:
                log(f"  Expected: {expected}")
                log(f"  Actual: {valid_samples_sorted}")
        
        # Verify accumulated samples
        expected_accumulated = accumulated_samples + logical_samples
        log(f"\nAccumulated samples check:")
        log(f"  Previous total: {accumulated_samples}")
        log(f"  This batch: {logical_samples}")
        log(f"  New total: {expected_accumulated}")
    
    if cu_seqlens is not None:
        log(f"\ncu_seqlens: {cu_seqlens.tolist()}")
        log(f"Number of sequences: {len(cu_seqlens) - 1}")
        if len(cu_seqlens) > 1:
            seq_lengths = [(cu_seqlens[i+1] - cu_seqlens[i]).item() for i in range(len(cu_seqlens)-1)]
            log(f"Sequence lengths: {seq_lengths}")
    
    # Video data information
    pixel_values_videos = batch.get("pixel_values_videos")
    video_grid_thw = batch.get("video_grid_thw")
    if pixel_values_videos is not None:
        log(f"\nVideo data:")
        log(f"  pixel_values_videos shape: {pixel_values_videos.shape}")
        if video_grid_thw is not None:
            log(f"  video_grid_thw shape: {video_grid_thw.shape}")
            log(f"  Number of video segments: {len(video_grid_thw)}")
            log(f"  video_grid_thw (all):")
            for i, thw in enumerate(video_grid_thw):
                log(f"    [{i}] t={thw[0].item()}, h={thw[1].item()}, w={thw[2].item()}")
    
    fast_pixel_values_videos = batch.get("fast_pixel_values_videos")
    fast_video_grid_thw = batch.get("fast_video_grid_thw")
    if fast_pixel_values_videos is not None:
        log(f"\nFast video data:")
        log(f"  fast_pixel_values_videos shape: {fast_pixel_values_videos.shape}")
        if fast_video_grid_thw is not None:
            log(f"  fast_video_grid_thw shape: {fast_video_grid_thw.shape}")
            log(f"  Number of fast video segments: {len(fast_video_grid_thw)}")
            log(f"  fast_video_grid_thw (all):")
            for i, thw in enumerate(fast_video_grid_thw):
                log(f"    [{i}] t={thw[0].item()}, h={thw[1].item()}, w={thw[2].item()}")
    
    # Image data information
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    if pixel_values is not None:
        log(f"\nImage data:")
        log(f"  pixel_values shape: {pixel_values.shape}")
        if image_grid_thw is not None:
            log(f"  image_grid_thw shape: {image_grid_thw.shape}")
            log(f"  Number of image segments: {len(image_grid_thw)}")
    
    # Loss mask summary
    if loss_mask is not None:
        loss_mask_flat = loss_mask.flatten()
        valid_tokens = (loss_mask_flat > 0).sum().item()
        total_tokens = loss_mask_flat.numel()
        log(f"\nLoss mask:")
        log(f"  Valid tokens (loss_mask > 0): {valid_tokens} / {total_tokens} ({100*valid_tokens/total_tokens:.2f}%)")
    
    log(f"{'='*80}\n")
    
    return num_samples_in_batch


def main():
    parser = argparse.ArgumentParser(description="Debug video dataset packing and sample_idx")
    parser.add_argument("--dataset-config", type=str, required=True,
                       help="Path to dataset config JSON file")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Model directory (for base_model_dir if not in config)")
    parser.add_argument("--num-batches", type=int, default=10,
                       help="Number of batches to analyze")
    parser.add_argument("--rank", type=int, default=0,
                       help="Rank for distributed dataset (default: 0)")
    parser.add_argument("--world-size", type=int, default=1,
                       help="World size for distributed dataset (default: 1)")
    
    args = parser.parse_args()
    
    # Initialize distributed environment (required by dataset internals)
    # Even for single GPU, we need to init process group for get_data_parallel_rank()
    
    # Get rank/world_size from environment (set by torchrun)
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = args.rank
        world_size = args.world_size
        local_rank = 0
    
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Override args with actual values
    args.rank = rank
    args.world_size = world_size
    
    # Helper function for logging with rank
    def log(msg):
        print_with_rank(msg, rank)
    
    log("Initializing distributed environment...")
    
    # Initialize model parallel (for get_data_parallel_rank/group)
    initialize_model_parallel(context_parallel_size=1)
    log(f"Distributed initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    
    # Load dataset config
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = json.load(f)
    
    # Override base_model_dir if provided
    if args.model_dir and not dataset_config.get("base_model_dir"):
        dataset_config["base_model_dir"] = args.model_dir
    
    # Fix num_workers: DistributedDataset requires num_workers >= 1 for file sharding
    # (total_workers = world_size * num_workers, used as slice step)
    if dataset_config.get("num_workers", 0) == 0:
        dataset_config["num_workers"] = 1
        log("Note: Setting dataset num_workers=1 (required for DistributedDataset file sharding)")
    
    # Add distributed info
    dataset_config["rank"] = args.rank
    dataset_config["world_size"] = args.world_size
    
    log("="*80)
    log("Video Dataset Debug Script (Multi-GPU)")
    log("="*80)
    log(f"Dataset config: {args.dataset_config}")
    log(f"Rank: {args.rank}, World size: {args.world_size}")
    log(f"Number of batches to analyze: {args.num_batches}")
    log("="*80)
    
    # Create dataset
    log("\nCreating dataset...")
    try:
        dataset = ChatCompletionVisionDataset_keye_vitrope_slowfast_video(**dataset_config)
        log("Dataset created successfully!")
    except Exception as e:
        log(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create dataloader
    # IMPORTANT: Use num_workers=0 to avoid multiprocessing issues
    # The dataset uses dist.get_rank() internally, which fails in worker processes
    # without distributed initialization
    log("\nCreating dataloader (num_workers=0 to avoid dist init issues)...")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Must be 0 for non-distributed debug mode
        collate_fn=lambda x: x[0]  # Unwrap single-element list
    )
    
    # Iterate through batches
    log(f"\nIterating through {args.num_batches} batches...")
    log("="*80)
    
    total_samples = 0
    total_tokens = 0
    sum_of_batch_samples = 0  # Sum of logical samples from each batch
    batch_samples_list = []   # List of samples per batch for verification
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            
            # Analyze batch and get samples count
            num_samples_in_batch = analyze_batch(batch, batch_idx, total_samples, rank)
            
            # Accumulate statistics
            sample_idx = batch.get("sample_idx")
            if sample_idx is not None:
                max_idx = sample_idx.max().item()
                logical_samples = max_idx + 1
                total_samples += logical_samples
                sum_of_batch_samples += num_samples_in_batch
                batch_samples_list.append(logical_samples)
            
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                total_tokens += input_ids.numel()
        
        # Print per-rank summary
        log("\n" + "="*80)
        log(f"PER-RANK SUMMARY (Rank {rank})")
        log("="*80)
        log(f"Total batches analyzed: {batch_idx + 1}")
        log(f"Total logical samples (accumulated): {total_samples}")
        log(f"Sum of batch samples: {sum_of_batch_samples}")
        log(f"Samples per batch: {batch_samples_list}")
        log(f"Sum verification: {sum(batch_samples_list)} == {total_samples} ? {sum(batch_samples_list) == total_samples}")
        log(f"Average samples per batch: {total_samples / (batch_idx + 1):.2f}")
        log(f"Total tokens: {total_tokens}")
        log(f"Average tokens per batch: {total_tokens / (batch_idx + 1):.2f}")
        log("="*80)
        
        # Synchronize all ranks before global summary
        dist.barrier()
        
        # Gather statistics from all ranks for global summary
        local_stats = torch.tensor([
            float(batch_idx + 1),  # num_batches
            float(total_samples),   # total_samples
            float(total_tokens),    # total_tokens
        ], device=torch.cuda.current_device())
        
        # All-reduce to sum across all ranks
        global_stats = local_stats.clone()
        dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)
        
        # Gather per-rank samples for detailed view (only on rank 0)
        all_samples = [torch.tensor([0.0], device=torch.cuda.current_device()) for _ in range(world_size)]
        dist.all_gather(all_samples, torch.tensor([float(total_samples)], device=torch.cuda.current_device()))
        
        all_tokens = [torch.tensor([0.0], device=torch.cuda.current_device()) for _ in range(world_size)]
        dist.all_gather(all_tokens, torch.tensor([float(total_tokens)], device=torch.cuda.current_device()))
        
        all_batches = [torch.tensor([0.0], device=torch.cuda.current_device()) for _ in range(world_size)]
        dist.all_gather(all_batches, torch.tensor([float(batch_idx + 1)], device=torch.cuda.current_device()))
        
        # Print global summary only on rank 0
        if rank == 0:
            print("\n" + "="*80, flush=True)
            print("GLOBAL SUMMARY (All Ranks Combined)", flush=True)
            print("="*80, flush=True)
            print(f"World size: {world_size}", flush=True)
            print(f"\nPer-rank breakdown:", flush=True)
            for r in range(world_size):
                print(f"  Rank {r}: {int(all_batches[r].item())} batches, "
                      f"{int(all_samples[r].item())} samples, "
                      f"{int(all_tokens[r].item())} tokens", flush=True)
            print(f"\nGlobal totals:", flush=True)
            print(f"  Total batches (all ranks): {int(global_stats[0].item())}", flush=True)
            print(f"  Total samples (all ranks): {int(global_stats[1].item())}", flush=True)
            print(f"  Total tokens (all ranks): {int(global_stats[2].item())}", flush=True)
            print(f"  Average samples per rank: {global_stats[1].item() / world_size:.2f}", flush=True)
            print(f"  Average tokens per rank: {global_stats[2].item() / world_size:.2f}", flush=True)
            print("="*80, flush=True)
        
        dist.barrier()
        
    except Exception as e:
        log(f"\nError during iteration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

