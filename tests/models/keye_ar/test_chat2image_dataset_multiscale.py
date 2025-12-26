"""
Demo test for Chat2ImageDataset with MultiScaleDatasetWrapper.

This test demonstrates how to use Chat2ImageDataset wrapped with MultiScaleDatasetWrapper
for multi-scale training, following the pattern from recipes/sana/train_sana_ar_dit.py
"""

import torch
import pytest
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Dict, Any

from muse.data.datasets import Chat2ImageDataset, MultiScaleDatasetWrapper
from muse.data.utils import ResolutionBudget, ResolutionBudgetConfig


def demo_chat2image_dataset_multiscale(
    sources: str = "path/to/parquet/files",
    processor_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    image_size: int = 1024,
    batch_size: int = 8,
    max_condition_length: int = 384,
    num_workers: int = 0,
    use_multi_scale: bool = True,
    resolution_budgets: List[tuple] = None,
) -> DataLoader:
    """
    Demo function showing how to use Chat2ImageDataset with MultiScaleDatasetWrapper.
    
    This demonstrates the typical pattern used in recipes/sana/train_sana_ar_dit.py
    for setting up multi-scale training with chat-style image generation datasets.
    
    Args:
        sources: Path to parquet files or directory containing dataset
        processor_path: Path to processor (e.g., "Qwen/Qwen2-VL-7B-Instruct")
        image_size: Base image size for fixed-size training
        batch_size: Batch size for fixed-size training
        max_condition_length: Maximum condition sequence length for processor
        num_workers: Number of dataloader workers
        use_multi_scale: Whether to enable multi-scale training
        resolution_budgets: List of (resolution, batch_size) tuples for multi-scale training
                           If None, creates a single-resolution config
    
    Returns:
        Configured DataLoader with appropriate collate_fn
    
    Example:
        >>> # Fixed-size training
        >>> dataloader = demo_chat2image_dataset_multiscale(
        ...     sources="/path/to/data",
        ...     use_multi_scale=False,
        ...     batch_size=8
        ... )
        
        >>> # Multi-scale training with custom budgets
        >>> dataloader = demo_chat2image_dataset_multiscale(
        ...     sources="/path/to/data",
        ...     use_multi_scale=True,
        ...     resolution_budgets=[(512, 32), (1024, 8)]
        ... )
    """
    
    # ====================================================================================
    # Step 1: Prepare dataset configuration
    # ====================================================================================
    dataset_config = {
        "sources": sources,
        "image_size": image_size,
        "processor_path": processor_path,
        "max_condition_length": max_condition_length,
        "center_crop": True,
        "multi_scale": use_multi_scale,
    }
    
    print(f"Building Chat2ImageDataset with config: {dataset_config}")
    
    # ====================================================================================
    # Step 2: Create the Chat2ImageDataset instance
    # ====================================================================================
    dataset = Chat2ImageDataset(**dataset_config)
    collate_fn = dataset.collate_fn
    
    print(f"Chat2ImageDataset created successfully")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Max condition length: {max_condition_length}")
    print(f"  Multi-scale enabled: {use_multi_scale}")
    
    # ====================================================================================
    # Step 3: Configure multi-scale training if enabled
    # ====================================================================================
    if use_multi_scale:
        # Create resolution budget configuration
        if resolution_budgets is None:
            # Default: single resolution (no multi-scale variation)
            # In this case, all samples are trained at the same resolution
            budget_config = ResolutionBudgetConfig(
                budgets=[ResolutionBudget(image_size, batch_size)],
            )
        else:
            # Custom multi-scale budgets
            # Example: [(512, 32), (1024, 8)] means:
            # - 512x512 resolution with batch size 32
            # - 1024x1024 resolution with batch size 8
            budgets = [
                ResolutionBudget(resolution, bs)
                for resolution, bs in resolution_budgets
            ]
            budget_config = ResolutionBudgetConfig(budgets=budgets)
        
        print(f"Multi-scale training configuration:")
        for b in budget_config.budgets:
            print(f"  {b.size}x{b.size}: batch_size={b.batch_size}")
        
        # ====================================================================================
        # Step 4: Wrap dataset with MultiScaleDatasetWrapper
        # ====================================================================================
        # This wrapper groups samples by (resolution, aspect_ratio) into buckets
        # and yields pre-batched samples when buckets are full
        multi_scale_wrapper = MultiScaleDatasetWrapper(
            dataset=dataset,
            config=budget_config,
            drop_last=True,  # Drop incomplete batches at the end
            max_bucket_size=10000,  # Maximum samples per bucket
        )
        
        print(f"MultiScaleDatasetWrapper created successfully")
        print(f"  Number of resolutions: {len(budget_config.budgets)}")
        print(f"  Drop last: True")
        
        # ====================================================================================
        # Step 5: Create DataLoader with wrapped dataset
        # ====================================================================================
        # Key points:
        # - batch_size=1 because MultiScaleDatasetWrapper already yields pre-batched lists
        # - The wrapper returns a list of samples (already batched)
        # - collate_fn receives x[0] which is the pre-batched list
        dataloader = DataLoader(
            multi_scale_wrapper,
            batch_size=1,  # Wrapper yields pre-batched lists
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x[0]),  # x is [(sample1, sample2, ...)]
            drop_last=False,  # Already handled by wrapper
        )
        
        print(f"DataLoader created for multi-scale training")
        
    else:
        # ====================================================================================
        # Alternative: Fixed-size training (no multi-scale wrapper)
        # ====================================================================================
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
        
        print(f"DataLoader created for fixed-size training")
        print(f"  Batch size: {batch_size}")
    
    return dataloader


# ====================================================================================
# Optional: Example usage and testing
# ====================================================================================

def example_iterate_dataloader(dataloader: DataLoader, max_batches: int = 2) -> None:
    """
    Example function showing how to iterate through the dataloader.
    
    Args:
        dataloader: The configured DataLoader
        max_batches: Maximum number of batches to iterate (for testing)
    """
    print(f"\nIterating through dataloader (showing first {max_batches} batches):")
    print("=" * 80)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: type={type(value)}")
        
        # Show example of accessing specific fields
        if "image" in batch:
            print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        
        if "input_ids" in batch:
            print(f"  Input IDs: {batch['input_ids'].shape}")
        
        if "pixel_values" in batch:
            print(f"  Pixel values: {batch['pixel_values'].shape}")


if __name__ == "__main__":
    """
    Quick test of the demo functions.
    Note: Requires valid dataset path and processor path to run fully.
    """
    print("Chat2ImageDataset + MultiScaleDatasetWrapper Demo")
    print("=" * 80)
    
    # Example 1: Multi-scale training with custom budgets
    print("\nExample 1: Multi-scale training with custom resolution budgets")
    print("-" * 80)
    try:
        dataloader = demo_chat2image_dataset_multiscale(
            sources="path/to/parquet/files",  # Replace with actual path
            processor_path="Qwen/Qwen2-VL-7B-Instruct",
            image_size=1024,
            use_multi_scale=True,
            resolution_budgets=[(512, 32), (1024, 8)],
            num_workers=0,
        )
        # example_iterate_dataloader(dataloader, max_batches=1)
        print("✓ Multi-scale dataloader created successfully")
    except Exception as e:
        print(f"✗ Error creating multi-scale dataloader: {e}")
    
    # Example 2: Fixed-size training
    print("\n\nExample 2: Fixed-size training (no multi-scale)")
    print("-" * 80)
    try:
        dataloader = demo_chat2image_dataset_multiscale(
            sources="path/to/parquet/files",  # Replace with actual path
            processor_path="Qwen/Qwen2-VL-7B-Instruct",
            image_size=1024,
            batch_size=8,
            use_multi_scale=False,
            num_workers=0,
        )
        # example_iterate_dataloader(dataloader, max_batches=1)
        print("✓ Fixed-size dataloader created successfully")
    except Exception as e:
        print(f"✗ Error creating fixed-size dataloader: {e}")
    
    # Example 3: Multi-scale with single resolution (default)
    print("\n\nExample 3: Multi-scale wrapper with single resolution (default)")
    print("-" * 80)
    try:
        dataloader = demo_chat2image_dataset_multiscale(
            sources="path/to/parquet/files",  # Replace with actual path
            processor_path="Qwen/Qwen2-VL-7B-Instruct",
            image_size=1024,
            batch_size=8,
            use_multi_scale=True,
            resolution_budgets=None,  # Uses default single-resolution config
            num_workers=0,
        )
        print("✓ Multi-scale dataloader (single resolution) created successfully")
    except Exception as e:
        print(f"✗ Error creating multi-scale dataloader: {e}")
    
    print("\n" + "=" * 80)
    print("Demo complete! Dataloader is ready for training.")
