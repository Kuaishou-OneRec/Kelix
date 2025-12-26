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
    sources: str = "/llm_reco/vlm/datahub/datasets/Sana_pretrain/0.0.0/index/parquet.json",
    processor_path: str = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted",
    image_size: int = 1024,
    batch_size: int = 1,
    max_condition_length: int = 324,
    num_workers: int = 0,
    use_multi_scale: bool = True,
    resolution_budgets: List[tuple] = None,
    center_crop: bool = True,
) -> DataLoader:
    """
    Demo function showing how to use Chat2ImageDataset with MultiScaleDatasetWrapper.
    
    This demonstrates the typical pattern used in recipes/sana/train_sana_ar_dit.py
    for setting up multi-scale training with chat-style image generation datasets.
    
    Based on: examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im.sh
    
    Configuration Sources:
    - Dataset config: examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im.json
    - Training script: recipes/sana/train_sana_ar_dit.py
    - Model/Processor: KEYE_AR_DIR from training config
    
    Args:
        sources: Path to parquet index file or directory containing dataset
                Default from run_ar_dit_lzx_4096_v2_1024im.json:
                /llm_reco/vlm/datahub/datasets/Sana_pretrain/0.0.0/index/parquet.json
        
        processor_path: Path to Keye AR processor/model for Chat2ImageDataset
                       Default from run_ar_dit_lzx_4096_v2_1024im.sh KEYE_AR_DIR:
                       /mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted
        
        image_size: Base image size in pixels (default: 1024)
                   From run_ar_dit_lzx_4096_v2_1024im.sh --image-size
        
        batch_size: Batch size for single-scale training (default: 1)
                   From run_ar_dit_lzx_4096_v2_1024im.sh --batch-size
        
        max_condition_length: Maximum condition sequence length for processor (default: 324)
                            From run_ar_dit_lzx_4096_v2_1024im.sh --max-condition-length
                            and run_ar_dit_lzx_4096_v2_1024im.json
        
        num_workers: Number of dataloader workers (default: 0)
        
        use_multi_scale: Whether to enable multi-scale training (default: True)
                        From run_ar_dit_lzx_4096_v2_1024im.sh --multi-scale flag
        
        resolution_budgets: List of (resolution, batch_size) tuples for multi-scale training
                           If None, uses single resolution config (default behavior when
                           --resolution-budgets is not specified in training script)
        
        center_crop: Whether to center crop images (default: True)
                    From run_ar_dit_lzx_4096_v2_1024im.json
    
    Returns:
        Configured DataLoader with appropriate collate_fn
    
    Example:
        >>> # Multi-scale training with single resolution (default, matches run_ar_dit_lzx_4096_v2_1024im.sh)
        >>> dataloader = demo_chat2image_dataset_multiscale()
        
        >>> # Fixed-size training (no multi-scale)
        >>> dataloader = demo_chat2image_dataset_multiscale(
        ...     use_multi_scale=False,
        ...     batch_size=1
        ... )
        
        >>> # Multi-scale with custom budgets
        >>> dataloader = demo_chat2image_dataset_multiscale(
        ...     use_multi_scale=True,
        ...     resolution_budgets=[(512, 32), (1024, 8)]
        ... )
    """
    
    # ====================================================================================
    # Step 1: Prepare dataset configuration
    # Based on: examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im.json
    # ====================================================================================
    dataset_config = {
        "sources": sources,
        "image_size": image_size,
        "processor_path": processor_path,
        "max_condition_length": max_condition_length,
        "center_crop": center_crop,
        "packing": False,
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
    # Note: When using --multi-scale flag without --resolution-budgets in the training script,
    # it creates a single-resolution budget (no curriculum scheduling).
    # See: recipes/sana/train_sana_ar_dit.py lines 1217-1221
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
    
    Configuration based on: examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im.sh
    """
    print("Chat2ImageDataset + MultiScaleDatasetWrapper Demo")
    print("=" * 80)
    
    # Example 1: Multi-scale training with single resolution (default config)
    # Mirrors: examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im.sh
    print("\nExample 1: Multi-scale training with single resolution (default)")
    print("-" * 80)
    try:
        dataloader = demo_chat2image_dataset_multiscale(
            sources="/llm_reco/vlm/datahub/datasets/Sana_pretrain/0.0.0/index/parquet.json",
            processor_path="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted",
            image_size=1024,
            batch_size=1,
            max_condition_length=324,
            use_multi_scale=True,
            num_workers=0,
        )
        # example_iterate_dataloader(dataloader, max_batches=1)
        print("✓ Multi-scale dataloader (single resolution) created successfully")
    except Exception as e:
        print(f"✗ Error creating multi-scale dataloader: {e}")
    
    # Example 2: Fixed-size training (no multi-scale wrapper)
    print("\n\nExample 2: Fixed-size training (no multi-scale)")
    print("-" * 80)
    try:
        dataloader = demo_chat2image_dataset_multiscale(
            sources="/llm_reco/vlm/datahub/datasets/Sana_pretrain/0.0.0/index/parquet.json",
            processor_path="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted",
            image_size=1024,
            batch_size=1,
            max_condition_length=324,
            use_multi_scale=False,
            num_workers=0,
        )
        # example_iterate_dataloader(dataloader, max_batches=1)
        print("✓ Fixed-size dataloader created successfully")
    except Exception as e:
        print(f"✗ Error creating fixed-size dataloader: {e}")
    
    # Example 3: Multi-scale with custom resolution budgets
    print("\n\nExample 3: Multi-scale training with custom resolution budgets")
    print("-" * 80)
    try:
        dataloader = demo_chat2image_dataset_multiscale(
            sources="/llm_reco/vlm/datahub/datasets/Sana_pretrain/0.0.0/index/parquet.json",
            processor_path="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted",
            image_size=1024,
            batch_size=1,
            max_condition_length=324,
            use_multi_scale=True,
            resolution_budgets=[(512, 32), (1024, 8)],
            num_workers=0,
        )
        # example_iterate_dataloader(dataloader, max_batches=1)
        print("✓ Multi-scale dataloader (custom budgets) created successfully")
    except Exception as e:
        print(f"✗ Error creating multi-scale dataloader: {e}")
    
    print("\n" + "=" * 80)
    print("Demo complete! Dataloader is ready for training.")
