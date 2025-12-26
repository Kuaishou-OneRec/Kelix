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
    Example function showing how to iterate through the dataloader and inspect samples.
    
    This function demonstrates how to fetch and analyze actual samples from the dataloader,
    including tensor shapes, data ranges, and field information.
    
    Args:
        dataloader: The configured DataLoader
        max_batches: Maximum number of batches to iterate and print (for testing)
    """
    print(f"\n{'=' * 100}")
    print(f"Iterating through dataloader (showing first {max_batches} batches)")
    print(f"{'=' * 100}")
    
    total_samples = 0
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            print(f"\n{'─' * 100}")
            print(f"Batch {batch_idx}:")
            print(f"{'─' * 100}")
            
            # Print batch keys
            print(f"Available fields: {list(batch.keys())}")
            
            # Iterate through all fields in the batch
            for key, value in batch.items():
                print(f"\n  Field: '{key}'")
                
                if isinstance(value, torch.Tensor):
                    print(f"    Type: torch.Tensor")
                    print(f"    Shape: {value.shape}")
                    print(f"    Data type: {value.dtype}")
                    print(f"    Device: {value.device}")
                    
                    # Show value ranges for numerical tensors
                    if value.dtype in [torch.float32, torch.float64, torch.float16]:
                        print(f"    Value range: [{value.min():.4f}, {value.max():.4f}]")
                        print(f"    Mean: {value.mean():.4f}, Std: {value.std():.4f}")
                    
                    # Show more details for specific fields
                    if key == "image":
                        total_samples += value.shape[0]
                        print(f"    ├─ Number of images: {value.shape[0]}")
                        print(f"    ├─ Image dimensions: {value.shape[-2:]} (H x W)")
                        print(f"    └─ Channels: {value.shape[1] if len(value.shape) > 3 else 'N/A'}")
                    
                    elif key == "pixel_values":
                        print(f"    ├─ Total patches: {value.shape[0]}")
                        print(f"    └─ Patch feature dim: {value.shape[-1]}")
                    
                    elif key == "image_grid_thw":
                        print(f"    ├─ Grid info shape: {value.shape}")
                        print(f"    └─ Contains (T, H, W) grid dimensions for each image")
                    
                    elif key in ["input_ids", "attention_mask"]:
                        print(f"    ├─ Sequence length (cumulative): {value.shape[-1] if len(value.shape) > 1 else value.shape[0]}")
                        if key == "input_ids" and len(value.shape) > 1:
                            print(f"    └─ Non-zero tokens: {(value > 0).sum().item()}")
                    
                    elif key == "cu_seqlens":
                        print(f"    ├─ Cumulative sequence lengths: {value}")
                        print(f"    └─ Number of samples: {len(value) - 1}")
                
                elif isinstance(value, dict):
                    print(f"    Type: dict")
                    print(f"    Keys: {list(value.keys())}")
                    print(f"    Number of items: {len(value)}")
                
                elif isinstance(value, list):
                    print(f"    Type: list")
                    print(f"    Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                
                else:
                    print(f"    Type: {type(value)}")
                    print(f"    Value: {value}")
            
            # Print summary statistics for the batch
            print(f"\n  Batch Summary:")
            print(f"    ├─ Total samples in batch: {total_samples}")
            if "image" in batch:
                print(f"    ├─ Image batch shape: {batch['image'].shape}")
            if "input_ids" in batch:
                print(f"    ├─ Text tokens shape: {batch['input_ids'].shape}")
            if "pixel_values" in batch:
                print(f"    └─ Vision patches shape: {batch['pixel_values'].shape}")
    
    except StopIteration:
        print(f"\nDataloader iteration completed.")
    except Exception as e:
        print(f"\n✗ Error during dataloader iteration: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'=' * 100}")
    print(f"Total samples fetched: {total_samples}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    """
    Quick test of the demo functions.
    Note: Requires valid dataset path and processor path to run fully.
    
    Configuration based on: examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im.sh
    """
    print("Chat2ImageDataset + MultiScaleDatasetWrapper Demo")
    print("=" * 100)
    
    # Example 1: Multi-scale training with single resolution (default config)
    # Mirrors: examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im.sh
    print("\n\n" + "█" * 100)
    print("Example 1: Multi-scale training with single resolution (default)")
    print("█" * 100)
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
        print("✓ Multi-scale dataloader (single resolution) created successfully")
        
        # Fetch and print actual samples from dataloader
        print("\nFetching samples from dataloader...")
        example_iterate_dataloader(dataloader, max_batches=1)
        
    except Exception as e:
        print(f"✗ Error creating/using multi-scale dataloader: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Fixed-size training (no multi-scale wrapper)
    print("\n\n" + "█" * 100)
    print("Example 2: Fixed-size training (no multi-scale)")
    print("█" * 100)
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
        print("✓ Fixed-size dataloader created successfully")
        
        # Fetch and print actual samples from dataloader
        print("\nFetching samples from dataloader...")
        example_iterate_dataloader(dataloader, max_batches=1)
        
    except Exception as e:
        print(f"✗ Error creating/using fixed-size dataloader: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 3: Multi-scale with custom resolution budgets
    print("\n\n" + "█" * 100)
    print("Example 3: Multi-scale training with custom resolution budgets")
    print("█" * 100)
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
        print("✓ Multi-scale dataloader (custom budgets) created successfully")
        
        # Fetch and print actual samples from dataloader
        print("\nFetching samples from dataloader...")
        example_iterate_dataloader(dataloader, max_batches=1)
        
    except Exception as e:
        print(f"✗ Error creating/using multi-scale dataloader: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n" + "=" * 100)
    print("Demo complete! All dataloader configurations tested with sample inspection.")
    print("=" * 100 + "\n")
