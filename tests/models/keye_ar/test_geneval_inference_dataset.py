"""
Demo test for GenEvalInferenceDataset.

This test demonstrates how to use GenEvalInferenceDataset with default parameters.
"""

import torch
import torch.distributed as dist
import pytest
from pathlib import Path
from typing import List, Dict, Any
import os

from muse.data.datasets import GenEvalInferenceDataset

def setup_distributed_environment() -> bool:
    """
    Initialize distributed environment for testing.
    
    In a single-process test environment, we initialize a local process group
    to avoid distributed training errors.
    
    Returns:
        True if distributed was already initialized or successfully initialized,
        False otherwise
    """
    if dist.is_available() and not dist.is_initialized():
        try:
            # For testing, use the default backend on CPU
            dist.init_process_group(
                backend='gloo',
                init_method='tcp://127.0.0.1:29500',
                rank=0,
                world_size=1,
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize distributed environment: {e}")
            print("Will attempt to run without distributed mode...")
            return False
    return True

def demo_geneval_inference_dataset(
    gen_eval_csv_path: str = None,
    template: str = '{}',
    systemp_prompt: str = "You are a helpful assistant.",
    initialize_dist: bool = True
) -> GenEvalInferenceDataset:
    """
    Demo function showing how to use GenEvalInferenceDataset with default parameters.
    
    Args:
        gen_eval_csv_path: Path to GenEval TSV file. If None, uses default path.
        template: Template string for processing questions. Default is '{}'.
        systemp_prompt: System prompt to use. Default is "You are a helpful assistant.".
        initialize_dist: Whether to initialize distributed environment (default: True).
    
    Returns:
        Initialized GenEvalInferenceDataset instance
    
    Example:
        >>> # Use default parameters
        >>> dataset = demo_geneval_inference_dataset()
        >>> first_sample = next(iter(dataset))
        >>> print(f"First processed sample: {first_sample.keys()}")
        
        >>> # Use custom CSV path
        >>> dataset = demo_geneval_inference_dataset(
        ...     gen_eval_csv_path="/path/to/GenEval.tsv"
        ... )
    """
    
    # ====================================================================================
    # Setup: Initialize distributed environment if needed
    # ====================================================================================
    if initialize_dist:
        setup_distributed_environment()
    
    # ====================================================================================
    # Step 1: Create the GenEvalInferenceDataset instance with default parameters
    # ====================================================================================
    print("Initializing GenEvalInferenceDataset...")
    
    dataset_args = {
        "template": template,
        "systemp_prompt": systemp_prompt
    }
    
    # Only add gen_eval_csv_path if provided (otherwise use default)
    if gen_eval_csv_path is not None:
        dataset_args["gen_eval_csv_path"] = gen_eval_csv_path
    
    dataset = GenEvalInferenceDataset(**dataset_args)
    
    print(f"GenEvalInferenceDataset created successfully")
    print(f"  Configured with:")
    print(f"    - gen_eval_csv_path: {dataset.gen_eval_csv_path}")
    print(f"    - template: {template}")
    print(f"    - systemp_prompt: {systemp_prompt}")
    
    # ====================================================================================
    # Step 2: Display dataset information
    # ====================================================================================
    if hasattr(dataset, 'all_data') and isinstance(dataset.all_data, list):
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(dataset.all_data)}")
        
        if len(dataset.all_data) > 0:
            # Use iter and next to get the first processed sample
            try:
                first_processed_sample = next(iter(dataset))
                print(f"\nFirst processed sample keys: {first_processed_sample.keys()}")
                print(f"  - Sample contains image: {'image' in first_processed_sample}")
                print(f"  - Sample contains input_ids: {'input_ids' in first_processed_sample}")
                print(f"  - Sample contains attention_mask: {'attention_mask' in first_processed_sample}")
                
                if 'input_ids' in first_processed_sample:
                    print(f"  - Input IDs shape: {first_processed_sample['input_ids'].shape}")
                if 'image' in first_processed_sample:
                    print(f"  - Image shape: {first_processed_sample['image'].shape}")
            except Exception as e:
                print(f"\nWarning: Failed to get processed sample: {e}")
                print("  - Raw first sample structure:")
                for key, value in dataset.all_data[0].items():
                    print(f"    - {key}: {value}")
            
            # Show tag distribution
            tags = [sample['tag'] for sample in dataset.all_data]
            tag_counts = {}
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            print(f"\nTag distribution:")
            for tag, count in tag_counts.items():
                print(f"    - {tag}: {count} samples")
    
    return dataset


def test_geneval_inference_dataset_basic():
    """
    Basic test for GenEvalInferenceDataset initialization.
    
    This test checks that the dataset can be initialized with default parameters
    and that the __iter__ method works correctly.
    """
    print("=== Running GenEvalInferenceDataset Basic Test ===")
    
    # Initialize with default parameters
    dataset = demo_geneval_inference_dataset()
    
    # Verify that all_data is loaded correctly
    assert hasattr(dataset, 'all_data'), "Dataset should have 'all_data' attribute"
    assert isinstance(dataset.all_data, list), "'all_data' should be a list"
    assert len(dataset.all_data) > 0, "'all_data' should not be empty"
    
    # Test __iter__ method by getting the first sample
    first_processed_sample = next(iter(dataset))
    print(f"== First processed sample: {first_processed_sample}")
    assert isinstance(first_processed_sample, dict), "Processed sample should be a dict"
    
    # Verify the shapes of processed tensors
    assert first_processed_sample['input_ids'].ndim >= 1, "Input IDs should be a tensor"
    
    print("✅ All basic tests passed!")
    return True



if __name__ == "__main__":
    # Run the demo
    print("=== GenEvalInferenceDataset Demo ===")
    dataset = demo_geneval_inference_dataset()
    
    # Run the test
    print("\n=== Running Test ===")
    test_geneval_inference_dataset_basic()