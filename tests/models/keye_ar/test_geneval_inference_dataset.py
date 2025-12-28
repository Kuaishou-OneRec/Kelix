"""
Demo test for GenEvalInferenceDataset.

This test demonstrates how to use GenEvalInferenceDataset with default parameters.
"""

import torch
import pytest
from pathlib import Path
from typing import List, Dict, Any
import os

from muse.data.datasets import GenEvalInferenceDataset

def demo_geneval_inference_dataset(
    gen_eval_csv_path: str = None,
    template: str = '{}',
    systemp_prompt: str = "You are a helpful assistant."
) -> GenEvalInferenceDataset:
    """
    Demo function showing how to use GenEvalInferenceDataset with default parameters.
    
    Args:
        gen_eval_csv_path: Path to GenEval TSV file. If None, uses default path.
        template: Template string for processing questions. Default is '{}'.
        systemp_prompt: System prompt to use. Default is "You are a helpful assistant.".
    
    Returns:
        Initialized GenEvalInferenceDataset instance
    
    Example:
        >>> # Use default parameters
        >>> dataset = demo_geneval_inference_dataset()
        >>> print(f"Loaded {len(dataset.all_data)} samples from GenEval dataset")
        >>> print(f"First sample: {dataset.all_data[0]}")
        
        >>> # Use custom CSV path
        >>> dataset = demo_geneval_inference_dataset(
        ...     gen_eval_csv_path="/path/to/GenEval.tsv"
        ... )
    """
    
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
            # Show first sample structure
            first_sample = dataset.all_data[0]
            print(f"\nFirst sample structure:")
            for key, value in first_sample.items():
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
    and that the _load_all_data method works correctly.
    """
    print("=== Running GenEvalInferenceDataset Basic Test ===")
    
    # Initialize with default parameters
    dataset = demo_geneval_inference_dataset()
    
    # Verify that all_data is loaded correctly
    assert hasattr(dataset, 'all_data'), "Dataset should have 'all_data' attribute"
    assert isinstance(dataset.all_data, list), "'all_data' should be a list"
    assert len(dataset.all_data) > 0, "'all_data' should not be empty"
    
    # Verify the structure of first sample
    first_sample = dataset.all_data[0]
    expected_keys = ['index', 'tag', 'include_class', 'include_count', 
                    'include_color', 'include_position', 'exclude_class', 
                    'exclude_count', 'question']
    
    for key in expected_keys:
        assert key in first_sample, f"Sample should contain key '{key}'"
    
    print("✅ All basic tests passed!")
    return True



if __name__ == "__main__":
    # Run the demo
    print("=== GenEvalInferenceDataset Demo ===")
    dataset = demo_geneval_inference_dataset()
    
    # Run the test
    print("\n=== Running Test ===")
    test_geneval_inference_dataset_basic()