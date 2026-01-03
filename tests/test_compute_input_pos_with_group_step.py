"""
Test script for compute_input_pos_with_group_step function
"""

import torch
from recipes.sana.train_sana_ar_dit import compute_input_pos_with_group_step

def test_compute_input_pos_with_group_step():
    # Test case 1: Basic case with scale=1
    print("=== Test Case 1: Basic case with scale=1 ===")
    image_grid_thw = torch.tensor([[1, 4, 4]])  # [B, 3] where each row is (t, h, w)
    cond_pos_scale = 1
    max_seq_len = 16
    device = torch.device("cpu")
    
    result = compute_input_pos_with_group_step(image_grid_thw, cond_pos_scale, max_seq_len, device)
    print("Input:")
    print(f"  image_grid_thw: {image_grid_thw.tolist()}")
    print(f"  cond_pos_scale: {cond_pos_scale}")
    print(f"  max_seq_len: {max_seq_len}")
    print("Output:")
    print(f"  height positions: {result['height'].tolist()}")
    print(f"  width positions: {result['width'].tolist()}")
    print(f"  Output sequence length: {len(result['height'])}")
    
    # Test case 2: Scale=2
    print("\n=== Test Case 2: Scale=2 ===")
    cond_pos_scale = 2
    result = compute_input_pos_with_group_step(image_grid_thw, cond_pos_scale, max_seq_len, device)
    print("Input:")
    print(f"  image_grid_thw: {image_grid_thw.tolist()}")
    print(f"  cond_pos_scale: {cond_pos_scale}")
    print(f"  max_seq_len: {max_seq_len}")
    print("Output:")
    print(f"  height positions: {result['height'].tolist()}")
    print(f"  width positions: {result['width'].tolist()}")
    print(f"  Output sequence length: {len(result['height'])}")
    
    # Test case 3: Larger grid with padding
    print("\n=== Test Case 3: Larger grid with padding ===")
    image_grid_thw = torch.tensor([[1, 8, 8]])  # Original 8x8 grid
    cond_pos_scale = 2
    max_seq_len = 20  # Require padding
    result = compute_input_pos_with_group_step(image_grid_thw, cond_pos_scale, max_seq_len, device)
    print("Input:")
    print(f"  image_grid_thw: {image_grid_thw.tolist()}")
    print(f"  cond_pos_scale: {cond_pos_scale}")
    print(f"  max_seq_len: {max_seq_len}")
    print("Output:")
    print(f"  height positions: {result['height'].tolist()}")
    print(f"  width positions: {result['width'].tolist()}")
    print(f"  Output sequence length: {len(result['height'])}")
    print(f"  Padding length: {max_seq_len - 16}")  # 16 is 4x4 (8//2 * 8//2)

if __name__ == "__main__":
    test_compute_input_pos_with_group_step()