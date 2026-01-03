"""
Test script for compute_input_pos_with_group_step function
"""

import torch
import torch.nn.functional as F

def mock_compute_input_pos(h: int, w: int, device: torch.device = None) -> dict:
    """Mock implementation of compute_input_pos for testing"""
    seq_len = h * w
    pos_ids = torch.arange(seq_len, device=device)
    height_ids = pos_ids // w
    width_ids = pos_ids % w
    return {"height": height_ids, "width": width_ids}

def mock_resize_hw(hw, max_tokens):
    """Mock implementation of resize_hw that keeps aspect ratio"""
    h, w = hw.tolist()
    # Simple scaling: reduce dimensions while keeping aspect ratio, not exceeding max_tokens
    scale_factor = min(1.0, (max_tokens / (h * w)) ** 0.5)
    h_out = max(1, int(h * scale_factor))
    w_out = max(1, int(w * scale_factor))
    # Adjust to not exceed max_tokens
    while h_out * w_out > max_tokens:
        if h_out > w_out:
            h_out -= 1
        else:
            w_out -= 1
    return torch.tensor([h_out, w_out])

def compute_input_pos_with_group_step(image_grid_thw, cond_pos_scale, max_seq_len, device):
    '''
    Compute position ids with group step sampling.
    Algorithm:
    1. Scale height and width by cond_pos_scale to get expanded position ids
    2. Sample every cond_pos_scale-th position id
    '''
    # Get height and width from image grid (after dividing by 2 and applying resize)
    hw_input = image_grid_thw[0][1:] // 2  # Assume original grid is divided by 2
    h_cond, w_cond = (mock_resize_hw(hw_input, max_seq_len) * int(cond_pos_scale)).tolist()
    
    # Compute full position grid
    cond_input_pos = mock_compute_input_pos(h_cond, w_cond, device=device)
    
    # Sample every cond_pos_scale-th position id along height and width
    cond_input_pos['height'] = cond_input_pos['height'][::cond_pos_scale]
    cond_input_pos['width'] = cond_input_pos['width'][::cond_pos_scale]
    
    # Calculate sequence length after sampling
    sampled_h = len(cond_input_pos['height'].unique())
    sampled_w = len(cond_input_pos['width'].unique())
    sampled_seq_len = sampled_h * sampled_w
    
    # Pad to max_seq_len if needed
    if sampled_seq_len < max_seq_len:
        pad_len = max_seq_len - sampled_seq_len
        cond_input_pos['height'] = F.pad(cond_input_pos['height'], (0, pad_len), value=0)
        cond_input_pos['width'] = F.pad(cond_input_pos['width'], (0, pad_len), value=0)
    elif sampled_seq_len > max_seq_len:
        # Truncate to max_seq_len
        cond_input_pos['height'] = cond_input_pos['height'][:max_seq_len]
        cond_input_pos['width'] = cond_input_pos['width'][:max_seq_len]
    
    return cond_input_pos

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
    print(f"  Unique height values: {result['height'].unique().tolist()}")
    print(f"  Unique width values: {result['width'].unique().tolist()}")
    
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
    print(f"  Unique height values: {result['height'].unique().tolist()}")
    print(f"  Unique width values: {result['width'].unique().tolist()}")
    
    # Test case 3: Different input sizes for comparison
    print("\n=== Test Case 3: Different grid sizes comparison ===")
    test_cases = [
        (torch.tensor([[1, 4, 4]]), 1, 16),
        (torch.tensor([[1, 8, 8]]), 1, 16),
        (torch.tensor([[1, 4, 4]]), 2, 16),
        (torch.tensor([[1, 8, 8]]), 2, 16),
    ]
    
    for grid, scale, max_len in test_cases:
        result = compute_input_pos_with_group_step(grid, scale, max_len, device)
        print(f"Grid: {grid[0][1:].tolist()}, Scale: {scale}, MaxLen: {max_len}")
        print(f"  Output length: {len(result['height'])}")
        print(f"  Height uniq: {result['height'].unique().tolist()}")
        print(f"  Width uniq: {result['width'].unique().tolist()}")

if __name__ == "__main__":
    test_compute_input_pos_with_group_step()