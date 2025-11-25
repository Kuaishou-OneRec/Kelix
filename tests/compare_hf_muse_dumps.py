"""
Compare HF and Muse dumped activations in detail.

Usage:
    python compare_hf_muse_dumps.py <dump_directory>
"""

import os
import sys
import torch
import numpy as np


def load_activation(dump_dir, name):
    """Load a single activation by name."""
    filepath = os.path.join(dump_dir, f"{name}.pt")
    if os.path.exists(filepath):
        return torch.load(filepath)
    return None


def compare_tensors(name, hf_tensor, muse_tensor, atol=1e-3, rtol=1e-3):
    """Compare two tensors and print detailed diff."""
    print(f"\n{name}:")
    print("-" * 60)
    
    if hf_tensor is None:
        print("  ⚠️  HF tensor not found")
        return False
    
    if muse_tensor is None:
        print("  ⚠️  Muse tensor not found")
        return False
    
    # Handle list/tuple inputs
    if isinstance(hf_tensor, (list, tuple)):
        if len(hf_tensor) == 0:
            print(f"  ⚠️  HF value is an empty list/tuple")
            return False
        elif len(hf_tensor) == 1 and isinstance(hf_tensor[0], torch.Tensor):
            hf_tensor = hf_tensor[0]
        elif len(hf_tensor) >= 2 and all(isinstance(item, torch.Tensor) for item in hf_tensor[:2]):
            # For attention input, HF might have (hidden_states, ...), take first
            # For Muse, it's (x, y) where x=y for self-attention, take first
            print(f"  ℹ️  HF value is a tuple with {len(hf_tensor)} items, using first tensor")
            hf_tensor = hf_tensor[0]
        else:
            print(f"  ⚠️  HF value is a list/tuple with {len(hf_tensor)} items:")
            for i, item in enumerate(hf_tensor):
                if isinstance(item, torch.Tensor):
                    print(f"    [{i}]: tensor with shape {item.shape}")
                else:
                    print(f"    [{i}]: {type(item)}")
            print(f"  Cannot compare directly - skipping")
            return False
    
    if isinstance(muse_tensor, (list, tuple)):
        if len(muse_tensor) == 0:
            print(f"  ⚠️  Muse value is an empty list/tuple")
            return False
        elif len(muse_tensor) == 1 and isinstance(muse_tensor[0], torch.Tensor):
            muse_tensor = muse_tensor[0]
        elif len(muse_tensor) >= 2 and all(isinstance(item, torch.Tensor) for item in muse_tensor[:2]):
            # For attention input, Muse has (x, y) where x=y for self-attention, take first
            print(f"  ℹ️  Muse value is a tuple with {len(muse_tensor)} items, using first tensor (x=y for self-attention)")
            muse_tensor = muse_tensor[0]
        else:
            print(f"  ⚠️  Muse value is a list/tuple with {len(muse_tensor)} items:")
            for i, item in enumerate(muse_tensor):
                if isinstance(item, torch.Tensor):
                    print(f"    [{i}]: tensor with shape {item.shape}")
                else:
                    print(f"    [{i}]: {type(item)}")
            print(f"  Cannot compare directly - skipping")
            return False
    
    # Check if both are tensors
    if not isinstance(hf_tensor, torch.Tensor):
        print(f"  ⚠️  HF value is not a tensor: {type(hf_tensor)}")
        return False
    
    if not isinstance(muse_tensor, torch.Tensor):
        print(f"  ⚠️  Muse value is not a tensor: {type(muse_tensor)}")
        return False
    
    # Handle shape differences
    if hf_tensor.shape != muse_tensor.shape:
        print(f"  Shape mismatch:")
        print(f"    HF:   {hf_tensor.shape}")
        print(f"    Muse: {muse_tensor.shape}")
        
        # Try to reshape if possible
        if len(hf_tensor.shape) == len(muse_tensor.shape) == 4:
            # Try transpose for [b, h, s, d] vs [b, s, h, d]
            if (hf_tensor.shape[1] == muse_tensor.shape[2] and 
                hf_tensor.shape[2] == muse_tensor.shape[1]):
                print(f"    Attempting transpose...")
                muse_tensor = muse_tensor.transpose(1, 2)
                print(f"    Muse after transpose: {muse_tensor.shape}")
        
        if hf_tensor.shape != muse_tensor.shape:
            print(f"  ⚠️  Cannot compare due to shape mismatch")
            return False
    
    # Compare values
    hf_tensor = hf_tensor.float()
    muse_tensor = muse_tensor.float()
    
    diff = (hf_tensor - muse_tensor).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Relative difference
    relative_diff = (diff / (hf_tensor.abs() + 1e-8)).max().item()
    
    print(f"  Shape: {hf_tensor.shape}")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Max relative diff: {relative_diff:.6e}")
    
    # Check if close
    is_close = torch.allclose(hf_tensor, muse_tensor, atol=atol, rtol=rtol)
    
    if is_close:
        print(f"  ✓ Match! (atol={atol}, rtol={rtol})")
        return True
    else:
        print(f"  ⚠️  Mismatch!")
        
        # Find position of max diff
        max_diff_idx = diff.argmax()
        max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
        print(f"  Max diff position: {max_diff_pos}")
        print(f"  HF value:   {hf_tensor[max_diff_pos].item():.6f}")
        print(f"  Muse value: {muse_tensor[max_diff_pos].item():.6f}")
        
        # Show sample values
        print(f"  Sample values (first 5 elements):")
        print(f"    HF:   {hf_tensor.flatten()[:5].cpu().numpy()}")
        print(f"    Muse: {muse_tensor.flatten()[:5].cpu().numpy()}")
        
        return False


def compare_dumps(dump_dir):
    """Compare HF and Muse dumps."""
    hf_dir = os.path.join(dump_dir, "hf")
    muse_dir = os.path.join(dump_dir, "muse")
    
    if not os.path.exists(hf_dir):
        print(f"Error: HF directory {hf_dir} does not exist")
        return
    
    if not os.path.exists(muse_dir):
        print(f"Error: Muse directory {muse_dir} does not exist")
        return
    
    print("=" * 80)
    print("HF vs Muse Layer 0 Detailed Comparison")
    print("=" * 80)
    
    # Load metadata
    metadata_path = os.path.join(dump_dir, "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
        print(f"\nInput: {metadata.get('input_info', {}).get('prompt', 'N/A')}")
        print(f"Seq len: {metadata.get('input_info', {}).get('seq_len', 'N/A')}")
    
    # Comparison pairs: (hf_name, muse_name, description)
    comparisons = [
        ("embedding_output", "embedding_output", "Embedding output"),
        ("layer0_input", "layer0_input", "Layer 0 input"),
        ("layer0_input_layernorm_output", "layer0_sa_norm_output", "SA norm output"),
        ("attn0_input", "attn0_input", "Attention module input"),
        ("attn0_q_proj_output", "attn0_q_proj_output", "Q projection output"),
        ("attn0_q_norm_output", "attn0_q_norm_output", "Q norm output"),
        ("attn0_k_proj_output", "attn0_k_proj_output", "K projection output"),
        ("attn0_k_norm_output", "attn0_k_norm_output", "K norm output"),
        ("attn0_v_proj_output", "attn0_v_proj_output", "V projection output"),
        ("attn0_output", "attn0_output", "Attention module output"),
        ("attn0_o_proj_input", "attn0_output_proj_input", "Output projection input"),
        ("attn0_o_proj_output", "attn0_output_proj_output", "Output projection output"),
        ("layer0_post_attention_layernorm_output", "layer0_mlp_norm_output", "MLP norm output"),
        ("layer0_mlp_gate_proj_output", "layer0_mlp_gate_proj_output", "MLP gate projection"),
        ("layer0_mlp_up_proj_output", "layer0_mlp_up_proj_output", "MLP up projection"),
        ("layer0_mlp_down_proj_output", "layer0_mlp_down_proj_output", "MLP down projection"),
        ("layer0_mlp_output", "layer0_mlp_output", "MLP output"),
        ("layer0_output", "layer0_output", "Layer 0 output"),
    ]
    
    matches = 0
    mismatches = 0
    skipped = 0
    
    for hf_name, muse_name, description in comparisons:
        hf_tensor = load_activation(hf_dir, hf_name)
        muse_tensor = load_activation(muse_dir, muse_name)
        
        result = compare_tensors(description, hf_tensor, muse_tensor)
        if result is True:
            matches += 1
        elif result is False:
            mismatches += 1
        else:
            skipped += 1
    
    # Special comparison for attention weights
    print("\n" + "=" * 80)
    print("Attention Weights Comparison")
    print("=" * 80)
    
    hf_attn_weights = load_activation(hf_dir, "attn0_attn_weights")
    if hf_attn_weights is not None:
        print(f"\nHF attention weights:")
        print(f"  Shape: {hf_attn_weights.shape}")
        print(f"  Range: [{hf_attn_weights.min().item():.6f}, {hf_attn_weights.max().item():.6f}]")
        print(f"  Mean: {hf_attn_weights.mean().item():.6f}, Std: {hf_attn_weights.std().item():.6f}")
        print(f"  Sum per row (should be ~1.0): min={hf_attn_weights.sum(dim=-1).min().item():.6f}, max={hf_attn_weights.sum(dim=-1).max().item():.6f}")
    
    # Check if Muse has attention function inputs (q, k, v)
    muse_q = load_activation(muse_dir, "q")
    muse_k = load_activation(muse_dir, "k")
    muse_v = load_activation(muse_dir, "v")
    
    if muse_q is not None and muse_k is not None:
        print(f"\nMuse attention function inputs:")
        print(f"  q shape: {muse_q.shape}")
        print(f"  k shape: {muse_k.shape}")
        print(f"  v shape: {muse_v.shape}")
        
        # Manual computation of attention weights from Muse qkv
        print(f"\nManually computing attention weights from Muse qkv...")
        # This would require the scaling factor and mask, which we can get from metadata
        # For now, just show the shapes
    
    print("\n" + "=" * 80)
    print(f"Summary: {matches} matches, {mismatches} mismatches, {skipped} skipped")
    print("=" * 80)
    
    return matches, mismatches


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_hf_muse_dumps.py <dump_directory>")
        sys.exit(1)
    
    dump_dir = sys.argv[1]
    if not os.path.exists(dump_dir):
        print(f"Error: Directory {dump_dir} does not exist")
        sys.exit(1)
    
    matches, mismatches = compare_dumps(dump_dir)
    
    print(f"\nComparison complete!")
    print(f"  Matches: {matches}")
    print(f"  Mismatches: {mismatches}")

