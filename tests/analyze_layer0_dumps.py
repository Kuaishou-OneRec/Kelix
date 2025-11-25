"""
Analyze dumped Layer 0 activations.

Usage:
    python analyze_layer0_dumps.py <dump_directory>
"""

import os
import sys
import torch
import numpy as np


def load_dumps(dump_dir):
    """Load all dumped activations."""
    activations = {}
    metadata = torch.load(os.path.join(dump_dir, "metadata.pt"))
    
    # Load all .pt files
    for filename in os.listdir(dump_dir):
        if filename.endswith(".pt") and filename != "metadata.pt":
            name = filename[:-3]  # Remove .pt extension
            activations[name] = torch.load(os.path.join(dump_dir, filename))
    
    return activations, metadata


def print_activation_info(name, value, indent=0):
    """Print information about an activation."""
    prefix = "  " * indent
    if isinstance(value, torch.Tensor):
        print(f"{prefix}{name}:")
        print(f"{prefix}  shape: {value.shape}")
        print(f"{prefix}  dtype: {value.dtype}")
        print(f"{prefix}  device: {value.device}")
        print(f"{prefix}  range: [{value.min().item():.6f}, {value.max().item():.6f}]")
        print(f"{prefix}  mean: {value.mean().item():.6f}, std: {value.std().item():.6f}")
        if value.numel() <= 20:
            print(f"{prefix}  values: {value.float().cpu().numpy()}")
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: list/tuple with {len(value)} items")
        for i, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                print_activation_info(f"  [{i}]", item, indent + 1)
    elif isinstance(value, dict):
        print(f"{prefix}{name}: dict with keys: {list(value.keys())}")
        for k, v in value.items():
            if isinstance(v, torch.Tensor):
                print_activation_info(f"  {k}", v, indent + 1)
    else:
        print(f"{prefix}{name}: {type(value)} = {value}")


def compare_activations(act1, act2, name1="Act1", name2="Act2", atol=1e-4):
    """Compare two activations."""
    if isinstance(act1, torch.Tensor) and isinstance(act2, torch.Tensor):
        if act1.shape != act2.shape:
            print(f"  Shape mismatch: {name1}={act1.shape}, {name2}={act2.shape}")
            return False
        
        diff = (act1 - act2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  {name1} vs {name2}:")
        print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        
        if max_diff < atol:
            print(f"    ✓ Match!")
            return True
        else:
            print(f"    ⚠️  Mismatch!")
            # Find position of max diff
            max_diff_idx = diff.argmax()
            max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
            print(f"    Max diff position: {max_diff_pos}")
            print(f"    {name1} value: {act1[max_diff_pos].item():.6f}")
            print(f"    {name2} value: {act2[max_diff_pos].item():.6f}")
            return False
    else:
        print(f"  Cannot compare: {name1}={type(act1)}, {name2}={type(act2)}")
        return False


def analyze_dumps(dump_dir):
    """Analyze dumped activations."""
    activations, metadata = load_dumps(dump_dir)
    
    print("=" * 80)
    print("Layer 0 Activation Analysis")
    print("=" * 80)
    
    print("\nMetadata:")
    print(f"  Config: {metadata['config']}")
    print(f"  Input info: {metadata['input_info']}")
    print(f"  Device: {metadata['device']}, dtype: {metadata['dtype']}")
    
    print("\n" + "=" * 80)
    print("Activation Flow (in order):")
    print("=" * 80)
    
    # Print activations in order
    order = [
        "embedding",
        "layer0",
        "layer0_sa_norm",
        "attn0",
        "attn0_q_proj",
        "attn0_q_norm",
        "attn0_rope_q",
        "attn0_k_proj",
        "attn0_k_norm",
        "attn0_rope_k",
        "attn0_v_proj",
        "q",  # attention function input
        "k",
        "v",
        "output",  # attention function output
        "attn0_output_proj",
        "attn0",  # attention module output
        "layer0_mlp_norm",
        "layer0_mlp_gate_proj",
        "layer0_mlp_up_proj",
        "layer0_mlp_down_proj",
        "layer0_mlp",
        "layer0",  # layer output
    ]
    
    for name in order:
        # Check for input/output variants
        for suffix in ["_input", "_output", ""]:
            full_name = name + suffix
            if full_name in activations:
                print(f"\n{full_name}:")
                print_activation_info("", activations[full_name], indent=1)
                break
    
    # Print remaining activations
    printed = set()
    for name in order:
        for suffix in ["_input", "_output", ""]:
            printed.add(name + suffix)
    
    remaining = [name for name in activations.keys() if name not in printed]
    if remaining:
        print("\n" + "=" * 80)
        print("Other Activations:")
        print("=" * 80)
        for name in sorted(remaining):
            print(f"\n{name}:")
            print_activation_info("", activations[name], indent=1)
    
    return activations, metadata


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_layer0_dumps.py <dump_directory>")
        sys.exit(1)
    
    dump_dir = sys.argv[1]
    if not os.path.exists(dump_dir):
        print(f"Error: Directory {dump_dir} does not exist")
        sys.exit(1)
    
    activations, metadata = analyze_dumps(dump_dir)
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print(f"\nTotal activations: {len(activations)}")
    print(f"Dump directory: {dump_dir}")

