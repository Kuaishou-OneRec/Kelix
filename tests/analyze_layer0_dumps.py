"""
Analyze and compare dumped Layer 0 activations from HF and Muse models.

Usage:
    python analyze_layer0_dumps.py <dump_directory> [--compare]
    
If --compare is specified, will compare HF and Muse activations side by side.
"""

import os
import sys
import torch
import numpy as np


def load_dumps(dump_dir, prefix=""):
    """Load all dumped activations."""
    activations = {}
    
    # Try to load metadata
    metadata_path = os.path.join(dump_dir, "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)
    else:
        metadata = None
    
    # Load all .pt files
    for filename in os.listdir(dump_dir):
        if filename.endswith(".pt") and filename != "metadata.pt":
            name = filename[:-3]  # Remove .pt extension
            if prefix:
                name = f"{prefix}_{name}"
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


def compare_hf_muse_activations(dump_dir):
    """Compare HF and Muse activations side by side."""
    hf_dir = os.path.join(dump_dir, "hf")
    muse_dir = os.path.join(dump_dir, "muse")
    
    if not os.path.exists(hf_dir):
        print(f"Error: HF directory {hf_dir} does not exist")
        return
    
    if not os.path.exists(muse_dir):
        print(f"Error: Muse directory {muse_dir} does not exist")
        return
    
    hf_activations, hf_metadata = load_dumps(hf_dir, prefix="hf")
    muse_activations, muse_metadata = load_dumps(muse_dir, prefix="muse")
    
    print("=" * 80)
    print("HF vs Muse Layer 0 Activation Comparison")
    print("=" * 80)
    
    # Mapping between HF and Muse activation names
    name_mapping = {
        # Embedding
        "hf_embedding_output": "muse_embedding_output",
        # Layer 0 input
        "hf_layer0_input": "muse_layer0_input",
        # SA norm
        "hf_layer0_input_layernorm_output": "muse_layer0_sa_norm_output",
        # Attention inputs
        "hf_attn0_input": "muse_attn0_input",
        # QKV projections
        "hf_attn0_q_proj_output": "muse_attn0_q_proj_output",
        "hf_attn0_k_proj_output": "muse_attn0_k_proj_output",
        "hf_attn0_v_proj_output": "muse_attn0_v_proj_output",
        # QK norms
        "hf_attn0_q_norm_output": "muse_attn0_q_norm_output",
        "hf_attn0_k_norm_output": "muse_attn0_k_norm_output",
        # Attention output
        "hf_attn0_output": "muse_attn0_output",
        "hf_attn0_attn_weights": None,  # Will compare with manual computation
        # Output projection
        "hf_attn0_o_proj_input": "muse_attn0_output_proj_input",
        "hf_attn0_o_proj_output": "muse_attn0_output_proj_output",
        # MLP norm
        "hf_layer0_post_attention_layernorm_output": "muse_layer0_mlp_norm_output",
        # MLP components
        "hf_layer0_mlp_gate_proj_output": "muse_layer0_mlp_gate_proj_output",
        "hf_layer0_mlp_up_proj_output": "muse_layer0_mlp_up_proj_output",
        "hf_layer0_mlp_down_proj_output": "muse_layer0_mlp_down_proj_output",
        "hf_layer0_mlp_output": "muse_layer0_mlp_output",
        # Layer output
        "hf_layer0_output": "muse_layer0_output",
    }
    
    print("\nComparing activations:")
    print("-" * 80)
    
    matches = 0
    mismatches = 0
    
    for hf_name, muse_name in name_mapping.items():
        if hf_name not in hf_activations:
            print(f"  ⚠️  {hf_name} not found in HF activations")
            continue
        
        hf_value = hf_activations[hf_name]
        
        if muse_name is None:
            print(f"\n  {hf_name}:")
            print(f"    HF shape: {hf_value.shape if isinstance(hf_value, torch.Tensor) else 'N/A'}")
            print(f"    (No Muse equivalent to compare)")
            continue
        
        if muse_name not in muse_activations:
            print(f"  ⚠️  {muse_name} not found in Muse activations")
            continue
        
        muse_value = muse_activations[muse_name]
        
        print(f"\n  {hf_name} vs {muse_name}:")
        
        if compare_activations(hf_value, muse_value, "HF", "Muse", atol=1e-3):
            matches += 1
        else:
            mismatches += 1
    
    print("\n" + "=" * 80)
    print(f"Summary: {matches} matches, {mismatches} mismatches")
    print("=" * 80)
    
    return hf_activations, muse_activations


def analyze_dumps(dump_dir, compare=False):
    """Analyze dumped activations."""
    # Check if this is a combined dump (has hf/ and muse/ subdirectories)
    hf_dir = os.path.join(dump_dir, "hf")
    muse_dir = os.path.join(dump_dir, "muse")
    
    if compare and os.path.exists(hf_dir) and os.path.exists(muse_dir):
        return compare_hf_muse_activations(dump_dir)
    
    # Single model analysis
    activations, metadata = load_dumps(dump_dir)
    
    print("=" * 80)
    print("Layer 0 Activation Analysis")
    print("=" * 80)
    
    if metadata:
        print("\nMetadata:")
        print(f"  Config: {metadata.get('config', 'N/A')}")
        print(f"  Input info: {metadata.get('input_info', 'N/A')}")
        print(f"  Device: {metadata.get('device', 'N/A')}, dtype: {metadata.get('dtype', 'N/A')}")
    else:
        print("\n⚠️  Metadata not found")
    
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
        print("Usage: python analyze_layer0_dumps.py <dump_directory> [--compare]")
        print("\nOptions:")
        print("  --compare: Compare HF and Muse activations side by side")
        sys.exit(1)
    
    dump_dir = sys.argv[1]
    compare = "--compare" in sys.argv
    
    if not os.path.exists(dump_dir):
        print(f"Error: Directory {dump_dir} does not exist")
        sys.exit(1)
    
    if compare:
        hf_activations, muse_activations = analyze_dumps(dump_dir, compare=True)
        print(f"\nTotal HF activations: {len(hf_activations)}")
        print(f"Total Muse activations: {len(muse_activations)}")
    else:
        activations, metadata = analyze_dumps(dump_dir, compare=False)
        print("\n" + "=" * 80)
        print("Analysis Complete")
        print("=" * 80)
        print(f"\nTotal activations: {len(activations)}")
    
    print(f"Dump directory: {dump_dir}")

