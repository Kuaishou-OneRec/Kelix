#!/usr/bin/env python3
"""
Gradient Comparison Script
==========================
Purpose: Compare gradients saved from two different repositories/frameworks.

This script:
1. Loads gradient files from both repositories
2. Matches parameter names between frameworks
3. Computes and reports gradient differences
4. Identifies parameters with significant discrepancies

Usage:
    python compare_gradients.py \
        --grad1 /tmp/gradients_msy_master_2.pt \
        --grad2 /tmp/gradients_end2end.pt \
        --output comparison_report.txt
"""

import os
import argparse
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_gradients(path: str) -> dict:
    """Load gradient file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gradient file not found: {path}")
    return torch.load(path, map_location="cpu")


def normalize_param_name(name: str, framework: str) -> str:
    """
    Normalize parameter names to enable matching between frameworks.
    
    Converts msy_master_2/muse naming to end2end/muse (HuggingFace) naming.
    Based on KeyeTokenizerEnd2EndVideo.convert_hf_state_dict()
    """
    normalized = name
    
    # Remove common prefixes
    prefixes_to_remove = ["module.", "_orig_mod."]
    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    
    # If it's msy_master_2 format, convert to HF format
    if framework == "msy_master_2/muse":
        # ============ LLM Model conversions ============
        # model.model.tok_embeddings.weight -> model.embed_tokens.weight
        if normalized == "model.model.tok_embeddings.weight":
            return "model.embed_tokens.weight"
        
        # model.model.norm.scale -> model.norm.weight
        if normalized == "model.model.norm.scale":
            return "model.norm.weight"
        
        # model.model.output.weight -> lm_head.weight
        if normalized == "model.model.output.weight":
            return "lm_head.weight"
        
        # model.model.layers.{i}.* -> model.layers.{i}.*
        if normalized.startswith("model.model.layers."):
            rest = normalized[len("model.model.layers."):]
            parts = rest.split(".", 1)
            if len(parts) == 2:
                layer_idx, layer_rest = parts
                
                # Attention: attn.* -> self_attn.*
                if layer_rest.startswith("attn."):
                    attn_rest = layer_rest[len("attn."):]
                    # output_proj -> o_proj
                    attn_rest = attn_rest.replace("output_proj", "o_proj")
                    # q_norm.scale -> q_norm.weight, k_norm.scale -> k_norm.weight
                    attn_rest = attn_rest.replace(".scale", ".weight")
                    return f"model.layers.{layer_idx}.self_attn.{attn_rest}"
                
                # MLP: w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj
                if layer_rest.startswith("mlp."):
                    mlp_rest = layer_rest[len("mlp."):]
                    mlp_rest = mlp_rest.replace("w1.", "gate_proj.")
                    mlp_rest = mlp_rest.replace("w2.", "down_proj.")
                    mlp_rest = mlp_rest.replace("w3.", "up_proj.")
                    return f"model.layers.{layer_idx}.mlp.{mlp_rest}"
                
                # LayerNorms: sa_norm.scale -> input_layernorm.weight
                if layer_rest == "sa_norm.scale":
                    return f"model.layers.{layer_idx}.input_layernorm.weight"
                
                # mlp_norm.scale -> post_attention_layernorm.weight
                if layer_rest == "mlp_norm.scale":
                    return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        
        # ============ Visual Tokenizer conversions ============
        if normalized.startswith("visual_tokenizer.visual."):
            vis_rest = normalized[len("visual_tokenizer.visual."):]
            
            # embeddings.* -> visual.vision_model.embeddings.*
            if vis_rest.startswith("embeddings."):
                return f"visual_tokenizer.visual.vision_model.{vis_rest}"
            
            # ln_post.* -> visual.vision_model.post_layernorm.*
            if vis_rest.startswith("ln_post."):
                suffix = vis_rest[len("ln_post."):]
                return f"visual_tokenizer.visual.vision_model.post_layernorm.{suffix}"
            
            # encoder.layers.{i}.* -> visual.vision_model.encoder.layers.{i}.*
            if vis_rest.startswith("encoder.layers."):
                enc_parts = vis_rest.split(".", 3)  # ["encoder", "layers", "{i}", "rest"]
                if len(enc_parts) >= 4:
                    layer_idx = enc_parts[2]
                    layer_rest = enc_parts[3]
                    
                    # sa_norm.* -> layer_norm1.*
                    if layer_rest.startswith("sa_norm."):
                        suffix = layer_rest[len("sa_norm."):]
                        return f"visual_tokenizer.visual.vision_model.encoder.layers.{layer_idx}.layer_norm1.{suffix}"
                    
                    # mlp_norm.* -> layer_norm2.*
                    if layer_rest.startswith("mlp_norm."):
                        suffix = layer_rest[len("mlp_norm."):]
                        return f"visual_tokenizer.visual.vision_model.encoder.layers.{layer_idx}.layer_norm2.{suffix}"
                    
                    # attn.* -> self_attn.*
                    if layer_rest.startswith("attn."):
                        attn_rest = layer_rest[len("attn."):]
                        attn_rest = attn_rest.replace("output_proj", "out_proj")
                        return f"visual_tokenizer.visual.vision_model.encoder.layers.{layer_idx}.self_attn.{attn_rest}"
                    
                    # mlp.w1.* -> mlp.fc1.*, mlp.w2.* -> mlp.fc2.*
                    if layer_rest.startswith("mlp."):
                        mlp_rest = layer_rest[len("mlp."):]
                        mlp_rest = mlp_rest.replace("w1.", "fc1.")
                        mlp_rest = mlp_rest.replace("w2.", "fc2.")
                        return f"visual_tokenizer.visual.vision_model.encoder.layers.{layer_idx}.mlp.{mlp_rest}"
    
    return normalized


def find_matching_params(
    grads1: Dict[str, dict],
    grads2: Dict[str, dict],
    framework1: str = "framework1",
    framework2: str = "framework2"
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Find matching parameters between two gradient dictionaries.
    
    Returns:
        matched_pairs: List of (name1, name2) tuples for matching parameters
        unmatched1: Parameters only in grads1
        unmatched2: Parameters only in grads2
    """
    # Normalize names
    norm_to_orig1 = {normalize_param_name(k, framework1): k for k in grads1.keys()}
    norm_to_orig2 = {normalize_param_name(k, framework2): k for k in grads2.keys()}
    
    norm_names1 = set(norm_to_orig1.keys())
    norm_names2 = set(norm_to_orig2.keys())
    
    # Find matches
    matched_norm = norm_names1 & norm_names2
    unmatched_norm1 = norm_names1 - norm_names2
    unmatched_norm2 = norm_names2 - norm_names1
    
    # Convert back to original names
    matched_pairs = [(norm_to_orig1[n], norm_to_orig2[n]) for n in sorted(matched_norm)]
    unmatched1 = [norm_to_orig1[n] for n in sorted(unmatched_norm1)]
    unmatched2 = [norm_to_orig2[n] for n in sorted(unmatched_norm2)]
    
    return matched_pairs, unmatched1, unmatched2


def compute_gradient_difference(
    stats1: dict,
    stats2: dict,
    tensor1: Optional[torch.Tensor] = None,
    tensor2: Optional[torch.Tensor] = None
) -> dict:
    """
    Compute various difference metrics between two gradient statistics.
    """
    diff = {}
    
    # Check if both have gradients
    if stats1.get("no_grad", False) or stats2.get("no_grad", False):
        diff["status"] = "no_grad"
        return diff
    
    if stats1.get("mean") is None or stats2.get("mean") is None:
        diff["status"] = "missing_grad"
        return diff
    
    diff["status"] = "ok"
    
    # Scalar statistics differences
    for key in ["mean", "std", "max", "min", "norm", "abs_mean"]:
        v1 = stats1.get(key, 0)
        v2 = stats2.get(key, 0)
        diff[f"{key}_diff"] = abs(v1 - v2)
        diff[f"{key}_1"] = v1
        diff[f"{key}_2"] = v2
        
        # Relative difference (avoid division by zero)
        max_val = max(abs(v1), abs(v2), 1e-10)
        diff[f"{key}_rel_diff"] = abs(v1 - v2) / max_val
    
    # Shape check
    diff["shape_match"] = stats1.get("shape") == stats2.get("shape")
    diff["shape_1"] = stats1.get("shape")
    diff["shape_2"] = stats2.get("shape")
    
    # If full tensors are available, compute more detailed comparisons
    if tensor1 is not None and tensor2 is not None:
        if tensor1.shape == tensor2.shape:
            # Element-wise differences
            abs_diff = (tensor1.float() - tensor2.float()).abs()
            diff["tensor_max_diff"] = abs_diff.max().item()
            diff["tensor_mean_diff"] = abs_diff.mean().item()
            diff["tensor_std_diff"] = abs_diff.std().item()
            
            # Cosine similarity
            flat1 = tensor1.flatten().float()
            flat2 = tensor2.flatten().float()
            cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
            diff["cosine_similarity"] = cos_sim
            
            # Correlation
            if flat1.std() > 1e-10 and flat2.std() > 1e-10:
                corr = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1].item()
                diff["correlation"] = corr
        else:
            diff["tensor_comparison"] = "shape_mismatch"
    
    return diff


def import_torch_nn_functional():
    """Import F for cosine similarity."""
    import torch.nn.functional as F
    return F

# Import F for compute_gradient_difference
import torch.nn.functional as F


def generate_report(
    data1: dict,
    data2: dict,
    matched_pairs: List[Tuple[str, str]],
    unmatched1: List[str],
    unmatched2: List[str],
    differences: Dict[str, dict],
    output_path: Optional[str] = None
) -> str:
    """Generate a human-readable comparison report."""
    
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("GRADIENT COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Metadata comparison
    meta1 = data1.get("metadata", {})
    meta2 = data2.get("metadata", {})
    
    lines.append("## Metadata Comparison")
    lines.append("-" * 40)
    lines.append(f"Framework 1: {meta1.get('framework', 'unknown')}")
    lines.append(f"Framework 2: {meta2.get('framework', 'unknown')}")
    lines.append(f"Model 1: {meta1.get('model_class', 'unknown')}")
    lines.append(f"Model 2: {meta2.get('model_class', 'unknown')}")
    lines.append(f"Seed 1: {meta1.get('seed', 'unknown')}")
    lines.append(f"Seed 2: {meta2.get('seed', 'unknown')}")
    lines.append(f"Total Loss 1: {meta1.get('total_loss', 'N/A')}")
    lines.append(f"Total Loss 2: {meta2.get('total_loss', 'N/A')}")
    lines.append(f"LM Loss 1: {meta1.get('lm_loss', 'N/A')}")
    lines.append(f"LM Loss 2: {meta2.get('lm_loss', 'N/A')}")
    lines.append("")
    
    # Input stats comparison
    input1 = data1.get("input_stats", {})
    input2 = data2.get("input_stats", {})
    lines.append("## Input Statistics Comparison")
    lines.append("-" * 40)
    lines.append(f"input_ids_sum 1: {input1.get('input_ids_sum', 'N/A')}")
    lines.append(f"input_ids_sum 2: {input2.get('input_ids_sum', 'N/A')}")
    lines.append(f"pixel_values_mean 1: {input1.get('pixel_values_mean', 'N/A')}")
    lines.append(f"pixel_values_mean 2: {input2.get('pixel_values_mean', 'N/A')}")
    lines.append(f"pixel_values_std 1: {input1.get('pixel_values_std', 'N/A')}")
    lines.append(f"pixel_values_std 2: {input2.get('pixel_values_std', 'N/A')}")
    lines.append("")
    
    # Summary
    lines.append("## Parameter Matching Summary")
    lines.append("-" * 40)
    lines.append(f"Matched parameters: {len(matched_pairs)}")
    lines.append(f"Unmatched in Framework 1: {len(unmatched1)}")
    lines.append(f"Unmatched in Framework 2: {len(unmatched2)}")
    lines.append("")
    
    # Unmatched parameters
    if unmatched1:
        lines.append("### Unmatched Parameters (Framework 1 only)")
        for name in unmatched1[:20]:  # Limit to first 20
            lines.append(f"  - {name}")
        if len(unmatched1) > 20:
            lines.append(f"  ... and {len(unmatched1) - 20} more")
        lines.append("")
    
    if unmatched2:
        lines.append("### Unmatched Parameters (Framework 2 only)")
        for name in unmatched2[:20]:
            lines.append(f"  - {name}")
        if len(unmatched2) > 20:
            lines.append(f"  ... and {len(unmatched2) - 20} more")
        lines.append("")
    
    # Gradient differences - sorted by largest difference
    lines.append("## Gradient Differences")
    lines.append("-" * 40)
    
    # Filter valid comparisons
    valid_diffs = [(k, v) for k, v in differences.items() if v.get("status") == "ok"]
    
    # Sort by norm relative difference
    sorted_diffs = sorted(valid_diffs, key=lambda x: x[1].get("norm_rel_diff", 0), reverse=True)
    
    # Categorize by severity
    critical = []  # rel_diff > 0.1 (10%)
    warning = []   # rel_diff > 0.01 (1%)
    ok = []        # rel_diff <= 0.01
    
    for name, diff in sorted_diffs:
        rel_diff = diff.get("norm_rel_diff", 0)
        if rel_diff > 0.1:
            critical.append((name, diff))
        elif rel_diff > 0.01:
            warning.append((name, diff))
        else:
            ok.append((name, diff))
    
    lines.append(f"\n### Summary: Critical={len(critical)}, Warning={len(warning)}, OK={len(ok)}")
    lines.append("")
    
    # Critical differences
    if critical:
        lines.append("### CRITICAL DIFFERENCES (>10% relative difference)")
        lines.append("-" * 40)
        for name, diff in critical[:30]:
            lines.append(f"\n{name}:")
            lines.append(f"  Shape: {diff.get('shape_1')} vs {diff.get('shape_2')} (match: {diff.get('shape_match')})")
            lines.append(f"  Norm: {diff.get('norm_1', 0):.6e} vs {diff.get('norm_2', 0):.6e} (diff: {diff.get('norm_diff', 0):.6e}, rel: {diff.get('norm_rel_diff', 0):.2%})")
            lines.append(f"  Mean: {diff.get('mean_1', 0):.6e} vs {diff.get('mean_2', 0):.6e} (diff: {diff.get('mean_diff', 0):.6e})")
            lines.append(f"  Std:  {diff.get('std_1', 0):.6e} vs {diff.get('std_2', 0):.6e}")
            if "cosine_similarity" in diff:
                lines.append(f"  Cosine Sim: {diff['cosine_similarity']:.6f}")
        if len(critical) > 30:
            lines.append(f"\n... and {len(critical) - 30} more critical parameters")
        lines.append("")
    
    # Warning differences
    if warning:
        lines.append("### WARNING DIFFERENCES (1-10% relative difference)")
        lines.append("-" * 40)
        for name, diff in warning[:20]:
            lines.append(f"\n{name}:")
            lines.append(f"  Norm: {diff.get('norm_1', 0):.6e} vs {diff.get('norm_2', 0):.6e} (rel: {diff.get('norm_rel_diff', 0):.2%})")
        if len(warning) > 20:
            lines.append(f"\n... and {len(warning) - 20} more warning parameters")
        lines.append("")
    
    # Summary of OK parameters
    lines.append(f"### OK Parameters (<1% relative difference): {len(ok)}")
    lines.append("")
    
    # Overall assessment
    lines.append("=" * 80)
    lines.append("## OVERALL ASSESSMENT")
    lines.append("=" * 80)
    
    if len(critical) == 0 and len(warning) == 0:
        lines.append("✅ ALL GRADIENTS MATCH WELL!")
        lines.append("The two frameworks produce essentially identical gradients.")
    elif len(critical) == 0:
        lines.append("⚠️  MINOR DIFFERENCES DETECTED")
        lines.append(f"Found {len(warning)} parameters with 1-10% relative difference.")
        lines.append("This may be due to numerical precision differences.")
    else:
        lines.append("❌ SIGNIFICANT DIFFERENCES DETECTED!")
        lines.append(f"Found {len(critical)} parameters with >10% relative difference.")
        lines.append("The two frameworks may have different implementations.")
        lines.append("\nRecommendation: Investigate the critical parameters listed above.")
    
    lines.append("")
    
    report = "\n".join(lines)
    
    # Save to file if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare gradients from two frameworks")
    parser.add_argument("--grad1", type=str, required=True,
                        help="Path to first gradient file (e.g., msy_master_2)")
    parser.add_argument("--grad2", type=str, required=True,
                        help="Path to second gradient file (e.g., end2end)")
    parser.add_argument("--output", type=str, default="gradient_comparison_report.txt",
                        help="Path to save comparison report")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gradient Comparison Tool")
    print("=" * 60)
    print(f"File 1: {args.grad1}")
    print(f"File 2: {args.grad2}")
    print(f"Output: {args.output}")
    print("")
    
    # Load gradient files
    print("Loading gradient files...")
    data1 = load_gradients(args.grad1)
    data2 = load_gradients(args.grad2)
    
    grads1 = data1.get("gradients", {})
    grads2 = data2.get("gradients", {})
    
    print(f"  Framework 1: {data1.get('metadata', {}).get('framework', 'unknown')}")
    print(f"  Framework 2: {data2.get('metadata', {}).get('framework', 'unknown')}")
    print(f"  Parameters 1: {len(grads1)}")
    print(f"  Parameters 2: {len(grads2)}")
    print("")
    
    # Match parameters
    print("Matching parameters...")
    framework1 = data1.get("metadata", {}).get("framework", "framework1")
    framework2 = data2.get("metadata", {}).get("framework", "framework2")
    
    matched_pairs, unmatched1, unmatched2 = find_matching_params(
        grads1, grads2, framework1, framework2
    )
    
    print(f"  Matched: {len(matched_pairs)}")
    print(f"  Unmatched in 1: {len(unmatched1)}")
    print(f"  Unmatched in 2: {len(unmatched2)}")
    print("")
    
    # Compute differences
    print("Computing gradient differences...")
    differences = {}
    
    for name1, name2 in matched_pairs:
        stats1 = grads1[name1]
        stats2 = grads2[name2]
        
        # Get full tensors if available
        tensor1 = stats1.get("grad_tensor")
        tensor2 = stats2.get("grad_tensor")
        
        diff = compute_gradient_difference(stats1, stats2, tensor1, tensor2)
        differences[name1] = diff
    
    # Generate report
    print("Generating report...")
    report = generate_report(
        data1, data2,
        matched_pairs, unmatched1, unmatched2,
        differences,
        args.output
    )
    
    # Print report to console
    print("\n" + report)
    
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()

