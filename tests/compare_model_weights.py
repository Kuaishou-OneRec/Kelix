"""
Compare Model Weights Between Two Directories
==============================================
用于比较 Muse 版本和 HF 版本模型权重是否一致
使用 convert_hf_state_dict 的映射逻辑进行 key 转换后比较
"""

import os
import sys
import torch
import tqdm
import logging
from pathlib import Path
from safetensors.torch import load_file
from collections import defaultdict

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =========================================================================
# Configuration - 修改这两个路径为你要比较的模型目录
# =========================================================================

# Muse 版本模型目录（已经是 Muse 格式的 key）
MUSE_MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image"

# HF 版本模型目录（需要转换 key）
HF_MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_video_baseline"

# 是否 tie_word_embeddings（影响 lm_head 是否被跳过）
TIE_WORD_EMBEDDINGS = True

# =========================================================================
# Key Conversion (from convert_hf_state_dict)
# =========================================================================

def convert_hf_key_to_muse(hf_key: str, tie_word_embeddings: bool = True) -> str:
    """
    将 HF 格式的 key 转换为 Muse 格式的 key
    基于 KeyeTokenizerEnd2EndVideo.convert_hf_state_dict 的逻辑
    
    Returns:
        转换后的 Muse key，如果应该跳过则返回 None
    """
    # ============ Visual Tokenizer ============
    if hf_key.startswith("visual_tokenizer."):
        rest_key = hf_key[len("visual_tokenizer."):]
        
        # visual.vision_model.* -> visual_tokenizer.visual.*
        if rest_key.startswith("visual.vision_model."):
            vision_rest = rest_key[len("visual.vision_model."):]
            
            # Skip pooling head
            if vision_rest.startswith("head."):
                return None
            
            # Handle embeddings
            if vision_rest.startswith("embeddings."):
                return f"visual_tokenizer.visual.{vision_rest}"
            
            # Handle post_layernorm -> ln_post
            if vision_rest.startswith("post_layernorm."):
                suffix = vision_rest.replace("post_layernorm.", "")
                return f"visual_tokenizer.visual.ln_post.{suffix}"
            
            # Handle encoder layers
            if vision_rest.startswith("encoder.layers."):
                parts = vision_rest.split(".", 3)
                if len(parts) >= 4:
                    layer_idx = parts[2]
                    layer_rest = parts[3]
                    
                    if layer_rest.startswith("layer_norm1."):
                        suffix = layer_rest.replace("layer_norm1.", "")
                        return f"visual_tokenizer.visual.encoder.layers.{layer_idx}.sa_norm.{suffix}"
                    
                    if layer_rest.startswith("layer_norm2."):
                        suffix = layer_rest.replace("layer_norm2.", "")
                        return f"visual_tokenizer.visual.encoder.layers.{layer_idx}.mlp_norm.{suffix}"
                    
                    if layer_rest.startswith("self_attn."):
                        attn_key = layer_rest.replace("self_attn.", "attn.")
                        attn_key = attn_key.replace("out_proj.", "output_proj.")
                        return f"visual_tokenizer.visual.encoder.layers.{layer_idx}.{attn_key}"
                    
                    if layer_rest.startswith("mlp."):
                        new_rest_key = layer_rest.replace("fc1", "w1").replace("fc2", "w2")
                        return f"visual_tokenizer.visual.encoder.layers.{layer_idx}.{new_rest_key}"
            
            return None  # Skip unknown vision keys
        
        # mlp_AR, pre_llm_aligner, encoder, quantizer - direct mapping
        return f"visual_tokenizer.{rest_key}"
    
    # ============ Quant Projector ============
    if hf_key.startswith("quant_projector."):
        return hf_key  # Direct mapping
    
    # ============ LLM Model ============
    # Skip lm_head if tie_word_embeddings is True
    if tie_word_embeddings and hf_key == "lm_head.weight":
        return None
    
    # Handle embedding layer
    if hf_key == "model.embed_tokens.weight":
        return "model.model.tok_embeddings.weight"
    
    # Handle final norm
    if hf_key == "model.norm.weight":
        return "model.model.norm.scale"
    
    # Handle lm_head (when not tied)
    if hf_key == "lm_head.weight":
        return "model.model.output.weight"
    
    # Handle transformer layers
    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".", 3)
        if len(parts) < 4:
            return None
        
        layer_idx = parts[2]
        rest_key = parts[3]
        
        # Handle attention
        if rest_key.startswith("self_attn."):
            attn_key = rest_key.replace("self_attn.", "attn.")
            attn_key = attn_key.replace("o_proj", "output_proj")
            attn_key = attn_key.replace("q_norm.weight", "q_norm.scale")
            attn_key = attn_key.replace("k_norm.weight", "k_norm.scale")
            return f"model.model.layers.{layer_idx}.{attn_key}"
        
        # Handle MLP
        if rest_key.startswith("mlp."):
            mlp_key = rest_key.replace("mlp.", "")
            if mlp_key == "gate_proj.weight":
                return f"model.model.layers.{layer_idx}.mlp.w1.weight"
            elif mlp_key == "up_proj.weight":
                return f"model.model.layers.{layer_idx}.mlp.w3.weight"
            elif mlp_key == "down_proj.weight":
                return f"model.model.layers.{layer_idx}.mlp.w2.weight"
            return None
        
        # Handle layer norms
        if rest_key == "input_layernorm.weight":
            return f"model.model.layers.{layer_idx}.sa_norm.scale"
        
        if rest_key == "post_attention_layernorm.weight":
            return f"model.model.layers.{layer_idx}.mlp_norm.scale"
    
    return None  # Skip unknown keys


def convert_hf_state_dict_keys(hf_state_dict: dict, tie_word_embeddings: bool = True) -> tuple:
    """
    将 HF state dict 的所有 key 转换为 Muse 格式
    
    Returns:
        (converted_dict, skipped_keys)
    """
    converted = {}
    skipped = []
    
    for hf_key, tensor in hf_state_dict.items():
        muse_key = convert_hf_key_to_muse(hf_key, tie_word_embeddings)
        if muse_key is not None:
            converted[muse_key] = tensor
        else:
            skipped.append(hf_key)
    
    return converted, skipped


# =========================================================================
# Helper Functions
# =========================================================================

def load_safetensors_from_dir(model_dir: str) -> dict:
    """加载目录下所有 safetensors 文件"""
    sd = {}
    safetensor_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    print(f"Found {len(safetensor_files)} safetensors files in {model_dir}")
    for f in tqdm.tqdm(safetensor_files, desc=f"Loading from {Path(model_dir).name}"):
        sd.update(load_file(os.path.join(model_dir, f)))
    return sd


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name: str, rtol: float = 1e-5, atol: float = 1e-8):
    """比较两个 tensor 是否相等"""
    if t1.shape != t2.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: {t1.shape} vs {t2.shape}",
            "max_diff": None,
            "mean_diff": None,
        }
    
    if t1.dtype != t2.dtype:
        t1 = t1.float()
        t2 = t2.float()
    
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    if torch.equal(t1, t2):
        return {
            "match": True,
            "reason": "exact match",
            "max_diff": 0.0,
            "mean_diff": 0.0,
        }
    
    if torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol):
        return {
            "match": True,
            "reason": f"close match (max_diff={max_diff:.2e})",
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }
    
    return {
        "match": False,
        "reason": f"values differ (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})",
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


def main():
    print("=" * 80)
    print("Model Weights Comparison (with Key Conversion)")
    print("=" * 80)
    print(f"\nMuse Model: {MUSE_MODEL_DIR}")
    print(f"HF Model:   {HF_MODEL_DIR}")
    print(f"Tie Word Embeddings: {TIE_WORD_EMBEDDINGS}")
    print()
    
    # 1. 加载权重
    print("Loading Muse model weights...")
    muse_sd = load_safetensors_from_dir(MUSE_MODEL_DIR)
    print(f"  -> {len(muse_sd)} keys loaded\n")
    
    print("Loading HF model weights...")
    hf_sd_raw = load_safetensors_from_dir(HF_MODEL_DIR)
    print(f"  -> {len(hf_sd_raw)} keys loaded\n")
    
    # 2. 转换 HF keys 到 Muse 格式
    print("=" * 80)
    print("Converting HF Keys to Muse Format")
    print("=" * 80)
    
    hf_sd_converted, skipped_hf_keys = convert_hf_state_dict_keys(hf_sd_raw, TIE_WORD_EMBEDDINGS)
    print(f"\n✅ Converted: {len(hf_sd_converted)} keys")
    print(f"⏭️  Skipped:   {len(skipped_hf_keys)} keys")
    
    if skipped_hf_keys:
        print(f"\nSkipped HF keys (first 10):")
        for k in skipped_hf_keys[:10]:
            print(f"  - {k}")
        if len(skipped_hf_keys) > 10:
            print(f"  ... and {len(skipped_hf_keys) - 10} more")
    
    # 3. 比较 keys
    muse_keys = set(muse_sd.keys())
    hf_converted_keys = set(hf_sd_converted.keys())
    
    print("\n" + "=" * 80)
    print("Key Analysis (After Conversion)")
    print("=" * 80)
    
    common_keys = muse_keys & hf_converted_keys
    muse_only_keys = muse_keys - hf_converted_keys
    hf_only_keys = hf_converted_keys - muse_keys
    
    print(f"\nCommon keys: {len(common_keys)}")
    print(f"Muse-only keys: {len(muse_only_keys)}")
    print(f"HF-only keys (after conversion): {len(hf_only_keys)}")
    
    if muse_only_keys:
        print(f"\n⚠️ Keys only in Muse model (first 20):")
        for k in sorted(muse_only_keys)[:20]:
            print(f"  - {k}")
        if len(muse_only_keys) > 20:
            print(f"  ... and {len(muse_only_keys) - 20} more")
    
    if hf_only_keys:
        print(f"\n⚠️ Keys only in HF model after conversion (first 20):")
        for k in sorted(hf_only_keys)[:20]:
            print(f"  - {k}")
        if len(hf_only_keys) > 20:
            print(f"  ... and {len(hf_only_keys) - 20} more")
    
    # 4. 比较 common keys 的值
    if not common_keys:
        print("\n❌ No common keys found after conversion!")
        print("The key conversion might be incomplete or models have different architectures.")
        return
    
    print("\n" + "=" * 80)
    print("Value Comparison (Common Keys)")
    print("=" * 80)
    
    match_count = 0
    mismatch_count = 0
    mismatches = []
    
    for key in tqdm.tqdm(sorted(common_keys), desc="Comparing tensors"):
        result = compare_tensors(muse_sd[key], hf_sd_converted[key], key)
        if result["match"]:
            match_count += 1
        else:
            mismatch_count += 1
            mismatches.append((key, result))
    
    print(f"\n✅ Matching keys: {match_count}/{len(common_keys)}")
    print(f"❌ Mismatching keys: {mismatch_count}/{len(common_keys)}")
    
    if mismatches:
        print(f"\n⚠️ Mismatched keys (first 30):")
        for key, result in mismatches[:30]:
            print(f"  - {key}")
            print(f"      Reason: {result['reason']}")
            if result['max_diff'] is not None:
                print(f"      Max diff: {result['max_diff']:.6e}, Mean diff: {result['mean_diff']:.6e}")
        if len(mismatches) > 30:
            print(f"  ... and {len(mismatches) - 30} more mismatches")
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if mismatch_count == 0 and len(muse_only_keys) == 0 and len(hf_only_keys) == 0:
        print("\n✅ Models are IDENTICAL! All keys and values match after conversion.")
    elif mismatch_count == 0:
        print(f"\n⚠️ All {len(common_keys)} common keys match!")
        if muse_only_keys or hf_only_keys:
            print(f"   But there are extra keys:")
            print(f"   - Muse-only: {len(muse_only_keys)}")
            print(f"   - HF-only (after conversion): {len(hf_only_keys)}")
    else:
        print(f"\n❌ Models have VALUE DIFFERENCES:")
        print(f"   - Mismatched values: {mismatch_count}/{len(common_keys)}")
        print(f"   - Muse-only keys: {len(muse_only_keys)}")
        print(f"   - HF-only keys: {len(hf_only_keys)}")
    
    # 6. 计算总体差异统计
    print("\n" + "=" * 80)
    print("Detailed Diff Statistics")
    print("=" * 80)
    
    all_max_diffs = []
    all_mean_diffs = []
    
    for key in common_keys:
        t1, t2 = muse_sd[key].float(), hf_sd_converted[key].float()
        if t1.shape == t2.shape:
            diff = (t1 - t2).abs()
            all_max_diffs.append(diff.max().item())
            all_mean_diffs.append(diff.mean().item())
    
    if all_max_diffs:
        print(f"\nAcross all {len(all_max_diffs)} comparable tensors:")
        print(f"  Overall max diff:  {max(all_max_diffs):.6e}")
        print(f"  Overall mean diff: {sum(all_mean_diffs)/len(all_mean_diffs):.6e}")
        
        # 统计完全匹配的比例
        exact_match = sum(1 for d in all_max_diffs if d == 0.0)
        close_match = sum(1 for d in all_max_diffs if d < 1e-6)
        print(f"\n  Exact matches (diff=0): {exact_match}/{len(all_max_diffs)}")
        print(f"  Close matches (diff<1e-6): {close_match}/{len(all_max_diffs)}")


if __name__ == "__main__":
    main()

