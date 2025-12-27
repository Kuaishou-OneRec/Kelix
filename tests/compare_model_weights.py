"""
Compare Model Weights Between Two Directories
==============================================
用于比较 Muse 版本和 HF 版本模型权重是否一致
"""

import os
import sys
import torch
import tqdm
from pathlib import Path
from safetensors.torch import load_file
from collections import defaultdict

# =========================================================================
# Configuration - 修改这两个路径为你要比较的模型目录
# =========================================================================

# Muse 版本模型目录
MUSE_MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"

# HF 版本模型目录
HF_MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_video_baseline"

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
        # 转换为相同 dtype 进行比较
        t1 = t1.float()
        t2 = t2.float()
    
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 检查是否完全相等
    if torch.equal(t1, t2):
        return {
            "match": True,
            "reason": "exact match",
            "max_diff": 0.0,
            "mean_diff": 0.0,
        }
    
    # 检查是否在容差范围内相等
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


def find_key_mapping(muse_keys: set, hf_keys: set):
    """尝试找出 key 的映射关系"""
    # 常见的 key 映射模式
    # Muse: model.layers.0.xxx -> HF: model.layers.0.xxx (可能相同)
    # 或者有前缀差异
    
    mappings = {}
    muse_only = set()
    hf_only = set()
    
    for mk in muse_keys:
        if mk in hf_keys:
            mappings[mk] = mk
        else:
            muse_only.add(mk)
    
    for hk in hf_keys:
        if hk not in muse_keys:
            hf_only.add(hk)
    
    return mappings, muse_only, hf_only


def main():
    print("=" * 80)
    print("Model Weights Comparison")
    print("=" * 80)
    print(f"\nMuse Model: {MUSE_MODEL_DIR}")
    print(f"HF Model:   {HF_MODEL_DIR}")
    print()
    
    # 1. 加载权重
    print("Loading Muse model weights...")
    muse_sd = load_safetensors_from_dir(MUSE_MODEL_DIR)
    print(f"  -> {len(muse_sd)} keys loaded\n")
    
    print("Loading HF model weights...")
    hf_sd = load_safetensors_from_dir(HF_MODEL_DIR)
    print(f"  -> {len(hf_sd)} keys loaded\n")
    
    # 2. 比较 keys
    muse_keys = set(muse_sd.keys())
    hf_keys = set(hf_sd.keys())
    
    print("=" * 80)
    print("Key Analysis")
    print("=" * 80)
    
    common_keys = muse_keys & hf_keys
    muse_only_keys = muse_keys - hf_keys
    hf_only_keys = hf_keys - muse_keys
    
    print(f"\nCommon keys: {len(common_keys)}")
    print(f"Muse-only keys: {len(muse_only_keys)}")
    print(f"HF-only keys: {len(hf_only_keys)}")
    
    if muse_only_keys:
        print(f"\n⚠️ Keys only in Muse model (first 20):")
        for k in sorted(muse_only_keys)[:20]:
            print(f"  - {k}")
        if len(muse_only_keys) > 20:
            print(f"  ... and {len(muse_only_keys) - 20} more")
    
    if hf_only_keys:
        print(f"\n⚠️ Keys only in HF model (first 20):")
        for k in sorted(hf_only_keys)[:20]:
            print(f"  - {k}")
        if len(hf_only_keys) > 20:
            print(f"  ... and {len(hf_only_keys) - 20} more")
    
    # 3. 比较 common keys 的值
    if not common_keys:
        print("\n❌ No common keys found! The models have completely different key names.")
        print("This might indicate different model architectures or naming conventions.")
        return
    
    print("\n" + "=" * 80)
    print("Value Comparison (Common Keys)")
    print("=" * 80)
    
    match_count = 0
    mismatch_count = 0
    mismatches = []
    
    for key in tqdm.tqdm(sorted(common_keys), desc="Comparing tensors"):
        result = compare_tensors(muse_sd[key], hf_sd[key], key)
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
    
    # 4. 总结
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if mismatch_count == 0 and len(muse_only_keys) == 0 and len(hf_only_keys) == 0:
        print("\n✅ Models are IDENTICAL! All keys and values match.")
    elif mismatch_count == 0:
        print(f"\n⚠️ Common keys match, but there are extra keys:")
        print(f"   - Muse-only: {len(muse_only_keys)}")
        print(f"   - HF-only: {len(hf_only_keys)}")
    else:
        print(f"\n❌ Models have DIFFERENCES:")
        print(f"   - Mismatched values: {mismatch_count}")
        print(f"   - Muse-only keys: {len(muse_only_keys)}")
        print(f"   - HF-only keys: {len(hf_only_keys)}")
    
    # 5. 计算总体差异统计
    if common_keys and mismatch_count > 0:
        print("\n" + "=" * 80)
        print("Detailed Diff Statistics")
        print("=" * 80)
        
        all_max_diffs = []
        all_mean_diffs = []
        
        for key in common_keys:
            t1, t2 = muse_sd[key].float(), hf_sd[key].float()
            if t1.shape == t2.shape:
                diff = (t1 - t2).abs()
                all_max_diffs.append(diff.max().item())
                all_mean_diffs.append(diff.mean().item())
        
        if all_max_diffs:
            print(f"\nAcross all {len(all_max_diffs)} comparable tensors:")
            print(f"  Overall max diff:  {max(all_max_diffs):.6e}")
            print(f"  Overall mean diff: {sum(all_mean_diffs)/len(all_mean_diffs):.6e}")


if __name__ == "__main__":
    main()

