"""
Debug Logits Difference Between Muse and HF Models
===================================================
权重一致但输出不同，定位问题出在哪里：
1. 检查 inputs 是否一致
2. 检查 logits 差异的分布
"""

import os
import sys
import torch
import numpy as np

# =========================================================================
# Configuration
# =========================================================================

# 两个脚本保存的 logits 路径
MUSE_LOGITS_PATH = "/llm_reco/maosiyang/muse_model_logits_video.pt"
HF_LOGITS_PATH = "/llm_reco/maosiyang/hf_logits_video_new.pt"

# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 80)
    print("Debug: Comparing Muse vs HF Logits")
    print("=" * 80)
    
    # 1. 加载两个 logits
    print(f"\nLoading Muse logits from: {MUSE_LOGITS_PATH}")
    muse_logits = torch.load(MUSE_LOGITS_PATH, map_location="cpu")
    print(f"  Shape: {muse_logits.shape}, Dtype: {muse_logits.dtype}")
    
    print(f"\nLoading HF logits from: {HF_LOGITS_PATH}")
    hf_logits = torch.load(HF_LOGITS_PATH, map_location="cpu")
    print(f"  Shape: {hf_logits.shape}, Dtype: {hf_logits.dtype}")
    
    # 2. 检查 shape 是否一致
    print("\n" + "=" * 80)
    print("Shape Comparison")
    print("=" * 80)
    
    if muse_logits.shape != hf_logits.shape:
        print(f"\n❌ Shape mismatch!")
        print(f"   Muse: {muse_logits.shape}")
        print(f"   HF:   {hf_logits.shape}")
        print("\n   This means the input sequence lengths are different!")
        print("   Check if both models use the same processor and inputs.")
        return
    else:
        print(f"\n✅ Shapes match: {muse_logits.shape}")
    
    # 3. 转换为 float32 进行比较
    muse_f32 = muse_logits.float()
    hf_f32 = hf_logits.float()
    
    # 4. 计算差异
    print("\n" + "=" * 80)
    print("Overall Difference Statistics")
    print("=" * 80)
    
    diff = muse_f32 - hf_f32
    abs_diff = diff.abs()
    
    print(f"\n  Max absolute diff:  {abs_diff.max().item():.6e}")
    print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
    print(f"  Std of diff:        {diff.std().item():.6e}")
    
    # 相对误差
    eps = 1e-8
    rel_diff = abs_diff / (hf_f32.abs() + eps)
    print(f"\n  Max relative diff:  {rel_diff.max().item():.6e}")
    print(f"  Mean relative diff: {rel_diff.mean().item():.6e}")
    
    # Cosine similarity
    muse_flat = muse_f32.view(-1)
    hf_flat = hf_f32.view(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        muse_flat.unsqueeze(0), hf_flat.unsqueeze(0)
    ).item()
    print(f"\n  Cosine similarity:  {cos_sim:.8f}")
    
    # 5. 按 token 位置分析差异
    print("\n" + "=" * 80)
    print("Per-Token Difference Analysis")
    print("=" * 80)
    
    seq_len = muse_logits.shape[1]
    per_token_max_diff = abs_diff[0].max(dim=-1).values  # [seq_len]
    per_token_mean_diff = abs_diff[0].mean(dim=-1)  # [seq_len]
    
    # 找出差异最大的 token 位置
    top_k = min(10, seq_len)
    top_diff_indices = per_token_max_diff.topk(top_k).indices
    
    print(f"\nTop {top_k} tokens with largest max diff:")
    for idx in top_diff_indices:
        print(f"  Token {idx.item():4d}: max_diff={per_token_max_diff[idx].item():.6e}, mean_diff={per_token_mean_diff[idx].item():.6e}")
    
    # 6. 检查前几个 token 和后几个 token 的差异
    print("\n" + "=" * 80)
    print("First & Last Tokens Analysis")
    print("=" * 80)
    
    print("\nFirst 10 tokens max diff:")
    for i in range(min(10, seq_len)):
        print(f"  Token {i:4d}: max_diff={per_token_max_diff[i].item():.6e}")
    
    print("\nLast 10 tokens max diff:")
    for i in range(max(0, seq_len - 10), seq_len):
        print(f"  Token {i:4d}: max_diff={per_token_max_diff[i].item():.6e}")
    
    # 7. 检查 logits 值本身
    print("\n" + "=" * 80)
    print("Logits Value Comparison (First Token)")
    print("=" * 80)
    
    print("\nFirst token, first 10 dims:")
    print(f"  Muse: {muse_f32[0, 0, :10].numpy()}")
    print(f"  HF:   {hf_f32[0, 0, :10].numpy()}")
    print(f"  Diff: {diff[0, 0, :10].numpy()}")
    
    print("\nFirst token, top-5 predicted tokens:")
    muse_top5 = muse_f32[0, 0].topk(5)
    hf_top5 = hf_f32[0, 0].topk(5)
    print(f"  Muse top5 indices: {muse_top5.indices.tolist()}")
    print(f"  HF   top5 indices: {hf_top5.indices.tolist()}")
    
    # 8. 检查最后一个 token（通常是生成的第一个 token）
    print("\n" + "=" * 80)
    print("Logits Value Comparison (Last Token)")
    print("=" * 80)
    
    print("\nLast token, first 10 dims:")
    print(f"  Muse: {muse_f32[0, -1, :10].numpy()}")
    print(f"  HF:   {hf_f32[0, -1, :10].numpy()}")
    print(f"  Diff: {diff[0, -1, :10].numpy()}")
    
    print("\nLast token, top-5 predicted tokens:")
    muse_top5 = muse_f32[0, -1].topk(5)
    hf_top5 = hf_f32[0, -1].topk(5)
    print(f"  Muse top5 indices: {muse_top5.indices.tolist()}")
    print(f"  HF   top5 indices: {hf_top5.indices.tolist()}")
    
    # 9. 总结
    print("\n" + "=" * 80)
    print("Summary & Next Steps")
    print("=" * 80)
    
    if cos_sim > 0.999:
        print("\n✅ Logits are very similar (cosine > 0.999)")
        print("   Small differences might be due to numerical precision (bf16 vs fp16).")
    elif cos_sim > 0.99:
        print("\n⚠️ Logits are roughly similar (cosine > 0.99)")
        print("   Check attention implementation differences.")
    else:
        print("\n❌ Logits have significant differences (cosine <= 0.99)")
        print("\nPossible causes:")
        print("  1. Input preprocessing differences (check input_ids, pixel_values)")
        print("  2. Position embedding / RoPE implementation differences")
        print("  3. Attention mask handling differences")
        print("  4. Visual tokenizer forward differences")
        print("\nNext steps:")
        print("  1. Compare input_ids between two scripts")
        print("  2. Add hooks to compare intermediate layer outputs")
        print("  3. Check if video frames are processed the same way")


if __name__ == "__main__":
    main()

