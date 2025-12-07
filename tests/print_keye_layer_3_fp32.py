"""
Keye Vision Precision Microscope (Find the 1e-8 source)
=======================================================

目标：定位 Layer 0 中 10^-8 误差的精确来源。
重点：Inv_Freq (频率基), Freqs (角度), Cos/Sin (三角函数值), Rotated Query.
"""

import torch
import numpy as np
import sys
import types
from transformers import PretrainedConfig
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.training.common import set_default_dtype

CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

# --- Config Hack ---
class HFKeyeVisionConfig(PretrainedConfig):
    model_type = "siglip_vision"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items(): setattr(self, k, v)
class HFKeyeConfig(PretrainedConfig):
    model_type = "keye"
    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_config = vision_config
def _ensure_origin_ready():
    mod = "muse.muse.models.keye_vit.configuration_keye"
    if mod in sys.modules: return
    c = types.ModuleType(mod)
    c.KeyeConfig = HFKeyeConfig
    c.KeyeVisionConfig = HFKeyeVisionConfig
    sys.modules[mod] = c
_ensure_origin_ready()
from muse.models.keye_vit import modeling_keye_origin as keye_origin
OriginKeyeVisionModel = keye_origin.SiglipVisionModel 

def compare(name, a, b):
    a = a.float().cpu().detach()
    b = b.float().cpu().detach()
    
    # 自动对齐 Shape
    if a.shape != b.shape:
        if a.numel() == b.numel(): b = b.view_as(a)
        else:
            print(f"{name:<30} | ❌ SHAPE: {a.shape} vs {b.shape}")
            return
    
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 评级
    if max_diff == 0: status = "✅ PERFECT (0.0)"
    elif max_diff < 1e-7: status = "⚠️ TINY (<1e-7)"
    else: status = "❌ DIFF (>1e-7)"
    
    print(f"{name:<30} | {status:<18} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    
    if max_diff > 0 and max_diff < 1e-5:
        # 打印前 3 个有差异的值
        mask = diff > 0
        indices = torch.nonzero(mask)
        if len(indices) > 0:
            idx = indices[0] # 第一个差异点
            val_a = a[tuple(idx)].item()
            val_b = b[tuple(idx)].item()
            print(f"   -> First Diff at {idx.tolist()}: Origin={val_a:.12f}, Muse={val_b:.12f}, Delta={val_a-val_b:.2e}")

def run_microscope():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # FP32 !!!
    
    print(f"Running Microscope in {dtype}...")
    
    # 1. Init Models
    muse_cfg = KeyeVisionConfig(use_qk_norm=False, attention_function="eager")
    origin_cfg = HFKeyeVisionConfig(**muse_cfg.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_cfg).to(device).eval()
        muse_model = MuseKeyeVisionModel(muse_cfg).to(device).eval()

    # (Skip Weight Loading if we just check math logic, but loading ensures dims are correct)
    # 这里我们只关注 RoPE 的算术逻辑，权重其实不影响 RoPE 的生成
    
    # 2. Extract RoPE Modules
    rope_origin = origin_model.vision_model.encoder.rotary_pos_emb
    
    # Muse 的 RoPE 在 encoder.rope (KeyeAxialRotaryEmbedding) -> height_rope / width_rope
    rope_muse_h = muse_model.encoder.rope.height_rope
    
    print("\n" + "="*60)
    print("Microscope Phase 1: Inv Freq (Base Frequencies)")
    print("="*60)
    
    # 获取 Inv Freq
    # Origin
    inv_o = rope_origin.inv_freq # [18]
    # Muse (H和W是一样的配置，取H即可)
    inv_m = rope_muse_h.inv_freq # [18]
    
    compare("Inv Freq", inv_o, inv_m)
    
    print("\n" + "="*60)
    print("Microscope Phase 2: Freqs (Angle = Pos * InvFreq)")
    print("="*60)
    
    # 模拟计算 Freqs
    seq_len = 196
    t = torch.arange(seq_len, device=device, dtype=dtype)
    
    # Origin Logic
    # freqs = outer(t, inv_freq)
    freqs_o = torch.outer(t, inv_o)
    
    # Muse Logic (Inside rotary_embedding.py -> forward or _update_cos_sin_tables)
    # Muse 通常会预计算 cache，我们看看 cache 里的值
    # cache: [max_len, dim] -> 我们取前 seq_len 行
    # 注意：Muse cache 可能是 [cos, sin] 或者 [freqs]，通常是已计算好的 cos/sin
    # 我们手动模拟 Muse 的 freqs 计算逻辑来对比
    freqs_m = torch.outer(t, inv_m) 
    
    compare("Calculated Freqs (Manual)", freqs_o, freqs_m)
    
    print("\n" + "="*60)
    print("Microscope Phase 3: Cos/Sin Values")
    print("="*60)
    
    # Origin Cos
    # emb = cat(freqs, freqs) -> cos = emb.cos()
    emb_o = torch.cat((freqs_o, freqs_o), dim=-1)
    cos_o = emb_o.cos()
    sin_o = emb_o.sin()
    
    # Muse Cos (From Cache)
    # Muse cache shape [max_seq, dim]
    # 我们直接取 cache 里的值（它是预计算好的）
    # Muse 可能是 lazy init，先跑一次 forward 确保 cache 更新
    _ = rope_muse_h(torch.zeros(1, seq_len, 36, device=device)) 
    
    # Muse 的 cache 通常包含 cos 和 sin。具体实现因库而异。
    # 假设 muse.layers.position_embeddings.RotaryPositionalEmbeddings 的 cache 是 [max_len, dim] 的 cos 和 sin
    # 查看源码或属性
    if hasattr(rope_muse_h, 'cos_cached'): # 旧版常见
        cache_cos = rope_muse_h.cos_cached[:seq_len]
        cache_sin = rope_muse_h.sin_cached[:seq_len]
    elif hasattr(rope_muse_h, 'cache'): # 新版常见，可能是 [max_len, dim, 2] (cos, sin)
        # 根据你之前贴的代码: gathered = cache[pos_ids] -> [..., 2]
        # cache 是 [max_len, dim, 2]
        cache_val = rope_muse_h.cache[:seq_len]
        cache_cos = cache_val[..., 0]
        cache_sin = cache_val[..., 1]
    else:
        print("Could not find cache in Muse RoPE module.")
        return

    # Muse cache layout: [H, H] (if dim=36)
    # Origin layout before repeat: [H, H] (emb = cat(freqs, freqs))
    
    compare("Cos Table", cos_o, cache_cos)
    compare("Sin Table", sin_o, cache_sin)
    
    print("\n" + "="*60)
    print("Microscope Phase 4: Full RoPE Construct")
    print("="*60)
    
    # 构造 Origin 完整的 [H, W, H, W]
    # 假设 Grid 为 [14, 14]
    h_ids = torch.arange(14, device=device).repeat_interleave(14)
    w_ids = torch.arange(14, device=device).repeat(14)
    pids = torch.stack([h_ids, w_ids], dim=-1)
    
    # Origin
    rope_full_o = origin_model.vision_model.encoder.rotary_pos_emb(30) # 足够大
    rope_idx_o = rope_full_o[pids].flatten(1) # [196, 36] (H+W Freqs)
    # Origin Repeat: [H, W, H, W]
    rope_final_o = rope_idx_o.repeat(1, 2) 
    cos_final_o = rope_final_o.cos()
    
    # Muse
    # Muse Cache Lookup
    cache_h = rope_muse_h.cache
    cache_w = muse_model.encoder.rope.width_rope.cache
    
    cos_h = cache_h[h_ids, ..., 0] # [196, 36] ([H, H])
    cos_w = cache_w[w_ids, ..., 0] # [196, 36] ([W, W])
    
    # Muse Interleave Logic
    h1, h2 = cos_h.chunk(2, dim=-1)
    w1, w2 = cos_w.chunk(2, dim=-1)
    cos_final_m = torch.cat([h1, w1, h2, w2], dim=-1)
    
    compare("Final RoPE Cos Map", cos_final_o, cos_final_m)

if __name__ == "__main__":
    run_microscope()