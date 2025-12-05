"""
Keye Vision Layer 0 Diagnosis: Final Fix V3 (Correct Weight Loading)
====================================================================

修复点：
1. 精心构造带有 `siglip.vision_model.` 前缀的 state_dict，确保 Muse 的转换器能工作。
2. 修复 RoPE 模拟的 Shape 问题。
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import types
from transformers import PretrainedConfig
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.training.common import set_default_dtype

CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

# --- Setup Origin Model Wrapper ---
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
    mod_name = "muse.muse.models.keye_vit.configuration_keye"
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        m.KeyeConfig = HFKeyeConfig
        m.KeyeVisionConfig = HFKeyeVisionConfig
        sys.modules[mod_name] = m

_ensure_origin_ready()
from muse.models.keye_vit import modeling_keye_origin as keye_origin
OriginKeyeVisionModel = keye_origin.SiglipVisionModel

def compare(name, a, b, atol=1e-3):
    a = a.float().cpu().detach()
    b = b.float().cpu().detach()
    # Auto squeeze batch dim if needed
    if a.shape != b.shape:
        if a.shape[0] == 1 and a.dim() == b.dim() + 1: a = a.squeeze(0)
        if b.shape[0] == 1 and b.dim() == a.dim() + 1: b = b.squeeze(0)
    
    if a.shape != b.shape:
        print(f"{name:<30} | ❌ SHAPE: {a.shape} vs {b.shape}")
        return

    diff = (a - b).abs().max().item()
    status = "✅" if diff < atol else "❌"
    print(f"{name:<30} | Diff: {diff:.4e} | {status}")

def run_diagnosis():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print("Loading models...")
    with set_default_dtype(dtype):
        # 强制 Muse 配置 QK Norm 为 False
        muse_cfg = KeyeVisionConfig(use_qk_norm=False)
        muse_model = MuseKeyeVisionModel(muse_cfg).to(device).eval()
        
        origin_cfg = HFKeyeVisionConfig(**muse_cfg.dict())
        origin_model = OriginKeyeVisionModel(origin_cfg).to(device).eval()

    print(f"Loading weights...")
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "module" in raw: raw = raw["module"]
    
    # --- 关键修正：构造 State Dict Key ---
    origin_state = {}
    for k, v in raw.items():
        # 1. 去掉乱七八糟的前缀，保留干净的 vision_model...
        clean = k
        for p in ["module.", "vision_tower.", "siglip."]:
            if clean.startswith(p): clean = clean[len(p):]
        
        if "vision_model" not in clean: clean = "vision_model." + clean
        
        # 2. [FIX] 加上 'siglip.' 前缀，满足 Muse Converter 的正则要求
        siglip_key = "siglip." + clean
        
        origin_state[siglip_key] = v
        # 同时为了 origin_model 加载，也存一份无 siglip 的 (OriginModel 类比较傻，不需要 siglip 前缀，因为它就是 SiglipVisionModel)
        # 但是 load_state_dict 不支持重复引用，我们做两份 dict
    
    # 转换 Muse (输入必须带 siglip.vision_model)
    print("Converting weights for Muse...")
    # 这里调用的是 Muse 模型实例自带的方法，也就是你改过 fc->w 逻辑的那个方法
    muse_state = muse_model.convert_hf_state_dict(origin_state)
    
    # 准备给 Origin 加载的 dict (去掉 siglip. 前缀)
    origin_load_dict = {k.replace("siglip.", ""): v for k, v in origin_state.items()}
    
    # 加载
    print("Loading state dicts...")
    origin_model.load_state_dict(origin_load_dict, strict=False)
    muse_model.load_state_dict(muse_state, strict=False)
    
    print("\n" + "="*50)
    print("Step 0: WEIGHT CHECK (Crucial)")
    print("="*50)
    
    # 1. Check Layer Norm Weights
    ln_o = origin_model.vision_model.encoder.layers[0].layer_norm1
    ln_m = muse_model.encoder.layers[0].sa_norm
    compare("Weight: LayerNorm", ln_o.weight, ln_m.weight, atol=1e-5)
    
    # 2. Check Q Projection Weights
    q_o = origin_model.vision_model.encoder.layers[0].self_attn.q_proj
    q_m = muse_model.encoder.layers[0].attn.q_proj
    compare("Weight: Q Proj", q_o.weight, q_m.weight, atol=1e-5)
    
    if (ln_o.weight - ln_m.weight).abs().max() > 1e-2:
        print("⚠️ 权重依然不匹配！后续计算必然错误，请检查 convert_hf_state_dict 日志。")
    
    print("\n" + "="*50)
    print("Step 1: Component Calculation")
    print("="*50)
    
    # Input
    bs, h, w = 1, 14, 14
    seq = h * w
    # 修正维度：[1, 196, 3, 14, 14]
    pixel_values = torch.randn(bs, seq, 3, h, w, device=device, dtype=dtype)
    grid_thw = [(1, h, w)]
    pos_ids = torch.arange(seq, device=device).unsqueeze(0)
    
    # Get Embeddings Output
    with torch.no_grad():
        x = origin_model.vision_model.embeddings(
            pixel_values, position_ids=pos_ids, image_grid_thw=grid_thw, interpolate_pos_encoding=True
        )
    
    # 1. LN Output
    with torch.no_grad():
        x_o_ln = ln_o(x)
        x_m_ln = ln_m(x)
    compare("Output: LayerNorm", x_o_ln, x_m_ln, atol=1e-3)
    
    x_in = x_o_ln # Force sync
    
    # 2. Projections
    attn_o = origin_model.vision_model.encoder.layers[0].self_attn
    attn_m = muse_model.encoder.layers[0].attn
    
    with torch.no_grad():
        q_o = attn_o.q_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2)
        q_m = attn_m.q_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2)
    compare("Output: Q Proj", q_o, q_m, atol=1e-3)
    
    # 3. RoPE Calculation (Debug)
    h_ids = torch.arange(h, device=device).repeat_interleave(w)
    w_ids = torch.arange(w, device=device).repeat(h)
    
    with torch.no_grad():
        # Origin RoPE
        pids = torch.stack([h_ids, w_ids], dim=-1) # [Seq, 2]
        rope_full = origin_model.vision_model.encoder.rotary_pos_emb(max(h,w)+1) 
        
        # --- DEBUG FIX: Correct Origin Logic ---
        # Origin: rope_emb = rope_full[pids].flatten(1).repeat(1, 2)
        # cos = rope_emb.cos().chunk(2, dim=-1)[0]
        
        rope_indexed = rope_full[pids] # [Seq, 2, 36]
        rope_val = rope_indexed.flatten(1) # [Seq, 72]
        rope_val = rope_val.repeat(1, 2)
        
        # Origin Logic: 它其实是把 [h_freq, w_freq] 变成了 [h_freq, w_freq, h_freq, w_freq]
        # 然后取前半部分，所以 cos 还是 [h_freq, w_freq]
        # 简单来说，cos_o 就是 rope_val.cos()
        
        cos_o = rope_val.cos() 
        sin_o = rope_val.sin()
        rope_full = origin_model.vision_model.encoder.rotary_pos_emb(max(h,w)+1)
        # print(f"\n[Origin RoPE Debug]")
        # 取 h_ids[0] (假设是0) 对应的频率
        # origin_freq_sample = rope_full[0].flatten().cpu().tolist()
        # print(f"Origin Freq (pos=0, first 3): {origin_freq_sample}")
        # Muse RoPE
        rope_mod = muse_model.encoder.rope
        cos_h, sin_h = rope_mod._lookup(rope_mod.height_rope, h_ids)
        cos_w, sin_w = rope_mod._lookup(rope_mod.width_rope, w_ids)
        
        ch1, ch2 = cos_h.chunk(2, dim=-1)
        cw1, cw2 = cos_w.chunk(2, dim=-1)
        cos_m = torch.cat([ch1, cw1, ch2, cw2], dim=-1).squeeze()

        sh1, sh2 = sin_h.chunk(2, dim=-1)
        sw1, sw2 = sin_w.chunk(2, dim=-1)
        sin_m = torch.cat([sh1, sw1, sh2, sw2], dim=-1).squeeze()

        
    compare("Values: RoPE Cos", cos_o, cos_m, atol=1e-4)
    compare("Values: RoPE Sin", sin_o, sin_m, atol=1e-4)
    
    # 4. Apply RoPE (With Safe Reshape)
    with torch.no_grad():
        def apply_rotary(x, cos, sin):
            # x: [B, H, S, D]
            # cos, sin: [S, D]
            B, H, S, D = x.shape
            
            # Ensure shape alignment [1, 1, S, D]
            cos = cos.view(1, 1, S, D)
            sin = sin.view(1, 1, S, D)
            
            x1, x2 = x.chunk(2, dim=-1)
            # Origin (interleaved=False): [-x2, x1]
            return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)
            
        q_o_roped = apply_rotary(q_o, cos_o, sin_o)
        
        # Muse Apply
        input_pos = {"height": h_ids.unsqueeze(0), "width": w_ids.unsqueeze(0)}
        q_m_in = q_m.transpose(1, 2) 
        q_m_roped = rope_mod(q_m_in, input_pos=input_pos).transpose(1, 2)
        
    compare("Output: Q after RoPE", q_o_roped, q_m_roped, atol=1e-2)

if __name__ == "__main__":
    run_diagnosis()