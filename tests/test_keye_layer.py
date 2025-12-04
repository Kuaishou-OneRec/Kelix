"""
Keye Vision Layer 0 Diagnosis: Final (Weight Check + RoPE Fix)
==============================================================

修复内容：
1. [关键] 在 Forward 前显式检查权重是否对齐。
2. 修正 RoPE 模拟代码，正确拼接 H/W 维度 (36+36=72)。
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
    if a.shape != b.shape:
        # Auto squeeze batch dim if needed
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
        muse_cfg = KeyeVisionConfig(use_qk_norm=False)
        muse_model = MuseKeyeVisionModel(muse_cfg).to(device).eval()
        
        origin_cfg = HFKeyeVisionConfig(**muse_cfg.dict())
        origin_model = OriginKeyeVisionModel(origin_cfg).to(device).eval()

    print(f"Loading weights...")
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "module" in raw: raw = raw["module"]
    
    # 手动构建 Origin State Dict
    origin_state = {}
    for k, v in raw.items():
        clean = k
        for p in ["module.", "vision_tower.", "siglip."]:
            if clean.startswith(p): clean = clean[len(p):]
        if "vision_model" not in clean: clean = "vision_model." + clean
        origin_state[clean] = v
    
    # 转换 Muse
    muse_state = muse_model.convert_hf_state_dict(origin_state)
    
    # 加载
    origin_model.load_state_dict(origin_state, strict=False)
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
    
    # 如果上面这步是 ❌，说明权重加载逻辑（convert_hf_state_dict）有问题
    # 如果是 ✅，说明是计算逻辑有问题
    
    print("\n" + "="*50)
    print("Step 1: Component Calculation")
    print("="*50)
    
    # Input
    bs, h, w = 1, 14, 14
    seq = h * w
    pixel_values = torch.randn(bs, seq, 3, h, w, device=device, dtype=dtype)
    grid_thw = [(1, h, w)]
    pos_ids = torch.arange(seq, device=device).unsqueeze(0)
    
    # Get Embeddings Output (Verified Match)
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
    
    # 3. RoPE Calculation (Corrected)
    h_ids = torch.arange(h, device=device).repeat_interleave(w)
    w_ids = torch.arange(w, device=device).repeat(h)
    
    with torch.no_grad():
        # --- Origin RoPE ---
        # Origin logic: cat(height_freq, width_freq)
        pids = torch.stack([h_ids, w_ids], dim=-1) # [Seq, 2]
        rope_full = origin_model.vision_model.encoder.rotary_pos_emb(max(h,w)+1) # [Max, 36]
        
        # Select: [Seq, 2, 36] -> Flatten: [Seq, 72]
        rope_val = rope_full[pids].flatten(1)
        
        # Origin applies repeat(1,2) later for cos/sin construction
        # rope_emb = rope_val.repeat(1, 2) # [Seq, 144]
        # cos_o = rope_emb.cos().chunk(2, dim=-1)[0] # [Seq, 72]
        # 简化逻辑：
        cos_o = rope_val.cos() # [Seq, 72]
        sin_o = rope_val.sin()
        
        # --- Muse RoPE ---
        rope_mod = muse_model.encoder.rope
        cos_h, sin_h = rope_mod._lookup(rope_mod.height_rope, h_ids)
        cos_w, sin_w = rope_mod._lookup(rope_mod.width_rope, w_ids)
        
        cos_m = torch.cat([cos_h, cos_w], dim=-1).squeeze()
        sin_m = torch.cat([sin_h, sin_w], dim=-1).squeeze()
        
    compare("Values: RoPE Cos", cos_o, cos_m, atol=1e-4)
    compare("Values: RoPE Sin", sin_o, sin_m, atol=1e-4)
    
    # 4. Apply RoPE
    with torch.no_grad():
        # 手动执行标准 LLaMA 风格 RoPE (flash_attn 默认非 interleaved)
        # rotate_half: [-x2, x1]
        def apply_rotary(x, cos, sin):
            # x: [B, H, S, D]
            # cos, sin: [S, D]
            cos = cos.view(1, 1, seq, 72)
            sin = sin.view(1, 1, seq, 72)
            x1, x2 = x.chunk(2, dim=-1)
            return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)
            
        q_o_roped = apply_rotary(q_o, cos_o, sin_o)
        
        # Muse Apply
        input_pos = {"height": h_ids.unsqueeze(0), "width": w_ids.unsqueeze(0)}
        q_m_in = q_m.transpose(1, 2) # Muse RoPE expects [B, S, H, D]
        q_m_roped = rope_mod(q_m_in, input_pos=input_pos).transpose(1, 2)
        
    compare("Output: Q after RoPE", q_o_roped, q_m_roped, atol=1e-2)

if __name__ == "__main__":
    run_diagnosis()