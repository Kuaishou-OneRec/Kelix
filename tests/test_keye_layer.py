"""
Keye Vision Layer 0 Diagnosis: Component-wise Check
===================================================

此脚本手动提取 Layer 0 的权重和输入，在脚本中重现 RoPE 和 Attention 的计算过程，
以精确定位 Muse 和 Origin 在哪一步开始出现差异。
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import types
from transformers import PretrainedConfig
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.training.common import set_default_dtype

# 请替换为你的 Checkpoint 路径
CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

# --- Setup Origin Model ---
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
    diff = (a - b).abs().max().item()
    status = "✅" if diff < atol else "❌"
    print(f"{name:<30} | Diff: {diff:.4e} | {status}")
    if diff >= atol:
        print(f"  Origin sample: {a.flatten()[:5].numpy()}")
        print(f"  Muse   sample: {b.flatten()[:5].numpy()}")

def manual_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def run_diagnosis():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # 1. Load Models & Weights
    print("Loading models...")
    with set_default_dtype(dtype):
        # 强制 Muse 配置 QK Norm 为 False (虽然默认就是 False)
        muse_cfg = KeyeVisionConfig(use_qk_norm=False)
        muse_model = MuseKeyeVisionModel(muse_cfg).to(device).eval()
        
        origin_cfg = HFKeyeVisionConfig(**muse_cfg.dict())
        origin_model = OriginKeyeVisionModel(origin_cfg).to(device).eval()

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "module" in raw: raw = raw["module"]
    
    # Simple loader
    origin_state = {}
    for k, v in raw.items():
        clean = k.replace("module.", "").replace("vision_tower.", "").replace("siglip.", "")
        if "vision_model" not in clean: clean = "vision_model." + clean
        origin_state[clean] = v
    
    muse_state = muse_model.convert_hf_state_dict(origin_state)
    origin_model.load_state_dict(origin_state, strict=False)
    muse_model.load_state_dict(muse_state, strict=False)
    
    # 2. Extract Components of Layer 0
    print("\nStarting Component-wise Diagnosis for Layer 0...")
    
    # Generate common input (Embeddings output)
    bs, h, w = 1, 14, 14
    seq = h * w
    pixel_values = torch.randn(bs, 3, h*14, w*14, device=device, dtype=dtype).unsqueeze(1) # 5D
    # Mocking standard inputs
    grid_thw = [(1, h, w)]
    pos_ids = torch.arange(seq, device=device).unsqueeze(0)
    
    with torch.no_grad():
        # Get standardized input x
        x = origin_model.vision_model.embeddings(
            pixel_values, position_ids=pos_ids, image_grid_thw=grid_thw, interpolate_pos_encoding=True
        )
        
    # --- Step 1: Layer Norm (Pre-Norm) ---
    origin_ln = origin_model.vision_model.encoder.layers[0].layer_norm1
    muse_ln = muse_model.encoder.layers[0].sa_norm
    
    with torch.no_grad():
        x_origin_ln = origin_ln(x)
        x_muse_ln = muse_ln(x)
    
    compare("1. Layer Norm", x_origin_ln, x_muse_ln, atol=1e-3)
    
    # Use Origin's LN output for next steps to isolate errors
    x_in = x_origin_ln
    
    # --- Step 2: Q/K/V Projections ---
    origin_attn = origin_model.vision_model.encoder.layers[0].self_attn
    muse_attn = muse_model.encoder.layers[0].attn
    
    with torch.no_grad():
        # Origin Proj
        q_o = origin_attn.q_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2) # [B, H, S, D]
        k_o = origin_attn.k_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2)
        v_o = origin_attn.v_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2)
        
        # Muse Proj
        q_m = muse_attn.q_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2)
        k_m = muse_attn.k_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2)
        v_m = muse_attn.v_proj(x_in).view(bs, seq, 16, 72).transpose(1, 2)
        
    compare("2. Q Projection", q_o, q_m, atol=1e-3)
    compare("3. K Projection", k_o, k_m, atol=1e-3)
    compare("4. V Projection", v_o, v_m, atol=1e-3)
    
    # --- Step 3: RoPE Calculation (Cos/Sin) ---
    # Construct Grid
    h_ids = torch.arange(h, device=device).repeat_interleave(w)
    w_ids = torch.arange(w, device=device).repeat(h)
    
    with torch.no_grad():
        # Origin RoPE Cos/Sin
        pids = torch.stack([h_ids, w_ids], dim=-1)
        # Origin logic simulation
        rope_emb_full = origin_model.vision_model.encoder.rotary_pos_emb(30) # get enough freqs
        rope_emb = rope_emb_full[pids].flatten(1).repeat(1, 2)
        cos_o = rope_emb.cos().chunk(2, dim=-1)[0] # [Seq, 72]
        sin_o = rope_emb.sin().chunk(2, dim=-1)[0]
        
        # Muse RoPE Cos/Sin
        # Muse logic simulation from KeyeAxialRotaryEmbedding
        rope_mod = muse_model.encoder.rope
        cos_h, sin_h = rope_mod._lookup(rope_mod.height_rope, h_ids)
        cos_w, sin_w = rope_mod._lookup(rope_mod.width_rope, w_ids)
        cos_m = torch.cat([cos_h, cos_w], dim=-1).squeeze() # [Seq, 72]
        sin_m = torch.cat([sin_h, sin_w], dim=-1).squeeze()
        
    compare("5. RoPE Cos", cos_o, cos_m, atol=1e-4)
    compare("6. RoPE Sin", sin_o, sin_m, atol=1e-4)
    
    # --- Step 4: Apply RoPE ---
    with torch.no_grad():
        # Origin Apply
        # Usually flash_attn apply_rotary_emb(q, cos, sin)
        # We manually simulate standard rotate_half logic to see if Origin matches Muse
        # Origin's apply_rotary_emb (interleaved=False default) does:
        # x1, x2 = x.chunk(2) -> return x1*cos - x2*sin, x2*cos + x1*sin
        # Muse's manual logic:
        # return x*cos + rotate_half(x)*sin
        # rotate_half(x) = cat(-x2, x1)
        # = (x1, x2)*cos + (-x2, x1)*sin
        # = (x1*cos - x2*sin, x2*cos + x1*sin)
        # MATHEMATICALLY THEY ARE THE SAME.
        
        # Let's verify standard math vs Origin output
        from flash_attn.layers.rotary import apply_rotary_emb
        # q_o is [B, H, S, D]. cos_o is [S, D] -> need unsqueeze for broadcast
        q_o_roped = apply_rotary_emb(
            q_o.transpose(1, 2), # Expects [B, S, H, D]
            cos_o.to(dtype=dtype), 
            sin_o.to(dtype=dtype)
        ).transpose(1, 2)
        
        # Muse Apply
        input_pos = {"height": h_ids.unsqueeze(0), "width": w_ids.unsqueeze(0)}
        # Must transpose q_m to [B, S, H, D] for Muse RoPE
        q_m_in = q_m.transpose(1, 2)
        q_m_roped = rope_mod(q_m_in, input_pos=input_pos).transpose(1, 2)
        
    compare("7. Q after RoPE", q_o_roped, q_m_roped, atol=1e-2)
    
    # --- Step 5: Attention (Manual Eager) ---
    # To bypass flash_attn differences, let's test pure math
    with torch.no_grad():
        from torch.nn.functional import scaled_dot_product_attention
        
        # Origin uses self.scale = head_dim**-0.5
        # Muse implicitly uses same scale
        
        # Origin
        out_o = scaled_dot_product_attention(q_o_roped, k_o, v_o, dropout_p=0.0)
        # Muse (using q_m_roped which ideally equals q_o_roped)
        out_m = scaled_dot_product_attention(q_m_roped, k_m, v_m, dropout_p=0.0)
        
    compare("8. Attn Output (Pre-Proj)", out_o, out_m, atol=1e-2)
    
    # --- Step 6: Output Projection ---
    with torch.no_grad():
        # Flatten [B, H, S, D] -> [B, S, H*D]
        out_o_flat = out_o.transpose(1, 2).reshape(bs, seq, -1)
        out_m_flat = out_m.transpose(1, 2).reshape(bs, seq, -1)
        
        res_o = origin_attn.out_proj(out_o_flat)
        res_m = muse_attn.output_proj(out_m_flat)
        
    compare("9. Final Layer Output", res_o, res_m, atol=1e-2)

if __name__ == "__main__":
    run_diagnosis()