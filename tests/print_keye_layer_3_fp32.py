"""
Keye Vision Precision Microscope (Fixed: theta vs inv_freq)
===========================================================

修复：
1. 适配 Muse RoPE 的属性名 `theta` (Origin 叫 `inv_freq`)。
2. 完整追踪 1e-8 误差的来源。
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
    
    # 评级 (FP32下)
    if max_diff == 0: status = "✅ PERFECT (0.0)"
    elif max_diff < 1e-7: status = "⚠️ TINY (<1e-7)" # 这里的差异通常是 float 精度极限
    else: status = "❌ DIFF (>1e-7)"
    
    print(f"{name:<30} | {status:<18} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    
    if max_diff > 1e-7:
        mask = diff > 0
        indices = torch.nonzero(mask)
        if len(indices) > 0:
            idx = indices[0]
            val_a = a[tuple(idx)].item()
            val_b = b[tuple(idx)].item()
            print(f"   -> First Diff at {idx.tolist()}: Origin={val_a:.20f}, Muse={val_b:.20f}")
            print(f"      Delta={val_a-val_b:.2e}")

def run_microscope():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # FP32 !!!
    
    print(f"Running Microscope in {dtype}...")
    
    muse_cfg = KeyeVisionConfig(use_qk_norm=False, attention_function="eager")
    origin_cfg = HFKeyeVisionConfig(**muse_cfg.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_cfg).to(device).eval()
        muse_model = MuseKeyeVisionModel(muse_cfg).to(device).eval()
    
    # 2. Weights
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "module" in raw: raw = raw["module"]
    origin_state = {}
    for k, v in raw.items():
        clean = k
        for p in ["module.", "vision_tower.", "siglip."]:
            if clean.startswith(p): clean = clean[len(p):]
        if "vision_model" not in clean: clean = "vision_model." + clean
        origin_state["siglip." + clean] = v.to(dtype)
    
    muse_state = muse_model.convert_hf_state_dict(origin_state)
    origin_load = {k.replace("siglip.", ""): v for k, v in origin_state.items()}
    origin_model.load_state_dict(origin_load, strict=False)
    muse_model.load_state_dict(muse_state, strict=False)
    
    # 3. Extract Modules
    rope_origin = origin_model.vision_model.encoder.rotary_pos_emb
    rope_muse_h = muse_model.encoder.rope.height_rope
    
    print("\n" + "="*60)
    print("Microscope Phase 1: Inv Freq (Base Frequencies)")
    print("="*60)
    
    # Origin: inv_freq
    inv_o = rope_origin.inv_freq
    
    # Muse: theta (Fix: use .theta instead of .inv_freq)
    if hasattr(rope_muse_h, "theta"):
        inv_m = rope_muse_h.theta
    elif hasattr(rope_muse_h, "inv_freq"):
        inv_m = rope_muse_h.inv_freq
    else:
        print("❌ Cannot find 'theta' or 'inv_freq' in Muse RoPE.")
        return
    
    compare("Inv Freq", inv_o, inv_m)
    
    print("\n" + "="*60)
    print("Microscope Phase 2: Cache (Cos/Sin Table)")
    print("="*60)
    
    bs, h, w = 1, 14, 14
    seq_len = h * w
    
    # Origin (On-the-fly)
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs_o = torch.outer(t, inv_o)
    emb_o = torch.cat((freqs_o, freqs_o), dim=-1)
    cos_o_raw = emb_o.cos()
    sin_o_raw = emb_o.sin()
    
    # Muse (Cached)
    # Trigger lazy init if needed
    _ = muse_model.encoder.rope(torch.zeros(1, seq_len, 16, 72, device=device), input_pos={"height": torch.zeros(1, device=device), "width": torch.zeros(1, device=device)})

    # Muse Cache: [max_len, dim, 2] -> [..., 0] is cos
    cache_val = rope_muse_h.cache
    cache_slice = cache_val[:seq_len] # [196, 36, 2]
    
    cos_m_raw = cache_slice[..., 0] # [196, 36]
    sin_m_raw = cache_slice[..., 1] # [196, 36]
    
    compare("Cos Table (Raw)", cos_o_raw, cos_m_raw)
    
    print("\n" + "="*60)
    print("Microscope Phase 3: Rotated Query")
    print("="*60)
    
    # Grid
    h_ids = torch.arange(h, device=device).repeat_interleave(w)
    w_ids = torch.arange(w, device=device).repeat(h)
    
    # Input Q
    x_in = torch.randn(1, seq_len, 1152, device=device, dtype=dtype)
    
    # Projections
    q_proj_o = origin_model.vision_model.encoder.layers[0].self_attn.q_proj
    q_proj_m = muse_model.encoder.layers[0].attn.q_proj
    
    with torch.no_grad():
        q_raw_o = q_proj_o(x_in).view(1, seq_len, 16, 72).transpose(1, 2)
        q_raw_m = q_proj_m(x_in).view(1, seq_len, 16, 72).transpose(1, 2)
    
    compare("Q Raw (Pre-RoPE)", q_raw_o, q_raw_m)
    
    # Origin Apply (Manual to guarantee logic)
    pids = torch.stack([h_ids, w_ids], dim=-1)
    rope_full_o = origin_model.vision_model.encoder.rotary_pos_emb(max(h,w)+1)
    rope_val_o = rope_full_o[pids].flatten(1).repeat(1, 2) # [196, 72]
    
    cos_o_final = rope_val_o.cos().view(1, 1, seq_len, 72)
    sin_o_final = rope_val_o.sin().view(1, 1, seq_len, 72)
    
    def apply_rotary_o(q, cos, sin):
        q1, q2 = q.chunk(2, dim=-1)
        return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        
    q_rot_o = apply_rotary_o(q_raw_o, cos_o_final, sin_o_final)
    
    # Muse Apply
    input_pos = {"height": h_ids.unsqueeze(0), "width": w_ids.unsqueeze(0)}
    q_m_in = q_raw_m.transpose(1, 2) 
    # Calling the FIXED KeyeAxialRotaryEmbedding.forward
    q_rot_m = muse_model.encoder.rope(q_m_in, input_pos=input_pos).transpose(1, 2)
    
    compare("Q Rotated", q_rot_o, q_rot_m)
    
    print("\n" + "="*60)
    print("Microscope Phase 4: Attn Score")
    print("="*60)
    
    scale = 72 ** -0.5
    # Self-attention score (using Q as K for simplicity to test accumulation)
    attn_o = torch.matmul(q_rot_o, q_rot_o.transpose(-2, -1)) * scale
    attn_m = torch.matmul(q_rot_m, q_rot_m.transpose(-2, -1)) * scale
    
    compare("Attn Score (Simulated)", attn_o, attn_m)

if __name__ == "__main__":
    run_microscope()