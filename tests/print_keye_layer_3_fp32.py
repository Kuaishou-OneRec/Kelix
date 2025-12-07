"""
Keye Vision Precision Microscope V2 (On-the-fly Logic Verification)
===================================================================

目标：验证新的 KeyeAxialRotaryEmbedding (FP32 On-the-fly) 是否消除了 1e-8 误差。
注意：不再检查 .cache，因为新模型已弃用它。
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
    if a.shape != b.shape:
        if a.numel() == b.numel(): b = b.view_as(a)
        else:
            print(f"{name:<30} | ❌ SHAPE: {a.shape} vs {b.shape}")
            return
    diff = (a - b).abs()
    max_diff = diff.max().item()
    status = "✅ PERFECT" if max_diff == 0 else ("⚠️ TINY" if max_diff < 1e-7 else "❌ DIFF")
    print(f"{name:<30} | {status:<15} | Max: {max_diff:.2e}")

def run_microscope():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # FP32 Verification
    
    print(f"Running Microscope V2 in {dtype}...")
    
    muse_cfg = KeyeVisionConfig(use_qk_norm=False, attention_function="eager")
    origin_cfg = HFKeyeVisionConfig(**muse_cfg.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_cfg).to(device).eval()
        muse_model = MuseKeyeVisionModel(muse_cfg).to(device).eval()
    
    # Load Weights
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
    
    # Extract Modules
    rope_origin = origin_model.vision_model.encoder.rotary_pos_emb
    rope_muse = muse_model.encoder.rope
    
    print("\n" + "="*60)
    print("Microscope Phase 1: Inv Freq (Base Theta)")
    print("="*60)
    inv_o = rope_origin.inv_freq
    inv_m = rope_muse.height_rope.theta
    compare("Inv Freq", inv_o, inv_m)
    
    print("\n" + "="*60)
    print("Microscope Phase 2: On-the-fly Calculation (FP32)")
    print("="*60)
    # 模拟输入 Grid
    bs, h, w = 1, 14, 14
    seq_len = h * w
    h_ids = torch.arange(h, device=device).repeat_interleave(w)
    w_ids = torch.arange(w, device=device).repeat(h)
    
    # --- Origin Calculation ---
    pids = torch.stack([h_ids, w_ids], dim=-1)
    # Origin 内部也是 On-the-fly: outer(seq, inv_freq)
    # 我们直接调它的 forward 拿到 freqs [max_len, 36]
    freqs_full_o = rope_origin(max(h,w)+1)
    freqs_o = freqs_full_o[pids].flatten(1) # [Seq, 36]
    
    # 构造完整的 Cos [Seq, 72] -> [H, W, H, W]
    emb_o = freqs_o.repeat(1, 2) 
    cos_o = emb_o.cos()
    sin_o = emb_o.sin()
    
    # --- Muse Calculation (New Method) ---
    # 调用新的 _compute_freqs 方法 (FP32)
    freqs_h_m = rope_muse._compute_freqs(rope_muse.height_rope.theta, h_ids).squeeze()
    freqs_w_m = rope_muse._compute_freqs(rope_muse.width_rope.theta, w_ids).squeeze()
    
    # 拼接 [H, W]
    freqs_m = torch.cat([freqs_h_m, freqs_w_m], dim=-1)
    
    # 构造完整的 Cos [H, W, H, W]
    emb_m = torch.cat([freqs_m, freqs_m], dim=-1)
    cos_m = emb_m.cos()
    sin_m = emb_m.sin()
    
    # 对比 Frequencies (角度值)
    compare("Frequencies (Angle)", freqs_o, freqs_m)
    # 对比 Cos/Sin (最终值)
    compare("Cos Table (On-Fly)", cos_o, cos_m)
    compare("Sin Table (On-Fly)", sin_o, sin_m)

    print("\n" + "="*60)
    print("Microscope Phase 3: Rotated Query")
    print("="*60)
    
    x_in = torch.randn(1, seq_len, 1152, device=device, dtype=dtype)
    q_proj_o = origin_model.vision_model.encoder.layers[0].self_attn.q_proj
    q_proj_m = muse_model.encoder.layers[0].attn.q_proj
    
    with torch.no_grad():
        q_raw_o = q_proj_o(x_in).view(1, seq_len, 16, 72).transpose(1, 2)
        q_raw_m = q_proj_m(x_in).view(1, seq_len, 16, 72).transpose(1, 2)

    # Origin Apply
    def apply_rotary_o(q, cos, sin):
        cos = cos.view(1, 1, seq_len, 72)
        sin = sin.view(1, 1, seq_len, 72)
        q1, q2 = q.chunk(2, dim=-1)
        return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    
    q_rot_o = apply_rotary_o(q_raw_o, cos_o, sin_o)
    
    # Muse Apply (Using actual forward)
    input_pos = {"height": h_ids.unsqueeze(0), "width": w_ids.unsqueeze(0)}
    q_m_in = q_raw_m.transpose(1, 2)
    q_rot_m = rope_muse(q_m_in, input_pos=input_pos).transpose(1, 2)
    
    compare("Q Rotated", q_rot_o, q_rot_m)
    
    print("\n" + "="*60)
    print("Microscope Phase 4: Attn Score")
    print("="*60)
    
    scale = 72 ** -0.5
    attn_o = torch.matmul(q_rot_o, q_rot_o.transpose(-2, -1)) * scale
    attn_m = torch.matmul(q_rot_m, q_rot_m.transpose(-2, -1)) * scale
    
    compare("Attn Score", attn_o, attn_m)

if __name__ == "__main__":
    run_microscope()