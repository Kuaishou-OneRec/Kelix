"""
Keye Vision Precision Microscope (Find the 1e-8 source)
=======================================================

目标：定位 Layer 0 中 10^-8 误差的精确来源。
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

# --- Setup Wrappers ---
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
        if a.numel() == b.numel(): b = b.view_as(a) # 尝试 reshape 对齐
        else:
            print(f"{name:<25} | ❌ SHAPE: {a.shape} vs {b.shape}")
            return
    
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 定义“完美”为 0 或 极小
    status = "✅ PERFECT" if max_diff == 0 else ( "⚠️ TINY" if max_diff < 1e-6 else "❌ DIFF")
    
    print(f"{name:<25} | {status} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")

def run_microscope():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # 必须用 FP32
    
    print(f"Running Microscope in {dtype}...")
    
    # 1. Init
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
    
    print("\n=== 1. RoPE 基础频率 (inv_freq) ===")
    # Origin 的 inv_freq
    # 路径: vision_model.encoder.rotary_pos_emb.inv_freq
    inv_o = origin_model.vision_model.encoder.rotary_pos_emb.inv_freq
    
    # Muse 的 inv_freq (Height 和 Width 应该是同一套参数)
    # 路径: encoder.rope.height_rope.inv_freq
    inv_m = muse_model.encoder.rope.height_rope.inv_freq
    
    compare("Inv Freq", inv_o, inv_m)
    
    print("\n=== 2. RoPE 三角函数表 (Cos/Sin) ===")
    bs, h, w = 1, 14, 14
    seq = h * w
    h_ids = torch.arange(h, device=device).repeat_interleave(w)
    w_ids = torch.arange(w, device=device).repeat(h)
    
    # --- Origin Cos/Sin ---
    pids = torch.stack([h_ids, w_ids], dim=-1)
    rope_full_o = origin_model.vision_model.encoder.rotary_pos_emb(max(h,w)+1)
    rope_val_o = rope_full_o[pids].flatten(1)
    # Origin Logic: Repeat to [H, W, H, W]
    rope_emb_o = rope_val_o.repeat(1, 2)
    cos_o = rope_emb_o.cos()
    
    # --- Muse Cos/Sin ---
    rope_mod = muse_model.encoder.rope
    cos_h, _ = rope_mod._lookup(rope_mod.height_rope, h_ids)
    cos_w, _ = rope_mod._lookup(rope_mod.width_rope, w_ids)
    # Muse Logic: Chunk & Interleave -> [H, W, H, W]
    ch1, ch2 = cos_h.chunk(2, dim=-1)
    cw1, cw2 = cos_w.chunk(2, dim=-1)
    cos_m = torch.cat([ch1, cw1, ch2, cw2], dim=-1).squeeze()
    
    compare("Cos Values", cos_o, cos_m)

    print("\n=== 3. 旋转后的 Q (Rotated Query) ===")
    # 构造相同的 Q 输入
    # 使用随机数，因为我们已经验证过 Q_Proj 是匹配的
    x_in = torch.randn(1, seq, 1152, device=device, dtype=dtype)
    
    # 获取 Q (通过已加载的权重)
    q_proj_o = origin_model.vision_model.encoder.layers[0].self_attn.q_proj
    q_proj_m = muse_model.encoder.layers[0].attn.q_proj
    
    with torch.no_grad():
        q_raw_o = q_proj_o(x_in).view(1, seq, 16, 72).transpose(1, 2)
        q_raw_m = q_proj_m(x_in).view(1, seq, 16, 72).transpose(1, 2)
    
    # 验证投影是否完美
    compare("Q Raw (Pre-RoPE)", q_raw_o, q_raw_m)
    
    # 应用 RoPE
    # Origin Apply (Manual FP32 Sim)
    sin_o = rope_emb_o.sin()
    def apply_origin_style(q, cos, sin):
        cos = cos.view(1, 1, seq, 72)
        sin = sin.view(1, 1, seq, 72)
        q1, q2 = q.chunk(2, dim=-1)
        return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    
    q_rot_o = apply_origin_style(q_raw_o, cos_o, sin_o)
    
    # Muse Apply (Using Actual Module)
    input_pos = {"height": h_ids.unsqueeze(0), "width": w_ids.unsqueeze(0)}
    q_m_in = q_raw_m.transpose(1, 2) # Muse Module expects [B, S, H, D]
    q_rot_m = rope_mod(q_m_in, input_pos=input_pos).transpose(1, 2)
    
    compare("Q Rotated (Post-RoPE)", q_rot_o, q_rot_m)
    
    print("\n=== 4. Attention Score (Q * K^T) ===")
    # 构造 K
    k_proj_o = origin_model.vision_model.encoder.layers[0].self_attn.k_proj
    k_raw_o = k_proj_o(x_in).view(1, seq, 16, 72).transpose(1, 2)
    k_rot_o = apply_origin_style(k_raw_o, cos_o, sin_o)
    
    # 计算 Dot Product (这是误差最容易放大的地方)
    scale = 72 ** -0.5
    attn_score_o = torch.matmul(q_rot_o, k_rot_o.transpose(-2, -1)) * scale
    
    # 假设 Muse 使用完全相同的输入 (为了隔离 RoPE 误差)
    attn_score_m = torch.matmul(q_rot_m, q_rot_m.transpose(-2, -1)) * scale # 这里的 K 暂用 Q 代替测试自相关，或者用 k_rot_m
    # 为了严谨，计算 Muse 的 K
    k_proj_m = muse_model.encoder.layers[0].attn.k_proj
    k_raw_m = k_proj_m(x_in).view(1, seq, 16, 72).transpose(1, 2)
    k_m_in = k_raw_m.transpose(1, 2)
    k_rot_m = rope_mod(k_m_in, input_pos=input_pos).transpose(1, 2)
    
    attn_score_m = torch.matmul(q_rot_m, k_rot_m.transpose(-2, -1)) * scale
    
    compare("Attn Score (QK^T)", attn_score_o, attn_score_m)
    
    print("\n=== 5. Softmax Output ===")
    attn_probs_o = torch.nn.functional.softmax(attn_score_o, dim=-1)
    attn_probs_m = torch.nn.functional.softmax(attn_score_m, dim=-1)
    
    compare("Attn Probs (Softmax)", attn_probs_o, attn_probs_m)

if __name__ == "__main__":
    run_microscope()