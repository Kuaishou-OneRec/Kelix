"""
Keye Vision Ultimate Verification (Final Fix V6: Shape Handling)
================================================================

修复：
1. 修正 Origin 输出解包逻辑：当返回单元素 List 时，直接取值而不 Stack，避免多余维度。
2. 保持 FP32 + Eager 模式，验证逻辑绝对正确性。
"""

import logging
import os
import sys
import types
import numpy as np
import torch
from PIL import Image
from transformers import PretrainedConfig
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.training.common import set_default_dtype

CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

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

# --- Helpers ---
def create_dummy_image(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)
def log_separator(title: str) -> None:
    logger.info(f"\n{'='*100}\n {title.center(98)} \n{'='*100}")

def compare_tensors_verbose(name: str, reference: torch.Tensor, candidate: torch.Tensor, atol: float = 1e-4) -> None:
    # Unpack tuples
    if isinstance(reference, (tuple, list)): reference = reference[0]
    if isinstance(candidate, (tuple, list)): candidate = candidate[0]
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    # [FIX] 增强的 Shape 对齐逻辑
    # 如果维度数不一样，尝试压缩掉大小为 1 的维度
    while ref.dim() > cand.dim() and ref.shape[0] == 1:
        ref = ref.squeeze(0)
    while cand.dim() > ref.dim() and cand.shape[0] == 1:
        cand = cand.squeeze(0)
        
    # 如果还是不一样，尝试转置 (针对 Linear 权重)
    if ref.shape != cand.shape and ref.dim() == 2 and ref.t().shape == cand.shape:
        ref = ref.t()

    if ref.shape != cand.shape:
        logger.error(f"{name:<40} | ❌ SHAPE ERR | Origin{ref.shape} vs Muse{cand.shape}")
        return

    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # FP32 严格判定
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max: {max_diff:.2e})"
    logger.info(f"{name:<40} | {status:<25} | MeanDiff: {mean_diff:.2e}")

# --- Main Test ---
def test_full_check():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Use Float32 for logic verification
    dtype = torch.float32
    logger.info(f"Running Validation in {dtype} (Eager Mode)...")
    
    muse_config = KeyeVisionConfig(attention_function="eager", use_qk_norm=False)
    origin_config = HFKeyeVisionConfig(**muse_config.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config).to(device).eval()
        muse_model = MuseKeyeVisionModel(muse_config).to(device).eval()
    
    # Force Origin to Eager
    for layer in origin_model.vision_model.encoder.layers:
        layer.self_attn.config._attn_implementation = "eager"

    # 2. Load Weights
    logger.info("Loading Weights...")
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
    muse_model.load_state_dict(muse_state, strict=False)
    origin_load = {k.replace("siglip.", ""): v for k, v in origin_state.items()}
    origin_model.load_state_dict(origin_load, strict=False)

    # 3. Hooks
    activations = {"origin": {}, "muse": {}}
    def get_hook(m, n):
        def hook(mod, inp, out):
            if isinstance(out, (tuple, list)): out = out[0]
            activations[m][n] = out.detach()
        return hook

    # Layer 0 Components
    origin_l0 = origin_model.vision_model.encoder.layers[0]
    muse_l0 = muse_model.encoder.layers[0]
    
    origin_l0.layer_norm1.register_forward_hook(get_hook("origin", "L0.1 LayerNorm1"))
    muse_l0.sa_norm.register_forward_hook(get_hook("muse", "L0.1 LayerNorm1"))
    origin_l0.self_attn.out_proj.register_forward_hook(get_hook("origin", "L0.3 Output Proj"))
    muse_l0.attn.output_proj.register_forward_hook(get_hook("muse", "L0.3 Output Proj"))
    origin_l0.mlp.register_forward_hook(get_hook("origin", "L0.5 MLP Output"))
    muse_l0.mlp.register_forward_hook(get_hook("muse", "L0.5 MLP Output"))
    origin_l0.register_forward_hook(get_hook("origin", "1. Encoder Layer 0 Block"))
    muse_l0.register_forward_hook(get_hook("muse", "1. Encoder Layer 0 Block"))

    # 4. Input
    proc = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    img = create_dummy_image(muse_config.image_size)
    pix = proc(img, return_tensors="pt")["pixel_values"].to(device, dtype)
    pix_patches = pix.unsqueeze(0) # [1, Seq, C, H, W]
    
    seq_len = pix.shape[0]
    side = int(seq_len ** 0.5)
    pids = torch.arange(seq_len, device=device).unsqueeze(0)
    grid = [(1, side, side)]
    
    log_separator("Running Forward (FP32 + Eager)")
    with torch.no_grad():
        # Origin
        origin_out = origin_model(
            pix_patches, 
            position_ids=pids, 
            image_grid_thw=grid, 
            cu_seqlens=None, # Force Eager
            interpolate_pos_encoding=True, 
            window_size=-1,
            use_rope=True
        )
        
        # [FIX] 修正 Origin 输出解包逻辑
        if hasattr(origin_out, "last_hidden_state"): origin_final = origin_out.last_hidden_state
        else: origin_final = origin_out
        
        if isinstance(origin_final, list): 
            # 如果是单元素 list，直接取出来；否则 stack
            if len(origin_final) == 1:
                origin_final = origin_final[0]
            else:
                origin_final = torch.stack(origin_final, dim=0)

        # Muse
        muse_out = muse_model(
            pix_patches, 
            position_ids=pids, 
            image_grid_thw=grid, 
            cu_seqlens=None, # Force Eager
            interpolate_pos_encoding=True, 
            has_learnable_position_embedding=True
        )
        muse_final = muse_out["last_hidden_state"]

    log_separator("Deep Dive Analysis")
    keys = ["L0.1 LayerNorm1", "L0.3 Output Proj", "L0.5 MLP Output", "1. Encoder Layer 0 Block"]
    tol = 1e-4 
    
    for k in keys:
        if k in activations["origin"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=tol)

    compare_tensors_verbose("Final Output", origin_final, muse_final, atol=tol)

if __name__ == "__main__":
    test_full_check()