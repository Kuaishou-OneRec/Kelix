"""
Keye Vision Full Check (With MLP & OutputProj Debug) - Final Fixed Input
========================================================================

修复：
1. Input Preparation: 适配 Muse Processor 返回的 [Seq, C, H, W] 格式，
   不再尝试对已经是 Patch 的数据进行二次切片，解决维度错误。
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
def compare_tensors_verbose(name: str, reference: torch.Tensor, candidate: torch.Tensor, atol: float = 1e-3) -> None:
    if isinstance(reference, (tuple, list)): reference = reference[0]
    if isinstance(candidate, (tuple, list)): candidate = candidate[0]
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    # Shape Alignment
    if ref.shape != cand.shape:
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2: ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2: cand = cand.squeeze(0)
        if ref.shape != cand.shape and ref.dim() == 3 and ref.transpose(1, 2).shape == cand.shape: ref = ref.transpose(1, 2)
    
    if ref.shape != cand.shape:
        logger.error(f"{name:<40} | ❌ SHAPE ERR | Origin{ref.shape} vs Muse{cand.shape}")
        return
    
    diff = (ref - cand).abs()
    status = "✅ MATCH" if diff.max().item() < atol else f"❌ MISMATCH (Max: {diff.max().item():.2e})"
    logger.info(f"{name:<40} | {status:<25} | MeanDiff: {diff.mean().item():.2e}")

# --- Setup & Hooks ---
def test_full_check():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    muse_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_config.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config).to(device).eval()
        muse_model = MuseKeyeVisionModel(muse_config).to(device).eval()
    
    # Load Weights
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

    # Hooks
    activations = {"origin": {}, "muse": {}}
    def get_hook(m, n):
        def hook(mod, inp, out):
            if isinstance(out, (tuple, list)): out = out[0]
            activations[m][n] = out.detach()
        return hook

    # Layer 0 Components
    origin_l0 = origin_model.vision_model.encoder.layers[0]
    muse_l0 = muse_model.encoder.layers[0]
    
    # 1. Norm 1 & QKV
    origin_l0.layer_norm1.register_forward_hook(get_hook("origin", "L0.1 LayerNorm1"))
    muse_l0.sa_norm.register_forward_hook(get_hook("muse", "L0.1 LayerNorm1"))
    origin_l0.self_attn.q_proj.register_forward_hook(get_hook("origin", "L0.2 Q Proj"))
    muse_l0.attn.q_proj.register_forward_hook(get_hook("muse", "L0.2 Q Proj"))

    # 2. Output Projection (Post-Attention)
    origin_l0.self_attn.out_proj.register_forward_hook(get_hook("origin", "L0.3 Output Proj"))
    muse_l0.attn.output_proj.register_forward_hook(get_hook("muse", "L0.3 Output Proj"))
    
    # 3. Layer Norm 2 (Pre-MLP)
    origin_l0.layer_norm2.register_forward_hook(get_hook("origin", "L0.4 LayerNorm2"))
    muse_l0.mlp_norm.register_forward_hook(get_hook("muse", "L0.4 LayerNorm2"))

    # 4. MLP Output
    origin_l0.mlp.register_forward_hook(get_hook("origin", "L0.5 MLP Output"))
    muse_l0.mlp.register_forward_hook(get_hook("muse", "L0.5 MLP Output"))

    # 5. Block Output
    origin_l0.register_forward_hook(get_hook("origin", "1. Encoder Layer 0 Block"))
    muse_l0.register_forward_hook(get_hook("muse", "1. Encoder Layer 0 Block"))

    # --- [FIXED] Input Preparation ---
    proc = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    img = create_dummy_image(muse_config.image_size)
    
    # Muse Processor 直接输出 [Seq_Patches, 3, 14, 14]
    inputs = proc(img, return_tensors="pt")
    pix = inputs["pixel_values"].to(device, dtype)
    
    # Origin 需要 [Batch, Seq_Patches, 3, 14, 14]
    # 我们只需加一个 Batch 维度
    pix_patches = pix.unsqueeze(0) 
    
    seq_len = pix.shape[0]
    side = int(seq_len ** 0.5)
    logger.info(f"Input Shape: {pix_patches.shape} (Grid {side}x{side})")

    pids = torch.arange(seq_len, device=device).unsqueeze(0)
    grid = [(1, side, side)]
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    log_separator("Running Forward")
    with torch.no_grad():
        origin_out = origin_model(pix_patches, position_ids=pids, image_grid_thw=grid, cu_seqlens=cu, interpolate_pos_encoding=True, window_size=-1,use_rope=True)
        muse_out = muse_model(pix_patches, position_ids=pids, image_grid_thw=grid, cu_seqlens=cu, interpolate_pos_encoding=True, has_learnable_position_embedding=True)

    log_separator("Deep Dive Analysis")
    keys = ["L0.1 LayerNorm1", "L0.2 Q Proj", "L0.3 Output Proj", "L0.4 LayerNorm2", "L0.5 MLP Output", "1. Encoder Layer 0 Block"]
    tol = 5e-2 # BF16 tolerance
    
    for k in keys:
        if k in activations["origin"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=tol)

if __name__ == "__main__":
    test_full_check()