"""
Keye Vision Ultimate Verification (FP32 + Eager Mode)
=====================================================

原理：
1. 强行使用 Float32 消除 BF16 的 0.0625 精度噪音。
2. 传入 cu_seqlens=None，强迫 Muse 和 Origin 关闭 FlashAttention，转而使用 Eager 模式。
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
    if ref.shape != cand.shape:
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2: ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2: cand = cand.squeeze(0)
        if ref.shape != cand.shape and ref.dim() == 3 and ref.transpose(1, 2).shape == cand.shape: ref = ref.transpose(1, 2)
    
    if ref.shape != cand.shape:
        logger.error(f"{name:<40} | ❌ SHAPE ERR | Origin{ref.shape} vs Muse{cand.shape}")
        return
    
    diff = (ref - cand).abs()
    # FP32 下，我们要求误差极小 (1e-4)
    status = "✅ MATCH" if diff.max().item() < atol else f"❌ MISMATCH (Max: {diff.max().item():.2e})"
    logger.info(f"{name:<40} | {status:<25} | MeanDiff: {diff.mean().item():.2e}")

# --- Main Test ---
def test_full_check():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # [FIX 1] 强制使用 Float32
    dtype = torch.float32
    logger.info(f"Running Validation in {dtype} (Eager Mode)...")
    
    # [FIX 2] Muse 使用 Eager
    muse_config = KeyeVisionConfig(
        attention_function="eager", 
        use_qk_norm=False
    )
    origin_config = HFKeyeVisionConfig(**muse_config.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config).to(device).eval()
        muse_model = MuseKeyeVisionModel(muse_config).to(device).eval()
    
    # [FIX 3] Origin 强制 Eager (虽然传 cu_seqlens=None 也会触发，但双重保险)
    for layer in origin_model.vision_model.encoder.layers:
        layer.self_attn.config._attn_implementation = "eager"

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
        origin_state["siglip." + clean] = v.to(dtype) # 转为 FP32
    
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

    # Layer 0 Hooks
    origin_l0 = origin_model.vision_model.encoder.layers[0]
    muse_l0 = muse_model.encoder.layers[0]
    
    origin_l0.layer_norm1.register_forward_hook(get_hook("origin", "L0.1 LayerNorm1"))
    muse_l0.sa_norm.register_forward_hook(get_hook("muse", "L0.1 LayerNorm1"))
    origin_l0.self_attn.out_proj.register_forward_hook(get_hook("origin", "L0.3 Output Proj"))
    muse_l0.attn.output_proj.register_forward_hook(get_hook("muse", "L0.3 Output Proj"))
    origin_l0.layer_norm2.register_forward_hook(get_hook("origin", "L0.4 LayerNorm2"))
    muse_l0.mlp_norm.register_forward_hook(get_hook("muse", "L0.4 LayerNorm2"))
    origin_l0.mlp.register_forward_hook(get_hook("origin", "L0.5 MLP Output"))
    muse_l0.mlp.register_forward_hook(get_hook("muse", "L0.5 MLP Output"))
    origin_l0.register_forward_hook(get_hook("origin", "1. Encoder Layer 0 Block"))
    muse_l0.register_forward_hook(get_hook("muse", "1. Encoder Layer 0 Block"))

    # Input Prep
    proc = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    img = create_dummy_image(muse_config.image_size)
    inputs = proc(img, return_tensors="pt")
    pix = inputs["pixel_values"].to(device, dtype)
    pix_patches = pix.unsqueeze(0) 
    
    seq_len = pix.shape[0]
    side = int(seq_len ** 0.5)
    pids = torch.arange(seq_len, device=device).unsqueeze(0)
    grid = [(1, side, side)]
    
    # [FIX 4] cu_seqlens = None 
    # 这会告诉 Origin 和 Muse 模型：“这不是 Packed Sequence，请用普通 Attention”
    # 从而避开 FlashAttn 的 BF16 限制
    cu = None 

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
        
        # Muse
        muse_out = muse_model(
            pix_patches, 
            position_ids=pids, 
            image_grid_thw=grid, 
            cu_seqlens=None, # Force Eager
            interpolate_pos_encoding=True, 
            has_learnable_position_embedding=True
        )

    log_separator("Deep Dive Analysis")
    keys = ["L0.1 LayerNorm1", "L0.3 Output Proj", "L0.4 LayerNorm2", "L0.5 MLP Output", "1. Encoder Layer 0 Block"]
    
    # FP32 下，即使有累积误差，通常也在 1e-5 级别
    tol = 1e-4 
    
    for k in keys:
        if k in activations["origin"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=tol)

if __name__ == "__main__":
    test_full_check()