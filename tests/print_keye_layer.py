"""
Keye Vision Full Integration Test (With Layer 0 Internal Hooks)
===============================================================

新增功能：
1. 深入 Layer 0 内部，Hook 抓取 LayerNorm、Q_Proj、K_Proj、V_Proj 的输出。
2. 帮助定位是投影层(Linear)算错了，还是 RoPE/Attention 算错了。
"""

import logging
import os
import sys
import types
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from PIL import Image
from transformers import PretrainedConfig

# Muse imports
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
    line = "=" * 120
    logger.info("\n%s", line)
    logger.info(" %s ", title.center(118))
    logger.info("%s", line)

def compare_tensors_verbose(name: str, reference: torch.Tensor, candidate: torch.Tensor, atol: float = 1e-3) -> None:
    # Unpack tuples
    if isinstance(reference, (tuple, list)): reference = reference[0]
    if isinstance(candidate, (tuple, list)): candidate = candidate[0]
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    # Shape Alignment
    if ref.shape != cand.shape:
        # Try squeezing batch dim
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2: ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2: cand = cand.squeeze(0)
        
        # Try transposing (Specific for Linear layer outputs [B, S, D] vs [B, D, S] if messed up)
        if ref.shape != cand.shape and ref.dim() == 3:
             if ref.transpose(1, 2).shape == cand.shape: ref = ref.transpose(1, 2)
    
    if ref.shape != cand.shape:
        logger.error(f"{name:<40} | ❌ SHAPE ERR | Origin{ref.shape} vs Muse{cand.shape}")
        return

    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max: {max_diff:.2e})"
    logger.info(f"{name:<40} | {status:<25} | MeanDiff: {mean_diff:.2e}")

# --- Weight Loading ---
def convert_and_load_weights(origin_model, muse_model, path, dtype, device):
    logger.info("Loading checkpoint...")
    raw = torch.load(path, map_location="cpu")
    if "module" in raw: raw = raw["module"]
    
    origin_state = {}
    for k, v in raw.items():
        clean = k
        for p in ["module.", "vision_tower.", "siglip."]:
            if clean.startswith(p): clean = clean[len(p):]
        if "vision_model" not in clean: clean = "vision_model." + clean
        origin_state["siglip." + clean] = v.to(dtype)

    # Muse Load
    muse_state = muse_model.convert_hf_state_dict(origin_state)
    muse_model.load_state_dict(muse_state, strict=False)
    
    # Origin Load
    origin_load = {k.replace("siglip.", ""): v for k, v in origin_state.items()}
    origin_model.load_state_dict(origin_load, strict=False)
    
    origin_model.to(device)
    muse_model.to(device)

# --- Input ---
def prepare_pixel_inputs(processor, image, device, dtype):
    processed = processor.preprocess(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    grid_info = processed["image_grid_thw"]
    if isinstance(grid_info, torch.Tensor): grid_info = grid_info.cpu().tolist()
    elif isinstance(grid_info, np.ndarray): grid_info = grid_info.tolist()
    
    image_grid_thw = [tuple(int(v) for v in grid) for grid in grid_info]
    patches = [int(np.prod(grid)) for grid in image_grid_thw]
    
    # 5D Input
    batched = []
    start = 0
    for count in patches:
        batched.append(pixel_values[start : start + count])
        start += count
    pixel_batch = torch.stack(batched, dim=0).to(device=device, dtype=dtype).contiguous()
    
    seq_len = patches[0]
    pids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).repeat(len(image_grid_thw), 1)
    cu_seq = torch.tensor([0] + [sum(patches[:i+1]) for i in range(len(patches))], dtype=torch.int32, device=device)
    
    return pixel_batch, image_grid_thw, pids, cu_seq

# --- Main Test ---
def test_full_check():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    muse_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_config.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config)
        muse_model = MuseKeyeVisionModel(muse_config)
    
    origin_model.eval()
    muse_model.eval()
    
    convert_and_load_weights(origin_model, muse_model, CHECKPOINT_PATH, dtype, device)
    
    # --- Hook System ---
    activations = {"origin": {}, "muse": {}}
    def get_hook(m, n):
        def hook(mod, inp, out):
            if isinstance(out, (tuple, list)): out = out[0]
            activations[m][n] = out.detach()
        return hook

    # 1. Standard Layer Hooks
    origin_model.vision_model.embeddings.register_forward_hook(get_hook("origin", "0. Embeddings"))
    muse_model.embeddings.register_forward_hook(get_hook("muse", "0. Embeddings"))
    
    origin_model.vision_model.encoder.layers[0].register_forward_hook(get_hook("origin", "1. Encoder Layer 0 Output"))
    muse_model.encoder.layers[0].register_forward_hook(get_hook("muse", "1. Encoder Layer 0 Output"))
    
    # 2. [NEW] Layer 0 Internals (Debug Hooks)
    # Layer Norm 1 (Pre-Attention)
    origin_model.vision_model.encoder.layers[0].layer_norm1.register_forward_hook(get_hook("origin", "L0.1 LayerNorm1"))
    muse_model.encoder.layers[0].sa_norm.register_forward_hook(get_hook("muse", "L0.1 LayerNorm1"))
    
    # Q/K/V Projections (Linear Output)
    # Origin: self_attn.q_proj
    origin_model.vision_model.encoder.layers[0].self_attn.q_proj.register_forward_hook(get_hook("origin", "L0.2 Q Proj"))
    origin_model.vision_model.encoder.layers[0].self_attn.k_proj.register_forward_hook(get_hook("origin", "L0.2 K Proj"))
    origin_model.vision_model.encoder.layers[0].self_attn.v_proj.register_forward_hook(get_hook("origin", "L0.2 V Proj"))
    
    # Muse: attn.q_proj
    muse_model.encoder.layers[0].attn.q_proj.register_forward_hook(get_hook("muse", "L0.2 Q Proj"))
    muse_model.encoder.layers[0].attn.k_proj.register_forward_hook(get_hook("muse", "L0.2 K Proj"))
    muse_model.encoder.layers[0].attn.v_proj.register_forward_hook(get_hook("muse", "L0.2 V Proj"))
    
    # Run Forward
    proc = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    img = create_dummy_image(muse_config.image_size)
    pix, grid, pids, cu = prepare_pixel_inputs(proc, img, device, dtype)
    
    log_separator("Running Forward")
    with torch.no_grad():
        # Origin
        origin_out = origin_model(pix, position_ids=pids, image_grid_thw=grid, cu_seqlens=cu, interpolate_pos_encoding=True, window_size=-1)
        if hasattr(origin_out, "last_hidden_state"): origin_final = origin_out.last_hidden_state
        else: origin_final = origin_out
        if isinstance(origin_final, list): origin_final = torch.stack(origin_final, dim=0)

        # Muse
        muse_out = muse_model(pix, position_ids=pids, image_grid_thw=grid, cu_seqlens=cu, interpolate_pos_encoding=True, has_learnable_position_embedding=True)
        muse_final = muse_out["last_hidden_state"]

    # Compare
    log_separator("Deep Dive: Layer 0 Analysis")
    # 打印顺序：Embeddings -> LN -> QKV -> Layer Output
    keys_to_check = [
        "0. Embeddings", 
        "L0.1 LayerNorm1",
        "L0.2 Q Proj", "L0.2 K Proj", "L0.2 V Proj",
        "1. Encoder Layer 0 Output"
    ]
    
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    for k in keys_to_check:
        if k in activations["origin"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=tol)

    log_separator("Final Output")
    compare_tensors_verbose("Final Output", origin_final, muse_final, atol=tol)

if __name__ == "__main__":
    test_full_check()