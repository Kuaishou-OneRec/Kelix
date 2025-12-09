"""
Integration test to ensure Muse SigLIP matches Hugging Face logits.
Refined for Verbose Logging, Mapping Verification, and Deep Internal Coverage.
"""

import os
import sys
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel as HFSiglipVisionModel
from muse.config import SiglipVisionConfig

# 假设 Muse 的 SiglipVisionTransformer 路径如下，请根据实际情况调整
from muse.models.Siglip import SiglipVisionTransformer as SiglipVisionModel
from muse.training.common import set_default_dtype

# 配置简单的日志格式
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)




@contextmanager
def _mock_context_parallel():
    """Stub context parallel helpers so tests can run without torch.distributed init."""
    patches = [
        patch("muse.training.parallel.get_context_parallel_world_size", new=lambda: 1),
        patch("muse.training.parallel.get_context_parallel_group", new=lambda backend="nccl": None),
        patch("muse.training.parallel.get_context_parallel_rank", new=lambda: 0),
        patch("muse.layers.attention.get_context_parallel_world_size", new=lambda: 1),
        patch("muse.layers.attention.get_context_parallel_group", new=lambda backend="nccl": None),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def _build_siglip_config(hf_cfg: Dict[str, Any]) -> SiglipVisionConfig:
    """Map Hugging Face config to Muse SiglipVisionConfig."""
    image_size = hf_cfg.get("image_size", 384)
    patch_size = hf_cfg.get("patch_size", 14)
    default_max_seq_len = (image_size // patch_size) ** 2
    
    return SiglipVisionConfig(
        model_class="SiglipVisionTransformer",
        image_size=image_size,
        patch_size=patch_size,
        num_channels=hf_cfg.get("num_channels", 3),
        hidden_size=hf_cfg.get("hidden_size", 1152),
        num_hidden_layers=hf_cfg.get("num_hidden_layers", 27),
        num_attention_heads=hf_cfg.get("num_attention_heads", 16),
        intermediate_size=hf_cfg.get("intermediate_size", 4304),
        max_seq_len=hf_cfg.get("max_seq_len", default_max_seq_len),
        layer_norm_eps=hf_cfg.get("layer_norm_eps", 1e-6),
        attention_dropout=hf_cfg.get("attention_dropout", 0.0),
        has_learnable_position_embedding=hf_cfg.get("has_learnable_position_embedding", False),
        use_qk_norm=hf_cfg.get("use_qk_norm", False),
        qk_norm_eps=hf_cfg.get("qk_norm_eps", 1e-6),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        attention_function="eager", # 建议这里先改回 eager 排查，如果环境允许 flash_attn 再开
        output_attentions=False,
        output_hidden_states=False,
    )

def create_dummy_image(size: int = 224) -> Image.Image:
    np.random.seed(42)
    data = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)

def format_tensor_val(t: torch.Tensor, n: int = 5) -> str:
    vals = t.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join([f"{x:.6f}" for x in vals]) + "]"

def log_separator(title: str):
    logger.info(f"\n{'='*100}")
    logger.info(f" {title.center(98)} ")
    logger.info(f"{'='*100}")

def compare_tensors_verbose(name: str, tensor_hf: torch.Tensor, tensor_muse: torch.Tensor, atol=1e-3):
    # 处理可能的 Tuple 输出
    if isinstance(tensor_hf, (tuple, list)): tensor_hf = tensor_hf[0]
    if isinstance(tensor_muse, (tuple, list)): tensor_muse = tensor_muse[0]

    t1 = tensor_hf.detach().float().cpu()
    t2 = tensor_muse.detach().float().cpu()
    
    # 尝试自动处理 batch 维度差异 (例如 [1, S, D] vs [S, D])
    if t1.shape != t2.shape:
        if t1.dim() == 3 and t1.shape[0] == 1 and t2.dim() == 2: t1 = t1.squeeze(0)
        elif t2.dim() == 3 and t2.shape[0] == 1 and t1.dim() == 2: t2 = t2.squeeze(0)

    if t1.shape != t2.shape:
        logger.error(f"❌ SHAPE MISMATCH [{name}]: HF={t1.shape} vs Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    match_status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max Diff: {max_diff:.2e})"
    
    logger.info(f"{name:<40} | {match_status:<25} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    
    if max_diff >= atol:
        logger.info(f"   -> HF   (first 3): {format_tensor_val(t1, 3)}")
        logger.info(f"   -> Muse (first 3): {format_tensor_val(t2, 3)}")

# =========================================================================
# Enhanced Hook System
# =========================================================================
activations = {"hf": {}, "muse": {}}

def make_hook(model_name, layer_name, capture_input=False):
    def hook(module, inp, out):
        target = inp if capture_input else out
        if isinstance(target, (tuple, list)):
            target = target[0]
        activations[model_name][layer_name] = target.detach()
    return hook

def register_debug_hooks(hf_model, muse_model):
    """
    Hook 注入：深入 Layer 0 的 Attention 内部
    """
    # ------------------------------------------------------------------
    # HF Hooks
    # ------------------------------------------------------------------
    hf_l0 = hf_model.vision_model.encoder.layers[0]
    
    # 1. Norm
    hf_l0.layer_norm1.register_forward_hook(make_hook("hf", "L0.1 LN1"))
    
    # 2. Attention Internals
    # Q/K/V Proj Outputs
    hf_l0.self_attn.q_proj.register_forward_hook(make_hook("hf", "L0.2.1 Q Proj"))
    hf_l0.self_attn.k_proj.register_forward_hook(make_hook("hf", "L0.2.2 K Proj"))
    hf_l0.self_attn.v_proj.register_forward_hook(make_hook("hf", "L0.2.3 V Proj"))
    
    # Raw Attn Context (Before Output Projection)
    # 技巧：Hook out_proj 的输入 (forward_pre_hook)
    hf_l0.self_attn.out_proj.register_forward_hook(make_hook("hf", "L0.2.4 Attn Context (Pre-OutProj)", capture_input=True))
    
    # Final Attn Output
    hf_l0.self_attn.out_proj.register_forward_hook(make_hook("hf", "L0.2.5 Attn Out (Post-OutProj)"))

    # 3. Post-Attn Norm
    hf_l0.layer_norm2.register_forward_hook(make_hook("hf", "L0.3 LN2"))

    # ------------------------------------------------------------------
    # Muse Hooks
    # ------------------------------------------------------------------
    muse_l0 = muse_model.encoder.layers[0]
    
    # 1. Norm
    # SigLIP 通常用 sa_norm 或 input_layernorm
    if hasattr(muse_l0, "sa_norm"): ln1 = muse_l0.sa_norm
    else: ln1 = muse_l0.input_layernorm
    ln1.register_forward_hook(make_hook("muse", "L0.1 LN1"))

    # 2. Attention Internals
    # Muse 的 Attention 模块通常叫 attn 或 self_attn
    attn_module = getattr(muse_l0, "attn", getattr(muse_l0, "self_attn", None))
    
    if attn_module:
        # Q/K/V Proj
        attn_module.q_proj.register_forward_hook(make_hook("muse", "L0.2.1 Q Proj"))
        attn_module.k_proj.register_forward_hook(make_hook("muse", "L0.2.2 K Proj"))
        attn_module.v_proj.register_forward_hook(make_hook("muse", "L0.2.3 V Proj"))
        
        # Raw Attn Context
        # Muse 通常叫 output_proj 或 out_proj
        out_proj = getattr(attn_module, "output_proj", getattr(attn_module, "out_proj", None))
        if out_proj:
            out_proj.register_forward_hook(make_hook("muse", "L0.2.4 Attn Context (Pre-OutProj)", capture_input=True))
            out_proj.register_forward_hook(make_hook("muse", "L0.2.5 Attn Out (Post-OutProj)"))
        else:
            logger.error("❌ Could not find output_proj in Muse attention module")
    else:
        logger.error("❌ Could not find attention module in Muse layer 0")

    # 3. Post-Attn Norm
    if hasattr(muse_l0, "mlp_norm"): ln2 = muse_l0.mlp_norm
    else: ln2 = muse_l0.post_attention_layernorm
    ln2.register_forward_hook(make_hook("muse", "L0.3 LN2"))

    logger.info("✅ Granular Attention hooks registered.")

def test_siglip_logits_align_with_hf_checkpoint():
    with _mock_context_parallel():
        _run_siglip_logits_align_with_hf_checkpoint()

def _run_siglip_logits_align_with_hf_checkpoint():
    torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    target_dtype = torch.float32()

    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"
    logger.info(f"Testing checkpoint: {checkpoint_dir}")

    # --- Load HF ---
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(checkpoint_dir, torch_dtype=target_dtype, device_map="auto")
    hf_model.eval()
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype

    # --- Load Muse ---
    hf_state_dict = hf_model.state_dict()
    muse_config = _build_siglip_config(hf_model.config.to_dict())
    
    with set_default_dtype(dtype):
        muse_model = SiglipVisionModel(muse_config)

    # Weights
    full_prefixed = {f"siglip.vision_model.{k}": v for k, v in hf_state_dict.items()}
    converted = muse_model.convert_hf_state_dict(full_prefixed)
    # Ensure dtype
    for k, v in converted.items():
        if isinstance(v, torch.Tensor): converted[k] = v.to(dtype=dtype)
    
    muse_model.load_state_dict(converted, strict=False)
    muse_model = muse_model.to(device=device, dtype=dtype).eval()

    # --- Forward ---
    register_debug_hooks(hf_model, muse_model)
    log_separator("Running Forward")

    image = create_dummy_image(hf_model.config.image_size)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]
    
    # 构造 SigLIP grid
    grid_thw = [(1, muse_config.image_size // 14, muse_config.image_size // 14)] * pixel_values.shape[0]

    with torch.no_grad():
        hf_model(**inputs)
        muse_model(pixel_values=pixel_values, image_grid_thw=grid_thw)

    # --- Analysis ---
    log_separator("Deep Dive: Attention Internals")
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-5
    
    keys = [
        "L0.1 LN1",
        "L0.2.1 Q Proj",
        "L0.2.2 K Proj",
        "L0.2.3 V Proj",
        "L0.2.4 Attn Context (Pre-OutProj)", # 关键点：Attention 算完，Linear 还没算
        "L0.2.5 Attn Out (Post-OutProj)",
        "L0.3 LN2"
    ]

    for k in keys:
        if k in activations["hf"] and k in activations["muse"]:
            compare_tensors_verbose(k, activations["hf"][k], activations["muse"][k], atol=tol)
        else:
            logger.warning(f"⚠️ Missing hook for {k}")

if __name__ == "__main__":
    test_siglip_logits_align_with_hf_checkpoint()