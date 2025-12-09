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


def _resolve_target_dtype() -> torch.dtype:
    """Prefer BF16; fall back to FP32 if unsupported."""
    if torch.cuda.is_available():
        is_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if is_supported is None or is_supported():
            return torch.bfloat16
        logger.warning("CUDA BF16 not supported on this device, falling back to float32.")
        return torch.float32
    logger.warning("CUDA not available; using float32.")
    return torch.float32


@contextmanager
def _mock_context_parallel():
    """Stub context parallel helpers so tests can run without torch.distributed init."""
    patches = [
        patch("muse.training.parallel.get_context_parallel_world_size", new=lambda: 1),
        patch("muse.training.parallel.get_context_parallel_group", new=lambda backend="nccl": None),
        patch("muse.training.parallel.get_context_parallel_rank", new=lambda: 0),
        # patch("muse.layers.attention.get_context_parallel_world_size", new=lambda: 1),
        # patch("muse.layers.attention.get_context_parallel_group", new=lambda backend="nccl": None),
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
        attention_function="flash_attention_2", # 建议这里先改回 eager 排查，如果环境允许 flash_attn 再开
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
# Hook Helper System
# =========================================================================
activations = {"hf": {}, "muse": {}}

def make_hook(model_name, layer_name):
    def hook(module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        activations[model_name][layer_name] = out.detach()
    return hook

def register_debug_hooks(hf_model, muse_model):
    """
    自动注册关键层 Hook
    """
    # 1. Embeddings (Output)
    # HF: vision_model.embeddings
    hf_model.vision_model.embeddings.register_forward_hook(make_hook("hf", "0. Embeddings Output"))
    # Muse: embeddings
    muse_model.embeddings.register_forward_hook(make_hook("muse", "0. Embeddings Output"))

    # 2. Encoder Layer 0 (检查最底层)
    # HF Layer 0
    hf_l0 = hf_model.vision_model.encoder.layers[0]
    hf_l0.layer_norm1.register_forward_hook(make_hook("hf", "L0.1 LayerNorm1 (Pre-Attn)"))
    hf_l0.self_attn.register_forward_hook(make_hook("hf", "L0.2 SelfAttn Output"))
    hf_l0.layer_norm2.register_forward_hook(make_hook("hf", "L0.3 LayerNorm2 (Pre-MLP)"))
    hf_l0.mlp.register_forward_hook(make_hook("hf", "L0.4 MLP Output"))
    
    # Muse Layer 0
    # 注意：这里假设 Muse 的命名结构。如果不一致，需要根据 Muse 代码调整
    muse_l0 = muse_model.encoder.layers[0]
    
    # 尝试探测 LayerNorm1
    if hasattr(muse_l0, "layer_norm1"): ln1 = muse_l0.layer_norm1
    elif hasattr(muse_l0, "input_layernorm"): ln1 = muse_l0.input_layernorm
    elif hasattr(muse_l0, "sa_norm"): ln1 = muse_l0.sa_norm
    else: ln1 = None
    if ln1: ln1.register_forward_hook(make_hook("muse", "L0.1 LayerNorm1 (Pre-Attn)"))

    # 尝试探测 Attention
    if hasattr(muse_l0, "self_attn"): attn = muse_l0.self_attn
    elif hasattr(muse_l0, "attn"): attn = muse_l0.attn
    else: attn = None
    if attn: attn.register_forward_hook(make_hook("muse", "L0.2 SelfAttn Output"))
    
    # 尝试探测 LayerNorm2
    if hasattr(muse_l0, "layer_norm2"): ln2 = muse_l0.layer_norm2
    elif hasattr(muse_l0, "post_attention_layernorm"): ln2 = muse_l0.post_attention_layernorm
    elif hasattr(muse_l0, "mlp_norm"): ln2 = muse_l0.mlp_norm
    else: ln2 = None
    if ln2: ln2.register_forward_hook(make_hook("muse", "L0.3 LayerNorm2 (Pre-MLP)"))

    # 尝试探测 MLP
    if hasattr(muse_l0, "mlp"): mlp = muse_l0.mlp
    else: mlp = None
    if mlp: mlp.register_forward_hook(make_hook("muse", "L0.4 MLP Output"))

    logger.info("Debug hooks registered for Layer 0.")

def test_siglip_logits_align_with_hf_checkpoint():
    with _mock_context_parallel():
        _run_siglip_logits_align_with_hf_checkpoint()


def _run_siglip_logits_align_with_hf_checkpoint():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    target_dtype = _resolve_target_dtype()

    # =========================================================================
    # 1. 配置路径
    # =========================================================================
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"
    logger.info(f"Testing checkpoint: {checkpoint_dir}")

    # =========================================================================
    # 2. 加载 HF 模型
    # =========================================================================
    logger.info(f"Loading HF model with dtype={target_dtype} ...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(
        checkpoint_dir,
        torch_dtype=target_dtype,
        device_map="auto"
    )
    hf_model.eval()

    device = next(hf_model.parameters()).device
    actual_dtype = next(hf_model.parameters()).dtype
    dtype = actual_dtype
    logger.info(f"Reference Model Device: {device}, Dtype: {dtype}")

    # =========================================================================
    # 3. 初始化 Muse 模型
    # =========================================================================
    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()
    muse_config = _build_siglip_config(hf_config_dict)

    with set_default_dtype(dtype):
        muse_model = SiglipVisionModel(muse_config)

    # =========================================================================
    # 4. 权重映射与加载
    # =========================================================================
    log_separator("Weight Analysis Phase")
    
    prefixed_keys_map = {k: f"siglip.vision_model.{k}" for k in hf_state_dict.keys()}
    full_prefixed_state_dict = {}
    for hf_key, prefixed_key in prefixed_keys_map.items():
        full_prefixed_state_dict[prefixed_key] = hf_state_dict[hf_key]

    logger.info("Converting and loading weights...")
    final_converted_state_dict = muse_model.convert_hf_state_dict(full_prefixed_state_dict)
    
    # 统一精度
    for key in final_converted_state_dict:
        if isinstance(final_converted_state_dict[key], torch.Tensor):
            final_converted_state_dict[key] = final_converted_state_dict[key].to(dtype=dtype)

    load_res = muse_model.load_state_dict(final_converted_state_dict, strict=False)
    
    # 检查关键权重缺失
    critical_missing = [k for k in load_res.missing_keys if k.endswith(".weight") or k.endswith(".bias")]
    if critical_missing:
        logger.error(f"❌ CRITICAL ERROR: {len(critical_missing)} params missing! Example: {critical_missing[0]}")
    else:
        logger.info("✅ Weight loading coverage looks good.")

    muse_model = muse_model.to(device=device, dtype=dtype)
    muse_model.eval()

    # =========================================================================
    # 5. 注册 Hook 并运行前向传播
    # =========================================================================
    register_debug_hooks(hf_model, muse_model)
    
    log_separator("Forward Pass Comparison")
    
    image = create_dummy_image(hf_model.config.image_size)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # --- HF Forward ---
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        hf_last_hidden = hf_outputs.last_hidden_state

    # --- Muse Forward ---
    pixel_values = inputs["pixel_values"]
    batch_size = pixel_values.shape[0]
    num_patches_per_side = muse_config.image_size // muse_config.patch_size
    image_grid_thw = [(1, num_patches_per_side, num_patches_per_side)] * batch_size

    with torch.no_grad():
        muse_outputs = muse_model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        # 兼容 dict 或 object 返回值
        if isinstance(muse_outputs, dict):
            muse_last_hidden = muse_outputs["last_hidden_state"]
        else:
            muse_last_hidden = muse_outputs.last_hidden_state

    # =========================================================================
    # 6. 深入分析 (Deep Dive Analysis)
    # =========================================================================
    log_separator("Deep Dive: Internal Layer Analysis")
    
    # 精度容忍度 (BF16 需要宽一点)
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-5
    
    keys_to_check = [
        "0. Embeddings Output",
        "L0.1 LayerNorm1 (Pre-Attn)",
        "L0.2 SelfAttn Output",
        "L0.3 LayerNorm2 (Pre-MLP)",
        "L0.4 MLP Output"
    ]
    
    for k in keys_to_check:
        if k in activations["hf"] and k in activations["muse"]:
            compare_tensors_verbose(k, activations["hf"][k], activations["muse"][k], atol=tol)
        else:
            logger.warning(f"⚠️  Skipping {k}: Missing hook capture (HF={k in activations['hf']}, Muse={k in activations['muse']})")

    # Final Output Comparison
    log_separator("Final Output Analysis")
    hf_res = hf_last_hidden.to(device, dtype=dtype)
    muse_res = muse_last_hidden.to(device, dtype=dtype)
    
    compare_tensors_verbose("Last Hidden State", hf_res, muse_res, atol=tol)

    final_diff = (hf_res - muse_res).abs().max().item()
    if final_diff < tol:
        logger.info(f"\n✅ SUCCESS: Outputs match! (Max Diff: {final_diff:.2e})")
    else:
        logger.error(f"\n❌ FAILURE: Outputs mismatch! (Max Diff: {final_diff:.2e})")

if __name__ == "__main__":
    test_siglip_logits_align_with_hf_checkpoint()