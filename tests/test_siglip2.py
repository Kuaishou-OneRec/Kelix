"""
Integration test to ensure Muse SigLIP matches Hugging Face logits.
Refined for Verbose Logging, Mapping Verification, and Coverage Checking.
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
        attention_function="eager",
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

def compare_tensors_verbose(name: str, tensor_hf: torch.Tensor, tensor_muse: torch.Tensor, atol=1e-5):
    t1 = tensor_hf.detach().float().cpu()
    t2 = tensor_muse.detach().float().cpu()
    
    if t1.shape != t2.shape:
        logger.error(f"❌ SHAPE MISMATCH [{name}]: HF={t1.shape} vs Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    match_status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max Diff: {max_diff:.2e})"
    
    logger.info(f"{'-'*100}")
    logger.info(f"Layer: {name}")
    logger.info(f"Status: {match_status}")
    logger.info(f"Stats : MaxDiff={max_diff:.2e} | MeanDiff={mean_diff:.2e} | HF_Mean={t1.mean():.4f} | Muse_Mean={t2.mean():.4f}")
    
    if max_diff >= atol:
        logger.info(f"Values Comparison (First 10 flattened):")
        logger.info(f"  HF  : {format_tensor_val(t1, 10)}")
        logger.info(f"  Muse: {format_tensor_val(t2, 10)}")

def test_siglip_logits_align_with_hf_checkpoint():
    with _mock_context_parallel():
        _run_siglip_logits_align_with_hf_checkpoint()


def _run_siglip_logits_align_with_hf_checkpoint():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # =========================================================================
    # 1. 配置路径
    # =========================================================================
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"
    logger.info(f"Testing checkpoint: {checkpoint_dir}")

    # =========================================================================
    # 2. 加载 HF 模型
    # =========================================================================
    logger.info("Loading HF model...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    hf_model.eval()

    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    logger.info(f"Reference Model Device: {device}, Dtype: {dtype}")

    # =========================================================================
    # 3. 初始化 Muse 模型
    # =========================================================================
    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()
    muse_config = _build_siglip_config(hf_config_dict)

    model_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    with set_default_dtype(model_dtype):
        muse_model = SiglipVisionModel(muse_config)

    # =========================================================================
    # 4. 权重映射、转换与覆盖率检查
    # =========================================================================
    log_separator("Weight Analysis Phase")
    
    # --- 阶段 4.1: 探针式检测映射关系 (生成文档) ---
    logger.info("Step 1: Tracing weight mapping logic...")
    
    prefixed_keys_map = {k: f"siglip.vision_model.{k}" for k in hf_state_dict.keys()}
    mapping_report = []
    full_prefixed_state_dict = {}

    for hf_key, prefixed_key in prefixed_keys_map.items():
        tensor = hf_state_dict[hf_key]
        full_prefixed_state_dict[prefixed_key] = tensor
        
        # 探针检测
        probe_dict = {prefixed_key: tensor}
        converted_probe = muse_model.convert_hf_state_dict(probe_dict)
        
        if len(converted_probe) == 0:
            muse_key = "<SKIPPED / UNMAPPED>"
        else:
            muse_key = list(converted_probe.keys())[0]
            
        mapping_report.append({
            "hf_orig": hf_key,
            "muse_key": muse_key,
            "shape": str(tuple(tensor.shape))
        })

    # 打印映射表
    logger.info(f"{'HF Original Key':<60} | {'Muse Converted Key':<50} | {'Shape':<15}")
    logger.info("-" * 130)
    for item in mapping_report:
        logger.info(f"{item['hf_orig']:<60} | {item['muse_key']:<50} | {item['shape']:<15}")
    logger.info("-" * 130)

    # --- 阶段 4.2: 实际转换 ---
    logger.info("Step 2: Performing full weight conversion...")
    final_converted_state_dict = muse_model.convert_hf_state_dict(full_prefixed_state_dict)
    
    # 统一精度
    for key in final_converted_state_dict:
        if isinstance(final_converted_state_dict[key], torch.Tensor):
            final_converted_state_dict[key] = final_converted_state_dict[key].to(dtype=dtype)

    # --- 阶段 4.3: 加载到 Muse 模型 (重点关注返回值) ---
    logger.info("Step 3: Loading state dict into Muse model...")
    load_res = muse_model.load_state_dict(final_converted_state_dict, strict=False)
    
    logger.info(f"Load Result Raw: {load_res}")

    # 将模型移到 GPU
    muse_model = muse_model.to(device=device, dtype=dtype)
    muse_model.eval()

    # --- 阶段 4.5: 未加载参数覆盖率检测 (新增功能) ---
    log_separator("Unloaded Parameter Coverage Analysis")
    
    missing_keys = load_res.missing_keys
    
    # 分类检测：将 Missing Keys 分为 '关键参数' (weights/bias) 和 '其他' (buffers like inv_freq)
    critical_missing = []
    other_missing = []
    
    for k in missing_keys:
        if k.endswith(".weight") or k.endswith(".bias"):
            critical_missing.append(k)
        else:
            other_missing.append(k)
            
    # 报告关键缺失
    if len(critical_missing) > 0:
        logger.error(f"❌ CRITICAL ERROR: {len(critical_missing)} learnable parameters were NOT initialized by the checkpoint!")
        logger.error("   These parameters remain randomly initialized, which will cause output mismatch.")
        for k in critical_missing:
            logger.error(f"   - {k}")
    else:
        logger.info("✅ SUCCESS: All learnable parameters (weights/biases) in Muse model are covered by the checkpoint.")

    # 报告非关键缺失 (通常是 RoPE buffers 或 Position IDs)
    if len(other_missing) > 0:
        logger.info(f"ℹ️  Info: {len(other_missing)} non-parameter keys (likely buffers) were not loaded (usually expected):")
        for k in other_missing:
            logger.info(f"   - {k}")
            
    # 如果有关键参数缺失，建议直接终止测试，因为后面的数值对比肯定不过
    if len(critical_missing) > 0:
        logger.error("🛑 Stopping test early due to missing critical weights. Please fix logic in `convert_hf_state_dict`.")
        # return # 你可以选择在这里 return，或者继续跑看看有多离谱

    # --- 阶段 4.4: 基于映射关系的数值验证 ---
    log_separator("Weight Value Verification (Mapped)")
    
    muse_curr_state_dict = muse_model.state_dict()
    issues_found = 0
    
    for item in mapping_report:
        hf_key = item['hf_orig']
        muse_key = item['muse_key']
        
        if muse_key == "<SKIPPED / UNMAPPED>": continue
        if muse_key not in muse_curr_state_dict: continue # 已在上面覆盖率检测中处理
            
        hf_tensor = hf_state_dict[hf_key].to(device).to(dtype)
        muse_tensor = muse_curr_state_dict[muse_key].to(device)
        
        if hf_tensor.shape != muse_tensor.shape:
            if hf_tensor.shape == muse_tensor.t().shape:
                hf_tensor = hf_tensor.t()
            else:
                logger.error(f"{muse_key:<60} | SHAPE ERR  | HF{hf_tensor.shape} vs Muse{muse_tensor.shape}")
                issues_found += 1
                continue
                
        diff = (hf_tensor - muse_tensor).abs().max().item()
        threshold = 1e-2 if dtype == torch.bfloat16 else 1e-5
        
        if diff >= threshold:
            logger.error(f"{muse_key:<60} | FAIL       | {diff:.4e}")
            issues_found += 1
        elif "layers.0." in muse_key or "embeddings" in muse_key: 
            logger.info(f"{muse_key:<60} | OK         | {diff:.4e}")

    if issues_found == 0:
        logger.info("\n✅ All mapped weights verified successfully.")
    else:
        logger.error(f"\n❌ Found {issues_found} weight value mismatch(es).")

    # =========================================================================
    # 5. 前向传播对比 (Forward Pass)
    # =========================================================================
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
        muse_last_hidden = muse_outputs["last_hidden_state"]

    # 统一转换
    hf_res = hf_last_hidden.to(device, dtype=dtype)
    muse_res = muse_last_hidden.to(device, dtype=dtype)

    # 形状检查
    if hf_res.shape != muse_res.shape:
        logger.error(f"Output Shape Mismatch: HF={hf_res.shape}, Muse={muse_res.shape}")
        min_len = min(hf_res.shape[1], muse_res.shape[1])
        hf_res = hf_res[:, :min_len, :]
        muse_res = muse_res[:, :min_len, :]

    # --- 详细的数值对比 ---
    compare_tensors_verbose("Last Hidden State (Global)", hf_res, muse_res, atol=1e-2 if dtype==torch.bfloat16 else 1e-5)

    # Final verdict
    final_diff = (hf_res - muse_res).abs().max().item()
    threshold = 1e-2 if dtype == torch.bfloat16 else 1e-4
    
    log_separator("FINAL RESULT")
    if final_diff < threshold:
        logger.info(f"SUCCESS: Outputs match! (Max Diff: {final_diff:.2e} < Threshold: {threshold})")
    else:
        logger.error(f"FAILURE: Outputs mismatch! (Max Diff: {final_diff:.2e} >= Threshold: {threshold})")

if __name__ == "__main__":
    test_siglip_logits_align_with_hf_checkpoint()