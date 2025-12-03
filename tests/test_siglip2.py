"""
Integration test to ensure Muse SigLIP matches Hugging Face logits.
Refined for Verbose Logging and Debugging.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel as HFSiglipVisionModel
from muse.config import SiglipVisionConfig

from muse.models.Siglip import SiglipVisionTransformer as SiglipVisionModel
from muse.training.common import set_default_dtype

# 配置简单的日志格式，方便nohup查看
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
    """Helper to format first n values of a tensor flat."""
    vals = t.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join([f"{x:.6f}" for x in vals]) + "]"

def log_separator(title: str):
    logger.info(f"\n{'='*80}")
    logger.info(f" {title.center(78)} ")
    logger.info(f"{'='*80}")

def compare_tensors_verbose(name: str, tensor_hf: torch.Tensor, tensor_muse: torch.Tensor, atol=1e-5):
    """
    Detailed comparison of two tensors.
    """
    t1 = tensor_hf.detach().float().cpu()
    t2 = tensor_muse.detach().float().cpu()
    
    if t1.shape != t2.shape:
        logger.error(f"❌ SHAPE MISMATCH [{name}]: HF={t1.shape} vs Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 判定是否匹配
    match_status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max Diff: {max_diff:.2e})"
    
    logger.info(f"{'-'*80}")
    logger.info(f"Layer: {name}")
    logger.info(f"Status: {match_status}")
    logger.info(f"Stats : MaxDiff={max_diff:.2e} | MeanDiff={mean_diff:.2e} | HF_Mean={t1.mean():.4f} | Muse_Mean={t2.mean():.4f}")
    
    if max_diff >= atol:
        # 如果不匹配，打印详细的前10个数值对比
        logger.info(f"Values Comparison (First 10 flattened):")
        logger.info(f"  HF  : {format_tensor_val(t1, 10)}")
        logger.info(f"  Muse: {format_tensor_val(t2, 10)}")
        
        # 打印差异最大的位置
        flat_diff = diff.flatten()
        top_val, top_idx = torch.topk(flat_diff, 1)
        idx_flat = top_idx[0].item()
        
        # 尝试还原多维坐标 (仅作参考)
        logger.info(f"  Worst diff index (flat): {idx_flat}, Val: {top_val[0].item():.6f}")
        logger.info(f"  HF Value at worst: {t1.flatten()[idx_flat]:.6f}")
        logger.info(f"  Muse Value at worst: {t2.flatten()[idx_flat]:.6f}")

def test_siglip_logits_align_with_hf_checkpoint():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # =========================================================================
    # 1. 配置路径
    # =========================================================================
    # 请确保路径正确
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"
    logger.info(f"Testing checkpoint: {checkpoint_dir}")

    # =========================================================================
    # 2. 加载 HF 模型
    # =========================================================================
    logger.info("Loading HF model...")
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"  # 自动放到GPU
    )
    hf_model.eval()

    # 获取设备和精度
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    logger.info(f"Reference Model Device: {device}, Dtype: {dtype}")

    # =========================================================================
    # 3. 初始化 Muse 模型
    # =========================================================================
    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()
    muse_config = _build_siglip_config(hf_config_dict)

    # 打印 Config 关键参数对比
    log_separator("Config Check")
    logger.info(f"{'Param':<30} | {'HF Value':<20} | {'Muse Value':<20}")
    check_params = ['image_size', 'patch_size', 'hidden_size', 'num_attention_heads', 'num_hidden_layers', 'layer_norm_eps']
    for p in check_params:
        hf_v = hf_config_dict.get(p, "N/A")
        muse_v = getattr(muse_config, p, "N/A")
        logger.info(f"{p:<30} | {str(hf_v):<20} | {str(muse_v):<20}")

    model_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    with set_default_dtype(model_dtype):
        muse_model = SiglipVisionModel(muse_config)

    # =========================================================================
    # 4. 权重转换与加载对比
    # =========================================================================
    log_separator("Weight Loading & Comparison")

    prefixed_state_dict = {}
    for key, value in hf_state_dict.items():
        prefixed_state_dict[f"siglip.vision_model.{key}"] = value

    # 执行转换
    converted_state_dict = muse_model.convert_hf_state_dict(prefixed_state_dict)
    
    # 将转换后的权重加载到 Muse 模型中 (在 CPU 上先加载，再转 device)
    # 注意：为了对比，我们手动把 converted_state_dict 转成正确的 dtype
    for key in converted_state_dict:
        if isinstance(converted_state_dict[key], torch.Tensor):
            converted_state_dict[key] = converted_state_dict[key].to(dtype=dtype)

    load_res = muse_model.load_state_dict(converted_state_dict, strict=False)
    logger.info(f"Load State Dict Result: {load_res}")
    
    # 将 Muse 模型移到 GPU
    muse_model = muse_model.to(device=device, dtype=dtype)
    muse_model.eval()

    # --- 详细对比每一层的权重数值 ---
    # 我们遍历 Muse 模型实际的 state_dict，反向查找或通过 convert 后的对应关系查找
    # 为了准确，我们利用 convert_hf_state_dict 的结果（作为 Muse 的 source）和 Muse 模型当前的参数对比
    # 同时，我们也可以尝试找到原始 HF 参数进行对比。
    
    # 建立映射: Muse Key -> HF Key (Reverse lookup is hard, so we assume correctness of converter and just compare loaded vs converted tensor)
    # 更直观的是：对比 converted_state_dict (理论值) 和 muse_model.state_dict() (实际值)
    # 以及重点抽查几个关键层对比 HF 原始 dict
    
    logger.info(f"\nComparing loaded weights layer by layer (Sampled)...")
    logger.info(f"{'Weight Name':<60} | {'Shape':<15} | {'Max Diff':<12} | {'Status'}")
    
    muse_curr_state_dict = muse_model.state_dict()
    
    # 遍历顺序
    sorted_keys = sorted(list(muse_curr_state_dict.keys()))
    
    issues_found = 0
    
    for key in sorted_keys:
        if key in converted_state_dict:
            target_w = converted_state_dict[key].to(device)
            current_w = muse_curr_state_dict[key]
            
            diff = (target_w - current_w).abs().max().item()
            status = "OK" if diff < 1e-5 else "FAIL"
            if diff >= 1e-5: issues_found += 1
            
            # 只打印有问题的或者每隔10层打印一次以节省日志
            if diff > 1e-5 or "embeddings" in key or "layers.0." in key or "layers.26." in key:
                 logger.info(f"{key:<60} | {str(tuple(current_w.shape)):<15} | {diff:.4e}   | {status}")
    
    if issues_found == 0:
        logger.info("✅ All weights loaded match the converted tensors exactly.")
    else:
        logger.error(f"❌ Found {issues_found} weight mismatches between converter output and loaded model.")

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
        # 尝试切片对齐以便查看数值
        min_len = min(hf_res.shape[1], muse_res.shape[1])
        hf_res = hf_res[:, :min_len, :]
        muse_res = muse_res[:, :min_len, :]
        logger.warning(f"  -> Sliced to {hf_res.shape} for value comparison")

    # --- 详细的数值对比 ---
    compare_tensors_verbose("Last Hidden State (Global)", hf_res, muse_res, atol=1e-2 if dtype==torch.bfloat16 else 1e-5)

    # --- Token 级详细对比 (表格形式) ---
    logger.info("\nDetailed Token Slices Comparison (First 5 dimensions):")
    logger.info(f"{'Location':<25} | {'HF Values':<40} | {'Muse Values':<40} | {'MaxDiff'}")
    
    # 定义要检查的位置: (batch, seq_idx)
    check_indices = [
        (0, 0, "First Token (CLS like)"),
        (0, 10, "Early Token"),
        (0, hf_res.shape[1]//2, "Middle Token"),
        (0, hf_res.shape[1]-1, "Last Token")
    ]
    
    for b_idx, s_idx, label in check_indices:
        if s_idx >= hf_res.shape[1]: continue
        
        hf_slice = hf_res[b_idx, s_idx, :5].float().cpu().numpy()
        muse_slice = muse_res[b_idx, s_idx, :5].float().cpu().numpy()
        
        diff_slice = np.abs(hf_slice - muse_slice).max()
        
        hf_str = ", ".join([f"{x:.4f}" for x in hf_slice])
        muse_str = ", ".join([f"{x:.4f}" for x in muse_slice])
        
        logger.info(f"{label:<25} | [{hf_str:<38}] | [{muse_str:<38}] | {diff_slice:.2e}")

    # --- 统计特征对比 ---
    logger.info(f"\nStatistics Summary:")
    logger.info(f"HF   -> Min: {hf_res.min():.4f}, Max: {hf_res.max():.4f}, Mean: {hf_res.mean():.4f}, Std: {hf_res.std():.4f}")
    logger.info(f"Muse -> Min: {muse_res.min():.4f}, Max: {muse_res.max():.4f}, Mean: {muse_res.mean():.4f}, Std: {muse_res.std():.4f}")

    # Final verdict
    final_diff = (hf_res - muse_res).abs().max().item()
    threshold = 1e-2 if dtype == torch.bfloat16 else 1e-4
    
    log_separator("FINAL RESULT")
    if final_diff < threshold:
        logger.info(f"SUCCESS: Outputs match! (Max Diff: {final_diff:.2e} < Threshold: {threshold})")
    else:
        logger.error(f"FAILURE: Outputs mismatch! (Max Diff: {final_diff:.2e} >= Threshold: {threshold})")

if __name__ == "__main__":
    # 默认只运行这一项最关键的测试
    test_siglip_logits_align_with_hf_checkpoint()