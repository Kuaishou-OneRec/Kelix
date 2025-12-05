"""
Keye Vision Full Integration Test & Debugging Script
====================================================

功能：
1. 验证 Muse Keye Vision 模型与 Origin 模型的完全对齐。
2. 包含 Config 修复、5D 输入构造、权重映射追踪、缺失参数报警、逐层权重对比及前向传播对比。
"""

import logging
import os
import sys
import types
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import PretrainedConfig

# Muse imports
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.training.common import set_default_dtype

# -----------------------------------------------------------------------------
# 基础配置
# -----------------------------------------------------------------------------

# 请确保路径正确
CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. 动态注入 Origin Config (解决 PretrainedConfig 报错)
# -----------------------------------------------------------------------------

class HFKeyeVisionConfig(PretrainedConfig):
    model_type = "siglip_vision"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 模拟 Pydantic 行为，允许属性访问
        for k, v in kwargs.items():
            setattr(self, k, v)

class HFKeyeConfig(PretrainedConfig):
    model_type = "keye"
    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_config = vision_config

def _ensure_origin_config_module() -> None:
    module_name = "muse.muse.models.keye_vit.configuration_keye"
    if module_name in sys.modules:
        return

    config_module = types.ModuleType(module_name)
    config_module.KeyeConfig = HFKeyeConfig
    config_module.KeyeVisionConfig = HFKeyeVisionConfig
    sys.modules[module_name] = config_module

_ensure_origin_config_module()
# 导入 Origin 模型
from muse.models.keye_vit import modeling_keye_origin as keye_origin
# 使用外层 Wrapper 类，它包含 .vision_model
OriginKeyeVisionModel = keye_origin.SiglipVisionModel 

# -----------------------------------------------------------------------------
# 2. 辅助函数
# -----------------------------------------------------------------------------

def create_dummy_image(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)

def format_tensor_val(tensor: torch.Tensor, n: int = 5) -> str:
    vals = tensor.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join(f"{x:.5f}" for x in vals) + "]"

def log_separator(title: str) -> None:
    line = "=" * 100
    logger.info("\n%s", line)
    logger.info(" %s ", title.center(98))
    logger.info("%s", line)

def compare_tensors_verbose(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-3,
) -> None:
    if isinstance(reference, (tuple, list)): reference = reference[0]
    if isinstance(candidate, (tuple, list)): candidate = candidate[0]

    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    # 简单的形状对齐 (处理 Batch dim)
    if ref.shape != cand.shape:
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2:
            ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2:
            cand = cand.squeeze(0)
    
    if ref.shape != cand.shape:
        # 尝试转置 (针对 Linear 权重)
        if ref.shape == cand.t().shape:
            ref = ref.t()
        else:
            logger.error(f"❌ {name:<60} | SHAPE ERR | Origin{ref.shape} vs Muse{cand.shape}")
            return

    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max: {max_diff:.2e})"

    if max_diff >= atol:
        logger.info(f"{name:<60} | {status:<25} | MeanDiff: {mean_diff:.2e}")
    else:
        # 为了不刷屏，Match 的只打印简略信息，除非是关键层
        if "Final" in name or "Embeddings" in name:
            logger.info(f"{name:<60} | {status:<25} | MaxDiff: {max_diff:.2e}")

# -----------------------------------------------------------------------------
# 3. 数据准备 (5D Input [Batch, Seq, C, H, W])
# -----------------------------------------------------------------------------

def prepare_pixel_inputs(
    processor: KeyeVisionImageProcessor,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]], torch.Tensor, torch.Tensor]:
    processed = processor.preprocess(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    grid_info = processed["image_grid_thw"]
    
    if isinstance(grid_info, torch.Tensor):
        grid_info = grid_info.cpu().tolist()
    elif isinstance(grid_info, np.ndarray):
        grid_info = grid_info.tolist()
    image_grid_thw = [tuple(int(v) for v in grid) for grid in grid_info]

    patches_per_image = [int(np.prod(grid)) for grid in image_grid_thw]
    
    # 构造 5D tensor: [Batch, Seq_in_Batch, Channels, H, W]
    batched = []
    start = 0
    for count in patches_per_image:
        batched.append(pixel_values[start : start + count])
        start += count
    
    pixel_batch = torch.stack(batched, dim=0).to(device=device, dtype=dtype).contiguous()
    
    seq_len = patches_per_image[0]
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).repeat(len(image_grid_thw), 1)

    cumsum = [0]
    for length in patches_per_image:
        cumsum.append(cumsum[-1] + length)
    cu_seqlens = torch.tensor(cumsum, dtype=torch.int32, device=device)
    
    return pixel_batch, image_grid_thw, position_ids, cu_seqlens

# -----------------------------------------------------------------------------
# 4. 核心：权重映射与加载逻辑
# -----------------------------------------------------------------------------

def convert_and_load_weights(origin_model, muse_model, path, dtype, device):
    logger.info("Loading raw checkpoint from: %s", path)
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        if "module" in raw: raw = raw["module"]
        # 有些 checkpoint 可能还要剥离其他层
    
    origin_state = {}
    muse_target_state = {}
    mapping_report = []

    # 1. 清洗 Origin Key 并构建映射
    prefixes_to_strip = ["module.", "vision_tower.", "siglip.", "vision_model."]
    
    for k, v in raw.items():
        # 清洗 Key
        clean_key = k
        for p in prefixes_to_strip:
            if clean_key.startswith(p): clean_key = clean_key[len(p):]
        
        # Origin 需要 vision_model. 前缀
        origin_key = f"vision_model.{clean_key}"
        origin_state[origin_key] = v.to(dtype)

        # 构建 Muse Key (扁平化 + 映射)
        muse_key = clean_key # 去掉 vision_model. 的版本
        
        # Skip Head
        if muse_key.startswith("head."):
            continue
            
        # [关键映射逻辑]
        if muse_key.startswith("post_layernorm."):
            muse_key = muse_key.replace("post_layernorm.", "ln_post.")
        elif muse_key.startswith("encoder.layers."):
            muse_key = (
                muse_key.replace("self_attn", "attn")
                .replace("layer_norm1", "sa_norm")
                .replace("layer_norm2", "mlp_norm")
                .replace("out_proj", "output_proj")
                # [FIX] 修复 MLP 映射
                .replace("mlp.fc1", "mlp.w1")
                .replace("mlp.fc2", "mlp.w2")
            )
            
        muse_target_state[muse_key] = v.to(dtype)
        mapping_report.append((origin_key, muse_key, tuple(v.shape)))

    # 2. 打印映射预览
    logger.info(f"{'Origin Key':<50} | {'Muse Key':<50} | {'Shape'}")
    logger.info("-" * 120)
    for o, m, s in mapping_report[:5]: # 只打前5个
        logger.info(f"{o:<50} | {m:<50} | {s}")
    logger.info(f"... (Total {len(mapping_report)} keys mapped)")

    # 3. 加载 Origin
    logger.info("Loading Origin Model...")
    origin_model.load_state_dict(origin_state, strict=False)

    # 4. 加载 Muse
    logger.info("Loading Muse Model...")
    load_res = muse_model.load_state_dict(muse_target_state, strict=False)
    
    # 5. [关键] 参数覆盖率检查
    log_separator("Parameter Coverage Check")
    missing_keys = load_res.missing_keys
    critical_missing = [k for k in missing_keys if k.endswith(".weight") or k.endswith(".bias")]
    
    if critical_missing:
        logger.error(f"❌ CRITICAL: {len(critical_missing)} learnable parameters NOT loaded!")
        for k in critical_missing:
            logger.error(f"   MISSING: {k}")
        # 如果你想在此处终止，取消注释下一行
        # raise RuntimeError("Critical weights missing in Muse model!")
    else:
        logger.info("✅ SUCCESS: All learnable parameters (weights/biases) in Muse model are loaded.")

    # 6. 静态权重数值校验
    log_separator("Static Weight Verification")
    issues = 0
    muse_curr_dict = muse_model.state_dict()
    
    for o_key, m_key, _ in mapping_report:
        if m_key not in muse_curr_dict: continue # 已经在上面报错了
        
        t_o = origin_state[o_key]
        t_m = muse_curr_dict[m_key].cpu()
        
        # 形状对齐
        if t_o.shape != t_m.shape:
             # 有些权重可能做了 Transpose (如 Linear)
             if t_o.t().shape == t_m.shape: t_o = t_o.t()
        
        diff = (t_o - t_m).abs().max().item()
        if diff > 1e-5:
            logger.error(f"{m_key:<60} | VAL MISMATCH | Diff: {diff:.4e}")
            issues += 1
            
    if issues == 0:
        logger.info("✅ All mapped weights match perfectly in value.")
    else:
        logger.error(f"❌ Found {issues} weight value mismatches.")

    # Move to device
    origin_model.to(device)
    muse_model.to(device)
    return mapping_report

# -----------------------------------------------------------------------------
# 5. 主测试逻辑
# -----------------------------------------------------------------------------

def test_full_pipeline():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # 1. Configs
    muse_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_config.dict())
    
    logger.info(f"Running on {device} with {dtype}")

    with set_default_dtype(dtype):
        # 初始化模型
        origin_model = OriginKeyeVisionModel(origin_config)
        muse_model = MuseKeyeVisionModel(muse_config)
    
    origin_model.eval()
    muse_model.eval()

    # 2. 加载权重并检查
    log_separator("Weight Loading & Analysis")
    convert_and_load_weights(origin_model, muse_model, CHECKPOINT_PATH, dtype, device)

    # 3. 注册 Hooks
    activations = {"origin": {}, "muse": {}}
    def get_hook(model_type, layer_name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)): output = output[0]
            activations[model_type][layer_name] = output.detach()
        return hook

    # Hook Points
    origin_model.vision_model.embeddings.register_forward_hook(get_hook("origin", "embeddings"))
    muse_model.embeddings.register_forward_hook(get_hook("muse", "embeddings"))

    origin_model.vision_model.encoder.layers[0].register_forward_hook(get_hook("origin", "layer_0"))
    muse_model.encoder.layers[0].register_forward_hook(get_hook("muse", "layer_0"))

    mid_idx = muse_config.num_hidden_layers // 2
    origin_model.vision_model.encoder.layers[mid_idx].register_forward_hook(get_hook("origin", f"layer_{mid_idx}"))
    muse_model.encoder.layers[mid_idx].register_forward_hook(get_hook("muse", f"layer_{mid_idx}"))

    last_idx = muse_config.num_hidden_layers - 1
    origin_model.vision_model.encoder.layers[last_idx].register_forward_hook(get_hook("origin", "layer_last"))
    muse_model.encoder.layers[last_idx].register_forward_hook(get_hook("muse", "layer_last"))

    # 4. 准备输入
    processor = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    dummy_image = create_dummy_image(muse_config.image_size)
    
    pixel_values, image_grid_thw, position_ids, cu_seqlens = prepare_pixel_inputs(
        processor, dummy_image, device, dtype
    )
    
    logger.info(f"Input Pixel Shape: {pixel_values.shape}") 

    # 5. 前向传播
    log_separator("Running Forward Pass")
    
    with torch.no_grad():
        # --- Origin Forward ---
        origin_out = origin_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True, 
            window_size=-1 
        )
        
        if hasattr(origin_out, "last_hidden_state"):
            origin_final = origin_out.last_hidden_state
        else:
            origin_final = origin_out
            
        if isinstance(origin_final, list): 
             origin_final = torch.stack(origin_final, dim=0)
        elif isinstance(origin_final, tuple):
             origin_final = origin_final[0]

        # --- Muse Forward ---
        muse_out = muse_model(
            pixel_values=pixel_values, 
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
            has_learnable_position_embedding=getattr(muse_config, "has_learnable_position_embedding", True)
        )
        muse_final = muse_out["last_hidden_state"]

    # 6. 比较
    log_separator("Layer-wise Analysis")
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-3

    compare_tensors_verbose("1. Embeddings", activations["origin"]["embeddings"], activations["muse"]["embeddings"], atol=tol)
    compare_tensors_verbose("2. Encoder Layer 0", activations["origin"]["layer_0"], activations["muse"]["layer_0"], atol=tol)
    compare_tensors_verbose(f"3. Encoder Layer {mid_idx}", activations["origin"][f"layer_{mid_idx}"], activations["muse"][f"layer_{mid_idx}"], atol=tol)
    compare_tensors_verbose("4. Encoder Last Layer", activations["origin"]["layer_last"], activations["muse"]["layer_last"], atol=tol)
    compare_tensors_verbose("5. Final Output (Post-LN)", origin_final, muse_final, atol=tol)
    
    # 7. 详细 Token 采样
    log_separator("Token Sample Check")
    batch_idx = 0
    seq_len = origin_final.shape[1]
    positions = [
        (0, "First Token"),
        (min(10, seq_len - 1), "Token 10"),
        (seq_len // 2, "Middle Token"),
        (seq_len - 1, "Last Token"),
    ]
    
    logger.info(f"{'Position':<20} | {'Origin (First 5)':<40} | {'Muse (First 5)':<40} | {'MaxDiff'}")
    logger.info("-" * 120)
    
    for pos, label in positions:
        ref_vals = origin_final[batch_idx, pos, :5].float().cpu().numpy()
        muse_vals = muse_final[batch_idx, pos, :5].float().cpu().numpy()
        diff_val = np.max(np.abs(ref_vals - muse_vals))
        
        r_str = ", ".join(f"{x:.4f}" for x in ref_vals)
        m_str = ", ".join(f"{x:.4f}" for x in muse_vals)
        
        logger.info(f"{label:<20} | [{r_str:<38}] | [{m_str:<38}] | {diff_val:.2e}")

    final_diff = (origin_final - muse_final).abs().max().item()
    log_separator("FINAL VERDICT")
    if final_diff < tol:
        logger.info(f"SUCCESS: Outputs match! (Max Diff: {final_diff:.2e} < Threshold: {tol})")
    else:
        logger.error(f"FAILURE: Outputs mismatch! (Max Diff: {final_diff:.2e} >= Threshold: {tol})")


if __name__ == "__main__":
    test_full_pipeline()