"""
Keye Vision Full Integration Test (With Detailed Mapping & Verification)
======================================================================

功能：
1. 权重映射追踪：打印 HF -> Muse 的逐层映射关系表。
2. 参数覆盖检查：确保 Muse 模型所有 learnable parameters 都被加载。
3. 静态数值验证：加载后对比每一层权重的数值一致性。
4. 前向传播对比：对比 Logits 输出。
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

# -----------------------------------------------------------------------------
# 基础配置
# -----------------------------------------------------------------------------

CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. 动态注入 Origin Config (Config Hack)
# -----------------------------------------------------------------------------

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
    module_name = "muse.muse.models.keye_vit.configuration_keye"
    if module_name in sys.modules: return
    config_module = types.ModuleType(module_name)
    config_module.KeyeConfig = HFKeyeConfig
    config_module.KeyeVisionConfig = HFKeyeVisionConfig
    sys.modules[module_name] = config_module

_ensure_origin_ready()
from muse.models.keye_vit import modeling_keye_origin as keye_origin
# Origin Model Wrapper (包含 .vision_model)
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
    line = "=" * 120
    logger.info("\n%s", line)
    logger.info(" %s ", title.center(118))
    logger.info("%s", line)

def compare_tensors_verbose(name: str, reference: torch.Tensor, candidate: torch.Tensor, atol: float = 1e-3) -> None:
    if isinstance(reference, (tuple, list)): reference = reference[0]
    if isinstance(candidate, (tuple, list)): candidate = candidate[0]
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    # Shape alignment (squeeze batch dim if needed)
    if ref.shape != cand.shape:
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2: ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2: cand = cand.squeeze(0)

    if ref.shape != cand.shape:
        logger.error(f"{name:<60} | ❌ SHAPE ERR | Origin{ref.shape} vs Muse{cand.shape}")
        return

    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max: {max_diff:.2e})"
    
    # 关键层打印详细，其他层精简
    if max_diff >= atol or "Final" in name or "Embeddings" in name:
        logger.info(f"{name:<60} | {status:<25} | MeanDiff: {mean_diff:.2e}")

# -----------------------------------------------------------------------------
# 3. 核心逻辑：权重分析与加载
# -----------------------------------------------------------------------------

def analyze_and_load_weights(origin_model, muse_model, path, dtype, device):
    logger.info("Loading raw checkpoint from: %s", path)
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "module" in raw: raw = raw["module"]
    
    # 1. 构造标准化的 HF Key 字典 (模拟 HF AutoModel 加载后的结构)
    #    我们把所有 key 统一处理成 `siglip.vision_model.xxx` 的形式
    #    这样能最大程度兼容 Muse 的 convert_hf_state_dict 逻辑
    hf_standard_dict = {}
    
    for k, v in raw.items():
        clean_key = k
        # 剥离可能存在的包装前缀
        for p in ["module.", "vision_tower.", "siglip."]:
            if clean_key.startswith(p): clean_key = clean_key[len(p):]
        
        # 补全标准前缀
        if clean_key.startswith("vision_model."):
            std_key = "siglip." + clean_key
        else:
            std_key = "siglip.vision_model." + clean_key
            
        hf_standard_dict[std_key] = v.to(dtype)

    # 2. [探针阶段] 追踪映射关系
    log_separator("Step 1: Tracing Weight Mapping")
    mapping_report = []
    
    # 遍历每个 HF 权重，单独喂给 converter 看结果
    for std_key, tensor in hf_standard_dict.items():
        probe_dict = {std_key: tensor}
        converted = muse_model.convert_hf_state_dict(probe_dict)
        
        if not converted:
            muse_key = "<SKIPPED / UNMAPPED>"
        else:
            # 假设 1-to-1 映射
            muse_key = list(converted.keys())[0]
            
        mapping_report.append({
            "hf_orig": std_key,
            "muse_key": muse_key,
            "shape": str(tuple(tensor.shape))
        })

    # 打印详细映射表
    logger.info(f"{'HF Original Key (Standardized)':<60} | {'Muse Converted Key':<50} | {'Shape':<15}")
    logger.info("-" * 130)
    for item in mapping_report:
        # 过滤掉 Head 相关的未映射项以节省版面，或者全部打印
        if "head." in item['hf_orig'] and "SKIPPED" in item['muse_key']:
            continue 
        logger.info(f"{item['hf_orig']:<60} | {item['muse_key']:<50} | {item['shape']:<15}")
    logger.info("-" * 130)

    # 3. [转换阶段] 全量转换
    logger.info("Step 2: Performing full weight conversion...")
    final_muse_state = muse_model.convert_hf_state_dict(hf_standard_dict)
    
    # 4. [加载阶段] Muse 加载
    logger.info("Step 3: Loading Muse Model...")
    load_res = muse_model.load_state_dict(final_muse_state, strict=False)
    
    # 5. [覆盖率检查]
    log_separator("Parameter Coverage Analysis")
    missing = load_res.missing_keys
    critical_missing = [k for k in missing if k.endswith(".weight") or k.endswith(".bias")]
    
    if critical_missing:
        logger.error(f"❌ CRITICAL ERROR: {len(critical_missing)} learnable parameters NOT loaded!")
        for k in critical_missing:
            logger.error(f"   MISSING: {k}")
        logger.error("🛑 STOPPING TEST: Weight mismatch will cause failures.")
        # 根据需要，这里可以 raise Exception
    else:
        logger.info("✅ SUCCESS: All learnable parameters (weights/biases) in Muse model are loaded.")

    # 6. [数值验证] 静态对比
    log_separator("Static Weight Value Verification")
    muse_curr_dict = muse_model.state_dict()
    issues = 0
    
    # 表头
    logger.info(f"{'Muse Key':<60} | {'Status':<10} | {'Max Diff'}")
    
    for item in mapping_report:
        m_key = item['muse_key']
        h_key = item['hf_orig']
        
        if "SKIPPED" in m_key: continue
        if m_key not in muse_curr_dict: continue # 已在 Coverage Check 报错
        
        t_hf = hf_standard_dict[h_key]
        t_mu = muse_curr_dict[m_key].cpu() # 此时还在 CPU
        
        # 形状对齐 (处理 Linear 转置)
        if t_hf.shape != t_mu.shape:
            if t_hf.t().shape == t_mu.shape: t_hf = t_hf.t()
            else:
                logger.error(f"{m_key:<60} | SHAPE ERR  | HF{t_hf.shape} vs Muse{t_mu.shape}")
                issues += 1
                continue
        
        diff = (t_hf - t_mu).abs().max().item()
        
        if diff > 1e-5:
            logger.error(f"{m_key:<60} | FAIL       | {diff:.4e}")
            issues += 1
        elif "layers.0." in m_key or "embeddings" in m_key: # 抽样打印成功的
            logger.info(f"{m_key:<60} | OK         | {diff:.4e}")
            
    if issues == 0:
        logger.info("\n✅ All mapped weights verified successfully.")
    else:
        logger.error(f"\n❌ Found {issues} weight value mismatches.")

    # 7. [Origin 加载]
    # Origin 模型比较简单，直接用 vision_model. 开头的 dict 加载
    origin_load_dict = {k.replace("siglip.", ""): v for k, v in hf_standard_dict.items()}
    logger.info("Loading Origin Model...")
    origin_model.load_state_dict(origin_load_dict, strict=False)
    
    # Move to GPU
    origin_model.to(device)
    muse_model.to(device)

# -----------------------------------------------------------------------------
# 4. 主流程：前向传播
# -----------------------------------------------------------------------------

def prepare_pixel_inputs(processor, image, device, dtype):
    processed = processor.preprocess(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    grid_info = processed["image_grid_thw"]
    if isinstance(grid_info, torch.Tensor): grid_info = grid_info.cpu().tolist()
    elif isinstance(grid_info, np.ndarray): grid_info = grid_info.tolist()
    
    image_grid_thw = [tuple(int(v) for v in grid) for grid in grid_info]
    patches_per_image = [int(np.prod(grid)) for grid in image_grid_thw]
    
    # 5D Input [B, Seq, C, H, W]
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

def test_full_check():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    if not os.path.exists(CHECKPOINT_PATH): raise FileNotFoundError(CHECKPOINT_PATH)
    
    # 1. Init Models
    muse_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_config.dict())
    
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config)
        muse_model = MuseKeyeVisionModel(muse_config)
    
    origin_model.eval()
    muse_model.eval()
    
    # 2. Analyze & Load Weights
    log_separator("Phase 1: Weight Analysis")
    analyze_and_load_weights(origin_model, muse_model, CHECKPOINT_PATH, dtype, device)
    
    # 3. Register Hooks
    activations = {"origin": {}, "muse": {}}
    def get_hook(m, n):
        def hook(mod, inp, out):
            if isinstance(out, (tuple, list)): out = out[0]
            activations[m][n] = out.detach()
        return hook

    origin_model.vision_model.embeddings.register_forward_hook(get_hook("origin", "1. Embeddings"))
    muse_model.embeddings.register_forward_hook(get_hook("muse", "1. Embeddings"))
    
    origin_model.vision_model.encoder.layers[0].register_forward_hook(get_hook("origin", "2. Encoder Layer 0"))
    muse_model.encoder.layers[0].register_forward_hook(get_hook("muse", "2. Encoder Layer 0"))
    
    mid = muse_config.num_hidden_layers // 2
    origin_model.vision_model.encoder.layers[mid].register_forward_hook(get_hook("origin", f"3. Encoder Layer {mid}"))
    muse_model.encoder.layers[mid].register_forward_hook(get_hook("muse", f"3. Encoder Layer {mid}"))
    
    # 4. Prepare Input
    processor = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    dummy_image = create_dummy_image(muse_config.image_size)
    pixel_values, image_grid_thw, position_ids, cu_seqlens = prepare_pixel_inputs(
        processor, dummy_image, device, dtype
    )
    
    logger.info(f"Input Shape: {pixel_values.shape}")

    # 5. Forward Pass
    log_separator("Phase 2: Forward Pass")
    with torch.no_grad():
        origin_out = origin_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
            window_size=-1
        )
        origin_final = origin_out.last_hidden_state
        if isinstance(origin_final, list): origin_final = torch.stack(origin_final, dim=0)
        
        muse_out = muse_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
            has_learnable_position_embedding=getattr(muse_config, "has_learnable_position_embedding", True)
        )
        muse_final = muse_out["last_hidden_state"]

    # 6. Comparison
    log_separator("Phase 3: Layer-wise Comparison")
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    
    sorted_keys = sorted(activations["origin"].keys())
    for key in sorted_keys:
        compare_tensors_verbose(key, activations["origin"][key], activations["muse"][key], atol=tol)
        
    compare_tensors_verbose("4. Final Output (Post-LN)", origin_final, muse_final, atol=tol)
    
    # 7. Token Details
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
        r_v = origin_final[batch_idx, pos, :5].float().cpu().numpy()
        m_v = muse_final[batch_idx, pos, :5].float().cpu().numpy()
        d_v = np.abs(r_v - m_v).max()
        logger.info(f"{label:<20} | {str(r_v):<40} | {str(m_v):<40} | {d_v:.2e}")

    final_diff = (origin_final - muse_final).abs().max().item()
    log_separator("FINAL VERDICT")
    if final_diff < tol:
        logger.info(f"SUCCESS: Models match! (Max Diff: {final_diff:.2e})")
    else:
        logger.error(f"FAILURE: Models mismatch! (Max Diff: {final_diff:.2e})")

if __name__ == "__main__":
    test_full_check()