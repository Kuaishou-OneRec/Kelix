"""
Keye Vision Layer-wise Debugging Script
=======================================

此脚本结合了 `test_keye.py` 的权重加载逻辑和 `test_siglip2.py` 的 Hook 调试逻辑。
用于定位 Origin 模型与 Muse 模型在第几层开始出现数值偏差。
"""

import logging
import os
import sys
import types
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from PIL import Image

# Muse imports
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.training.common import set_default_dtype

# -----------------------------------------------------------------------------
# 基础配置 & 日志
# -----------------------------------------------------------------------------

CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. 动态注入 Origin Config (保持原逻辑)
# -----------------------------------------------------------------------------

def _ensure_origin_config_module() -> None:
    module_name = "muse.muse.models.keye_vit.configuration_keye"
    if module_name in sys.modules:
        return

    config_module = types.ModuleType(module_name)

    class DummyKeyeConfig:
        def __init__(self, vision_config=None, **kwargs):
            self.vision_config = vision_config or KeyeVisionConfig()
            for key, value in kwargs.items():
                setattr(self, key, value)

    config_module.KeyeConfig = DummyKeyeConfig
    config_module.KeyeVisionConfig = KeyeVisionConfig
    sys.modules[module_name] = config_module

_ensure_origin_config_module()
from muse.models.keye_vit import modeling_keye_origin as keye_origin
OriginKeyeVisionModel = keye_origin.SiglipVisionTransformer

# -----------------------------------------------------------------------------
# 2. 辅助工具函数
# -----------------------------------------------------------------------------

def create_dummy_image(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)

def log_separator(title: str) -> None:
    line = "=" * 80
    logger.info("\n%s", line)
    logger.info(" %s ", title.center(78))
    logger.info("%s", line)

def format_tensor_val(tensor: torch.Tensor, n: int = 5) -> str:
    vals = tensor.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join(f"{x:.5f}" for x in vals) + "]"

def compare_tensors_verbose(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-3,
) -> None:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    # 处理 Tuple 输出 (HuggingFace 风格有时返回 tuple)
    if isinstance(reference, tuple):
        ref = reference[0].detach().float().cpu()
    if isinstance(candidate, tuple):
        cand = candidate[0].detach().float().cpu()

    if ref.shape != cand.shape:
        logger.error("❌ %s SHAPE MISMATCH: Origin=%s vs Muse=%s", name, ref.shape, cand.shape)
        return

    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max Diff: {max_diff:.2e})"

    logger.info("-" * 80)
    logger.info("Layer   : %s", name)
    logger.info("Status  : %s", status)
    logger.info(
        "Stats   : MaxDiff=%.3e | MeanDiff=%.3e",
        max_diff, mean_diff
    )

    if max_diff >= atol:
        logger.info("Origin vals : %s", format_tensor_val(ref, 5))
        logger.info("Muse vals   : %s", format_tensor_val(cand, 5))

# -----------------------------------------------------------------------------
# 3. 数据准备函数
# -----------------------------------------------------------------------------

def build_position_ids(image_grid_thw: List[Tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    seq_lens = [int(t * h * w) for t, h, w in image_grid_thw]
    seq_len = seq_lens[0]
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    return position_ids.unsqueeze(0).repeat(len(image_grid_thw), 1)

def build_cu_seqlens(image_grid_thw: List[Tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    seq_lens = [int(t * h * w) for t, h, w in image_grid_thw]
    cumsum = [0]
    for length in seq_lens:
        cumsum.append(cumsum[-1] + length)
    return torch.tensor(cumsum, dtype=torch.int32, device=device)

def prepare_pixel_inputs(
    processor: KeyeVisionImageProcessor,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]], torch.Tensor, torch.Tensor]:
    processed = processor.preprocess(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    grid_info = processed["image_grid_thw"]
    
    # 转换 grid info 格式
    if isinstance(grid_info, torch.Tensor):
        grid_info = grid_info.cpu().tolist()
    elif isinstance(grid_info, np.ndarray):
        grid_info = grid_info.tolist()
    image_grid_thw = [tuple(int(v) for v in grid) for grid in grid_info]

    # Batching logic
    patches_per_image = [int(np.prod(grid)) for grid in image_grid_thw]
    batched = []
    start = 0
    for count in patches_per_image:
        batched.append(pixel_values[start : start + count])
        start += count
    pixel_batch = torch.stack(batched, dim=0).to(device=device, dtype=dtype).contiguous()

    position_ids = build_position_ids(image_grid_thw, device)
    cu_seqlens = build_cu_seqlens(image_grid_thw, device)
    
    return pixel_batch, image_grid_thw, position_ids, cu_seqlens

# -----------------------------------------------------------------------------
# 4. 权重处理逻辑 (复用自原脚本)
# -----------------------------------------------------------------------------

def load_checkpoint_and_convert(path: str) -> Tuple[Dict, Dict]:
    logger.info("Loading checkpoint from: %s", path)
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("module", "state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break

    # 1. Extract Origin Keys
    origin_state = {}
    prefixes = ("module.", "model.", "state_dict.", "vision_tower.", "vision_backbone.", "siglip.")
    for key, value in raw.items():
        new_key = key
        for prefix in prefixes:
            while new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        
        if "vision_model." in new_key:
            new_key = new_key.split("vision_model.", 1)[1]
        elif new_key.startswith("vision_model."):
            new_key = new_key[len("vision_model.") :]

        # 保留 head 以便完整加载 Origin 模型
        if new_key.startswith(("embeddings.", "encoder.", "post_layernorm.", "ln_post.", "head.")):
            origin_state[new_key] = value

    # 2. Convert to Muse Keys
    muse_state = {}
    for key, value in origin_state.items():
        if key.startswith("head."):
            continue # Muse 无 head
        new_key = key.replace("post_layernorm.", "ln_post.")
        if "encoder.layers." in new_key:
            new_key = (
                new_key.replace("self_attn", "attn")
                .replace("layer_norm1", "sa_norm")
                .replace("layer_norm2", "mlp_norm")
                .replace("out_proj", "output_proj")
                .replace("mlp.fc1", "mlp.gate_proj")
                .replace("mlp.fc2", "mlp.down_proj")
            )
        muse_state[new_key] = value

    return origin_state, muse_state

# -----------------------------------------------------------------------------
# 5. 主测试逻辑：Layer-by-Layer Debug
# -----------------------------------------------------------------------------

def test_layer_by_layer():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # --- Config ---
    config = KeyeVisionConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 检测是否支持 bf16
    bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if bf16_supported else torch.float32
    
    logger.info(f"Running on {device} with {dtype}")

    # --- Initialize Models ---
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(config)
        muse_model = MuseKeyeVisionModel(config)
    
    origin_model.eval()
    muse_model.eval()

    # --- Load Weights ---
    origin_state, muse_state = load_checkpoint_and_convert(CHECKPOINT_PATH)
    
    def _to_dtype(d):
        return {k: v.to(dtype=dtype) for k,v in d.items()}

    logger.info("Loading Origin Weights...")
    origin_model.load_state_dict(_to_dtype(origin_state), strict=False)
    
    logger.info("Loading Muse Weights...")
    muse_model.load_state_dict(_to_dtype(muse_state), strict=False)

    origin_model.to(device)
    muse_model.to(device)

    # --- Hook Registration ---
    log_separator("Registering Hooks")
    activations = {"origin": {}, "muse": {}}

    def get_hook(model_type, layer_name):
        def hook(module, input, output):
            # 处理 tuple 输出 (hidden, attn_weights)
            if isinstance(output, tuple):
                output = output[0]
            activations[model_type][layer_name] = output.detach()
        return hook

    # Hook Points:
    # 1. Embeddings (Patch + Pos)
    # Origin 结构通常为: vision_model.embeddings
    # Muse 结构通常为: embeddings
    origin_model.vision_model.embeddings.register_forward_hook(get_hook("origin", "embeddings"))
    muse_model.embeddings.register_forward_hook(get_hook("muse", "embeddings"))

    # 2. Layer 0 (Check Attention & MLP of first block)
    origin_model.vision_model.encoder.layers[0].register_forward_hook(get_hook("origin", "layer_0"))
    muse_model.encoder.layers[0].register_forward_hook(get_hook("muse", "layer_0"))

    # 3. Middle Layer (Check accumulation)
    mid_idx = config.num_hidden_layers // 2
    origin_model.vision_model.encoder.layers[mid_idx].register_forward_hook(get_hook("origin", f"layer_{mid_idx}"))
    muse_model.encoder.layers[mid_idx].register_forward_hook(get_hook("muse", f"layer_{mid_idx}"))

    # 4. Last Layer (Before final LN)
    last_idx = config.num_hidden_layers - 1
    origin_model.vision_model.encoder.layers[last_idx].register_forward_hook(get_hook("origin", "layer_last"))
    muse_model.encoder.layers[last_idx].register_forward_hook(get_hook("muse", "layer_last"))

    # --- Prepare Inputs ---
    processor = KeyeVisionImageProcessor(patch_size=config.patch_size)
    dummy_image = create_dummy_image(config.image_size)
    
    pixel_values, image_grid_thw, position_ids, cu_seqlens = prepare_pixel_inputs(
        processor, dummy_image, device, dtype
    )

    logger.info("Inputs Prepared. Grid: %s", image_grid_thw)

    # --- Forward Pass ---
    log_separator("Running Forward Pass")
    
    with torch.no_grad():
        # Origin Forward
        origin_out = origin_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True, 
        )
        origin_final = origin_out.last_hidden_state
        if isinstance(origin_out.last_hidden_state, list): # 有时返回 list
             origin_final = torch.stack(origin_out.last_hidden_state, dim=0)

        # Muse Forward
        muse_out = muse_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
        )
        muse_final = muse_out["last_hidden_state"]

    # --- Comparisons ---
    log_separator("Layer-wise Analysis")
    
    # 设定容差：bf16 下 1e-2 是合理的，fp32 可用 1e-4
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    # 1. Compare Embeddings
    # 关键点：如果这里挂了，查 PatchEmbed 的实现、Position Embedding 的插值/加法。
    compare_tensors_verbose("1. Embeddings", 
                           activations["origin"]["embeddings"], 
                           activations["muse"]["embeddings"], atol=tol)

    # 2. Compare Layer 0
    # 关键点：如果 Embeddings 对了但这里挂了，查 Attention (RoPE, QK Norm, Head拆分) 或 MLP (Act func)。
    compare_tensors_verbose("2. Encoder Layer 0", 
                           activations["origin"]["layer_0"], 
                           activations["muse"]["layer_0"], atol=tol)

    # 3. Compare Mid Layer
    compare_tensors_verbose(f"3. Encoder Layer {mid_idx}", 
                           activations["origin"][f"layer_{mid_idx}"], 
                           activations["muse"][f"layer_{mid_idx}"], atol=tol)

    # 4. Compare Last Layer Output
    compare_tensors_verbose("4. Encoder Last Layer", 
                           activations["origin"]["layer_last"], 
                           activations["muse"]["layer_last"], atol=tol)

    # 5. Final Output (Post-LN)
    # 关键点：如果 Last Layer 对了但这里挂了，查 Final LayerNorm 的 eps 或权重。
    compare_tensors_verbose("5. Final Output (Post-LN)", origin_final, muse_final, atol=tol)


if __name__ == "__main__":
    test_layer_by_layer()