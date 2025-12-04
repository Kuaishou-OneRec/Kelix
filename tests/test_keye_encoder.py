"""
Keye Vision Layer-wise Debugging Script (Corrected)
===================================================

修复内容：
1. 使用 SiglipVisionModel 作为 Origin 类，匹配包含 vision_model 的结构。
2. 调整权重 Key 的映射逻辑，确保 Origin 模型能正确加载 vision_model 前缀的参数。
3. 确保输入 Pixel Values 是 5D 格式 (Batch, Seq, Channel, Height, Width)。
"""

import logging
import os
import sys
import types
from typing import Dict, List, Tuple, Union

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

# 【修改点 1】使用最外层的 Model 类，它包含 .vision_model 属性
OriginKeyeVisionModel = keye_origin.SiglipVisionModel 

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
    # 处理 Tuple 输出
    if isinstance(reference, (tuple, list)):
        reference = reference[0]
    if isinstance(candidate, (tuple, list)):
        candidate = candidate[0]

    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    # 形状对齐尝试：Origin 可能是 (B, L, D)，Muse 可能是 (B, L, D) 或 (L, B, D)
    if ref.shape != cand.shape:
        # 尝试 squeeze (Muse 有时输出 shape [1, L, D] 而 Origin [L, D])
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2:
            ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2:
            cand = cand.squeeze(0)
    
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
    pixel_values = processed["pixel_values"] # [N_patches, C, H, W]
    grid_info = processed["image_grid_thw"]
    
    if isinstance(grid_info, torch.Tensor):
        grid_info = grid_info.cpu().tolist()
    elif isinstance(grid_info, np.ndarray):
        grid_info = grid_info.tolist()
    image_grid_thw = [tuple(int(v) for v in grid) for grid in grid_info]

    patches_per_image = [int(np.prod(grid)) for grid in image_grid_thw]
    batched = []
    start = 0
    for count in patches_per_image:
        batched.append(pixel_values[start : start + count])
        start += count
    
    # 堆叠为 Batch
    pixel_batch = torch.stack(batched, dim=0).to(device=device, dtype=dtype).contiguous()
    # [Batch, Seq_Patches, C, H, W] -> 这是 Origin 代码 Forward 要求的 5D 格式
    
    position_ids = build_position_ids(image_grid_thw, device)
    cu_seqlens = build_cu_seqlens(image_grid_thw, device)
    
    return pixel_batch, image_grid_thw, position_ids, cu_seqlens

# -----------------------------------------------------------------------------
# 4. 权重处理逻辑
# -----------------------------------------------------------------------------

def load_checkpoint_and_convert(path: str) -> Tuple[Dict, Dict]:
    logger.info("Loading checkpoint from: %s", path)
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("module", "state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break

    origin_state = {}
    muse_state = {}

    prefixes = ("module.", "model.", "state_dict.", "vision_tower.", "vision_backbone.", "siglip.")
    
    for key, value in raw.items():
        # 清理通用前缀，找到 "vision_model" 这一层
        clean_key = key
        for prefix in prefixes:
            while clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        
        # --- 1. Origin Key 构建 ---
        # OriginKeyeVisionModel 是 SiglipVisionModel，它包含 self.vision_model
        # 所以它的 state_dict 期望的 key 应该是 "vision_model.embeddings.weight" 等
        
        # 此时 clean_key 可能是 "vision_model.embeddings.weight" (理想)
        # 或者可能是 "embeddings.weight" (如果 checkpoint 没存 vision_model 前缀)
        
        origin_key = clean_key
        # 如果 checkpoint 里有 vision_model 前缀，保留它
        # 如果 checkpoint 直接是 embeddings 开头，我们需要补上 vision_model. 前缀
        if not origin_key.startswith("vision_model."):
            origin_key = "vision_model." + origin_key
            
        # 过滤掉非 Vision 参数 (比如 text model 的)
        if "vision_model." in origin_key:
             origin_state[origin_key] = value

        # --- 2. Muse Key 构建 ---
        # Muse 模型 (KeyeVisionTransformer) 结构是平铺的，没有 vision_model 前缀
        # 或者是 self.embeddings, self.encoder
        
        # 去掉 vision_model. 前缀来构建 Muse key
        muse_key_base = origin_key.replace("vision_model.", "")
        
        if muse_key_base.startswith("head."):
            continue # Muse 无 head

        new_muse_key = muse_key_base.replace("post_layernorm.", "ln_post.")
        if "encoder.layers." in new_muse_key:
            new_muse_key = (
                new_muse_key.replace("self_attn", "attn")
                .replace("layer_norm1", "sa_norm")
                .replace("layer_norm2", "mlp_norm")
                .replace("out_proj", "output_proj")
                .replace("mlp.fc1", "mlp.gate_proj")
                .replace("mlp.fc2", "mlp.down_proj")
            )
        muse_state[new_muse_key] = value

    return origin_state, muse_state

# -----------------------------------------------------------------------------
# 5. 主测试逻辑
# -----------------------------------------------------------------------------

def test_layer_by_layer():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    config = KeyeVisionConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if bf16_supported else torch.float32
    
    logger.info(f"Running on {device} with {dtype}")

    with set_default_dtype(dtype):
        # Origin: SiglipVisionModel (Wrapper)
        origin_model = OriginKeyeVisionModel(config)
        # Muse: KeyeVisionTransformer (Inner)
        muse_model = MuseKeyeVisionModel(config)
    
    origin_model.eval()
    muse_model.eval()

    origin_state, muse_state = load_checkpoint_and_convert(CHECKPOINT_PATH)
    
    def _to_dtype(d):
        return {k: v.to(dtype=dtype) for k,v in d.items()}

    logger.info(f"Origin Params: {len(origin_state)}, Muse Params: {len(muse_state)}")
    
    # 加载权重
    load_res_origin = origin_model.load_state_dict(_to_dtype(origin_state), strict=False)
    # logger.info(f"Origin Missing: {load_res_origin.missing_keys}")
    
    load_res_muse = muse_model.load_state_dict(_to_dtype(muse_state), strict=False)
    # logger.info(f"Muse Missing: {load_res_muse.missing_keys}")

    origin_model.to(device)
    muse_model.to(device)

    # --- Hook Registration ---
    log_separator("Registering Hooks")
    activations = {"origin": {}, "muse": {}}

    def get_hook(model_type, layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[model_type][layer_name] = output.detach()
        return hook

    # 现在 origin_model 是 SiglipVisionModel，它有 vision_model 属性
    # Muse KeyeVisionTransformer 直接有 embeddings 属性
    
    # 1. Embeddings
    origin_model.vision_model.embeddings.register_forward_hook(get_hook("origin", "embeddings"))
    muse_model.embeddings.register_forward_hook(get_hook("muse", "embeddings"))

    # 2. Layer 0
    origin_model.vision_model.encoder.layers[0].register_forward_hook(get_hook("origin", "layer_0"))
    muse_model.encoder.layers[0].register_forward_hook(get_hook("muse", "layer_0"))

    # 3. Middle Layer
    mid_idx = config.num_hidden_layers // 2
    origin_model.vision_model.encoder.layers[mid_idx].register_forward_hook(get_hook("origin", f"layer_{mid_idx}"))
    muse_model.encoder.layers[mid_idx].register_forward_hook(get_hook("muse", f"layer_{mid_idx}"))

    # 4. Last Layer
    last_idx = config.num_hidden_layers - 1
    origin_model.vision_model.encoder.layers[last_idx].register_forward_hook(get_hook("origin", "layer_last"))
    muse_model.encoder.layers[last_idx].register_forward_hook(get_hook("muse", "layer_last"))

    # --- Prepare Inputs ---
    processor = KeyeVisionImageProcessor(patch_size=config.patch_size)
    dummy_image = create_dummy_image(config.image_size)
    
    # pixel_values shape: [B, Seq, C, H, W]
    pixel_values, image_grid_thw, position_ids, cu_seqlens = prepare_pixel_inputs(
        processor, dummy_image, device, dtype
    )

    # --- Forward Pass ---
    log_separator("Running Forward Pass")
    
    with torch.no_grad():
        # Origin Forward
        # 传递 5D pixel_values, origin code 会在内部处理
        origin_out = origin_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True, 
        )
        origin_final = origin_out.last_hidden_state
        
        # 适配 Origin 的输出格式 (Sample pooling)
        if isinstance(origin_final, list): 
             # Origin 返回的是 list of tensors (每个 sample 一个)
             origin_final = torch.stack(origin_final, dim=0)

        # Muse Forward
        # 注意：Muse 可能期望 5D 或 4D 输入，根据你的实现。
        # 如果 Muse ImageProcessor 输出 4D，但这里为了 Origin 变成了 5D，
        # 你的 Muse 模型如果不支持 5D 可能会报错。
        # 如果报错，请尝试: muse_input = pixel_values.flatten(0, 1) 
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
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    compare_tensors_verbose("1. Embeddings", 
                           activations["origin"]["embeddings"], 
                           activations["muse"]["embeddings"], atol=tol)

    compare_tensors_verbose("2. Encoder Layer 0", 
                           activations["origin"]["layer_0"], 
                           activations["muse"]["layer_0"], atol=tol)

    compare_tensors_verbose(f"3. Encoder Layer {mid_idx}", 
                           activations["origin"][f"layer_{mid_idx}"], 
                           activations["muse"][f"layer_{mid_idx}"], atol=tol)

    compare_tensors_verbose("4. Encoder Last Layer", 
                           activations["origin"]["layer_last"], 
                           activations["muse"]["layer_last"], atol=tol)

    compare_tensors_verbose("5. Final Output (Post-LN)", origin_final, muse_final, atol=tol)


if __name__ == "__main__":
    test_layer_by_layer()