"""
Keye Vision Layer-wise Debugging Script (Final Fix)
===================================================

功能：
1. 解决 Config 类型检查报错 (构造 HF 兼容 Config)。
2. 对齐 Origin 和 Muse 的权重加载逻辑。
3. 逐层 (Embeddings -> Layer0 -> Mid -> Last -> Final) 比较输出。
"""

import logging
import os
import sys
import types
from typing import Dict, List, Tuple, Any, Union

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
# 1. 动态注入 Origin Config (解决 ValueError)
# -----------------------------------------------------------------------------

# 定义一个符合 HuggingFace 标准的 Config 类
class HFKeyeVisionConfig(PretrainedConfig):
    model_type = "siglip_vision"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 允许通过属性访问字典中的值，模仿 Pydantic 行为
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
    # 注入假的 Config 类
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
    # 固定种子生成随机图
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
    # 解包 tuple/list
    if isinstance(reference, (tuple, list)): reference = reference[0]
    if isinstance(candidate, (tuple, list)): candidate = candidate[0]

    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    # 简单的形状对齐 (Squeeze batch dim if needed)
    if ref.shape != cand.shape:
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
    logger.info("Stats   : MaxDiff=%.3e | MeanDiff=%.3e", max_diff, mean_diff)

    if max_diff >= atol:
        logger.info("Origin vals : %s", format_tensor_val(ref, 5))
        logger.info("Muse vals   : %s", format_tensor_val(cand, 5))

# -----------------------------------------------------------------------------
# 3. 数据准备 (5D Input)
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
    
    # 手动构建 5D tensor: [Batch, Seq_in_Batch, Channels, H, W]
    # 注意：这里假设 dummy image 只有一张图，且被切分成了 patches
    batched = []
    start = 0
    for count in patches_per_image:
        batched.append(pixel_values[start : start + count])
        start += count
    
    # [Batch=1, Seq, C, H, W]
    pixel_batch = torch.stack(batched, dim=0).to(device=device, dtype=dtype).contiguous()
    
    # 构建 Position IDs
    seq_lens = [int(t * h * w) for t, h, w in image_grid_thw]
    seq_len = seq_lens[0]
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).repeat(len(image_grid_thw), 1) # [Batch, Seq]

    # 构建 CU Seqlens
    cumsum = [0]
    for length in seq_lens:
        cumsum.append(cumsum[-1] + length)
    cu_seqlens = torch.tensor(cumsum, dtype=torch.int32, device=device)
    
    return pixel_batch, image_grid_thw, position_ids, cu_seqlens

# -----------------------------------------------------------------------------
# 4. 权重加载与转换
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
        # 1. 清洗 key 得到相对干净的名字
        clean_key = key
        for prefix in prefixes:
            while clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        
        # 2. 构建 Origin Key (需要以 vision_model. 开头，因为我们实例化的是 SiglipVisionModel)
        origin_key = clean_key
        if not origin_key.startswith("vision_model."):
            origin_key = "vision_model." + origin_key
        
        # 过滤掉非 vision 参数
        if "vision_model." in origin_key:
             origin_state[origin_key] = value

        # 3. 构建 Muse Key (扁平结构，去除 vision_model.)
        muse_key_base = origin_key.replace("vision_model.", "")
        
        if muse_key_base.startswith("head."):
            continue # Muse 没有 head

        # 映射规则
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
# 5. Config 转换 (Muse -> HF)
# -----------------------------------------------------------------------------

def convert_muse_config_to_hf(muse_cfg: KeyeVisionConfig) -> HFKeyeVisionConfig:
    # 提取 Muse Config 的所有字段
    cfg_dict = {}
    # Pydantic 用 .dict() 或 .model_dump()，普通类用 __dict__
    if hasattr(muse_cfg, "dict"):
        cfg_dict = muse_cfg.dict()
    elif hasattr(muse_cfg, "model_dump"):
        cfg_dict = muse_cfg.model_dump()
    else:
        # 普通类
        for k in dir(muse_cfg):
            if not k.startswith("_") and not callable(getattr(muse_cfg, k)):
                cfg_dict[k] = getattr(muse_cfg, k)
    
    return HFKeyeVisionConfig(**cfg_dict)

# -----------------------------------------------------------------------------
# 6. 主测试逻辑
# -----------------------------------------------------------------------------

def test_layer_by_layer():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # 1. Configs
    muse_config = KeyeVisionConfig()
    origin_config = convert_muse_config_to_hf(muse_config)
    
    # 2. Setup Device/Dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 为了 Debug 方便，如果不是必须，建议先用 float32，排查是否是精度问题
    # 但如果显存不够或想复现原环境，保持 bfloat16
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    logger.info(f"Running on {device} with {dtype}")

    with set_default_dtype(dtype):
        # 初始化模型
        origin_model = OriginKeyeVisionModel(origin_config)
        muse_model = MuseKeyeVisionModel(muse_config)
    
    origin_model.eval()
    muse_model.eval()

    # 3. 加载权重
    origin_state, muse_state = load_checkpoint_and_convert(CHECKPOINT_PATH)
    logger.info(f"Origin Params Count: {len(origin_state)}")
    logger.info(f"Muse Params Count  : {len(muse_state)}")
    
    def _to_dtype(d):
        return {k: v.to(dtype=dtype) for k,v in d.items()}

    # 加载 Origin
    keys_origin = origin_model.load_state_dict(_to_dtype(origin_state), strict=False)
    # Origin 可能会有一些 missing keys (如 head)，这是正常的，只要 vision_model 完整即可
    
    # 加载 Muse
    keys_muse = muse_model.load_state_dict(_to_dtype(muse_state), strict=False)
    if len(keys_muse.missing_keys) > 0:
        logger.warning(f"Muse Missing Keys: {keys_muse.missing_keys}")

    origin_model.to(device)
    muse_model.to(device)

    # 4. 注册 Hooks
    log_separator("Registering Hooks")
    activations = {"origin": {}, "muse": {}}

    def get_hook(model_type, layer_name):
        def hook(module, input, output):
            # 解包 tuple
            if isinstance(output, tuple): output = output[0]
            if isinstance(output, list): output = output[0] # Handle list output if any
            activations[model_type][layer_name] = output.detach()
        return hook

    # Hook Points:
    
    # [Point 1] Embeddings Output
    # Origin: model.vision_model.embeddings
    origin_model.vision_model.embeddings.register_forward_hook(get_hook("origin", "embeddings"))
    # Muse: model.embeddings
    muse_model.embeddings.register_forward_hook(get_hook("muse", "embeddings"))

    # [Point 2] Layer 0 (Transformer Block)
    origin_model.vision_model.encoder.layers[0].register_forward_hook(get_hook("origin", "layer_0"))
    muse_model.encoder.layers[0].register_forward_hook(get_hook("muse", "layer_0"))

    # [Point 3] Middle Layer
    mid_idx = muse_config.num_hidden_layers // 2
    origin_model.vision_model.encoder.layers[mid_idx].register_forward_hook(get_hook("origin", f"layer_{mid_idx}"))
    muse_model.encoder.layers[mid_idx].register_forward_hook(get_hook("muse", f"layer_{mid_idx}"))

    # [Point 4] Last Layer (Before Final LN)
    last_idx = muse_config.num_hidden_layers - 1
    origin_model.vision_model.encoder.layers[last_idx].register_forward_hook(get_hook("origin", "layer_last"))
    muse_model.encoder.layers[last_idx].register_forward_hook(get_hook("muse", "layer_last"))

    # 5. 准备输入
    processor = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    dummy_image = create_dummy_image(muse_config.image_size)
    
    pixel_values, image_grid_thw, position_ids, cu_seqlens = prepare_pixel_inputs(
        processor, dummy_image, device, dtype
    )
    
    logger.info(f"Input Pixel Shape: {pixel_values.shape}") # 应为 [1, Seq, C, H, W]

    # 6. 前向传播
    log_separator("Running Forward Pass")
    
    with torch.no_grad():
        # --- Origin Forward ---
        # Origin 代码期望 5D 输入，并处理 interpolation
        origin_out = origin_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True, 
            # Origin 代码里 window_size 默认为 -1，这里显式传一下以防万一
            window_size=-1 
        )
        
        # Origin 输出处理
        if hasattr(origin_out, "last_hidden_state"):
            origin_final = origin_out.last_hidden_state
        else:
            origin_final = origin_out
            
        # Origin 返回的是 list (Sample Pooling)，stack 起来
        if isinstance(origin_final, list): 
             origin_final = torch.stack(origin_final, dim=0)
        elif isinstance(origin_final, tuple):
             origin_final = origin_final[0]

        # --- Muse Forward ---
        # Muse 代码也做了 5D 检查
        muse_out = muse_model(
            pixel_values=pixel_values, 
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
            # Muse 里的 has_learnable_position_embedding 默认可能是 False，检查 Config
            has_learnable_position_embedding=getattr(muse_config, "has_learnable_position_embedding", True)
        )
        muse_final = muse_out["last_hidden_state"]

    # 7. 比较
    log_separator("Layer-wise Analysis")
    
    # 宽松度：bfloat16 精度下 1e-2 是正常的，如果很大则有问题
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-3

    # 1. Embeddings
    compare_tensors_verbose("1. Embeddings", 
                           activations["origin"]["embeddings"], 
                           activations["muse"]["embeddings"], atol=tol)

    # 2. Layer 0
    compare_tensors_verbose("2. Encoder Layer 0", 
                           activations["origin"]["layer_0"], 
                           activations["muse"]["layer_0"], atol=tol)

    # 3. Middle
    compare_tensors_verbose(f"3. Encoder Layer {mid_idx}", 
                           activations["origin"][f"layer_{mid_idx}"], 
                           activations["muse"][f"layer_{mid_idx}"], atol=tol)

    # 4. Last Layer
    compare_tensors_verbose("4. Encoder Last Layer", 
                           activations["origin"]["layer_last"], 
                           activations["muse"]["layer_last"], atol=tol)

    # 5. Final
    compare_tensors_verbose("5. Final Output (Post-LN)", origin_final, muse_final, atol=tol)

if __name__ == "__main__":
    test_layer_by_layer()