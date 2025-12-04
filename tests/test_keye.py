"""
Keye Vision 对比测试脚本
========================

参考 `tests/test_siglip2.py` 的结构，加载同一路径下的权重，
同时实例化：

1) `modeling_keye_origin.py` 中的原始 Keye Vision Transformer（视为 ground truth）
2) `modeling.py` / `image_processing_keye.py` / `model_config.py` 中实现的 Keye Vision Transformer

对两者进行权重转换、加载，并在同一张随机图像上比较前向输出。
"""

import logging
import os
import sys
import types
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

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
# 解决 modeling_keye_origin.py 对缺失 configuration_keye 的依赖
# -----------------------------------------------------------------------------

def _ensure_origin_config_module() -> None:
    """在运行时注入最精简的 configuration_keye 模块以便导入原始模型。"""
    module_name = "muse.muse.models.keye_vit.configuration_keye"
    if module_name in sys.modules:
        return

    config_module = types.ModuleType(module_name)

    class DummyKeyeConfig:
        """占位符，使得 origin 代码可以顺利 import。"""

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
# 辅助函数
# -----------------------------------------------------------------------------

def create_dummy_image(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)


def format_tensor_val(tensor: torch.Tensor, n: int = 5) -> str:
    vals = tensor.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join(f"{x:.6f}" for x in vals) + "]"


def log_separator(title: str) -> None:
    line = "=" * 80
    logger.info("\n%s", line)
    logger.info(" %s ", title.center(78))
    logger.info("%s", line)


def compare_tensors_verbose(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-5,
) -> None:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()

    if ref.shape != cand.shape:
        logger.error("❌ %s shape mismatch: %s vs %s", name, ref.shape, cand.shape)
        return

    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max Diff: {max_diff:.2e})"

    logger.info("%s", "-" * 80)
    logger.info("Tensor  : %s", name)
    logger.info("Status  : %s", status)
    logger.info(
        "Stats   : MaxDiff=%.3e | MeanDiff=%.3e | RefMean=%.4f | CandMean=%.4f",
        max_diff,
        mean_diff,
        ref.mean().item(),
        cand.mean().item(),
    )

    if max_diff >= atol:
        logger.info("Ref vals : %s", format_tensor_val(ref, 10))
        logger.info("Cand vals: %s", format_tensor_val(cand, 10))


def _flatten_grid_entry(entry) -> List[Tuple[int, int, int]]:
    if isinstance(entry, tuple) and len(entry) == 3:
        return [tuple(int(x) for x in entry)]

    flattened: List[Tuple[int, int, int]] = []
    if isinstance(entry, (list, tuple)):
        for item in entry:
            flattened.extend(_flatten_grid_entry(item))
        return flattened

    raise ValueError(f"Unsupported grid format: {entry}")


def build_position_ids(image_grid_thw: List[Tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    seq_lens = [int(t * h * w) for t, h, w in image_grid_thw]
    if len(set(seq_lens)) != 1:
        raise ValueError("当前测试假设 batch 内每个样本的 patch 数量一致。")
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
    pixel_values: torch.Tensor = processed["pixel_values"]  # [total_patches, C, patch, patch]
    grid_info = processed["image_grid_thw"]
    if isinstance(grid_info, torch.Tensor):
        grid_info = grid_info.cpu().tolist()
    elif isinstance(grid_info, np.ndarray):
        grid_info = grid_info.tolist()

    image_grid_thw = [tuple(int(v) for v in grid) for grid in grid_info]
    patches_per_image = [int(np.prod(grid)) for grid in image_grid_thw]
    total_patches = sum(patches_per_image)
    if total_patches != pixel_values.shape[0]:
        raise ValueError(
            f"Patch 数量不匹配: expected {total_patches}, got {pixel_values.shape[0]}"
        )

    batched = []
    start = 0
    for count in patches_per_image:
        batched.append(pixel_values[start : start + count])
        start += count

    pixel_batch = torch.stack(batched, dim=0)  # [B, Seq, C, patch, patch]
    pixel_batch = pixel_batch.to(device=device, dtype=dtype).contiguous()

    position_ids = build_position_ids(image_grid_thw, device)
    cu_seqlens = build_cu_seqlens(image_grid_thw, device)
    return pixel_batch, image_grid_thw, position_ids, cu_seqlens


def load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    logger.info("Loading checkpoint from: %s", path)
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("module", "state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                logger.info("Unpacking nested '%s' key", key)
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint 格式不正确，未找到 state_dict。")
    return raw


def extract_vision_state_dict(state_dict: Dict[str, torch.Tensor], keep_head: bool) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "model.",
        "state_dict.",
        "vision_tower.",
        "vision_backbone.",
        "siglip.",
    )
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            while new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]

        if "vision_model." in new_key:
            new_key = new_key.split("vision_model.", 1)[1]
        elif new_key.startswith("vision_model."):
            new_key = new_key[len("vision_model.") :]

        if not new_key.startswith(("embeddings.", "encoder.", "post_layernorm.", "ln_post.", "head.")):
            continue
        if not keep_head and new_key.startswith("head."):
            continue
        filtered[new_key] = value
    return filtered


def convert_to_muse_keys(origin_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}
    for key, value in origin_state_dict.items():
        if key.startswith("head."):
            # Muse 模型不包含 head
            continue
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
        converted[new_key] = value
    return converted


def check_loaded_weights(model: torch.nn.Module, reference_state: Dict[str, torch.Tensor], name: str) -> None:
    issues = 0
    model_state = model.state_dict()
    for key, ref_tensor in reference_state.items():
        if key not in model_state:
            continue
        diff = (model_state[key].detach().cpu() - ref_tensor.detach().cpu()).abs().max().item()
        if diff >= 1e-5:
            issues += 1
            logger.warning("⚠️ %s weight mismatch on %s (max diff %.3e)", name, key, diff)
    if issues == 0:
        logger.info("✅ %s 权重与参考张量完全一致。", name)
    else:
        logger.error("❌ %s 有 %d 个参数存在差异。", name, issues)


# -----------------------------------------------------------------------------
# 主测试逻辑
# -----------------------------------------------------------------------------

def test_keye_logits_align_with_origin_checkpoint():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint 不存在: {CHECKPOINT_PATH}")

    log_separator("Config Summary")
    config = KeyeVisionConfig()
    config_fields = ["image_size", "patch_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]
    for field in config_fields:
        logger.info("%-30s : %s", field, getattr(config, field))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if torch.cuda.is_available() else False
    run_dtype = torch.bfloat16 if bf16_supported else torch.float32
    model_dtype = torch.bfloat16 if run_dtype == torch.bfloat16 else torch.float32

    with set_default_dtype(model_dtype):
        muse_model = MuseKeyeVisionModel(config)
        origin_model = OriginKeyeVisionModel(config)

    muse_model.eval()
    origin_model.eval()

    raw_state = load_checkpoint(CHECKPOINT_PATH)
    origin_state = extract_vision_state_dict(raw_state, keep_head=True)
    muse_ready_state = convert_to_muse_keys(origin_state)

    logger.info("原始 Keye 参数数量: %d", len(origin_state))
    logger.info("Muse Keye 参数数量 : %d", len(muse_ready_state))

    def _to_dtype(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ready = {}
        for key, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                ready[key] = tensor.to(dtype=model_dtype)
            else:
                ready[key] = tensor
        return ready

    origin_loaded = origin_model.load_state_dict(_to_dtype(origin_state), strict=False)
    logger.info("Origin missing keys   : %s", origin_loaded.missing_keys)
    logger.info("Origin unexpected keys: %s", origin_loaded.unexpected_keys)

    muse_loaded = muse_model.load_state_dict(_to_dtype(muse_ready_state), strict=False)
    logger.info("Muse missing keys     : %s", muse_loaded.missing_keys)
    logger.info("Muse unexpected keys  : %s", muse_loaded.unexpected_keys)

    check_loaded_weights(origin_model, _to_dtype(origin_state), "Origin")
    check_loaded_weights(muse_model, _to_dtype(muse_ready_state), "Muse")

    muse_model = muse_model.to(device=device, dtype=run_dtype)
    origin_model = origin_model.to(device=device, dtype=run_dtype)

    processor = KeyeVisionImageProcessor(patch_size=config.patch_size)
    dummy_image = create_dummy_image(config.image_size)
    pixel_values, image_grid_thw, position_ids, cu_seqlens = prepare_pixel_inputs(
        processor, dummy_image, device, run_dtype
    )

    log_separator("Forward Pass Comparison")
    with torch.no_grad():
        origin_outputs = origin_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
        )
        muse_outputs = muse_model(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
        )

    origin_hidden = origin_outputs.last_hidden_state
    if isinstance(origin_hidden, list):
        origin_hidden = torch.stack(origin_hidden, dim=0)
    muse_hidden = muse_outputs["last_hidden_state"]

    compare_tensors_verbose(
        "Last Hidden State",
        origin_hidden,
        muse_hidden,
        atol=1e-2 if run_dtype == torch.bfloat16 else 1e-4,
    )

    batch_idx = 0
    seq_len = origin_hidden.shape[1]
    positions = [
        (0, "First Token"),
        (min(10, seq_len - 1), "Token 10"),
        (seq_len // 2, "Middle Token"),
        (seq_len - 1, "Last Token"),
    ]
    logger.info("\n详细 Token 采样（前 5 维特征）")
    for pos, label in positions:
        if pos < 0 or pos >= seq_len:
            continue
        ref_vals = origin_hidden[batch_idx, pos, :5].float().cpu().numpy()
        muse_vals = muse_hidden[batch_idx, pos, :5].float().cpu().numpy()
        diff_val = np.max(np.abs(ref_vals - muse_vals))
        logger.info(
            "%-15s | Ref: [%s] | Muse: [%s] | MaxDiff %.2e",
            label,
            ", ".join(f"{x:.4f}" for x in ref_vals),
            ", ".join(f"{x:.4f}" for x in muse_vals),
            diff_val,
        )

    final_diff = (origin_hidden - muse_hidden).abs().max().item()
    threshold = 1e-2 if run_dtype == torch.bfloat16 else 1e-4
    log_separator("FINAL RESULT")
    if final_diff < threshold:
        logger.info("SUCCESS: Max diff %.3e < threshold %.3e", final_diff, threshold)
    else:
        logger.error("FAILURE: Max diff %.3e >= threshold %.3e", final_diff, threshold)


if __name__ == "__main__":
    test_keye_logits_align_with_origin_checkpoint()