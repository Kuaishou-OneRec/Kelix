"""
等价性冒烟测试：加载同一权重到原版 (`modeling_keye_origin.py`) 与 Muse
(`modeling.py`) 的 KeyeForConditionalGeneration，构造最小视觉+文本输入并断言
logits 全零差异。

用法：
  KEYE_VL_CHECKPOINT=/path/to/ckpt.pt pytest -q tests/test_keye_vl_zero_diff.py

可选：
  设置 KEYE_SLOWFAST_MODEL_DIR 使用 SlowFastVisionPadder（下面类，基于
  AutoProcessor 的占位构造）生成更接近 SlowFast 输入风格的样例；未设置则走最简
  单 patch 随机输入。

说明：
- 权重应为常规 state_dict（支持含 module/state_dict 包裹）。
- 默认严格对齐 logits（atol/rtol=0），如需放宽可自行调整断言。
"""

import os
import glob
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import torch
import pytest
from transformers import AutoProcessor

from muse.models.keye_tokenizer_video import modeling as muse_mod
from muse.models.keye_tokenizer_video import modeling_keye_origin as origin_mod
from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig


DEFAULT_CKPT = (
    "/mmu_mllm_hdd_2/maosiyang/output/Keye/vq_end2end_video/discrete/"
    "run_exp0.0.1_stage1_baseline/step16000/global_step16000/converted"
)
SLOWFAST_MODEL_DIR_ENV = "KEYE_SLOWFAST_MODEL_DIR"


def _load_safetensors_file(file_path: str, dtype: torch.dtype) -> dict:
    from safetensors.torch import safe_open

    tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).to(dtype)
    return tensors


def _load_checkpoint(path: str, dtype: torch.dtype) -> dict:
    p = Path(path)

    # If directory, load all model-*.safetensors shards
    if p.is_dir():
        shard_files = sorted(glob.glob(str(p / "model-*.safetensors")))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors shards found in {path}")
        state = {}
        for shard in shard_files:
            state.update(_load_safetensors_file(shard, dtype))
        return state

    # If single safetensors file
    if p.suffix == ".safetensors":
        return _load_safetensors_file(str(p), dtype)

    # Fallback: torch load
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "module" in ckpt:
        ckpt = ckpt["module"]
    return {k: v.to(dtype) for k, v in ckpt.items()}


def _maybe_convert_for_muse(model: torch.nn.Module, state_dict: dict, tie_word_embeddings: bool = True) -> dict:
    """
    如果模型实现了 convert_hf_state_dict（常见于 vision 子模块），则先做转换；
    否则直接返回原字典。
    """
    # Check if convert_hf_state_dict exists on the class or instance
    model_cls = model.__class__
    convert_fn = getattr(model_cls, "convert_hf_state_dict", None)
    if convert_fn is None:
        convert_fn = getattr(model, "convert_hf_state_dict", None)
    
    if callable(convert_fn):
        try:
            return convert_fn(state_dict, tie_word_embeddings=tie_word_embeddings)
        except Exception as e:
            print(f"Warning: convert_hf_state_dict failed with {e}, falling back to original state_dict")
            # 转换失败则回退原权重，再由 strict=False 兜底
            return state_dict
    return state_dict


class SlowFastVisionPadder:
    """
    极简版 SlowFast padding 构造器，按照用户提供的片段实现，只保留需要的字段。
    - 仅生成 image_pad（不使用 fast_video 以避免额外依赖）。
    - position_ids 交由模型内部生成，确保两端一致。
    """

    def __init__(self, model_dir: str):
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = processor
        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        assert (
            self.merge_size == 2
        ), f"SlowFastVisionPadder only supports merge_size==2, got {self.merge_size}"

        self.image_pad = processor.tokenizer.encode("<|image_pad|>")[0]
        self.video_pad = processor.tokenizer.encode("<|video_pad|>")[0]
        fast_video_pad = processor.tokenizer.encode("<|fast_video_pad|>")
        assert len(fast_video_pad) == 1, f"Decode fast_video_pad failed: {fast_video_pad}"
        self.fast_video_pad = fast_video_pad[0]
        self.vision_start = processor.tokenizer.encode("<|vision_start|>")[0]
        self.vision_end = processor.tokenizer.encode("<|vision_end|>")[0]
        self.frame = processor.tokenizer.encode("<|frame|>")[0]

    def gen_img_pad(self, n_merged_slow_tokens: int = 1) -> Dict[str, Any]:
        input_ids = [self.vision_start] + [self.image_pad] * n_merged_slow_tokens + [self.vision_end]
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64),
            "attention_mask": torch.tensor([[1] * (n_merged_slow_tokens + 2)], dtype=torch.int64),
            # merge_size=2 -> 每个 merged token 对应 2x2 patch = 4 patch tokens
            "pixel_values": torch.rand(
                n_merged_slow_tokens * 4, 3, self.patch_size, self.patch_size
            ).float(),
            "image_grid_thw": torch.tensor([[1, 2, n_merged_slow_tokens * 2]], dtype=torch.int64),
            "loss_mask": torch.zeros(len(input_ids), dtype=torch.int64),
        }
        # 让模型内部生成 position_ids，保持两端一致
        return inputs


def _build_inputs(
    image_token_id: int, device: torch.device, dtype: torch.dtype, slowfast_dir: Union[str, None]
) -> Tuple[dict, torch.Tensor]:
    """
    优先使用 SlowFastVisionPadder 生成占位输入；否则退回最小随机样例。
    Both models expect image_grid_thw as tensor [num_images, 3].
    """
    if slowfast_dir:
        padder = SlowFastVisionPadder(slowfast_dir)
        img_pad = padder.gen_img_pad(n_merged_slow_tokens=1)
        # image_grid_thw as tensor [num_images, 3]
        grid_thw = img_pad["image_grid_thw"].to(device)  # [1, 3]
        inputs = {
            "input_ids": img_pad["input_ids"].to(device),
            "attention_mask": img_pad["attention_mask"].to(device),
            "pixel_values": img_pad["pixel_values"].to(device, dtype),  # [num_patches, 3, H, W]
            "image_grid_thw": grid_thw,
            # 让模型内部根据 image_token_id 生成 mask
        }
        vision_token_mask = None
        return inputs, vision_token_mask

    # fallback: One 14x14 patch -> one vision token
    # pixel_values: [num_patches, C, H, W] where num_patches = t*h*w = 1*1*1 = 1
    pixel_values = torch.randn(1, 3, 14, 14, device=device, dtype=dtype)
    # image_grid_thw: [num_images, 3] where num_images = 1
    image_grid_thw = torch.tensor([[1, 1, 1]], device=device, dtype=torch.long)
    input_ids = torch.tensor([[image_token_id, 1, 2, 3]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    vision_token_mask = input_ids == image_token_id
    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "vision_token_mask": vision_token_mask,
        },
        vision_token_mask,
    )


@pytest.mark.parametrize("dtype", [torch.float16])
def test_keye_vl_zero_diff(dtype):
    ckpt_path = os.environ.get("KEYE_VL_CHECKPOINT", DEFAULT_CKPT)
    assert Path(ckpt_path).exists(), f"Checkpoint not found: {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 优先环境变量，否则默认用同一 converted 目录（其中包含处理器/processing）
    slowfast_dir = os.environ.get(SLOWFAST_MODEL_DIR_ENV, DEFAULT_CKPT)

    # ------------------ load config.json ------------------
    def _load_config_json(ckpt_path: str) -> Dict[str, Any]:
        p = Path(ckpt_path)
        cfg_path = p / "config.json" if p.is_dir() else p.with_name("config.json")
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found near {ckpt_path}")
        with open(cfg_path, "r") as f:
            return json.load(f)

    raw_cfg = _load_config_json(ckpt_path)

    # LLM (Qwen3) config mapping — use raw fields directly to match checkpoint shapes
    qwen_cfg = Qwen3Config(
        model_class="Qwen3Model",
        vocab_size=raw_cfg["vocab_size"],
        embed_dim=raw_cfg["hidden_size"],            # 1024
        num_layers=raw_cfg["num_hidden_layers"],     # 28
        num_heads=raw_cfg["num_attention_heads"],    # 16 -> q_proj out = 16*128=2048
        num_kv_heads=raw_cfg["num_key_value_heads"], # 8
        head_dim=raw_cfg["head_dim"],                # 128
        attn_dropout=raw_cfg.get("attention_dropout", 0.0),
        attention_function="flash_attention_2",
        q_proj_bias=raw_cfg.get("attention_bias", False),
        k_proj_bias=raw_cfg.get("attention_bias", False),
        v_proj_bias=raw_cfg.get("attention_bias", False),
        intermediate_dim=raw_cfg["intermediate_size"],  # 3072
        max_seq_len=raw_cfg["max_position_embeddings"], # 40960
        rope_base=float(raw_cfg.get("rope_theta", 1_000_000)),
        norm_eps=raw_cfg.get("rms_norm_eps", 1e-6),
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=raw_cfg.get("tie_word_embeddings", True),
    )

    # Vision config: use inner vision_config["vision_config"]
    outer_vcfg = raw_cfg["vision_config"]
    inner_vcfg = outer_vcfg["vision_config"]
    vision_cfg = KeyeVisionConfig(
        hidden_size=inner_vcfg["hidden_size"],               # 1152
        intermediate_size=inner_vcfg["intermediate_size"],   # 4304
        num_hidden_layers=inner_vcfg["num_hidden_layers"],   # 27
        num_attention_heads=inner_vcfg["num_attention_heads"], # 16
        num_channels=inner_vcfg["num_channels"],             # 3
        image_size=inner_vcfg["image_size"],                 # 384
        patch_size=inner_vcfg["patch_size"],                 # 14
        layer_norm_eps=inner_vcfg.get("layer_norm_eps", 1e-6),
        attention_dropout=inner_vcfg.get("attention_dropout", 0.0),
        rope_theta=inner_vcfg.get("rope_theta", 10000),
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_function=inner_vcfg.get("_attn_implementation", "flash_attention_2"),
        use_qk_norm=inner_vcfg.get("use_qk_norm", False),
    )

    # Tokenizer config from outer vision_config (tokenizer block)
    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=outer_vcfg.get("llm_hidden_size", 4096),  # align to ckpt: 4096
        embedding_dim=outer_vcfg.get("embedding_dim", 128),
        init_embedding_dim=outer_vcfg.get("init_embedding_dim", 4096),
        codebook_size=outer_vcfg.get("codebook_size", 65536),
        n_q_tokens=outer_vcfg.get("n_q_tokens", 8),
        split_voc=outer_vcfg.get("split_voc", 1),
        add_voc_reducer=outer_vcfg.get("add_voc_reducer", False),
        split_dim=outer_vcfg.get("split_dim", False),
        vq_sampling_mode=outer_vcfg.get("vq_sampling_mode", "argmin"),
    )

    # 当使用 SlowFast padder 时，将 image_token_id 与 padder 的 image_pad 对齐
    padder_token_id = None
    if slowfast_dir:
        padder_token_id = SlowFastVisionPadder(slowfast_dir).image_pad

    image_token_id = padder_token_id if padder_token_id is not None else raw_cfg.get("image_token_id", 151655)

    # Build origin config from raw config (HuggingFace style)
    # KeyeConfig takes vision_config as dict or KeyeImageTokenizerConfig
    origin_cfg = origin_mod.KeyeConfig(
        vocab_size=raw_cfg["vocab_size"],
        hidden_size=raw_cfg["hidden_size"],
        intermediate_size=raw_cfg["intermediate_size"],
        num_hidden_layers=raw_cfg["num_hidden_layers"],
        num_attention_heads=raw_cfg["num_attention_heads"],
        num_key_value_heads=raw_cfg["num_key_value_heads"],
        max_position_embeddings=raw_cfg["max_position_embeddings"],
        rms_norm_eps=raw_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=raw_cfg.get("rope_theta", 1_000_000),
        attention_dropout=raw_cfg.get("attention_dropout", 0.0),
        attention_bias=raw_cfg.get("attention_bias", False),
        tie_word_embeddings=raw_cfg.get("tie_word_embeddings", True),
        vision_config=outer_vcfg,  # Pass the dict, KeyeConfig will convert it
        image_token_id=image_token_id,
        video_token_id=raw_cfg.get("video_token_id", 151656),
        vision_start_token_id=raw_cfg.get("vision_start_token_id", 151652),
        fast_video_token_id=raw_cfg.get("fast_video_token_id", 151678),
    )
    
    muse_model = muse_mod.KeyeForConditionalGeneration(
        qwen_config=qwen_cfg,
        vision_config=vision_cfg,
        tokenizer_config=tokenizer_cfg,
        image_token_id=image_token_id,
    ).to(device, dtype)
    origin_model = origin_mod.KeyeForConditionalGeneration(origin_cfg).to(device, dtype)

    state_dict = _load_checkpoint(ckpt_path, dtype)
    tie_word_embeddings = qwen_cfg.tie_word_embeddings
    muse_state = _maybe_convert_for_muse(muse_model, state_dict, tie_word_embeddings=tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)
    origin_model.load_state_dict(state_dict, strict=False)

    inputs, vision_mask = _build_inputs(image_token_id, device, dtype, slowfast_dir)

    muse_model.eval()
    origin_model.eval()
    with torch.no_grad():
        origin_out = origin_model(**inputs)
        muse_out = muse_model(**inputs)

    # Align logits tensors
    origin_logits = origin_out.logits
    muse_logits = muse_out["logits"] if isinstance(muse_out, dict) else muse_out.logits

    torch.testing.assert_close(
        origin_logits, muse_logits, atol=0.0, rtol=0.0, msg="Origin vs Muse logits differ"
    )


if __name__ == "__main__":
    test_keye_vl_zero_diff(torch.float32)

