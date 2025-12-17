"""
Keye-VL Pipeline: Muse vs Origin Model Comparison
==================================================
Input: 100x100 Generated Circle Image (No Text Prompt)
Output: Compare Muse and Origin Model Logits
        And save them to /llm_reco/maosiyang/
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'  # Disable sequence parallel for Origin model

import sys
import logging
import glob
import json
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image, ImageDraw

# === 导入 Muse 模型 ===
from muse.models.keye_tokenizer_end2end_image import modeling as muse_mod
from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig
from muse.training.common import set_default_dtype

# === 导入 Origin 模型 ===
from tests.models.tokenizer_end2end_mt_1drope_v2.configuration_keye import KeyeConfig
from tests.models.tokenizer_end2end_mt_1drope_v2.modeling_keye import KeyeForConditionalGeneration

# === 导入 Processor 相关 ===
from transformers import AutoTokenizer, AutoProcessor

try:
    from tests.models.keye_vl_tokenizer_image.processing_keye import KeyeProcessor
    from tests.models.tokenizer_end2end_mt_1drope_v2.keye_vl_utils import process_vision_info
except ImportError:
    sys.path.append(os.getcwd())
    from tests.models.keye_vl_tokenizer_image.processing_keye import KeyeProcessor
    from tests.models.tokenizer_end2end_mt_1drope_v2.keye_vl_utils import process_vision_info

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_CKPT = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/"

# =========================================================================
# Helper Functions (严格对齐 Origin)
# =========================================================================

def generate_circle_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    与 Origin 代码完全一致的生成函数
    """
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color,
                 outline=outline_color,
                 width=outline_width)
    return image

def _load_config_json(ckpt_path: str) -> Dict[str, Any]:
    p = Path(ckpt_path)
    base_dir = p if p.is_dir() else p.parent
    cfg_path = base_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {base_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)

def _load_checkpoint_robust(path_str: str, device="cpu") -> Dict[str, torch.Tensor]:
    path = Path(path_str)
    if path.is_file():
        state_dict = torch.load(path, map_location=device)
        return state_dict.get("module", state_dict)

    state_dict = {}
    import tqdm
    from safetensors.torch import load_file

    # 优先处理 safetensors 文件
    print(f"Loading weights from {path_str}...")
    for f in tqdm.tqdm(os.listdir(path_str)):
        if f.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(path_str, f)))

    # 如果没有 safetensors 文件，回退到 bin 文件
    if not state_dict:
        bin_files = sorted(glob.glob(str(path / "*.bin")))
        if bin_files:
            for f in bin_files:
                if any(x in f for x in ["training_args", "optimizer", "scheduler"]): continue
                part = torch.load(f, map_location=device)
                if "module" in part: part = part["module"]
                state_dict.update(part)

    return state_dict

# =========================================================================
# Input Preparation (Aligned with Origin process_message)
# =========================================================================

def prepare_inputs_common(ckpt_path: str, device: str, dtype: torch.dtype):
    """
    准备通用输入（供两个模型共享）
    严格对齐新 HF 代码的 process_message 函数
    """
    logger.info("⚙️ Loading Tokenizer & ImageProcessor...")
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)

    # [Align] 使用 100x100
    logger.info("🎨 Generating Circle Image (100x100)...")
    image = generate_circle_image(size=(100, 100)) 
    
    # [Align] 只有图片，没有文本
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
            ],
        }
    ]

    logger.info("📝 Applying Chat Template...")
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    logger.info(f"   -> Text Prompt: {repr(text)}")
    
    # [关键对齐] 使用 process_vision_info 提取图片，与新 HF 代码一致
    image_inputs, video_inputs = process_vision_info(messages)
    logger.info(f"   -> process_vision_info: images={len(image_inputs) if image_inputs else 0}, videos={len(video_inputs) if video_inputs else 0}")

    logger.info("🔄 Running Processor...")
    # [Align] 参数与 Origin 完全一致：padding=False, truncation=False
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False, 
        truncation=False,
        return_tensors="pt",
    ).to(device)

    return inputs, processor, messages

# def prepare_inputs_for_muse(inputs, device: str, dtype: torch.dtype):
#     """
#     为 Muse 模型准备输入
#     """
#     model_inputs = {
#         "input_ids": inputs["input_ids"].to(device),
#         "attention_mask": inputs["attention_mask"].to(device),
#         "pixel_values": inputs["pixel_values"].to(device, dtype=dtype),
#         "image_grid_thw": inputs["image_grid_thw"].to(device)
#     }
    
#     # [Muse Specific] Muse Model 期望 pixel_values 是 [N, C, H, W]
#     if model_inputs["pixel_values"].dim() == 5 and model_inputs["pixel_values"].shape[0] == 1:
#         model_inputs["pixel_values"] = model_inputs["pixel_values"].squeeze(0)

#     logger.info(f"   [Muse] Input IDs Shape: {model_inputs['input_ids'].shape}")
#     logger.info(f"   [Muse] Pixel Values Shape: {model_inputs['pixel_values'].shape}")
#     logger.info(f"   [Muse] Image Grid: {model_inputs['image_grid_thw'].tolist()}")
    
#     return model_inputs

# def prepare_inputs_for_origin(inputs, device):
#     """
#     为 Origin 模型准备输入
#     [关键对齐] 与新 HF 代码完全一致：直接 .to(device)，不转换 dtype
#     """
#     # 直接把整个 inputs 移到 device 上，保持原始 dtype（与新 HF 代码一致）
#     model_inputs = inputs.to(device)

#     logger.info(f"   [Origin] Input IDs Shape: {model_inputs['input_ids'].shape}")
#     logger.info(f"   [Origin] Pixel Values Shape: {model_inputs['pixel_values'].shape}")
#     logger.info(f"   [Origin] Pixel Values Dtype: {model_inputs['pixel_values'].dtype}")
#     logger.info(f"   [Origin] Image Grid: {model_inputs['image_grid_thw'].tolist()}")
    
#     return model_inputs

# =========================================================================
# Model Loading Functions
# =========================================================================

def load_muse_model(ckpt_path: str, raw_cfg: Dict, device: str, dtype: torch.dtype):
    """
    加载 Muse 模型
    """
    logger.info("\n" + "="*60)
    logger.info("🚀 Loading Muse Model...")
    logger.info("="*60)
    
    rope_scaling = raw_cfg.get("rope_scaling")
    mrope_section = rope_scaling.get("mrope_section") if rope_scaling else None
    
    qwen_cfg = Qwen3Config(
        model_class="Qwen3Model",
        vocab_size=raw_cfg["vocab_size"],
        embed_dim=raw_cfg["hidden_size"],
        num_layers=raw_cfg["num_hidden_layers"],
        num_heads=raw_cfg["num_attention_heads"],
        num_kv_heads=raw_cfg["num_key_value_heads"],
        head_dim=raw_cfg["head_dim"],
        intermediate_dim=raw_cfg["intermediate_size"],
        max_seq_len=raw_cfg["max_position_embeddings"],
        hidden_act=raw_cfg.get("hidden_act", "silu"),
        attention_bias=raw_cfg.get("attention_bias", False),
        rope_base=float(raw_cfg.get("rope_theta", 1_000_000)),
        rope_theta=float(raw_cfg.get("rope_theta", 1_000_000)),
        rope_scaling=rope_scaling,
        attention_function=raw_cfg.get("_attn_implementation", "flash_attention_2"),
        use_sliding_window=raw_cfg.get("use_sliding_window", False),
        sliding_window=raw_cfg.get("sliding_window"),
        norm_eps=raw_cfg.get("norm_eps", 1e-6),
        rms_norm_eps=raw_cfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=raw_cfg.get("tie_word_embeddings", True),
        use_multimodal_rope=True,
        mrope_section=mrope_section,
    )
    
    outer_vcfg = raw_cfg["vision_config"]
    inner_vcfg = outer_vcfg["vision_config"]
    
    vision_cfg = KeyeVisionConfig(
        hidden_size=inner_vcfg["hidden_size"],
        num_hidden_layers=inner_vcfg["num_hidden_layers"],
        num_attention_heads=inner_vcfg["num_attention_heads"],
        image_size=inner_vcfg["image_size"],
        patch_size=inner_vcfg["patch_size"],
        intermediate_size=inner_vcfg["intermediate_size"],
        hidden_act=inner_vcfg.get("hidden_act", "gelu_pytorch_tanh"),
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_dropout=inner_vcfg.get("attention_dropout", 0.0),
        rope_theta=inner_vcfg.get("rope_theta", 10000.0),
        use_qk_norm=inner_vcfg.get("use_qk_norm", False),
        qk_norm_eps=inner_vcfg.get("qk_norm_eps", 1e-6),
        attention_function=raw_cfg.get("_attn_implementation", "flash_attention_2"),
    )
    vision_cfg._attn_implementation = "flash_attention_2"

    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=outer_vcfg.get("llm_hidden_size", 4096),
        embedding_dim=outer_vcfg.get("embedding_dim", 128),
        init_embedding_dim=outer_vcfg.get("init_embedding_dim", 4096),
        codebook_size=outer_vcfg.get("codebook_size", 65536),
        n_q_tokens=outer_vcfg.get("n_q_tokens", 8),
        split_voc=outer_vcfg.get("split_voc", 1),
        add_voc_reducer=outer_vcfg.get("add_voc_reducer", False),
        split_dim=outer_vcfg.get("split_dim", False),
        vq_sampling_mode="argmin",
    )

    with set_default_dtype(dtype):
        muse_model = muse_mod.KeyeTokenizerEnd2EndImage(
            qwen_config=qwen_cfg,
            vision_config=vision_cfg,
            tokenizer_config=tokenizer_cfg,
            image_token_id=raw_cfg.get("image_token_id", 151655),
            pool="sum"
        ).to(device)

    logger.info("📥 Loading Muse Weights...")
    state_dict = _load_checkpoint_robust(ckpt_path, device="cpu")
    muse_state = muse_model.convert_hf_state_dict(state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)
    muse_model.to(device, dtype)
    muse_model.eval()
    
    logger.info("✅ Muse Model Loaded.")
    return muse_model

def load_origin_model(ckpt_path: str, device: str, dtype: torch.dtype):
    """
    加载 Origin 模型 (KeyeForConditionalGeneration)
    """
    logger.info("\n" + "="*60)
    logger.info("🚀 Loading Origin Model (KeyeForConditionalGeneration)...")
    logger.info("="*60)
    
    # 先用 from_pretrained 获取 config
    origin_model = KeyeForConditionalGeneration.from_pretrained(
        ckpt_path, 
        _attn_implementation="flash_attention_2", 
        torch_dtype=dtype, 
        low_cpu_mem_usage=True
    )
    
    # 重新初始化模型（与 test_run.py 保持一致）
    origin_model = KeyeForConditionalGeneration(origin_model.config)
    origin_model._attn_implementation = "flash_attention_2"
    origin_model = origin_model.to(device).to(dtype)
    
    # 加载权重
    logger.info("📥 Loading Origin Weights...")
    state_dict = _load_checkpoint_robust(ckpt_path, device="cpu")
    origin_model.load_state_dict(state_dict, strict=True)
    origin_model.eval()
    
    logger.info("✅ Origin Model Loaded.")
    return origin_model

# =========================================================================
# Comparison Functions
# =========================================================================

def compare_logits(muse_logits: torch.Tensor, origin_logits: torch.Tensor):
    """
    对比两个模型的 logits
    """
    logger.info("\n" + "="*60)
    logger.info("📊 Comparing Logits...")
    logger.info("="*60)
    
    logger.info(f"   Muse Logits Shape:   {muse_logits.shape}")
    logger.info(f"   Origin Logits Shape: {origin_logits.shape}")
    
    # 确保形状一致
    if muse_logits.shape != origin_logits.shape:
        logger.warning(f"⚠️ Shape mismatch! Cannot compare directly.")
        return
    
    # 转换为 float32 进行比较
    muse_f32 = muse_logits.float()
    origin_f32 = origin_logits.float()
    
    # 计算差异
    diff = muse_f32 - origin_f32
    abs_diff = diff.abs()
    
    # 统计信息
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # 相对误差 (避免除零)
    eps = 1e-8
    rel_diff = abs_diff / (origin_f32.abs() + eps)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Cosine Similarity (展平后计算)
    muse_flat = muse_f32.view(-1)
    origin_flat = origin_f32.view(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        muse_flat.unsqueeze(0), origin_flat.unsqueeze(0)
    ).item()
    
    logger.info(f"\n📈 Comparison Results:")
    logger.info(f"   Max Absolute Diff:  {max_abs_diff:.6e}")
    logger.info(f"   Mean Absolute Diff: {mean_abs_diff:.6e}")
    logger.info(f"   Max Relative Diff:  {max_rel_diff:.6e}")
    logger.info(f"   Mean Relative Diff: {mean_rel_diff:.6e}")
    logger.info(f"   Cosine Similarity:  {cos_sim:.8f}")
    
    # 判断是否对齐
    if cos_sim > 0.99 and max_abs_diff < 1e-2:
        logger.info("\n✅ Models are well aligned! (cosine > 0.99, max_diff < 1e-2)")
    elif cos_sim > 0.95:
        logger.info("\n⚠️ Models are roughly aligned. (cosine > 0.95)")
    else:
        logger.info("\n❌ Models have significant differences. (cosine <= 0.95)")
    
    # 打印第一个 token 的 logits 对比
    logger.info(f"\n🔍 First Token Logits (top 10 dims):")
    logger.info(f"   Muse:   {muse_logits[0, 0, :10].float().cpu().numpy()}")
    logger.info(f"   Origin: {origin_logits[0, 0, :10].float().cpu().numpy()}")
    
    # 打印最后一个 token 的 logits 对比
    logger.info(f"\n🔍 Last Token Logits (top 10 dims):")
    logger.info(f"   Muse:   {muse_logits[0, -1, :10].float().cpu().numpy()}")
    logger.info(f"   Origin: {origin_logits[0, -1, :10].float().cpu().numpy()}")
    
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "cosine_similarity": cos_sim,
    }

# =========================================================================
# Main Execution
# =========================================================================

def run_comparison():
    ckpt_path = DEFAULT_CKPT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 
    
    logger.info(f"🔧 Device: {device}, Dtype: {dtype}")
    logger.info(f"📂 Checkpoint: {ckpt_path}")
    
    # 加载配置
    raw_cfg = _load_config_json(ckpt_path)
    
    # 准备通用输入
    inputs, processor, messages = prepare_inputs_common(ckpt_path, device, dtype)
    
    # ========== Muse Model ==========
    muse_model = load_muse_model(ckpt_path, raw_cfg, device, dtype)
    muse_inputs = inputs
    
    logger.info("\n🔥 Running Muse Forward...")
    with torch.no_grad():
        muse_outputs = muse_model(**muse_inputs)
    
    if isinstance(muse_outputs, dict):
        muse_logits = muse_outputs.get("logits")
    elif hasattr(muse_outputs, "logits"):
        muse_logits = muse_outputs.logits
    else:
        muse_logits = muse_outputs[0]
    
    logger.info(f"   Muse Logits Shape: {muse_logits.shape}")
    
    # ========== Origin Model ==========
    origin_model = load_origin_model(ckpt_path, device, dtype)
    # [关键对齐] 与新 HF 代码一致，直接 .to(device)
    # origin_inputs = prepare_inputs_for_origin(inputs, device)
    origin_inputs = inputs
    
    logger.info("\n🔥 Running Origin Forward...")
    with torch.no_grad():
        origin_outputs = origin_model(**origin_inputs)
    
    if isinstance(origin_outputs, dict):
        origin_logits = origin_outputs.get("logits")
    elif hasattr(origin_outputs, "logits"):
        origin_logits = origin_outputs.logits
    else:
        origin_logits = origin_outputs[0]
    
    logger.info(f"   Origin Logits Shape: {origin_logits.shape}")
    
    # ========== Compare ==========
    comparison_results = compare_logits(muse_logits, origin_logits)
    
    # ========== Save Results ==========
    save_dir = Path("/llm_reco/maosiyang/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # [FIX] Save individual logits files as well
    torch.save(muse_logits.detach().cpu(), save_dir / "muse_model_logits.pt")
    torch.save(origin_logits.detach().cpu(), save_dir / "origin_model_logits.pt")
    
    # Save dictionary
    torch.save({
        "muse_logits": muse_logits.detach().cpu(),
        "origin_logits": origin_logits.detach().cpu(),
        "comparison": comparison_results,
    }, save_dir / "comparison_results.pt")
    
    logger.info(f"\n💾 Results saved to {save_dir}")
    logger.info("\n✅ Comparison Completed!")

if __name__ == "__main__":
    run_comparison()

