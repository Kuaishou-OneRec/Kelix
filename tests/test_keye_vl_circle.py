"""
Keye-VL Pipeline: Muse Model Inference Only
===========================================
Input: Generated Circle Image
Output: Muse Model Logits
"""

import os
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

# === 导入 Processor 相关 ===
from transformers import AutoTokenizer, AutoProcessor
from tests.models.keye_vl_tokenizer_image.image_processing_keye import SiglipImageProcessor

# 假设 KeyeProcessor 在 tests 目录下，如果路径不同请调整
try:
    from tests.models.keye_vl_tokenizer_image.processing_keye import KeyeProcessor
except ImportError:
    sys.path.append(os.getcwd())
    from tests.models.keye_vl_tokenizer_image.processing_keye import KeyeProcessor

# 配置日志
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# CKPT 路径
DEFAULT_CKPT = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/"

# =========================================================================
# Helper Functions
# =========================================================================

def generate_circle_image(size=(384, 384), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """生成一个包含圆形的测试图片"""
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2 - outline_width
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
    
    # 文件夹加载逻辑 (SafeTensors / Bin / PT)
    state_dict = {}
    st_files = sorted(glob.glob(str(path / "*.safetensors")))
    if st_files:
        from safetensors.torch import safe_open
        for f in st_files:
            with safe_open(f, framework="pt", device=device) as open_f:
                for k in open_f.keys(): state_dict[k] = open_f.get_tensor(k)
        return state_dict
    
    bin_files = sorted(glob.glob(str(path / "*.bin")))
    if bin_files:
        for f in bin_files:
            if any(x in f for x in ["training_args", "optimizer", "scheduler"]): continue
            part = torch.load(f, map_location=device)
            if "module" in part: part = part["module"]
            state_dict.update(part)
        return state_dict
        
    pt_files = sorted(glob.glob(str(path / "*.pt")))
    if pt_files:
         for f in pt_files:
            part = torch.load(f, map_location=device)
            if "module" in part: part = part["module"]
            state_dict.update(part)
         return state_dict
    raise ValueError(f"No checkpoint found in {path}")

# =========================================================================
# Input Preparation
# =========================================================================

def prepare_inputs(ckpt_path: str, device: str, dtype: torch.dtype):
    """生成圆形图片并通过 Processor 处理为模型输入"""
    logger.info("⚙️ Loading Tokenizer & ImageProcessor...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    image_processor = SiglipImageProcessor.from_pretrained(ckpt_path)
    processor = KeyeProcessor(image_processor=image_processor, tokenizer=tokenizer)

    logger.info("🎨 Generating Circle Image...")
    image = generate_circle_image(size=(384, 384))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."}, 
            ],
        }
    ]

    logger.info("📝 Applying Chat Template...")
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 提取图片对象
    image_inputs = [image] # 简单起见直接使用上面生成的对象，如果是多图需从 messages 解析

    logger.info("🔄 Running Processor...")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=None,
        padding=False, # 强制关闭 Padding
        truncation=False,
        return_tensors="pt",
    )

    # 转移到 Device
    model_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].to(device, dtype=dtype),
        "image_grid_thw": inputs["image_grid_thw"].to(device)
    }
    
    # 如果 Processor 返回 [1, num_patches, ...]，去掉 batch 维以匹配模型输入预期
    if model_inputs["pixel_values"].dim() == 5 and model_inputs["pixel_values"].shape[0] == 1:
        model_inputs["pixel_values"] = model_inputs["pixel_values"].squeeze(0)

    logger.info(f"   -> Input IDs Shape: {model_inputs['input_ids'].shape}")
    return model_inputs

# =========================================================================
# Main Execution
# =========================================================================

def run_muse_inference():
    ckpt_path = DEFAULT_CKPT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 
    
    logger.info(f"Loading Config from: {ckpt_path}")
    raw_cfg = _load_config_json(ckpt_path)
    
    # --- 构建 Muse Configs ---
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
    
    # [FIX] 手动强制设置 _attn_implementation 避免 SiglipAttention 报错
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

    # --- 初始化 Muse 模型 ---
    with set_default_dtype(dtype):
        logger.info("🚀 Initializing Muse Model...")
        muse_model = muse_mod.KeyeTokenizerEnd2EndImage(
            qwen_config=qwen_cfg,
            vision_config=vision_cfg,
            tokenizer_config=tokenizer_cfg,
            image_token_id=raw_cfg.get("image_token_id", 151655),
            pool="sum"
        ).to(device)

    # --- 加载权重 ---
    logger.info("📥 Loading Weights...")
    state_dict = _load_checkpoint_robust(ckpt_path, device="cpu")
    # 转换 HF 权重到 Muse 格式
    muse_state = muse_model.convert_hf_state_dict(state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)
    muse_model.to(device, dtype)
    muse_model.eval()

    # --- 准备输入 ---
    inputs = prepare_inputs(ckpt_path, device, dtype)

    # --- 前向传播 ---
    logger.info("🔥 Running Muse Forward...")
    with torch.no_grad():
        outputs = muse_model(**inputs)

# --- 提取 Logits ---
    # Muse 的输出可能是 dict 或 object，这里做兼容处理
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        # 如果 outputs 是 tuple，通常 logits 是第一个元素
        logits = outputs[0]

    if logits is not None:
        logger.info(f"\n✅ Muse Logits Obtained!")
        logger.info(f"   Shape: {logits.shape}")
        logger.info(f"   Dtype: {logits.dtype}")
        
        # 打印部分值供检查
        first_token_logits = logits[0, 0, :5].float().cpu().numpy()
        last_token_logits = logits[0, -1, :5].float().cpu().numpy()
        logger.info(f"   First token logits (top 5 dims): {first_token_logits}")
        logger.info(f"   Last token logits  (top 5 dims): {last_token_logits}")

        # === [新增] 保存 Logits 到 .pt 文件 ===
        save_path = "/llm_reco/maosiyang/muse_logits.pt"
        logger.info(f"💾 Saving logits to {save_path}...")
        # detach并转到cpu保存，方便后续加载查看
        torch.save(logits.detach().cpu(), save_path)
        logger.info("✅ Save Completed.")
    else:
        logger.error("❌ Failed to extract logits from model output.")

if __name__ == "__main__":
    run_muse_inference()