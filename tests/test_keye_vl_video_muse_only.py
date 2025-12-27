"""
Keye-VL Pipeline: Muse Model Inference Only
==================================================
Input: Video
Output: Save Muse Model Logits to /llm_reco/maosiyang/

使用与 test_keye_vl_video_circle.py 相同的加载方式，
从 HF config.json 手动构建 Muse config，确保前向对齐。
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'  # Disable sequence parallel

import sys
import logging
import glob
import json
import tqdm
import torch
from pathlib import Path
from typing import Dict, Any
from safetensors.torch import load_file
from transformers import AutoProcessor

# === 导入 Muse 模型 ===
from muse.models.keye_tokenizer_end2end_video import modeling as muse_mod
from muse.config import KeyeTokenizerEnd2EndVideoConfig
from muse.training.common import set_default_dtype

# === 导入 Processor 相关 ===
try:
    from tests.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info
except ImportError:
    sys.path.append(os.getcwd())
    from tests.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

# 模型目录（包含 config.json 和 *.safetensors 文件，HF 格式）
MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_video_baseline"

# 输入视频路径
VIDEO_PATH = "/llm_reco/maosiyang/23b77760a4304e9092eb3b45b7bf8050.mp4"

# 输出保存路径
SAVE_PATH = "/llm_reco/maosiyang/muse_model_logits_video.pt"

# 设备和数据类型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# =========================================================================
# Helper Functions (与 test_keye_vl_video_circle.py 一致)
# =========================================================================

def _load_config_json(ckpt_path: str) -> Dict[str, Any]:
    """加载 HF 格式的 config.json"""
    p = Path(ckpt_path)
    base_dir = p if p.is_dir() else p.parent
    cfg_path = base_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {base_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _load_checkpoint_robust(path_str: str, device="cpu") -> Dict[str, torch.Tensor]:
    """加载 safetensors 或 bin 文件"""
    path = Path(path_str)
    if path.is_file():
        state_dict = torch.load(path, map_location=device)
        return state_dict.get("module", state_dict)

    state_dict = {}
    # 优先处理 safetensors 文件
    logger.info(f"Loading weights from {path_str}...")
    for f in tqdm.tqdm(os.listdir(path_str)):
        if f.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(path_str, f)))

    # 如果没有 safetensors 文件，回退到 bin 文件
    if not state_dict:
        bin_files = sorted(glob.glob(str(path / "*.bin")))
        if bin_files:
            for f in bin_files:
                if any(x in f for x in ["training_args", "optimizer", "scheduler"]): 
                    continue
                part = torch.load(f, map_location=device)
                if "module" in part: 
                    part = part["module"]
                state_dict.update(part)

    return state_dict


def load_muse_model(ckpt_path: str, raw_cfg: Dict, device: str, dtype: torch.dtype):
    """
    从 HF config.json 手动构建 Muse config 并加载模型
    与 test_keye_vl_video_circle.py 完全一致
    """
    from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig
    
    logger.info("\n" + "="*60)
    logger.info("🚀 Loading Muse Model (from HF config)...")
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

    model_cfg = KeyeTokenizerEnd2EndVideoConfig(
        qwen_config=qwen_cfg,
        vision_config=vision_cfg,
        tokenizer_config=tokenizer_cfg,
        image_token_id=raw_cfg.get("image_token_id", 151655),
        video_token_id=raw_cfg.get("video_token_id", 151656),
        pool="sum",
    )

    with set_default_dtype(dtype):
        muse_model = muse_mod.KeyeTokenizerEnd2EndVideo(model_cfg).to(device)

    logger.info("📥 Loading Muse Weights (with HF->Muse conversion)...")
    state_dict = _load_checkpoint_robust(ckpt_path, device="cpu")
    muse_state = muse_model.convert_hf_state_dict(state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)
    muse_model.to(device, dtype)
    muse_model.eval()
    
    logger.info("✅ Muse Model Loaded.")
    return muse_model


# =========================================================================
# Main
# =========================================================================

def main():
    logger.info(f"🔧 Device: {DEVICE}, Dtype: {DTYPE}")
    logger.info(f"📂 Model Dir: {MODEL_DIR}")
    
    # --- 1. 加载 HF 格式的 config.json ---
    logger.info(f"📄 Loading HF config.json...")
    raw_cfg = _load_config_json(MODEL_DIR)
    
    # --- 2. 使用与 test_keye_vl_video_circle.py 相同的方式加载模型 ---
    model = load_muse_model(MODEL_DIR, raw_cfg, DEVICE, DTYPE)
    
    # --- 4. 加载 Processor ---
    logger.info("⚙️ Loading Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    # --- 5. 构造输入 ---
    logger.info(f"📹 Processing video: {VIDEO_PATH}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_PATH},
            ],
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    logger.info(f"📝 Prompt text: {repr(text)}")
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    logger.info(f"   -> process_vision_info: images={len(image_inputs) if image_inputs else 0}, videos={len(video_inputs) if video_inputs else 0}")
    
    # Run processor
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    ).to(DEVICE)
    
    logger.info(f"   Input IDs Shape: {inputs['input_ids'].shape}")
    if 'pixel_values_videos' in inputs:
        logger.info(f"   Video Pixel Values Shape: {inputs['pixel_values_videos'].shape}")
    if 'video_grid_thw' in inputs:
        logger.info(f"   Video Grid THW: {inputs['video_grid_thw'].tolist()}")
    
    # --- 6. 执行推理并保存 Logits ---
    logger.info("🔥 Running forward pass...")
    inputs.pop("num_frames", None)
    with torch.no_grad():
        outputs = model(**inputs)
        
        # 提取 logits
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]
    
    logger.info(f"📊 Logits shape: {logits.shape}")
    logger.info(f"   Logits dtype: {logits.dtype}")
    logger.info(f"   First token logits (top 10): {logits[0, 0, :10].float().cpu().numpy()}")
    logger.info(f"   Last token logits (top 10): {logits[0, -1, :10].float().cpu().numpy()}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # 保存到 CPU
    torch.save(logits.detach().cpu(), SAVE_PATH)
    logger.info(f"💾 Saved logits to {SAVE_PATH}")
    logger.info("✅ Done!")


if __name__ == "__main__":
    main()

