"""
Keye-VL Pipeline: Muse Model Inference Only
==================================================
Input: Video
Output: Save Muse Model Logits to /llm_reco/maosiyang/
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'  # Disable sequence parallel

import sys
import logging
import tqdm
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoProcessor

# === 导入 Muse 模型 ===
from muse.models import get_model_class
from muse.config import load_config
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

# 模型目录（包含 muse_config.json 和 *.safetensors 文件）
MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"

# 输入视频路径
VIDEO_PATH = "/llm_reco/maosiyang/23b77760a4304e9092eb3b45b7bf8050.mp4"

# 输出保存路径
SAVE_PATH = "/llm_reco/maosiyang/muse_model_logits_video.pt"

# 设备和数据类型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# =========================================================================
# Main
# =========================================================================

def main():
    logger.info(f"🔧 Device: {DEVICE}, Dtype: {DTYPE}")
    logger.info(f"📂 Model Dir: {MODEL_DIR}")
    
    # --- 1. 加载配置 ---
    config_path = Path(MODEL_DIR) / "muse_config.json"
    logger.info(f"📄 Loading config from {config_path}...")
    model_config = load_config(config_path)
    
    # --- 2. 获取模型类并创建模型 ---
    model_class_name = model_config.model_class
    logger.info(f"🚀 Creating model: {model_class_name}...")
    model_cls = get_model_class(model_class_name)
    
    with set_default_dtype(DTYPE):
        model = model_cls(model_config)
    
    # --- 3. 加载权重 ---
    logger.info(f"📥 Loading weights from {MODEL_DIR}...")
    sd = {}
    safetensor_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")]
    for f in tqdm.tqdm(safetensor_files, desc="Loading safetensors"):
        sd.update(load_file(os.path.join(MODEL_DIR, f)))
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if missing_keys:
        logger.warning(f"⚠️ Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        logger.warning(f"⚠️ Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    model = model.to(DEVICE).to(DTYPE)
    model.eval()
    logger.info("✅ Model loaded successfully!")
    
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

