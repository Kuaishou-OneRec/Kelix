"""
Test Script for Muse SigLIP (v2/RoPE/NaViT)
Target: Load raw training checkpoint (.pt) and verify forward pass.
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Union

# 假设你的模型代码在 muse.models.Siglip
# 请根据实际情况调整 import
try:
    from muse.config import SiglipVisionConfig
    from muse.models.Siglip import SiglipVisionTransformer as SiglipVisionModel
    from muse.training.common import set_default_dtype
except ImportError:
    # 如果路径不对，尝试直接引用当前目录（假设你把模型代码放在同级目录）
    sys.path.append(os.getcwd())
    # 这里需要你确保能 import 到你刚才贴出的 SiglipVisionTransformer
    pass


from muse.muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as KeyeVisionModel
# Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. Configuration (Manual Setup)
# =============================================================================

def get_so400m_config():
    """
    手动定义 SigLIP-SO400M (SigLIP 2) 的配置。
    根据 checkpoint 的实际情况，可能需要微调 (例如 image_size, patch_size)。
    """
    return SiglipVisionConfig(
        model_class="SiglipVisionTransformer",
        # 标准 SO400M 参数
        image_size=384,
        patch_size=14,
        num_channels=3,
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
        # 自动计算 max_seq_len
        max_seq_len=(384 // 14) ** 2, 
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # SigLIP 2 特性
        has_learnable_position_embedding=False, # 通常 v2 是 False, 靠 RoPE
        use_qk_norm=False, # 视具体训练配置而定，Paligemma 是 True，纯 Siglip2 可能是 False
        qk_norm_eps=1e-6,
        rope_theta=10000.0, # RoPE Base
        attention_function="eager",
        output_attentions=False,
        output_hidden_states=False,
    )

# =============================================================================
# 2. Weight Loading Logic (Smart Converter)
# =============================================================================

def load_checkpoint_to_muse(model, checkpoint_path, device):
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # 1. Load Raw State Dict
    try:
        # map_location='cpu' 防止显存爆炸
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return False

    # 处理可能的嵌套 (DeepSpeed/Megatron 常见结构)
    if "module" in state_dict:
        logger.info("Found 'module' key, unpacking...")
        state_dict = state_dict["module"]
    elif "state_dict" in state_dict:
        logger.info("Found 'state_dict' key, unpacking...")
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # 2. Key Conversion
    # 目标是去除不需要的前缀，匹配 Muse 的 keys
    # Muse keys 示例: 
    #   embeddings.patch_embedding.weight
    #   encoder.layers.0.attn.q_proj.weight
    #   encoder.rope.inv_freq (buffer, 不用加载)
    
    converted_dict = {}
    skipped_keys = []
    
    # 常见的前缀垃圾
    prefixes_to_strip = [
        "module.", 
        "vision_tower.", 
        "vision_model.", 
        "siglip.vision_model.",
        "model.vision_model."
    ]
    
    logger.info("Converting keys...")
    for k, v in state_dict.items():
        new_k = k
        
        # 1. 剥离前缀
        clean = False
        while not clean:
            clean = True
            for p in prefixes_to_strip:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    clean = False # 继续检查是否有双重前缀
        
        # 2. 映射特定层名称 (如果原始权重名称和 Muse 不一样)
        # 你的 Muse 模型使用的是:
        #   self_attn -> attn
        #   layer_norm1 -> sa_norm
        #   layer_norm2 -> mlp_norm
        #   out_proj -> output_proj
        #   mlp.fc1 -> mlp.w1 (或者保持 fc1, 看你的代码)
        
        # 根据你贴出的代码:
        # SiglipEncoderLayer 有 self_attn, layer_norm1, mlp, layer_norm2
        # 但是 Muse convert_hf_state_dict 里映射成了 sa_norm, mlp_norm, attn 等
        # **你的模型定义中使用的是**：
        #   self.sa_norm = nn.LayerNorm...
        #   self.mlp_norm = nn.LayerNorm...
        #   self.attn = MultiHeadAttention...
        
        # 处理 Encoder Layer 内部命名
        if "encoder.layers." in new_k:
            parts = new_k.split(".")
            # 假设结构 encoder.layers.{i}.{submodule}
            # 我们需要重命名 submodule 部分
            
            # self_attn -> attn
            new_k = new_k.replace("self_attn", "attn")
            # out_proj -> output_proj (Muse MultiHeadAttention 常用名)
            new_k = new_k.replace("out_proj", "output_proj")
            
            # layer_norm1 -> sa_norm
            new_k = new_k.replace("layer_norm1", "sa_norm")
            # layer_norm2 -> mlp_norm
            new_k = new_k.replace("layer_norm2", "mlp_norm")
            
            # mlp.fc1 -> mlp.gate_proj (视你的 SiglipMLP 实现而定)
            # 你的 SiglipMLP 代码: gate_proj=fc1, down_proj=fc2
            # 你的 convert 代码: mlp.fc1 -> w1 ?? 
            # 让我们看你的 SiglipMLP 类:
            #   self.fc1 = nn.Linear...
            #   self.fc2 = nn.Linear...
            #   return FeedForward(gate_proj=fc1, down_proj=fc2...)
            # 所以 Muse 模型里实际的 Parameter 名字是:
            #   encoder.layers.0.mlp.gate_proj.weight (如果 FeedForward 把 fc1 赋给了 gate_proj)
            #   **但是**，你的 SiglipMLP 返回的是 FeedForward 对象。
            #   如果 FeedForward 是简单的赋值，名字可能是 mlp.gate_proj.weight
            
            # 假设 state_dict 里是 mlp.fc1，我们需要根据 FeedForward 的内部结构去改。
            # 简单起见，我们假设你的 FeedForward 只是包装，内部名字取决于 SiglipMLP.__init__
            # 在 SiglipMLP.__init__ 中: self.fc1 = ...
            # 如果 SiglipMLP 是 nn.Module 且作为 FeedForward 的一部分...
            # **修正**: 你的代码 SiglipMLP 返回的是一个 FeedForward **对象**。
            # 这意味着 SiglipMLP 这个函数是工厂函数。
            # FeedForward 类通常有 `w1`, `w2`, `w3` 或者 `gate_proj`, `up_proj`, `down_proj`。
            # 假设 FeedForward 的定义标准（如 Llama）：
            #   fc1 (gate) -> w1/gate_proj
            #   fc2 (down) -> w2/down_proj
            
            # 这里做一个通用尝试，你可能需要根据报错微调
            if "mlp.fc1" in new_k:
                new_k = new_k.replace("mlp.fc1", "mlp.gate_proj") 
            if "mlp.fc2" in new_k:
                new_k = new_k.replace("mlp.fc2", "mlp.down_proj")

        converted_dict[new_k] = v

    # 3. Load
    logger.info(f"Loading {len(converted_dict)} keys into Muse model...")
    missing, unexpected = model.load_state_dict(converted_dict, strict=False)
    
    if len(missing) > 0:
        logger.warning(f"⚠️ Missing Keys ({len(missing)}): {missing[:5]} ...")
    if len(unexpected) > 0:
        logger.warning(f"⚠️ Unexpected Keys ({len(unexpected)}): {unexpected[:5]} ...")
        
    return True

# =============================================================================
# 3. Data Preparation (Mocking the Image Processor)
# =============================================================================

def get_dummy_input(config, device, batch_size=1):
    """
    模拟 SiglipImageProcessor 的输出。
    Muse 的 SiglipVisionEmbeddings 要求 5D 输入: [Batch, Seq(T), Channel, Height, Width]
    """
    H, W = config.image_size, config.image_size # 384
    C = config.num_channels
    P = config.patch_size
    
    # 构造 dummy image tensor
    # 假设单帧图片 (T=1)
    # Shape: [B, T, C, H, W] -> [1, 1, 3, 384, 384]
    pixel_values = torch.randn(batch_size, 1, C, H, W, device=device, dtype=torch.float32)
    
    # 构造 Grid
    # grid_h = 384 // 14 = 27
    # grid_w = 384 // 14 = 27
    grid_h = H // P
    grid_w = W // P
    
    # SiglipVisionEmbeddings 中: for t, h, w in flatten_image_grid_thw:
    # 格式通常是 (T_grid, H_grid, W_grid)
    # 对于单图，T_grid=1
    image_grid_thw = [(1, grid_h, grid_w)] * batch_size
    
    return pixel_values, image_grid_thw

# =============================================================================
# 4. Main Test Function
# =============================================================================

def test_new_model_checkpoint():
    # 1. Setup
    checkpoint_path = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        # return # 注释掉以便在没有文件时也能测试代码逻辑(用随机权重)
        logger.warning("Continuing with Random Weights for Logic Testing...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    # 2. Config & Model
    config = get_so400m_config()
    logger.info(f"Model Config: {config}")
    
    with set_default_dtype(dtype):
        model = SiglipVisionModel(config)
        
    model = model.to(device)
    model.eval()
    
    # 3. Load Weights
    if os.path.exists(checkpoint_path):
        success = load_checkpoint_to_muse(model, checkpoint_path, device)
        if success:
            logger.info("✅ Checkpoint loaded successfully (with potential strict=False warnings).")
    
    # 4. Prepare Input
    logger.info("Preparing inputs...")
    pixel_values, image_grid_thw = get_dummy_input(config, device)
    
    # Cast to model dtype
    pixel_values = pixel_values.to(dtype=dtype)
    
    logger.info(f"Input Shape: {pixel_values.shape}")
    logger.info(f"Grid Info: {image_grid_thw}")
    
    # 5. Forward Pass
    logger.info("Running forward pass...")
    try:
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
            
        last_hidden = outputs["last_hidden_state"]
        
        logger.info("\n" + "="*40)
        logger.info("FORWARD PASS SUCCESS")
        logger.info("="*40)
        logger.info(f"Output Shape: {last_hidden.shape}")
        logger.info(f"Output Dtype: {last_hidden.dtype}")
        logger.info(f"Output Stats: Mean={last_hidden.mean().item():.4f}, Std={last_hidden.std().item():.4f}")
        logger.info(f"First 10 values: {last_hidden[0, 0, :10].float().cpu().numpy()}")
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_model_checkpoint()