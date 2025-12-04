"""
Deep Integration Test for SigLIP (Layer-wise Debugging with Hooks)
"""

import os
import sys
import logging
from typing import Any, Dict

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel as HFSiglipVisionModel
from muse.config import SiglipVisionConfig
from muse.models.Siglip import SiglipVisionTransformer as SiglipVisionModel
from muse.training.common import set_default_dtype

# Setup Logging
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def _build_siglip_config(hf_cfg: Dict[str, Any]) -> SiglipVisionConfig:
    """Map Hugging Face config to Muse SiglipVisionConfig."""
    image_size = hf_cfg.get("image_size", 384)
    patch_size = hf_cfg.get("patch_size", 14)
    default_max_seq_len = (image_size // patch_size) ** 2
    
    return SiglipVisionConfig(
        model_class="SiglipVisionTransformer",
        image_size=image_size,
        patch_size=patch_size,
        num_channels=hf_cfg.get("num_channels", 3),
        hidden_size=hf_cfg.get("hidden_size", 1152),
        num_hidden_layers=hf_cfg.get("num_hidden_layers", 27),
        num_attention_heads=hf_cfg.get("num_attention_heads", 16),
        intermediate_size=hf_cfg.get("intermediate_size", 4304),
        max_seq_len=hf_cfg.get("max_seq_len", default_max_seq_len),
        layer_norm_eps=hf_cfg.get("layer_norm_eps", 1e-6),
        attention_dropout=hf_cfg.get("attention_dropout", 0.0),
        has_learnable_position_embedding=hf_cfg.get("has_learnable_position_embedding", False),
        use_qk_norm=hf_cfg.get("use_qk_norm", False),
        qk_norm_eps=hf_cfg.get("qk_norm_eps", 1e-6),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        attention_function="eager",
        output_attentions=False,
        output_hidden_states=False,
    )

def create_dummy_image(size: int = 384) -> Image.Image:
    np.random.seed(42)
    data = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)

def format_tensor_val(t: torch.Tensor, n: int = 5) -> str:
    """Helper to format first n values of a tensor flat."""
    vals = t.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join([f"{x:.5f}" for x in vals]) + "]"

def log_separator(title: str):
    logger.info(f"\n{'='*80}\n {title.center(78)} \n{'='*80}")

def compare_tensors_verbose(name: str, tensor_hf: torch.Tensor, tensor_muse: torch.Tensor, atol=1e-3):
    """Detailed comparison of two tensors."""
    t1 = tensor_hf.detach().float().cpu()
    t2 = tensor_muse.detach().float().cpu()
    
    # Auto-align shapes (HF sometimes outputs tuple, or [B, N, D] vs [B, D, H, W])
    if t1.ndim == 4 and t2.ndim == 3: # Conv vs Flattened
        t1 = t1.flatten(2).transpose(1, 2)
    
    if t1.shape != t2.shape:
        logger.error(f"❌ SHAPE MISMATCH [{name}]: HF={t1.shape} vs Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max Diff: {max_diff:.2e})"
    logger.info(f"{name:<35} | {status} | MaxDiff: {max_diff:.2e} | MeanDiff: {mean_diff:.2e}")
    
    if max_diff >= atol:
        logger.info(f"   Sample HF  : {format_tensor_val(t1, 5)}")
        logger.info(f"   Sample Muse: {format_tensor_val(t2, 5)}")

# =============================================================================
# Main Debug Logic
# =============================================================================

def test_siglip_layer_by_layer():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"
    logger.info(f"Testing checkpoint: {checkpoint_dir}")

    # 1. Load HF Model
    logger.info("Loading HF model...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(
        checkpoint_dir, torch_dtype="auto", device_map="auto"
    )
    hf_model.eval()
    
    device = hf_model.device
    dtype = hf_model.dtype
    logger.info(f"Model Device: {device}, Dtype: {dtype}")

    # 2. Load Muse Model
    hf_state_dict = hf_model.state_dict()
    muse_config = _build_siglip_config(hf_model.config.to_dict())
    
    with set_default_dtype(dtype):
        muse_model = SiglipVisionModel(muse_config)

    # 3. Intelligent Weight Conversion & Loading
    log_separator("Weight Loading")
    
    # Prepare prefix mapping
    prefixed_state_dict = {}
    for key, value in hf_state_dict.items():
        # HF key: "vision_model.embeddings..." -> Muse expects "siglip.vision_model.embeddings..."
        # Or HF key: "embeddings..." -> Muse expects "siglip.vision_model.embeddings..."
        if key.startswith("vision_model."):
            new_key = f"siglip.{key}"
        else:
            new_key = f"siglip.vision_model.{key}"
        prefixed_state_dict[new_key] = value

    converted_state_dict = muse_model.convert_hf_state_dict(prefixed_state_dict)
    
    # Convert dtype
    for key in converted_state_dict:
        converted_state_dict[key] = converted_state_dict[key].to(dtype=dtype)

    load_res = muse_model.load_state_dict(converted_state_dict, strict=False)
    logger.info(f"Load Result: {load_res}")
    
    if len(load_res.missing_keys) > 0:
        logger.error("⚠️ Stop! Weights are missing. Fix loading first.")
        # return # You can comment this out if you still want to run

    muse_model = muse_model.to(device=device, dtype=dtype)
    muse_model.eval()

    # =========================================================================
    # 4. HOOKS Setup (The Magic Part)
    # =========================================================================
    activations = {"hf": {}, "muse": {}}

    def get_hook(model_name, layer_name):
        def hook(module, input, output):
            # HF outputs are often tuples (hidden, attn), we want hidden (idx 0)
            if isinstance(output, tuple):
                output = output[0]
            activations[model_name][layer_name] = output.detach()
        return hook

    # --- Register Hooks ---
    
    # 1. Embeddings (Post Conv + PosEmbed)
    hf_model.vision_model.embeddings.register_forward_hook(get_hook("hf", "embeddings"))
    muse_model.embeddings.register_forward_hook(get_hook("muse", "embeddings"))

    # 2. Layer 0 (Check RoPE and first MLP)
    hf_model.vision_model.encoder.layers[0].register_forward_hook(get_hook("hf", "layer_0"))
    muse_model.encoder.layers[0].register_forward_hook(get_hook("muse", "layer_0"))

    # 3. Middle Layer
    mid_layer = 13
    hf_model.vision_model.encoder.layers[mid_layer].register_forward_hook(get_hook("hf", "layer_mid"))
    muse_model.encoder.layers[mid_layer].register_forward_hook(get_hook("muse", "layer_mid"))

    # 4. Last Layer (Before final LN)
    last_layer = 26
    hf_model.vision_model.encoder.layers[last_layer].register_forward_hook(get_hook("hf", "layer_last"))
    muse_model.encoder.layers[last_layer].register_forward_hook(get_hook("muse", "layer_last"))


    # =========================================================================
    # 5. Forward Pass
    # =========================================================================
    log_separator("Forward Pass & Layer Debug")
    
    image = create_dummy_image(muse_config.image_size)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]
    
    # Verify Input
    logger.info(f"Input Shape: {pixel_values.shape}")
    logger.info(f"Input Stats: Mean={pixel_values.mean():.4f}, Std={pixel_values.std():.4f}")

    # Run HF
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        hf_final = hf_outputs.last_hidden_state

    # Run Muse
    # Calculate Grid Manually (mimic automatic logic) to ensure it's passed
    batch_size, _, height, width = pixel_values.shape
    h_grid = height // muse_config.patch_size
    w_grid = width // muse_config.patch_size
    image_grid_thw = [(1, h_grid, w_grid)] * batch_size

    with torch.no_grad():
        muse_outputs = muse_model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw # Explicitly pass to match your logic
        )
        muse_final = muse_outputs["last_hidden_state"]

    # =========================================================================
    # 6. Comparisons
    # =========================================================================
    
    # 阈值设置：bfloat16 精度较低，需要宽容一点 (1e-2)；float32 可以严格 (1e-5)
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    # 1. Embeddings Comparison
    # 如果这里挂了 -> Conv2d 权重、Position Embedding Resize 逻辑、或者 Input Preprocessing 有问题
    compare_tensors_verbose("1. Embeddings Output", 
                           activations["hf"]["embeddings"], 
                           activations["muse"]["embeddings"], atol=tol)

    # 2. Layer 0 Comparison
    # 如果 Embeddings 对了但这里挂了 -> Attention 计算错误（尤其是 RoPE 实现！）或者 MLP Bias 丢失
    compare_tensors_verbose("2. Encoder Layer 0 Output", 
                           activations["hf"]["layer_0"], 
                           activations["muse"]["layer_0"], atol=tol)

    # 3. Mid Layer Comparison
    compare_tensors_verbose("3. Encoder Layer 13 Output", 
                           activations["hf"]["layer_mid"], 
                           activations["muse"]["layer_mid"], atol=tol)
                           
    # 4. Last Layer Comparison
    compare_tensors_verbose("4. Encoder Layer 26 Output", 
                           activations["hf"]["layer_last"], 
                           activations["muse"]["layer_last"], atol=tol)

    # 5. Final Output (Post LayerNorm)
    compare_tensors_verbose("5. Final Output (Post-LN)", hf_final, muse_final, atol=tol)


if __name__ == "__main__":
    test_siglip_layer_by_layer()