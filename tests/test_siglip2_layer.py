"""
SigLIP Layer 0 Internal Debugging
Focus: Q/K/V Projections, Attention Output, and MLP
"""

import os
import sys
import logging
import torch
from transformers import AutoImageProcessor, SiglipVisionModel as HFSiglipVisionModel
from muse.config import SiglipVisionConfig
from muse.models.Siglip import SiglipVisionTransformer as SiglipVisionModel
from muse.training.common import set_default_dtype
import numpy
# Logging Setup
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def compare_tensors(name, t1, t2, atol=1e-3):
    t1 = t1.detach().float().cpu()
    t2 = t2.detach().float().cpu()
    
    # Handle Shape Mismatch (Conv vs Flattened)
    if t1.ndim == 4 and t2.ndim == 3:
        t1 = t1.flatten(2).transpose(1, 2)
    if t2.ndim == 4 and t1.ndim == 3:
        t2 = t2.flatten(2).transpose(1, 2)

    if t1.shape != t2.shape:
        logger.error(f"❌ {name} SHAPE MISMATCH: HF={t1.shape}, Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Thresholds: Strict for Linear layers, looser for Attention output (due to accumulation)
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max: {max_diff:.2e})"
    
    logger.info(f"{name:<35} | {status} | MeanDiff: {mean_diff:.2e}")
    if max_diff >= atol:
        logger.info(f"   HF Sample  : {t1.flatten()[:5].numpy()}")
        logger.info(f"   Muse Sample: {t2.flatten()[:5].numpy()}")

def _build_siglip_config(hf_cfg):
    """Build Muse config from HF config."""
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

# =============================================================================
# Main Test
# =============================================================================

def test_layer0_internals():
    torch.manual_seed(0)
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"
    
    logger.info("1. Loading Models...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(checkpoint_dir, device_map="auto", torch_dtype="auto")
    hf_model.eval()
    
    device = hf_model.device
    dtype = hf_model.dtype
    
    # Muse Setup
    config_dict = hf_model.config.to_dict()
    muse_config = _build_siglip_config(config_dict)
    with set_default_dtype(dtype):
        muse_model = SiglipVisionModel(muse_config)

    # Weight Loading (Using your corrected logic implicitly via convert_hf_state_dict)
    logger.info("2. Loading Weights...")
    hf_state_dict = hf_model.state_dict()
    prefixed_dict = {}
    for k, v in hf_state_dict.items():
        if k.startswith("vision_model."): new_k = f"siglip.{k}"
        else: new_k = f"siglip.vision_model.{k}"
        prefixed_dict[new_k] = v
        
    converted_dict = muse_model.convert_hf_state_dict(prefixed_dict)
    for k in converted_dict: converted_dict[k] = converted_dict[k].to(dtype)
    muse_model.load_state_dict(converted_dict, strict=False)
    muse_model = muse_model.to(device)
    muse_model.eval()

    # =========================================================================
    # 3. Hook Internals of Layer 0
    # =========================================================================
    activations = {"hf": {}, "muse": {}}

    def get_hook(model_name, layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple): output = output[0]
            activations[model_name][layer_name] = output.detach()
        return hook

    # --- Hooking Specific Sub-Modules of Layer 0 ---
    
    # 1. Norm 1 (Input to Attention)
    hf_model.vision_model.encoder.layers[0].layer_norm1.register_forward_hook(get_hook("hf", "norm1"))
    muse_model.encoder.layers[0].sa_norm.register_forward_hook(get_hook("muse", "norm1"))

    # 2. Q Projection (Before RoPE)
    hf_model.vision_model.encoder.layers[0].self_attn.q_proj.register_forward_hook(get_hook("hf", "q_proj"))
    muse_model.encoder.layers[0].attn.q_proj.register_forward_hook(get_hook("muse", "q_proj"))

    # 3. K Projection (Before RoPE)
    hf_model.vision_model.encoder.layers[0].self_attn.k_proj.register_forward_hook(get_hook("hf", "k_proj"))
    muse_model.encoder.layers[0].attn.k_proj.register_forward_hook(get_hook("muse", "k_proj"))

    # 4. Attention Output (After RoPE + Softmax + V + OutProj)
    hf_model.vision_model.encoder.layers[0].self_attn.out_proj.register_forward_hook(get_hook("hf", "attn_out"))
    muse_model.encoder.layers[0].attn.output_proj.register_forward_hook(get_hook("muse", "attn_out"))

    # 5. MLP Output (Final layer check)
    hf_model.vision_model.encoder.layers[0].mlp.register_forward_hook(get_hook("hf", "mlp_out"))
    muse_model.encoder.layers[0].mlp.register_forward_hook(get_hook("muse", "mlp_out"))

    # =========================================================================
    # 4. Forward & Compare
    # =========================================================================
    logger.info("3. Running Forward...")
    # Using random image to avoid Processor path issues, purely testing logic
    image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        hf_model(**inputs)
        # Manually construct grid
        h, w = 384//14, 384//14
        image_grid = [(1, h, w)] * inputs["pixel_values"].shape[0]
        muse_model(pixel_values=inputs["pixel_values"], image_grid_thw=image_grid)

    logger.info("\n" + "="*60)
    logger.info("LAYER 0 INTERNAL DIAGNOSIS")
    logger.info("="*60)

    # Tols
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-4

    # Check 1: Norm 1 (Should match perfectly as Embeddings matched)
    compare_tensors("1. Norm1 Output", activations["hf"]["norm1"], activations["muse"]["norm1"], atol=tol)

    # Check 2: Linear Projections (Should match perfectly if weights loaded)
    compare_tensors("2. Q_Proj Output (Pre-RoPE)", activations["hf"]["q_proj"], activations["muse"]["q_proj"], atol=tol)
    compare_tensors("3. K_Proj Output (Pre-RoPE)", activations["hf"]["k_proj"], activations["muse"]["k_proj"], atol=tol)

    # Check 3: Attention Output (Will FAIL if RoPE is wrong)
    compare_tensors("4. Attn Output (Post-RoPE)", activations["hf"]["attn_out"], activations["muse"]["attn_out"], atol=tol)

    # Check 4: MLP Output
    compare_tensors("5. MLP Output", activations["hf"]["mlp_out"], activations["muse"]["mlp_out"], atol=tol)

if __name__ == "__main__":
    test_layer0_internals()