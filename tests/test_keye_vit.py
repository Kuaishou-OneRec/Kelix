"""
Integration test to ensure Muse KeyeVisionTransformer matches Hugging Face (Origin) implementation.
"""

import os
import sys
import types
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Muse imports
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.training.common import set_default_dtype

# === Mock/Import HF Configs to support Origin Model loading ===
from transformers import PretrainedConfig

class HFKeyeVisionConfig(PretrainedConfig):
    model_type = "siglip_vision"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items(): setattr(self, k, v)

class HFKeyeConfig(PretrainedConfig):
    model_type = "keye"
    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_config = vision_config

def _ensure_origin_ready():
    # Helper to inject config classes so Origin model can import them
    mod = "muse.muse.models.keye_vit.configuration_keye"
    if mod not in sys.modules:
        c = types.ModuleType(mod)
        c.KeyeConfig = HFKeyeConfig
        c.KeyeVisionConfig = HFKeyeVisionConfig
        sys.modules[mod] = c

_ensure_origin_ready()
# Import the Reference Implementation (Origin)
# Assuming this file exists in your path as per previous debug sessions
from muse.models.keye_vit import modeling_keye_origin as keye_origin
OriginKeyeVisionModel = keye_origin.SiglipVisionModel 

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _build_muse_config(hf_cfg: Dict[str, Any]) -> KeyeVisionConfig:
    """Map Hugging Face/Origin config dict to Muse KeyeVisionConfig."""
    return KeyeVisionConfig(
        hidden_size=hf_cfg.get("hidden_size", 1152),
        intermediate_size=hf_cfg.get("intermediate_size", 4304),
        num_hidden_layers=hf_cfg.get("num_hidden_layers", 27),
        num_attention_heads=hf_cfg.get("num_attention_heads", 16),
        num_channels=hf_cfg.get("num_channels", 3),
        image_size=hf_cfg.get("image_size", 384),
        patch_size=hf_cfg.get("patch_size", 14),
        layer_norm_eps=hf_cfg.get("layer_norm_eps", 1e-6),
        attention_dropout=hf_cfg.get("attention_dropout", 0.0),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        # Ensure we use eager for comparison to avoid kernel nondeterminism
        attention_function="eager", 
        # Additional params
        use_qk_norm=hf_cfg.get("use_qk_norm", False),
        qk_norm_eps=hf_cfg.get("qk_norm_eps", 1e-6),
        vision_use_head=False, # We are testing the vision tower
        has_learnable_position_embedding=hf_cfg.get("has_learnable_position_embedding", False)
    )

def create_dummy_image(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)

def check_weights(hf_state_dict: Dict[str, torch.Tensor], 
                  muse_model: nn.Module, 
                  device: torch.device, 
                  dtype: torch.dtype):
    """Compare weights layer by layer between HF dict and Muse model."""
    print(f"\n{'='*60}")
    print("Weight Value Comparison")
    print(f"{'='*60}")

    muse_state_dict = muse_model.state_dict()
    config = muse_model.config
    
    total_checked = 0
    total_matched = 0
    issues = []

    # 1. Check Embeddings
    # Origin: embeddings.patch_embedding.weight
    # Muse: embeddings.patch_embedding.weight
    embed_map = {
        "vision_model.embeddings.patch_embedding.weight": "embeddings.patch_embedding.weight",
        "vision_model.embeddings.patch_embedding.bias": "embeddings.patch_embedding.bias",
        "vision_model.embeddings.position_embedding.weight": "embeddings.position_embedding.weight",
    }
    
    for hf_key, muse_key in embed_map.items():
        # Remove 'siglip.' prefix if present in hf_state_dict keys for lookup
        lookup_key = "siglip." + hf_key
        if lookup_key in hf_state_dict and muse_key in muse_state_dict:
            total_checked += 1
            hf_w = hf_state_dict[lookup_key].to(device=device, dtype=dtype)
            muse_w = muse_state_dict[muse_key]
            diff = (hf_w - muse_w).abs().max().item()
            if diff > 1e-5:
                issues.append(f"{muse_key}: diff={diff:.6e}")
            else:
                total_matched += 1

    # 2. Check Encoder Layers
    for i in range(config.num_hidden_layers):
        # Mappings for one layer
        # Origin: vision_model.encoder.layers.0.self_attn.q_proj.weight
        # Muse: encoder.layers.0.attn.q_proj.weight
        layer_map = [
            # Norms
            (f"vision_model.encoder.layers.{i}.layer_norm1.weight", f"encoder.layers.{i}.sa_norm.weight"),
            (f"vision_model.encoder.layers.{i}.layer_norm1.bias",   f"encoder.layers.{i}.sa_norm.bias"),
            (f"vision_model.encoder.layers.{i}.layer_norm2.weight", f"encoder.layers.{i}.mlp_norm.weight"),
            (f"vision_model.encoder.layers.{i}.layer_norm2.bias",   f"encoder.layers.{i}.mlp_norm.bias"),
            # Attn
            (f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight", f"encoder.layers.{i}.attn.q_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias",   f"encoder.layers.{i}.attn.q_proj.bias"),
            (f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight", f"encoder.layers.{i}.attn.k_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias",   f"encoder.layers.{i}.attn.k_proj.bias"),
            (f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight", f"encoder.layers.{i}.attn.v_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias",   f"encoder.layers.{i}.attn.v_proj.bias"),
            (f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.attn.output_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias",   f"encoder.layers.{i}.attn.output_proj.bias"),
            # MLP (KeyeMLP: fc1->gate_proj(w1), fc2->down_proj(w2))
            # Note: Muse FeedForward maps w1=gate, w2=down.
            (f"vision_model.encoder.layers.{i}.mlp.fc1.weight", f"encoder.layers.{i}.mlp.w1.weight"),
            (f"vision_model.encoder.layers.{i}.mlp.fc1.bias",   f"encoder.layers.{i}.mlp.w1.bias"),
            (f"vision_model.encoder.layers.{i}.mlp.fc2.weight", f"encoder.layers.{i}.mlp.w2.weight"),
            (f"vision_model.encoder.layers.{i}.mlp.fc2.bias",   f"encoder.layers.{i}.mlp.w2.bias"),
        ]

        for hf_k, muse_k in layer_map:
            lookup_key = "siglip." + hf_k
            if lookup_key in hf_state_dict and muse_k in muse_state_dict:
                total_checked += 1
                hf_w = hf_state_dict[lookup_key].to(device=device, dtype=dtype)
                muse_w = muse_state_dict[muse_k]
                
                # Reshape handling if needed (e.g. Linear vs Conv)
                # But here everything should align if convert_hf_state_dict is correct
                if hf_w.shape != muse_w.shape:
                     if hf_w.transpose(0,1).shape == muse_w.shape: hf_w = hf_w.transpose(0,1)

                diff = (hf_w - muse_w).abs().max().item()
                if diff > 1e-5:
                    issues.append(f"Layer {i} {muse_k}: diff={diff:.6e}")
                else:
                    total_matched += 1

    # 3. Final Norm
    final_map = {
        "vision_model.post_layernorm.weight": "ln_post.weight",
        "vision_model.post_layernorm.bias": "ln_post.bias"
    }
    for hf_k, muse_k in final_map.items():
        lookup_key = "siglip." + hf_k
        if lookup_key in hf_state_dict and muse_k in muse_state_dict:
            total_checked += 1
            hf_w = hf_state_dict[lookup_key].to(device=device, dtype=dtype)
            muse_w = muse_state_dict[muse_k]
            diff = (hf_w - muse_w).abs().max().item()
            if diff > 1e-5:
                issues.append(f"{muse_k}: diff={diff:.6e}")
            else:
                total_matched += 1

    print(f"Total weights checked: {total_checked}")
    print(f"Weights matched: {total_matched}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} weight mismatches:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10: print(f"  ... {len(issues)-10} more")
    else:
        print("✓ All checked weights match!")


def test_keye_vision_align_with_hf_checkpoint():
    """Ensure Muse KeyeVisionTransformer aligns with the Origin implementation."""
    
    # === 1. Configuration ===
    checkpoint_path = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # Using BF16 as per your debugging requirement
    
    torch.manual_seed(0)
    
    print(f"Running alignment test on device={device}, dtype={dtype}")
    print(f"Checkpoint: {checkpoint_path}")

    # === 2. Load Origin (HF-style) Model ===
    print("\nLoading Origin Model...")
    # Use default config structure, populated with Muse defaults but HF class
    muse_dummy_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_dummy_config.dict())
    
    # Initialize empty Origin model
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config)
    
    # Load Weights manually from .pt
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "module" in raw_state_dict: raw_state_dict = raw_state_dict["module"]
    
    # Filter for vision model keys and format for Origin
    origin_load_dict = {}
    for k, v in raw_state_dict.items():
        # Key format in checkpoint: "vision_tower.siglip.vision_model.xxx" or similar
        # We need relative keys for Origin model which starts at "vision_model" or internal
        clean_k = k
        for prefix in ["module.", "vision_tower.", "siglip."]:
            if clean_k.startswith(prefix): clean_k = clean_k[len(prefix):]
        
        # Origin model's state_dict keys usually start after "siglip." if it's the full model,
        # but here we initialized `SiglipVisionModel`. Its keys start with `vision_model.`?
        # Let's inspect Origin model keys
        # Actually, SiglipVisionModel keys: "vision_model.embeddings...", "vision_model.encoder..."
        
        if "vision_model" not in clean_k: 
            clean_k = "vision_model." + clean_k
        
        # Map to what origin_model expects (remove "vision_model." prefix because OriginModel wraps it?
        # No, SiglipVisionModel usually has `self.vision_model`. 
        # Let's clean it to match `origin_model.state_dict()` keys.
        # Assuming origin_model.state_dict() has keys like "vision_model.embeddings..."
        origin_load_dict[clean_k] = v.to(dtype)

    # Load into Origin
    missing, unexpected = origin_model.load_state_dict(origin_load_dict, strict=False)
    # Ignore missing text model keys
    unexpected = [k for k in unexpected if "text_model" not in k]
    if len(unexpected) > 0:
        print(f"Origin Model Unexpected: {unexpected[:5]}...")

    origin_model.to(device)
    origin_model.eval()

    # === 3. Initialize Muse Model ===
    print("\nInitializing Muse Model...")
    muse_config = _build_muse_config(origin_config.to_dict())
    
    with set_default_dtype(dtype):
        muse_model = MuseKeyeVisionModel(muse_config)

    # === 4. Weight Conversion & Loading ===
    # We construct a "full" HF state dict to pass to converter
    hf_full_state_dict = {"siglip." + k: v for k, v in origin_load_dict.items()}
    
    print("Converting weights...")
    converted_state_dict = muse_model.convert_hf_state_dict(hf_full_state_dict)
    
    # Load into Muse
    m_missing, m_unexpected = muse_model.load_state_dict(converted_state_dict, strict=False)
    if m_missing: print(f"Muse Missing: {m_missing}")
    if m_unexpected: print(f"Muse Unexpected: {m_unexpected}")

    muse_model.to(device)
    muse_model.eval()

    # === 5. Weight Verification ===
    check_weights(hf_full_state_dict, muse_model, device, dtype)

    # === 6. Input Preparation ===
    print("\nPreparing Inputs...")
    processor = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    image = create_dummy_image(muse_config.image_size)
    
    processed = processor.preprocess(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    
    # Grid info
    grid_thw = processed["image_grid_thw"]
    if isinstance(grid_thw, torch.Tensor): grid_thw = grid_thw.tolist()
    image_grid_thw = [tuple(int(x) for x in g) for g in grid_thw]
    
    # Pack Inputs for Muse (5D tensor: [1, Seq, C, H, W])
    # Note: Origin likely needs this too based on previous debugs
    num_patches = [int(np.prod(g)) for g in image_grid_thw]
    seq_len = num_patches[0]
    
    # pixel_values from processor is [Seq, 3, 14, 14]
    # Muse expects [1, Seq, 3, 14, 14] for batch=1
    pixel_inputs = pixel_values.unsqueeze(0).to(device, dtype)
    
    # Position IDs & Cu Seqlens
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # === 7. Forward Pass & Comparison ===
    print(f"\n{'='*60}")
    print("Forward Pass & Hidden States Comparison")
    print(f"{'='*60}")

    with torch.no_grad():
        print("Running Origin Model Forward...")
        # Origin signature based on debug history
        origin_out = origin_model(
            pixel_inputs, 
            position_ids=position_ids, 
            image_grid_thw=image_grid_thw, 
            cu_seqlens=cu_seqlens, 
            interpolate_pos_encoding=True, 
            window_size=-1, 
            use_rope=True
        )
        # Extract tensor
        if hasattr(origin_out, "last_hidden_state"):
            hf_output = origin_out.last_hidden_state
        else:
            hf_output = origin_out
        
        # Handle list output
        if isinstance(hf_output, list):
            hf_output = torch.stack(hf_output, dim=0)

        print("Running Muse Model Forward...")
        muse_out_dict = muse_model(
            pixel_inputs, 
            position_ids=position_ids, 
            image_grid_thw=image_grid_thw, 
            cu_seqlens=cu_seqlens, 
            interpolate_pos_encoding=True, 
            has_learnable_position_embedding=True
        )
        muse_output = muse_out_dict["last_hidden_state"]

        # Ensure comparison on same device/dtype
        hf_output = hf_output.to(device, dtype)
        muse_output = muse_output.to(device, dtype)

        # === 8. Statistics ===
        print(f"\nOutput Shapes:")
        print(f"  HF:   {hf_output.shape}")
        print(f"  Muse: {muse_output.shape}")

        if hf_output.shape != muse_output.shape:
             print("⚠️ Shape Mismatch! Attempting squeeze...")
             if hf_output.dim() == 3 and hf_output.shape[0] == 1: hf_output = hf_output.squeeze(0)
             if muse_output.dim() == 3 and muse_output.shape[0] == 1: muse_output = muse_output.squeeze(0)

        diff = (hf_output - muse_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Relative diff
        rel_diff = diff / (hf_output.abs() + 1e-8)
        max_rel = rel_diff.max().item()

        print(f"\nDifference Statistics:")
        print(f"  Max Absolute Diff: {max_diff:.6e}")
        print(f"  Mean Absolute Diff: {mean_diff:.6e}")
        print(f"  Max Relative Diff: {max_rel:.6e}")

        # === 9. Pass/Fail Decision ===
        # BF16 tolerance: usually around 1e-2 is acceptable for complex attention models
        # FP32 tolerance: should be < 1e-5
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        
        print(f"\nTolerance Check (thresh={tol}):")
        if max_diff < tol:
            print("✓✓✓ SUCCESS: Outputs match within tolerance!")
        else:
            print("✗ FAILURE: Outputs differ beyond tolerance.")
            # Debug info
            idx = torch.argmax(diff)
            flat_hf = hf_output.flatten()
            flat_muse = muse_output.flatten()
            print(f"  Max diff index: {idx.item()}")
            print(f"  HF Value:   {flat_hf[idx].item()}")
            print(f"  Muse Value: {flat_muse[idx].item()}")

if __name__ == "__main__":
    test_keye_vision_align_with_hf_checkpoint()