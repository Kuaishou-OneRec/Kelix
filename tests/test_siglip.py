"""
Integration test to ensure Muse SigLIP matches Hugging Face logits.
"""

import os
from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel as HFSiglipVisionModel
from muse.config import SiglipVisionConfig

from muse.models.siglip import SiglipVisionTransformer as SiglipVisionModel
from muse.training.common import set_default_dtype



def _build_siglip_config(hf_cfg: Dict[str, Any]) -> SiglipVisionConfig:
    """Map Hugging Face config to Muse SiglipVisionConfig."""
    return SiglipVisionConfig(
        model_class="SiglipVisionModel",
        image_size=hf_cfg.get("image_size", 224),
        patch_size=hf_cfg.get("patch_size", 16),
        num_channels=hf_cfg.get("num_channels", 3),
        hidden_size=hf_cfg.get("hidden_size", 1152),
        num_hidden_layers=hf_cfg.get("num_hidden_layers", 24),
        num_attention_heads=hf_cfg.get("num_attention_heads", 16),
        intermediate_size=hf_cfg.get("intermediate_size", 4304),
        layer_norm_eps=hf_cfg.get("layer_norm_eps", 1e-5),
        attention_dropout=hf_cfg.get("attention_dropout", 0.0),
        has_learnable_position_embedding=hf_cfg.get("has_learnable_position_embedding", False),
        use_qk_norm=hf_cfg.get("use_qk_norm", False),
        qk_norm_eps=hf_cfg.get("qk_norm_eps", 1e-6),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        attention_function="eager",  # Use eager for comparison
        vision_use_head=hf_cfg.get("vision_use_head", True),
        spatial_merge_size=hf_cfg.get("spatial_merge_size", 2),
        output_attentions=False,
        output_hidden_states=False,
    )


def create_dummy_image(size: int = 224) -> Image.Image:
    """Create a dummy RGB image for testing."""
    np.random.seed(42)
    data = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)


def test_siglip_logits_align_with_hf_checkpoint():
    """Ensure Muse SigLIP outputs match the Hugging Face reference model."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Change this to your SigLIP checkpoint path
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"  # e.g., "google/siglip-base-patch16-224"

    # Load the processor and the model
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"
    )

    # Prepare the model input
    image = create_dummy_image(hf_model.config.image_size)
    inputs = processor(images=image, return_tensors="pt").to(hf_model.device)

    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()

    muse_config = _build_siglip_config(hf_config_dict)
    
    # Get target device and dtype from HF model
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    
    # Create Muse model with correct dtype
    model_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    with set_default_dtype(model_dtype):
        muse_model = SiglipVisionModel(muse_config)

    # For HF SiglipVisionModel, we need to add prefix for conversion
    # The HF state dict keys are like: embeddings.patch_embedding.weight
    # We need to add "siglip.vision_model." prefix for our converter
    prefixed_state_dict = {}
    for key, value in hf_state_dict.items():
        prefixed_state_dict[f"siglip.vision_model.{key}"] = value

    # Convert and load state dict
    state_dict = muse_model.convert_hf_state_dict(prefixed_state_dict)
    
    # Get Muse model's expected state dict keys
    muse_model_state_dict = muse_model.state_dict()
    muse_expected_keys = set(muse_model_state_dict.keys())
    converted_keys = set(state_dict.keys())
    
    # Check key matching
    print(f"\n{'='*60}")
    print("Weight Loading Check")
    print(f"{'='*60}")
    print(f"HF state dict keys: {len(hf_state_dict)}")
    print(f"Converted state dict keys: {len(state_dict)}")
    print(f"Muse model expected keys: {len(muse_expected_keys)}")
    
    # Find missing and extra keys
    missing_in_converted = muse_expected_keys - converted_keys
    extra_in_converted = converted_keys - muse_expected_keys
    
    if missing_in_converted:
        print(f"\n⚠️  Missing keys in converted state dict ({len(missing_in_converted)}):")
        for key in sorted(list(missing_in_converted))[:20]:
            print(f"  - {key}")
        if len(missing_in_converted) > 20:
            print(f"  ... and {len(missing_in_converted) - 20} more")
    
    if extra_in_converted:
        print(f"\n⚠️  Extra keys in converted state dict ({len(extra_in_converted)}):")
        for key in sorted(list(extra_in_converted))[:20]:
            print(f"  - {key}")
        if len(extra_in_converted) > 20:
            print(f"  ... and {len(extra_in_converted) - 20} more")
    
    # Convert state dict tensors to target dtype and device
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            state_dict[key] = tensor.to(device=device, dtype=dtype)
    
    # Handle missing keys
    missing_keys, unexpected_keys = muse_model.load_state_dict(
        state_dict, strict=False
    )
    
    if missing_keys:
        print(f"\n⚠️  Missing keys after load_state_dict ({len(missing_keys)}):")
        for key in missing_keys[:20]:
            print(f"  - {key}")
        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")
    else:
        print(f"\n✓ All expected keys loaded successfully")
    
    if unexpected_keys:
        print(f"\n⚠️  Unexpected keys after load_state_dict ({len(unexpected_keys)}):")
        for key in unexpected_keys[:20]:
            print(f"  - {key}")
        if len(unexpected_keys) > 20:
            print(f"  ... and {len(unexpected_keys) - 20} more")
    
    # Compare some key weights to verify correctness
    print(f"\n{'='*60}")
    print("Weight Value Comparison (Sample)")
    print(f"{'='*60}")
    
    # Check patch embedding layer
    if "embeddings.patch_embedding.weight" in state_dict:
        hf_embed_key = "embeddings.patch_embedding.weight"
        if hf_embed_key in hf_state_dict:
            hf_embed = hf_state_dict[hf_embed_key].to(device=device, dtype=dtype)
            muse_embed = state_dict["embeddings.patch_embedding.weight"]
            embed_diff = (hf_embed - muse_embed).abs()
            print(f"Patch embedding layer:")
            print(f"  Shape: HF={hf_embed.shape}, Muse={muse_embed.shape}")
            print(f"  Max diff: {embed_diff.max().item():.6e}")
            print(f"  Mean diff: {embed_diff.mean().item():.6e}")
            if embed_diff.max().item() > 1e-5:
                print(f"  ⚠️  Large difference detected!")
    
    # Check all transformer layers
    print(f"\nChecking all {muse_config.num_hidden_layers} transformer layers...")
    layer_issues = []
    total_checked = 0
    total_matched = 0
    
    for layer_idx in range(muse_config.num_hidden_layers):
        # Check attention weights
        attn_weights = [
            ("self_attn.q_proj.weight", "attn.q_proj.weight"),
            ("self_attn.k_proj.weight", "attn.k_proj.weight"),
            ("self_attn.v_proj.weight", "attn.v_proj.weight"),
            ("self_attn.out_proj.weight", "attn.output_proj.weight"),
            ("self_attn.q_proj.bias", "attn.q_proj.bias"),
            ("self_attn.k_proj.bias", "attn.k_proj.bias"),
            ("self_attn.v_proj.bias", "attn.v_proj.bias"),
            ("self_attn.out_proj.bias", "attn.output_proj.bias"),
        ]
        
        for hf_weight_name, muse_weight_name in attn_weights:
            hf_key = f"encoder.layers.{layer_idx}.{hf_weight_name}"
            muse_key = f"encoder.layers.{layer_idx}.{muse_weight_name}"
            if hf_key in hf_state_dict and muse_key in state_dict:
                total_checked += 1
                hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
                muse_weight = state_dict[muse_key]
                weight_diff = (hf_weight - muse_weight).abs()
                max_diff = weight_diff.max().item()
                if max_diff > 1e-5:
                    layer_issues.append(
                        f"Layer {layer_idx} {hf_weight_name}: "
                        f"max_diff={max_diff:.6e}"
                    )
                else:
                    total_matched += 1
        
        # Check MLP weights
        mlp_mapping = [
            ("mlp.fc1.weight", "mlp.fc1.weight"),
            ("mlp.fc1.bias", "mlp.fc1.bias"),
            ("mlp.fc2.weight", "mlp.fc2.weight"),
            ("mlp.fc2.bias", "mlp.fc2.bias"),
        ]
        for hf_weight_name, muse_weight_name in mlp_mapping:
            hf_key = f"encoder.layers.{layer_idx}.{hf_weight_name}"
            muse_key = f"encoder.layers.{layer_idx}.{muse_weight_name}"
            if hf_key in hf_state_dict and muse_key in state_dict:
                total_checked += 1
                hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
                muse_weight = state_dict[muse_key]
                weight_diff = (hf_weight - muse_weight).abs()
                max_diff = weight_diff.max().item()
                if max_diff > 1e-5:
                    layer_issues.append(
                        f"Layer {layer_idx} MLP {hf_weight_name}: "
                        f"max_diff={max_diff:.6e}"
                    )
                else:
                    total_matched += 1
        
        # Check layer norms
        layer_norms = [
            ("layer_norm1.weight", "sa_norm.weight"),
            ("layer_norm1.bias", "sa_norm.bias"),
            ("layer_norm2.weight", "mlp_norm.weight"),
            ("layer_norm2.bias", "mlp_norm.bias"),
        ]
        for hf_weight_name, muse_weight_name in layer_norms:
            hf_key = f"encoder.layers.{layer_idx}.{hf_weight_name}"
            muse_key = f"encoder.layers.{layer_idx}.{muse_weight_name}"
            if hf_key in hf_state_dict and muse_key in state_dict:
                total_checked += 1
                hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
                muse_weight = state_dict[muse_key]
                weight_diff = (hf_weight - muse_weight).abs()
                max_diff = weight_diff.max().item()
                if max_diff > 1e-5:
                    layer_issues.append(
                        f"Layer {layer_idx} {hf_weight_name}: "
                        f"max_diff={max_diff:.6e}"
                    )
                else:
                    total_matched += 1
    
    # Report layer issues
    print(f"\nWeight comparison summary:")
    print(f"  Total weights checked: {total_checked}")
    print(f"  Weights matched (diff < 1e-5): {total_matched}")
    print(f"  Weights with issues: {len(layer_issues)}")
    
    if layer_issues:
        print(f"\n⚠️  Found {len(layer_issues)} weight mismatches:")
        for issue in layer_issues[:50]:
            print(f"  - {issue}")
        if len(layer_issues) > 50:
            print(f"  ... and {len(layer_issues) - 50} more issues")
    else:
        print(f"✓ All transformer layer weights match!")
    
    # Check post layer norm
    hf_norm_key = "post_layernorm.weight"
    muse_norm_key = "ln_post.weight"
    if hf_norm_key in hf_state_dict and muse_norm_key in state_dict:
        hf_norm = hf_state_dict[hf_norm_key].to(device=device, dtype=dtype)
        muse_norm = state_dict[muse_norm_key]
        norm_diff = (hf_norm - muse_norm).abs()
        print(f"\nPost LayerNorm:")
        print(f"  Shape: HF={hf_norm.shape}, Muse={muse_norm.shape}")
        print(f"  Max diff: {norm_diff.max().item():.6e}")
        print(f"  Mean diff: {norm_diff.mean().item():.6e}")
        if norm_diff.max().item() > 1e-5:
            print(f"  ⚠️  Large difference detected!")
    
    print(f"{'='*60}\n")

    # Move Muse model to same device and dtype as HF model
    muse_model = muse_model.to(device=device, dtype=dtype)
    
    # Double-check that all parameters are in the correct dtype
    for name, param in muse_model.named_parameters():
        if param.dtype != dtype:
            print(f"Warning: Parameter {name} has dtype {param.dtype}, expected {dtype}")
            param.data = param.data.to(dtype=dtype)
    
    for name, buffer in muse_model.named_buffers():
        if buffer.dtype != dtype:
            print(f"Warning: Buffer {name} has dtype {buffer.dtype}, expected {dtype}")
            buffer.data = buffer.data.to(dtype=dtype)
    
    muse_model.eval()
    hf_model.eval()
    
    print(f"\n{'='*60}")
    print("Forward Pass & Output Comparison")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # HF forward
        print("Running HF model forward pass...")
        hf_outputs = hf_model(**inputs)
        hf_hidden_states = hf_outputs.last_hidden_state
        
        # Muse forward - prepare inputs
        # Muse model expects pixel_values and image_grid_thw
        print("Running Muse model forward pass...")
        pixel_values = inputs["pixel_values"]
        
        # For standard input, create image_grid_thw
        batch_size = pixel_values.shape[0]
        num_patches_per_side = muse_config.image_size // muse_config.patch_size
        image_grid_thw = [(1, num_patches_per_side, num_patches_per_side)] * batch_size
        
        muse_outputs = muse_model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        muse_hidden_states = muse_outputs["last_hidden_state"]
        
        # Ensure both outputs are on same device and dtype for comparison
        hf_hidden_states = hf_hidden_states.to(device=device, dtype=dtype)
        muse_hidden_states = muse_hidden_states.to(device=device, dtype=dtype)
        
        # Compare shapes
        print(f"\nOutput shapes:")
        print(f"  HF:   {hf_hidden_states.shape}")
        print(f"  Muse: {muse_hidden_states.shape}")
        
        if hf_hidden_states.shape != muse_hidden_states.shape:
            print(f"  ⚠️  Shape mismatch!")
            # Try to align shapes if possible
            min_seq_len = min(hf_hidden_states.shape[1], muse_hidden_states.shape[1])
            hf_hidden_states = hf_hidden_states[:, :min_seq_len, :]
            muse_hidden_states = muse_hidden_states[:, :min_seq_len, :]
            print(f"  Using aligned shapes: HF={hf_hidden_states.shape}, Muse={muse_hidden_states.shape}")
        
        # Check for NaN/Inf
        hf_has_nan = torch.isnan(hf_hidden_states).any().item()
        hf_has_inf = torch.isinf(hf_hidden_states).any().item()
        muse_has_nan = torch.isnan(muse_hidden_states).any().item()
        muse_has_inf = torch.isinf(muse_hidden_states).any().item()
        
        print(f"\nNaN/Inf checks:")
        print(f"  HF:   NaN={hf_has_nan}, Inf={hf_has_inf}")
        print(f"  Muse: NaN={muse_has_nan}, Inf={muse_has_inf}")
        
        if hf_has_nan or hf_has_inf:
            print(f"  ⚠️  HF outputs contain NaN/Inf!")
        if muse_has_nan or muse_has_inf:
            print(f"  ⚠️  Muse outputs contain NaN/Inf!")
        
        # Calculate differences
        output_diff = (hf_hidden_states - muse_hidden_states).abs()
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()
        median_diff = output_diff.median().item()
        
        # Calculate relative differences
        hf_abs = hf_hidden_states.abs()
        relative_diff = output_diff / (hf_abs + 1e-8)
        max_relative_diff = relative_diff.max().item()
        mean_relative_diff = relative_diff.mean().item()
        
        # Calculate per-token statistics
        per_token_max_diff = output_diff.max(dim=-1)[0]  # [b, s]
        per_token_mean_diff = output_diff.mean(dim=-1)   # [b, s]
        
        print(f"\nOutput difference statistics:")
        print(f"  Max absolute diff:     {max_diff:.6e}")
        print(f"  Mean absolute diff:    {mean_diff:.6e}")
        print(f"  Median absolute diff:  {median_diff:.6e}")
        print(f"  Max relative diff:     {max_relative_diff:.6e}")
        print(f"  Mean relative diff:    {mean_relative_diff:.6e}")
        
        print(f"\nPer-token statistics:")
        print(f"  Max diff per token - Max: {per_token_max_diff.max().item():.6e}")
        print(f"  Max diff per token - Mean: {per_token_max_diff.mean().item():.6e}")
        print(f"  Mean diff per token - Max: {per_token_mean_diff.max().item():.6e}")
        print(f"  Mean diff per token - Mean: {per_token_mean_diff.mean().item():.6e}")
        
        # Check if differences are within acceptable tolerance
        if dtype == torch.bfloat16:
            tolerance = 1e-2  # More lenient for bfloat16
        else:
            tolerance = 1e-5  # Stricter for float32
        
        print(f"\nTolerance check (tolerance={tolerance:.0e}):")
        within_tolerance = max_diff < tolerance
        print(f"  Max diff within tolerance: {within_tolerance}")
        
        if not within_tolerance:
            print(f"  ⚠️  Outputs differ beyond tolerance!")
            
            # Find positions with largest differences
            flat_diff = output_diff.flatten()
            top_k = min(10, len(flat_diff))
            top_k_values, top_k_indices = torch.topk(flat_diff, top_k)
            
            print(f"\n  Top {top_k} largest differences:")
            for i, (val, idx) in enumerate(zip(top_k_values, top_k_indices)):
                batch_idx = idx // (muse_hidden_states.shape[1] * muse_hidden_states.shape[2])
                remainder = idx % (muse_hidden_states.shape[1] * muse_hidden_states.shape[2])
                seq_idx = remainder // muse_hidden_states.shape[2]
                hidden_idx = remainder % muse_hidden_states.shape[2]
                
                hf_val = hf_hidden_states[batch_idx, seq_idx, hidden_idx].item()
                muse_val = muse_hidden_states[batch_idx, seq_idx, hidden_idx].item()
                
                print(f"    [{i+1}] pos=({batch_idx}, {seq_idx}, {hidden_idx}), "
                      f"diff={val.item():.6e}, HF={hf_val:.6e}, Muse={muse_val:.6e}")
        else:
            print(f"  ✓ Outputs match within tolerance!")
        
        # Compare output value ranges
        print(f"\nOutput value ranges:")
        print(f"  HF:   min={hf_hidden_states.min().item():.4f}, max={hf_hidden_states.max().item():.4f}, "
              f"mean={hf_hidden_states.mean().item():.4f}, std={hf_hidden_states.std().item():.4f}")
        print(f"  Muse: min={muse_hidden_states.min().item():.4f}, max={muse_hidden_states.max().item():.4f}, "
              f"mean={muse_hidden_states.mean().item():.4f}, std={muse_hidden_states.std().item():.4f}")
        
        print(f"\n{'='*60}\n")
        
        # Summary
        if within_tolerance:
            print("✓✓✓ SUCCESS: Outputs match within tolerance!")
        else:
            print("✗ FAILURE: Outputs differ beyond tolerance")


def test_siglip_from_checkpoint():
    """Test loading Muse SigLIP from a converted checkpoint."""
    hf_checkpoint_dir = "/path/to/siglip/hf/model"
    checkpoint_dir = "/path/to/siglip/muse/model"
    
    with set_default_dtype(torch.bfloat16):
        model = SiglipVisionModel.from_pretrained(checkpoint_dir)
    
    # Load the processor and the HF model
    processor = AutoProcessor.from_pretrained(hf_checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(
        hf_checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Prepare the model input
    image = create_dummy_image(hf_model.config.image_size)
    inputs = processor(images=image, return_tensors="pt").to(hf_model.device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)

    with torch.no_grad():
        # HF forward
        print("Running HF model forward pass...")
        hf_outputs = hf_model(**inputs)
        hf_hidden_states = hf_outputs.last_hidden_state
        
        # Muse forward
        print("Running Muse model forward pass...")
        pixel_values = inputs["pixel_values"].to(device)
        
        batch_size = pixel_values.shape[0]
        num_patches_per_side = model.config.image_size // model.config.patch_size
        image_grid_thw = [(1, num_patches_per_side, num_patches_per_side)] * batch_size
        
        muse_outputs = model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        muse_hidden_states = muse_outputs["last_hidden_state"]

        # Calculate differences
        output_diff = (hf_hidden_states.to(device) - muse_hidden_states).abs()
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()
        median_diff = output_diff.median().item()
        
        # Calculate relative differences
        hf_abs = hf_hidden_states.abs()
        relative_diff = output_diff / (hf_abs.to(device) + 1e-8)
        max_relative_diff = relative_diff.max().item()
        mean_relative_diff = relative_diff.mean().item()

        print(f"Max diff: {max_diff:.6e}")
        print(f"Mean diff: {mean_diff:.6e}")
        print(f"Median diff: {median_diff:.6e}")
        print(f"Max relative diff: {max_relative_diff:.6e}")
        print(f"Mean relative diff: {mean_relative_diff:.6e}")

        print(f"{'='*60}\n")

        # Summary
        if max_diff < 1e-2:  # Use 1e-2 for bfloat16
            print("✓✓✓ SUCCESS: Outputs match within tolerance!")
        else:
            print("✗ FAILURE: Outputs differ beyond tolerance")


def test_siglip_weight_conversion():
    """Test that weight conversion produces correct key mappings."""
    # This test verifies the convert_hf_state_dict function without needing a real checkpoint
    
    print(f"\n{'='*60}")
    print("Testing Weight Conversion Logic")
    print(f"{'='*60}")
    
    # Create a minimal config
    config = SiglipVisionConfig(
        model_class="SiglipVisionModel",
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    
    # Create model
    with set_default_dtype(torch.float32):
        model = SiglipVisionModel(config)
    
    # Create a fake HF state dict with expected key patterns
    fake_hf_state_dict = {}
    
    # Embeddings
    fake_hf_state_dict["siglip.vision_model.embeddings.patch_embedding.weight"] = torch.randn(768, 3, 16, 16)
    fake_hf_state_dict["siglip.vision_model.embeddings.patch_embedding.bias"] = torch.randn(768)
    fake_hf_state_dict["siglip.vision_model.embeddings.position_embedding.weight"] = torch.randn(196, 768)
    fake_hf_state_dict["siglip.vision_model.embeddings.packing_position_embedding.weight"] = torch.randn(32768, 768)
    
    # Encoder layers
    for layer_idx in range(2):
        # Layer norm 1 -> sa_norm
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm1.weight"] = torch.randn(768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm1.bias"] = torch.randn(768)
        
        # Self attention
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.randn(768, 768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias"] = torch.randn(768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.randn(768, 768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias"] = torch.randn(768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.randn(768, 768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias"] = torch.randn(768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight"] = torch.randn(768, 768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias"] = torch.randn(768)
        
        # Layer norm 2 -> mlp_norm
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm2.weight"] = torch.randn(768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm2.bias"] = torch.randn(768)
        
        # MLP
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc1.weight"] = torch.randn(3072, 768)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc1.bias"] = torch.randn(3072)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc2.weight"] = torch.randn(768, 3072)
        fake_hf_state_dict[f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc2.bias"] = torch.randn(768)
    
    # Post layer norm -> ln_post
    fake_hf_state_dict["siglip.vision_model.post_layernorm.weight"] = torch.randn(768)
    fake_hf_state_dict["siglip.vision_model.post_layernorm.bias"] = torch.randn(768)
    
    # Add some keys that should be skipped
    fake_hf_state_dict["siglip.logit_scale"] = torch.tensor(1.0)
    fake_hf_state_dict["siglip.logit_bias"] = torch.randn(100)
    fake_hf_state_dict["siglip.text_model.embeddings.token_embedding.weight"] = torch.randn(1000, 768)
    fake_hf_state_dict["siglip.vision_model.head.probe"] = torch.randn(768)
    
    # Convert
    converted = model.convert_hf_state_dict(fake_hf_state_dict)
    
    # Verify key mappings
    expected_mappings = {
        # Embeddings
        "embeddings.patch_embedding.weight": "siglip.vision_model.embeddings.patch_embedding.weight",
        "embeddings.patch_embedding.bias": "siglip.vision_model.embeddings.patch_embedding.bias",
        "embeddings.position_embedding.weight": "siglip.vision_model.embeddings.position_embedding.weight",
        "embeddings.packing_position_embedding.weight": "siglip.vision_model.embeddings.packing_position_embedding.weight",
        # Post layer norm
        "ln_post.weight": "siglip.vision_model.post_layernorm.weight",
        "ln_post.bias": "siglip.vision_model.post_layernorm.bias",
    }
    
    # Add layer mappings
    for layer_idx in range(2):
        layer_mappings = {
            f"encoder.layers.{layer_idx}.sa_norm.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm1.weight",
            f"encoder.layers.{layer_idx}.sa_norm.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm1.bias",
            f"encoder.layers.{layer_idx}.attn.q_proj.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight",
            f"encoder.layers.{layer_idx}.attn.q_proj.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias",
            f"encoder.layers.{layer_idx}.attn.k_proj.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight",
            f"encoder.layers.{layer_idx}.attn.k_proj.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias",
            f"encoder.layers.{layer_idx}.attn.v_proj.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight",
            f"encoder.layers.{layer_idx}.attn.v_proj.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias",
            f"encoder.layers.{layer_idx}.attn.output_proj.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight",
            f"encoder.layers.{layer_idx}.attn.output_proj.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias",
            f"encoder.layers.{layer_idx}.mlp_norm.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm2.weight",
            f"encoder.layers.{layer_idx}.mlp_norm.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.layer_norm2.bias",
            f"encoder.layers.{layer_idx}.mlp.fc1.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc1.weight",
            f"encoder.layers.{layer_idx}.mlp.fc1.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc1.bias",
            f"encoder.layers.{layer_idx}.mlp.fc2.weight": f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc2.weight",
            f"encoder.layers.{layer_idx}.mlp.fc2.bias": f"siglip.vision_model.encoder.layers.{layer_idx}.mlp.fc2.bias",
        }
        expected_mappings.update(layer_mappings)
    
    # Check all expected keys are present
    all_passed = True
    for muse_key, hf_key in expected_mappings.items():
        if muse_key not in converted:
            print(f"  ✗ Missing key: {muse_key} (from {hf_key})")
            all_passed = False
        else:
            # Verify tensor is the same
            if not torch.equal(converted[muse_key], fake_hf_state_dict[hf_key]):
                print(f"  ✗ Tensor mismatch: {muse_key}")
                all_passed = False
    
    # Check that skipped keys are not in converted
    skipped_keys = [
        "siglip.logit_scale",
        "siglip.logit_bias", 
        "siglip.text_model.embeddings.token_embedding.weight",
        "siglip.vision_model.head.probe",
    ]
    for key in skipped_keys:
        # These should not appear in any form in converted
        for converted_key in converted.keys():
            if "logit" in converted_key or "text_model" in converted_key or "head.probe" in converted_key:
                print(f"  ✗ Key should have been skipped: {converted_key}")
                all_passed = False
    
    print(f"\nExpected {len(expected_mappings)} keys, got {len(converted)} keys")
    
    if all_passed and len(converted) == len(expected_mappings):
        print("✓✓✓ All weight conversion tests passed!")
    else:
        print("✗ Some weight conversion tests failed")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "convert":
            test_siglip_weight_conversion()
        elif sys.argv[1] == "checkpoint":
            test_siglip_from_checkpoint()
        else:
            test_siglip_logits_align_with_hf_checkpoint()
    else:
        # Default: run weight conversion test (doesn't need checkpoint)
        test_siglip_weight_conversion()

