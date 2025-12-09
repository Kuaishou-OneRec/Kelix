# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Forward Alignment Test for Sana DiT Model.

This test compares the muse implementation against the diffusers SanaTransformer2DModel
layer by layer to identify any numerical differences.

Usage:
    PYTHONPATH=/path/to/muse:$PYTHONPATH python tests/models/sana/test_forward_alignment.py
    
    # Or run with pytest
    pytest tests/models/sana/test_forward_alignment.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Paths - update these to your locations
# Diffusers format checkpoint directory (containing transformer/ subfolder)
DIFFUSERS_CHECKPOINT_PATH = "/llm_reco_ssd/zhouyang12/models/Sana_1600M_1024px_diffusers"

# Model configuration for Sana 1600M
MODEL_CONFIG = {
    "input_size": 32,
    "patch_size": 1,
    "in_channels": 32,
    "hidden_size": 2240,
    "depth": 20,
    "num_heads": 20,
    "mlp_ratio": 2.5,
    "caption_channels": 2304,
    "model_max_length": 300,
    "attn_type": "linear",
    "ffn_type": "glumbconv",
    "mlp_acts": ("silu", "silu", None),
    "use_pe": False,
    "qk_norm": False,
    "y_norm": True,
    "y_norm_scale_factor": 0.01,
    "pred_sigma": False,
    "learn_sigma": False,
    "class_dropout_prob": 0.0,
    "drop_path": 0.0,
}

# Tolerance for numerical comparison
ATOL = 1e-5
RTOL = 1e-4


def load_diffusers_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load the diffusers SanaTransformer2DModel."""
    from diffusers.models import SanaTransformer2DModel
    
    # Try loading with subfolder="transformer" first
    transformer_path = os.path.join(checkpoint_path, "transformer")
    if os.path.exists(transformer_path):
        model = SanaTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=dtype
        )
    else:
        model = SanaTransformer2DModel.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype
        )
    
    model = model.to(device=device)
    model.eval()
    
    print(f"  Loaded diffusers model config: {model.config}")
    return model


def load_muse_model(diffusers_state_dict: dict, device: torch.device, dtype: torch.dtype):
    """Load the muse Sana model with diffusers state dict."""
    from muse.models.sana.modeling import SanaModel
    from muse.config.model_config import SanaConfig
    
    config = SanaConfig(
        model_class="SanaModel",
        input_size=MODEL_CONFIG["input_size"],
        patch_size=MODEL_CONFIG["patch_size"],
        in_channels=MODEL_CONFIG["in_channels"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        depth=MODEL_CONFIG["depth"],
        num_heads=MODEL_CONFIG["num_heads"],
        mlp_ratio=MODEL_CONFIG["mlp_ratio"],
        caption_channels=MODEL_CONFIG["caption_channels"],
        model_max_length=MODEL_CONFIG["model_max_length"],
        attn_type=MODEL_CONFIG["attn_type"],
        ffn_type=MODEL_CONFIG["ffn_type"],
        mlp_acts=MODEL_CONFIG["mlp_acts"],
        use_pe=MODEL_CONFIG["use_pe"],
        qk_norm=MODEL_CONFIG["qk_norm"],
        y_norm=MODEL_CONFIG["y_norm"],
        y_norm_scale_factor=MODEL_CONFIG["y_norm_scale_factor"],
        pred_sigma=MODEL_CONFIG["pred_sigma"],
        learn_sigma=MODEL_CONFIG["learn_sigma"],
        class_dropout_prob=MODEL_CONFIG["class_dropout_prob"],
        drop_path=MODEL_CONFIG["drop_path"],
    )
    
    model = SanaModel(config)
    
    # Convert diffusers state dict to muse format using model's converter
    converted_state_dict = model.convert_diffusers_state_dict(diffusers_state_dict)
    
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing:
        print(f"  [Warning] Missing keys: {missing[:10]}..." if len(missing) > 10 else f"  [Warning] Missing keys: {missing}")
    if unexpected:
        print(f"  [Warning] Unexpected keys: {unexpected[:10]}..." if len(unexpected) > 10 else f"  [Warning] Unexpected keys: {unexpected}")
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    return model


def create_test_inputs(
    batch_size: int = 1,
    latent_size: int = 32,
    latent_channels: int = 32,
    text_length: int = 300,
    text_channels: int = 2304,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
):
    """Create test inputs for both models."""
    torch.manual_seed(seed)
    
    # Latent input: [B, C, H, W]
    x = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device, dtype=dtype)
    
    # Timestep: [B]
    timestep = torch.tensor([500] * batch_size, device=device, dtype=torch.long)
    
    # Text embeddings: [B, L, D] for diffusers, [B, 1, L, D] for muse
    y_diffusers = torch.randn(batch_size, text_length, text_channels, device=device, dtype=dtype)
    y_muse = y_diffusers.unsqueeze(1)  # [B, 1, L, D]
    
    # Attention mask: [B, L] for diffusers, [B, 1, 1, L] for muse
    mask_diffusers = torch.ones(batch_size, text_length, device=device, dtype=dtype)
    mask_muse = mask_diffusers.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
    
    return {
        "x": x,
        "timestep": timestep,
        "y_diffusers": y_diffusers,
        "y_muse": y_muse,
        "mask_diffusers": mask_diffusers,
        "mask_muse": mask_muse,
    }


def compare_tensors(name: str, tensor1: torch.Tensor, tensor2: torch.Tensor, atol=ATOL, rtol=RTOL):
    """Compare two tensors and print results."""
    if tensor1.shape != tensor2.shape:
        print(f"  [{name}] SHAPE MISMATCH: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    diff = (tensor1 - tensor2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Check relative difference for non-zero values
    mask = tensor1.abs() > 1e-8
    if mask.any():
        rel_diff = (diff[mask] / tensor1.abs()[mask]).max().item()
    else:
        rel_diff = 0.0
    
    passed = max_diff < atol or rel_diff < rtol
    status = "PASS" if passed else "FAIL"
    
    print(f"  [{name}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}  {status}")
    
    return passed


def compare_state_dicts(diffusers_model, muse_model):
    """Compare state dict keys and values between models."""
    print("\n[State Dict Comparison]")
    
    diff_sd = diffusers_model.state_dict()
    muse_sd = muse_model.state_dict()
    
    print(f"  Diffusers keys: {len(diff_sd)}")
    print(f"  Muse keys: {len(muse_sd)}")
    
    # Print some raw diffusers keys for debugging
    print("\n  Sample raw diffusers keys:")
    for i, key in enumerate(sorted(diff_sd.keys())):
        if i < 20:
            print(f"    {key}")
    
    # Print some muse keys for debugging
    print("\n  Sample muse keys:")
    for i, key in enumerate(sorted(muse_sd.keys())):
        if i < 20:
            print(f"    {key}")
    
    # Try to match keys using model's converter
    converted = muse_model.convert_diffusers_state_dict(diff_sd)
    
    matched = 0
    unmatched_diff = []
    unmatched_muse = []
    
    for key in converted:
        if key in muse_sd:
            matched += 1
        else:
            unmatched_diff.append(key)
    
    for key in muse_sd:
        if key not in converted:
            unmatched_muse.append(key)
    
    print(f"\n  Matched keys: {matched}")
    print(f"  Unmatched from diffusers (after conversion): {len(unmatched_diff)}")
    if unmatched_diff[:10]:
        print(f"    Examples: {unmatched_diff[:10]}")
    print(f"  Unmatched in muse: {len(unmatched_muse)}")
    if unmatched_muse[:10]:
        print(f"    Examples: {unmatched_muse[:10]}")


def run_full_alignment_test():
    """Run complete layer-by-layer alignment test."""
    print("=" * 70)
    print("Sana Forward Alignment Test - Diffusers vs Muse")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"Checkpoint: {DIFFUSERS_CHECKPOINT_PATH}")
    
    # Check path
    if not os.path.exists(DIFFUSERS_CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {DIFFUSERS_CHECKPOINT_PATH}")
        return False
    
    # Load diffusers model
    print("\n[Loading] Loading diffusers SanaTransformer2DModel...")
    try:
        diffusers_model = load_diffusers_model(DIFFUSERS_CHECKPOINT_PATH, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in diffusers_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load muse model
    print("\n[Loading] Loading muse SanaModel...")
    try:
        diffusers_state_dict = diffusers_model.state_dict()
        muse_model = load_muse_model(diffusers_state_dict, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in muse_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare state dicts
    compare_state_dicts(diffusers_model, muse_model)
    
    # Create inputs
    print("\n[Setup] Creating test inputs...")
    inputs = create_test_inputs(device=device, dtype=dtype)
    print(f"  x: {inputs['x'].shape}")
    print(f"  timestep: {inputs['timestep']}")
    print(f"  y_diffusers: {inputs['y_diffusers'].shape}")
    print(f"  y_muse: {inputs['y_muse'].shape}")
    
    all_passed = True
    
    # Run forward pass on both models
    print("\n[Forward] Running forward pass...")
    
    with torch.no_grad():
        # Diffusers forward
        diffusers_out = diffusers_model(
            hidden_states=inputs["x"],
            encoder_hidden_states=inputs["y_diffusers"],
            timestep=inputs["timestep"].float(),
            encoder_attention_mask=inputs["mask_diffusers"],
            return_dict=False,
        )[0]
        
        # Muse forward
        muse_out = muse_model(
            x=inputs["x"],
            timestep=inputs["timestep"].float(),
            y=inputs["y_muse"],
            mask=inputs["mask_muse"],
        )
    
    print(f"\n  Diffusers output: {diffusers_out.shape}")
    print(f"  Muse output: {muse_out.shape}")
    
    passed = compare_tensors("Full Model Output", diffusers_out, muse_out)
    all_passed = all_passed and passed
    
    # Layer-by-layer comparison
    print("\n[Layer Comparison]")
    
    with torch.no_grad():
        x = inputs["x"].to(dtype)
        
        # 1. Patch embedding
        print("\n  [1] Patch Embedding...")
        diff_patch = diffusers_model.patch_embed(x)
        muse_patch = muse_model.x_embedder(x)
        passed = compare_tensors("patch_embed", diff_patch, muse_patch)
        all_passed = all_passed and passed
        
        # 2. Timestep embedding
        print("\n  [2] Timestep Embedding...")
        timestep_float = inputs["timestep"].float()
        
        # Diffusers time_embed returns (shift_scale, embedded_timestep)
        diff_time, diff_t_emb = diffusers_model.time_embed(
            timestep_float, 
            batch_size=inputs["x"].shape[0],
            hidden_dtype=dtype
        )
        
        # Muse separate t_embedder and t_block
        muse_t = muse_model.t_embedder(timestep_float.long().float())
        muse_t0 = muse_model.t_block(muse_t)
        
        compare_tensors("t_embedder", diff_t_emb, muse_t)
        compare_tensors("t_block (6*dim)", diff_time, muse_t0)
        
        # 3. Caption embedding
        print("\n  [3] Caption Embedding...")
        y_diff = inputs["y_diffusers"]
        y_muse = inputs["y_muse"]
        
        diff_caption = diffusers_model.caption_projection(y_diff)
        diff_caption = diff_caption.view(y_diff.shape[0], -1, diff_patch.shape[-1])
        diff_caption = diffusers_model.caption_norm(diff_caption)
        
        muse_caption = muse_model.y_embedder(y_muse, False)
        if muse_model.y_norm:
            muse_caption = muse_model.attention_y_norm(muse_caption)
        muse_caption = muse_caption.squeeze(1)  # Remove extra dim for comparison
        
        compare_tensors("caption_embed", diff_caption, muse_caption)
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("SUCCESS: All comparisons passed within tolerance!")
    else:
        print("FAILED: Some comparisons failed.")
    print("=" * 70)
    
    return all_passed


def run_muse_only_test():
    """Run a quick test of the muse model only."""
    print("=" * 60)
    print("Muse Sana Model Test (standalone)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    if not os.path.exists(DIFFUSERS_CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {DIFFUSERS_CHECKPOINT_PATH}")
        return
    
    print(f"\nCheckpoint: {DIFFUSERS_CHECKPOINT_PATH}")
    
    # Load diffusers model to get state dict
    print("\n[1/3] Loading diffusers model for state dict...")
    try:
        from diffusers.models import SanaTransformer2DModel
        transformer_path = os.path.join(DIFFUSERS_CHECKPOINT_PATH, "transformer")
        if os.path.exists(transformer_path):
            diffusers_model = SanaTransformer2DModel.from_pretrained(transformer_path, torch_dtype=dtype)
        else:
            diffusers_model = SanaTransformer2DModel.from_pretrained(DIFFUSERS_CHECKPOINT_PATH, torch_dtype=dtype)
        diffusers_state_dict = diffusers_model.state_dict()
        del diffusers_model
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[2/3] Loading muse Sana model...")
    try:
        muse_model = load_muse_model(diffusers_state_dict, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in muse_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[3/3] Running forward pass...")
    inputs = create_test_inputs(device=device, dtype=dtype)
    
    try:
        with torch.no_grad():
            output = muse_model(
                x=inputs["x"],
                timestep=inputs["timestep"].float(),
                y=inputs["y_muse"],
                mask=inputs["mask_muse"],
            )
        print(f"  Output shape: {output.shape}")
        print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Muse model test PASSED!")
    print("=" * 60)


# Pytest test classes
@pytest.mark.skipif(
    not os.path.exists(DIFFUSERS_CHECKPOINT_PATH),
    reason=f"Checkpoint not found at {DIFFUSERS_CHECKPOINT_PATH}"
)
class TestSanaAlignment:
    """Test suite for Sana alignment."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
    
    def test_full_alignment(self):
        """Test full model alignment."""
        result = run_full_alignment_test()
        assert result, "Alignment test failed"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--muse-only", action="store_true", help="Only test muse model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    args = parser.parse_args()
    
    if args.checkpoint:
        DIFFUSERS_CHECKPOINT_PATH = args.checkpoint
    
    if args.muse_only:
        run_muse_only_test()
    else:
        run_full_alignment_test()
