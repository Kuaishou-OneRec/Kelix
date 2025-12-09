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

This test compares the muse implementation against the official Sana implementation
layer by layer to identify any numerical differences.

Usage:
    # Full alignment test (requires Sana repo in PYTHONPATH)
    PYTHONPATH=/Users/zhouyang/code/Sana:/Users/zhouyang/code/muse:$PYTHONPATH \
        python tests/models/sana/test_forward_alignment.py
    
    # Or run with pytest
    pytest tests/models/sana/test_forward_alignment.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Paths - update these to your locations
CHECKPOINT_PATH = "/llm_reco_ssd/zhouyang12/models/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"
SANA_REPO_PATH = "/llm_reco_ssd/zhouyang12/code/dev/Sana"

# Model configuration for SanaMS_1600M_P1_D20 (from Sana_1600M_img1024.yaml)
MODEL_CONFIG = {
    "input_size": 32,  # For 1024px image with 32x downsample
    "patch_size": 1,
    "in_channels": 32,  # DC-AE latent channels
    "hidden_size": 2240,
    "depth": 20,
    "num_heads": 20,
    "mlp_ratio": 2.5,
    "caption_channels": 2304,  # gemma-2-2b-it hidden size
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
    "class_dropout_prob": 0.0,  # Disable dropout for deterministic test
    "drop_path": 0.0,
}

# Tolerance for numerical comparison
ATOL = 1e-5
RTOL = 1e-4


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and handle git-lfs pointers."""
    # Check if checkpoint is a git-lfs pointer file
    with open(checkpoint_path, 'rb') as f:
        header = f.read(50)
    if b'git-lfs' in header:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is a git-lfs pointer file.\n"
            f"Please run 'git lfs pull' in the model directory to download the actual weights."
        )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    
    return new_state_dict


def load_official_model(state_dict: dict, device: torch.device, dtype: torch.dtype):
    """Load the official Sana model from the Sana repo."""
    try:
        from diffusion.model.nets.sana_multi_scale import SanaMS_1600M_P1_D20
    except ImportError:
        raise ImportError(
            "Cannot import official Sana model. Make sure:\n"
            "1. Sana repo is in PYTHONPATH\n"
            "2. All Sana dependencies are installed\n"
            f"   PYTHONPATH={SANA_REPO_PATH}:$PYTHONPATH"
        )
    
    # Create model with matching config
    model = SanaMS_1600M_P1_D20(
        input_size=MODEL_CONFIG["input_size"],
        in_channels=MODEL_CONFIG["in_channels"],
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
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    return model


def load_muse_model(state_dict: dict, device: torch.device, dtype: torch.dtype):
    """Load the muse Sana model."""
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
    converted_state_dict = model.convert_sana_state_dict(state_dict)
    
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing:
        print(f"  [Warning] Missing keys: {missing}")
    if unexpected:
        print(f"  [Warning] Unexpected keys: {unexpected}")
    
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
    """Create test inputs for the model."""
    torch.manual_seed(seed)
    
    x = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device, dtype=dtype)
    timestep = torch.tensor([500.0] * batch_size, device=device, dtype=torch.float32)
    y = torch.randn(batch_size, 1, text_length, text_channels, device=device, dtype=dtype)
    mask = torch.ones(batch_size, 1, 1, text_length, device=device, dtype=dtype)
    
    return x, timestep, y, mask


def compare_tensors(name: str, official: torch.Tensor, muse: torch.Tensor, atol=ATOL, rtol=RTOL):
    """Compare two tensors and print results."""
    if official.shape != muse.shape:
        print(f"  [{name}] SHAPE MISMATCH: official={official.shape}, muse={muse.shape}")
        return False
    
    diff = (official - muse).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Check relative difference for non-zero values
    mask = official.abs() > 1e-8
    if mask.any():
        rel_diff = (diff[mask] / official.abs()[mask]).max().item()
    else:
        rel_diff = 0.0
    
    passed = max_diff < atol or rel_diff < rtol
    status = "PASS" if passed else "FAIL"
    
    print(f"  [{name}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}  {status}")
    
    return passed


def compare_patch_embed(official_model, muse_model, x, dtype):
    """Compare patch embedding outputs."""
    print("\n[1] Comparing Patch Embedding (x_embedder)...")
    
    x_off = x.to(dtype)
    x_muse = x.to(dtype)
    
    with torch.no_grad():
        off_out = official_model.x_embedder(x_off)
        muse_out = muse_model.x_embedder(x_muse)
    
    return compare_tensors("x_embedder", off_out, muse_out)


def compare_timestep_embed(official_model, muse_model, timestep):
    """Compare timestep embedding outputs."""
    print("\n[2] Comparing Timestep Embedding (t_embedder + t_block)...")
    
    # Official uses timestep.long().to(torch.float32)
    t_off = timestep.long().to(torch.float32)
    t_muse = timestep.long().to(torch.float32)
    
    with torch.no_grad():
        # t_embedder
        off_t = official_model.t_embedder(t_off)
        muse_t = muse_model.t_embedder(t_muse)
        passed1 = compare_tensors("t_embedder", off_t, muse_t)
        
        # t_block
        off_t0 = official_model.t_block(off_t)
        muse_t0 = muse_model.t_block(muse_t)
        passed2 = compare_tensors("t_block", off_t0, muse_t0)
    
    return passed1 and passed2


def compare_caption_embed(official_model, muse_model, y, mask, dtype):
    """Compare caption embedding outputs."""
    print("\n[3] Comparing Caption Embedding (y_embedder)...")
    
    y_off = y.to(dtype)
    y_muse = y.to(dtype)
    
    with torch.no_grad():
        # y_embedder (set training=False for deterministic behavior)
        official_model.eval()
        muse_model.eval()
        
        off_y = official_model.y_embedder(y_off, False, mask=mask)
        muse_y = muse_model.y_embedder(y_muse, False, mask=mask)
        passed1 = compare_tensors("y_embedder", off_y, muse_y)
        
        # y_norm (attention_y_norm) if enabled
        if MODEL_CONFIG["y_norm"]:
            off_y_norm = official_model.attention_y_norm(off_y)
            muse_y_norm = muse_model.attention_y_norm(muse_y)
            passed2 = compare_tensors("attention_y_norm", off_y_norm, muse_y_norm)
        else:
            passed2 = True
    
    return passed1 and passed2


def compare_single_block(block_idx, official_block, muse_block, x, y, t0, y_lens, HW):
    """Compare a single transformer block."""
    with torch.no_grad():
        off_out = official_block(x.clone(), y.clone(), t0.clone(), y_lens, HW)
        muse_out = muse_block(x.clone(), y.clone(), t0.clone(), y_lens, HW)
    
    return compare_tensors(f"Block {block_idx}", off_out, muse_out)


def compare_final_layer(official_model, muse_model, x, t):
    """Compare final layer outputs."""
    print("\n[5] Comparing Final Layer...")
    
    with torch.no_grad():
        off_out = official_model.final_layer(x.clone(), t.clone())
        muse_out = muse_model.final_layer(x.clone(), t.clone())
    
    return compare_tensors("final_layer", off_out, muse_out)


def run_full_alignment_test():
    """Run complete layer-by-layer alignment test."""
    print("=" * 70)
    print("Sana Forward Alignment Test - Layer by Layer Comparison")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    # Check paths
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return False
    
    # Load checkpoint
    print("\n[Loading] Loading checkpoint...")
    state_dict = load_checkpoint(CHECKPOINT_PATH)
    print(f"  Loaded {len(state_dict)} keys")
    
    # Load official model
    print("\n[Loading] Loading official Sana model...")
    try:
        official_model = load_official_model(state_dict, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in official_model.parameters()):,}")
    except ImportError as e:
        print(f"  ERROR: {e}")
        print("\n  To run the full alignment test, ensure Sana repo is in PYTHONPATH:")
        print(f"  PYTHONPATH={SANA_REPO_PATH}:$PYTHONPATH python {__file__}")
        return False
    
    # Load muse model
    print("\n[Loading] Loading muse Sana model...")
    muse_model = load_muse_model(state_dict, device, dtype)
    print(f"  Parameters: {sum(p.numel() for p in muse_model.parameters()):,}")
    
    # Create inputs
    print("\n[Setup] Creating test inputs...")
    x, timestep, y, mask = create_test_inputs(device=device, dtype=dtype)
    print(f"  x: {x.shape}")
    print(f"  timestep: {timestep}")
    print(f"  y: {y.shape}")
    print(f"  mask: {mask.shape}")
    
    all_passed = True
    
    # Compare patch embedding
    passed = compare_patch_embed(official_model, muse_model, x, dtype)
    all_passed = all_passed and passed
    
    # Compare timestep embedding
    passed = compare_timestep_embed(official_model, muse_model, timestep)
    all_passed = all_passed and passed
    
    # Compare caption embedding
    passed = compare_caption_embed(official_model, muse_model, y, mask, dtype)
    all_passed = all_passed and passed
    
    # Prepare intermediate values for block comparison
    print("\n[4] Comparing Transformer Blocks...")
    
    with torch.no_grad():
        # Prepare x
        x_off = x.to(dtype)
        x_muse = x.to(dtype)
        
        # Get patch embeddings
        x_off = official_model.x_embedder(x_off)
        x_muse = muse_model.x_embedder(x_muse)
        
        # Get timestep embeddings
        t_off = timestep.long().to(torch.float32)
        t_off = official_model.t_embedder(t_off)
        t0_off = official_model.t_block(t_off)
        
        t_muse = timestep.long().to(torch.float32)
        t_muse = muse_model.t_embedder(t_muse)
        t0_muse = muse_model.t_block(t_muse)
        
        # Get caption embeddings
        y_off = y.to(dtype)
        y_muse = y.to(dtype)
        y_off = official_model.y_embedder(y_off, False, mask=mask)
        y_muse = muse_model.y_embedder(y_muse, False, mask=mask)
        
        if MODEL_CONFIG["y_norm"]:
            y_off = official_model.attention_y_norm(y_off)
            y_muse = muse_model.attention_y_norm(y_muse)
        
        # Set up HW
        patch_size = MODEL_CONFIG["patch_size"]
        H = W = x.shape[-1] // patch_size
        HW = (H, W)
        
        # Process mask and y for cross-attention (matching official logic)
        # Check xformers availability
        try:
            import xformers
            _xformers_available = True
        except ImportError:
            _xformers_available = False
        
        mask_off = mask.to(torch.int16).squeeze(1).squeeze(1)
        mask_muse = mask.to(torch.int16).squeeze(1).squeeze(1)
        
        if _xformers_available:
            y_off = y_off.squeeze(1).masked_select(mask_off.unsqueeze(-1) != 0).view(1, -1, x_off.shape[-1])
            y_lens_off = mask_off.sum(dim=1).tolist()
            
            y_muse = y_muse.squeeze(1).masked_select(mask_muse.unsqueeze(-1) != 0).view(1, -1, x_muse.shape[-1])
            y_lens_muse = mask_muse.sum(dim=1).tolist()
        else:
            y_lens_off = [y_off.shape[2]] * y_off.shape[0]
            y_off = y_off.squeeze(1).view(1, -1, x_off.shape[-1])
            y_lens_muse = y_lens_off
            y_muse = y_muse.squeeze(1).view(1, -1, x_muse.shape[-1])
        
        # Compare each block
        for i, (off_block, muse_block) in enumerate(zip(official_model.blocks, muse_model.blocks)):
            # Run both blocks
            x_off_new = off_block(x_off.clone(), y_off.clone(), t0_off.clone(), y_lens_off, HW)
            x_muse_new = muse_block(x_muse.clone(), y_muse.clone(), t0_muse.clone(), y_lens_muse, HW)
            
            passed = compare_tensors(f"Block {i}", x_off_new, x_muse_new)
            all_passed = all_passed and passed
            
            if not passed:
                # If block fails, debug sub-components
                print(f"    Debugging Block {i} sub-components...")
                debug_block(i, off_block, muse_block, x_off, y_off, t0_off, y_lens_off, HW,
                           x_muse, y_muse, t0_muse, y_lens_muse)
            
            # Update x for next block
            x_off = x_off_new
            x_muse = x_muse_new
        
        # Compare final layer
        passed = compare_final_layer(official_model, muse_model, x_off, t_off)
        all_passed = all_passed and passed
        
        # Compare full model output
        print("\n[6] Comparing Full Model Output...")
        x_input, timestep_input, y_input, mask_input = create_test_inputs(device=device, dtype=dtype)
        
        off_final = official_model(x_input, timestep_input, y_input, mask=mask_input)
        muse_final = muse_model(x_input, timestep_input, y_input, mask=mask_input)
        
        passed = compare_tensors("Full Model", off_final, muse_final)
        all_passed = all_passed and passed
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("SUCCESS: All layers match within tolerance!")
    else:
        print("FAILED: Some layers have differences beyond tolerance.")
    print("=" * 70)
    
    return all_passed


def debug_block(block_idx, off_block, muse_block, x_off, y_off, t0_off, y_lens_off, HW,
                x_muse, y_muse, t0_muse, y_lens_muse):
    """Debug a single block by comparing its sub-components."""
    B, N, C = x_off.shape
    
    with torch.no_grad():
        # Get modulation parameters
        shift_msa_off, scale_msa_off, gate_msa_off, shift_mlp_off, scale_mlp_off, gate_mlp_off = (
            off_block.scale_shift_table[None] + t0_off.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        
        shift_msa_muse, scale_msa_muse, gate_msa_muse, shift_mlp_muse, scale_mlp_muse, gate_mlp_muse = (
            muse_block.scale_shift_table[None] + t0_muse.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        
        compare_tensors(f"  Block {block_idx} scale_shift_table", 
                       off_block.scale_shift_table, muse_block.scale_shift_table)
        compare_tensors(f"  Block {block_idx} shift_msa", shift_msa_off, shift_msa_muse)
        
        # Check norm1
        norm1_off = off_block.norm1(x_off)
        norm1_muse = muse_block.norm1(x_muse)
        compare_tensors(f"  Block {block_idx} norm1", norm1_off, norm1_muse)
        
        # Check self-attention input (after modulation)
        def t2i_modulate(x, shift, scale):
            return x * (1 + scale) + shift
        
        attn_input_off = t2i_modulate(norm1_off, shift_msa_off, scale_msa_off)
        attn_input_muse = t2i_modulate(norm1_muse, shift_msa_muse, scale_msa_muse)
        compare_tensors(f"  Block {block_idx} attn_input", attn_input_off, attn_input_muse)
        
        # Check self-attention output
        attn_out_off = off_block.attn(attn_input_off, HW=HW)
        attn_out_muse = muse_block.attn(attn_input_muse, HW=HW)
        compare_tensors(f"  Block {block_idx} self_attn", attn_out_off, attn_out_muse)
        
        # Check cross-attention
        x_after_self_off = x_off + gate_msa_off * attn_out_off
        x_after_self_muse = x_muse + gate_msa_muse * attn_out_muse
        
        cross_attn_off = off_block.cross_attn(x_after_self_off, y_off, y_lens_off)
        cross_attn_muse = muse_block.cross_attn(x_after_self_muse, y_muse, y_lens_muse)
        compare_tensors(f"  Block {block_idx} cross_attn", cross_attn_off, cross_attn_muse)


def run_muse_only_test():
    """Run a quick test of the muse model only (no official comparison)."""
    print("=" * 60)
    print("Muse Sana Model Test (standalone)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return
    
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    
    print("\n[1/3] Loading muse Sana model...")
    try:
        state_dict = load_checkpoint(CHECKPOINT_PATH)
        muse_model = load_muse_model(state_dict, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in muse_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[2/3] Creating test inputs...")
    x, timestep, y, mask = create_test_inputs(device=device, dtype=dtype)
    print(f"  x: {x.shape}")
    print(f"  timestep: {timestep}")
    print(f"  y: {y.shape}")
    print(f"  mask: {mask.shape}")
    
    print("\n[3/3] Running forward pass...")
    try:
        with torch.no_grad():
            output = muse_model(x, timestep, y, mask=mask)
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
    not os.path.exists(CHECKPOINT_PATH),
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}"
)
class TestSanaMuseModel:
    """Test suite for muse Sana model."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.state_dict = load_checkpoint(CHECKPOINT_PATH)
    
    def test_model_loading(self):
        muse_model = load_muse_model(self.state_dict, self.device, self.dtype)
        muse_params = sum(p.numel() for p in muse_model.parameters())
        assert muse_params > 1_000_000_000
        assert muse_params < 2_000_000_000
    
    def test_forward_pass(self):
        muse_model = load_muse_model(self.state_dict, self.device, self.dtype)
        x, timestep, y, mask = create_test_inputs(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            output = muse_model(x, timestep, y, mask=mask)
        
        assert output.shape == (1, MODEL_CONFIG["in_channels"], 32, 32)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--muse-only", action="store_true", help="Only test muse model")
    args = parser.parse_args()
    
    if args.muse_only:
        run_muse_only_test()
    else:
        # Try full alignment test, fall back to muse-only if official not available
        try:
            run_full_alignment_test()
        except ImportError as e:
            print(f"\nCannot run full alignment test: {e}")
            print("\nFalling back to muse-only test...")
            print("-" * 60)
            run_muse_only_test()
