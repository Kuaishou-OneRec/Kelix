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

This test verifies that the muse implementation of Sana can load checkpoints
and run forward pass correctly. For full alignment testing against the official
implementation, run on a system with the Sana repo and all dependencies installed.

Usage:
    pytest tests/models/sana/test_forward_alignment.py -v
    
    # Or run directly:
    python tests/models/sana/test_forward_alignment.py
"""

import os
import sys

# Mock out problematic imports before importing muse modules
class MockFlashAttn:
    def flash_attn_func(*args, **kwargs):
        raise NotImplementedError("flash_attn not available")
    def flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError("flash_attn not available")

sys.modules['flash_attn'] = MockFlashAttn

import pytest
import torch
import torch.nn as nn

# Checkpoint path - update this to your checkpoint location
CHECKPOINT_PATH = "/llm_reco_ssd/zhouyang12/models/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"


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


def load_muse_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load the muse Sana model."""
    # Direct import to avoid loading other models that have heavy dependencies
    from muse.models.sana.modeling import SanaModel
    from muse.config.model_config import SanaConfig
    
    # Create config
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
    
    # Create model
    model = SanaModel(config)
    
    # Check if checkpoint is a git-lfs pointer file
    with open(checkpoint_path, 'rb') as f:
        header = f.read(50)
    if b'git-lfs' in header:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is a git-lfs pointer file.\n"
            f"Please run 'git lfs pull' in the model directory to download the actual weights."
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    
    # Convert state dict using model's converter
    converted_state_dict = model.convert_sana_state_dict(new_state_dict)
    
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
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
    
    # Latent input: [B, C, H, W]
    x = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device, dtype=dtype)
    
    # Timestep: scalar or 1D tensor
    timestep = torch.tensor([500.0], device=device, dtype=torch.float32)
    
    # Text embeddings: [B, 1, L, D]
    y = torch.randn(batch_size, 1, text_length, text_channels, device=device, dtype=dtype)
    
    # Attention mask: [B, 1, 1, L] - all ones (no padding)
    mask = torch.ones(batch_size, 1, 1, text_length, device=device, dtype=dtype)
    
    return x, timestep, y, mask


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH),
    reason=f"Checkpoint not found at {CHECKPOINT_PATH}"
)
class TestSanaMuseModel:
    """Test suite for muse Sana model."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.dtype = torch.float32
    
    def test_model_loading(self):
        """Test that muse model can be loaded successfully."""
        muse_model = load_muse_model(CHECKPOINT_PATH, self.device, self.dtype)
        
        # Check parameter count
        muse_params = sum(p.numel() for p in muse_model.parameters())
        print(f"Muse model parameters: {muse_params:,}")
        
        # Expected ~1.6B parameters
        assert muse_params > 1_000_000_000, "Model should have >1B parameters"
        assert muse_params < 2_000_000_000, "Model should have <2B parameters"
    
    def test_forward_pass(self):
        """Test that forward pass runs without errors."""
        muse_model = load_muse_model(CHECKPOINT_PATH, self.device, self.dtype)
        
        # Create inputs
        x, timestep, y, mask = create_test_inputs(
            device=self.device,
            dtype=self.dtype,
        )
        
        # Forward pass
        with torch.no_grad():
            output = muse_model(x, timestep, y, mask=mask)
        
        # Check output shape
        expected_shape = (1, MODEL_CONFIG["in_channels"], 32, 32)
        assert output.shape == expected_shape, \
            f"Output shape mismatch: got {output.shape}, expected {expected_shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        print(f"Output shape: {output.shape}")
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    def test_forward_deterministic(self):
        """Test that forward pass is deterministic."""
        muse_model = load_muse_model(CHECKPOINT_PATH, self.device, self.dtype)
        
        # Run twice with same seed
        x1, t1, y1, m1 = create_test_inputs(seed=123, device=self.device, dtype=self.dtype)
        x2, t2, y2, m2 = create_test_inputs(seed=123, device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            out1 = muse_model(x1, t1, y1, mask=m1)
            out2 = muse_model(x2, t2, y2, mask=m2)
        
        torch.testing.assert_close(out1, out2, rtol=0, atol=0)
        print("Forward pass is deterministic: PASSED")
    
    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        muse_model = load_muse_model(CHECKPOINT_PATH, self.device, self.dtype)
        
        for batch_size in [1, 2]:
            x, timestep, y, mask = create_test_inputs(
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )
            timestep = timestep.expand(batch_size)
            
            with torch.no_grad():
                output = muse_model(x, timestep, y, mask=mask)
            
            expected_shape = (batch_size, MODEL_CONFIG["in_channels"], 32, 32)
            assert output.shape == expected_shape, \
                f"Batch {batch_size}: got {output.shape}, expected {expected_shape}"
            print(f"Batch size {batch_size}: PASSED")


def run_muse_test():
    """Run a quick test of the muse model."""
    print("=" * 60)
    print("Muse Sana Model Test")
    print("=" * 60)
    
    device = torch.device("cpu")
    dtype = torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    # Check checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please update CHECKPOINT_PATH to point to your Sana checkpoint.")
        return
    
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    
    # Load model
    print("\n[1/3] Loading muse Sana model...")
    try:
        muse_model = load_muse_model(CHECKPOINT_PATH, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in muse_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create inputs
    print("\n[2/3] Creating test inputs...")
    x, timestep, y, mask = create_test_inputs(device=device, dtype=dtype)
    print(f"  x: {x.shape}")
    print(f"  timestep: {timestep}")
    print(f"  y: {y.shape}")
    print(f"  mask: {mask.shape}")
    
    # Forward pass
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
    print("✓ Muse model test PASSED!")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Full Alignment Test Instructions")
    print("=" * 60)
    print("""
To run full alignment test against the official Sana implementation:

1. Install Sana dependencies:
   cd /path/to/Sana
   pip install -e .

2. Make sure xformers is installed (required by official Sana):
   pip install xformers

3. Run the comparison script (create a separate script with both models):

   ```python
   import torch
   from diffusion.model.nets.sana_multi_scale import SanaMS_1600M_P1_D20
   from muse.models.sana import SanaModel
   from muse.config.model_config import SanaConfig
   
   # Load both models with same checkpoint
   checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
   
   # ... load official_model
   # ... load muse_model
   
   # Create identical inputs
   x = torch.randn(1, 32, 32, 32)
   timestep = torch.tensor([500.0])
   y = torch.randn(1, 1, 300, 2304)
   mask = torch.ones(1, 1, 1, 300)
   
   # Compare outputs
   with torch.no_grad():
       official_out = official_model(x, timestep, y, mask=mask)
       muse_out = muse_model(x, timestep, y, mask=mask)
   
   print(f"Max diff: {(official_out - muse_out).abs().max()}")
   ```
""")


if __name__ == "__main__":
    run_muse_test()
