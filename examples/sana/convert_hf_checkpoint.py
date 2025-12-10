#!/usr/bin/env python3
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
Convert Diffusers Sana checkpoint to Muse format.

This script converts a Diffusers SanaTransformer2DModel checkpoint to the Muse
SanaModel format.

Usage:
    python examples/sana/convert_hf_checkpoint.py \
        --hf-path Efficient-Large-Model/Sana_1600M_1024px_diffusers \
        --output-dir /path/to/muse/sana_checkpoint

Or with a local path:
    python examples/sana/convert_hf_checkpoint.py \
        --hf-path /path/to/diffusers/sana \
        --output-dir /path/to/muse/sana_checkpoint
"""

import argparse
import logging
from typing import Dict, Any, Optional

import torch

from muse.config import SanaConfig
from muse.models.sana import SanaModel
from muse.training.common import set_default_dtype

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Diffusers Sana checkpoint to Muse format"
    )
    
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Path to Diffusers Sana checkpoint (HuggingFace repo or local path)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for Muse format checkpoint"
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="transformer",
        help="Subfolder containing the transformer model (default: transformer)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for conversion"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for conversion"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by comparing forward outputs"
    )
    parser.add_argument(
        "--vae-pretrained",
        type=str,
        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        help="VAE model path (for config)"
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="google/gemma-2-2b-it",
        help="Text encoder model name (for config)"
    )
    
    return parser.parse_args()


def _build_sana_config(hf_cfg: Dict[str, Any], 
                       vae_pretrained: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                       text_encoder_name: str = "google/gemma-2-2b-it") -> SanaConfig:
    """Map Diffusers SanaTransformer2DModel config to Muse SanaConfig.
    
    Args:
        hf_cfg: Diffusers model config dict
        vae_pretrained: VAE model path
        text_encoder_name: Text encoder model name
    
    Returns:
        SanaConfig for Muse model
    """
    # Extract dimensions
    num_attention_heads = hf_cfg.get("num_attention_heads", 70)
    attention_head_dim = hf_cfg.get("attention_head_dim", 32)
    hidden_size = num_attention_heads * attention_head_dim
    
    # Number of heads for cross attention (used as num_heads in Muse)
    num_cross_attention_heads = hf_cfg.get("num_cross_attention_heads", 20)
    
    # QK norm: Diffusers uses string or None, Muse uses bool
    qk_norm_str = hf_cfg.get("qk_norm", None)
    qk_norm = qk_norm_str is not None and qk_norm_str != ""
    
    # Linear head dim (for LiteLA attention)
    linear_head_dim = attention_head_dim
    
    # Input/output channels
    in_channels = hf_cfg.get("in_channels", 32)
    out_channels = hf_cfg.get("out_channels", in_channels)
    
    # Check if model predicts sigma (variance)
    pred_sigma = out_channels == in_channels * 2
    
    # Determine attention type based on model structure
    # Diffusers Sana uses linear attention by default
    attn_type = "linear"
    cross_attn_type = "flash"
    
    config = SanaConfig(
        model_class="SanaModel",
        # Image/Latent dimensions
        input_size=hf_cfg.get("sample_size", 32),
        patch_size=hf_cfg.get("patch_size", 1),
        in_channels=in_channels,
        # Architecture dimensions
        hidden_size=hidden_size,
        depth=hf_cfg.get("num_layers", 20),
        num_heads=num_cross_attention_heads,
        mlp_ratio=hf_cfg.get("mlp_ratio", 2.5),
        # Text encoder configuration
        caption_channels=hf_cfg.get("caption_channels", 2304),
        model_max_length=300,  # Default for Gemma
        # Attention configuration
        attn_type=attn_type,
        cross_attn_type=cross_attn_type,
        linear_head_dim=linear_head_dim,
        qk_norm=qk_norm,
        cross_norm=qk_norm,  # Same as qk_norm by default
        # FFN configuration
        ffn_type="glumbconv",
        mlp_acts=("silu", "silu", None),
        # Output configuration
        pred_sigma=pred_sigma,
        learn_sigma=pred_sigma,
        # Position embedding
        use_pe=hf_cfg.get("interpolation_scale") is not None,
        pe_interpolation=hf_cfg.get("interpolation_scale", 1.0) or 1.0,
        # Normalization
        y_norm=True,
        y_norm_scale_factor=0.01,
        norm_eps=hf_cfg.get("norm_eps", 1e-6),
        # Training
        class_dropout_prob=0.1,
        drop_path=0.0,
        # VAE configuration
        vae_type="AutoencoderDC",
        vae_pretrained=vae_pretrained,
        vae_downsample_rate=32,
        # Text encoder configuration
        text_encoder_name=text_encoder_name,
    )
    
    return config


def load_diffusers_model(hf_path: str, subfolder: str, dtype: torch.dtype, device: torch.device):
    """Load Diffusers SanaTransformer2DModel.
    
    Args:
        hf_path: HuggingFace repo or local path
        subfolder: Subfolder containing the transformer
        dtype: Model dtype
        device: Target device
    
    Returns:
        Loaded Diffusers model
    """
    try:
        from diffusers import SanaTransformer2DModel
    except ImportError:
        raise ImportError(
            "diffusers is required for conversion. "
            "Install it with: pip install diffusers"
        )
    
    logger.info(f"Loading Diffusers model from {hf_path}/{subfolder}")
    
    hf_model = SanaTransformer2DModel.from_pretrained(
        hf_path,
        subfolder=subfolder,
        torch_dtype=dtype,
    )
    hf_model = hf_model.to(device)
    hf_model.eval()
    
    logger.info(f"Diffusers model loaded: {type(hf_model).__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in hf_model.parameters()):,}")
    
    return hf_model


def convert_hf_checkpoint(
    hf_path: str,
    output_dir: str,
    subfolder: str = "transformer",
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None,
    verify: bool = False,
    vae_pretrained: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
    text_encoder_name: str = "google/gemma-2-2b-it",
):
    """Convert Diffusers Sana checkpoint to Muse format.
    
    Args:
        hf_path: Path to Diffusers checkpoint
        output_dir: Output directory for Muse checkpoint
        subfolder: Subfolder containing transformer model
        dtype: Model dtype
        device: Target device
        verify: Whether to verify conversion
        vae_pretrained: VAE model path
        text_encoder_name: Text encoder name
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Diffusers model
    hf_model = load_diffusers_model(hf_path, subfolder, dtype, device)
    
    # Get config dict - handle both regular config and FrozenDict
    if hasattr(hf_model.config, 'to_dict'):
        hf_config_dict = hf_model.config.to_dict()
    else:
        # FrozenDict or similar - convert directly to dict
        hf_config_dict = dict(hf_model.config)
    
    logger.info("Diffusers config:")
    for key, value in sorted(hf_config_dict.items()):
        logger.info(f"  {key}: {value}")
    
    # Build Muse config
    config = _build_sana_config(hf_config_dict, vae_pretrained, text_encoder_name)
    logger.info(f"\nMuse config:")
    logger.info(f"  hidden_size: {config.hidden_size}")
    logger.info(f"  depth: {config.depth}")
    logger.info(f"  num_heads: {config.num_heads}")
    logger.info(f"  input_size: {config.input_size}")
    logger.info(f"  patch_size: {config.patch_size}")
    logger.info(f"  in_channels: {config.in_channels}")
    logger.info(f"  caption_channels: {config.caption_channels}")
    logger.info(f"  mlp_ratio: {config.mlp_ratio}")
    logger.info(f"  qk_norm: {config.qk_norm}")
    logger.info(f"  pred_sigma: {config.pred_sigma}")
    
    # Create Muse model
    logger.info("\nCreating Muse SanaModel...")
    with set_default_dtype(str(dtype).split('.')[-1]):
        model = SanaModel(config)
    
    # Convert state dict
    logger.info("Converting state dict...")
    hf_state_dict = hf_model.state_dict()
    muse_state_dict = model.convert_hf_state_dict(hf_state_dict)
    
    # Log conversion details
    logger.info(f"\nOriginal keys: {len(hf_state_dict)}")
    logger.info(f"Converted keys: {len(muse_state_dict)}")
    
    # Load state dict
    logger.info("\nLoading converted state dict...")
    missing_keys, unexpected_keys = model.load_state_dict(muse_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}):")
        for key in missing_keys[:20]:
            logger.warning(f"  {key}")
        if len(missing_keys) > 20:
            logger.warning(f"  ... and {len(missing_keys) - 20} more")
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:20]:
            logger.warning(f"  {key}")
        if len(unexpected_keys) > 20:
            logger.warning(f"  ... and {len(unexpected_keys) - 20} more")
    
    # Move to device and dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # Ensure all parameters have correct dtype
    for name, param in model.named_parameters():
        if param.dtype != dtype:
            logger.warning(f"Parameter {name} has dtype {param.dtype}, expected {dtype}")
            param.data = param.data.to(dtype=dtype)
    
    for name, buffer in model.named_buffers():
        if buffer.dtype not in [dtype, torch.int64, torch.int32, torch.int16, torch.bool]:
            logger.warning(f"Buffer {name} has dtype {buffer.dtype}")
    
    # Verify conversion (optional)
    if verify:
        logger.info("\n" + "=" * 60)
        logger.info("Verifying conversion...")
        verify_conversion(hf_model, model, config, device, dtype)
    
    # Save Muse model
    logger.info(f"\nSaving Muse model to {output_dir}")
    model.save_pretrained(output_dir)
    
    logger.info("Conversion completed successfully!")
    logger.info(f"Output directory: {output_dir}")


@torch.no_grad()
def verify_conversion(
    hf_model,
    muse_model,
    config: SanaConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    """Verify conversion by comparing forward outputs.
    
    Args:
        hf_model: Diffusers model
        muse_model: Muse model
        config: Muse config
        device: Target device
        dtype: Model dtype
    """
    batch_size = 1
    latent_size = config.input_size
    in_channels = config.in_channels
    caption_channels = config.caption_channels
    max_length = 64  # Use shorter length for testing
    
    logger.info(f"Test input shapes:")
    logger.info(f"  Latent: [{batch_size}, {in_channels}, {latent_size}, {latent_size}]")
    logger.info(f"  Text: [{batch_size}, {max_length}, {caption_channels}]")
    
    # Create random inputs
    latents = torch.randn(
        batch_size, in_channels, latent_size, latent_size,
        device=device, dtype=dtype
    )
    timestep = torch.tensor([500], device=device, dtype=dtype)
    
    # Text embeddings
    text_embeds = torch.randn(
        batch_size, max_length, caption_channels,
        device=device, dtype=dtype
    )
    attention_mask = torch.ones(batch_size, max_length, device=device, dtype=torch.int16)
    
    # Diffusers forward
    logger.info("Running Diffusers forward...")
    hf_output = hf_model(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=text_embeds,
        encoder_attention_mask=attention_mask,
    ).sample
    
    # Muse forward
    logger.info("Running Muse forward...")
    # Muse expects text_embeds in [B, 1, L, D] format
    text_embeds_muse = text_embeds[:, None]
    attention_mask_muse = attention_mask[:, None, None]
    
    muse_output = muse_model.forward_with_dpmsolver(
        x=latents,
        timestep=timestep,
        y=text_embeds_muse,
        mask=attention_mask_muse,
    )
    
    # Compare outputs
    diff = (hf_output - muse_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()
    
    # Relative differences
    hf_abs = hf_output.abs()
    relative_diff = diff / (hf_abs + 1e-8)
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()
    
    logger.info(f"\nOutput comparison:")
    logger.info(f"  Max diff: {max_diff:.6e}")
    logger.info(f"  Mean diff: {mean_diff:.6e}")
    logger.info(f"  Median diff: {median_diff:.6e}")
    logger.info(f"  Max relative diff: {max_relative_diff:.6e}")
    logger.info(f"  Mean relative diff: {mean_relative_diff:.6e}")
    
    logger.info("=" * 60)
    
    if max_diff < 1e-3:
        logger.info("SUCCESS: Outputs match within tolerance!")
    elif max_diff < 1e-2:
        logger.info("WARNING: Small differences detected, but within acceptable range")
    else:
        logger.warning("FAILURE: Significant differences detected")
        logger.warning("This may be due to attention implementation differences")


def main():
    args = get_args()
    
    # Setup dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)
    
    convert_hf_checkpoint(
        hf_path=args.hf_path,
        output_dir=args.output_dir,
        subfolder=args.subfolder,
        dtype=dtype,
        device=device,
        verify=args.verify,
        vae_pretrained=args.vae_pretrained,
        text_encoder_name=args.text_encoder,
    )


if __name__ == "__main__":
    main()
