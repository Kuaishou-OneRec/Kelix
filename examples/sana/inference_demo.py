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
Sana Inference Demo.

This script demonstrates loading a Sana model checkpoint and generating images
from text prompts using the muse framework.

Usage:
    python examples/sana/inference_demo.py \
        --model-dir /path/to/muse/checkpoint \
        --prompt "A cat sitting on a couch" \
        --output output.png \
        --num-steps 20 \
        --cfg-scale 4.5
"""

import argparse
import logging
import os
from typing import Optional

import torch
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sana Inference Demo")
    
    # Model arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to Muse format checkpoint directory (contains config.json and model.safetensors)"
    )
    
    # VAE arguments
    parser.add_argument(
        "--vae-pretrained",
        type=str,
        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        help="Pretrained VAE model path or HuggingFace repo"
    )
    
    # Text encoder arguments
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="google/gemma-2-2b-it",
        help="Text encoder model name"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=300,
        help="Maximum text sequence length"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean, photorealistic, 8k",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for CFG"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Output image size (square)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    
    return parser.parse_args()


def load_model(model_dir: str, device: torch.device, dtype: torch.dtype):
    """Load Sana model from Muse format checkpoint.
    
    Args:
        model_dir: Path to checkpoint directory
        device: Target device
        dtype: Model dtype
    
    Returns:
        Loaded SanaModel
    """
    from muse.models.sana import SanaModel
    
    logger.info(f"Loading model from {model_dir}")
    model = SanaModel.from_pretrained(model_dir)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def load_vae(vae_pretrained: str, device: torch.device, dtype: torch.dtype):
    """Load VAE model.
    
    Args:
        vae_pretrained: VAE model path or HuggingFace repo
        device: Target device
        dtype: Model dtype
    
    Returns:
        Loaded VAE model
    """
    from muse.models.sana import get_vae
    
    logger.info(f"Loading VAE from {vae_pretrained}")
    vae = get_vae(
        vae_type="AutoencoderDC",
        vae_pretrained=vae_pretrained,
        device=device,
        dtype=dtype,
    )
    
    return vae


def load_text_encoder(text_encoder_name: str, device: torch.device, dtype: torch.dtype):
    """Load text encoder and tokenizer.
    
    Args:
        text_encoder_name: Text encoder model name
        device: Target device
        dtype: Model dtype
    
    Returns:
        Tuple of (tokenizer, text_encoder)
    """
    from muse.models.sana import get_text_encoder
    
    logger.info(f"Loading text encoder from {text_encoder_name}")
    tokenizer, text_encoder = get_text_encoder(
        name=text_encoder_name,
        device=device,
        dtype=dtype,
    )
    
    return tokenizer, text_encoder


def encode_prompts(
    tokenizer,
    text_encoder,
    prompts: list,
    max_length: int,
    device: torch.device,
):
    """Encode text prompts to embeddings.
    
    Args:
        tokenizer: Tokenizer instance
        text_encoder: Text encoder model
        prompts: List of text prompts
        max_length: Maximum sequence length
        device: Target device
    
    Returns:
        Tuple of (text_embeds, attention_mask)
    """
    from muse.models.sana import encode_text
    
    text_embeds, attention_mask = encode_text(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        texts=prompts,
        max_length=max_length,
        device=device,
    )
    
    return text_embeds, attention_mask


def get_scheduler(num_steps: int, flow_shift: float = 3.0):
    """Get diffusers scheduler for sampling.
    
    Args:
        num_steps: Number of sampling steps
        flow_shift: Flow shift parameter
    
    Returns:
        Configured scheduler
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=flow_shift,
    )
    scheduler.set_timesteps(num_steps)
    
    return scheduler


@torch.no_grad()
def generate(
    model,
    vae,
    scheduler,
    text_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    uncond_embeds: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    cfg_scale: float = 4.5,
    image_size: int = 1024,
    latent_channels: int = 32,
    vae_downsample: int = 32,
    device: torch.device = None,
    dtype: torch.dtype = None,
    seed: int = 42,
):
    """Generate images from text embeddings.
    
    Args:
        model: Sana diffusion model
        vae: VAE decoder
        scheduler: Diffusion scheduler
        text_embeds: Text embeddings [B, 1, L, D]
        attention_mask: Attention mask [B, 1, 1, L]
        uncond_embeds: Unconditional embeddings for CFG
        uncond_mask: Unconditional attention mask
        cfg_scale: Classifier-free guidance scale
        image_size: Output image size
        latent_channels: Number of latent channels
        vae_downsample: VAE downsample factor
        device: Target device
        dtype: Model dtype
        seed: Random seed
    
    Returns:
        Generated images as PIL Image list
    """
    from muse.models.sana import vae_decode
    
    batch_size = text_embeds.shape[0]
    latent_size = image_size // vae_downsample
    
    # Set random seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Initialize latents with random noise
    latents = torch.randn(
        (batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    
    # Scale initial noise by scheduler
    latents = latents * scheduler.init_noise_sigma
    
    # Prepare CFG: concat conditional and unconditional
    do_cfg = cfg_scale > 1.0 and uncond_embeds is not None
    if do_cfg:
        text_embeds_cfg = torch.cat([uncond_embeds, text_embeds], dim=0)
        attention_mask_cfg = torch.cat([uncond_mask, attention_mask], dim=0) if attention_mask is not None else None
    else:
        text_embeds_cfg = text_embeds
        attention_mask_cfg = attention_mask
    
    # Sampling loop
    logger.info(f"Starting sampling with {len(scheduler.timesteps)} steps...")
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        
        # Scale latents (some schedulers require this)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Prepare timestep
        timestep = torch.tensor([t] * latent_model_input.shape[0], device=device, dtype=dtype)
        
        # Model prediction (velocity)
        noise_pred = model.forward_with_dpmsolver(
            latent_model_input,
            timestep,
            text_embeds_cfg,
            mask=attention_mask_cfg,
        )
        
        # Apply CFG
        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        if (i + 1) % 5 == 0 or i == len(scheduler.timesteps) - 1:
            logger.info(f"Step {i + 1}/{len(scheduler.timesteps)}")
    
    # Decode latents to images
    logger.info("Decoding latents to images...")
    images = vae_decode(vae, latents)
    
    # Convert to PIL images
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in images]
    
    return pil_images


def main():
    args = get_args()
    
    # Setup device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    logger.info(f"Device: {device}, dtype: {dtype}")
    
    # Load models
    model = load_model(args.model_dir, device, dtype)
    vae = load_vae(args.vae_pretrained, device, dtype)
    tokenizer, text_encoder = load_text_encoder(args.text_encoder, device, dtype)
    
    # Get scheduler
    scheduler = get_scheduler(args.num_steps)
    
    # Encode prompts
    logger.info(f"Prompt: {args.prompt}")
    text_embeds, attention_mask = encode_prompts(
        tokenizer,
        text_encoder,
        [args.prompt],
        args.max_text_length,
        device,
    )
    
    # Encode negative prompt for CFG
    uncond_embeds, uncond_mask = None, None
    if args.cfg_scale > 1.0:
        negative_prompt = args.negative_prompt if args.negative_prompt else ""
        logger.info(f"Negative prompt: '{negative_prompt}'")
        uncond_embeds, uncond_mask = encode_prompts(
            tokenizer,
            text_encoder,
            [negative_prompt],
            args.max_text_length,
            device,
        )
    
    # Generate images
    logger.info("Generating image...")
    images = generate(
        model=model,
        vae=vae,
        scheduler=scheduler,
        text_embeds=text_embeds,
        attention_mask=attention_mask,
        uncond_embeds=uncond_embeds,
        uncond_mask=uncond_mask,
        cfg_scale=args.cfg_scale,
        image_size=args.image_size,
        latent_channels=model.config.in_channels,
        vae_downsample=model.config.vae_downsample_rate,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    
    # Save images
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    images[0].save(output_path)
    logger.info(f"Image saved to {output_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
