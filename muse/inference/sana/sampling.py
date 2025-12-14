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
Sampling utilities for Sana image generation.

Provides reusable functions for:
- Text prompt encoding
- Image generation with DPM-Solver
- Image generation with Euler scheduler
- Latent decoding
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_prompts(
    tokenizer,
    text_encoder,
    prompts: List[str],
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
        - text_embeds: [B, 1, L, D] tensor
        - attention_mask: [B, 1, 1, L] tensor
    """
    # Tokenize
    tokens = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    
    # Encode
    with torch.no_grad():
        outputs = text_encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        
        # Get hidden states
        if hasattr(outputs, 'last_hidden_state'):
            text_embeds = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            text_embeds = outputs[0]
        else:
            text_embeds = outputs
    
    # Add dimension for cross attention: [B, 1, L, D]
    text_embeds = text_embeds[:, None]
    attention_mask = attention_mask[:, None, None]
    
    return text_embeds, attention_mask


@torch.no_grad()
def decode_latents(
    vae,
    latents: torch.Tensor,
    return_pil: bool = True,
) -> Union[List[Image.Image], torch.Tensor]:
    """Decode latents to images.
    
    Args:
        vae: VAE decoder model
        latents: Latent tensor [B, C, H, W]
        return_pil: If True, return PIL images; otherwise return tensor
    
    Returns:
        List of PIL images or tensor [B, C, H, W] in [0, 1] range
    """
    # Apply inverse scaling factor
    if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
        latents = latents / vae.config.scaling_factor
    
    images = vae.decode(latents).sample
    
    # Convert to [0, 1] range
    images = (images / 2 + 0.5).clamp(0, 1)
    
    if return_pil:
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images
    else:
        return images


@torch.no_grad()
def generate_with_dpm_solver(
    model,
    vae,
    text_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    uncond_embeds: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    cfg_scale: float = 4.5,
    num_steps: int = 20,
    flow_shift: float = 3.0,
    image_size: int = 1024,
    latent_channels: int = 32,
    vae_downsample: int = 32,
    device: torch.device = None,
    dtype: torch.dtype = None,
    seed: int = 42,
    return_pil: bool = True,
) -> Union[List[Image.Image], torch.Tensor]:
    """Generate images using Flow-DPM-Solver (faster sampling).
    
    Args:
        model: Sana model
        vae: VAE decoder
        text_embeds: Conditional text embeddings [B, 1, L, D]
        attention_mask: Text attention mask [B, 1, 1, L]
        uncond_embeds: Unconditional embeddings for CFG
        uncond_mask: Unconditional attention mask
        cfg_scale: Classifier-free guidance scale
        num_steps: Number of sampling steps
        flow_shift: Flow shift parameter
        image_size: Output image size
        latent_channels: Number of latent channels
        vae_downsample: VAE downsampling factor
        device: Target device
        dtype: Data type
        seed: Random seed
        return_pil: If True, return PIL images; otherwise return tensor
    
    Returns:
        List of PIL images or tensor
    """
    from muse.inference import create_flow_dpm_solver
    
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
    
    # Prepare model function that accepts (x, t, cond, mask) 
    def model_fn(x, t, cond, mask=None):
        return model.forward_with_dpmsolver(x, t, cond, mask=mask)
    
    # Prepare condition and uncondition for DPM-Solver
    # DPM-Solver handles CFG internally
    condition = text_embeds
    uncondition = uncond_embeds if uncond_embeds is not None else torch.zeros_like(text_embeds)
    
    # Create model kwargs with attention mask
    model_kwargs = {"mask": attention_mask}
    
    # Create DPM-Solver
    logger.info(f"Using DPM-Solver with {num_steps} steps, flow_shift={flow_shift}")
    solver = create_flow_dpm_solver(
        model_fn,
        condition=condition,
        uncondition=uncondition,
        cfg_scale=cfg_scale,
        model_kwargs=model_kwargs,
    )
    
    # Sample
    latents = solver.sample(
        latents,
        steps=num_steps,
        flow_shift=flow_shift,
        order=2,
        skip_type="time_uniform_flow",
        method="multistep",
        lower_order_final=True,
    )
    
    # Decode latents to images
    logger.info("Decoding latents to images...")
    return decode_latents(vae, latents, return_pil=return_pil)


@torch.no_grad()
def generate_with_euler(
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
    return_pil: bool = True,
) -> Union[List[Image.Image], torch.Tensor]:
    """Generate images using Euler scheduler (from diffusers).
    
    Args:
        model: Sana model
        vae: VAE decoder
        scheduler: Diffusers scheduler instance
        text_embeds: Conditional text embeddings [B, 1, L, D]
        attention_mask: Text attention mask [B, 1, 1, L]
        uncond_embeds: Unconditional embeddings for CFG
        uncond_mask: Unconditional attention mask
        cfg_scale: Classifier-free guidance scale
        image_size: Output image size
        latent_channels: Number of latent channels
        vae_downsample: VAE downsampling factor
        device: Target device
        dtype: Data type
        seed: Random seed
        return_pil: If True, return PIL images; otherwise return tensor
    
    Returns:
        List of PIL images or tensor
    """
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
    
    # Scale initial noise by scheduler (if applicable)
    if hasattr(scheduler, 'init_noise_sigma'):
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
    logger.info(f"Starting Euler sampling with {len(scheduler.timesteps)} steps...")
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        
        # Scale latents (some schedulers require this)
        if hasattr(scheduler, 'scale_model_input'):
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
    return decode_latents(vae, latents, return_pil=return_pil)
