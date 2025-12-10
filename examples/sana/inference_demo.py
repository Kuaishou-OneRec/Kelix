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

Supports two sampling methods:
- dpm-solver (default): Flow-DPM-Solver for faster sampling with fewer steps
- euler: Standard Euler scheduler from diffusers

Usage:
    # Using DPM-Solver (recommended, faster)
    python examples/sana/inference_demo.py \
        --model-dir /path/to/checkpoint \
        --tokenizer google/gemma-2-2b-it \
        --prompt "A cat sitting on a couch" \
        --output output.png \
        --num-steps 20 \
        --cfg-scale 4.5 \
        --sampler dpm-solver \
        --flow-shift 3.0
    
    # Using Euler scheduler
    python examples/sana/inference_demo.py \
        --model-dir /path/to/checkpoint \
        --tokenizer google/gemma-2-2b-it \
        --prompt "A cat sitting on a couch" \
        --output output.png \
        --num-steps 20 \
        --cfg-scale 4.5 \
        --sampler euler
"""

import argparse
import logging
import os
from pathlib import Path
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
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--transformer-subfolder",
        type=str,
        default="transformer",
        help="Subfolder for transformer in Diffusers format (default: transformer)"
    )
    
    # VAE arguments
    parser.add_argument(
        "--vae-pretrained",
        type=str,
        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        help="Pretrained VAE model path or HuggingFace repo"
    )
    
    # Tokenizer and text encoder arguments
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-2-2b-it",
        help="Tokenizer path or HuggingFace repo"
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default=None,
        help="Text encoder path (defaults to same as tokenizer)"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=300,
        help="Maximum text sequence length"
    )
    
    # Scheduler arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Scheduler path (optional, will create default if not specified)"
    )
    
    # Sampler arguments
    parser.add_argument(
        "--sampler",
        type=str,
        default="dpm-solver",
        choices=["dpm-solver", "euler"],
        help="Sampling algorithm: dpm-solver (faster) or euler"
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=3.0,
        help="Flow shift parameter for DPM-Solver (default: 3.0)"
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


def load_model(model_dir: str, device: torch.device, dtype: torch.dtype, subfolder: str = "transformer"):
    """Load Sana model from checkpoint.
    
    Args:
        model_dir: Path to checkpoint directory
        device: Target device
        dtype: Model dtype
        subfolder: Subfolder for transformer in Diffusers format
    
    Returns:
        Loaded SanaModel
    """
    from muse.models.sana import SanaModel
    
    model_path = Path(model_dir)
    
    # Check if it's Muse format (config.json at root without transformer subfolder)
    # or Diffusers format (has transformer/ subfolder)
    if (model_path / "config.json").exists() and not (model_path / subfolder).exists():
        # Muse format
        logger.info(f"Loading Muse format model from {model_dir}")
        model = SanaModel.from_pretrained(model_dir)
    else:
        # Diffusers format - need to convert
        logger.info(f"Loading Diffusers format model from {model_dir}/{subfolder}")
        from examples.sana.convert_hf_checkpoint import load_diffusers_model, _build_sana_config
        from muse.training.common import set_default_dtype
        
        hf_model = load_diffusers_model(model_dir, subfolder, dtype, device)
        
        # Get config
        if hasattr(hf_model.config, 'to_dict'):
            hf_config_dict = hf_model.config.to_dict()
        else:
            hf_config_dict = dict(hf_model.config)
        
        config = _build_sana_config(hf_config_dict)
        
        # Create and load model
        with set_default_dtype(str(dtype).split('.')[-1]):
            model = SanaModel(config)
        
        muse_state_dict = model.convert_hf_state_dict(hf_model.state_dict())
        model.load_state_dict(muse_state_dict, strict=False)
    
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
    from diffusers import AutoencoderDC
    
    logger.info(f"Loading VAE from {vae_pretrained}")
    vae = AutoencoderDC.from_pretrained(vae_pretrained, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    return vae


def load_tokenizer_and_encoder(
    tokenizer_path: str,
    text_encoder_path: Optional[str],
    device: torch.device,
    dtype: torch.dtype
):
    """Load tokenizer and text encoder.
    
    Args:
        tokenizer_path: Tokenizer path or HuggingFace repo
        text_encoder_path: Text encoder path (defaults to tokenizer_path)
        device: Target device
        dtype: Model dtype
    
    Returns:
        Tuple of (tokenizer, text_encoder)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    encoder_path = text_encoder_path or tokenizer_path
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    logger.info(f"Loading text encoder from {encoder_path}")
    model = AutoModelForCausalLM.from_pretrained(encoder_path, torch_dtype=dtype)
    
    # Get decoder if available
    if hasattr(model, 'get_decoder'):
        text_encoder = model.get_decoder()
    else:
        text_encoder = model
    
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    
    return tokenizer, text_encoder


def load_scheduler(scheduler_path: Optional[str], num_steps: int):
    """Load scheduler.
    
    Args:
        scheduler_path: Scheduler path (optional)
        num_steps: Number of sampling steps
    
    Returns:
        Configured scheduler
    """
    from diffusers import DPMSolverMultistepScheduler
    
    if scheduler_path:
        logger.info(f"Loading scheduler from {scheduler_path}")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(scheduler_path)
    else:
        logger.info("Creating default DPMSolverMultistepScheduler")
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            flow_shift=3.0,
            use_flow_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )
    
    scheduler.set_timesteps(num_steps)
    return scheduler


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
):
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
    
    Returns:
        List of PIL images
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
    
    # Apply inverse scaling factor
    if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
        latents = latents / vae.config.scaling_factor
    
    images = vae.decode(latents).sample
    
    # Convert to PIL images
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in images]
    
    return pil_images


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
):
    """Generate images using Euler scheduler (from diffusers)."""
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
    
    # Apply inverse scaling factor
    if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
        latents = latents / vae.config.scaling_factor
    
    images = vae.decode(latents).sample
    
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
    logger.info(f"Sampler: {args.sampler}")
    
    # Load models
    model = load_model(args.model_dir, device, dtype, args.transformer_subfolder)
    vae = load_vae(args.vae_pretrained, device, dtype)
    tokenizer, text_encoder = load_tokenizer_and_encoder(
        args.tokenizer, args.text_encoder, device, dtype
    )
    
    # Only load scheduler for euler sampler
    scheduler = None
    if args.sampler == "euler":
        scheduler = load_scheduler(args.scheduler, args.num_steps)
    
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
    
    if args.sampler == "dpm-solver":
        images = generate_with_dpm_solver(
            model=model,
            vae=vae,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            flow_shift=args.flow_shift,
            image_size=args.image_size,
            latent_channels=model.config.in_channels,
            vae_downsample=model.config.vae_downsample_rate,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
    else:  # euler
        images = generate_with_euler(
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
