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
Sana AE Inference Script.

This script performs image reconstruction inference using a trained Sana AE model.
The condition is from KeyeImageTokenizer.

Usage:
    python recipes/inference_sana_ae.py \
        --model-dir /path/to/model \
        --vae-dir mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
        --image-tokenizer-dir /path/to/tokenizer \
        --input-dir /path/to/images \
        --output-dir /path/to/output
"""

from typing import Optional
import os
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from diffusers import FlowMatchEulerDiscreteScheduler

from muse.models.base import Model
from muse.training.common import get_torch_dtype

# Import helper functions from train_sana_ae
from recipes.train_sana_ae import (
    load_vae,
    load_image_tokenizer,
    load_visualization_images,
    tokenize_images,
    vae_encode,
)


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Sana AE Inference")

    # Model args
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the trained model directory (HuggingFace format)")
    
    # VAE args
    parser.add_argument("--vae-dir", type=str,
                        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                        help="Pretrained VAE model path")

    # Image Tokenizer args
    parser.add_argument("--image-tokenizer-dir", type=str, required=True,
                        help="Image tokenizer model path")
    
    parser.add_argument("--max-condition-length", type=int, default=324,
                        help="Maximum condition sequence length")

    # Input/Output args
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input images")
    
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output images")

    # Inference args
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Image size for inference")
    
    parser.add_argument("--cfg-scale", type=float, default=4.5,
                        help="CFG scale for sampling")
    
    parser.add_argument("--num-sampling-steps", type=int, default=20,
                        help="Number of Euler sampling steps")
    
    parser.add_argument("--flow-shift", type=float, default=3.0,
                        help="Flow shift parameter")
    
    parser.add_argument("--num-images", type=int, default=None,
                        help="Maximum number of images to process (default: all)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Device args
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Data type for inference")

    return parser


@torch.no_grad()
def run_inference(
    model,
    vae,
    image_tokenizer,
    processor,
    input_dir: str,
    output_dir: str,
    cfg_scale: float,
    num_sampling_steps: int,
    flow_shift: float,
    max_condition_length: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
    num_images: Optional[int] = None,
):
    """Run DiT reconstruction inference.
    
    Creates comparison images showing: Original | VAE Reconstruction | DiT Reconstruction
    
    Args:
        model: DiT model
        vae: VAE model
        image_tokenizer: Image tokenizer
        processor: AutoProcessor for image preprocessing
        input_dir: Directory containing source images
        output_dir: Directory to save results
        cfg_scale: CFG scale for sampling
        num_sampling_steps: Number of Euler sampling steps
        flow_shift: Flow shift parameter
        max_condition_length: Maximum condition sequence length
        image_size: Image size
        device: Device to run on
        dtype: Data type
        seed: Random seed
        num_images: Maximum number of images to process
    """
    print(f"Running inference...")
    print(f"  Input dir: {input_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  Sampling steps: {num_sampling_steps}")
    print(f"  Flow shift: {flow_shift}")
    
    # Load and preprocess images
    result = load_visualization_images(
        image_dir=input_dir,
        processor=processor,
        image_size=image_size,
        max_condition_length=max_condition_length,
        device=device,
        dtype=dtype,
        num_images=num_images,
    )
    
    if result[0] is None:
        print("No images found, exiting...")
        return
    
    original_images, pixel_values, image_grid_thw, vae_input_images = result
    batch_size = len(original_images)
    print(f"  Processing {batch_size} images...")
    
    # 1. VAE Reconstruction: encode -> decode
    print("  VAE encoding...")
    latents = vae_encode(vae, vae_input_images)
    latent_channels = latents.shape[1]
    latent_size = latents.shape[2]
    
    print("  VAE decoding (reconstruction)...")
    vae_recon_latents = latents / vae.config.scaling_factor
    vae_recon_images = vae.decode(vae_recon_latents).sample
    vae_recon_images = (vae_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 2. Get condition embeddings from image tokenizer
    print("  Getting condition embeddings...")
    cond_embeds, cond_mask = tokenize_images(
        tokenizer=image_tokenizer,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        batch_size=batch_size,
        max_condition_length=max_condition_length,
    )
    
    # Prepare unconditional embeddings using model's null embedding for CFG
    null_embed = model.y_embedder.y_embedding  # [token_num, caption_channels]
    seq_len = min(null_embed.shape[0], max_condition_length)
    uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
    
    # Pad to max_condition_length if needed
    if seq_len < max_condition_length:
        padding = torch.zeros(
            batch_size, max_condition_length - seq_len, uncond_embeds.shape[-1],
            device=device, dtype=dtype
        )
        uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
    uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
    
    # Mask: mark the valid part of null embedding as 1
    uncond_mask = torch.zeros(batch_size, max_condition_length, device=device)
    uncond_mask[:, :seq_len] = 1
    uncond_mask = uncond_mask[:, None, None, :]  # [B, 1, 1, L]
    
    # 3. DiT sampling with Euler scheduler
    print(f"  Euler sampling ({num_sampling_steps} steps, cfg={cfg_scale})...")
    
    # Create Euler scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_sampling_steps, device=device)
    
    # Initialize with random noise
    generator = torch.Generator(device=device).manual_seed(seed)
    dit_latents = torch.randn(
        (batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    
    # Prepare CFG inputs
    cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
    mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)
    
    # Euler sampling loop
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_input = torch.cat([dit_latents] * 2)
        timestep = t.expand(latent_input.shape[0])
        
        # Model prediction
        noise_pred = model.forward_with_dpmsolver(
            latent_input, timestep, cond_embeds_cfg, mask=mask_cfg
        )
        
        # CFG combination
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        
        # Scheduler step
        dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]
    
    # Decode DiT latents to images
    print("  Decoding DiT latents...")
    dit_recon_latents = dit_latents / vae.config.scaling_factor
    dit_recon_images = vae.decode(dit_recon_latents).sample
    dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 4. Create comparison images and save
    print("  Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    vae_recon_np = vae_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    dit_recon_np = dit_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    
    for i, orig_img in enumerate(original_images):
        # Get reconstructed images
        vae_img = Image.fromarray((vae_recon_np[i] * 255).round().astype("uint8"))
        dit_img = Image.fromarray((dit_recon_np[i] * 255).round().astype("uint8"))
        
        # Create side-by-side comparison: Original | VAE | DiT
        comparison = Image.new('RGB', (image_size * 3, image_size))
        comparison.paste(orig_img, (0, 0))
        comparison.paste(vae_img, (image_size, 0))
        comparison.paste(dit_img, (image_size * 2, 0))
        
        # Save comparison image
        comparison.save(os.path.join(output_dir, f"comparison_{i}.png"))
        
        # Also save individual images
        orig_img.save(os.path.join(output_dir, f"original_{i}.png"))
        vae_img.save(os.path.join(output_dir, f"vae_recon_{i}.png"))
        dit_img.save(os.path.join(output_dir, f"dit_recon_{i}.png"))
    
    print(f"  Results saved to {output_dir}")
    print(f"  Processed {batch_size} images")


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    
    # Setup device and dtype
    device = torch.device(args.device)
    dtype = get_torch_dtype(args.dtype)
    
    print("=" * 60)
    print("Sana AE Inference")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"VAE: {args.vae_dir}")
    print(f"Image Tokenizer: {args.image_tokenizer_dir}")
    print(f"Device: {device}, Dtype: {dtype}")
    print("=" * 60)
    
    # Load DiT model
    print("Loading DiT model...")
    model = Model.from_pretrained(args.model_dir)
    model = model.to(device=device, dtype=dtype).eval()
    print(f"  Model loaded: {type(model).__name__}")
    
    # Load VAE
    print("Loading VAE...")
    vae = load_vae(args.vae_dir, device=device, dtype=dtype)
    
    # Load Image Tokenizer
    print("Loading Image Tokenizer...")
    image_tokenizer = load_image_tokenizer(
        args.image_tokenizer_dir, device=device, dtype=dtype
    )
    
    # Load processor for image preprocessing
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.image_tokenizer_dir, trust_remote_code=True)
    
    # Run inference
    run_inference(
        model=model,
        vae=vae,
        image_tokenizer=image_tokenizer,
        processor=processor,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        flow_shift=args.flow_shift,
        max_condition_length=args.max_condition_length,
        image_size=args.image_size,
        device=device,
        dtype=dtype,
        seed=args.seed,
        num_images=args.num_images,
    )
    
    print("=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
