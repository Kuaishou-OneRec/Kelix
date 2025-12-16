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
Sana DiT Training Script.

This script implements training for the Sana AE model,
the condition is from KeyeImageTokenizer.

"""

from typing import Dict, Any, Union, Optional, List
import os
import torch
import datetime
import contextlib
import argparse
import time
import json
import logging
import itertools
import gc
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.profiler import record_function
from transformers import AutoProcessor
from keye_vl_utils import process_vision_info

torch.autograd.set_detect_anomaly(True)
gc.disable()

process_group_timeout = datetime.timedelta(minutes=60*24)

# Muse imports
from muse.models import get_model_class, list_models
from muse.config import load_config
from muse.config.model_config import SanaConfig
from muse.training.distributed import (
    shard_model, 
    load_from_full_model_state_dict,
    initialize_model_params
)
from muse.training.checkpoint import (
    AppState, 
    DistributedCheckpointer,
    load_hf_checkpoint,
    get_checkpoint_path,
    save_checkpoint
)
from muse.training.common import (
    set_default_dtype,
    get_torch_dtype,
    clip_grad_by_value, 
    compute_fsdp_zero2_grad_norm
)
from muse.utils.common import Timer
from muse.training.lr_schedulers import get_scheduler
from muse.training.activations import set_activation_checkpointing
from muse.training.parallel import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_local_sequence,
    initialize_model_parallel,
    gather_by_group
)
from muse.utils.common import (
    set_random_seed, 
    print_rank_0,
    print_rank_n,
    to_cuda,
    dist_reduce_dict
)
from muse.data.datasets import Token2ImageDataset, MultiScaleDatasetWrapper
from muse.losses.diffusion import FlowMatchingLoss

from muse.utils.metrics import Logger, StdoutBackend, CSVBackend, TensorBoardBackend
from muse.training.common import initialize_metrics, StepScheduler
from muse.training.ema import EMAModel, ema_update


logger = logging.getLogger(__name__)


def parse_config_overrides(overrides: list) -> dict:
    """Parse config override strings into a dictionary.
    
    Args:
        overrides: List of strings in format "key=value"
        
    Returns:
        Dictionary of parsed overrides with appropriate types
        
    Example:
        >>> parse_config_overrides(["use_pe=true", "pe_interpolation=1.0"])
        {"use_pe": True, "pe_interpolation": 1.0}
    """
    result = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")
        
        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Parse value to appropriate type
        if value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        elif value.lower() == "none":
            result[key] = None
        else:
            # Try to parse as number
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                # Keep as string
                result[key] = value
    
    return result


def get_argument_parser():
    parser = argparse.ArgumentParser()

    ############ Model args ############
    parser.add_argument("--model-config", type=str, default=None,
                        help="The config file path of the model to train")

    parser.add_argument("--model-dir", type=str, default=None,
                      help="The directory of the pretrained model (required for continue training).")

    parser.add_argument("--model-config-overrides", type=str, nargs="*", default=[],
                        help="Override model config fields. Format: key=value. "
                             "Example: --model-config-overrides use_pe=true pe_interpolation=1.0")

    ############ VAE args ############
    parser.add_argument("--vae-dir", type=str,
                        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                        help="Pretrained VAE model path")

    ############ Image Tokenizer args ############
    parser.add_argument("--image-tokenizer-dir", type=str,
                        default=None,
                        help="Image tokenizer model name")
    
    parser.add_argument("--max-condition-length", type=int, default=324,
                        help="Maximum condition sequence length")

    ############ Dataset args ############
    parser.add_argument("--dataset-config", type=str, required=True,
                        help="The config file path of the dataset to train")

    parser.add_argument("--image-size", type=int, default=1024,
                        help="Training image size")
    
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per GPU")
    
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    
    parser.add_argument("--multi-scale", action="store_true",
                        help="Enable multi-scale training with variable aspect ratios")

    ############ Diffusion args ############
    parser.add_argument("--num-timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    
    parser.add_argument("--flow-shift", type=float, default=3.0,
                        help="Flow shift parameter")
    
    parser.add_argument("--weighting-scheme", type=str, default="logit_normal",
                        choices=["logit_normal", "mode", "uniform"],
                        help="Timestep sampling scheme")
    
    parser.add_argument("--logit-mean", type=float, default=0.0,
                        help="Mean for logit-normal sampling")
    
    parser.add_argument("--logit-std", type=float, default=1.0,
                        help="Std for logit-normal sampling")

    ############ Checkpoint args ############
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory to resume from")

    parser.add_argument("--checkpoint-id", type=str, default=None,
                        help="Checkpoint id to resume from")
    
    parser.add_argument("--save-checkpoint-per-step", type=int, default=1000,
                        help="Save checkpoint every N steps")

    parser.add_argument("--save-checkpoint-every-epoch", action="store_true",
                      help="Save checkpoint at the end of every epoch")

    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    
    parser.add_argument("--model-dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype")

    ############ FSDP Args ############
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload to CPU")

    parser.add_argument("--reshard-after-forward", action="store_true",
                        help="Reshard after forward (Zero3)")

    parser.add_argument("--prefetch-params-in-forward", action="store_true",
                        help="Prefetch parameters in forward pass.")

    parser.add_argument("--fp32-weight", action="store_true",
                        help="Use fp32 for model weight updating")

    parser.add_argument("--fp32-reduce", action="store_true",
                        help="Use fp32 for model gradient reduction")

    parser.add_argument("--allow-random-init-params", type=str, default='',
                        help="Parameter names to allow random initialization")
    
    parser.add_argument("--compile", action="store_true",
                        help="Compile model with torch.compile")

    ############ Optimizer & Learning Rate Args ############
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        help="Learning rate scheduler type")

    parser.add_argument("--num-warmup-steps", type=int, default=1000,
                        help="Number of warmup steps")
    
    parser.add_argument("--num-training-steps", type=int, default=100000,
                        help="Total number of training steps")

    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate")

    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Peak learning rate")

    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay")
    
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="AdamW beta1")

    parser.add_argument("--beta2", type=float, default=0.999,
                        help="AdamW beta2")
    
    parser.add_argument("--clip-range", type=float, default=1.0,
                        help="Gradient clipping range")

    ############ Training Args ############
    parser.add_argument("--enable-gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing")

    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")

    parser.add_argument("--logging-per-step", type=int, default=100,
                        help="Log every N steps")

    parser.add_argument("--comment", type=str, default="sana_training",
                        help="Experiment comment")

    parser.add_argument("--commit-id", type=str, default="dev",
                        help="Git commit id")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    ############ Visualization Args ############
    
    parser.add_argument("--cfg-scale", type=float, default=4.5,
                        help="CFG scale for validation sampling")
    
    parser.add_argument("--num-sampling-steps", type=int, default=20,
                        help="Number of sampling steps for validation")
    
    parser.add_argument("--visualize-dir", type=str, default=None,
                        help="Directory containing images for reconstruction visualization")
    
    parser.add_argument("--visualize-per-step", type=int, default=1000,
                        help="Visualize reconstruction every N steps")
    
    parser.add_argument("--num-vis-images", type=int, default=None,
                        help="Max number of images to visualize (default: all)")

    ############ Debug Args ############
    parser.add_argument("--enable-profile", action="store_true",
                        help="Enable torch profiler")

    parser.add_argument("--overfit-batches", type=int, default=None,
                        help="Number of batches to cache for overfitting (debug mode)")

    return parser


def load_vae(vae_dir: str, device: torch.device, dtype: torch.dtype):
    """Load VAE model from diffusers.
    
    Reference: Sana/diffusion/model/builder.py
    """
    from diffusers import AutoencoderDC
    
    print_rank_0(f"Loading VAE from {vae_dir}")
    vae = AutoencoderDC.from_pretrained(vae_dir, torch_dtype=dtype)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    
    return vae

def vae_encode(vae, images: torch.Tensor) -> torch.Tensor:
    """Encode images to latent space.
    
    Reference: Sana/diffusion/model/builder.py vae_encode for AutoencoderDC
    """
    with torch.no_grad():
        # VAE runs in float32 for precision, images should already be float32
        # Use indexing [0] which works for both tuple and EncoderOutput
        z = vae.encode(images)[0]
        z = z * vae.config.scaling_factor
    return z

def load_image_tokenizer(tokenizer_dir: str, device: torch.device, dtype: torch.dtype):
    from muse.models.keye_tokenizer import KeyeImageTokenizer
    with set_default_dtype(dtype), torch.device(device):
        tokenizer = KeyeImageTokenizer.from_pretrained(tokenizer_dir).eval()
        tokenizer.requires_grad_(False)

    return tokenizer

def tokenize_images(tokenizer,
                    pixel_values: torch.Tensor,
                    image_grid_thw: torch.Tensor,
                    batch_size: int,
                    max_condition_length: int) -> torch.Tensor:
    """Tokenize images.
    
    Args:
        tokenizer: Image tokenizer
        pixel_values: Pixel values tensor [num_total_patches, ...]
        image_grid_thw: Grid info tensor [B, 3] where each row is (t, h, w)
        batch_size: Batch size
        max_condition_length: Maximum condition sequence length for padding
    
    Returns:
        fused_embeddings: [B, max_condition_length, embed_dim]
        attention_mask: [B, 1, 1, max_condition_length]
    """
    with torch.no_grad():
        # List of tensor: [num_total_patches, embed_dim]
        embeddings: List[torch.Tensor] = tokenizer(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )["z_q"]
        
        # Sum embeddings across all codebooks
        fused_embeddings = torch.sum(torch.stack(embeddings, dim=1), dim=1)
        _, embed_dim = fused_embeddings.shape
        
        # Split by image_grid_thw and pad to max_condition_length
        # image_grid_thw: [B, 3] where each row is (t, h, w)
        # 4=merge_length
        lengths = (image_grid_thw[:, 1] * image_grid_thw[:, 2] // 4).tolist()  # h * w for each image
        
        # Split fused_embeddings according to lengths
        split_embeddings = torch.split(fused_embeddings, lengths, dim=0)
        
        # Pad each to max_condition_length and stack
        padded = []
        for emb in split_embeddings:
            seq_len = emb.shape[0]
            if seq_len < max_condition_length:
                padding = torch.zeros(max_condition_length - seq_len, embed_dim, 
                                      device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, padding], dim=0)
            else:
                emb = emb[:max_condition_length]
            padded.append(emb)
        
        fused_embeddings = torch.stack(padded, dim=0)  # [B, max_condition_length, embed_dim]

    # Create attention mask based on actual lengths
    attention_mask = torch.zeros(batch_size, max_condition_length, device=fused_embeddings.device)
    for i, length in enumerate(lengths):
        attention_mask[i, :min(length, max_condition_length)] = 1
    attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, max_condition_length]

    return fused_embeddings.unsqueeze(1), attention_mask


def load_visualization_images(
    image_dir: str,
    processor,
    image_size: int,
    max_condition_length: int,
    device: torch.device,
    dtype: torch.dtype,
    num_images: Optional[int] = None
) -> tuple:
    """Load and preprocess images from a directory for visualization.
    
    Args:
        image_dir: Directory containing images (jpg, png, jpeg)
        processor: AutoProcessor instance for image preprocessing
        image_size: Target image size
        max_condition_length: Maximum condition sequence length for image tokenizer
        device: Target device
        dtype: Target dtype
        num_images: Maximum number of images to load (None for all)
    
    Returns:
        Tuple of (original_images, pixel_values, image_grid_thw, vae_input_images)
        - original_images: List of PIL images (for visualization)
        - pixel_values: Tensor for image tokenizer
        - image_grid_thw: Grid info tensor for image tokenizer
        - vae_input_images: Tensor for VAE encoding [B, C, H, W] in [-1, 1]
    """
    from PIL import Image
    from torchvision import transforms
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []
    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in image_extensions:
            image_files.append(os.path.join(image_dir, f))
    
    if num_images is not None:
        image_files = image_files[:num_images]
    
    if not image_files:
        print_rank_0(f"Warning: No images found in {image_dir}")
        return None, None, None, None
    
    print_rank_0(f"Loading {len(image_files)} images from {image_dir}")
    
    # Load images
    original_images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        # Resize to target size
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        original_images.append(img)
    
    # Prepare messages format for processor (keye_vl_utils format)
    fake_messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
                "min_pixels": 4 * 28 * 28,
                "max_pixels": max_condition_length * 28 * 28
            } for img in original_images],
    }]
    text = processor.apply_chat_template(
        fake_messages,
        tokenize=False
    )
    # Process using keye_vl_utils
    image_inputs, _, _ = process_vision_info(fake_messages)
    
    # Use processor to get pixel_values and image_grid_thw
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    
    # Prepare images for VAE (normalize to [-1, 1])
    vae_transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    vae_input_images = torch.stack([vae_transform(img) for img in original_images])
    vae_input_images = vae_input_images.to(device=device, dtype=dtype)
    
    return original_images, pixel_values, image_grid_thw, vae_input_images


@torch.no_grad()
def visualize_reconstruction(
    model,
    vae,
    image_tokenizer,
    processor,
    image_dir: str,
    output_dir: str,
    global_step: int,
    cfg_scale: float,
    num_sampling_steps: int,
    flow_shift: float,
    max_condition_length: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    tb_writer=None,
    num_images: Optional[int] = None,
):
    """Visualize DiT reconstruction results.
    
    Creates comparison images showing: Original | VAE Reconstruction | DiT Reconstruction
    
    Args:
        model: DiT model
        vae: VAE model
        image_tokenizer: Image tokenizer
        processor: AutoProcessor for image preprocessing
        image_dir: Directory containing source images
        output_dir: Directory to save visualization results
        global_step: Current training step
        cfg_scale: CFG scale for sampling
        num_sampling_steps: Number of Euler sampling steps
        flow_shift: Flow shift parameter
        max_condition_length: Maximum condition sequence length
        image_size: Image size
        device: Device to run on
        dtype: Data type
        tb_writer: TensorBoard SummaryWriter (optional)
        num_images: Maximum number of images to visualize
    """
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    print_rank_0(f"[Step {global_step}] Running visualization...")
    
    # Load and preprocess images
    result = load_visualization_images(
        image_dir=image_dir,
        processor=processor,
        image_size=image_size,
        max_condition_length=max_condition_length,
        device=device,
        dtype=dtype,
        num_images=num_images,
    )
    
    if result[0] is None:
        print_rank_0("No images to visualize, skipping...")
        return
    
    original_images, pixel_values, image_grid_thw, vae_input_images = result
    batch_size = len(original_images)
    
    # 1. VAE Reconstruction: encode -> decode
    print_rank_0("  VAE encoding...")
    latents = vae_encode(vae, vae_input_images)
    latent_channels = latents.shape[1]
    latent_size = latents.shape[2]
    
    print_rank_0("  VAE decoding (reconstruction)...")
    vae_recon_latents = latents / vae.config.scaling_factor
    vae_recon_images = vae.decode(vae_recon_latents).sample
    vae_recon_images = (vae_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 2. Get condition embeddings from image tokenizer
    print_rank_0("  Getting condition embeddings...")
    cond_embeds, cond_mask = tokenize_images(
        tokenizer=image_tokenizer,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        batch_size=batch_size,
        max_condition_length=max_condition_length,
    )
    
    # Prepare unconditional embeddings using model's null embedding for CFG
    # Get the null embedding from model's y_embedder
    null_embed = model.y_embedder.y_embedding  # [token_num, caption_channels]
    # Truncate/pad to max_condition_length and expand to batch
    seq_len = min(null_embed.shape[0], max_condition_length)
    uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq_len, C]
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
    print_rank_0(f"  Euler sampling ({num_sampling_steps} steps, cfg={cfg_scale})...")
    
    # Create Euler scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_sampling_steps, device=device)
    
    # Initialize with random noise
    generator = torch.Generator(device=device).manual_seed(42)
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
    print_rank_0("  Decoding DiT latents...")
    dit_recon_latents = dit_latents / vae.config.scaling_factor
    dit_recon_images = vae.decode(dit_recon_latents).sample
    dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 4. Create comparison images and save
    print_rank_0("  Saving comparison images...")
    vis_dir = os.path.join(output_dir, "visualization", f"step_{global_step}")
    os.makedirs(vis_dir, exist_ok=True)
    
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
        
        # Save to file
        comparison.save(os.path.join(vis_dir, f"comparison_{i}.png"))
    
    # 5. Write to TensorBoard
    if tb_writer is not None:
        # Create a grid of all comparisons
        all_images = []
        for i, orig_img in enumerate(original_images):
            # Original
            orig_tensor = torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).float() / 255.0
            all_images.append(orig_tensor)
            # VAE reconstruction
            all_images.append(vae_recon_images[i].cpu().float())
            # DiT reconstruction  
            all_images.append(dit_recon_images[i].cpu().float())
        
        # Stack and add to tensorboard
        grid = torch.stack(all_images)  # [N*3, C, H, W]
        from torchvision.utils import make_grid
        grid_img = make_grid(grid, nrow=3, padding=2)  # 3 images per row
        tb_writer.add_image(f"visualization/reconstruction", grid_img, global_step)
    
    print_rank_0(f"  Visualization saved to {vis_dir}")


def _init_profiler(output_dir, with_stack=False) -> None:
    """Initialize torch profiler with TensorBoard support.
    
    Args:
        output_dir: Directory to save profiler output
        with_stack: Whether to record Python call stacks (slower but more detailed)
    
    Returns:
        Configured torch profiler instance
    """
    profile_dir = os.path.join(output_dir, "torch_profile")
    if dist.get_rank() == 0:
        os.makedirs(profile_dir, exist_ok=True)
    dist.barrier()  # Ensure directory is created before other ranks proceed

    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=5,       # Skip first 5 steps (initial warmup)
            warmup=2,     # 2 steps for profiler warmup
            active=10,    # Record 10 steps
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    )
    return torch_profiler


def train():
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    assert all([args.commit_id, args.seed, args.comment]), \
        "Git commit, seed, and comment is required for reproducibility"

    assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
        "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
        "save_checkpoint_every_epoch should be set."

    # TODO: move to muse.training.distributed
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        rank=rank, world_size=world_size,
        timeout=process_group_timeout
    )
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))

    ### initialize model parallel group
    initialize_model_parallel()
    print_rank_0(f"Data parallel size: {get_data_parallel_world_size()}")

    # Use rank-specific seed for training
    # This ensures different noise/timesteps across ranks while maintaining reproducibility
    # Model weights are loaded from checkpoint, not randomly initialized, so this is safe
    training_seed = args.seed + rank
    set_random_seed(training_seed)
    print_rank_0(f"Random seed: base={args.seed}, training_seed={training_seed} (rank={rank})")


    if dist.get_rank() == 0:
        args_str = json.dumps(vars(args), indent=2, ensure_ascii=False)
        print_rank_0(f"Training Arguments:\n{args_str}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(os.path.join(args.output_dir,
                f"args-{args.commit_id}-{timestamp}.json"), 'w',
                encoding="utf-8") as f:
            f.write(args_str + "\n")

    # TODO: support wandb
    tb_writer = None
    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
        tb_writer.add_text("comment", args.comment, 0)
        tb_writer.add_text("comment_id", args.commit_id, 0)
    
    # Setup logging
    if rank == 0:
        stdout_logger = Logger("stdout", [StdoutBackend()])
        csv_logger = Logger("csv", [CSVBackend(os.path.join(args.output_dir, "metrics.csv"))])
        tb_logger = Logger("tb", [TensorBoardBackend(args.output_dir)])
        loggers = [stdout_logger, csv_logger, tb_logger]
    else:
        loggers = []

    metrics = initialize_metrics(
        acc_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step,
        loggers=loggers
    )

    # Determine training mode and get model_class
    if args.model_dir:
        # Continue pretrain mode: get model_class from model_dir/config.json
        model_config_path = Path(args.model_dir) / "config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {model_config_path}. "
                f"Cannot continue pretrain without config.json in {args.model_dir}"
            )
        model_config = load_config(model_config_path)
    elif args.model_config:
        # Train from scratch mode: get model_class from model_config
        model_config = load_config(args.model_config)
    else:
        raise ValueError(
            "Either --model-dir (for continue pretrain) or --model-config "
            "(for train from scratch) must be provided.")
    
    # Apply model config overrides from command line
    # caption_channels = 128
    if args.model_config_overrides:
        overrides = parse_config_overrides(args.model_config_overrides)
        print_rank_0(f"Applying model config overrides: {overrides}")
        for key, value in overrides.items():
            if hasattr(model_config, key):
                old_value = getattr(model_config, key)
                setattr(model_config, key, value)
                print_rank_0(f"  {key}: {old_value} -> {value}")
            else:
                raise ValueError(f"Unknown model config field: {key}")
    
    model_class_name = model_config.model_class
    # Get model class from registry
    print_rank_0(f"Available models: {list_models()}")
    print_rank_0(f"Loading model class: {model_class_name}")
    
    try:
        model_cls = get_model_class(model_class_name)
        print_rank_0(f"Get model class: {model_cls.__name__}")
    except KeyError:
        print_rank_0(
        f"Unavailable model: {model_class_name}, " \
        f"please choose from available models: {list_models()}")
        return

    # Load state dict and convert using model's converter (only for continue pretrain)
    state_dict = None
    
    # Load state_dict to CPU only on rank 0 to avoid CPU OOM
    if args.model_dir:
        # Continue pretrain: load weights from checkpoint
        if dist.get_rank() == 0:
            with set_default_dtype(args.model_dtype):
                print_rank_0(f"Loading checkpoint from: {args.model_dir}")
                state_dict = load_hf_checkpoint(args.model_dir)
        dist.barrier()
    else:
        # Train from scratch: no weights to load
        state_dict = None
        dist.barrier()

    # Create model
    with set_default_dtype(args.model_dtype), torch.device("meta"):
        print_rank_0(f"Creating model from config")
        model = model_cls(model_config)
        print_rank_0(f"Model instantiated: {type(model).__name__}")

    if args.enable_gradient_checkpointing:
        print_rank_0("Enabling gradient checkpointing")
        set_activation_checkpointing(
            model, auto_wrap_policy=model.get_checkpointable_module_classes()
        )

    # upcast fp32 to maintain master weight.
    # We need to save a fp32 model weight, otherwise the precision of the optimizer 
    # updating the weight will be reduced, affecting convergence
    if args.fp32_weight:
        model = model.float()

    # Shard model for distributed training
    shard_model(
        model=model,
        cpu_offload=args.cpu_offload,
        reshard_after_forward=args.reshard_after_forward,
        dp_mesh=device_mesh,
        fp32_weight=args.fp32_weight,
        prefetch_params_in_forward=args.prefetch_params_in_forward,
        fp32_reduce=args.fp32_reduce
    )
    dist.barrier()
    # 需要保证每个rank都执行了参数初始化或加载
    if args.model_dir:
        with Timer("Load state dict"):
            # Convert meta tensors to CUDA tensors
            # distribute the state_dict from rank 0 to all ranks
            load_from_full_model_state_dict(
                model=model, full_sd=state_dict,
                allow_random_init_params=args.allow_random_init_params
            )
    else:
        # Train from scratch: initialize model parameters randomly
        with Timer("Initialize model parameters"):
            initialize_model_params(model)

    with torch.device(torch.cuda.current_device()):
        # Initialize RoPE, if the buffer is not in the state_dict,
        # it still on meta device, so we need to initialize it here
        for m in model.modules():
            # RoPE is not covered in state dict
            if hasattr(m, "rope_init"):
                print_rank_0("Initialize RoPE")
                m.rope_init()

    # Check if all parameters & buffers are initialized
    for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        assert tensor.device != torch.device("meta"), \
        f"{name} not initialized, device={tensor.device}"

    if args.compile:
        # Compile model for better performance
        model = torch.compile(model)
        print_rank_0("Model compiled")

    if state_dict is not None:
        # Free the state_dict to save memory
        del state_dict

    # Print trainable parameters
    print_rank_0("=" * 50)
    print_rank_0("Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank_0(f"  {name}: {param.shape}")
        else:
            print_rank_0(f"  {name}: {param.shape} (not trainable)")
    print_rank_0("=" * 50)

    ############## Load VAE and text encoder ##############
    # VAE & Text Encoder is not trainable
    vae = None

    # VAE uses bfloat16 to match model compute dtype
    vae = load_vae(
        vae_dir=args.vae_dir,
        device=torch.cuda.current_device(),
        dtype=get_torch_dtype(args.model_dtype)
    )
    image_tokenizer = load_image_tokenizer(
        tokenizer_dir=args.image_tokenizer_dir,
        device=torch.cuda.current_device(),
        dtype=args.model_dtype
    )
    
    # Load processor for visualization (only needed if visualize_dir is set)
    vis_processor = None
    if args.visualize_dir:
        vis_processor = AutoProcessor.from_pretrained(
            args.image_tokenizer_dir,
            trust_remote_code=True
        )
        print_rank_0(f"Loaded processor for visualization from {args.image_tokenizer_dir}")
    ############## Load VAE and tokenizer ##############

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.get_optimizer_grouped_parameters(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        ),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=1e-8
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        min_lr=args.min_lr
    )

    # Create loss function
    loss_fn = FlowMatchingLoss(
        num_timesteps=args.num_timesteps,
        flow_shift=args.flow_shift,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        pred_sigma=model_config.pred_sigma if hasattr(model_config, 'pred_sigma') else False,
    )

    # Setup checkpointing
    app_state = AppState(model=model, optimizer=optimizer)
    dist_checkpointer = DistributedCheckpointer()

    if args.checkpoint_dir:
        print_rank_0(f"Resuming from checkpoint: {args.checkpoint_dir}")
        state_dict = {"app": app_state}
        checkpoint_path = get_checkpoint_path(args.checkpoint_dir, args.checkpoint_id)
        dist_checkpointer.load_checkpoint(
            state_dict=state_dict,
            checkpoint_path=checkpoint_path,
        )
        print_rank_0("Checkpoint loaded successfully")

    ############## Prepare dataset config ##############
    with open(args.dataset_config, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())

    dataset = dataset_config.pop("name")

    model_class_name = model_config.model_class
    dataset_config["model_class"] = model_class_name
    
    ## Overwrite dataset config
    if args.image_size:
        dataset_config["image_size"] = args.image_size
        print_rank_0(f"Set image_size of dataset to: {args.image_size}")
    
    # Set tokenizer_path from model_dir if not specified
    if not dataset_config.get("processor_path") and args.image_tokenizer_dir:
        dataset_config["processor_path"] = args.image_tokenizer_dir
        print_rank_0(f"Set tokenizer_path of dataset to: {args.image_tokenizer_dir}")
    
    # Enable multi-scale training if requested
    if args.multi_scale:
        dataset_config["multi_scale"] = True
        print_rank_0("Multi-scale training enabled with variable aspect ratios")

    # Add distributed rank/world_size to dataset config for proper data sharding
    if dist.is_initialized():
        dataset_config["rank"] = dist.get_rank()
        dataset_config["world_size"] = dist.get_world_size()
        print_rank_0(f"Dataset sharding: rank={dataset_config['rank']}, world_size={dataset_config['world_size']}")

    print_rank_0(f"Building dataset with config: {dataset_config}")
    dataset = Token2ImageDataset(**dataset_config)
    collate_fn = dataset.collate_fn
    if args.multi_scale:
        dataset = MultiScaleDatasetWrapper(
            dataset=dataset,
            batch_size=args.batch_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            collate_fn=lambda x: collate_fn(x[0])
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )

    # Training loop
    print_rank_0("Starting training...")
    model.train()

    # Setup profiler
    torch_profiler = _init_profiler(
        output_dir=args.output_dir
    ) if args.enable_profile else None

    # Initialize step scheduler for training loop management
    scheduler = StepScheduler(args)

    # Setup data iterator
    if dataloader is not None:
        if args.overfit_batches:
            # Overfit debug mode: cache n batches and cycle through them
            print_rank_0(f"=== OVERFIT DEBUG MODE: Caching {args.overfit_batches} batches ===")
            print_rank_0(f"Checkpoint saving will be disabled in overfit mode")
            cached_batches = []
            temp_iter = iter(gather_by_group(dataloader, get_context_parallel_group()))
            for i in range(args.overfit_batches):
                try:
                    batch = next(temp_iter)
                    cached_batches.append(batch)
                except StopIteration:
                    print_rank_0(f"Warning: Only {i} batches available, less than requested {args.overfit_batches}")
                    break
            print_rank_0(f"Successfully cached {len(cached_batches)} batches for overfitting")
            print_rank_0(f"Model will cycle through these batches indefinitely")
            # Create infinite iterator that cycles through cached batches
            data_iter = iter(itertools.cycle(cached_batches))
        else:
            # Normal mode: use dataloader as-is
            data_iter = iter(gather_by_group(dataloader, get_context_parallel_group()))
    else:
        print_rank_0("Warning: No dataloader available. Training loop will not run.")
        data_iter = iter([])

    # Step 0 visualization: show model state before any optimization
    if args.visualize_dir and dist.get_rank() == 0:
        print_rank_0("Running step 0 visualization (before training)...")
        model.eval()
        with Timer("visualization step 0"):
            visualize_reconstruction(
                model=model,
                vae=vae,
                image_tokenizer=image_tokenizer,
                processor=vis_processor,
                image_dir=args.visualize_dir,
                output_dir=args.output_dir,
                global_step=0,
                cfg_scale=args.cfg_scale,
                num_sampling_steps=args.num_sampling_steps,
                flow_shift=args.flow_shift,
                max_condition_length=args.max_condition_length,
                image_size=args.image_size,
                device=torch.cuda.current_device(),
                dtype=get_torch_dtype(args.model_dtype),
                tb_writer=tb_writer,
                num_images=args.num_vis_images,
            )
        model.train()

    if args.visualize_dir:
        dist.barrier()  # Sync all ranks after step 0 visualization

    while scheduler.global_step < args.num_training_steps:
        with contextlib.ExitStack() as ctx:
            if torch_profiler:
                ctx.enter_context(torch_profiler)
            
            # 1. DataLoader
            with record_function("DataLoader"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

            # 2. Data Transfer to GPU
            with record_function("DataTransfer"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(
                            device=torch.cuda.current_device(),
                            dtype=get_torch_dtype(args.model_dtype) if v.is_floating_point() else None
                        )

            scheduler.step()

            # 3. VAE Encode (get latents)
            with record_function("VAE_Encode"):
                if "latents" in batch:
                    latents = batch["latents"]
                elif "image" in batch and vae is not None:
                    latents = vae_encode(vae, batch["image"])
                else:
                    raise ValueError("No latents or images in batch")

            # 4. Text Encoder
            with record_function("TextEncoder"):
                token_embeds, attention_mask = tokenize_images(
                    image_tokenizer,
                    batch["pixel_values"],
                    batch["image_grid_thw"],
                    args.batch_size,
                    args.max_condition_length,
                )

            # 5. Forward + Loss Computation
            with record_function("Forward_Loss"):
                loss_dict = loss_fn(
                    model=model,
                    x_start=latents,
                    y=token_embeds,
                    mask=attention_mask,
                )
                loss = loss_dict["loss"]

            # Pass detached tensor directly - .item() will be called in metrics.step()
            # to avoid CPU-GPU sync during the training hot path
            metrics.loss.append(loss.detach())

            # 6. Backward Pass
            with record_function("Backward"):
                loss.backward()

            # 7. Gradient Clipping
            with record_function("GradClip"):
                clip_grad_by_value(model, args.clip_range)

            # Update optimizer at gradient accumulation boundaries
            if scheduler.is_gradient_accumulation_boundary():
                # 8. Gradient Norm Computation
                with record_function("GradNorm"):
                    grad_norm = compute_fsdp_zero2_grad_norm(model)
                metrics.grad_norm.append(grad_norm)
                learning_rate = lr_scheduler.get_last_lr()[0]
                metrics.learning_rate.append(learning_rate)

                # 9. Optimizer Step
                with record_function("OptimizerStep"):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            metrics.step_time.tick()
            metrics.step()

            # Logging
            if scheduler.should_logging():
                metrics.write_logs(scheduler.global_step)

            # Save checkpoint
            if scheduler.should_save_checkpoint():
                torch.cuda.empty_cache()
                gc.collect()
                with Timer("save checkpoint"):
                    save_checkpoint(
                        app_state=app_state,
                        dist_checkpointer=dist_checkpointer,
                        checkpoint_dir=args.output_dir,
                        global_step=scheduler.global_step
                    )

            # Visualization (only on rank 0)
            if (args.visualize_dir and 
                scheduler.global_step > 0 and 
                scheduler.global_step % args.visualize_per_step == 0 and
                dist.get_rank() == 0):
                model.eval()
                with Timer("visualization"):
                    visualize_reconstruction(
                        model=model,
                        vae=vae,
                        image_tokenizer=image_tokenizer,
                        processor=vis_processor,
                        image_dir=args.visualize_dir,
                        output_dir=args.output_dir,
                        global_step=scheduler.global_step,
                        cfg_scale=args.cfg_scale,
                        num_sampling_steps=args.num_sampling_steps,
                        flow_shift=args.flow_shift,
                        max_condition_length=args.max_condition_length,
                        image_size=args.image_size,
                        device=torch.cuda.current_device(),
                        dtype=get_torch_dtype(args.model_dtype),
                        tb_writer=tb_writer,
                        num_images=args.num_vis_images,
                    )
                model.train()
                dist.barrier()  # Sync after visualization

            if torch_profiler:
                torch_profiler.step()


    # Save final checkpoint
    print_rank_0("Training completed. Saving final checkpoint...")
    save_checkpoint(
        app_state=app_state,
        dist_checkpointer=dist_checkpointer,
        checkpoint_dir=args.output_dir,
        global_step=scheduler.global_step
    )
    print_rank_0("Done!")

if __name__ == "__main__":
    train()
