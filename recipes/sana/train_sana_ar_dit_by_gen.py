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

from typing import Dict, Any, Union, Optional, List, Tuple
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
from recipes.sana.utils import (
    compute_input_pos, load_vae, vae_encode
)

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
    compute_fsdp_zero2_grad_norm,
    freeze_params_by_pattern
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
    dist_reduce_dict,
    parse_config_overrides
)
from muse.data.datasets import (
    Token2ImageDataset, 
    MultiScaleDatasetWrapper,
    Chat2ImageDataset
)
from muse.data.utils import (
    parse_resolution_budgets, 
    DEFAULT_RESOLUTION_BUDGETS,
    ResolutionBudget,
    ResolutionBudgetConfig,
)
from muse.losses.diffusion import FlowMatchingLoss

from muse.utils.metrics import Logger, StdoutBackend, CSVBackend, TensorBoardBackend
from muse.training.common import initialize_metrics, StepScheduler
from muse.training.ema import EMAModel, ema_update


logger = logging.getLogger(__name__)


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
    parser.add_argument("--keye-ar-dir", type=str,
                        default=None,
                        help="keye ar model name")

    parser.add_argument("--max-condition-length", type=int, default=324,
                        help="Maximum condition sequence length")
    
    parser.add_argument("--cond-pos-scale", type=float, default=1.0,
                        help="Scale factor for condition position embeddings")

    parser.add_argument("--condition-on-special-tokens", action="store_true",
                        help="Condition on special tokens")

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
                        help="Enable multi-scale training with variable aspect ratios. "
                             "Use --resolution-budgets for curriculum scheduling.")

    parser.add_argument("--resolution-budgets", type=str, default=None,
                        help="Resolution budgets as 'size:batch_size,...' "
                             "Example: '512:32,768:16,1024:8'")
    
    parser.add_argument("--max-seq-length", type=int, default=24000,
                        help="Maximum sequence length for training")

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
    
    parser.add_argument("--skip-load-params", type=str, default='',
                        help="Parameter name patterns to skip loading from checkpoint (comma-separated). "
                             "Uses 'contains' matching. These params will be randomly initialized. "
                             "E.g., 'y_embedder,cross_attn' skips all params containing these substrings")
    
    parser.add_argument("--freeze-params", type=str, default='',
                        help="Parameter name patterns to freeze (comma-separated). Uses 'contains' matching. "
                             "E.g., 'y_embedder,cross_attn' freezes all params containing these substrings. "
                             "Use ^ prefix for inverse: '^y_embedder,^cross_attn' freezes all EXCEPT matching params")
    
    parser.add_argument("--compile", action="store_true",
                        help="Compile model with torch.compile")

    ############ Optimizer & Learning Rate Args ############
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        help="Learning rate scheduler type")

    parser.add_argument("--num-warmup-steps", type=int, default=1000,
                        help="Number of warmup steps")
    
    parser.add_argument("--num-training-steps", type=int, default=100000,
                        help="Total number of training steps")

    parser.add_argument("--num-decay-steps", type=int, default=4000,
                        help="Number of steps for learning rate decay")

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
                        
    parser.add_argument("--im-token-generation-length", type=int, default=None,
                        help="Generation length for image tokens, if None, use max_condition_length")

    ############ Visualization Args ############
    
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="CFG scale for validation sampling")
    
    parser.add_argument("--num-sampling-steps", type=int, default=20,
                        help="Number of sampling steps for validation")
    
    parser.add_argument("--visualize-parquet-path", type=str, default=None,
                        help="Parquet file path containing images for reconstruction visualization")
    
    parser.add_argument("--visualize-per-step", type=int, default=1000,
                        help="Visualize reconstruction every N steps")
    
    parser.add_argument("--num-vis-images", type=int, default=None,
                        help="Max number of images to visualize (default: all)")

    ############ Debug Args ############
    parser.add_argument("--enable-profile", action="store_true",
                        help="Enable torch profiler")

    parser.add_argument("--overfit-batches", type=int, default=None,
                        help="Number of batches to cache for overfitting (debug mode)")
    
    parser.add_argument("--run_data_iter", action="store_true", help="Run data iterator")

    return parser


class VisReconstructionLoader:
    """Visualize DiT reconstruction results."""
    loaded = None
    
    @classmethod
    def __call__(cls,
                 parquet_path: str,
                 dataset,
                 image_size: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 num_images: Optional[int] = None,
                 tb_writer=None,
                 vae=None
                 ):
        if cls.loaded: return cls.loaded

        # Load and preprocess images from parquet file
        result = load_visualization_images(
            parquet_path=parquet_path,  # 改为parquet_path
            dataset=dataset,  # 传入dataset用于处理方法
            image_size=image_size,
            device=device,
            dtype=dtype,
            num_images=num_images,
        )
        
        texts, original_images, pixel_values, image_grid_thw, vae_input_images, input_ids = result
        batch_size = len(original_images)
        print_rank_0(f"Loaded {len(original_images)} images for visualization, pixel_values shape: {pixel_values.shape}")
        # Add text information to TensorBoard
        if tb_writer is not None and texts:
            for i, text in enumerate(texts):
                # Truncate text if too long for TensorBoard display
                truncated_text = text[:200] + "..." if len(text) > 200 else text
                tb_writer.add_text(tag=f"visualization/text_sample_{i}", text_string=truncated_text, global_step=0)

        # 1. VAE Reconstruction: encode -> decode
        print_rank_0("  VAE encoding...")
        latents = vae_encode(vae, vae_input_images)
        latent_channels = latents.shape[1]
        latent_size = latents.shape[2]
        
        print_rank_0(f"  VAE decoding (reconstruction)...\nlatent_channels={latent_channels}, latent_size={latent_size}")
        vae_recon_latents = latents / vae.config.scaling_factor
        vae_recon_images = vae.decode(vae_recon_latents).sample
        vae_recon_images = (vae_recon_images / 2 + 0.5).clamp(0, 1)

        import easydict
        cls.loaded = easydict.EasyDict(
            texts=texts,
            original_images=original_images,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vae_input_images=vae_input_images,
            input_ids=input_ids,
            batch_size=batch_size,
            latents=latents,
            latent_channels=latent_channels,
            latent_size=latent_size,
            vae_recon_images=vae_recon_images,


        )
        return cls.loaded

def load_keye_ar(tokenizer_dir: str, device: torch.device, dtype: torch.dtype, output_last_hidden_states_only=True):

    from muse.models.keye_ar import KeyeARModel
    with set_default_dtype(dtype), torch.device(device):
        tokenizer = KeyeARModel.from_pretrained(tokenizer_dir).eval()
        if torch.distributed.get_rank() == 0:
            print(f"tokenizer={tokenizer}")
        tokenizer.config.qwen_config.output_last_hidden_states_only = output_last_hidden_states_only
        tokenizer.model.model.output_last_hidden_states_only = output_last_hidden_states_only
        tokenizer.requires_grad_(False)

    return tokenizer

def tokenize_images(tokenizer,
                    pixel_values: torch.Tensor,
                    image_grid_thw: torch.Tensor,
                    batch_size: int,
                    max_condition_length: int,
                    input_ids: Optional[torch.Tensor] = None,
                    cu_seqlens: Optional[torch.Tensor] = None,
                    cond_embeds_op = None,
                    condition_on_special_tokens: bool = False,
                    ar_processor = None,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    from recipes.sana.inference_ar2image import tokenize_images as tokenize_images_ar2image
    cond_embeds = []
    cond_mask = []
    token_embed_lengths = []
    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        input_ids_sample = input_ids[i:i + 1]
        per_sample_cond_embeds, per_sample_cond_mask, per_sample_token_embed_lengths = tokenize_images_ar2image(
            ar_model=tokenizer,
            ar_processor=ar_processor,
            batch_size=batch_size,
            max_condition_length=max_condition_length,
            input_ids=input_ids_sample,
            condition_on_special_tokens=condition_on_special_tokens,
            teacher_forcing=False,
        )
        cond_embeds.append(per_sample_cond_embeds)
        cond_mask.append(per_sample_cond_mask)
        token_embed_lengths.append(per_sample_token_embed_lengths)

    cond_embeds = torch.cat(cond_embeds, dim=0)
    cond_mask = torch.cat(cond_mask, dim=0)
    print(f"cond_mask000={cond_mask.shape}")
    token_embed_lengths = sum(token_embed_lengths, [])
    embed_dim = cond_embeds.shape[2]

    # Handle padding to max_condition_length
    current_seq_len = cond_embeds.shape[1]
    if current_seq_len < max_condition_length:
        # Pad to max_condition_length
        padding_embeddings = torch.zeros(batch_size, max_condition_length - current_seq_len, embed_dim,
                                        device=cond_embeds.device, dtype=cond_embeds.dtype)
        cond_embeds = torch.cat([cond_embeds, padding_embeddings], dim=1)
        
        # Extend attention mask with zeros for padding
        padding_mask = torch.zeros(batch_size, max_condition_length - current_seq_len,
                                    device=cond_mask.device, dtype=cond_mask.dtype)
        cond_mask = torch.cat([cond_mask, padding_mask], dim=1)
    elif current_seq_len > max_condition_length:
        # Truncate to max_condition_length
        cond_embeds = cond_embeds[:, :max_condition_length, :]
        cond_mask = cond_mask[:, :max_condition_length]

    # Reshape attention_mask to [B, 1, 1, max_condition_length]
    cond_mask = cond_mask[:, None, None, :]
    
    if cond_embeds_op is not None:
        cond_embeds = cond_embeds_op(cond_embeds)

    max_seq_len = max_condition_length
    print(f"after tokenizetion: cond_embeds={cond_embeds.shape}, cond_mask={cond_mask.shape}, max_seq_len={max_seq_len}, token_embed_lengths={token_embed_lengths}")
    return cond_embeds, cond_mask, max_seq_len, token_embed_lengths
    
def load_visualization_images(
    parquet_path: str,  # 改为接收parquet_path参数
    dataset,  # 保留dataset参数用于处理方法
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_images: Optional[int] = None
) -> tuple:
    """Load and preprocess images from parquet file for visualization.
    
    Args:
        parquet_path: Path to parquet file containing samples
        dataset: Chat2ImageDataset instance for processing methods
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
    import pandas as pd

    # Read parquet file
    df = pd.read_parquet(parquet_path)
    if num_images is not None:
        df = df.head(num_images)
    
    
    print(f"Loading {len(df)} samples from {parquet_path}")
    
    # Process samples using dataset's process method
    processed_samples = []
    texts = []
    original_images = []
    for _, row in df.iterrows():
        # Convert parquet row to sample format expected by dataset
        sample = row.to_dict()
        # messages=[{'role': 'user', 'content': [{'type': 'text', 'text': '这是第0张图像的描述'}]}, {'role': 'assistant', 'content': [{'type': 'image', 'image': '/tmp/tmpmah5htt0/images/image_0.jpg'}]}]
        # Use dataset's process method
        processed_sample = dataset.process(sample, valid_hw_range=(0,10000))
        if processed_sample is None: continue
        processed_samples.append(processed_sample)
        text = processed_sample["text"]
        texts.append(text)

        img = Image.open(processed_sample["image"]).convert('RGB')
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        original_images.append(img)
    

    # Use dataset's collate_fn to batch the samples
    batch = dataset.collate_fn(processed_samples)
    
    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device=device)
    
    # Prepare images for VAE (normalize to [-1, 1]) - BASELINE LOGIC
    vae_transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    vae_input_images = torch.stack([vae_transform(img) for img in original_images])
    vae_input_images = vae_input_images.to(device=device, dtype=dtype)
    return texts, original_images, pixel_values, image_grid_thw, vae_input_images, batch["input_ids"]


@torch.no_grad()
def visualize_reconstruction(
    model,
    vae,
    image_tokenizer,
    parquet_path: str,  # 改为parquet_path参数
    dataset,  # 保留dataset参数用于处理方法
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
    args = None,
):
    """Visualize DiT reconstruction results.
    
    Creates comparison images showing: Original | VAE Reconstruction | DiT Reconstruction
    
    Args:
        model: DiT model
        vae: VAE model
        image_tokenizer: Image tokenizer
        processor: AutoProcessor for image preprocessing
        parquet_path: Path to parquet file containing samples (changed from image_dir)
        dataset: Chat2ImageDataset instance for processing methods
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
    import time
    
    t0 = time.time()
    loaded = VisReconstructionLoader()(
               parquet_path,
               dataset,
               image_size,
               device,
               dtype,
               num_images,
               tb_writer,
               vae
          )
    # 2. Get condition embeddings from image tokenizer
    print_rank_0("  Getting condition embeddings...")
    cond_embeds, cond_mask, max_seq_len, token_embed_lengths = tokenize_images(  # pyright: ignore[reportAssignmentType]
        tokenizer=image_tokenizer,
        pixel_values=loaded.pixel_values.to(device=device),
        image_grid_thw=loaded.image_grid_thw.to(device=device),
        batch_size=loaded.batch_size,
        max_condition_length=max_condition_length,
        input_ids=loaded.input_ids.to(device=device),
        cond_embeds_op=model.diffusion_connector,
        condition_on_special_tokens=args.condition_on_special_tokens,
        ar_processor=dataset.processor,
    )
    
    # Prepare unconditional embeddings using model's null embedding for CFG
    # Get the null embedding from model's y_embedder
    null_embed = model.y_embedder.y_embedding  # [token_num, caption_channels]
    # Truncate/pad to max_condition_length and expand to batch
    seq_len = min(null_embed.shape[0], max_condition_length)
    uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(loaded.batch_size, -1, -1)  # [B, seq_len, C]
    # Pad to max_condition_length if needed
    if seq_len < max_condition_length:
        padding = torch.zeros(
            loaded.batch_size, max_condition_length - seq_len, uncond_embeds.shape[-1],
            device=device, dtype=dtype
        )
        uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
    uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
    # Mask: mark the valid part of null embedding as 1
    uncond_mask = torch.zeros(loaded.batch_size, max_condition_length, device=device)
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
        (loaded.batch_size, loaded.latent_channels, loaded.latent_size, loaded.latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    print(f"uncond_mask={uncond_mask.shape}, cond_mask={cond_mask.shape}")
    # Prepare CFG inputs
    cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
    mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)
    
    pos_args = compute_pos_args(
        latent_hw=(loaded.latent_size, loaded.latent_size), 
        image_grid_thw=torch.tensor([1, 2 * args.im_token_generation_length**0.5, 2*args.im_token_generation_length**0.5])[None], 
        max_seq_len=args.max_condition_length, 
        device=device, 
        cond_pos_scale=args.cond_pos_scale,
        image_size=args.image_size,
        token_embed_lengths=token_embed_lengths,
        )
    
    model_kwargs={
        **pos_args,
        "is_y_connected": True,
    }

    # Euler sampling loop
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_input = torch.cat([dit_latents] * 2)
        timestep = t.expand(latent_input.shape[0])
        
        # Model prediction
        noise_pred = model.forward_with_dpmsolver(
            latent_input, timestep, cond_embeds_cfg, mask=mask_cfg, **model_kwargs
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
    vae_recon_np = loaded.vae_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    dit_recon_np = dit_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    
    for i, orig_img in enumerate(loaded.original_images):
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
        for i, orig_img in enumerate(loaded.original_images):
            # Original
            orig_tensor = torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).float() / 255.0
            all_images.append(orig_tensor)
            # VAE reconstruction
            all_images.append(loaded.vae_recon_images[i].cpu().float())
            # DiT reconstruction  
            all_images.append(dit_recon_images[i].cpu().float())
        
        # Stack and add to tensorboard
        grid = torch.stack(all_images)  # [N*3, C, H, W]
        from torchvision.utils import make_grid
        grid_img = make_grid(grid, nrow=3, padding=2)  # 3 images per row
        tb_writer.add_image("visualization/comparison_grid", grid_img, global_step)
        
        # Add text information to TensorBoard (already done above)
        print_rank_0(f"  Added {len(loaded.texts)} text samples to TensorBoard")
    print_rank_0(f"  Visualization time: {time.time() - t0:.4f}s")


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


def resize_hw(hw, max_tokens):
    import keye_vl_utils
    return torch.tensor(keye_vl_utils.smart_resize(*hw.tolist(), factor=1, min_pixels=1, max_pixels=max_tokens))


'''
latent_hw=(32, 32), image_grid_thw=tensor([[ 1., 36., 36.]]), maxseq_len=324
args={'x_input_pos': {'height': tensor([ 0,  0,  0,  ..., 31, 31, 31], device='cuda:0'), 'width': tensor([ 0,  1,  2,  ..., 29, 30, 31], device='cuda:0')}, 'cond_input_pos': {'height': tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
         4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
         6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
         7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
         8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
         9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
       device='cuda:0'), 'width': tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17],
       device='cuda:0')}, 'H_y': 18, 'W_y': 18, 'H_x': 36, 'W_x': 36}
'''

def compute_pos_args(latent_hw, image_grid_thw, max_seq_len, device, cond_pos_scale=1, image_size=1024, token_embed_lengths=None):
    import math
    
    # Compute 2D position ids for RoPE
    # x_input_pos: for diffusion model's latent patches
    # latents shape: [N, C, H_latent, W_latent], grid size = H_latent x W_latent (with patch_size=1)
    h_latent, w_latent = latent_hw
    x_input_pos = compute_input_pos(h_latent, w_latent, device=device)
    
    # cond_input_pos: for condition tokens from image tokenizer
    # image_grid_thw: [B, 3] where each row is (t, h, w), 14x14 patch size
    # Use the first sample's grid (assuming same grid for all samples in batch)
    ## divide by 2 because the token embeddings is merged by 2x2 patches
    h_cond, w_cond = (resize_hw(image_grid_thw[0][1:] // 2, max_seq_len) ).tolist()

    if token_embed_lengths is not None:
        redundant_tokens = token_embed_lengths[0] - (h_cond * w_cond)
    else:
        redundant_tokens = 0

    w_cond_correction = w_cond + math.ceil(redundant_tokens / h_cond)

    cond_input_pos = compute_input_pos(h_cond, w_cond_correction, device=device)
    cond_input_pos = {k: (v * cond_pos_scale).long() for k, v in cond_input_pos.items()}

    # Pad cond_input_pos to max_seq_len (matching tokenize_images dynamic padding)
    cond_seq_len = h_cond * w_cond_correction
    pad_len = max_seq_len - cond_seq_len
    if pad_len > 0:
        cond_input_pos = {
            "height": F.pad(cond_input_pos["height"], (0, pad_len), value=0),
            "width": F.pad(cond_input_pos["width"], (0, pad_len), value=0),
        }
    
    args = {
        "x_input_pos": x_input_pos,
        "cond_input_pos": cond_input_pos,
        "H_y": h_cond,
        "W_y": w_cond_correction,

        "H_x": image_size // 28,
        "W_x": image_size // 28,
        
    }

    return args



def train():
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    if args.im_token_generation_length is None:
        args.im_token_generation_length = args.max_condition_length
        print_rank_0(f"im_token_generation_length: {args.im_token_generation_length}")

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
    
    # Save model config to output_dir before training
    if dist.get_rank() == 0:
        config_save_path = os.path.join(args.output_dir, "config.json")
        model_config.save(config_save_path)
        print_rank_0(f"Saved model config to: {config_save_path}")
    
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
                allow_random_init_params=args.allow_random_init_params,
                skip_load_params=args.skip_load_params
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

    # Freeze specified parameters
    if args.freeze_params:
        freeze_patterns = [p.strip() for p in args.freeze_params.split(',') if p.strip()]
        if freeze_patterns:
            frozen_count = freeze_params_by_pattern(model, freeze_patterns)
            print_rank_0(f"Frozen {frozen_count} parameters with patterns: {freeze_patterns}")

    if args.compile:
        # Compile model for better performance
        model = torch.compile(model)
        print_rank_0("Model compiled")

    if state_dict is not None:
        # Free the state_dict to save memory
        del state_dict

    # Print trainable and frozen parameters
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.shape))
        else:
            frozen_params.append((name, param.shape))

    print_rank_0("=" * 60)
    print_rank_0(f"Trainable Parameters ({len(trainable_params)}):")
    for name, shape in trainable_params:
        print_rank_0(f"  {name}: {shape}")
    print_rank_0("-" * 60)
    print_rank_0(f"Frozen Parameters ({len(frozen_params)}):")
    for name, shape in frozen_params:
        print_rank_0(f"  {name}: {shape}")
    print_rank_0("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(f"Total params: {total_params:,}, Trainable: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")

    ############## Load VAE and text encoder ##############
    # VAE & Text Encoder is not trainable
    vae = None

    # VAE uses bfloat16 to match model compute dtype
    vae = load_vae(
        vae_dir=args.vae_dir,
        device=torch.cuda.current_device(),
        dtype=get_torch_dtype(args.model_dtype)
    )
    image_tokenizer = load_keye_ar(
        tokenizer_dir=args.keye_ar_dir,
        device=torch.cuda.current_device(),
        dtype=args.model_dtype,
        output_last_hidden_states_only=False
    )

    # Setup visualization model (for FSDP mode)
    # In FSDP mode, we need a separate model instance for inference
    # because the training model has sharded weights
    model_for_vis = None
    if args.visualize_parquet_path and dist.get_rank() == 0:
        with set_default_dtype(args.model_dtype), torch.device("cpu"):
            model_for_vis = model_cls(model_config)
        print_rank_0("Created model instance for visualization (on CPU)")
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
        num_decay_steps=args.num_decay_steps,
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
    if not dataset_config.get("processor_path") and args.keye_ar_dir:
        dataset_config["processor_path"] = args.keye_ar_dir
        print_rank_0(f"Set tokenizer_path of dataset to: {args.keye_ar_dir}")
    
    # Add distributed rank/world_size to dataset config for proper data sharding
    if dist.is_initialized():
        dataset_config["rank"] = dist.get_rank()
        dataset_config["world_size"] = dist.get_world_size()
        print_rank_0(f"Dataset sharding: rank={dataset_config['rank']}, world_size={dataset_config['world_size']}")
    
    dataset_config["multi_scale"] = args.multi_scale
    dataset_config["max_condition_length"] = args.max_condition_length


    print_rank_0(f"Building dataset with config: {dataset_config}")
    dataset = Chat2ImageDataset(**dataset_config)
    collate_fn = dataset.collate_fn
    ar_processor = dataset.processor
    if args.multi_scale:
        # Parse resolution budget config or create single-resolution default
        if args.resolution_budgets:
            budget_config = parse_resolution_budgets(args.resolution_budgets)
        else:
            # Single resolution default - no curriculum scheduling
            budget_config = ResolutionBudgetConfig(
                budgets=[ResolutionBudget(args.image_size, args.batch_size)],
            )
        
        print_rank_0(f"Multi-scale training configuration:")
        for b in budget_config.budgets:
            print_rank_0(f"  {b.size}x{b.size}: batch_size={b.batch_size}")
        
        # Wrap with multi-scale wrapper (supports both fixed and dynamic resolution)
        multi_scale_wrapper = MultiScaleDatasetWrapper(
            dataset=dataset,
            config=budget_config,
            drop_last=True
        )
        
        dataloader = DataLoader(
            multi_scale_wrapper,
            batch_size=1,  # Wrapper yields pre-batched lists
            num_workers=args.num_workers,
            collate_fn=lambda x: collate_fn(x[0])
        )
        
        print_rank_0("Multi-scale training enabled")
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

    if args.run_data_iter:
        while True:
            it = 0
            for batch in data_iter:
                print_rank_0(f"Batch {it}: rank={dist.get_rank()}")
                it += 1

    # Step 0 visualization: show model state before any optimization
    if args.visualize_parquet_path:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        
        print_rank_0("Running step 0 visualization (before training)...")
        # Collect full state dict from FSDP model (all ranks participate)
        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        
        # Only rank 0 does the actual visualization
        if dist.get_rank() == 0 and model_for_vis is not None:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load weights to visualization model and move to GPU
            model_for_vis.load_state_dict(state_dict)
            model_for_vis.to(torch.cuda.current_device())
            model_for_vis.to(get_torch_dtype(args.model_dtype))
            model_for_vis.eval()
            
            with Timer("visualization step 0"):
                visualize_reconstruction(
                    model=model_for_vis,
                    vae=vae,
                    image_tokenizer=image_tokenizer,
                    parquet_path=args.visualize_parquet_path,  # 改为parquet_path参数
                    dataset=dataset,  # 传入dataset用于处理方法
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
                    args=args
                )
            
            # Move model back to CPU to save memory
            model_for_vis.cpu()
            torch.cuda.empty_cache()
        
        # Sync all ranks after step 0 visualization
        dist.barrier()

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

            if batch["input_ids"].numel() > args.max_seq_length:
                print_rank_0(f"rank={dist.get_rank()}, Skipping batch with input_ids.numel()={batch['input_ids'].numel()} > args.max_seq_length={args.max_seq_length}")
                continue

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
                    raise ValueError(f"No latents or images in batch. batch keys={batch.keys()}, vae={type(vae)}")

            # 4. Text Encoder
            with record_function("TextEncoder"):
                token_embeds, attention_mask, max_seq_len, token_embed_lengths = tokenize_images(
                    image_tokenizer,
                    batch["pixel_values"],
                    batch["image_grid_thw"],
                    batch["image"].shape[0],
                    args.max_condition_length,
                    input_ids=batch.get("input_ids"),
                    cu_seqlens=batch.get("cu_seqlens"),
                    condition_on_special_tokens=args.condition_on_special_tokens,
                    ar_processor=ar_processor,
                )
            
            pos_args = compute_pos_args(
                latent_hw=(latents.shape[2], latents.shape[3]),
                image_grid_thw=batch["image_grid_thw"],
                max_seq_len=max_seq_len,
                device=latents.device,
                cond_pos_scale=args.cond_pos_scale,
                image_size=args.image_size,
                token_embed_lengths=token_embed_lengths,
            )

            if np.random.rand() < 0.0001:
                print(f"token_embeds={token_embeds.shape}, pos_args={pos_args}")

            # 5. Forward + Loss Computation
            with record_function("Forward_Loss"):
                loss_dict = loss_fn(
                    model=model,
                    x_start=latents,
                    y=token_embeds.unsqueeze(1),
                    mask=attention_mask,  # Use attention_mask instead of None
                    model_kwargs={
                        **pos_args
                    },
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
                try:
                    metrics.write_logs(scheduler.global_step)
                except Exception as e:
                    print(f"Logging failed: {e}")

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

            # Visualization: generate sample images every N steps
            # All ranks participate in get_model_state_dict, only rank 0 does visualization
            if (args.visualize_parquet_path and 
                scheduler.global_step > 0 and 
                scheduler.global_step % args.visualize_per_step == 0):
                from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
                
                # Collect full state dict from FSDP model (all ranks participate)
                state_dict = get_model_state_dict(
                    model,
                    options=StateDictOptions(
                        full_state_dict=True,
                        cpu_offload=True,
                    )
                )
                
                # Only rank 0 does the actual visualization
                if dist.get_rank() == 0 and model_for_vis is not None:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Load weights to visualization model and move to GPU
                    model_for_vis.load_state_dict(state_dict)
                    model_for_vis.to(torch.cuda.current_device())
                    model_for_vis.to(get_torch_dtype(args.model_dtype))
                    model_for_vis.eval()
                    
                    with Timer("visualization"):
                        visualize_reconstruction(
                            model=model_for_vis,
                            vae=vae,
                            image_tokenizer=image_tokenizer,
                            parquet_path=args.visualize_parquet_path,  # 改为parquet_path参数
                            dataset=dataset,  # 传入dataset用于处理方法
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
                            args=args
                        )
                    
                    # Move model back to CPU to save memory
                    model_for_vis.cpu()
                    torch.cuda.empty_cache()
                
                # Sync all ranks after visualization
                dist.barrier()

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
