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

This script implements training for the Sana diffusion transformer model,
following the exact logic from the original Sana codebase while using
muse's training infrastructure.

Reference:
- Sana/train_scripts/train.py Lines 260-575
- Sana/diffusion/model/gaussian_diffusion.py
"""

from typing import Dict, Any, Union, Optional
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
    get_checkpoint_path,
    save_checkpoint
)
from muse.training.common import (
    set_default_dtype, 
    clip_grad_by_value, 
    compute_fsdp_zero2_grad_norm
)
from muse.utils.common import Timer
from muse.training.lr_schedulers import get_scheduler
from muse.training.activations import set_activation_checkpointing
from muse.training.parallel import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    initialize_model_parallel,
)
from muse.utils.common import (
    set_random_seed, 
    print_rank_0,
    print_rank_n,
    to_cuda,
    dist_reduce_dict
)
from muse.data.datasets import ImageTextDataset
from muse.losses.diffusion import FlowMatchingLoss

from muse.utils.metrics import Logger, StdoutBackend, CSVBackend, TensorBoardBackend
from muse.training.common import initialize_metrics, StepScheduler


logger = logging.getLogger(__name__)


def get_argument_parser():
    parser = argparse.ArgumentParser()

    ############ Model args ############
    parser.add_argument("--model-config", type=str, required=True,
                        help="The config file path of the model to train")

    parser.add_argument("--sana-checkpoint", type=str, default=None,
                        help="Path to Sana checkpoint to resume training from")
    
    parser.add_argument("--null-embed-path", type=str, default=None,
                        help="Path to null embedding file for CFG")

    ############ VAE args ############
    parser.add_argument("--vae-pretrained", type=str,
                        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                        help="Pretrained VAE model path")

    ############ Text Encoder args ############
    parser.add_argument("--text-encoder", type=str,
                        default="google/gemma-2-2b-it",
                        help="Text encoder model name")
    
    parser.add_argument("--max-text-length", type=int, default=300,
                        help="Maximum text sequence length")

    ############ Dataset args ############
    parser.add_argument("--dataset-config", type=str, required=True,
                        help="The config file path of the dataset to train")

    parser.add_argument("--image-size", type=int, default=1024,
                        help="Training image size")
    
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per GPU")
    
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")

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

    ############ Debug Args ############
    parser.add_argument("--enable-profile", action="store_true",
                        help="Enable torch profiler")
    
    parser.add_argument("--skip-vae-encode", action="store_true",
                        help="Skip VAE encoding (use pre-encoded latents)")
    
    parser.add_argument("--skip-text-encode", action="store_true",
                        help="Skip text encoding (use pre-encoded embeddings)")

    return parser


def load_vae(vae_pretrained: str, device: torch.device, dtype: torch.dtype):
    """Load VAE model from diffusers.
    
    Reference: Sana/diffusion/model/builder.py
    """
    from diffusers import AutoencoderDC
    
    print_rank_0(f"Loading VAE from {vae_pretrained}")
    vae = AutoencoderDC.from_pretrained(vae_pretrained, torch_dtype=dtype)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    
    return vae


def vae_encode(vae, images: torch.Tensor, sample_posterior: bool = True) -> torch.Tensor:
    """Encode images to latent space.
    
    Reference: Sana/train_scripts/train.py Lines 100-110
    """
    with torch.no_grad():
        posterior = vae.encode(images)
        if hasattr(posterior, 'latent_dist'):
            posterior = posterior.latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = z * vae.config.scaling_factor
    return z


def load_text_encoder(text_encoder_name: str, device: torch.device, dtype: torch.dtype):
    """Load text encoder and tokenizer.
    
    Reference: Sana/diffusion/model/builder.py Lines 53-89
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print_rank_0(f"Loading text encoder from {text_encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    
    # Load the full model and get the decoder
    model = AutoModelForCausalLM.from_pretrained(
        text_encoder_name,
        torch_dtype=dtype,
    )
    text_encoder = model.get_decoder() if hasattr(model, 'get_decoder') else model
    text_encoder = text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)
    
    return tokenizer, text_encoder


def encode_text(
    tokenizer,
    text_encoder,
    texts: list,
    max_length: int,
    device: torch.device,
) -> tuple:
    """Encode text to embeddings.
    
    Reference: Sana/train_scripts/train.py Lines 300-310
    """
    tokens = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = text_encoder(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
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
    attention_mask = tokens.attention_mask[:, None, None]
    
    return text_embeds, attention_mask


def load_sana_checkpoint(checkpoint_path: str, model, null_embed_path: Optional[str] = None):
    """Load Sana checkpoint and convert to muse format.
    
    Reference: Sana/diffusion/utils/checkpoint.py
    """
    print_rank_0(f"Loading Sana checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get state dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    
    # Convert using model's converter
    converted_state_dict = model.convert_sana_state_dict(
        new_state_dict,
        null_embed_path=null_embed_path,
    )
    
    # Load state dict
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing:
        print_rank_0(f"Missing keys: {missing}")
    if unexpected:
        print_rank_0(f"Unexpected keys: {unexpected}")
    
    return checkpoint.get("global_step", 0)


def _init_profiler(output_dir) -> None:
    """Initialize torch profiler."""
    if not os.path.exists(output_dir):
        if dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)

    def trace_handler(prof):
        prof.export_chrome_trace(
            os.path.join(
                output_dir, str(prof.step_num) + f"_w{dist.get_rank()}" + ".json")
        )

    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=50,
            warmup=1,
            active=10,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
    )
    return torch_profiler


def train():
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    # torch init
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))

    print_rank_n(f"torch init rank={rank}, local_rank={local_rank}, world_size={world_size}")
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        torch.distributed.init_process_group(
            rank=rank, world_size=world_size,
            timeout=process_group_timeout
        )
        device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
    else:
        device_mesh = None

    # Initialize model parallel
    initialize_model_parallel(context_parallel_size=1)
    print_rank_0(f"Data parallel size: {get_data_parallel_world_size()}")

    set_random_seed(args.seed)

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        args_str = json.dumps(vars(args), indent=2, ensure_ascii=False)
        print_rank_0(f"Training Arguments:\n{args_str}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(os.path.join(args.output_dir, f"args-{timestamp}.json"), 'w') as f:
            f.write(args_str + "\n")

    # Setup dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.model_dtype]
    device = torch.device(f"cuda:{local_rank}")

    # Load model config
    print_rank_0(f"Loading model config from {args.model_config}")
    model_config = load_config(args.model_config)
    
    # Get model class
    print_rank_0(f"Available models: {list_models()}")
    model_cls = get_model_class(model_config.model_class)
    print_rank_0(f"Model class: {model_cls.__name__}")

    # Load VAE and text encoder (only needed if not using pre-encoded data)
    vae = None
    tokenizer = None
    text_encoder = None
    
    if not args.skip_vae_encode:
        vae = load_vae(args.vae_pretrained, device, model_dtype)
    
    if not args.skip_text_encode:
        tokenizer, text_encoder = load_text_encoder(args.text_encoder, device, model_dtype)

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

    # Shard model for distributed training
    if device_mesh is not None:
        shard_model(
            model=model,
            cpu_offload=args.cpu_offload,
            reshard_after_forward=args.reshard_after_forward,
            dp_mesh=device_mesh,
        )
        if dist.is_initialized():
            dist.barrier()
    
    # Load checkpoint or initialize
    start_step = 0
    if args.sana_checkpoint:
        # Load from Sana checkpoint
        if rank == 0:
            start_step = load_sana_checkpoint(
                args.sana_checkpoint,
                model,
                args.null_embed_path,
            )
        if dist.is_initialized():
            dist.barrier()
    else:
        # Initialize model parameters randomly
        with Timer("Initialize model parameters"):
            initialize_model_params(model)

    # Move model to device and set dtype
    model = model.to(device=device, dtype=model_dtype)

    if args.compile:
        model = torch.compile(model)
        print_rank_0("Model compiled")

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(f"Total parameters: {total_params:,}")
    print_rank_0(f"Trainable parameters: {trainable_params:,}")

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

    # Build dataset
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)

    print_rank_0(f"Building dataset with config: {dataset_config}")
    dataset = ImageTextDataset(
        sources=dataset_config.get("sources", dataset_config.get("data_path")),
        image_size=args.image_size,
        vae=vae if not args.skip_vae_encode else None,
        tokenizer=tokenizer if not args.skip_text_encode else None,
        text_encoder=text_encoder if not args.skip_text_encode else None,
        max_text_length=args.max_text_length,
        rank=get_data_parallel_rank(),
        world_size=get_data_parallel_world_size(),
        num_workers=args.num_workers,
        seed=args.seed,
        **{k: v for k, v in dataset_config.items() if k not in ["sources", "data_path", "name"]}
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Workers already in DistributedDataset
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
    )

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

    scheduler = StepScheduler(args)

    # Setup profiler
    torch_profiler = _init_profiler(
        output_dir=os.path.join(args.output_dir, "torch_profile")
    ) if args.enable_profile else None

    # Training loop
    print_rank_0("Starting training...")
    model.train()
    
    data_iter = iter(dataloader)

    while scheduler.global_step < args.num_training_steps:
        with contextlib.ExitStack() as ctx:
            if torch_profiler:
                ctx.enter_context(torch_profiler)

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device=device, dtype=model_dtype if v.is_floating_point() else v.dtype)

            scheduler.step()

            # Get latents (either pre-encoded or encode on-the-fly)
            if "latents" in batch:
                latents = batch["latents"]
            elif "image" in batch and vae is not None:
                latents = vae_encode(vae, batch["image"])
            else:
                raise ValueError("No latents or images in batch")

            # Get text embeddings
            if "text_embeds" in batch:
                text_embeds = batch["text_embeds"]
                attention_mask = batch.get("attention_mask")
            elif "text" in batch and tokenizer is not None and text_encoder is not None:
                text_embeds, attention_mask = encode_text(
                    tokenizer,
                    text_encoder,
                    batch["text"],
                    args.max_text_length,
                    device,
                )
            else:
                raise ValueError("No text embeddings or text in batch")

            # Compute loss
            loss_dict = loss_fn(
                model=model,
                x_start=latents,
                y=text_embeds,
                mask=attention_mask,
            )
            loss = loss_dict["loss"]

            metrics.loss.append(loss.detach().item())

            # Backward pass
            loss.backward()
            clip_grad_by_value(model, args.clip_range)

            # Update optimizer at gradient accumulation boundaries
            if scheduler.is_gradient_accumulation_boundary():
                grad_norm = compute_fsdp_zero2_grad_norm(model)
                metrics.grad_norm.append(grad_norm)
                learning_rate = lr_scheduler.get_last_lr()[0]
                metrics.learning_rate.append(learning_rate)
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
