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
from muse.data.datasets import Text2ImageDataset
from muse.losses.diffusion import FlowMatchingLoss

from muse.utils.metrics import Logger, StdoutBackend, CSVBackend, TensorBoardBackend
from muse.training.common import initialize_metrics, StepScheduler


logger = logging.getLogger(__name__)


def get_argument_parser():
    parser = argparse.ArgumentParser()

    ############ Model args ############
    parser.add_argument("--model-config", type=str, default=None,
                        help="The config file path of the model to train")

    parser.add_argument("--model-dir", type=str, default=None,
                      help="The directory of the pretrained model (required for continue training).")

    ############ VAE args ############
    parser.add_argument("--vae-dir", type=str,
                        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                        help="Pretrained VAE model path")

    ############ Text Encoder args ############
    parser.add_argument("--text-encoder-dir", type=str,
                        default="google/gemma-2-2b-it",
                        help="Text encoder model name")

    parser.add_argument("--tokenizer-dir", type=str,
                        default="google/gemma-2-2b-it",
                        help="Tokenizer model name")
    
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

def load_text_encoder(text_encoder_dir: str, device: torch.device, dtype: torch.dtype):
    """Load text encoder.
    
    Reference: Sana/diffusion/model/builder.py Lines 53-89
    """
    from transformers import AutoModelForCausalLM
    
    # Load the full model and get the decoder
    text_encoder = AutoModelForCausalLM.from_pretrained(
        text_encoder_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    text_encoder = text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)
    
    return text_encoder


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

    set_random_seed(args.seed)


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
    tokenizer = None
    text_encoder = None

    vae = load_vae(
        vae_pretrained=args.vae_pretrained,
        device=torch.cuda.current_device(),
        dtype=get_torch_dtype(args.model_dtype)
    )
    text_encoder = load_text_encoder(
        text_encoder_dir=args.text_encoder_dir,
        device=torch.cuda.current_device(),
        dtype=get_torch_dtype(args.model_dtype)
    )
    ############## Load VAE and text encoder ##############

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
    
    if args.max_length:
        dataset_config["max_length"] = args.max_length
    
    # Set tokenizer_path from model_dir if not specified
    if not dataset_config.get("tokenizer_path") and args.tokenizer_path:
        dataset_config["tokenizer_path"] = args.tokenizer_path

    print_rank_0(f"Building dataset with config: {dataset_config}")
    dataset = Text2ImageDataset(**dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    # Training loop
    print_rank_0("Starting training...")
    model.train()

    # Setup profiler
    torch_profiler = _init_profiler(
        output_dir=os.path.join(args.output_dir, "torch_profile")
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
                    batch[k] = v.to(
                        device=device, dtype=model_dtype \
                            if v.is_floating_point() else v.dtype)

            scheduler.step()

            # Get latents (either pre-encoded or encode on-the-fly)
            if "latents" in batch:
                latents = batch["latents"]
            elif "image" in batch and vae is not None:
                latents = vae_encode(vae, batch["image"])
            else:
                raise ValueError("No latents or images in batch")

            text_embeds, attention_mask = encode_text(
                tokenizer,
                text_encoder,
                batch["input_ids"],
                batch.get("attention_mask"),
                device,
            )

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
