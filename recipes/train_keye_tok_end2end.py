# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework - KeyeTokenizerEnd2EndImage training
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
KeyeTokenizerEnd2EndImage Training Script.

This script implements training for the KeyeTokenizerEnd2EndImage model,
which combines a visual tokenizer (KeyeImageTokenizer) with a language model (Qwen3).

Usage:
    python recipes/train_keye_tok_end2end.py \
        --model-dir /path/to/model \
        --output-dir /path/to/output \
        --dataset-config examples/keye_tokenizer_end2end_image/config.json \
        ...
"""

from typing import Dict, Any, Union, Optional
import os
import torch
import datetime
import contextlib
import argparse
import time
import collections
import json
import logging
import threading
import itertools
import queue
import traceback
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.profiler import record_function

from collections import defaultdict

torch.autograd.set_detect_anomaly(True)
import gc
gc.disable()

process_group_timeout = datetime.timedelta(minutes=60*24)

# Muse imports
from muse.models import get_model_class, list_models
from muse.config import get_config
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
    to_device,
    dist_reduce_dict
)
from muse.data.datasets import ChatCompletionVisionDataset,ChatCompletionVisionDataset_keye_vitrope_slowfast

from muse.config import load_config

from muse.utils.metrics import Logger, StdoutBackend, CSVBackend, TensorBoardBackend
from muse.training.common import initialize_metrics, StepScheduler
from muse.losses import CrossEntropyLoss

logger = logging.getLogger(__name__)


def get_argument_parser():
    parser = argparse.ArgumentParser()

    ############ Model args ############
    parser.add_argument("--model-config", type=str, default=None,
                        help="The config file path of the model to train (required for train from scratch), e.g. model_dir/config.json")

    ############ Dataset args ############
    parser.add_argument("--dataset-class", type=str, default=None,
                        help="The dataset class name registered in muse.datasets.")

    parser.add_argument("--dataset-config", type=str, default=None,
                        help="The config file path of the dataset to train.")

    parser.add_argument("--max-length", type=int, default=None,
                        help="Max tokens per sentence in corpus")
    
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training")

    parser.add_argument("--shuffle-buffer-size", type=int, default=0,
                        help="Size of shuffle buffer for local data shuffling (0 to disable)")

    parser.add_argument("--use-dataset-load-balance", action="store_true",
                        help="Use load balance for dataset")

    parser.add_argument("--packing", action="store_true", default=True,
                        help="Whether to use packing for dataset")
    
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")

    ############ Checkpoint args ############
    parser.add_argument("--model-dir", type=str, default=None,
                        help="The directory of the pretrained model (required for continue pretrain).")

    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Specify the checkpoint directory to resume from.")

    parser.add_argument("--checkpoint-id", type=str, default=None,
                        help="Specify the checkpoint id to resume from, e.g. global_step1000")
    
    parser.add_argument("--resume-dataloader", action="store_true", default=True,
                        help="Whether to resume dataloader checkpoint")
    
    parser.add_argument("--no-resume-dataloader", action="store_false", dest="resume_dataloader",
                        help="Don't resume dataloader checkpoint")
    
    parser.add_argument("--resume-optimizer", action="store_true", default=True,
                        help="Whether to resume optimizer checkpoint")
    
    parser.add_argument("--no-resume-optimizer", action="store_false", dest="resume_optimizer",
                        help="Don't resume optimizer checkpoint")
    
    parser.add_argument("--save-checkpoint-per-step", type=int, default=1000,
                        help="The number of steps to save a checkpoint")

    parser.add_argument("--save-checkpoint-every-epoch", action="store_true",
                        help="Save checkpoint at the end of every epoch")
    
    parser.add_argument("--output-dir", type=str, default=None,
                        help="The directory to write the trained model")
    
    parser.add_argument("--model-dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="The dtype of the model.")

    parser.add_argument("--enable-dataset-checkpointing", action="store_true",
                        help="Enable dataset checkpoint recovery")
    
    ############ FSDP Args ############
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Whether to offload parameters, gradients, and optimizer states to CPU")

    parser.add_argument("--fp32-weight", action="store_true",
                        help="Whether use fp32 for model weight updating")

    parser.add_argument("--fp32-reduce", action="store_true",
                        help="Whether use fp32 for model gradient reduction")

    parser.add_argument("--reshard-after-forward", action="store_true",
                        help="Reshard params after forward pass, aka Zero3.")
    
    parser.add_argument("--prefetch-params-in-forward", action="store_true",
                        help="Prefetch parameters in forward pass.")

    parser.add_argument("--compile", action="store_true",
                        help="compile model.")

    parser.add_argument("--allow-random-init-params", type=str, default='',
                        help="Parameter names to allow random initialization")

    ############ Optimizer & Learning Rate Args ############
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        help="The type of learning rate scheduler.")

    parser.add_argument("--num-warmup-steps", type=int, default=0,
                        help="The number of warmup steps to do.")
    
    parser.add_argument("--num-decay-steps", type=int, default=1000,
                        help="The number of steps to decay.")

    parser.add_argument("--num-training-steps", type=int, default=1000,
                        help="The number of training steps to do.")

    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of epochs to train, no effect for pretraining.")
    
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="The minimum learning rate to reach after the cosine schedule.")

    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="The peak learning rate for optimizer.")

    # For AdamW optimizer
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="The weight decay for Adam Optimizer")
    
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="beta1 for Adam Optimizer")

    parser.add_argument("--beta2", type=float, default=0.95,
                        help="beta2 for Adam Optimizer")
    
    parser.add_argument("--clip-range", type=float, default=1.0,
                        help="The gradient clip range.")

    ############ Training Args ############

    parser.add_argument("--use-flash-attention-2", action="store_true",
                        help="Whether to use flash attention 2")

    parser.add_argument("--enable-gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing during training")

    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    
    parser.add_argument("--context-parallel-size", type=int, default=1,
                        help="Context parallelism size")

    parser.add_argument("--logging-per-step", type=int, default=100,
                        help="The number of steps to log training info")

    parser.add_argument("--comment", type=str, default=None,
                        help="Comment of this experiment.")

    parser.add_argument("--commit-id", type=str, default=None,
                        help="Git commit id for experiment.")

    parser.add_argument("--seed", type=int, default=123,
                        help="Manual seed for RNG")

    ############ Loss weights ############
    parser.add_argument("--codebook_loss_weight", type=float, default=1.0,
                        help="Weight for codebook loss")
    
    parser.add_argument("--commitment_loss_weight", type=float, default=0.25,
                        help="Weight for commitment loss")

    ############ Profile Args ############

    parser.add_argument("--enable-profile", action="store_true",
                        help="Enable torch profile")

    ############ Debug Args ############

    parser.add_argument("--overfit-batches", type=int, default=None,
                        help="Number of batches to cache for overfitting (debug mode)")

    ############ Freeze Args ############
    parser.add_argument("--freeze_llm", action="store_true",
                        help="Freeze LLM parameters, only train visual_tokenizer and quant_projector")
    
    parser.add_argument("--freeze_projector", action="store_true",
                        help="Freeze quant_projector parameters")
    
    parser.add_argument("--freeze_tokenizer", action="store_true",
                        help="Freeze visual_tokenizer parameters, train quant_projector parameters")
    
    parser.add_argument("--freeze_navit", action="store_true",
                        help="Freeze NaViT (visual_tokenizer.visual) parameters, train VQ parameters")
    
    parser.add_argument("--freeze_navit_mlp_ar", action="store_true",
                        help="Freeze visual_tokenizer.mlp_AR parameters")

    return parser





def freeze_params(args, model):
    """Freeze specific model parameters based on command line arguments.
    
    Args:
        args: Command line arguments containing freeze flags
        model: The model to freeze parameters on
    """
    if args.freeze_llm:
        print_rank_0("Freeze LLM parameters.")
        for name, param in model.named_parameters():
            if not (name.startswith("visual_tokenizer") or name.startswith("quant_projector")):
                print_rank_0(f"Disable LLM grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)
    
    if args.freeze_projector:
        print_rank_0("Freeze quant_projector parameters.")
        for name, param in model.named_parameters():
            if name.startswith("quant_projector"):
                print_rank_0(f"Disable quant_projector grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)
    
    if args.freeze_tokenizer:
        print_rank_0("Freeze tokenizer parameters. Train quant_projector parameters")
        for name, param in model.named_parameters():
            if name.startswith("visual_tokenizer"):
                print_rank_0(f"Disable visual_tokenizer grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)
    
    if args.freeze_navit:
        print_rank_0("Freeze NaViT parameters. Train VQ parameters")
        for name, param in model.named_parameters():
            if name.startswith("visual_tokenizer.visual") and not name.startswith("visual_tokenizer.mlp_AR"):
                print_rank_0(f"Disable visual_tokenizer_navit grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)
    
    if args.freeze_navit_mlp_ar:
        print_rank_0("Freeze mlp_AR parameters.")
        for name, param in model.named_parameters():
            if name.startswith("visual_tokenizer.mlp_AR"):
                print_rank_0(f"Disable visual_tokenizer.mlp_AR grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)


def _init_profiler(output_dir) -> None:
    """Initialize torch profiler with TensorBoard support."""
    profile_dir = os.path.join(output_dir, "torch_profile")
    if dist.get_rank() == 0:
        os.makedirs(profile_dir, exist_ok=True)
    dist.barrier()

    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=5,
            warmup=2,
            active=10,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
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

    # Get distributed training info from MPI environment
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

    ##############
    with open(args.dataset_config, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())

    
    # Determine training mode and get model_class
    if args.model_dir:
        # Continue pretrain mode: get model_class from model_dir/config.json
        model_config_path = Path(args.model_dir) / "muse_config.json"
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
            "(for train from scratch) must be provided."
        )

    if args.use_flash_attention_2:
        model_config.qwen_config.attention_function = "flash_attention_2"
        print_rank_0("Use flash attention 2")
    else:
        print_rank_0("Warning: Use eager attention, performance may be degraded.")

    model_class_name = model_config.model_class
    dataset_config["model_class"] = model_class_name
    
    if args.max_length:
        dataset_config["max_length"] = args.max_length
    
    # Set base_model_dir from model_dir if not specified
    if not dataset_config.get("base_model_dir") and args.model_dir:
        dataset_config["base_model_dir"] = args.model_dir

    # torch init
    print_rank_n(f"torch init rank={rank}, local_rank={local_rank}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        rank=rank, world_size=world_size,
        timeout=process_group_timeout
    )
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))

    ### initialize model parallel group
    # Currently only support context parallelism
    initialize_model_parallel(context_parallel_size=args.context_parallel_size)
    print_rank_0(f"Context parallel size: {get_context_parallel_world_size()}")
    print_rank_0(f"Data parallel size: {get_data_parallel_world_size()}")

    set_random_seed(args.seed)

    if dist.get_rank() == 0:
        args_str = json.dumps(vars(args), indent=2, ensure_ascii=False)
        print_rank_0(f"Training Arguments:\n{args_str}")
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(os.path.join(args.output_dir,
                f"args-{args.commit_id}-{timestamp}.json"), 'w',
                encoding="utf-8") as f:
            f.write(args_str + "\n")

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

    # Setup TensorBoard writer
    tb_writer = None
    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
        tb_writer.add_text("comment", args.comment, 0)
        tb_writer.add_text("comment_id", args.commit_id, 0)

    # Instantiate model on meta device, this is to avoid OOM
    with set_default_dtype(args.model_dtype), torch.device("meta"):
        print_rank_0(f"Creating model from config: {args.model_config or args.model_dir}")
        model = model_cls(model_config)
        print_rank_0(f"Model instantiated from config: {type(model).__name__}")
    
    if args.enable_gradient_checkpointing:
        print_rank_0("Enable gradient checkpointing")
        set_activation_checkpointing(
            model, auto_wrap_policy=model.get_checkpointable_module_classes()
        )

    # upcast fp32 to maintain master weight.
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
    
    # Load weights or initialize parameters
    if args.model_dir:
        with Timer("Load state dict"):
            load_from_full_model_state_dict(
                model=model, full_sd=state_dict,
                allow_random_init_params=args.allow_random_init_params
            )
    else:
        with Timer("Initialize model parameters"):
            initialize_model_params(model)

    with torch.device(torch.cuda.current_device()):
        # Initialize RoPE if needed
        for m in model.modules():
            if hasattr(m, "rope_init"):
                print_rank_0("Initialize RoPE")
                m.rope_init()


    # Fix: Materialize buffers that are still on meta device (e.g. position_ids)
    # 修复：手动实例化那些不在 checkpoint 中且仍停留在 meta 设备上的 buffers
    for name, module in model.named_modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer.device.type == "meta":
                print_rank_0(f"Materializing buffer '{name}.{buffer_name}' from meta to {torch.cuda.current_device()}")
                
                # 如果是 position_ids，通常需要初始化为 [0, 1, 2, ...]
                if "position_ids" in buffer_name:
                    # 获取序列长度 (通常是最后一个维度)
                    seq_len = buffer.shape[-1]
                    # 创建 [0, 1, ..., seq_len-1]
                    new_buffer = torch.arange(seq_len, device=torch.cuda.current_device(), dtype=buffer.dtype)
                    # 如果原始 shape 是 [1, seq_len]，需要 expand
                    if buffer.ndim > 1:
                        new_buffer = new_buffer.expand(buffer.shape)
                else:
                    # 其他 buffer 默认初始化为全 0
                    new_buffer = torch.zeros_like(buffer, device=torch.cuda.current_device())
                
                # 将实例化的 buffer 注册回模块，替换掉 meta buffer
                module.register_buffer(buffer_name, new_buffer)

    # Check if all parameters & buffers are initialized
    for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        assert tensor.device != torch.device("meta"), \
            f"{name} not initialized, device={tensor.device}"

    if args.compile:
        model = torch.compile(model)
        print_rank_0("Model compiled")

    if state_dict is not None:
        del state_dict

    # Freeze specific parameters based on args
    freeze_params(args, model)

    # Print frozen and trainable parameters separately
    print_rank_0("=" * 50)
    print_rank_0("Frozen Parameters:")
    frozen_count = 0
    frozen_numel = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print_rank_0(f"  [FROZEN] {name}: {param.shape}")
            frozen_count += 1
            frozen_numel += param.numel()
    print_rank_0(f"Total frozen: {frozen_count} params, {frozen_numel:,} elements")
    print_rank_0("=" * 50)
    
    print_rank_0("Trainable Parameters:")
    trainable_count = 0
    trainable_numel = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank_0(f"  [TRAINABLE] {name}: {param.shape}")
            trainable_count += 1
            trainable_numel += param.numel()
    print_rank_0(f"Total trainable: {trainable_count} params, {trainable_numel:,} elements")
    print_rank_0("=" * 50)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        model.get_optimizer_grouped_parameters(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        ),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=1.0e-8
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        min_lr=args.min_lr
    )

    app_state = AppState(model=model, optimizer=optimizer)
    dist_checkpointer = DistributedCheckpointer()
    if args.checkpoint_dir:
        print_rank_0(
            f"Resume from checkpoint: {args.checkpoint_dir}, tag={args.checkpoint_id}")

        state_dict = {"app": app_state}
        checkpoint_path = get_checkpoint_path(
            args.checkpoint_dir, args.checkpoint_id)

        dist_checkpointer.load_checkpoint(
            state_dict=state_dict,
            checkpoint_path=checkpoint_path,
        )

        print_rank_0(f"Successfully loaded model using distributed checkpoint")

    dist.barrier()

    if dist.get_rank() == 0:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(os.path.join(args.output_dir,
                f"dataset-{args.commit_id}-{timestamp}.json"), 'w',
                encoding="utf-8") as f:
            f.write(json.dumps(
                dataset_config, ensure_ascii=False, indent=2) + "\n")

    # Build dataloader with ChatCompletionVisionDataset
    dataloader = None
    with Timer("Build dataloader"):
        print_rank_0(f"Building dataloader with config: {dataset_config}")
        
        # Add distributed rank/world_size to dataset config for proper data sharding
        if dist.is_initialized():
            dataset_config["rank"] = dist.get_rank()
            dataset_config["world_size"] = dist.get_world_size()
            print_rank_0(f"Dataset sharding: rank={dataset_config['rank']}, world_size={dataset_config['world_size']}")
        
        dataset = ChatCompletionVisionDataset_keye_vitrope_slowfast(**dataset_config)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Each sample is already batched in ChatCompletionVisionDataset
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda x: x[0]  # Unwrap single-element list
        )

    ##############
    torch_profiler = _init_profiler(
        output_dir=args.output_dir) \
        if args.enable_profile else None

    # Setup logging
    if dist.get_rank() == 0:
        stdout_logger = Logger("stdout", [StdoutBackend()])
        csv_logger = Logger("csv", [CSVBackend(os.path.join(args.output_dir, "metrics.csv"))])
        tb_logger = Logger("tb", [TensorBoardBackend(args.output_dir)])
        loggers = [stdout_logger, csv_logger, tb_logger]
    else:
        loggers = []

    # Initialize metrics and step scheduler
    metrics = initialize_metrics(
        acc_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step,
        loggers=loggers
    )
    
    # Add extra metrics for KeyeTokenizerEnd2EndImage training
    metrics.new("lm_loss", dtype="float", reduce="mean")
    metrics.new("codebook_loss", dtype="float", reduce="mean")
    metrics.new("commitment_loss", dtype="float", reduce="mean")
    
    # Track extra metrics for logging
    acc_steps = args.gradient_accumulation_steps
    logging_per_step = args.logging_per_step
    
    avg_lm_loss = metrics.lm_loss.avg(window=acc_steps)[::acc_steps][1:]
    avg_codebook_loss = metrics.codebook_loss.avg(window=acc_steps)[::acc_steps][1:]
    avg_commitment_loss = metrics.commitment_loss.avg(window=acc_steps)[::acc_steps][1:]
    
    # Use metrics.logger to track to all registered loggers
    metrics.logger.track(
        avg_lm_loss.avg(window=logging_per_step)[::logging_per_step],
        name="lm_loss", group="training")
    metrics.logger.track(
        avg_codebook_loss.avg(window=logging_per_step)[::logging_per_step],
        name="codebook_loss", group="training")
    metrics.logger.track(
        avg_commitment_loss.avg(window=logging_per_step)[::logging_per_step],
        name="commitment_loss", group="training")
    
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
            data_iter = iter(itertools.cycle(cached_batches))
        else:
            data_iter = iter(gather_by_group(dataloader, get_context_parallel_group()))
    else:
        print_rank_0("Warning: No dataloader available. Training loop will not run.")
        data_iter = iter([])

    # Initialize loss function for external LM loss computation
    # shift_labels=False because we will pre-shift labels in the training loop
    loss_fn = CrossEntropyLoss(
        ignore_index=-100, 
        return_token_loss=True, 
        shift_labels=False
    )

    print_rank_0("Starting training...")
    model.train()
    
    while True:
        with contextlib.ExitStack() as ctx:
            if torch_profiler:
                ctx.enter_context(torch_profiler)

            # 1. DataLoader
            with record_function("DataLoader"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

            # 2. Data Transfer to GPU
            with record_function("DataTransfer"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(
                            device=torch.cuda.current_device(),
                            dtype=get_torch_dtype(args.model_dtype) if v.is_floating_point() else None
                        )
            
            scheduler.step()

            # Extract batch data for KeyeTokenizerEnd2EndImage
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            loss_mask = batch.get("loss_mask", None)
            pixel_values = batch.get("pixel_values", None)
            image_grid_thw = batch.get("image_grid_thw", None)
            pixel_values_videos = batch.get("pixel_values_videos", None)
            video_grid_thw = batch.get("video_grid_thw", None)
            
            # Process input_ids: set negative values to 0
            input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)
            
            # Generate labels based on loss_mask: mask tokens (e.g., image tokens) should not be predicted
            if loss_mask is not None:
                labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask)
                labels = labels.to(torch.int64)
            else:
                # Fallback: use input_ids as labels if no loss_mask provided
                labels = input_ids.clone()

            num_tokens = input_ids.numel()
            metrics.tokens.append(num_tokens)
            metrics.samples.append(input_ids.shape[0])

            # ================================================ Forward pass ================================================
            with record_function("Forward"):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    labels=labels,
                )
            
            # Get logits from model output
            logits = output["logits"]
            
            # ============ Compute LM loss externally (shifted labels) ============
            # Shift labels: remove first token, pad end with ignore_index
            # This aligns labels with logits for autoregressive prediction
            pad = torch.full(
                (labels.shape[0], 1), 
                loss_fn.ignore_index,
                dtype=labels.dtype
            ).to(device=labels.device, non_blocking=True)
            shifted_labels = torch.cat([labels[:, 1:], pad], dim=-1)
            
            # Get local sequence for context parallel (if enabled)
            local_labels = get_local_sequence(shifted_labels, seq_idx=1)
            
            # Compute LM loss and per-token loss
            lm_loss, per_token_loss = loss_fn(logits=logits, labels=local_labels)
            
            # Extract auxiliary losses from model output
            codebook_loss = output.get("codebook_loss", torch.tensor(0.0))
            commitment_loss = output.get("commitment_loss", torch.tensor(0.0))
            
            if isinstance(codebook_loss, (list, tuple)):
                # 如果是列表，求平均
                codebook_loss = sum(codebook_loss) / len(codebook_loss)

            if isinstance(commitment_loss, (list, tuple)):
                commitment_loss = sum(commitment_loss) / len(commitment_loss)
            
            total_loss = lm_loss + args.codebook_loss_weight * codebook_loss + args.commitment_loss_weight * commitment_loss
            
            # Record metrics
            metrics.loss.append(total_loss.detach())
            metrics.lm_loss.append(lm_loss.detach().item())
            metrics.codebook_loss.append(codebook_loss.detach().item() if isinstance(codebook_loss, torch.Tensor) else codebook_loss)
            metrics.commitment_loss.append(commitment_loss.detach().item() if isinstance(commitment_loss, torch.Tensor) else commitment_loss)
            # ================================================ End of Forward pass ================================================

            # ================================================ Backward pass ================================================
            with record_function("Backward"):
                total_loss.backward()
            
            with record_function("GradClip"):
                clip_grad_by_value(model, args.clip_range)

            # Update optimizer at gradient accumulation boundaries
            if scheduler.is_gradient_accumulation_boundary():
                with record_function("GradNorm"):
                    grad_norm = compute_fsdp_zero2_grad_norm(model)
                metrics.grad_norm.append(grad_norm)
                learning_rate = lr_scheduler.get_last_lr()[0]
                metrics.learning_rate.append(learning_rate)
                
                with record_function("OptimizerStep"):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            # ================================================ End of Backward pass ================================================

            metrics.step_time.tick()
            metrics.step()

            # Logging at specified intervals
            if scheduler.should_logging():
                metrics.write_logs(scheduler.global_step)

            # Save checkpoint at specified intervals
            if scheduler.should_save_checkpoint():
                if args.overfit_batches:
                    print_rank_0(f"Skipping checkpoint save at step {scheduler.global_step} (overfit debug mode)")
                else:
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
    if not args.overfit_batches:
        save_checkpoint(
            app_state=app_state,
            dist_checkpointer=dist_checkpointer,
            checkpoint_dir=args.output_dir,
            global_step=scheduler.global_step)
    else:
        print_rank_0(f"Skipping final checkpoint save (overfit debug mode)")
    
    print_rank_0("Training completed!")

if __name__ == "__main__":
    train()
