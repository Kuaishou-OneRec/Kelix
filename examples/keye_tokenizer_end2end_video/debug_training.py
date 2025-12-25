#!/usr/bin/env python3
"""
Debug script for testing training loop with fixed data.

This script loads a model and dataset, caches one packed batch,
and repeatedly trains on it to observe metrics like num_samples
and codebook_usage.
"""

import argparse
import json
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
import itertools
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from muse.models import get_model_class
from muse.config import load_config
from muse.data.datasets import ChatCompletionVisionDataset_keye_vitrope_slowfast_video
from muse.training.distributed import shard_model, load_from_full_model_state_dict, initialize_model_params
from muse.training.checkpoint import load_hf_checkpoint
from muse.training.common import set_default_dtype, get_torch_dtype, clip_grad_by_value, compute_fsdp_zero2_grad_norm
from muse.training.parallel import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_local_sequence,
    initialize_model_parallel,
    gather_by_group
)
from muse.training.lr_schedulers import get_scheduler
from muse.training.activations import set_activation_checkpointing
from muse.losses import CrossEntropyLoss
from muse.utils.common import (
    set_random_seed,
    print_rank_0,
    to_cuda,
    Timer
)
from muse.utils.metrics import Logger, StdoutBackend
from muse.training.common import initialize_metrics, StepScheduler
def compute_codebook_metrics(
    indices: list,
    codebook_size: int,
    n_q_tokens: int = 8
) -> tuple:
    """
    计算codebook的perplexity和usage指标。
    
    Args:
        indices: VQ indices列表，每个元素是一个codebook的indices tensor
        codebook_size: 码本大小
        n_q_tokens: 量化token数量
        
    Returns:
        global_perplexities: 每个codebook的perplexity列表
        codebook_usages: 每个codebook的usage列表
    """
    if indices is None:
        return [], []
    
    global_perplexities = []
    codebook_usages = []
    
    with torch.no_grad():
        for i, vq_indices in enumerate(indices):
            local_indices = vq_indices.flatten()
            
            # Handle single GPU case
            if dist.is_initialized():
                local_batch_size = local_indices.shape[0]
                world_size = dist.get_world_size()
                batch_sizes = torch.zeros(world_size, dtype=torch.long, device=local_indices.device)
                dist.all_gather_into_tensor(
                    batch_sizes, 
                    torch.tensor([local_batch_size], dtype=torch.long, device=local_indices.device)
                )
                
                max_batch_size = batch_sizes.max().item()
                padded_indices = torch.zeros(max_batch_size, dtype=local_indices.dtype, device=local_indices.device)
                padded_indices[:local_batch_size] = local_indices
                
                gathered_indices_list = [
                    torch.zeros(max_batch_size, dtype=local_indices.dtype, device=local_indices.device) 
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered_indices_list, padded_indices)
                
                global_indices = []
                for rank_idx, rank_indices in enumerate(gathered_indices_list):
                    valid_size = batch_sizes[rank_idx].item()
                    global_indices.append(rank_indices[:valid_size])
                global_indices = torch.cat(global_indices, dim=0)
            else:
                # Single GPU: use local indices directly
                global_indices = local_indices
            
            counts = torch.bincount(global_indices.long(), minlength=codebook_size)
            total_samples = global_indices.shape[0]
            
            avg_probs = counts.float() / total_samples
            non_zero_probs = avg_probs[avg_probs > 0]
            entropy = -torch.sum(non_zero_probs * torch.log(non_zero_probs + 1e-10))
            global_perplexity = torch.exp(entropy)
            codebook_usage = (counts > 0).sum().float() / codebook_size
            
            global_perplexities.append(global_perplexity.item())
            codebook_usages.append(codebook_usage.item())
    
    return global_perplexities, codebook_usages


def print_batch_info(batch, step):
    """Print detailed information about the batch."""
    sample_idx = batch.get("sample_idx")
    input_ids = batch.get("input_ids")
    
    print_rank_0(f"\n{'='*80}")
    print_rank_0(f"Step {step} - Batch Information")
    print_rank_0(f"{'='*80}")
    
    if input_ids is not None:
        print_rank_0(f"input_ids shape: {input_ids.shape}")
        print_rank_0(f"Total tokens: {input_ids.numel()}")
    
    if sample_idx is not None:
        sample_idx_flat = sample_idx.flatten()
        unique_samples = torch.unique(sample_idx_flat)
        max_idx = sample_idx.max().item()
        logical_samples = max_idx + 1
        
        print_rank_0(f"\nsample_idx:")
        print_rank_0(f"  shape: {sample_idx.shape}")
        print_rank_0(f"  unique values: {unique_samples.tolist()}")
        print_rank_0(f"  max index: {max_idx}")
        print_rank_0(f"  logical_samples (max_idx + 1): {logical_samples}")
        
        # Count tokens per sample
        for sample_id in unique_samples:
            if sample_id >= 0:
                mask = (sample_idx_flat == sample_id)
                count = mask.sum().item()
                print_rank_0(f"    Sample {sample_id.item()}: {count} tokens")
        
        padding_count = (sample_idx_flat == -1).sum().item()
        if padding_count > 0:
            print_rank_0(f"    Padding (-1): {padding_count} tokens")
    
    cu_seqlens = batch.get("cu_seqlens")
    if cu_seqlens is not None:
        print_rank_0(f"\ncu_seqlens: {cu_seqlens.tolist()}")
        print_rank_0(f"Number of sequences: {len(cu_seqlens) - 1}")


def print_metrics(metrics, step, n_q_tokens, codebook_size):
    """Print training metrics."""
    print_rank_0(f"\n{'='*80}")
    print_rank_0(f"Step {step} - Metrics")
    print_rank_0(f"{'='*80}")
    
    # Sample count
    if hasattr(metrics, 'samples') and len(metrics.samples) > 0:
        total_samples = sum(metrics.samples)
        avg_samples = metrics.samples.avg() if hasattr(metrics.samples, 'avg') else total_samples / len(metrics.samples)
        print_rank_0(f"\nSamples:")
        print_rank_0(f"  Total accumulated: {total_samples:.2f}")
        print_rank_0(f"  Average per step: {avg_samples:.2f}")
        print_rank_0(f"  Last step: {metrics.samples[-1]:.2f}")
    
    # Loss
    if hasattr(metrics, 'loss') and len(metrics.loss) > 0:
        print_rank_0(f"\nLoss:")
        print_rank_0(f"  Total: {metrics.loss[-1]:.4f}")
        if hasattr(metrics, 'lm_loss') and len(metrics.lm_loss) > 0:
            print_rank_0(f"  LM: {metrics.lm_loss[-1]:.4f}")
        if hasattr(metrics, 'codebook_loss') and len(metrics.codebook_loss) > 0:
            print_rank_0(f"  Codebook: {metrics.codebook_loss[-1]:.4f}")
        if hasattr(metrics, 'commitment_loss') and len(metrics.commitment_loss) > 0:
            print_rank_0(f"  Commitment: {metrics.commitment_loss[-1]:.4f}")
    
    # Codebook usage (image)
    if hasattr(metrics, 'avg_codebook_usage') and len(metrics.avg_codebook_usage) > 0:
        print_rank_0(f"\nCodebook Usage (Image):")
        print_rank_0(f"  Average: {metrics.avg_codebook_usage[-1]:.4f}")
        for i in range(n_q_tokens):
            usage_metric = getattr(metrics, f'codebook_usage_{i}', None)
            if usage_metric and len(usage_metric) > 0:
                print_rank_0(f"  Codebook {i}: {usage_metric[-1]:.4f}")
    
    # Codebook perplexity (image)
    if hasattr(metrics, 'avg_perplexity') and len(metrics.avg_perplexity) > 0:
        print_rank_0(f"\nCodebook Perplexity (Image):")
        print_rank_0(f"  Average: {metrics.avg_perplexity[-1]:.2f}")
        for i in range(n_q_tokens):
            ppl_metric = getattr(metrics, f'perplexity_{i}', None)
            if ppl_metric and len(ppl_metric) > 0:
                print_rank_0(f"  Codebook {i}: {ppl_metric[-1]:.2f}")
    
    # Codebook usage (video)
    if hasattr(metrics, 'video_avg_codebook_usage') and len(metrics.video_avg_codebook_usage) > 0:
        print_rank_0(f"\nCodebook Usage (Video):")
        print_rank_0(f"  Average: {metrics.video_avg_codebook_usage[-1]:.4f}")
        for i in range(n_q_tokens):
            usage_metric = getattr(metrics, f'video_codebook_usage_{i}', None)
            if usage_metric and len(usage_metric) > 0:
                print_rank_0(f"  Codebook {i}: {usage_metric[-1]:.4f}")
    
    # Codebook perplexity (video)
    if hasattr(metrics, 'video_avg_perplexity') and len(metrics.video_avg_perplexity) > 0:
        print_rank_0(f"\nCodebook Perplexity (Video):")
        print_rank_0(f"  Average: {metrics.video_avg_perplexity[-1]:.2f}")
        for i in range(n_q_tokens):
            ppl_metric = getattr(metrics, f'video_perplexity_{i}', None)
            if ppl_metric and len(ppl_metric) > 0:
                print_rank_0(f"  Codebook {i}: {ppl_metric[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Debug training loop with fixed data")
    
    # Model args
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Model directory path")
    
    # Dataset args
    parser.add_argument("--dataset-config", type=str, required=True,
                       help="Dataset config JSON file")
    parser.add_argument("--max-length", type=int, default=None,
                       help="Max sequence length (overrides config)")
    
    # Training args
    parser.add_argument("--num-training-steps", type=int, default=100,
                       help="Number of training steps")
    parser.add_argument("--overfit-batches", type=int, default=1,
                       help="Number of batches to cache and overfit")
    parser.add_argument("--logging-per-step", type=int, default=1,
                       help="Log every N steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    
    # Optimizer args
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--vision_lr", type=float, default=2e-5,
                       help="Vision learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                       help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Beta1 for Adam")
    parser.add_argument("--beta2", type=float, default=0.95,
                       help="Beta2 for Adam")
    
    # Model config
    parser.add_argument("--model-dtype", type=str, default="bfloat16",
                       choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--use-flash-attention-2", action="store_true",
                       help="Use flash attention 2")
    parser.add_argument("--enable-gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--context-parallel-size", type=int, default=1,
                       help="Context parallel size")
    
    # Loss weights
    parser.add_argument("--codebook-loss-weight", type=float, default=1.0,
                       help="Codebook loss weight")
    parser.add_argument("--commitment-loss-weight", type=float, default=0.25,
                       help="Commitment loss weight")
    
    # Freeze args
    parser.add_argument("--freeze-llm", action="store_true",
                       help="Freeze LLM")
    parser.add_argument("--freeze-navit", action="store_true",
                       help="Freeze NaViT")
    parser.add_argument("--freeze-navit-mlp-ar", action="store_true",
                       help="Freeze NaViT MLP AR")
    
    parser.add_argument("--seed", type=int, default=19260817,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize distributed training (single GPU for now)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        print_rank_0(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    else:
        print_rank_0("Running in single GPU mode")
    
    device_mesh = torch.distributed.device_mesh.init_device_mesh("cuda", mesh_shape=(world_size,))
    
    # Initialize model parallel
    initialize_model_parallel(context_parallel_size=args.context_parallel_size)
    print_rank_0(f"Context parallel size: {get_context_parallel_world_size()}")
    
    set_random_seed(args.seed)
    
    # Load dataset config
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = json.load(f)
    
    if args.max_length:
        dataset_config["max_length"] = args.max_length
    
    if not dataset_config.get("base_model_dir") and args.model_dir:
        dataset_config["base_model_dir"] = args.model_dir
    
    # Load model config
    model_config_path = Path(args.model_dir) / "muse_config.json"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Config not found: {model_config_path}")
    model_config = load_config(model_config_path)
    
    if args.use_flash_attention_2:
        model_config.qwen_config.attention_function = "flash_attention_2"
    
    model_class_name = model_config.model_class
    dataset_config["model_class"] = model_class_name
    
    # Get model class
    model_cls = get_model_class(model_class_name)
    print_rank_0(f"Model class: {model_cls.__name__}")
    
    # Load state dict
    state_dict = None
    if dist.get_rank() == 0:
        with set_default_dtype(args.model_dtype):
            print_rank_0(f"Loading checkpoint from: {args.model_dir}")
            state_dict = load_hf_checkpoint(args.model_dir)
    if world_size > 1:
        dist.barrier()
    
    # Create model
    with set_default_dtype(args.model_dtype), torch.device("meta"):
        print_rank_0("Creating model...")
        model = model_cls(model_config)
    
    if args.enable_gradient_checkpointing:
        from muse.training.activations import set_activation_checkpointing
        set_activation_checkpointing(
            model, auto_wrap_policy=model.get_checkpointable_module_classes()
        )
    
    if args.model_dtype == "float32" or hasattr(args, 'fp32_weight') and args.fp32_weight:
        model = model.float()
    
    # Shard model
    shard_model(
        model=model,
        cpu_offload=False,
        reshard_after_forward=False,
        dp_mesh=device_mesh,
        fp32_weight=False,
        prefetch_params_in_forward=False,
        fp32_reduce=False
    )
    if world_size > 1:
        dist.barrier()
    
    # Load weights
    if state_dict is not None:
        with Timer("Load state dict"):
            load_from_full_model_state_dict(model=model, full_sd=state_dict)
    else:
        with Timer("Initialize model parameters"):
            initialize_model_params(model)
    
    # Initialize RoPE
    with torch.device(torch.cuda.current_device()):
        for m in model.modules():
            if hasattr(m, "rope_init"):
                print_rank_0("Initialize RoPE")
                m.rope_init()
    
    # Freeze parameters
    if args.freeze_llm:
        for name, param in model.named_parameters():
            if not (name.startswith("visual_tokenizer") or name.startswith("quant_projector")):
                param.requires_grad = False
    
    if args.freeze_navit:
        for name, param in model.named_parameters():
            if name.startswith("visual_tokenizer.visual") and not name.startswith("visual_tokenizer.mlp_AR"):
                param.requires_grad = False
    
    if args.freeze_navit_mlp_ar:
        for name, param in model.named_parameters():
            if name.startswith("visual_tokenizer.mlp_AR"):
                param.requires_grad = False
    
    # Setup optimizer
    vision_lr = args.vision_lr if args.vision_lr > 0 else args.lr
    optimizer = torch.optim.AdamW(
        model.get_optimizer_grouped_parameters(
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            vision_learning_rate=vision_lr,
            vision_lr_layer_decay=1.0
        ),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1.0e-8
    )
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_training_steps,
        min_lr=args.min_lr
    )
    
    # Build dataset
    if dist.is_initialized():
        dataset_config["rank"] = dist.get_rank()
        dataset_config["world_size"] = dist.get_world_size()
    
    print_rank_0("Building dataset...")
    dataset = ChatCompletionVisionDataset_keye_vitrope_slowfast_video(**dataset_config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataset_config.get("num_workers", 0),
        collate_fn=lambda x: x[0]
    )
    
    # Cache batches for overfitting
    print_rank_0(f"Caching {args.overfit_batches} batch(es) for overfitting...")
    cached_batches = []
    if world_size > 1 and dist.is_initialized():
        temp_iter = iter(gather_by_group(dataloader, get_context_parallel_group()))
    else:
        temp_iter = iter(dataloader)
    for i in range(args.overfit_batches):
        try:
            batch = next(temp_iter)
            cached_batches.append(batch)
            print_rank_0(f"  Cached batch {i+1}/{args.overfit_batches}")
        except StopIteration:
            print_rank_0(f"Warning: Only {i} batches available")
            break
    
    if len(cached_batches) == 0:
        print_rank_0("Error: No batches cached!")
        return
    
    data_iter = iter(itertools.cycle(cached_batches))
    
    # Initialize metrics
    metrics = initialize_metrics(
        acc_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step,
        loggers=[Logger("stdout", [StdoutBackend()])] if dist.get_rank() == 0 else []
    )
    
    n_q_tokens = model_config.tokenizer_config.n_q_tokens
    codebook_size = model_config.tokenizer_config.codebook_size
    
    metrics.new("lm_loss", dtype="float", reduce="mean")
    metrics.new("codebook_loss", dtype="float", reduce="mean")
    metrics.new("commitment_loss", dtype="float", reduce="mean")
    metrics.new("avg_perplexity", dtype="float", reduce="mean")
    metrics.new("avg_codebook_usage", dtype="float", reduce="mean")
    metrics.new("video_avg_perplexity", dtype="float", reduce="mean")
    metrics.new("video_avg_codebook_usage", dtype="float", reduce="mean")
    
    for i in range(n_q_tokens):
        metrics.new(f"perplexity_{i}", dtype="float", reduce="mean")
        metrics.new(f"codebook_usage_{i}", dtype="float", reduce="mean")
        metrics.new(f"video_perplexity_{i}", dtype="float", reduce="mean")
        metrics.new(f"video_codebook_usage_{i}", dtype="float", reduce="mean")
    
    scheduler = StepScheduler(args)
    loss_fn = CrossEntropyLoss(ignore_index=-100, return_token_loss=True, shift_labels=False)
    
    print_rank_0("\n" + "="*80)
    print_rank_0("Starting training loop...")
    print_rank_0("="*80)
    
    model.train()
    
    for step in range(args.num_training_steps):
        # Get batch
        batch = next(data_iter)
        
        # Move to GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(
                    device=torch.cuda.current_device(),
                    dtype=get_torch_dtype(args.model_dtype) if v.is_floating_point() else None
                )
        
        scheduler.step()
        
        # Extract data
        sample_idx = batch.get("sample_idx")
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        loss_mask = batch.get("loss_mask", None)
        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)
        pixel_values_videos = batch.get("pixel_values_videos", None)
        video_grid_thw = batch.get("video_grid_thw", None)
        
        # Process input_ids
        input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)
        
        # Generate labels
        if loss_mask is not None:
            labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask)
            labels = labels.to(torch.int64)
        else:
            labels = input_ids.clone()
        
        # Record tokens and samples
        num_tokens = input_ids.numel()
        metrics.tokens.append(num_tokens)
        
        if sample_idx is not None:
            logical_samples = (sample_idx.max() + 1).item() / get_context_parallel_world_size()
            metrics.samples.append(logical_samples)
        else:
            metrics.samples.append(input_ids.shape[0] / get_context_parallel_world_size())
        
        # Forward pass
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
        )
        
        logits = output["logits"]
        
        # Compute loss
        pad = torch.full(
            (labels.shape[0], 1),
            loss_fn.ignore_index,
            dtype=labels.dtype
        ).to(device=labels.device, non_blocking=True)
        shifted_labels = torch.cat([labels[:, 1:], pad], dim=-1)
        local_labels = get_local_sequence(shifted_labels, seq_idx=1)
        
        lm_loss, per_token_loss = loss_fn(logits=logits, labels=local_labels)
        
        codebook_loss_raw = output.get("codebook_loss", torch.tensor(0.0))
        commitment_loss_raw = output.get("commitment_loss", torch.tensor(0.0))
        
        codebook_loss_list = codebook_loss_raw if isinstance(codebook_loss_raw, (list, tuple)) else [codebook_loss_raw]
        commitment_loss_list = commitment_loss_raw if isinstance(commitment_loss_raw, (list, tuple)) else [commitment_loss_raw]
        
        codebook_loss = sum(codebook_loss_list) / len(codebook_loss_list)
        commitment_loss = sum(commitment_loss_list) / len(commitment_loss_list)
        
        total_loss = lm_loss + args.codebook_loss_weight * codebook_loss + args.commitment_loss_weight * commitment_loss
        
        # Record metrics
        metrics.loss.append(total_loss.detach().item())
        metrics.lm_loss.append(lm_loss.detach().item())
        metrics.codebook_loss.append(codebook_loss.detach().item() if isinstance(codebook_loss, torch.Tensor) else codebook_loss)
        metrics.commitment_loss.append(commitment_loss.detach().item() if isinstance(commitment_loss, torch.Tensor) else commitment_loss)
        
        # Compute codebook metrics (image)
        vq_indices = output.get("indices", None)
        if vq_indices is not None:
            global_perplexities, codebook_usages = compute_codebook_metrics(
                indices=vq_indices,
                codebook_size=codebook_size,
                n_q_tokens=n_q_tokens
            )
            if global_perplexities:
                metrics.avg_perplexity.append(sum(global_perplexities) / len(global_perplexities))
                metrics.avg_codebook_usage.append(sum(codebook_usages) / len(codebook_usages))
                for i, (ppl, usage) in enumerate(zip(global_perplexities, codebook_usages)):
                    getattr(metrics, f"perplexity_{i}").append(ppl)
                    getattr(metrics, f"codebook_usage_{i}").append(usage)
        
        # Compute codebook metrics (video)
        video_vq_indices = output.get("video_indices", None)
        if video_vq_indices is not None:
            video_global_perplexities, video_codebook_usages = compute_codebook_metrics(
                indices=video_vq_indices,
                codebook_size=codebook_size,
                n_q_tokens=n_q_tokens
            )
            if video_global_perplexities:
                metrics.video_avg_perplexity.append(sum(video_global_perplexities) / len(video_global_perplexities))
                metrics.video_avg_codebook_usage.append(sum(video_codebook_usages) / len(video_codebook_usages))
                for i, (ppl, usage) in enumerate(zip(video_global_perplexities, video_codebook_usages)):
                    getattr(metrics, f"video_perplexity_{i}").append(ppl)
                    getattr(metrics, f"video_codebook_usage_{i}").append(usage)
        
        # Backward pass
        total_loss.backward()
        clip_grad_by_value(model, 1.0)
        
        if scheduler.is_gradient_accumulation_boundary():
            learning_rate = lr_scheduler.get_last_lr()[0]
            metrics.learning_rate.append(learning_rate)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        metrics.step()
        
        # Print detailed info
        if step % args.logging_per_step == 0 or step == 0:
            print_batch_info(batch, step)
            print_metrics(metrics, step, n_q_tokens, codebook_size)
        
        # Cleanup
        del output, logits, total_loss, lm_loss, per_token_loss
        del codebook_loss, commitment_loss
        if vq_indices is not None:
            del vq_indices
        if video_vq_indices is not None:
            del video_vq_indices
        del input_ids, labels, shifted_labels, local_labels
        if attention_mask is not None:
            del attention_mask
        if loss_mask is not None:
            del loss_mask
        if pixel_values is not None:
            del pixel_values
        if pixel_values_videos is not None:
            del pixel_values_videos
        del batch
        
        if step % 10 == 0:
            torch.cuda.empty_cache()
    
    print_rank_0("\n" + "="*80)
    print_rank_0("Training completed!")
    print_rank_0("="*80)


if __name__ == "__main__":
    main()

