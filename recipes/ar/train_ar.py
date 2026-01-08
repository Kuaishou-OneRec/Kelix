"""AR（自回归）模型训练脚本。

该脚本仿照 `recipes/train_keye_tok_end2end.py` 的组织方式，提供：
- 参数解析（训练/日志/分布式/恢复等）
- 配置加载（与 muse config 体系对齐）
- 模型构建与并行初始化（FSDP/TP/CP 等由框架内部控制）
- 数据集与 DataLoader 构建
- 训练主循环（交叉熵 LM loss，支持 loss_mask 生成 labels）
- checkpoint 保存/恢复

注：当前实现以 KeyeAR 模型的 forward 形参为准（tokens/cu_seqlens/input_pos/pixel_values/image_grid_thw）。
如果你使用的 AR 模型 forward 签名不同，请告诉我模型名和 batch 字段，我再对齐。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# muse imports
from muse.config import load_config
from muse.data.datasets import ChatCompletionVisionDataset_keye_vitrope_slowfast
from muse.losses import CrossEntropyLoss
from muse.models import get_model_class
from muse.training.checkpoint import (
    AppState,
    DistributedCheckpointer,
    get_checkpoint_path,
    load_from_full_model_state_dict,
    load_hf_checkpoint,
    save_checkpoint,
)
from muse.training.common import (
    clip_grad_by_value,
    compute_fsdp_zero2_grad_norm,
    get_torch_dtype,
    initialize_metrics,
    set_default_dtype,
    StepScheduler,
)
from muse.training.distributed import initialize_model_params, shard_model
from muse.training.lr_schedulers import get_scheduler
from muse.training.parallel import (
    gather_by_group,
    get_context_parallel_group,
    initialize_model_parallel,
)
from muse.utils.common import Timer, print_rank_0, set_random_seed, to_cuda
from muse.utils.metrics import CSVBackend, Logger, StdoutBackend, TensorBoardBackend

logger = logging.getLogger(__name__)

gc.disable()
process_group_timeout = datetime.timedelta(minutes=60 * 24)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Muse AR training")

    # basic
    parser.add_argument("--model-dir", type=str, default=None, help="HF checkpoint dir or model config dir")
    parser.add_argument("--model-name", type=str, default="KeyeARModel", help="muse.models registry name")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True, help="dataset json config")

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--clip-range", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", help="see muse.training.lr_schedulers")
    parser.add_argument("--warmup-steps", type=int, default=100)

    # precision
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    # data
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")

    # logging/checkpoint
    parser.add_argument("--logging-per-step", type=int, default=10)
    parser.add_argument("--save-per-step", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--overfit-batches", type=int, default=0)

    # distributed init
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--init-method", type=str, default="env://")

    # tensorboard
    parser.add_argument("--tensorboard", action="store_true")

    return parser


def _setup_distributed(args: argparse.Namespace) -> None:
    if dist.is_initialized():
        return
    dist.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        timeout=process_group_timeout,
    )


def _build_loggers(args: argparse.Namespace) -> list[Logger]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loggers: list[Logger] = []
    loggers.append(Logger(backend=StdoutBackend()))
    loggers.append(Logger(backend=CSVBackend(str(output_dir / "train.csv"))))
    if args.tensorboard:
        loggers.append(Logger(backend=TensorBoardBackend(str(output_dir / "tb"))))
    return loggers


def _load_dataset_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_dataloader(args: argparse.Namespace) -> DataLoader:
    ds_cfg = _load_dataset_config(args.dataset_config)

    dataset = ChatCompletionVisionDataset_keye_vitrope_slowfast(**ds_cfg)

    # dataset 内部通常会提供 collate_fn（如果没有，就用默认）
    collate_fn = getattr(dataset, "collate_fn", None)

    dataloader = DataLoader(
        dataset,
        batch_size=ds_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return dataloader


def _prepare_labels(input_ids: torch.Tensor, loss_mask: Optional[torch.Tensor], ignore_index: int) -> torch.Tensor:
    # input_ids: (bs, seq_len)
    input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)
    if loss_mask is None:
        return input_ids.clone().to(torch.int64)
    labels = input_ids * loss_mask + ignore_index * (1 - loss_mask)
    return labels.to(torch.int64)


def train() -> None:
    args = _build_arg_parser().parse_args()

    _setup_distributed(args)
    initialize_model_parallel()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # logging
    loggers = _build_loggers(args)
    print_rank_0(f"rank/world_size: {rank}/{world_size}")
    print_rank_0(f"output_dir: {args.output_dir}")

    # dtype & seed
    torch_dtype = get_torch_dtype(args.dtype)
    set_default_dtype(torch_dtype)
    set_random_seed(args.seed)

    # model
    model_cls = get_model_class(args.model_name)
    model = model_cls.from_pretrained(args.model_dir) if args.model_dir else model_cls()

    # init/shard
    initialize_model_params(model)
    model = shard_model(model)
    model.train()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # lr scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # loss
    loss_fn = CrossEntropyLoss(ignore_index=-100, shift_labels=True)

    # checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    app_state = AppState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    dist_checkpointer = DistributedCheckpointer(app_state=app_state)

    if args.resume:
        ckpt_path = get_checkpoint_path(str(output_dir), args.resume_step)
        print_rank_0(f"Resuming from checkpoint: {ckpt_path}")
        load_from_full_model_state_dict(app_state, dist_checkpointer.load(ckpt_path))
    elif args.model_dir:
        # 允许从 HF 目录加载（如果里面是 muse 兼容的权重）
        try:
            load_hf_checkpoint(app_state, args.model_dir)
            print_rank_0(f"Loaded HF checkpoint from {args.model_dir}")
        except Exception as e:
            print_rank_0(f"Skip HF checkpoint load: {e}")

    # data
    dataloader = _build_dataloader(args)

    # metrics & scheduler
    metrics = initialize_metrics(
        acc_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step,
        loggers=loggers,
    )
    step_scheduler = StepScheduler(args)

    # iterator
    if args.overfit_batches and args.overfit_batches > 0:
        print_rank_0(f"=== OVERFIT DEBUG MODE: Caching {args.overfit_batches} batches ===")
        cached_batches = []
        tmp_iter = iter(gather_by_group(dataloader, get_context_parallel_group()))
        for i in range(args.overfit_batches):
            try:
                cached_batches.append(next(tmp_iter))
            except StopIteration:
                print_rank_0(f"Warning: only {i} batches available")
                break
        data_iter = iter(itertools.cycle(cached_batches))
    else:
        data_iter = iter(gather_by_group(dataloader, get_context_parallel_group()))

    # train loop
    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        to_cuda(batch)

        step_scheduler.step()

        input_ids = batch["input_ids"]
        loss_mask = batch.get("loss_mask", None)
        position_ids = batch.get("position_ids", None)
        cu_seqlens = batch.get("cu_seqlens", None)
        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)

        labels = _prepare_labels(input_ids, loss_mask, ignore_index=loss_fn.ignore_index)

        # forward
        with contextlib.nullcontext():
            output = model(
                tokens=input_ids,
                cu_seqlens=cu_seqlens,
                input_pos=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        logits = output.logits if hasattr(output, "logits") else output
        loss = loss_fn(logits, labels)

        metrics.loss.append(loss.detach().item())
        metrics.tokens.append(input_ids.shape[1])
        if cu_seqlens is not None:
            metrics.samples.append(cu_seqlens.shape[0])

        # backward
        loss.backward()
        clip_grad_by_value(model, args.clip_range)

        if step_scheduler.is_gradient_accumulation_boundary():
            grad_norm = compute_fsdp_zero2_grad_norm(model)
            metrics.grad_norm.append(grad_norm)

            lr = lr_scheduler.get_last_lr()[0]
            metrics.learning_rate.append(lr)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metrics.step_time.tick()
        metrics.step()

        if step_scheduler.should_logging():
            metrics.write_logs(step_scheduler.global_step)

        if step_scheduler.should_save_checkpoint():
            if args.overfit_batches and args.overfit_batches > 0:
                print_rank_0(
                    f"Skipping checkpoint save at step {step_scheduler.global_step} (overfit debug mode)"
                )
            else:
                torch.cuda.empty_cache()
                with Timer("save checkpoint"):
                    save_checkpoint(
                        app_state=app_state,
                        dist_checkpointer=dist_checkpointer,
                        checkpoint_dir=args.output_dir,
                        global_step=step_scheduler.global_step,
                    )

        if step_scheduler.global_step >= args.max_steps:
            break

    # final save
    if not (args.overfit_batches and args.overfit_batches > 0):
        save_checkpoint(
            app_state=app_state,
            dist_checkpointer=dist_checkpointer,
            checkpoint_dir=args.output_dir,
            global_step=step_scheduler.global_step,
        )


if __name__ == "__main__":
    train()
