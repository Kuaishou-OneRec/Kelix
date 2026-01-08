"""AR（自回归）模型训练脚本。

- 训练muse/models/keye_ar/modeling.py的KeyeARModel模型
- 参数解析（训练/日志/分布式/恢复等）
- 配置加载（与 muse config 体系对齐）
- 模型构建与并行初始化（FSDP/TP/CP 等由框架内部控制）
- 数据集与 DataLoader 构建
- 训练主循环（交叉熵 LM loss，支持 loss_mask 生成 labels）
- checkpoint 保存/恢复

注：当前实现以 KeyeAR 模型的 forward 形参为准（tokens/cu_seqlens/input_pos/pixel_values/image_grid_thw）。

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
from muse.config import KeyeARConfig, load_config
from muse.data.datasets import ChatCompletionVisionDataset_keye_vitrope_slowfast
from muse.losses import CrossEntropyLoss
from muse.models import get_model_class
from muse.training.checkpoint import (
    AppState,
    DistributedCheckpointer,
    get_checkpoint_path,
    load_hf_checkpoint,
    save_checkpoint,
)
from muse.training.common import (
    clip_grad_by_value,
    compute_fsdp_zero2_grad_norm,
    freeze_params_by_pattern,
    get_torch_dtype,
    initialize_metrics,
    set_default_dtype,
    StepScheduler,
)
from muse.training.distributed import initialize_model_params, shard_model, load_from_full_model_state_dict
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
    parser.add_argument("--model-dir", type=str, default=None, help="checkpoint dir (optional)")
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="The config file path of the model to train (optional).",
    )
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
    parser.add_argument("--save-checkpoint-per-step", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--overfit-batches", type=int, default=0)

    # freeze
    parser.add_argument(
        "--freeze-params",
        type=str,
        default="",
        help=(
            "Parameter name patterns to freeze (comma-separated). Uses 'contains' matching. "
            "E.g., 'y_embedder,cross_attn' freezes all params containing these substrings. "
            "Use ^ prefix for inverse: '^y_embedder,^cross_attn' freezes all EXCEPT matching params"
        ),
    )

    # legacy/shortcut flags: --freeze_xxx will be converted to --freeze-params xxx
    parser.add_argument("--freeze-llm", action="store_true", help="Legacy alias -> freeze-params=llm")
    parser.add_argument("--freeze-navit", action="store_true", help="Legacy alias -> freeze-params=navit")
    parser.add_argument("--freeze-vision", action="store_true", help="Legacy alias -> freeze-params=vision")

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

    # 对齐 recipes/sana/train_sana_ar_dit.py：只有 rank0 写日志，避免多卡重复写文件
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return []

    loggers: list[Logger] = []
    loggers.append(Logger(backend=StdoutBackend()))
    loggers.append(Logger(backend=CSVBackend(str(output_dir / "metrics.csv"))))
    loggers.append(Logger(backend=TensorBoardBackend(str(output_dir))))

    if args.tensorboard:
        # 兼容旧习惯：也可以写到 tb 子目录
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
    # KeyeARModel 需要 KeyeARConfig
    model_cls = get_model_class(args.model_name)

    if args.model_dir:
        # continue pretrain mode: model_dir should contain config.json & weights
        model = model_cls.from_pretrained(args.model_dir)
    elif args.model_config:
        # train from scratch mode: build config from a json
        cfg = load_config(args.model_config, config_class=KeyeARConfig)
        model = model_cls(cfg)
    else:
        raise ValueError(
            "Either --model-dir (for continue pretrain) or --model-config (for train from scratch) must be provided."
        )

    # init/shard
    initialize_model_params(model)
    model = shard_model(model)
    model.train()

    # Freeze specified parameters (align with recipes/sana/train_sana_ar_dit.py)
    freeze_patterns: list[str] = []

    if args.freeze_params:
        freeze_patterns.extend([p.strip() for p in args.freeze_params.split(",") if p.strip()])

    # Convert legacy flags into patterns
    if getattr(args, "freeze_llm", False):
        freeze_patterns.append("llm")
    if getattr(args, "freeze_navit", False):
        freeze_patterns.append("navit")
    if getattr(args, "freeze_vision", False):
        freeze_patterns.append("vision")

    if freeze_patterns:
        frozen_count = freeze_params_by_pattern(model, freeze_patterns)
        print_rank_0(f"Frozen {frozen_count} parameters with patterns: {freeze_patterns}")

    # optimizer
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
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
        # KeyeARModel.forward:
        #   - tokens: (b, s)
        #   - input_pos: (b, s)
        #   - **kwargs 支持 cu_seqlens（packed sequences for flash-attn）
        #   - pixel_values+image_grid_thw: 提供则内部用visual_tokenizer得到image ids并expand
        #   - 返回：通常是 logits tensor (b, s, vocab)
        position_ids = batch.get("input_pos", None)
        if position_ids is None:
            position_ids = batch.get("position_ids", None)
        cu_seqlens = batch.get("cu_seqlens", None)
        pixel_values = batch.get("pixel_values", None)
        image_grid_thw = batch.get("image_grid_thw", None)

        if position_ids is None:
            if cu_seqlens is not None:
                # 参考 recipes/sana/train_sana_ar_dit.py：根据cu_seqlens构造packed input_pos
                input_pos_list = []
                for i in range(len(cu_seqlens) - 1):
                    seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
                    pos_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
                    input_pos_list.append(pos_ids)
                position_ids = torch.cat(input_pos_list, dim=0).unsqueeze(0)
            else:
                # 未packed fallback
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
                if position_ids.shape[0] != input_ids.shape[0]:
                    position_ids = position_ids.expand_as(input_ids)

        # labels: 用loss_mask将非监督位置置为ignore_index
        labels = _prepare_labels(input_ids, loss_mask, ignore_index=loss_fn.ignore_index)

        # forward
        with contextlib.nullcontext():
            logits = model(
                tokens=input_ids,
                input_pos=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                cu_seqlens=cu_seqlens,
            )

        loss = loss_fn(logits, labels)

        # 对齐 sana：append detached tensor，避免 hot path `.item()` 触发 CPU-GPU sync
        metrics.loss.append(loss.detach())
        metrics.tokens.append(input_ids.shape[1])

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
