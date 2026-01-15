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

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"



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
from muse.training.activations import set_activation_checkpointing

# muse imports
from muse.config import KeyeARConfig, load_config, model_config
from muse.data.datasets import ARChatCompletionVisionDataset
from muse.losses import CrossEntropyLoss, ChunkedLossComputer

from muse.models import get_model_class
from muse.training.checkpoint import (
    AppState,
    DistributedCheckpointer,
    get_checkpoint_path,
    load_hf_checkpoint,
    save_checkpoint,
)
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

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
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate (对齐 sana)")
    parser.add_argument("--max-length", type=int, default=16000, help="Maximum sequence length for packing (对齐 keye_tokenizer_end2end_video)")

    # precision
    parser.add_argument("--model-dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # data
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--chuncked-loss-compute-size", type=int, default=0, help="Chunk size for loss computation, 0 for not chunked")

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

    # FSDP options (对齐 sana)
    parser.add_argument("--cpu-offload", action="store_true", help="Offload to CPU") 
    parser.add_argument("--reshard-after-forward", action="store_true", help="Reshard after forward (Zero3)")
    parser.add_argument("--prefetch-params-in-forward", action="store_true", help="Prefetch parameters in forward pass") 
    parser.add_argument("--fp32-weight", action="store_true", help="Use fp32 for model weight updating")
    parser.add_argument("--fp32-reduce", action="store_true", help="Use fp32 for model gradient reduction")
    parser.add_argument("--allow-random-init-params", type=str, default="",
                        help="Parameter names to allow random initialization")
    parser.add_argument("--skip-load-params", type=str, default="",
                        help="Parameter name patterns to skip loading from checkpoint (comma-separated)")


    # distributed init
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--init-method", type=str, default="env://")

    # tensorboard
    parser.add_argument("--tensorboard", action="store_true")

    # metadata (对齐 sana 脚本)
    parser.add_argument("--comment", type=str, default="", help="Experiment comment/description")
    parser.add_argument("--commit-id", type=str, default="", help="Git commit hash")
    parser.add_argument("--enable-gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    return parser


def _setup_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    """Setup distributed training environment using OMPI environment variables.
    
    Returns:
        tuple: (rank, world_size, local_rank)
    """
    # 对齐 recipes/sana/train_sana_ar_dit.py：使用 OMPI 环境变量
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend=args.backend,
            rank=rank,
            world_size=world_size,
            timeout=process_group_timeout,
        )
    
    return rank, world_size, local_rank





def _load_dataset_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_config(args: argparse.Namespace) -> KeyeARConfig:
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


def _build_dataloader(args: argparse.Namespace) -> DataLoader:
    ds_cfg = _load_dataset_config(args.dataset_config)

    # 传递 max_length 到 dataset config（对齐 train_keye_tok_end2end_video）
    if args.max_length:
        ds_cfg["max_length"] = args.max_length

    dataset = ARChatCompletionVisionDataset(**ds_cfg)

    # dataset 内部通常会提供 collate_fn（如果没有，就用默认）
    collate_fn = getattr(dataset, "collate_fn", None)

    dataloader = DataLoader(
        dataset,
        batch_size=ds_cfg.get("batch_size", 1),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return dataloader


def _prepare_shifted_labels(input_ids: torch.Tensor, logits: torch.Tensor, loss_mask: Optional[torch.Tensor], ignore_index: int, model_config: KeyeARConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert input_ids.ndim == 3 and input_ids.shape[0] == 1 and input_ids.shape[2] == model_config.qwen_config.n_q_tokens + 1, \
        f"input_ids shape must be (1, seq_len, {model_config.qwen_config.n_q_tokens + 1}), but got {input_ids.shape}"
    assert loss_mask.ndim == 2 and loss_mask.shape[0] == 1 and loss_mask.shape[1] == input_ids.shape[1], \
        f"loss_mask shape must be (1, seq_len), but got {loss_mask.shape}"
    assert logits.shape[:2] == input_ids.shape[:2], \
        f"logits shape must be {input_ids.shape[:2] + (model_config.qwen_config.vocab_size,)}, but got {logits.shape}"
    loss_mask = loss_mask[:,:,None].repeat(1, 1, model_config.qwen_config.n_q_tokens + 1)
    q_eos_token = model_config.qwen_config.q_eos_token
    is_q_eos_token = input_ids == q_eos_token
    acc_q_eos_token = is_q_eos_token.cumsum(dim=2)
    loss_mask = loss_mask * (acc_q_eos_token <= 1)
    labels = input_ids * loss_mask + ignore_index * (1 - loss_mask)
    assert labels.shape == input_ids.shape, f"labels shape must be {input_ids.shape}, but got {labels.shape}"

    vocab_size = model_config.qwen_config.vocab_size

    n_text_token_per_pos = 2
    n_image_token_per_pos = model_config.qwen_config.n_q_tokens + 1
    
    text_weight = (n_text_token_per_pos + n_image_token_per_pos) / 2 / n_text_token_per_pos
    image_weight = (n_text_token_per_pos + n_image_token_per_pos) / 2 / n_image_token_per_pos
    is_text_token = (input_ids[:,:,:1] < vocab_size).repeat(1, 1, model_config.qwen_config.n_q_tokens + 1) & loss_mask
    is_image_token = (input_ids[:,:,:1] >= vocab_size).repeat(1, 1, model_config.qwen_config.n_q_tokens + 1) & loss_mask
    weights = text_weight * is_text_token + image_weight * is_image_token

    assert weights.shape == input_ids.shape, f"weights shape must be {input_ids.shape}, but got {weights.shape}"

    # shift for loss computation
    labels = labels[:,1:]
    weights = weights[:,1:]
    loss_mask = loss_mask[:,1:]
    logits = logits[:,:-1,:]
    return logits, labels.to(torch.int64), weights, loss_mask


def _load_model_config(args: argparse.Namespace) -> KeyeARConfig:
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
    return model_config


def _load_state_dict(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
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
    return state_dict


def freeze_params(args, model) -> None:
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

def squeeze0_if_dim_is_3(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    return tensor

def train() -> None:
    args = _build_arg_parser().parse_args()

    rank, world_size, local_rank = _setup_distributed(args)

    print_rank_0(f"rank/world_size/local_rank: {rank}/{world_size}/{local_rank}")
    print_rank_0(f"output_dir: {args.output_dir}")
    
    # 保存训练参数到 output_dir（对齐 sana）
    if rank == 0:
        args_str = json.dumps(vars(args), indent=2, ensure_ascii=False)
        print_rank_0(f"Training Arguments:\n{args_str}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(os.path.join(args.output_dir, f"args-{args.commit_id}-{timestamp}.json"), 'w', encoding="utf-8") as f:
            f.write(args_str + "\n")

    device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
    initialize_model_parallel()

    
    training_seed = args.seed + rank
    set_random_seed(training_seed)
    print_rank_0(f"Random seed: base={args.seed}, training_seed={training_seed} (rank={rank})")

    model_config = _load_model_config(args)

        
    # model
    # KeyeARModel 需要 KeyeARConfig
    with set_default_dtype(args.model_dtype), torch.device("meta"):
        model_cls = get_model_class(args.model_name)
        model_config.qwen_config.skip_output_layer = bool(args.chuncked_loss_compute_size > 0)
        model = model_cls(model_config)

    state_dict: Dict[str, Any] | None = _load_state_dict(args)

    # init/shard (对齐 sana)
    if args.enable_gradient_checkpointing:
        print_rank_0("Enable gradient checkpointing")
        set_activation_checkpointing(
            model, auto_wrap_policy=model.get_checkpointable_module_classes()
        )

    # fp32_weight mode: convert model to float32 before sharding
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
        fp32_reduce=args.fp32_reduce,
    )


    # Load weights or initialize parameters
    if args.model_dir:
        # Filter out buffers that should be initialized by rope_init
        # These buffers may exist in checkpoint but should not be loaded
        # because they will be re-initialized dynamically
        rope_buffer_patterns = [
            "position_ids",
            "inv_freq",
        ]
        if state_dict is not None and dist.get_rank() == 0:
            keys_to_remove = []
            for key in state_dict.keys():
                for pattern in rope_buffer_patterns:
                    if pattern in key:
                        keys_to_remove.append(key)
                        break
            for key in keys_to_remove:
                print_rank_0(f"Removing buffer from state_dict (will be initialized by rope_init): {key}")
                del state_dict[key]
        dist.barrier()

    # if rank == 0:
    #     print(f"state_dict")
    #     for k, v in state_dict.items():
    #         print(f"{k}: {v.shape}")

    #     print(f"\nmodel")
    #     for name, param in model.named_parameters():
    #         print(f"{name}: {param.shape}/{param.dtype}/{param.device}")


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

    freeze_params(args, model)

    # optimizer
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # lr scheduler (对齐 sana)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        min_lr=args.min_lr,
    )

    # loss
    loss_fn = CrossEntropyLoss(ignore_index=-100, shift_labels=False, return_token_loss=True)
    chunked_loss_computer = ChunkedLossComputer(
        lm_head=model.lm_head,
        loss_fn=loss_fn,
        minibatch_size=args.chuncked_loss_compute_size,
        shift_labels=False
    )

    # checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    app_state = AppState(model=model, optimizer=optimizer)
    dist_checkpointer = DistributedCheckpointer()

    # data
    dataloader = _build_dataloader(args)

    # Setup logging (对齐 sana)
    if rank == 0:
        stdout_logger = Logger("stdout", [StdoutBackend()])
        csv_logger = Logger("csv", [CSVBackend(os.path.join(args.output_dir, "metrics.csv"))])
        tb_logger = Logger("tb", [TensorBoardBackend(args.output_dir)])
        loggers = [stdout_logger, csv_logger, tb_logger]
    else:
        loggers = []

    # metrics & scheduler
    metrics = initialize_metrics(
        acc_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step,
        loggers=loggers,
    )
    step_scheduler = StepScheduler(args)

    model_config = _load_model_config(args)

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

        if image_grid_thw is not None: image_grid_thw = image_grid_thw[0]

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
        # labels = _prepare_labels(input_ids, loss_mask, ignore_index=loss_fn.ignore_index)
        if pixel_values.ndim == 5:
            pixel_values = pixel_values.squeeze(0)

        input_ids = squeeze0_if_dim_is_3(input_ids)
        position_ids = squeeze0_if_dim_is_3(position_ids)
        loss_mask = squeeze0_if_dim_is_3(loss_mask)
        # labels = squeeze0_if_dim_is_3(labels)
        cu_seqlens = cu_seqlens.flatten()

        # forward
        with contextlib.nullcontext():
            logits, expanded_ids = model(
                tokens=input_ids,
                input_pos=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                cu_seqlens=cu_seqlens,
                return_expanded_ids=True
            )

        logits, labels, weights, loss_mask = _prepare_shifted_labels(expanded_ids, logits, loss_mask, ignore_index=loss_fn.ignore_index, model_config=model.config)

        logits = logits.flatten(0,1)
        labels = labels.flatten(0,1)
        weights = weights.flatten(0,1)
        loss, per_token_loss = chunked_loss_computer.forward_and_backward(logits, labels, tokenwise_loss_weight=weights)

        # 对齐 sana：append detached tensor，避免 hot path `.item()` 触发 CPU-GPU sync
        metrics.loss.append(loss.detach())
        metrics.tokens.append(input_ids.shape[1])

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
