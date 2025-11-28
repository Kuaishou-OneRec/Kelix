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
from transformers import AutoTokenizer

from collections import defaultdict

torch.autograd.set_detect_anomaly(True)
import gc
gc.disable()

process_group_timeout = datetime.timedelta(minutes=60*24)

# TODO:
# 1. Add dataset checkpointing
# 2. Add wandb support
# 3. Add perf logging support

# Muse imports
from muse.models import get_model_class, list_models
from muse.config import get_config
from muse.training.distributed import (
    shard_model, 
    load_from_full_model_state_dict
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
from muse.data.datasets import TextDataset
from muse.losses.ce import CrossEntropyLoss

from muse.config import load_config

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

  parser.add_argument("--allow-random-init-params", type=str, default='',
                      help="Parameter names to allow random initialization")
  
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

  ############ Profile Args ############

  parser.add_argument("--enable-profile", action="store_true",
                      help="Enable torch profile")

  return parser

# TODO: move to muse.utils
def _init_profiler(output_dir) -> None:
    import torch.distributed as D
    import os
    if not os.path.exists(output_dir):
      if D.get_rank() == 0:
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

# TODO: move to muse.training.checkpoint

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

  ##############
  with open(args.dataset_config, encoding="utf-8") as f:
    dataset_config = json.loads(f.read())

  dataset = dataset_config.pop("name")
  
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
      "(for train from scratch) must be provided."
    )

  if args.use_flash_attention_2:
    model_config.attention_function = "flash_attention_2"
    print_rank_0("Use flash attention 2")
  else:
    print_rank_0("Warning: Use eager attention, performance may be degraded.")

  model_class_name = model_config.model_class
  dataset_config["model_class"] = model_class_name
  
  if args.max_length:
    dataset_config["max_length"] = args.max_length
  
  # Set tokenizer_path from model_dir if not specified
  if not dataset_config.get("tokenizer_path") and args.model_dir:
    dataset_config["tokenizer_path"] = args.model_dir

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

  # TODO: support wandb
  tb_writer = None
  if dist.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
    tb_writer.add_text("comment", args.comment, 0)
    tb_writer.add_text("comment_id", args.commit_id, 0)

  # Instantiate model on meta device, this is to avoid OOM
  with set_default_dtype(args.model_dtype), torch.device("meta"):
    # Train from scratch: create model with random initialization
    print_rank_0(f"Creating model from config: {args.model_config}")
    model = model_cls(model_config)
    print_rank_0(f"Model instantiated from config: {type(model).__name__}")
  
  if args.enable_gradient_checkpointing:
    print_rank_0("Enable gradient checkpointing")
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
  # 需要保证每个rank都执行了load_from_full_model_state_dict
  if args.model_dir:
    with Timer("Load state dict"):
      # Convert meta tensors to CUDA tensors
      # distribute the state_dict from rank 0 to all ranks
      load_from_full_model_state_dict(
        model=model, full_sd=state_dict,
        allow_random_init_params=args.allow_random_init_params
      )

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

  # TODO: support other optimizers
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

  global_step = 0
  app_state = AppState(model=model, optimizer=optimizer)
  dist_checkpointer = DistributedCheckpointer()
  if args.checkpoint_dir:
    print_rank_0(
      f"Resume from checkpoint: {args.checkpoint_dir}, tag={args.checkpoint_id}"
      f"load_weights_only={args.load_weights_only}")

    state_dict = {"app": app_state}
    # TODO: add get_checkpoint_path to utils
    checkpoint_path = get_checkpoint_path(
      args.checkpoint_dir, args.checkpoint_id)

    dist_checkpointer.load_checkpoint(
        state_dict=state_dict,
        checkpoint_path=checkpoint_path,
    )

    print_rank_0(f"Successfully loaded model using distributed checkpoint")

  dist.barrier()

  # Load tokenizer for decoding/debugging purposes
  tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

  if dist.get_rank() == 0:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(args.output_dir,
        f"dataset-{args.commit_id}-{timestamp}.json"), 'w',
        encoding="utf-8") as f:
      f.write(json.dumps(
        dataset_config, ensure_ascii=False, indent=2) + "\n")

  # Build dataloader
  # Note: For now, assume dataloader is provided via dataset config
  # TODO: Implement proper dataloader creation using muse.data
  dataloader = None
  with Timer("Build dataloader"):
    # This would need to be implemented based on muse.data API
    # For now, skipping to allow script to run for testing
    print_rank_0(f"Building dataloader with config: {dataset_config}")
    dataset = TextDataset(**dataset_config)
    dataloader = DataLoader(
      dataset,
      batch_size=1,
      shuffle=False,
      num_workers=dataset_config["num_workers"],
      collate_fn=lambda x: x[0]
    )

  ##############
  torch_profiler = _init_profiler(
    output_dir=os.path.join(args.output_dir, "torch_profile")) \
      if args.enable_profile else None

  # TODO: move to muse.losses
  # Simple cross-entropy loss for language modeling

  loss_fn = CrossEntropyLoss(ignore_index=-100, shift_labels=True)

  start_time = time.time()

  grad_norm = 0.0
  micro_step = 0
  acc_avg_loss = 0.0

  # Setup data iterator
  if dataloader is not None:
    data_iter = iter(gather_by_group(dataloader, get_context_parallel_group()))
  else:
    print_rank_0("Warning: No dataloader available. Training loop will not run.")
    data_iter = iter([])

  tb_metrics_q = queue.Queue(maxsize=8)
  def write_tb_async(tb_writer, metrics_queue):
    while True:
      global_step, log_dict = metrics_queue.get()
      for name, data in log_dict.items():
        if data is not None and tb_writer:
          tb_writer.add_scalar(
            name, data, global_step=global_step, new_style=True)

  if dist.get_rank() == 0:
    tb_writer_t = threading.Thread(
      target=write_tb_async, args=(
        tb_writer, tb_metrics_q))
    tb_writer_t.daemon = True
    tb_writer_t.start()

  while True:
    with contextlib.ExitStack() as ctx:
      if torch_profiler:
        ctx.enter_context(torch_profiler)

      try:
        batch = next(data_iter)
      except StopIteration:
        break

      micro_step += 1

      to_cuda(batch)

      # Extract batch data
      input_ids = batch["input_ids"]
      loss_mask = batch["loss_mask"]

      # Prepare labels for loss computation
      input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)
      labels = input_ids * loss_mask + (-100) * (1 - loss_mask)

      # Forward pass
      with Timer("Forward"):
        output = model(input_ids=input_ids)
        
        # Compute loss for language modeling
        logits = output.logits if hasattr(output, 'logits') else output
        loss = loss_fn(logits, labels)

      # Backward pass
      with Timer("Backward"):
        loss.backward()
        clip_grad_by_value(model, args.clip_range)

        if (micro_step + 1) % args.gradient_accumulation_steps == 0:
          grad_norm = compute_fsdp_zero2_grad_norm(model)
          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          global_step += 1

      # Accumulate loss
      avg_loss = loss.detach().item()
      acc_avg_loss += avg_loss

      # Logging

      if global_step % args.logging_per_step == 0 and \
        (micro_step + 1) % args.gradient_accumulation_steps == 0:

        if dist.get_rank() == 0:
          learning_rate = lr_scheduler.get_last_lr()[0]
          end_time = time.time()

          avg_loss_value = acc_avg_loss / args.logging_per_step
          
          log_dict = {
            "training/loss": avg_loss_value,
            "training/grad_norm": grad_norm,
            "training/learning_rate": learning_rate
          }
          start_time = end_time

          metrics_info = (global_step, log_dict)
          tb_metrics_q.put(metrics_info)
          
          print_rank_0(
            f"Step: {global_step}, Loss: {avg_loss_value:.4f}, "
            f"Learning Rate: {learning_rate:.2e}, GradNorm: {grad_norm:.2f}"
          )
    
      if (global_step % args.save_checkpoint_per_step == 0 or global_step in [100, 200]) and \
          global_step > 0 and (micro_step + 1) % args.gradient_accumulation_steps == 0:
        
        torch.cuda.empty_cache()
        gc.collect()

        with Timer("save checkpoint"):
          save_checkpoint(
            app_state=app_state,
            dist_checkpointer=dist_checkpointer,
            checkpoint_dir=args.output_dir,
            global_step=global_step
          )

      if torch_profiler:
        torch_profiler.step()

  save_checkpoint(
    app_state=app_state,
    dist_checkpointer=dist_checkpointer,
    checkpoint_dir=args.output_dir,
    global_step=global_step)

if __name__ == "__main__":
  train()
