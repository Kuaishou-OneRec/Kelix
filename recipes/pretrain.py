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
from transformers import AutoTokenizer

from collections import defaultdict

torch.autograd.set_detect_anomaly(True)
import gc
gc.disable()

process_group_timeout = datetime.timedelta(minutes=60*24)

# Muse imports
from muse.models import get_model_class, list_models
from muse.config import Qwen3Config
from muse.training.distributed import (
    shard_model, 
    load_from_full_model_state_dict,
    init_device_mesh
)
from muse.training.checkpoint import (
    AppState, 
    DistributedCheckpointer,
    load_hf_checkpoint
)
from muse.training.common import (
    set_default_dtype, 
    clip_grad_by_value, 
    compute_fsdp_zero2_grad_norm,
    Timer
)
from muse.training.lr_schedulers import get_scheduler
from muse.training.activations import set_activation_checkpointing
from muse.training.parallel import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_data_parallel_group,
    get_local_sequence,
    initialize_model_parallel,
    gather_by_group
)
from muse.utils.common import (
    set_random_seed, 
    print_rank_0,
    to_cuda,
    to_device,
    dist_reduce_dict
)

def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Model args ############
  parser.add_argument("--model-class", type=str, default="Qwen3Model",
                      help="The model class name registered in muse.models.",)

  parser.add_argument("--model-config", type=str, default=None,
                      help="The config file path of the model to train, e.g. model_dir/config.json")

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
                      help="The directory of the pretrained model.")

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
  parser.add_argument("--lr-scheduler-type", type=str, default="cosine_with_min_lr",
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
  
  parser.add_argument("--sequence-parallel-size", type=int, default=1,
                      help="Sequence parallelism size")

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
def save_model_checkpoint(
    model,
    save_dir: str,
    tag: str = None,
    client_state=None,
    optimizer = None,
    dataloader = None,
    lr_scheduler=None,
    app_state: AppState = None,
    dist_checkpointer: DistributedCheckpointer = None,
    global_step: int = None,
):
    """保存FSDP+TP模型的checkpoint

    Args:
        model: FSDP wrapped model
        save_dir: The directory to save the checkpoint
        checkpoint_id: The id of the checkpoint, if not specified, use timestamp
        client_state: The additional state to save
        dataloader: The dataloader to save
        lr_scheduler: The learning rate scheduler to save
        app_state: The app state to save
        dist_checkpointer: The dist checkpointer to save
        global_step: The global step to save
    """
    if dist.get_rank() == 0:
      os.makedirs(save_dir, exist_ok=True)
    
    # 生成checkpoint标签
    if tag is None:
      if global_step:
        tag = f"global_step{global_step}"
      else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = f"ckpt_{timestamp}"


    ckpt_path = os.path.join(save_dir, tag)
    if dist.get_rank() == 0:
      os.makedirs(ckpt_path, exist_ok=True)

      # Update latest file
      with open(os.path.join(save_dir, "latest"), "w") as f:
        f.write(tag)
    
    dist.barrier()

    # Configure FSDP state_dict
    full_state_dict_config = FullStateDictConfig(
      offload_to_cpu=True,
      rank0_only=True,
    )

    try:
        dist_checkpointer.save_checkpoint(
          state_dict={"app": app_state},
          output_dir=ckpt_path,           
          tag=tag
        )

        # Save dataloader state (if any)
        if dataloader is not None:
          try:
            dataloader_state = {
              "dataloader_state_dict": dataloader.state_dict()
            }
            dataloader_path = os.path.join(ckpt_path, "dataloader_ckpt")
            if dist.get_rank() == 0:
              os.makedirs(dataloader_path, exist_ok=True)
            dist.barrier()
            
            # Save dataloader state for each rank
            torch.save(
              dataloader_state,
              os.path.join(dataloader_path, f"rank{dist.get_rank()}.pt")
            )
            print_rank_0(f"Saved dataloader state to {dataloader_path}")
          except:
              import traceback
              logging.error(
                f"Failed to save dataloader state! dataloader" \
                f"({type(dataloader)})={dataloader} \n" \
                f"traceback:{traceback.format_exc()}")

        optimizer_path = os.path.join(ckpt_path, "optimizer_ckpt")
        optimizer_state = {
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": lr_scheduler.state_dict(),
        }
        if dist.get_rank() == 0:
          os.makedirs(optimizer_path, exist_ok=True)
        dist.barrier()
        torch.save(
          optimizer_state,
          os.path.join(optimizer_path, f"rank{dist.get_rank()}.pt")
        )
        print_rank_0(f"Saved dataloader state to {optimizer_path}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {str(e)}")
        raise e
    
    finally:
        # Ensure all processes are synchronized
        dist.barrier()


def get_resume_info(args):
  if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
    checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_id:
      checkpoint_id = args.checkpoint_id
    else:
      latest_file = os.path.join(checkpoint_dir, "latest")
      if os.path.exists(latest_file):
        with open(latest_file, encoding="utf-8") as f:
          checkpoint_id = f.read().strip()
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_id)
    if not os.path.exists(checkpoint_path):
      raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")
    return checkpoint_dir, checkpoint_id
  else:
    return None, None

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
  dataset_config["model_class"] = args.model_class
  if args.max_length:
    dataset_config["max_length"] = args.max_length
  
  # Add shuffle buffer and checkpoint parameters
  dataset_config["shuffle_buffer_size"] = args.shuffle_buffer_size
  dataset_config["enable_checkpointing"] = args.enable_dataset_checkpointing
  use_dataset_load_balance = dataset_config.get("use_load_balance", False) or \
    args.use_dataset_load_balance


  # torch init
  torch.cuda.set_device(local_rank)
  torch.distributed.init_process_group(
    rank=rank, world_size=world_size,
    timeout=process_group_timeout
  )
  device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))


  ### initialize model parallel group
  # Currently only support sequence parallelism
  initialize_model_parallel(sequence_parallel_size=args.sequence_parallel_size)
  print_rank_0(f"Sequence parallel size: {get_sequence_parallel_world_size()}")
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
  print_rank_0(f"Loading model class: {args.model_class}")
  
  try:
    model_cls = get_model_class(args.model_class)
    print_rank_0(f"Get model class: {model_cls.__name__}")
  except KeyError:
    print_rank_0(
      f"Unavailable model: {args.model_class}, " \
      f"please choose from available models: {list_models()}")
    return

  # Load state dict and convert using model's converter
  state_dict = None
  
  if args.model_dir and dist.get_rank() == 0:
    with set_default_dtype(args.model_dtype):
      print_rank_0(f"Loading checkpoint from: {args.model_dir}")
      hf_state_dict = load_hf_checkpoint(args.model_dir)
      # convert hf_state_dict to model_cls state_dict
      state_dict = model_cls.convert_hf_state_dict(hf_state_dict)

  dist.barrier()

  # TODO: support wandb
  tb_writer = None
  if dist.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
    tb_writer.add_text("comment", args.comment, 0)
    tb_writer.add_text("comment_id", args.commit_id, 0)

  # Instantiate model on meta device
  with set_default_dtype(args.model_dtype), torch.device("meta"):
    # don't load weights, only instantiate model from model_dir/config.json
    model = model_cls.from_pretrained(args.model_dir, load_weights=False)
    print_rank_0(f"Model instantiated: {type(model).__name__}")
  
  if args.enable_gradient_checkpointing:
    print_rank_0("Enable gradient checkpointing")
    set_activation_checkpointing(
      model, auto_wrap_policy=model.get_checkpointable_module_classes()
    )

  # upcast fp32 to maintain master weight.
  # 需要保存一个fp32的模型权重，否则优化器更新权重的精度会降低，影响收敛
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

  with Timer("Load state dict"):
    # Convert meta tensors to CUDA tensors
    load_from_full_model_state_dict(
      model=model, full_sd=state_dict,
      allow_random_init_params=args.allow_random_init_params
    )

  with torch.device(torch.cuda.current_device()):
    for m in model.modules():
      # RoPE is not covered in state dict
      if hasattr(m, "rope_init"):
        print_rank_0("Initialize RoPE")
        m.rope_init()

  # Check if all tensors are initialized
  for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
    assert tensor.device != torch.device("meta"), \
        f"{name} not initialized, device={tensor.device}"

  if args.compile:
    model = torch.compile(model)
    print_rank_0("Model compiled")

  if state_dict is not None:
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
  # prepare optimizer
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

  total_num_tokens = 0
  total_num_samples = 0
  global_step = 0
  app_state = AppState(model=model, optimizer=optimizer)
  dist_checkpointer = DistributedCheckpointer()
  if args.checkpoint_dir:
    print_rank_0(
      f"Resume from checkpoint: {args.checkpoint_dir}, tag={args.checkpoint_id}"
      f"load_weights_only={args.load_weights_only}")

    state_dict = {"app": app_state}
    # TODO: add get_checkpoint_path to utils
    checkpoint_path = dist_checkpointer.get_checkpoint_path(
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
  if not use_flops_balance:
    with Timer("Build dataloader"):
      # This would need to be implemented based on muse.data API
      # For now, skipping to allow script to run for testing
      print_rank_0("Warning: Dataloader creation not yet implemented for new API")
      print_rank_0(f"Dataset config: {dataset_config}")

  ##############
  torch_profiler = _init_profiler(
    output_dir=os.path.join(args.output_dir, "torch_profile")) \
      if args.enable_profile else None

  # TODO: move to muse.losses
  # Simple cross-entropy loss for language modeling
  def compute_loss(logits, labels, ignore_index=-100):
    """Compute cross-entropy loss for language modeling."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
      shift_logits.view(-1, shift_logits.size(-1)),
      shift_labels.view(-1),
      ignore_index=ignore_index
    )
    return loss

  start_time = time.time()
  start_time0 = start_time
  show_cnt = 1

  grad_norm = 0.0
  micro_step = 0

  # Setup data iterator
  if dataloader is not None:
    data_iter = iter(gather_by_group(dataloader, get_sequence_parallel_group()))
  else:
    print_rank_0("Warning: No dataloader available. Training loop will not run.")
    data_iter = iter([])

  tb_metrics_q = queue.Queue(maxsize=8)
  def write_tb_async(tb_writer, metrics_queue):
    while True:
      # metrics = metrics_queue.get()
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
      attention_mask = batch.get("attention_mask", None)
      sample_idx = batch.get("sample_idx", None)

      # Prepare labels for loss computation
      input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)
      labels = input_ids * loss_mask + (-100) * (1 - loss_mask)

      # Forward pass
      with Timer("Forward"):
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss for language modeling
        logits = output.logits if hasattr(output, 'logits') else output
        loss = compute_loss(logits, labels, ignore_index=-100)

      # print(f"X=111, rank={dist.get_rank()} current_gpu_memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
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
      log_acc_step = args.logging_per_step * args.gradient_accumulation_steps

      if global_step % args.logging_per_step == 0 and \
              (micro_step + 1) % args.gradient_accumulation_steps == 0:

        if dist.get_rank() == 0:
          learning_rate = lr_scheduler.get_last_lr()[0]
          end_time = time.time()
          sec_per_step = (end_time - start_time) / args.logging_per_step

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
          save_model_checkpoint(
            model=model,
            save_dir=args.output_dir,
            tag=f"step{global_step}",
            global_step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            app_state=app_state.set_call_back(converter.revert),
            dist_checkpointer=dist_checkpointer
          )

      if torch_profiler:
        torch_profiler.step()


  # Save final dataset checkpoint if enabled
  if args.enable_dataset_checkpointing and dataloader is not None and hasattr(dataloader, 'dataset'):
    try:
      dataloader.dataset.save_checkpoint(dist.get_rank(), global_step)
      print_rank_0(f"Saved final dataset checkpoint at step {global_step}")
    except Exception as e:
      print_rank_0(f"Failed to save final dataset checkpoint: {e}")

  save_model_checkpoint(
                      model=model,
                      save_dir=args.output_dir,
                      tag=f"step{global_step}",
                      global_step=global_step,
                      client_state={
                          "total_num_valid_tokens": total_num_valid_tokens,
                          "total_num_tokens": total_num_tokens,
                          "total_num_samples": total_num_samples,
                          "total_data_source_samples": total_data_source_samples,
                          "total_data_source_tokens": total_data_source_tokens,
                      },
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      dataloader=data_iter if use_flops_balance else dataloader,
                      app_state=app_state.set_call_back(converter.revert), # app_state.set_call_back(state_dict),
                      dist_checkpointer=dist_checkpointer,
                  )

if __name__ == "__main__":
  train()





