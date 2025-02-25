from typing import Dict, Any, Union, Optional

import contextlib
import gc
import argparse
import time
import datetime
import os
import glob
import json
import logging
import collections
import pickle
import itertools

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from transformers import AutoTokenizer, AutoProcessor
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

from recovlm.data.dataloaders import get_dataloader
from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.losses import CrossEntropyLoss
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0, \
  get_optimizer_grouped_parameters, dist_reduce_dict, Timer, heart_beat
from recovlm.training.lr_schedulers import get_scheduler

from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_rank, get_sequence_parallel_world_size, \
  get_local_sequence_boundary, initialize_model_parallel, gather_by_group, \
  get_local_sequence, get_data_parallel_group, get_data_parallel_world_size, \
  get_data_parallel_rank

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

from recovlm.training.distributed import shard_model, get_shard_conditions, \
  load_from_full_model_state_dict
from recovlm.training.checkpoint import load_hf_checkpoint

from recovlm.training.activations import set_activation_checkpointing

from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value

from recovlm.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer, Qwen2VLVisionBlock

# Logger 初始化
logging.basicConfig(level=logging.INFO)  # 设置日志级别
logger = logging.getLogger(__name__)  # 创建 logger 实例

def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Checkpoint args ############
  parser.add_argument("--model_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  parser.add_argument("--resume_from", type=str, default=None,
                      help="Specify the checkpoint directory to resume from.")

  parser.add_argument("--resume_from_tag", type=str, default=None,
                      help="Specify the checkpoint tag to resume from.")
  
  parser.add_argument("--resume_dataloader", action="store_true",
                      help="Whether to resume dataloader checkpoint")
  
  parser.add_argument("--auto_resume_local_latest", action="store_true",
                      help="Auto resume checkpoint from output dir if the latest ckpt exists." \
                            "Note: If the latest ckpt exists and the this option is enabled, " \
                            "the --resume_dataloader switch will be turned on, " \
                            "while the --load_weights_only option will be turned off.")
  
  parser.add_argument("--save_checkpoint_per_step", type=int, default=1000,
                      help="The number of steps to save a checkpoint")

  parser.add_argument("--save_checkpoint_every_epoch", action="store_true",
                      help="Save checkpoint at the end of every epoch")
  
  parser.add_argument("--load_weights_only", action="store_true",
                      help="Only load model weights.")

  parser.add_argument("--merge_checkpoint", action="store_true",
                      help="Merge the checkpoint files into a single file")

  parser.add_argument("--merge_checkpoint_dtype", type=str, default="fp16",
                      choices=["fp32", "fp16", "bf16"],
                      help="The dtype of the merged checkpoint file")

  parser.add_argument(
      "--merge_checkpoint_output_file",
      type=str,
      default="pytorch_model.bin",
      help="The name of the merged checkpoint file")
  
  parser.add_argument("--output_dir", type=str, default=None,
                      help="The directory to write the trained model")

  ############ Dataset args ############
  parser.add_argument("--dataset_config", type=str, default=None,
                      help="The comma seperated path of indexed json file.")

  parser.add_argument("--dataset", type=str, default=None,
                      help="The comma seperated path of indexed json file.")
  
  parser.add_argument("--data_format", type=str, default="chatml",
                      help="The data format of training, one of `chatml` and `completion`")

  parser.add_argument("--min_visual_tokens", type=int, default=16,
                      help="The max visual tokens to use")

  parser.add_argument("--max_visual_tokens", type=int, default=512,
                      help="The max visual tokens to use")

  parser.add_argument("--max_length", type=int, default=None,
                      help="Max tokens per sentence in corpus")
  
  ############ Learning Rate Args ############
  parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr",
                      help="The type of learning rate scheduler.")

  parser.add_argument("--num_warmup_steps", type=int, default=0,
                      help="The number of warmup steps to do.")
  
  parser.add_argument("--num_decay_steps", type=int, default=1000,
                      help="The number of steps to decay.")

  parser.add_argument("--num_training_steps", type=int, default=1000,
                      help="The number of training steps to do.")

  parser.add_argument("--num_epochs", type=int, default=1,
                      help="Number of epochs to train, no effect for pretraining.")
  
  parser.add_argument("--min_lr", type=float, default=1e-6,
                      help="The minimum learning rate to reach after the cosine schedule.")
  
  ############ Optimizer Args ############
  parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="The peak learning rate for optimizer.")
  
  parser.add_argument("--vision_learning_rate", type=float, default=-1.0,
                      help="The peak vit learning rate for optimizer." \
                           "Note: vision_learning_rate will be set to learning_rate if vision_learning_rate < 0.0")
  
  parser.add_argument("--vision_lr_layer_decay", type=float, default=1.0,
                      help="Decay vit learning rate by layers.")

  parser.add_argument("--weight_decay", type=float, default=0.1,
                      help="The weight decay for Adam Optimizer")
  
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="beta1 for Adam Optimizer")

  parser.add_argument("--beta2", type=float, default=0.95,
                      help="beta2 for Adam Optimizer")

  ############ Training Args ############

  parser.add_argument("--clip_range", type=float, default=None,
                      help="The gradient clip range.")

  parser.add_argument("--freeze_llm", action="store_true",
                      help="Freeze LLM parameters.")

  parser.add_argument("--freeze_visual", action="store_true",
                      help="Freeze visual encoder parameters.")
  
  parser.add_argument("--freeze_visual_without_adapter", action="store_true",
                      help="Only freeze visual encoder parameters, train adapter parameters.")

  parser.add_argument("--use_flash_attention_2", action="store_true",
                      help="Whether to use flash attention 2")

  parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                      help="Enable gradient checkpointing during training")
  
  parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Gradient accumulation steps")
  
  parser.add_argument("--sequence_parallel_size", type=int, default=1,
                      help="Enable gradient checkpointing during training")

  parser.add_argument("--logging_per_step", type=int, default=100,
                      help="The number of steps to log training info")

  parser.add_argument("--comment", type=str, default=None,
                      help="Comment of this experiment.")

  parser.add_argument("--commit_id", type=str, default=None,
                      help="Git commit id for experiment.")

  parser.add_argument("--seed", type=int, default=123,
                      help="Manual seed for RNG")

  parser.add_argument("--monitor_datasource_loss", action="store_true",
                      help="Whether to monitor loss of each datasource")

  parser.add_argument("--monitor_datasource_cnt", action="store_true",
                      help="Whether to monitor cnt of each datasource")

  ############ System Vars ############

  parser.add_argument("--kml_id", type=str, default=None,
                      help="KML_ID")

  parser.add_argument("--kml_task_id", type=str, default=None,
                      help="KML_TASK_ID")
  
  parser.add_argument("--heartbeat_monitor", action="store_true",
                      help="Whether to upload heartbeat to remote")
  

  return parser

def get_resume_info(args):
  # return: ckpt_folder, ckpt_tag, rewrite_flag
  if not args.auto_resume_local_latest:
    # Add validation for manual resume path
    if args.resume_from and not os.path.exists(args.resume_from):
      raise ValueError(f"Resume checkpoint directory {args.resume_from} does not exist")
    
    if args.resume_from and args.resume_from_tag:
      ckpt_path = os.path.join(args.resume_from, args.resume_from_tag)
      if not os.path.exists(ckpt_path):
        raise ValueError(f"Resume checkpoint path {ckpt_path} does not exist")
      
    return args.resume_from, args.resume_from_tag, False
  else:
    # check local ckpt
    latest_file = os.path.join(args.output_dir, "latest")
    if os.path.exists(latest_file):
      with open(latest_file, encoding="utf-8") as f:
        ckpt_id = f.read().strip()  # Add strip() to remove whitespace
      
      # Validate checkpoint exists
      ckpt_path = os.path.join(args.output_dir, ckpt_id)
      if not os.path.exists(ckpt_path):
        raise ValueError(f"Latest checkpoint path {ckpt_path} does not exist")
        
      print_rank_0(f"Check output_ckpt exists, auto resume from output_folder." \
                   f"checkpoint: resume_from={args.output_dir}, resume_tag={ckpt_id}")
      return args.output_dir, ckpt_id, True
    else:
      return args.resume_from, args.resume_from_tag, False

def train():
  arg_parser = get_argument_parser()
  args = arg_parser.parse_args()

  # check vision_lr
  assert args.learning_rate > 0.0
  if args.vision_learning_rate < 0.0:
    args.vision_learning_rate = args.learning_rate

  assert all([args.commit_id, args.seed, args.comment]), \
    "Git commit, seed, and comment is required for reproducibility"

  assert all([args.kml_id, args.kml_task_id]), \
    "Kml task infomation, for task alive monitor."

  assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
      "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
      "save_checkpoint_every_epoch should be set."

  # init model params
  os.environ["KML_ID"] = args.kml_id
  os.environ["KML_TASK_ID"] = args.kml_task_id
  rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
  world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
  local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
  # torch init
  torch.cuda.set_device(local_rank)
  torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
  device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))

  ### initialize model parallel group
  initialize_model_parallel(args.sequence_parallel_size)
  print_rank_0(f"Sequence parallel size: {get_sequence_parallel_world_size()}")

  set_random_seed(args.seed)

  state_dict = None
  if dist.get_rank() == 0:
    with set_default_dtype(torch.bfloat16):
      state_dict = load_hf_checkpoint(args.model_dir)

  dist.barrier()

  if dist.get_rank() == 0:
    args_dict = vars(args)
    args_str = json.dumps(args_dict, indent=4, ensure_ascii=False)
    print_rank_0(f"Training Arguments:\n{args_str}")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(args.output_dir,
          f"args-{args.commit_id}-{timestamp}.json"), 'w',
        encoding="utf-8") as f:
      f.write(args_str + "\n")

  tb_writer = None
  if dist.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
    tb_writer.add_text("comment", args.comment, 0)
    tb_writer.add_text("comment_id", args.commit_id, 0)
    tb_writer.add_text("kml_id", args.kml_id, 0)
    tb_writer.add_text("kml_task_id", args.kml_task_id, 0)

  with set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
      args.model_dir, _attn_implementation="flash_attention_2",
      use_cache=False
    )
  
  # check all param & buffer on meta device 
  for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")

  if args.enable_gradient_checkpointing:
    print_rank_0("Enable gradient checkpointing")
    # 使用FSDP时，hf的gradient_checkpointing_enable()不会生效
    # model.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={"use_reentrant": False})
    set_activation_checkpointing(
      model, auto_wrap_policy={Qwen2VLDecoderLayer, Qwen2VLVisionBlock}
    )

  shard_model(
    model=model,
    shard_conditions=[get_shard_conditions],
    cpu_offload=False,
    reshard_after_forward=True,
    dp_mesh=device_mesh,
  )
  dist.barrier()

  with Timer("Load state dict"):
    load_from_full_model_state_dict(model=model, full_sd=state_dict)
  
  if state_dict is not None:
    del state_dict

  with torch.device(torch.cuda.current_device()):
    for m in model.modules():
      # RoPE is not covered in state dict
      if hasattr(m, "rope_init"):
        print_rank_0("Initialize RoPE")
        m.rope_init()
  
  # 确保任何参数都被正确初始化
  for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
    assert not tensor.device == torch.device("meta"), \
      f"{name} not initialized, device={tensor.device}"

  if args.freeze_llm:
    print_rank_0("Freeze LLM parameters.")
    for name, param in model.named_parameters():
      if not name.startswith("visual"):
        print_rank_0(f"Disable LLM grad: {name}")
        param.requires_grad = False
    print_rank_0("=" * 50)

  if args.freeze_visual:
    print_rank_0("Freeze visual encoder parameters.")
    for name, param in model.named_parameters():
      if name.startswith("visual"):
        print_rank_0(f"Disable visual encoder grad: {name}")
        param.requires_grad = False
    print_rank_0("=" * 50)

  if args.freeze_visual_without_adapter:
    print_rank_0("Freeze visual encoder parameters. Train visual adapter parameters")
    for name, param in model.named_parameters():
      if name.startswith("visual") and not name.startswith("visual.merger."):
        print_rank_0(f"Disable visual encoder grad: {name}")
        param.requires_grad = False
    print_rank_0("=" * 50)
  
  # print train params log
  for name, param in model.named_parameters():
    if param.requires_grad:
      print_rank_0(f"params not freeze: {name}")
  print_rank_0("=" * 50)

  # Split weights in two groups, one with weight decay and the other not.
  optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model,
    learning_rate=args.learning_rate,
    vision_learning_rate=args.vision_learning_rate,
    weight_decay=args.weight_decay,
    no_decay_name_list=[
      "bias", "norm1", "norm2", "visual.merger.ln_q",
      "input_layernorm",
      "post_attention_layernorm",
      "model.norm"
    ],
    vision_learning_rate_layer_dacay=args.vision_lr_layer_decay
  )

  # prepare optimizer
  optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
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
  total_num_valid_tokens = 0
  dataloader_state_dict = None
  local_acc_data_source_samples = collections.defaultdict(int)
  total_data_source_tokens = collections.defaultdict(int)

  resume_from, ckpt_id, rewrite_resume_flag = get_resume_info(args)

  if rewrite_resume_flag:
    args.resume_dataloader = True
    args.load_weights_only = False
    print_rank_0(f"WARN: --resume_dataloader is rewrited to True \n" \
                 f"WARN: --load_weights_only is rewrited to False \n")
    
  if ckpt_id:
    ckpt_path = os.path.join(resume_from, ckpt_id)
    print_rank_0(
      f"Resume from checkpoint: {ckpt_path}, "
      f"load_weights_only={args.load_weights_only}")
    
    if not os.path.exists(ckpt_path):
      raise ValueError(f"Checkpoint path {ckpt_path} does not exist")
      
    _, client_state = model.load_checkpoint(
      resume_from, ckpt_id, load_module_only=args.load_weights_only)

    if args.resume_dataloader:
      dataloader_resume_path = os.path.join(resume_from, "dataloader_ckpt", f"rank{dist.get_rank()}_{ckpt_id}.pth")
      # Add validation for dataloader checkpoint
      if not os.path.exists(dataloader_resume_path):
        print_rank_0(f"Warning: Dataloader checkpoint {dataloader_resume_path} does not exist")
        print_rank_0("Will start training without resuming dataloader state")
        dataloader_state_dict = None
      else:
        try:
          dataloader_state_dict = torch.load(dataloader_resume_path)["dataloader_state_dict"]
          print_rank_0(f"Successfully loaded dataloader state from {dataloader_resume_path}")
        except Exception as e:
          print_rank_0(f"Error loading dataloader checkpoint: {str(e)}")
          print_rank_0("Will start training without resuming dataloader state")
          dataloader_state_dict = None

    if not args.load_weights_only:
      total_num_tokens = client_state.get("total_num_tokens", 0)
      total_num_samples = client_state.get("total_num_samples", 0)
      total_num_valid_tokens = client_state.get("total_num_valid_tokens", 0)

      # accumulate total_data_source_samples to rank0, 0 init others.
      if dist.get_rank() == 0:
        local_acc_data_source_samples.update(client_state.get("total_data_source_samples", {}))
        total_data_source_tokens.update(client_state.get("total_data_source_tokens", {}))

  dist.barrier()

  processor = Qwen2VLProcessor.from_pretrained(args.model_dir)

  ##############
  with open(args.dataset_config, encoding="utf-8") as f:
    dataset_config = json.loads(f.read())
  dataset = dataset_config.pop("name")
  if args.max_length:
    print_rank_0(
      f"Overwrite max_length in dataset_config: "
      f"{dataset_config['max_length']} -> {args.max_length}")
    dataset_config["max_length"] = args.max_length
  
  if dist.get_rank() == 0:
    with open(os.path.join(args.output_dir,
        f"dataset-{args.commit_id}-{timestamp}.json"), 'w',
        encoding="utf-8") as f:
      f.write(json.dumps(
        dataset_config, ensure_ascii=False, indent=2) + "\n")

  with Timer("Build dataloader"):
    dataloader = get_dataloader(name=dataset, **dataset_config)
    if args.resume_dataloader and dataloader_state_dict is not None:
      dataloader.load_state_dict(dataloader_state_dict)

  ##############

  loss_fn = CrossEntropyLoss(
    ignore_index=-100, return_token_loss=True, shift_labels=False)

  start_time = time.time()
  show_cnt = 1

  # Metrics, acc_ account for gradient accumulation
  # TODO: use mestrics manager
  acc_avg_loss = 0.0
  acc_num_tokens = 0
  acc_num_samples = 0
  acc_valid_num_tokens = 0
  batch_data_source_loss = collections.defaultdict(float)
  batch_data_source_tokens = collections.defaultdict(int)
  valid_data_source_tokens = collections.defaultdict(int)
  grad_norm = 0.0
  global_step = 0
  # get_sequence_parallel_group("gloo")
  for micro_step, batch in enumerate(gather_by_group(dataloader, get_sequence_parallel_group())):
    if show_cnt > 0 and dist.get_rank() == 0:
      with Timer("Show data"):
        input_text = processor.tokenizer.decode(batch['input_ids'][0])
        print_rank_0(
            f"Input Text:\n\n{input_text}\n" + "=" * 100 + "\n\n")
        print_rank_0(batch)
        show_cnt -= 1
    data_source = batch.pop("data_source", None) # dataset source list cur batch
    to_cuda(batch)
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch.get("attention_mask", None)
    pixel_values = batch.get("pixel_values", None)
    pixel_values_videos = batch.get("pixel_values_videos", None)
    image_grid_thw = batch.get("image_grid_thw", None)
    video_grid_thw = batch.get("video_grid_thw", None)
    cu_seqlens = batch.get("cu_seqlens", None)
    sample_idx = batch["sample_idx"]

    # 打印 token 数量
    token_count = input_ids.numel()  # 计算 token 数量
    print_rank_0(f"Iteration {micro_step}: Token count = {token_count}")

    num_tokens = input_ids.numel()
    num_samples = (sample_idx.max() + 1).sum()
    num_valid_tokens = num_tokens - (sample_idx == -1).sum()

    token_metrics = torch.tensor(
      [num_tokens, num_samples, num_valid_tokens]).cuda()
    dist.all_reduce(
      token_metrics, op=dist.ReduceOp.SUM, group=get_data_parallel_group())

    num_tokens = token_metrics[0]
    num_samples = token_metrics[1]
    num_valid_tokens = token_metrics[2]

    total_num_samples += num_samples.item()
    total_num_tokens += num_tokens.item()
    total_num_valid_tokens += num_valid_tokens.item()

    acc_num_samples += num_samples.item()
    acc_num_tokens += num_tokens.item()
    acc_valid_num_tokens += num_valid_tokens.item()

    input_ids = input_ids * (input_ids > 0).to(torch.int64)
    labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask)

    with Timer("Fwd"):
      output = model(
        input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
        cu_seqlens=cu_seqlens
      )
      # (b, N/P, V)
      logits = output.logits

      # 提前shift logits & labels
      pad = torch.full((labels.shape[0], 1), loss_fn.ignore_index,
          dtype=labels.dtype).to(device=labels.device)
      labels = torch.cat([labels[:, 1:], pad], dim=-1) # shift
      local_labels = get_local_sequence(labels, seq_idx=1)
      loss, per_token_loss = loss_fn(logits=logits, labels=local_labels)

      # del logits
      # del labels
      # del local_labels

    with Timer("bwd"):
      loss.backward(loss)
      clip_grad_by_value(model, args.clip_range)
      if (micro_step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        grad_norm = get_global_grad_norm(model)
        print_rank_0(grad_norm, [grad_norm])
        optimizer.zero_grad()
        global_step += 1

    ########## dataset source monitor ###############
    if args.monitor_datasource_loss:
      # WARN: assume batch_size = 1
      local_sample_idx = get_local_sequence(sample_idx).squeeze()
      unique_sample_idx = local_sample_idx.unique()

      for s_idx in unique_sample_idx:
        if s_idx < 0:
          continue
        mask = local_sample_idx == s_idx
        sum_loss = per_token_loss[mask].sum()

        key = data_source[int(s_idx.item())]
        batch_data_source_loss[key] += sum_loss.item()
        batch_data_source_tokens[key] += mask.sum().item()
        valid_data_source_tokens[key] += mask[local_labels.squeeze() != loss_fn.ignore_index].sum().item()

    if args.monitor_datasource_cnt:
      for data_source_name in data_source:
        local_acc_data_source_samples[data_source_name] += 1
  
    #########################################
    avg_loss = torch.tensor(loss.item()).cuda()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    acc_avg_loss += avg_loss

    if global_step % args.logging_per_step == 0 and \
            (micro_step + 1) % args.gradient_accumulation_steps == 0:

      with Timer("reduce data source metrics"):
        batch_data_source_loss = dist_reduce_dict(batch_data_source_loss)
        batch_data_source_tokens = dist_reduce_dict(batch_data_source_tokens)
        valid_data_source_tokens = dist_reduce_dict(valid_data_source_tokens)
        total_data_source_samples = dist_reduce_dict(
          local_acc_data_source_samples, group=get_data_parallel_group())
        for ds_key, ds_num_tokens in batch_data_source_tokens.items():
          total_data_source_tokens[ds_key] += ds_num_tokens
        

      if dist.get_rank() == 0:
        model_lrs = lr_scheduler.get_last_lr()
        learning_rate = model_lrs[0]
        if len(model_lrs) > 2:
          vision_learning_rate = lr_scheduler.get_lr()[2]
        else:
          vision_learning_rate = lr_scheduler.get_lr()[1]
        end_time = time.time()
        sec_per_step = (end_time - start_time) / args.gradient_accumulation_steps
        tokens_per_sec_per_gpu = \
          acc_num_tokens / dist.get_world_size() / (end_time - start_time)
        samples_per_sec_per_gpu = \
          acc_num_samples / dist.get_world_size() / (end_time - start_time)
        valid_tokens_per_sec_per_gpu = \
          acc_valid_num_tokens / dist.get_world_size() / (end_time - start_time)
        avg_loss = acc_avg_loss / args.gradient_accumulation_steps
        start_time = end_time
        log_dict = {
          "training/loss": avg_loss,
          "training/grad_norm": grad_norm,
          "training/learning_rate": learning_rate,
          "training/vision_learning_rate": vision_learning_rate,
          "perf/sec_per_step": sec_per_step,
          "perf/tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
          "perf/samples_per_sec_per_gpu": samples_per_sec_per_gpu,
          "perf/total_num_tokens": total_num_tokens,
          "perf/total_num_samples": total_num_samples,
          "perf/valid_total_num_tokens": total_num_valid_tokens,
          "perf/valid_tokens_per_sec_per_gpu": valid_tokens_per_sec_per_gpu,
          "perf/valid_token_ratio": total_num_valid_tokens / total_num_tokens,
        }

        for name, data in log_dict.items():
          if data is not None and tb_writer:
            tb_writer.add_scalar(
                name,
                data,
                global_step=global_step,
                new_style=True)

            # log metric by valid tokens
            if name.startswith("training/"):
              tb_writer.add_scalar(
                f"x_token_{name}",
                data,
                global_step=total_num_valid_tokens,
                new_style=True
              )

        if args.monitor_datasource_loss and tb_writer:
          for key, loss_sum in batch_data_source_loss.items():
            tb_writer.add_scalar(
                  f"data_source_loss/{key}",
                  loss_sum / valid_data_source_tokens[key],
                  global_step=global_step,
                  new_style=True)

        if args.monitor_datasource_cnt and tb_writer:
          for key, samples in total_data_source_samples.items():
            tb_writer.add_scalar(
                f"data_source_sample_ratio/{key}",
                1.0 * samples / total_num_samples,
                global_step=global_step,
                new_style=True)

          for key, num_tokens in total_data_source_tokens.items():
            tb_writer.add_scalar(
                f"data_source_token_ratio/{key}",
                1.0 * num_tokens / total_num_valid_tokens,
                global_step=global_step,
                new_style=True)

        print_rank_0(
          f"Step: {global_step}, Loss: {avg_loss}, "
          f"Grad Nrom: {grad_norm}",
          f"Learning Rate: {learning_rate}, "
          f"Sec per Step: {sec_per_step}",
          f"tokens_per_sec_per_gpu: {tokens_per_sec_per_gpu}",
          f"samples_per_sec_per_gpu: {samples_per_sec_per_gpu}",
          f"total_num_tokens: {total_num_tokens}",
          f"total_num_samples: {total_num_samples}",
          f"valid_tokens_per_sec_per_gpu: {valid_tokens_per_sec_per_gpu}, "
          f"total_num_tokens: {total_num_tokens}, "
          f"total_num_samples: {total_num_samples}, "
          f"total_num_valid_tokens: {total_num_valid_tokens}, "
          f"valid_tokens_ratio: {1.0 * total_num_valid_tokens / total_num_tokens}, "
        )

        # upload heart_beat to remote
        if args.heartbeat_monitor:
          heart_beat(int(acc_num_tokens))

      acc_avg_loss = 0.0
      acc_num_samples = 0
      acc_num_tokens = 0
      acc_valid_num_tokens = 0
      batch_data_source_loss = collections.defaultdict(float)
      batch_data_source_tokens = collections.defaultdict(int)
      valid_data_source_tokens = collections.defaultdict(int)

    if global_step % args.save_checkpoint_per_step == 0 and \
        global_step > 0 and (micro_step + 1) % args.gradient_accumulation_steps == 0:
      
      torch.cuda.empty_cache()

      with Timer("save checkpoint"):
        model.save_checkpoint(
          save_dir=args.output_dir, client_state = {
            "total_num_valid_tokens": total_num_valid_tokens,
            "total_num_tokens": total_num_tokens,
            "total_num_samples": total_num_samples,
            "total_data_source_samples": total_data_source_samples,
            "total_data_source_tokens": total_data_source_tokens,
          }
        )
        try:
          dataloader_state_dict = {
            "dataloader_state_dict": dataloader.state_dict()
          }
        except:
          dataloader_state_dict = None
          logging.error(f"Dataloader cannot dump state_dict!!!!!!!!")
        if dataloader_state_dict is not None:
          # dataloader ckpt
          dataloader_path = os.path.join(args.output_dir, "dataloader_ckpt")
          if dist.get_rank() == 0:
            os.makedirs(dataloader_path, exist_ok=True)
          dist.barrier()
          torch.save(
            dataloader_state_dict,
            os.path.join(
              dataloader_path,
              f"rank{dist.get_rank()}_global_step{global_step}.pth")
            )

  print_rank_0("Save checkpoint..")
  model.save_checkpoint(save_dir=args.output_dir, client_state = {
      "total_num_valid_tokens": total_num_valid_tokens,
      "total_num_tokens": total_num_tokens,
      "total_num_samples": total_num_samples,
      "total_data_source_samples": total_data_source_samples,
      "total_data_source_tokens": total_data_source_tokens
    }
  )
  try:
    # dataloader ckpt
    dataloader_state_dict = {
      "dataloader_state_dict": dataloader.state_dict()
    }
  except:
    dataloader_state_dict = None
    logging.error(f"Dataloader cannot dump state_dict!!!!!!!!")
    
  if dataloader_state_dict is not None:
    if dist.get_rank() == 0:
      os.makedirs(dataloader_path, exist_ok=True)
    dist.barrier()
    torch.save(
      dataloader_state_dict, os.path.join(dataloader_path,
      f"rank{dist.get_rank()}_global_step{global_step}.pth")
    )

  if args.merge_checkpoint and dist.get_rank() == 0:
    convert_zero_checkpoint_to_state_dict(
        args.output_dir,
        output_file=args.merge_checkpoint_output_file,
        dtype=args.merge_checkpoint_dtype
    )

  if dist.get_rank() == 0:
    logging.info("Training finished!")

if __name__ == "__main__":
  train()
