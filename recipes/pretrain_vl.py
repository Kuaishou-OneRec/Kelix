import argparse
import time
import datetime
import os
import glob
import json
import logging
import collections
import pickle

import torch
import deepspeed
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam

# from transformers import AutoTokenizer, AutoProcessor
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

from recovlm.data.dataloaders import get_dataloader
from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.losses import CrossEntropyLoss
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0, \
  get_optimizer_grouped_parameters, dist_reduce_dict
from recovlm.training.lr_schedulers import get_scheduler

from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_rank, get_sequence_parallel_world_size, \
  get_local_sequence_boundary, initialize_model_parallel, gather_by_group

def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Checkpoint args ############
  parser.add_argument("--model_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  parser.add_argument("--resume_from", type=str, default=None,
                      help="Specify the checkpoint directory to resume from.")

  parser.add_argument("--resume_from_tag", type=str, default=None,
                      help="Specify the checkpoint tag to resume from.")
  
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
  
  parser.add_argument("--weight_decay", type=float, default=0.1,
                      help="The weight decay for Adam Optimizer")
  
  parser.add_argument("--beta1", type=float, default=0.9,
                      help="beta1 for Adam Optimizer")

  parser.add_argument("--beta2", type=float, default=0.95,
                      help="beta2 for Adam Optimizer")

  ############ Training Args ############

  parser.add_argument("--freeze_llm", action="store_true",
                      help="Freeze LLM parameters.")

  parser.add_argument("--freeze_visual", action="store_true",
                      help="Freeze visual encoder parameters.")

  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Reserved for deepspeed framework")

  parser.add_argument("--use_flash_attention_2", action="store_true",
                      help="Whether to use flash attention 2")

  parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                      help="Enable gradient checkpointing during training")
  
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

  return parser

def train():
  arg_parser = get_argument_parser()
  arg_parser = deepspeed.add_config_arguments(arg_parser)
  args = arg_parser.parse_args()

  assert all([args.commit_id, args.seed, args.comment]), \
    "Git commit, seed, and comment is required for reproducibility"

  assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
      "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
      "save_checkpoint_every_epoch should be set."

  deepspeed.init_distributed()

  ### initialize model parallel group
  initialize_model_parallel(args.sequence_parallel_size)
  print_rank_0(f"Sequence parallel size: {get_sequence_parallel_world_size()}")

  set_random_seed(args.seed)
  torch.distributed.barrier()


  if dist.get_rank() == 0:
    args_dict = vars(args)
    args_str = json.dumps(args_dict, indent=4, ensure_ascii=False)
    print_rank_0(f"Training Arguments:\n{args_str}")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(args.output_dir, f"args-{args.commit_id}-{timestamp}.json"), 'w',
        encoding="utf-8") as f:
      f.write(args_str + "\n")

  tb_writer = None
  if dist.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))

  # enabled=False when zero stage < 3
  with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config, enabled=False):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
      args.model_dir, _attn_implementation="flash_attention_2", use_cache=False
    )

  if args.freeze_llm:
    print_rank_0("Freeze LLM parameters.")
    for name, param in model.named_parameters():
      if not name.startswith("visual"):
        print_rank_0(f"Disable LLM grad: {name}")
        param.requires_grad = False

  if args.freeze_visual:
    print_rank_0("Freeze visual encoder parameters.")
    for name, param in model.named_parameters():
      if name.startswith("visual"):
        print_rank_0(f"Disable visual encoder grad: {name}")
        param.requires_grad = False

  if args.enable_gradient_checkpointing:
    print_rank_0("Enable gradient checkpointing")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

  # Split weights in two groups, one with weight decay and the other not.
  optimizer_grouped_parameters = get_optimizer_grouped_parameters(
      model, args.weight_decay)

  # prepare optimizer
  optimizer = FusedAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        betas=(args.beta1, args.beta2),
                        eps=1.0e-8)
  lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
    min_lr=args.min_lr,
    num_stop_steps=20
  )

  model.train()
  model, optimizer, _, lr_scheduler = deepspeed.initialize(args=args,
                                                           model=model,
                                                           optimizer=optimizer,
                                                           lr_scheduler=lr_scheduler)

  total_num_tokens = 0
  total_num_samples = 0
  total_data_source_cnt = {}
  if not args.resume_from:
    args.resume_from = args.output_dir
  ckpt_id = args.resume_from_tag
  latest = os.path.join(args.resume_from, "latest")
  if not ckpt_id and os.path.exists(latest):
    with open(latest, encoding="utf-8") as f:
      ckpt_id = f.read()
  if ckpt_id:
    print_rank_0(
      f"Resume from checkpoint: {os.path.join(args.resume_from, ckpt_id)}, "
      f"load_weights_only={args.load_weights_only}")
    _, client_state = model.load_checkpoint(
      args.resume_from, ckpt_id, load_module_only=args.load_weights_only)
    if not args.load_weights_only:
      total_num_tokens = client_state.get("total_num_tokens", 0)
      total_num_samples = client_state.get("total_num_samples", 0)

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

  dataloader = get_dataloader(name=dataset, **dataset_config)
  ##############

  loss_fn = CrossEntropyLoss(
    ignore_index=-100, return_token_loss=True, shift_labels=False)

  start_time = time.time()
  show_cnt = 1

  acc_step = 0
  acc_avg_loss = 0.0
  acc_num_tokens = 0
  acc_num_samples = 0
  acc_fwd_time = 0.0
  acc_bwd_time = 0.0
  acc_data_fetch_time = 0.0

  s = time.time()
  for batch in gather_by_group(dataloader, get_sequence_parallel_group("gloo")):
    acc_data_fetch_time = time.time() - s
    if show_cnt > 0 and dist.get_rank() == 0:
      print_rank_0(batch)
      print_rank_0(
          f"Input Text:\n\n{processor.tokenizer.decode(batch['input_ids'][0])}\n"
          f"=" * 100 + "\n\n")
      show_cnt -= 1

    data_source = batch.pop("data_source") # dataset source list cur batch
    to_cuda(batch)
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch.get("attention_mask", None)
    pixel_values = batch.get("pixel_values", None)
    pixel_values_videos = batch.get("pixel_values_videos", None)
    image_grid_thw = batch.get("image_grid_thw", None)
    video_grid_thw = batch.get("video_grid_thw", None)
    cu_seqlens = batch.get("cu_seqlens", None)
    sample_idx = batch["sample_idx"].to(torch.int32)

    num_tokens = torch.tensor(input_ids.shape[-1]).cuda()
    num_samples = torch.tensor(cu_seqlens.shape[-1] - 1).cuda()

    dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    total_num_tokens += num_tokens.item()
    total_num_samples += num_samples.item()

    acc_num_tokens += num_tokens.item()
    acc_num_samples += num_samples.item()

    input_ids = input_ids * (input_ids > 0).to(torch.int64)
    labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask)

    s = time.time()
    output = model(
      input_ids, attention_mask=attention_mask,
      pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
      image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
      cu_seqlens=cu_seqlens
    )
    # (b, N/P, V)
    logits = output.logits
  
    # 提前shirft logits & labels
    start, end = get_local_sequence_boundary(labels.shape[-1])
    pad = torch.full((labels.shape[0], 1), loss_fn.ignore_index, dtype=labels.dtype).to(
        device=torch.cuda.current_device())
    labels = torch.cat([labels[:, 1:], pad], dim=-1) # shirft
    local_labels = labels[:, start:end]

    loss, token_loss = loss_fn(logits=logits, labels=local_labels)

    del logits
    del labels
    t = time.time()
    acc_fwd_time += (t - s)

    s = time.time()
    model.backward(loss)
    model.step()
    t = time.time()
    acc_bwd_time += (t - s)

    ########## dataset source monitor ###############
    if args.monitor_datasource_cnt:
      data_source_cnt = {}
      for data_source_name in data_source:
        data_source_cnt.setdefault(data_source_name, 0.0)
        data_source_cnt[data_source_name] += 1
      data_source_cnt = dist_reduce_dict(data_source_cnt)
      for k, v in data_source_cnt.items():
        total_data_source_cnt.setdefault(k, 0)
        total_data_source_cnt[k] += v
    #########################################

    avg_loss = torch.tensor(loss.item()).cuda()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    acc_avg_loss += avg_loss
    acc_step += 1

    iteration = model.global_steps
    if iteration % args.logging_per_step == 0 and dist.get_rank() == 0 and \
            model.is_gradient_accumulation_boundary():
      learning_rate = model.lr_scheduler.get_lr()[0]
      end_time = time.time()
      sec_per_step = (end_time - start_time) / acc_step
      tokens_per_sec_per_gpu = acc_num_tokens / dist.get_world_size() / (end_time - start_time)
      samples_per_sec_per_gpu = acc_num_samples / dist.get_world_size() / (end_time - start_time)
      avg_loss = acc_avg_loss / acc_step
      fwd_time = acc_fwd_time / acc_step
      bwd_time = acc_bwd_time / acc_step
      data_fetch_time = acc_data_fetch_time / acc_step
      start_time = end_time
      log_dict = {
        "losses/loss": avg_loss,
        "learning_rate": learning_rate,
        "losses/grad_norm": model.get_global_grad_norm(),
        "perf/sec_per_step": sec_per_step,
        "perf/tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
        "perf/samples_per_sec_per_gpu": samples_per_sec_per_gpu,
        "perf/total_num_tokens": total_num_tokens,
        "perf/total_num_samples": total_num_samples,
        "perf/fwd_time": fwd_time,
        "perf/bwd_time": bwd_time,
        "perf/data_fetch_time": data_fetch_time
      }

      for name, data in log_dict.items():
        if data is not None and tb_writer:
          tb_writer.add_scalar(
              name,
              data,
              global_step=iteration,
              new_style=True)

      if args.monitor_datasource_loss and tb_writer:
        sample_idx = sample_idx.squeeze()[:-1]   # shift idx
        unique_sample_idx = sample_idx.unique()
      
        data_source_loss = {}
        for s_idx in unique_sample_idx:
          mask = (sample_idx == s_idx)
          sum_loss = token_loss[mask].sum()
          token_num = mask.sum()
          key = data_source[int(s_idx.item())]
          data_source_loss.setdefault(key, [0.0, 0.0])
          data_source_loss[key][0] += sum_loss.item()
          data_source_loss[key][1] += token_num.item()
      
        def data_source_loss_reduce(gathered_dicts):
          sum_loss_dict = {}
          token_num_dict = {}
          loss_mean = {}
          for tmp_dict in gathered_dicts:
            for k, v in tmp_dict.items():
              sum_loss, token_num = v
              sum_loss_dict.setdefault(k, 0.0)
              token_num_dict.setdefault(k, 0.0)
              sum_loss_dict[k] += sum_loss
              token_num_dict[k] += token_num
          for k in sum_loss_dict:
            loss_mean[k] = sum_loss_dict[k] / token_num_dict[k]
          return loss_mean

        data_source_mean_loss = dist_reduce_dict(data_source_loss, data_source_loss_reduce)
        for k, v in loss_mean.items():
          tb_writer.add_scalar(
                f"data_source_loss/{key}",
                v,
                global_step=iteration,
                new_style=True)

      if args.monitor_datasource_cnt and tb_writer:
        source_ratio_dict = {}
        for key, cnt in total_data_source_cnt.items():
          source_ratio_dict[f"{key}"] = 1.0 * cnt / total_num_samples
          tb_writer.add_scalar(
              f"data_source_sample_ratio/{key}",
              1.0 * cnt / total_num_samples,
              global_step=iteration,
              new_style=True)

      print_rank_0(
        f"Step: {iteration}, Loss: {avg_loss}, "
        f"Learning Rate: {learning_rate}, "
        f"Grad Norm: {model.get_global_grad_norm()}, "
        f"Sec per Step: {sec_per_step}",
        f"tokens_per_sec_per_gpu: {tokens_per_sec_per_gpu}",
        f"samples_per_sec_per_gpu: {samples_per_sec_per_gpu}",
        f"total_num_tokens: {total_num_tokens}",
        f"total_num_samples: {total_num_samples}",
        f"fwd_time: {fwd_time}",
        f"bwd_time: {bwd_time}",
        f"data_fetch_time: {data_fetch_time}"
      )

      acc_step = 0
      acc_avg_loss = 0.0
      acc_num_samples = 0
      acc_num_tokens = 0
      acc_fwd_time = 0.0
      acc_bwd_time = 0.0

      if iteration % args.save_checkpoint_per_step == 0 and \
          iteration > 0 and model.is_gradient_accumulation_boundary():
        torch.cuda.empty_cache()
        model.save_checkpoint(
          save_dir=args.output_dir, client_state = {
            "total_num_tokens": total_num_tokens,
            "total_num_samples": total_num_samples
          }
        )

    s = time.time()

  print_rank_0("Save checkpoint..")
  model.save_checkpoint(save_dir=args.output_dir, client_state = {
    "total_num_tokens": total_num_tokens,
    "total_num_samples": total_num_samples}
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
