import argparse
import time
import wids
import os
import glob
import logging
import collections

import torch
import deepspeed
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam
from transformers import (
    SchedulerType,
    get_scheduler,
)

# from transformers import AutoTokenizer, AutoProcessor
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

from recovlm.data.dataloaders import get_indexed_dataloader
from recovlm.data.datasets import ImageTextPairDatasetWithPacking, VisionTextDatasetWithPacking
from recovlm.data.collators import ImageTextPackingCollator
from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.losses import CrossEntropyLoss
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0, \
  get_optimizer_grouped_parameters


def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Checkpoint args ############
  parser.add_argument("--model_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  parser.add_argument("--resume_from", type=str, default=None,
                      help="Specify the checkpoint tag to resume from.")
  
  parser.add_argument("--save_checkpoint_per_step", type=int, default=1000,
                      help="The number of steps to save a checkpoint")

  parser.add_argument("--save_checkpoint_every_epoch", action="store_true",
                      help="Save checkpoint at the end of every epoch")
  

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
  parser.add_argument("--dataset", type=str, default=None,
                      help="The comma seperated path of indexed json file.")
  
  parser.add_argument("--data_format", type=str, default="chatml",
                      help="The data format of training, one of `chatml` and `completion`")

  parser.add_argument("--min_visual_tokens", type=int, default=64,
                      help="The max visual tokens to use")

  parser.add_argument("--max_visual_tokens", type=int, default=512,
                      help="The max visual tokens to use")

  parser.add_argument("--max_length", type=int, default=2048,
                      help="Max tokens per sentence in corpus")
  
  ############ Learning Rate Args ############
  parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr",
                      help="The type of learning rate scheduler.")

  parser.add_argument("--num_warmup_steps", type=int, default=0,
                      help="The number of warmup steps to do.")
  
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

  parser.add_argument("--logging_per_step", type=int, default=100,
                      help="The number of steps to log training info")

  parser.add_argument("--seed", type=int, default=123,
                      help="Manual seed for RNG")

  return parser

def train():
  arg_parser = get_argument_parser()
  arg_parser = deepspeed.add_config_arguments(arg_parser)
  args = arg_parser.parse_args()
  torch.manual_seed(args.seed)

  assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
      "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
      "save_checkpoint_every_epoch should be set."

  deepspeed.init_distributed()

  set_random_seed(args.seed)
  torch.distributed.barrier()

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

  # if args.enable_gradient_checkpointing:
  #   print_rank_0("Enable gradient checkpointing")
  #   model.gradient_checkpointing_enable(
  #       gradient_checkpointing_kwargs={"use_reentrant": False})

  # Split weights in two groups, one with weight decay and the other not.
  optimizer_grouped_parameters = get_optimizer_grouped_parameters(
      model, args.weight_decay)

  # prepare optimizer
  optimizer = FusedAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        betas=(0.9, 0.95),
                        eps=1.0e-8)
  # TODO: pack args
  if args.lr_scheduler_type in ["cosine_with_min_lr"]:
    scheduler_specific_kwargs = {
      "min_lr": args.min_lr
    }
  lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
    scheduler_specific_kwargs=scheduler_specific_kwargs
  )

  model.train()
  model, optimizer, _, lr_scheduler = deepspeed.initialize(args=args,
                                                           model=model,
                                                           optimizer=optimizer,
                                                           lr_scheduler=lr_scheduler)

  total_num_tokens = 0
  total_num_samples = 0
  ckpt_id = args.resume_from
  latest = os.path.join(args.output_dir, "latest")
  if not ckpt_id and os.path.exists(latest):
    with open(latest, encoding="utf-8") as f:
      ckpt_id = f.read()
  if ckpt_id:
    print_rank_0(f"Resume from checkpoint: {os.path.join(args.output_dir, ckpt_id)}")
    _, client_state = model.load_checkpoint(args.output_dir, ckpt_id)
    total_num_tokens = client_state.get("total_num_tokens", 0)
    total_num_samples = client_state.get("total_num_samples", 0)

  dist.barrier()

  # TODO: remove hard code, dataloader配置化
  processor = Qwen2VLProcessor.from_pretrained(args.model_dir)

  # dataset = ImageTextPairDatasetWithPacking(
  #   sources = args.dataset,
  #   processor = processor,
  #   max_length = args.max_length,
  #   min_visual_tokens = args.min_visual_tokens,
  #   max_visual_tokens = args.max_visual_tokens,
  #   spatial_merge_size = 2,
  #   image_token_id = 151655,
  #   video_token_id = 151656,
  #   vision_start_token_id = 151652,
  #   patch_size = 14,
  #   shrink_ratio = 0.7,
  #   max_retry = 10,
  #   multiple_of = 8
  # )
  dataset = VisionTextDatasetWithPacking(
    sources = args.dataset,
    processor = processor,
    max_length = args.max_length,
    min_visual_tokens = args.min_visual_tokens,
    max_visual_tokens = args.max_visual_tokens,
    n_frames = 20,
    min_video_visual_tokens = args.min_visual_tokens * 5,
    max_video_visual_tokens = args.max_visual_tokens * 5,
    spatial_merge_size = 2,
    image_token_id = 151655,
    video_token_id = 151656,
    vision_start_token_id = 151652,
    patch_size = 14,
    shrink_ratio = 0.7,
    max_retry = 10,
    multiple_of = 8
  )
  
  ### packing, batching size=1; shuffle in dataset
  dataloader = DataLoader(
    dataset=dataset,
    shuffle=False,
    batch_size=1,
    num_workers=8,
    collate_fn=lambda x: x[0]
  )
  ##############

  loss_fn = CrossEntropyLoss(ignore_index=-100)
  start_time = time.time()
  show_cnt = 3
  for epoch in range(args.num_epochs):
    for batch in dataloader:
      if show_cnt > 0 and dist.get_rank() == 0:
        print_rank_0(batch)
        print_rank_0(
            f"Input Text:\n\n{processor.tokenizer.decode(batch['input_ids'][0])}\n"
            f"=" * 100 + "\n\n")
        show_cnt -= 1
      to_cuda(batch)
      input_ids = batch["input_ids"]
      loss_mask = batch["loss_mask"]
      attention_mask = batch.get("attention_mask", None)
      pixel_values = batch.get("pixel_values", None)
      pixel_values_videos = batch.get("pixel_values_videos", None)
      image_grid_thw = batch.get("image_grid_thw", None)
      video_grid_thw = batch.get("video_grid_thw", None)
      cu_seqlens = batch.get("cu_seqlens", None)

      num_tokens = torch.tensor(input_ids.shape[-1]).cuda()
      num_samples = torch.tensor(cu_seqlens.shape[-1] - 1).cuda()
      dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
      total_num_tokens += num_tokens.item()
      total_num_samples += num_samples.item()

      input_ids = input_ids * (input_ids > 0).to(torch.int64)
      labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask)

      output = model(
        input_ids, labels=labels, attention_mask=attention_mask,
        pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
        cu_seqlens=cu_seqlens
      )

      logits = output.logits
      loss = loss_fn(logits=logits, labels=labels)

      del logits
      del labels
      model.backward(loss)
      model.step()
      iteration = model.global_steps

      avg_loss = torch.tensor(loss.item()).cuda()
      dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
      avg_loss = avg_loss.item() / dist.get_world_size()
      if iteration % args.logging_per_step == 0 and dist.get_rank() == 0 and \
              model.is_gradient_accumulation_boundary():
        learning_rate = model.lr_scheduler.get_lr()[0]
        end_time = time.time()
        sec_per_step = (end_time - start_time) / args.logging_per_step
        tokens_per_sec_per_gpu = num_tokens / dist.get_world_size() / (end_time - start_time)
        samples_per_sec_per_gpu = num_samples / dist.get_world_size() / (end_time - start_time)
        start_time = end_time
        log_dict = {
          "loss": avg_loss,
          "learning_rate": learning_rate,
          "grad_norm": model.get_global_grad_norm(),
          "sec_per_step": sec_per_step,
          "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
          "samples_per_sec_per_gpu": samples_per_sec_per_gpu,
          "total_num_tokens": total_num_tokens,
          "total_num_samples": total_num_samples
        }
        for name, data in log_dict.items():
          if data is not None and tb_writer:
            tb_writer.add_scalar(
                name,
                data,
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
        )

      if iteration % args.save_checkpoint_per_step == 0 and \
          iteration > 0 and model.is_gradient_accumulation_boundary():
        model.save_checkpoint(
          save_dir=args.output_dir, client_state = {
            "total_num_tokens": total_num_tokens,
            "total_num_samples": total_num_samples
          }
        )

    print_rank_0(f"Epoch {epoch} finished.")
    if args.save_checkpoint_every_epoch:
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
