from rich import print
import argparse

import time
import os
import glob
import logging
import collections
from typing import List

import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from transformers import AutoTokenizer, Qwen2ForCausalLM
from safetensors import safe_open

from qwen_vl_utils import process_vision_info

from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.losses import CrossEntropyLoss
from recovlm.utils import print_rank_0, move_to_cuda
from recovlm.training import load_zero3_state_dict

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F


def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  ############ Dataset args ############
  parser.add_argument("--dataset", type=str, default=None,
                      help="The path of training data.")

  # Change to Vision Dataset
  # parser.add_argument("--chat_template", type=str,
  #                     default="chat_template_with_generation_tag",
  #                     help="The chat template to use")

  # parser.add_argument(
  #     "--system_prompt",
  #     type=str,
  #     default=None,
  #     help="Override default SYSTEM prompt, `None` means no SYSTEM used.")

  # parser.add_argument("--input_key", type=str, default="messages",
  #                     help="The column name for input data.")

  # parser.add_argument("--role_key", type=str, default="role",
  #                     help="The key name for role field.")

  # parser.add_argument("--content_key", type=str, default="content",
  #                     help="The key name for content field.")

  # parser.add_argument("--user_name", type=str, default="user",
  #                     help="The name of user role")

  # parser.add_argument("--assistant_name", type=str, default="assistant",
  #                     help="The name of assistant.")

  # parser.add_argument("--file_format", type=str, default="jsonl",
  #                     help="The format of file.")
  ################################################

  parser.add_argument("--max_length", type=int, default=1024,
                      help="Max tokens per sentence in corpus")

  parser.add_argument("--num_epochs", type=int, default=1,
                      help="Number of epochs to train")

  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Reserved for deepspeed framework")

  parser.add_argument("--output_dir", type=str, default=None,
                      help="The directory to write the trained model")

  parser.add_argument("--save_checkpoint_per_step", type=int, default=None,
                      help="The number of steps to save a checkpoint")

  parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                      help="Enable gradient checkpointing during training")

  parser.add_argument("--use_flash_attention_2", action="store_true",
                      help="Whether to use flash attention 2")

  parser.add_argument("--save_checkpoint_every_epoch", action="store_true",
                      help="Save checkpoint at the end of every epoch")

  parser.add_argument("--logging_per_step", type=int, default=100,
                      help="The number of steps to log training info")

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

  parser.add_argument("--seed", type=int, default=123,
                      help="Manual seed for RNG")

  return parser


def train():
  arg_parser = get_argument_parser()
  arg_parser = deepspeed.add_config_arguments(arg_parser)
  args = arg_parser.parse_args()

  assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
      "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
      "save_checkpoint_every_epoch should be set."

  deepspeed.init_distributed()

  tb_writer = None
  if dist.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))

  with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
    # TODO: add support for other models
    model_config = Qwen2ForCausalLM.config_class.from_pretrained(
        args.model_dir)
    model_config.attn_implementation = \
        "flash_attention_2" if args.use_flash_attention_2 else "eager"
    model_config.use_cache = False
    model = Qwen2ForCausalLM(model_config)

  load_zero3_state_dict(model, args.model_dir)
  model.train()
  if args.enable_gradient_checkpointing:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

  model_engine, _, _, _ = deepspeed.initialize(args=args,
                                               model=model)

  tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
  # Blended Datasets
  dataset = ChatCompletionDataset(
      source=args.dataset,
      tokenizer=tokenizer,
      input_key=args.input_key,
      system_prompt=args.system_prompt,
      chat_template=args.chat_template,
      role_key=args.role_key,
      content_key=args.content_key,
      user_name=args.user_name,
      assistant_name=args.assistant_name,
      file_format=args.file_format,
      max_length=args.max_length
  )
  loss_fn = CrossEntropyLoss(ignore_index=-100)
  sampler = DistributedSampler(dataset)
  start_time = time.time()
  for epoch in range(args.num_epochs):
    for batch in torch.utils.data.DataLoader(
            dataset,
            batch_size=model_engine._config.train_micro_batch_size_per_gpu,
            sampler=sampler,
            collate_fn=dataset.collate_fn):
      move_to_cuda(batch)
      input_ids = batch["input_ids"]
      loss_mask = batch["loss_mask"]
      attention_mask = batch.get("attention_mask", None)
      pixel_values = batch.get("pixel_values", None)
      pixel_values_videos = batch.get("pixel_values_videos", None)
      image_grid_thw = batch.get("image_grid_thw", None)
      video_grid_thw = batch.get("video_grid_thw", None)

      input_ids = input_ids * (input_ids > 0).to(torch.int64)
      labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask)

      empty_loss_ratio = 1 - (labels != loss_fn.ignore_index).sum(-1).mean()

      if empty_loss_ratio > 0.3:
        print_rank_0(
            f"WARN: {empty_loss_ratio * 100}% samples doesn't have loss.")

      output = model_engine(
          input_ids, labels=labels, attention_mask=attention_mask,
          pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
          image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw)

      logits = output.logits

      loss = loss_fn(logits=logits, labels=labels)

      model_engine.backward(loss)
      model_engine.step()
      model_engine.zero_grad()
      iteration = model_engine.global_steps
      if not args.save_checkpoint_every_epoch and \
          iteration % args.save_checkpoint_per_step == 0 and \
              iteration > 0 and model_engine.is_gradient_accumulation_boundary():
        model_engine.save_checkpoint(save_dir=args.output_dir)

      avg_loss = torch.tensor(loss.item()).cuda()
      dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
      avg_loss = avg_loss.item() / dist.get_world_size()
      if iteration % args.logging_per_step == 0 and dist.get_rank() == 0 and \
              model_engine.is_gradient_accumulation_boundary():
        learning_rate = model_engine.lr_scheduler.get_lr()[0]
        end_time = time.time()
        sec_per_step = (end_time - start_time) / args.logging_per_step
        start_time = end_time
        log_dict = {
            "loss": avg_loss,
            "learning_rate": learning_rate,
            "grad_norm": model_engine.get_global_grad_norm(),
            "sec_per_step": sec_per_step
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
            f"Grad Norm: {model_engine.get_global_grad_norm()}, "
            f"Sec per Step: {sec_per_step}")

    print_rank_0(f"Epoch {epoch} finished, save checkpoint...")
    if args.save_checkpoint_every_epoch:
      model_engine.save_checkpoint(save_dir=args.output_dir)

  if not args.save_checkpoint_every_epoch:
    model_engine.save_checkpoint(save_dir=args.output_dir)

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
