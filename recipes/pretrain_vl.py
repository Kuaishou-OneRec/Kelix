from rich import print
import argparse

import time
import wids
import os
import glob
import logging
import collections
from typing import List

os.environ["NCCL_IB_QPS_PER_CONNECTION"] = "2"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["NCCL_IB_GID_INDEX"] = "3"
os.environ["NCCL_IB_HCA"] = "mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
# os.environ["NCCL_TOPO_FILE"] = "/share/huzhiwen/baidu/topo_a800_hpc_bcc.xml"
os.environ["NCCL_ALGO"]= "^NVLS,NVLSTree"

import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
# from transformers import AutoTokenizer, AutoProcessor
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration
from safetensors import safe_open

from qwen_vl_utils import process_vision_info

from recovlm.data.dataloaders import get_indexed_dataloader
from recovlm.data.datasets import ImageTextPairDatasetWithPacking
from recovlm.data.collators import ImageTextPackingCollator
from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.losses import CrossEntropyLoss

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F


def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  ############ Dataset args ############
  parser.add_argument("--dataset", type=str, default=None,
                      help="The comma seperated path of indexed json file.")
  
  parser.add_argument("--packing_batch_size", type=int, default=1,
                      help="The batch size for sample packing.")

  ################################################

  parser.add_argument("--freeze_llm", action="store_true",
                      help="Freeze LLM parameters.")

  parser.add_argument("--freeze_visual", action="store_true",
                      help="Freeze visual encoder parameters.")

  parser.add_argument("--max_length", type=int, default=2048,
                      help="Max tokens per sentence in corpus")

  parser.add_argument("--num_epochs", type=int, default=1,
                      help="Number of epochs to train")

  parser.add_argument("--local_rank", type=int, default=-1,
                      help="Reserved for deepspeed framework")

  parser.add_argument("--output_dir", type=str, default=None,
                      help="The directory to write the trained model")

  parser.add_argument("--use_flash_attention_2", action="store_true",
                      help="Whether to use flash attention 2")
  
  parser.add_argument("--save_checkpoint_per_step", type=int, default=1000,
                      help="The number of steps to save a checkpoint")

  parser.add_argument("--save_checkpoint_every_epoch", action="store_true",
                      help="Save checkpoint at the end of every epoch")

  parser.add_argument("--logging_per_step", type=int, default=100,
                      help="The number of steps to log training info")

  parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                      help="Enable gradient checkpointing during training")

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


def print_rank_n(*msg, rank=0):
  if dist.get_rank() == rank:
    print(*msg)


def print_rank_0(*msg):
  print_rank_n(*msg, rank=0)


def move_to_cuda(batch):
  for key in list(batch.keys()):
    batch[key] = batch[key].cuda(torch.cuda.current_device())


def load_safetensors(path):
  tensors = {}
  with safe_open(path, framework="pt", device="cpu") as f:
    for k in f.keys():
      tensors[k] = f.get_tensor(k)
  return tensors


def load_zero3_state_dict(model, model_dir):

  missing_keys: List[str] = []
  unexpected_keys: List[str] = []
  error_msgs: List[str] = []

  if dist.get_rank() == 0:
    state_dict = collections.OrderedDict()
    patterns = glob.glob(os.path.join(model_dir, "model-*.safetensors"))
    for model_path in patterns:
      # state_dict.update(torch.load(model_path, map_location="cpu"))
      state_dict.update(load_safetensors(model_path))

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
      # mypy isn't aware that "_metadata" exists in state_dict
      state_dict._metadata = metadata  # type: ignore[attr-defined]
  else:
    state_dict = None

  dist.barrier()

  def load(module, local_state_dict, prefix=""):
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
      if dist.get_rank() == 0:
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        print_rank_0(f"Load: {prefix}")
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True,
            missing_keys, unexpected_keys, error_msgs
        )

    for name, child in module._modules.items():
      if child is not None:
        child_prefix = prefix + name + '.'
        if state_dict:
          child_state_dict = {
              k: v for k, v in local_state_dict.items() if
              k.startswith(child_prefix)}
        else:
          child_state_dict = None
        load(child, child_state_dict, child_prefix)

  load(model, state_dict, prefix="")


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

  # torch.cuda.memory._record_memory_history()

  # with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
  #   # TODO: add support for other models

  #   model_config = Qwen2VLForConditionalGeneration.config_class.from_pretrained(
  #       args.model_dir)
  #   model_config._attn_implementation = \
  #       "flash_attention_2" if args.use_flash_attention_2 else "eager"
  #   model_config.use_cache = False
  #   model = Qwen2VLForConditionalGeneration(model_config)

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

  # load_zero3_state_dict(model, args.model_dir)
  model.train()
  model_engine, _, _, _ = deepspeed.initialize(args=args,
                                               model=model)

  # TODO: 检查下预训练的tokenizer配置是否需要改变
  # TODO: fix hard code
  # processor.image_processor.min_pixels / 28 ** 2
  processor = Qwen2VLProcessor.from_pretrained(args.model_dir)
  
  dataset = ImageTextPairDatasetWithPacking(
      dataset=wids.ShardListDataset(args.dataset),
      processor = processor,
      max_length = args.max_length,
      min_visual_tokens = 1,
      max_visual_tokens = 512,
      spatial_merge_size = 2,
      image_token_id = 151655,
      video_token_id = 151656,
      vision_start_token_id = 151652,
      patch_size = 14,
      shrink_ratio = 0.9,
      max_retry = 5,
      multiple_of = 8
  )
  # sources = args.dataset.split(",")
  # dataloader = get_indexed_dataloader(
  #     sources=sources,
  #     processor=processor,
  #     batch_size=args.packing_batch_size,
  #     num_workers=8,
  #     shuffle=True,
  #     max_length=1024,
  #     rank=dist.get_rank(),
  #     collator=collator)
  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: x
  )
  loss_fn = CrossEntropyLoss(ignore_index=-100)
  start_time = time.time()
  show_cnt = 3
  total_num_tokens = 0
  for epoch in range(args.num_epochs):
    for batch in dataloader:
      if show_cnt > 0 and dist.get_rank() == 0:
        print_rank_0(batch)
        print_rank_0(
            f"Input Text:\n\n{processor.tokenizer.decode(batch['input_ids'][0])}")
        show_cnt -= 1
      move_to_cuda(batch)
      input_ids = batch["input_ids"]
      loss_mask = batch["loss_mask"]
      attention_mask = batch.get("attention_mask", None)
      pixel_values = batch.get("pixel_values", None)
      pixel_values_videos = batch.get("pixel_values_videos", None)
      image_grid_thw = batch.get("image_grid_thw", None)
      video_grid_thw = batch.get("video_grid_thw", None)
      cu_seqlens = batch.get("cu_seqlens", None)

      num_tokens = torch.tensor(input_ids.shape[-1]).cuda()
      dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
      total_num_tokens += num_tokens.item()

      input_ids = input_ids * (input_ids > 0).to(torch.int64)
      labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask)

      output = model_engine(
        input_ids, labels=labels, attention_mask=attention_mask,
        pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
        cu_seqlens=cu_seqlens
      )

      logits = output.logits

      loss = loss_fn(logits=logits, labels=labels)

      del logits
      del labels
      model_engine.backward(loss)
      model_engine.step()
      # model_engine.zero_grad()
      iteration = model_engine.global_steps

      avg_loss = torch.tensor(loss.item()).cuda()
      dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
      avg_loss = avg_loss.item() / dist.get_world_size()
      if iteration % args.logging_per_step == 0 and dist.get_rank() == 0 and \
              model_engine.is_gradient_accumulation_boundary():
        learning_rate = model_engine.lr_scheduler.get_lr()[0]
        end_time = time.time()
        sec_per_step = (end_time - start_time) / args.logging_per_step
        tokens_per_sec_per_gpu = num_tokens / dist.get_world_size() / (end_time - start_time)
        start_time = end_time
        log_dict = {
            "loss": avg_loss,
            "learning_rate": learning_rate,
            "grad_norm": model_engine.get_global_grad_norm(),
            "sec_per_step": sec_per_step,
            "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
            "total_num_tokens": total_num_tokens
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
            f"Sec per Step: {sec_per_step}",
            f"tokens_per_sec_per_gpu: {tokens_per_sec_per_gpu}",
            f"total_num_tokens: {total_num_tokens}")

      if iteration % args.save_checkpoint_per_step == 0 and \
          iteration > 0 and model_engine.is_gradient_accumulation_boundary():
        model_engine.save_checkpoint(save_dir=args.output_dir)

    print_rank_0(f"Epoch {epoch} finished.")
    if args.save_checkpoint_every_epoch:
      print_rank_0("Save checkpoint..")
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
