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

from recovlm.data.datasets import ChatCompletionDataset
from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict

from torch.utils.tensorboard import SummaryWriter


def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  parser.add_argument("--dataset", type=str, default=None,
                      help="The path of training data.")

  parser.add_argument("--chat_template", type=str,
                      default="chat_template_with_generation_tag",
                      help="The chat template to use")

  parser.add_argument(
      "--system_prompt",
      type=str,
      default=None,
      help="Override default SYSTEM prompt, `None` means no SYSTEM used.")

  parser.add_argument("--input_key", type=str, default="messages",
                      help="The column name for input data.")

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


def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        average_log_prob: bool = False) -> torch.FloatTensor:
  """Compute the log probabilities of the given labels under the given logits.

  Args:
      logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
      labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
      average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

  Returns:
      A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
  """
  assert average_log_prob == False
  assert logits.shape[:-1] == labels.shape

  labels = labels[:, 1:].clone()
  logits = logits[:, :-1, :]

  loss_masks = attention_mask.clone().bool()
  # mask prompts
  for mask, source_len in zip(loss_masks, prompt_id_lens):
    mask[:source_len] = False
  loss_masks = loss_masks[:, 1:]

  # dummy token; we'll ignore the losses on these tokens later
  labels[loss_masks == False] = 0
  per_token_logps = torch.gather(
      logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

  logprobs_sums = (per_token_logps * loss_masks).sum(-1)
  logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
  return logprobs_sums, logprobs_means


def concatenated_inputs(chosen_input_ids: torch.Tensor,
                        chosen_attention_mask: torch.Tensor,
                        rejected_input_ids: torch.Tensor,
                        rejected_attention_mask: torch.Tensor,
                        pad_token_id: int = 0):
  """Concatenate the chosen and rejected inputs into a single tensor.

  Args:
      batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
        which are tensors of shape (batch_size, sequence_length).

  Returns:
      A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
  """

  def pad_to_length(tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
      return tensor
    else:
      pad_size = list(tensor.shape)
      pad_size[dim] = length - tensor.size(dim)
      return torch.cat(
          [tensor, pad_value * torch.ones(
              *pad_size, dtype=tensor.dtype, device=tensor.device)],
          dim=dim
      )

  max_length = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
  inputs_ids = torch.cat(
      (
          pad_to_length(chosen_input_ids, max_length, pad_token_id),
          pad_to_length(rejected_input_ids, max_length, pad_token_id),
      ),
      dim=0,
  )
  max_length = max(
      chosen_attention_mask.shape[1],
      rejected_attention_mask.shape[1])
  attention_masks = torch.cat(
      (pad_to_length(chosen_attention_mask, max_length, 0),
       pad_to_length(rejected_attention_mask, max_length, 0)), dim=0)
  return inputs_ids, attention_masks


def concatenated_forward(model,
                         chosen_input_ids: torch.Tensor,
                         chosen_attention_mask: torch.Tensor,
                         rejected_input_ids: torch.Tensor,
                         rejected_attention_mask: torch.Tensor):
  """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

  We do this to avoid doing two forward passes, because it's faster for FSDP.
  """
  input_ids, attention_masks = concatenated_inputs(
      chosen_input_ids, chosen_attention_mask,
      rejected_input_ids, rejected_attention_mask,
  )
  output = model(input_ids, attention_mask=attention_masks, return_output=True)
  all_logits = output["logits"]
  all_logps_sum, all_logps_mean = get_batch_logps(
      all_logits, input_ids, attention_masks, average_log_prob=False
  )
  chosen_logps = all_logps_sum[:chosen_input_ids.shape[0]]
  rejected_logps = all_logps_sum[chosen_input_ids.shape[0]:]
  aux_loss = output.aux_loss if "aux_loss" in output else []
  return chosen_logps, rejected_logps, aux_loss, \
      -all_logps_mean[: chosen_input_ids.shape[0]].mean()


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
    model = Qwen2ForCausalLM(model_config)

  load_zero3_state_dict(model, args.model_dir)
  model.train()
  if args.enable_gradient_checkpointing:
    model.gradient_checkpointing_enable()

  with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
    # TODO: add support for other models
    ref_model = Qwen2ForCausalLM(model_config)

  load_zero3_state_dict(ref_model, args.model_dir)
  ref_model.eval()

  for p in ref_model.parameters():
    p.requires_grad = False

  # 使用 deepspeed 初始化模型
  print_rank_0("====dpo==== Initializing deepspeed...")
  with Timer("Initialize deepspeed model."):
    model.train()
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # 为参考模型创建一个虚拟优化器（参数不会被更新）
    ref_model.eval()  # 确保在评估模式
    dummy_optimizer = FusedAdam(
        [{'params': [p for p in ref_model.parameters()]}],
        lr=0.0,  # 学习率设为0
        betas=(0.9, 0.999)
    )
    
    # 同样初始化参考模型
    ref_model, _, _, _ = deepspeed.initialize(
        args=args,
        model=ref_model,
        optimizer=dummy_optimizer
    )

    # 确保参考模型参数被冻结
    for param in ref_model.parameters():
        param.requires_grad = False

  model_engine, _, _, _ = deepspeed.initialize(args=args,
                                               model=model)

  ref_model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                   model=ref_model)

  dataset = PreferenceDataset(
      source=args.dataset,
      tokenizer=args.model_dir,
      input_key=args.input_key,
      system_prompt=args.system_prompt,
      chat_template=args.chat_template,
      max_length=args.max_length
  )
  sampler = DistributedSampler(dataset)
  start_time = time.time()
  for epoch in range(args.num_epochs):
    for batch in torch.utils.data.DataLoader(
            dataset,
            batch_size=model_engine._config.train_micro_batch_size_per_gpu,
            sampler=sampler,
            collate_fn=dataset.collate_fn):
      move_to_cuda(batch)
      chosen_input_ids = batch["chosen_input_ids"]
      chosen_attention_mask = batch.get("chosen_attention_mask", None)

      rejected_input_ids = batch["rejected_input_ids"]
      rejected_attention_mask = batch.get("rejected_attention_mask", None)

      chosen_logps, rejected_logps, aux_loss, nll_loss = concatenated_forward(
          model,
          chosen_input_ids, chosen_attention_mask,
          rejected_input_ids, rejected_attention_mask
      )
      with torch.no_grad():
        reference_chosen_logps, reference_rejected_logps, _, _ = concatenated_forward(
            ref_model,
            chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask
        )

      loss = model_engine(
          input_ids, labels=labels, attention_mask=attention_mask).loss

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
