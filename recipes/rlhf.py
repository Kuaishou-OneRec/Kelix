import argparse
import time
import os
import glob
import logging
import collections
from typing import List, Tuple, Union
import json
import datetime

import torch
import deepspeed
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from deepspeed.ops.adam import FusedAdam

from recovlm.losses import CrossEntropyLoss
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from recovlm.data.dataloaders import get_dataloader
from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0, \
    get_optimizer_grouped_parameters, dist_reduce_dict, Timer, heart_beat
from recovlm.training.lr_schedulers import get_scheduler
from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_rank, get_sequence_parallel_world_size, \
  get_local_sequence_boundary, initialize_model_parallel, gather_by_group, \
  get_local_sequence, get_data_parallel_group, get_data_parallel_world_size, \
  get_data_parallel_rank

from recovlm.data.datasets import ChatCompletionDataset


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
                      help="Auto resume checkpoint from output dir if the latest ckpt exists.")
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
    parser.add_argument("--merge_checkpoint_output_file", type=str, default="pytorch_model.bin",
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
                      help="The min visual tokens to use")
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
                      help="The peak vit learning rate for optimizer. " \
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
    parser.add_argument("--freeze_llm", action="store_true",
                      help="Freeze LLM parameters.")
    parser.add_argument("--freeze_visual", action="store_true",
                      help="Freeze visual encoder parameters.")
    parser.add_argument("--freeze_visual_without_adapter", action="store_true",
                      help="Only freeze visual encoder parameters, train adapter parameters.")
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
    parser.add_argument("--monitor_datasource_loss", action="store_true",
                      help="Whether to monitor loss of each datasource")
    parser.add_argument("--monitor_datasource_cnt", action="store_true",
                      help="Whether to monitor cnt of each datasource")
    parser.add_argument("--loss_style", type=str, choices=["token", "sample"], default="sample",
                      help="Token wise loss or sample wise loss")

    ############ Processor Args ############
    parser.add_argument("--pad_id", type=int, default=151643,
                      help="Processor pad token id")
    parser.add_argument("--eos_id", type=int, default=151645,
                      help="Processor eos token id")

    ############ RLHF specific args ############
    # parser.add_argument("--rlhf_beta", type=float, default=0.1,
    #                   help="The beta parameter for RLHF loss")
    # parser.add_argument("--label_smoothing", type=float, default=0.0,
    #                   help="Label smoothing parameter for RLHF loss")
    # parser.add_argument("--rlhf_reference_free", action="store_true",
    #                   help="Whether to use reference-free RLHF training")

    ############ System Vars ############
    parser.add_argument("--kml_id", type=str, default=None,
                      help="KML_ID")
    parser.add_argument("--kml_task_id", type=str, default=None,
                      help="KML_TASK_ID")
    parser.add_argument("--heartbeat_monitor", action="store_true",
                      help="Whether to upload heartbeat to remote")
    parser.add_argument("--comment", type=str, default=None,
                      help="Comment of this experiment.")
    parser.add_argument("--commit_id", type=str, default=None,
                      help="Git commit id for experiment.")
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

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

def get_batch_rewards(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    mask: torch.BoolTensor,
    ignore_index: int = -100,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """计算每个序列的对数概率
    Args:
        logits: shape (batch_size, seq_len/parallel_size, 1)
        labels: shape (batch_size, seq_len) 
        mask: shape (batch_size, seq_len)
        ignore_index: 忽略的标签值
        average_log_prob: 是否返回平均对数概率
    Returns:
        log_probs: shape (batch_size,)
    """
    # 添加输入验证
    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain inf or nan values")
    
    # 确保输入在同一设备上
    assert logits.device == labels.device == mask.device
    
    # 处理padding和标签
    labels = labels * (labels > 0).to(torch.int64)  # 将负值替换为0
    labels = labels * mask + ignore_index * (1 - mask)  # 用ignore_index填充非mask位置
    
    # shift labels
    pad = torch.full((labels.shape[0], 1), ignore_index, 
                    dtype=labels.dtype, device=labels.device)
    labels = torch.cat([labels[:, 1:], pad], dim=-1)  # shift labels
    
    # 获取本地序列部分
    local_labels = get_local_sequence(labels, seq_idx=1)
    valid_mask = get_local_sequence(mask, seq_idx=1)

    return logits.squeeze(-1), local_labels, valid_mask

    # log_probs = F.log_softmax(logits.float(), dim=-1)
    
    # # 创建新的张量进行标签处理
    # valid_labels = local_labels.clone()
    # valid_labels[valid_labels == ignore_index] = 0
    
    # # 使用新的张量进行gather操作
    # token_log_probs = log_probs.gather(
    #     dim=-1, 
    #     index=valid_labels.unsqueeze(-1)
    # ).squeeze(-1)
    
    # # 将ignore_index位置的概率置为0
    # token_log_probs = token_log_probs * (local_labels != ignore_index).float()
        
    # return token_log_probs


# tensor_list = [None for _ in range(world_size)]
#         dist.all_gather_object(
#             object_list=tensor_list, obj=tensor,
#             group=group
#         )
#         tensor_list = [x.to(torch.cuda.current_device()) for x in tensor_list]
#         return torch.concat(tensor_list, dim=dim)
# group = get_sequence_parallel_group()
#     world_size = dist.get_world_size(group)
class DisCoGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        if not dist.is_initialized():
            raise "torch.distributed is not initialized"

        group = get_sequence_parallel_group()
        world_size = dist.get_world_size(group)
        # world_size = torch.distributed.get_world_size()
        ctx.bs = tensor.shape[0]
        ctx.rank = dist.get_rank(group=group)

        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(gathered_tensors, tensor.contiguous(), group=group)

        gathered_tensors = torch.cat(gathered_tensors, dim=1)
        gathered_tensors.requires_grad_(True)

        return gathered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        lengths = grad_output.shape[1]
        world_size = get_sequence_parallel_world_size()
        local_lengths = lengths // world_size
        dist.all_reduce(grad_output, op=torch.distributed.ReduceOp.AVG)
        return grad_output[:, ctx.rank * local_lengths: local_lengths * (ctx.rank + 1)]



def disco_gather(tensor):
    return DisCoGather.apply(tensor)


def compute_rlhf_loss(
    chosen_rewards: torch.FloatTensor,
    rejected_rewards: torch.FloatTensor,
    chosen_token_ids: torch.LongTensor,
    rejected_token_ids: torch.LongTensor,
    chosen_sample_idx: torch.IntTensor,
    rejected_sample_idx: torch.IntTensor,
    eos_token_id: int = 151645,
    pad_id: int = 151643,
    newline_id: int = 198,
    loss_style="sample",
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    # # 计算rewards，确保保持梯度
    # chosen_rewards = reward_chosen_logps
    # rejected_rewards = reward_rejected_logps

    assert loss_style in ["sample", "token"]

    group = get_sequence_parallel_group()
    world_size = dist.get_world_size(group)
    
    def gather_concat_tensor(tensor, dim=1):
        tensor_list = [None for _ in range(world_size)]
        dist.all_gather_object(
            object_list=tensor_list, obj=tensor,
            group=group
        )
        tensor_list = [x.to(torch.cuda.current_device()) for x in tensor_list]
        return torch.concat(tensor_list, dim=dim)

    def get_token_rewards(batch_rewards, batch_token_ids, batch_sample_idx, only_eos=False):
        rewards_list = list()
        if batch_sample_idx.dim() == 1:
            batch_sample_idx = batch_cu_sample_idx[None, :]

        for i in range(batch_rewards.shape[0]):
            token_ids = batch_token_ids[i]
            rewards = batch_rewards[i]
            sample_idx = batch_sample_idx[i]

            unique_sample_idx = torch.unique(sample_idx)
            unique_sample_idx, _ = unique_sample_idx.sort()
            if unique_sample_idx[0].item() == -1:
                unique_sample_idx = unique_sample_idx[1:]

            for idx in unique_sample_idx:
                sample_indices = (sample_idx == idx.item()).nonzero().flatten()
                if only_eos:
                    sample_indices = sample_indices[-1:]
                    assert sample_indices[0].item() == pad_id
                rewards_list.append(rewards[sample_indices])
            # assert cu_seqlens[0] == 0, cu_seqlens

            # eos_indices = list()
            # for j in range(1, len(cu_seqlens)):
            #     index = cu_seqlens[j]
            #     print("[ZDJ]", token_ids[index - 2: index + 2])
            #     assert token_ids[index - 1] == pad_id, token_ids[index - 1]
            #     # assert token_ids[index - 2] == eos_token_id, token_ids[index - 2]
            #     eos_indices.append(index - 1)

            # # token_is_eos = (token_ids == eos_token_id)
            # # prev_is_eos = (torch.roll(token_ids, 1) == eos_token_id)
            # # prev_is_eos[0] = False
            # # eos_indices = (token_is_eos & (~prev_is_eos)).nonzero().flatten()

            # eos_rewards = rewards[eos_indices]
            # batch_eos_rewards.append(eos_rewards)
        return rewards_list

    def get_sample_all_token_rewards(batch_rewards, batch_token_ids):
        all_token_rewards = list()
        for i in range(batch_rewards.shape[0]):
            token_ids = batch_token_ids[i]
            rewards = batch_rewards[i]

            token_is_eos = (token_ids == eos_token_id)
            prev_is_eos = (torch.roll(token_ids, 1) == eos_token_id)
            prev_is_eos[0] = False
            eos_indices = (token_is_eos & (~prev_is_eos)).nonzero().flatten()

            prev_is_eos[0] = True
            bos_indices = (prev_is_eos & (~token_is_eos)).nonzero().flatten()
            assert eos_indices.shape[0] == bos_indices.shape[0], "{} {}".format(eos_indices, bos_indices)

            for start, end in zip(bos_indices, eos_indices):
                all_token_rewards.append(batch_rewards[start: end + 1])
        return all_token_rewards

    def pad_fixed_length_1d(tensor, length, value, left_padding=False):
        if tensor.shape[0] >= length:
            return tensor
        padding_length = length - tensor.shape[0]
        padding = tensor.new_full((padding_length, ), value, dtype=tensor.dtype)
        if left_padding:
            new_tensor = torch.concat([padding, tensor], dim=0)
        else:
            new_tensor = torch.concat([tensor, padding], dim=0)
        return new_tensor

    gathered_chosen_rewards = gather_concat_tensor(chosen_rewards)
    gathered_rejected_rewards = gather_concat_tensor(rejected_rewards)

    chosen_size = gathered_chosen_rewards.shape[1]
    rejected_size = gathered_rejected_rewards.shape[1]
    batch_size = gathered_chosen_rewards.shape[0]

    if loss_style == "sample":
        batch_chosen_eos_rewards = get_token_rewards(gathered_chosen_rewards, chosen_token_ids, chosen_sample_idx, only_eos=True)
        batch_rejected_eos_rewards = get_token_rewards(gathered_rejected_rewards, rejected_token_ids, rejected_sample_idx, only_eos=True)

        losses = 0.
        chosen_rewards_sum = 0.
        rejected_rewards_sum = 0.

        chosen_num_samples = sum([x.numel() for x in batch_chosen_eos_rewards])
        rejected_num_samples = sum([x.numel() for x in batch_rejected_eos_rewards])
        assert chosen_num_samples == rejected_num_samples and chosen_num_samples > 0, "{} {}".format(chosen_token_ids, rejected_token_ids)

        for chosen_eos_rewards, rejected_eos_rewards in zip(batch_chosen_eos_rewards, batch_rejected_eos_rewards):
            assert chosen_eos_rewards.shape == rejected_eos_rewards.shape

            chosen_rewards_sum += chosen_eos_rewards.sum()
            rejected_rewards_sum += rejected_eos_rewards.sum()
    
            logits = chosen_eos_rewards - rejected_eos_rewards

            # 检查数值稳定性
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print_rank_0("====rlhf==== ERROR: logits contains nan or inf values!")
                print_rank_0(f"====rlhf==== chosen_eos_rewards: {chosen_eos_rewards}")
                print_rank_0(f"====rlhf==== rejected_eos_rewards: {rejected_eos_rewards}")
                raise ValueError("logits contains nan or inf values")
            
            loss = -F.logsigmoid(logits)
            losses += loss.sum()
        losses = losses / world_size / chosen_num_samples
        return losses, chosen_rewards_sum / chosen_num_samples, rejected_rewards_sum / chosen_num_samples

    losses = 0.
    chosen_rewards_sum = 0.
    rejected_rewards_sum = 0.

    chosen_token_rewards_list = get_sample_all_token_rewards(gathered_chosen_rewards, chosen_token_ids)
    rejected_token_rewards_list = get_sample_all_token_rewards(gathered_rejected_rewards, rejected_token_ids)

    assert len(chosen_token_rewards_list) == len(rejected_token_rewards_list)

    for chosen_token_rewards, rejected_token_rewards in zip(chosen_token_rewards_list, rejected_token_rewards_list):
        max_length = max(chosen_token_rewards.shape[0], rejected_token_rewards.shape[0])
        
        chosen_token_rewards = pad_fixed_length_1d(chosen_token_rewards, max_length, 0.)
        rejected_token_rewards = pad_fixed_length_1d(rejected_token_rewards, max_length, 0.)

        padding_chosen_token_ids = pad_fixed_length_1d(chosen_token_ids, max_length, pad_id)
        padding_rejected_token_ids = pad_fixed_length_1d(rejected_token_ids, max_length, pad_id)

        divergence_token_indices = (padding_chosen_token_ids != padding_rejected_token_ids).nonzero().flatten()[0]
        partial_chosen_rewards = chosen_token_rewards[divergence_token_indices:]
        partial_rejected_rewards = rejected_token_rewards[divergence_token_indices:]

        logits = partial_chosen_rewards - partial_rejected_rewards

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print_rank_0("====rlhf==== ERROR: logits contains nan or inf values!")
            print_rank_0(f"====rlhf==== partial_chosen_rewards: {partial_chosen_rewards}")
            print_rank_0(f"====rlhf==== partial_rejected_rewards: {partial_rejected_rewards}")
            raise ValueError("logits contains nan or inf values")
        
        loss = -F.logsigmoid(logits)
        losses += loss.mean()
        chosen_rewards_sum += partial_chosen_rewards.mean()
        rejected_rewards_sum += partial_rejected_rewards.mean()

    chosen_num_samples = len(chosen_token_rewards)
    losses = losses / chosen_num_samples / world_size
    
    return losses, chosen_rewards_sum / chosen_num_samples, rejected_rewards_sum / chosen_num_samples


def get_resume_info(args):
    """获取恢复训练的信息
    Args:
        args: 参数
    Returns:
        resume_from: 恢复checkpoint的目录
        ckpt_id: checkpoint的标识
        rewrite_flag: 是否重写了resume相关参数
    """
    if not args.auto_resume_local_latest:
        # 检查手动指定的恢复路径
        if args.resume_from and not os.path.exists(args.resume_from):
            raise ValueError(f"Resume checkpoint directory {args.resume_from} does not exist")
        
        if args.resume_from and args.resume_from_tag:
            ckpt_path = os.path.join(args.resume_from, args.resume_from_tag)
            if not os.path.exists(ckpt_path):
                raise ValueError(f"Resume checkpoint path {ckpt_path} does not exist")
            
        return args.resume_from, args.resume_from_tag, False
    else:
        # 检查本地最新checkpoint
        latest_file = os.path.join(args.output_dir, "latest")
        if os.path.exists(latest_file):
            with open(latest_file, encoding="utf-8") as f:
                ckpt_id = f.read().strip()
            
            ckpt_path = os.path.join(args.output_dir, ckpt_id)
            if not os.path.exists(ckpt_path):
                raise ValueError(f"Latest checkpoint path {ckpt_path} does not exist")
                
            print_rank_0(f"====rlhf==== Check output_ckpt exists, auto resume from output_folder. " \
                        f"checkpoint: resume_from={args.output_dir}, resume_tag={ckpt_id}")
            return args.output_dir, ckpt_id, True
        else:
            return args.resume_from, args.resume_from_tag, False


def pad_to_multiple_of_8(tensor, pad_value=0):
    """将tensor的最后一个维度填充到8的倍数
    Args:
        tensor: 输入tensor
        pad_value: 填充值
    Returns:
        padded_tensor: 填充后的tensor
        original_size: 原始大小
    """
    if tensor is None:
        return None, None
        
    size = tensor.size()
    last_dim = size[-1]
    pad_size = (8 - (last_dim % 8)) % 8
    
    if pad_size == 0:
        return tensor, last_dim
        
    # 创建填充规格 (左边0, 右边pad_size)
    pad_spec = [0, 0] * (len(size) - 1) + [0, pad_size]
    padded_tensor = F.pad(tensor, pad_spec, value=pad_value)
    
    return padded_tensor, last_dim


def concatenate_inputs(chosen_inputs, rejected_inputs):
    """合并chosen和rejected输入，并记录原始序列长度"""
    combined_inputs = {}
    sequence_lengths = {
        "chosen_length": chosen_inputs["input_ids"].size(1),
        "rejected_length": rejected_inputs["input_ids"].size(1)
    }

    # 处理序列类输入（在序列长度维度上拼接）
    for key in ["input_ids", "attention_mask", "loss_mask"]:
        if key in chosen_inputs:
            combined_inputs[key] = torch.cat([
                chosen_inputs[key],
                rejected_inputs[key]
            ], dim=1)  # 在序列长度维度上拼接

    # 特殊处理 cu_seqlens
    if "cu_seqlens" in chosen_inputs and chosen_inputs["cu_seqlens"] is not None:
        # 获取chosen部分的最大值
        chosen_max = chosen_inputs["cu_seqlens"][-1]
        
        # 将rejected的值加上chosen的最大值
        shifted_rejected = rejected_inputs["cu_seqlens"] + chosen_max
        
        # 连接两个张量
        combined_inputs["cu_seqlens"] = torch.cat([
            chosen_inputs["cu_seqlens"],
            shifted_rejected
        ], dim=-1)

    # 处理其他输入（在批次维度上拼接）
    for key in chosen_inputs:
        # 添加调试信息
        if isinstance(chosen_inputs[key], torch.Tensor):
            print_rank_0(f"Input shapes - {key}: {chosen_inputs[key].shape}, {rejected_inputs[key].shape}")
        if key in ["input_ids", "attention_mask", "loss_mask", "cu_seqlens"]:
            continue

        if isinstance(chosen_inputs[key], torch.Tensor):
            if key in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
                if chosen_inputs[key] is not None and rejected_inputs[key] is not None:
                    combined_inputs[key] = torch.cat([
                        chosen_inputs[key],
                        rejected_inputs[key]
                    ], dim=0)  # 图像和视频在批次维度上拼接
            else:
                combined_inputs[key] = torch.cat([
                    chosen_inputs[key],
                    rejected_inputs[key]
                ], dim=-1)
        elif isinstance(chosen_inputs[key], list):
            if len(chosen_inputs[key]) > 0:
                combined_inputs[key] = chosen_inputs[key] + rejected_inputs[key]
            else:
                combined_inputs[key] = []
        else:
            combined_inputs[key] = chosen_inputs[key]

    return combined_inputs, sequence_lengths


def split_outputs(outputs, sequence_lengths):
    """将模型输出分割回chosen和rejected部分，只处理必要的字段
    Args:
        outputs: 模型输出
        sequence_lengths: 原始序列长度信息 {"chosen_length": int, "rejected_length": int}
    Returns:
        chosen_outputs: chosen部分的输出
        rejected_outputs: rejected部分的输出
    """
    chosen_outputs = type('Outputs', (), {})()
    rejected_outputs = type('Outputs', (), {})()
    
    # 只处理reward_logits字段
    if hasattr(outputs, 'reward_logits'):
        reward_logits = outputs.reward_logits.squeeze(-1)  # shape: [batch_size, local_seq_len]
        
        # 获取local sequence的chosen部分长度
        chosen_length = sequence_lengths["chosen_length"] // get_sequence_parallel_world_size()
        
        # 在序列维度上分割reward_logits
        chosen_outputs.reward_logits = reward_logits[:, :chosen_length]
        rejected_outputs.reward_logits = reward_logits[:, chosen_length:]
    
    if hasattr(outputs, 'logits'):
        logits = outputs.logits  # shape: [batch_size, local_seq_len, vocab_size]
        
        # 获取local sequence的chosen部分长度
        chosen_length = sequence_lengths["chosen_length"] // get_sequence_parallel_world_size()
        
        # 在序列维度上分割reward_logits
        chosen_outputs.logits = logits[:, :chosen_length]
        rejected_outputs.logits = logits[:, chosen_length:]
    
    return chosen_outputs, rejected_outputs


def train():
    # 1. 解析参数
    arg_parser = get_argument_parser()
    arg_parser = deepspeed.add_config_arguments(arg_parser)
    args = arg_parser.parse_args()

    # 2. 初始化分布式
    deepspeed.init_distributed()

    # 3. 检查参数
    assert args.learning_rate > 0.0
    if args.vision_learning_rate < 0.0:
        args.vision_learning_rate = args.learning_rate
        print_rank_0("====rlhf==== Setting vision_learning_rate to learning_rate")

    assert all([args.commit_id, args.seed, args.comment]), \
        "Git commit, seed, and comment is required for reproducibility"

    assert all([args.kml_id, args.kml_task_id]), \
        "Kml task information, for task alive monitor."

    assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
        "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
        "save_checkpoint_every_epoch should be set."

    # 4. 设置环境变量
    print_rank_0("====rlhf==== Setting environment variables...")
    os.environ["KML_ID"] = args.kml_id
    os.environ["KML_TASK_ID"] = args.kml_task_id

    # 5. 初始化模型并行
    initialize_model_parallel(args.sequence_parallel_size)
    print_rank_0(f"====rlhf==== Sequence parallel size: {get_sequence_parallel_world_size()}")

    # 6. 设置随机种子
    set_random_seed(args.seed)
    dist.barrier()

    # 7. 打印参数
    if dist.get_rank() == 0:
        args_dict = vars(args)
        args_str = json.dumps(args_dict, indent=4, ensure_ascii=False)
        print_rank_0(f"====rlhf==== Training Arguments:\n{args_str}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir,
              f"args-{args.commit_id}-{timestamp}.json"), 'w',
            encoding="utf-8") as f:
            f.write(args_str + "\n")

    # 8. 初始化tensorboard
    print_rank_0("====rlhf==== Initializing tensorboard writer...")
    tb_writer = None
    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
        tb_writer.add_text("comment", args.comment, 0)
        tb_writer.add_text("comment_id", args.commit_id, 0)
        tb_writer.add_text("kml_id", args.kml_id, 0)
        tb_writer.add_text("kml_task_id", args.kml_task_id, 0)

    # 9. 初始化模型
    print_rank_0("====rlhf==== Initializing models...")
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config,
                           enabled=False):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_dir, _attn_implementation="flash_attention_2",
            use_cache=False)
        # ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     args.model_dir, _attn_implementation="flash_attention_2",
        #     use_cache=False)
        
        # 冻结参考模型的所有参数
        # print_rank_0("====rlhf==== Freezing reference model parameters...")
        # for param in ref_model.parameters():
        #     param.requires_grad = False
        # ref_model.eval()  # 设置为评估模式

    # 10. 设置reward模型的参数冻结（如果需要）
    if args.freeze_llm:
        print_rank_0("====rlhf==== Freezing LLM parameters...")
        for name, param in model.named_parameters():
            if not name.startswith("visual"):
                print_rank_0(f"====rlhf==== Disable LLM grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)

    if args.freeze_visual:
        print_rank_0("====rlhf==== Freezing visual encoder parameters...")
        for name, param in model.named_parameters():
            if name.startswith("visual"):
                print_rank_0(f"====rlhf==== Disable visual encoder grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)

    if args.freeze_visual_without_adapter:
        print_rank_0("====rlhf==== Freezing visual encoder parameters (except adapter)...")
        for name, param in model.named_parameters():
            if name.startswith("visual") and not name.startswith("visual.merger."):
                print_rank_0(f"====rlhf==== Disable visual encoder grad: {name}")
                param.requires_grad = False
        print_rank_0("=" * 50)

    # 打印训练参数日志
    print_rank_0("====rlhf==== Parameters requiring gradients:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank_0(f"====rlhf==== params not freeze: {name}")
    print_rank_0("=" * 50)

    if args.enable_gradient_checkpointing:
      print_rank_0("Enable gradient checkpointing")
      model.gradient_checkpointing_enable(
          gradient_checkpointing_kwargs={"use_reentrant": False})
      
    # 准备优化器
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,
        learning_rate=args.learning_rate,
        vision_learning_rate=args.vision_learning_rate,
        weight_decay=args.weight_decay,
        no_decay_name_list=["bias", "norm1", "norm2", "visual.merger.ln_q", "input_layernorm", "post_attention_layernorm", "model.norm"],
        vision_learning_rate_layer_dacay=args.vision_lr_layer_decay
    )

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

    # 使用 deepspeed 初始化模型
    print_rank_0("====rlhf==== Initializing deepspeed...")
    with Timer("Initialize deepspeed model."):
        # 首先初始化主模型
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )

        # # 首先确保参考模型所有参数都被冻结
        # ref_model.eval()
        # for param in ref_model.parameters():
        #     param.requires_grad = False

        # # 使用 DeepSpeed 引擎直接包装模型
        # ref_model = deepspeed.init_inference(
        #     model=ref_model,
        #     dtype=torch.bfloat16,
        #     replace_with_kernel_inject=False
        # )

        # # 确保模型在正确的设备上
        # if torch.cuda.is_available():
        #     ref_model = ref_model.cuda()

    # 11. 初始化统计变量
    total_num_tokens = 0
    total_num_samples = 0
    total_num_valid_tokens = 0
    total_data_source_samples = collections.defaultdict(int)
    total_data_source_tokens = collections.defaultdict(int)

    # 12. 获取resume信息
    resume_from, ckpt_id, rewrite_resume_flag = get_resume_info(args)

    if rewrite_resume_flag:
        args.resume_dataloader = True
        args.load_weights_only = False
        print_rank_0(f"====rlhf==== WARN: --resume_dataloader is rewritten to True\n" \
                     f"====rlhf==== WARN: --load_weights_only is rewritten to False\n")

    # 13. 如果需要从checkpoint恢复
    if ckpt_id:
        ckpt_path = os.path.join(resume_from, ckpt_id)
        print_rank_0(
            f"====rlhf==== Resume from checkpoint: {ckpt_path}, "
            f"load_weights_only={args.load_weights_only}")
        
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint path {ckpt_path} does not exist")
            
        _, client_state = model.load_checkpoint(
            resume_from, ckpt_id, load_module_only=args.load_weights_only)

        if args.resume_dataloader:
            dataloader_resume_path = os.path.join(resume_from, "dataloader_ckpt", 
                                                f"rank{dist.get_rank()}_{ckpt_id}.pth")
            if not os.path.exists(dataloader_resume_path):
                print_rank_0(f"====rlhf==== Warning: Dataloader checkpoint {dataloader_resume_path} does not exist")
                print_rank_0("====rlhf==== Will start training without resuming dataloader state")
                dataloader_state_dict = None
            else:
                try:
                    dataloader_state_dict = torch.load(dataloader_resume_path)["dataloader_state_dict"]
                    print_rank_0(f"====rlhf==== Successfully loaded dataloader state from {dataloader_resume_path}")
                except Exception as e:
                    print_rank_0(f"====rlhf==== Error loading dataloader checkpoint: {str(e)}")
                    print_rank_0("====rlhf==== Will start training without resuming dataloader state")
                    dataloader_state_dict = None

        if not args.load_weights_only:
            total_num_tokens = client_state.get("total_num_tokens", 0)
            total_num_samples = client_state.get("total_num_samples", 0)
            total_num_valid_tokens = client_state.get("total_num_valid_tokens", 0)

            if dist.get_rank() == 0:
                total_data_source_samples.update(client_state.get("total_data_source_samples", {}))
                total_data_source_tokens.update(client_state.get("total_data_source_tokens", {}))

    dist.barrier()

    ##############
    with open(args.dataset_config, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    
    # 获取数据集名称和配置
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

    # loss_fn_chosen = CrossEntropyLoss(
    #     ignore_index=-100, return_token_loss=True, shift_labels=False)
    # loss_fn_rejected = CrossEntropyLoss(
    #     ignore_index=-100, return_token_loss=True, shift_labels=False)
    # loss_fn_ref = CrossEntropyLoss(
    #     ignore_index=-100, return_token_loss=True, shift_labels=False)
    
    # 加载处理器
    processor = Qwen2VLProcessor.from_pretrained(args.model_dir)
    show_cnt = 1
    
    # 训练统计初始化
    total_num_tokens = 0
    total_num_samples = 0
    total_num_valid_tokens = 0
    acc_step = 0
    acc_avg_loss = 0.0
    acc_num_tokens = 0
    acc_num_samples = 0
    acc_valid_num_tokens = 0
    iteration = 0  # 添加iteration初始化
    acc_chosen_loss = 0.0
    acc_rejected_loss = 0.0
    # 数据源监控
    batch_data_source_loss = collections.defaultdict(float)
    batch_data_source_tokens = collections.defaultdict(int)
    valid_data_source_tokens = collections.defaultdict(int)
    total_data_source_samples = collections.defaultdict(int)
    total_data_source_tokens = collections.defaultdict(int)

    # 在初始化模型之后，确保两个模型都在正确的设备上并设置正确的数据类型
    print_rank_0("====rlhf==== Moving models to device and setting dtype...")
    device = model.device
    dtype = torch.bfloat16  # 或者使用 torch.bfloat16，取决于你的需求
    
    # # 修改这部分代码来检查ref_model的设备和数据类型
    # print_rank_0(f"====rlhf==== Model device: {model.device}, dtype: {model.dtype}")
    # # 对于ref_model，我们检查其模块参数而不是直接访问device和dtype
    # ref_model_param = next(ref_model.module.parameters())  # 使用.module访问底层PyTorch模型
    # print_rank_0(f"====rlhf==== Ref model device: {ref_model_param.device}, dtype: {ref_model_param.dtype}")
    
    # 确保模型参数使用正确的数据类型
    model = model.to(dtype)
    # ref_model = ref_model.to(dtype)
    
    # 训练循环
    model.train()
    # ref_model.eval()

    # 在训练开始前添加
    # print_rank_0(f"====rlhf==== RLHF training configuration:")
    # print_rank_0(f"Beta: {args.rlhf_beta}")
    # print_rank_0(f"Label smoothing: {args.label_smoothing}")
    # print_rank_0(f"Reference free: {args.rlhf_reference_free}")

    # 在模型forward之前添加类型检查和转换
    def ensure_input_types(inputs):
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"]* (inputs["input_ids"] > 0).to(dtype=torch.int64)
        # if "attention_mask" in inputs:
        #     inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.bool)
        # if "pixel_values" in inputs:
        #     inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)  # dtype是之前定义的float16或bfloat16
        # if "pixel_values_videos" in inputs:
        #     inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype=dtype)
        # if "image_grid_thw" in inputs:
        #     inputs["image_grid_thw"] = inputs["image_grid_thw"].to(dtype=torch.long)
        # if "video_grid_thw" in inputs:
        #     inputs["video_grid_thw"] = inputs["video_grid_thw"].to(dtype=torch.long)
        return inputs

    # 使用 gather_by_group 来处理数据
    for iteration, batch in enumerate(gather_by_group(dataloader, get_sequence_parallel_group())):
        chosen_inputs, rejected_inputs = batch
        to_cuda(chosen_inputs)
        to_cuda(rejected_inputs)
        
        # 确保输入类型正确
        chosen_inputs = ensure_input_types(chosen_inputs)
        rejected_inputs = ensure_input_types(rejected_inputs)

        # 合并输入以提高效率
        combined_inputs, sequence_lengths = concatenate_inputs(
            chosen_inputs, rejected_inputs)
        
        # # Forward pass
        # with torch.no_grad():
        #     # 参考模型的 forward pass，确保使用相同的输入格式
        #     ref_outputs = ref_model(
        #         input_ids=combined_inputs["input_ids"],
        #         attention_mask=combined_inputs.get("attention_mask", None),
        #         pixel_values=combined_inputs.get("pixel_values", None),
        #         image_grid_thw=combined_inputs.get("image_grid_thw", None),
        #         pixel_values_videos=combined_inputs.get("pixel_values_videos", None),
        #         video_grid_thw=combined_inputs.get("video_grid_thw", None),
        #         cu_seqlens=combined_inputs.get("cu_seqlens", None)
        #     )
        #     if torch.isnan(ref_outputs.logits).any() or torch.isinf(ref_outputs.logits).any():
        #         print_rank_0("====rlhf==== ERROR: ref_outputs.logits contains nan or inf values!")
        #         raise ValueError("ref_outputs.logits contains nan or inf values")
            
        #     ref_chosen_outputs, ref_rejected_outputs = split_outputs(
        #         ref_outputs, sequence_lengths)

        # reward模型的 forward pass
        reward_outputs = model(
            input_ids=combined_inputs["input_ids"],
            attention_mask=combined_inputs.get("attention_mask", None),
            pixel_values=combined_inputs.get("pixel_values", None),
            image_grid_thw=combined_inputs.get("image_grid_thw", None),
            pixel_values_videos=combined_inputs.get("pixel_values_videos", None),
            video_grid_thw=combined_inputs.get("video_grid_thw", None),
            cu_seqlens=combined_inputs.get("cu_seqlens", None)
        )
        # if torch.isnan(reward_outputs.logits).any() or torch.isinf(reward_outputs.logits).any():
        #     print_rank_0("====rlhf==== ERROR: reward_outputs.logits contains nan or inf values!")
        #     raise ValueError("reward_outputs.logits contains nan or inf values")  
        
        if torch.isnan(reward_outputs.reward_logits).any() or torch.isinf(reward_outputs.reward_logits).any():
            print_rank_0("====rlhf==== ERROR: reward_outputs.reward_logits contains nan or inf values!")
            raise ValueError("reward_outputs.reward_logits contains nan or inf values")
        
        reward_chosen_outputs, reward_rejected_outputs = split_outputs(
            reward_outputs, sequence_lengths)

        # 计算 log probabilities，使用 mask 确保只考虑有效 token
        chosen_mask = chosen_inputs["loss_mask"]
        rejected_mask = rejected_inputs["loss_mask"]

        # with torch.no_grad():
        #     ref_chosen_logps = get_batch_rewards(
        #         ref_chosen_outputs.logits, 
        #         chosen_inputs["input_ids"],
        #         chosen_mask,
        #         average_log_prob=True
        #     )
        #     ref_rejected_logps = get_batch_rewards(
        #         ref_rejected_outputs.logits,
        #         rejected_inputs["input_ids"],
        #         rejected_mask,
        #         average_log_prob=True
        #     )

        # 计算reward模型的reward，保持梯度
        reward_chosen, chosen_labels, chosen_masks = get_batch_rewards(
            reward_chosen_outputs.reward_logits,
            chosen_inputs["input_ids"],
            chosen_mask,
            average_log_prob=True
        )
        reward_rejected, rejected_labels, rejected_masks = get_batch_rewards(
            reward_rejected_outputs.reward_logits,
            rejected_inputs["input_ids"],
            rejected_mask,
            average_log_prob=True
        )

        # 计算 RLHF loss
        loss, chosen_rewards, rejected_rewards = compute_rlhf_loss(
            chosen_rewards=reward_chosen,
            rejected_rewards=reward_rejected,
            chosen_token_ids=chosen_inputs["input_ids"],
            rejected_token_ids=rejected_inputs["input_ids"],
            chosen_sample_idx=chosen_inputs["sample_idx"],
            rejected_sample_idx=rejected_inputs["sample_idx"],
            loss_style=args.loss_style,
            eos_token_id=args.eos_id,
            pad_id=args.pad_id
        )

        loss_fn = CrossEntropyLoss(
            ignore_index=-100, return_token_loss=True, shift_labels=False)
        chosen_input_ids = chosen_inputs["input_ids"] * (chosen_inputs["input_ids"] > 0).to(torch.int64)
        chosen_labels = chosen_input_ids * chosen_mask + loss_fn.ignore_index * (1 - chosen_mask)
        chosen_pad = torch.full((chosen_labels.shape[0], 1), loss_fn.ignore_index, dtype=chosen_labels.dtype).to(device=chosen_labels.device)
        chosen_labels = torch.cat([chosen_labels[:, 1:], chosen_pad], dim=-1) # shift
        local_chosen_labels = get_local_sequence(chosen_labels, seq_idx=1)
        chosen_loss, per_token_chosen_loss = loss_fn(logits=reward_chosen_outputs.logits, labels=local_chosen_labels)

        rejected_input_ids = rejected_inputs["input_ids"] * (rejected_inputs["input_ids"] > 0).to(torch.int64)
        rejected_labels = rejected_input_ids * rejected_mask + loss_fn.ignore_index * (1 - rejected_mask)
        rejected_pad = torch.full((rejected_labels.shape[0], 1), loss_fn.ignore_index, dtype=rejected_labels.dtype).to(device=rejected_labels.device)
        rejected_labels = torch.cat([rejected_labels[:, 1:], rejected_pad], dim=-1) # shift
        local_rejected_labels = get_local_sequence(rejected_labels, seq_idx=1)
        rejected_loss, per_token_rejected_loss = loss_fn(logits=reward_rejected_outputs.logits, labels=local_rejected_labels)

        # 使用 DeepSpeed 进行反向传播
        print_rank_0("====rlhf==== Backward...")
        model.backward(loss)
        print_rank_0(f"====rlhf==== Loss: {loss.item()}")
        model.step()
        print_rank_0(f"====rlhf==== Step... {acc_step}")
        # 统计信息
        input_ids = combined_inputs["input_ids"]
        # sample 只考虑 chosen 的
        sample_idx = chosen_inputs["sample_idx"]
        num_tokens = input_ids.numel()
        num_samples = (sample_idx.max() + 1).sum()
        num_valid_tokens = num_tokens - (input_ids == -1).sum()

        # 使用相同的方式在data parallel group中同步统计信息
        token_metrics = torch.tensor([num_tokens, num_samples, num_valid_tokens]).cuda()
        dist.all_reduce(token_metrics, op=dist.ReduceOp.SUM, group=get_data_parallel_group())

        num_tokens = token_metrics[0]
        num_samples = token_metrics[1]
        num_valid_tokens = token_metrics[2]

        # 更新总计数
        total_num_samples += num_samples.item()
        total_num_tokens += num_tokens.item()
        total_num_valid_tokens += num_valid_tokens.item()

        # 更新累积计数
        acc_num_samples += num_samples.item()
        acc_num_tokens += num_tokens.item()
        acc_valid_num_tokens += num_valid_tokens.item()

        start_time = time.time()
        
        avg_loss = torch.tensor(loss.item()).cuda()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()
        acc_avg_loss += avg_loss
        acc_step += 1

        avg_chosen_loss = torch.tensor(chosen_loss.item()).cuda()
        dist.all_reduce(avg_chosen_loss, op=dist.ReduceOp.SUM)
        avg_chosen_loss = avg_chosen_loss.item() / dist.get_world_size()
        acc_chosen_loss += avg_chosen_loss

        avg_rejected_loss = torch.tensor(rejected_loss.item()).cuda()
        dist.all_reduce(avg_rejected_loss, op=dist.ReduceOp.SUM)
        avg_rejected_loss = avg_rejected_loss.item() / dist.get_world_size()
        acc_rejected_loss += avg_rejected_loss

        # 在反向传播和优化器步骤之后添加日志记录逻辑
        if iteration % args.logging_per_step == 0 and model.is_gradient_accumulation_boundary():
            # 计算平均 loss
            avg_loss = acc_avg_loss / acc_step
            avg_chosen_loss = acc_chosen_loss / acc_step
            avg_rejected_loss = acc_rejected_loss / acc_step
            
            # 获取学习率
            model_lrs = model.lr_scheduler.get_lr()
            learning_rate = model_lrs[0]
            if len(model_lrs) > 2:
                vision_learning_rate = model.lr_scheduler.get_lr()[2]
            else:
                vision_learning_rate = model.lr_scheduler.get_lr()[1]
            
            # 计算性能指标
            end_time = time.time()
            sec_per_step = (end_time - start_time) / acc_step
            tokens_per_sec_per_gpu = acc_num_tokens / dist.get_world_size() / (end_time - start_time)
            samples_per_sec_per_gpu = acc_num_samples / dist.get_world_size() / (end_time - start_time)
            valid_tokens_per_sec_per_gpu = acc_valid_num_tokens / dist.get_world_size() / (end_time - start_time)
            
            # 日志字典
            log_dict = {
                "training/loss": avg_loss,
                "training/chosen_loss": avg_chosen_loss,
                "training/rejected_loss": avg_rejected_loss,
                "training/chosen_rewards": chosen_rewards.mean().item(),
                "training/rejected_rewards": rejected_rewards.mean().item(),
                "training/reward_gap": (chosen_rewards - rejected_rewards).mean().item(),
                "training/grad_norm": model.get_global_grad_norm(),
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
            
            # TensorBoard 记录
            if dist.get_rank() == 0 and tb_writer:
                for name, data in log_dict.items():
                    if data is not None:
                        tb_writer.add_scalar(
                        name,
                        data, 
                        global_step=iteration,
                        new_style=True
                    )

                    # 按有效token数记录指标
                    if name.startswith("training/"):
                        tb_writer.add_scalar(
                            f"x_token_{name}",
                            data,
                            global_step=total_num_valid_tokens,
                            new_style=True
                        )
            
            # 打印训练信息,格式与 pretrain_vl.py 保持一致
            print_rank_0(
                f"Step: {iteration}, Loss: {avg_loss}, "
                f"Chosen Loss: {avg_chosen_loss}, Rejected Loss: {avg_rejected_loss}, "
                f"Learning Rate: {learning_rate}, "
                f"Grad Norm: {model.get_global_grad_norm()}, "
                f"Sec per Step: {sec_per_step}, "
                f"tokens_per_sec_per_gpu: {tokens_per_sec_per_gpu}, "
                f"samples_per_sec_per_gpu: {samples_per_sec_per_gpu}, "
                f"total_num_tokens: {total_num_tokens}, "
                f"total_num_samples: {total_num_samples}, "
                f"valid_tokens_per_sec_per_gpu: {valid_tokens_per_sec_per_gpu}, "
                f"total_num_valid_tokens: {total_num_valid_tokens}, "
                f"valid_tokens_ratio: {1.0 * total_num_valid_tokens / total_num_tokens}, "
                f"chosen_rewards: {chosen_rewards.mean().item():.4f}, "
                f"rejected_rewards: {rejected_rewards.mean().item():.4f}"
            )
            
            # 心跳监控
            if args.heartbeat_monitor:
                heart_beat(int(acc_num_tokens))

            # 重置统计
            acc_step = 0
            acc_avg_loss = 0.0
            acc_chosen_loss = 0.0
            acc_rejected_loss = 0.0
            acc_num_samples = 0
            acc_num_tokens = 0
            acc_valid_num_tokens = 0
            start_time = end_time
    
    # 在训练循环结束后保存最终checkpoint
    print_rank_0("====rlhf==== Saving final checkpoint...")
    model.save_checkpoint(
        save_dir=args.output_dir,
        client_state={
            "total_num_valid_tokens": total_num_valid_tokens,
            "total_num_tokens": total_num_tokens,
            "total_num_samples": total_num_samples,
            "total_data_source_samples": total_data_source_samples,
            "total_data_source_tokens": total_data_source_tokens
        }
    )

    # 保存dataloader状态
    try:
        dataloader_state_dict = {
            "dataloader_state_dict": dataloader.state_dict()
        }
    except:
        dataloader_state_dict = None
        logging.error("====rlhf==== Dataloader cannot dump state_dict!!!!!!!!")

    if dataloader_state_dict is not None:
        dataloader_path = os.path.join(args.output_dir, "dataloader_ckpt")
        if dist.get_rank() == 0:
            os.makedirs(dataloader_path, exist_ok=True)
        dist.barrier()
        torch.save(
            dataloader_state_dict,
            os.path.join(dataloader_path, f"rank{dist.get_rank()}_final.pth")
        )
    
    # 合并检查点
    if args.merge_checkpoint and dist.get_rank() == 0:
        convert_zero_checkpoint_to_state_dict(
            args.output_dir,
            output_file=args.merge_checkpoint_output_file,
            dtype=args.merge_checkpoint_dtype
        )
    
    if dist.get_rank() == 0:
        logging.info("====rlhf==== Training finished!")

    # 为了调试sequence parallel问题，添加同步点和shape检查
    dist.barrier()  # 确保所有进程同步到这里
    print_rank_0(f"====rlhf==== Rank {dist.get_rank()} finished forward pass")
    print_rank_0(f"====rlhf==== chosen_input_ids shape: {chosen_inputs['input_ids'].shape}")
    print_rank_0(f"====rlhf==== rejected_input_ids shape: {rejected_inputs['input_ids'].shape}")

    # 在训练循环中添加定期保存checkpoint的逻辑
    if iteration % args.save_checkpoint_per_step == 0 and \
        iteration > 0 and model.is_gradient_accumulation_boundary():
        
        torch.cuda.empty_cache()

        with Timer("save checkpoint"):
            model.save_checkpoint(
                save_dir=args.output_dir,
                client_state={
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
                logging.error(f"====rlhf==== Dataloader cannot dump state_dict!!!!!!!!")
            
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
                        f"rank{dist.get_rank()}_global_step{iteration}.pth"
                    )
                )



if __name__ == "__main__":
    train()
