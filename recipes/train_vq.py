import os

import torch
torch.autograd.set_detect_anomaly(True)  # 开启异常检测

#torch.use_deterministic_algorithms(True)

from typing import Dict, Any, Union, Optional
import datetime
process_group_timeout = datetime.timedelta(minutes=60*24)

import contextlib
import gc
gc.disable()
import argparse
import time
import collections
from collections import defaultdict
import datetime
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import json
import logging
import pickle
import itertools
import contextlib
import multiprocessing as mp
import psutil
import threading
import queue
import traceback
from functools import partial
from tools.mfu.flops_counter import MFUStats

from recovlm.training.checkpoint import AppState, DistributedCheckpointer
from recovlm.models.qwen2_vl.checkpoint import Qwen2VLCheckpointConverter
from recovlm.models.internvl.checkpoint import InternVLCheckpointConverter
from recovlm.models.qwen_2_5_vl.checkpoint import Qwen2_5_VL_moonvitCheckpointConverter
from recovlm.models.qwen_2_5_vl.checkpoint import Qwen2_5_VL_siglipCheckpointConverter

from recovlm.models.tokenizer.keye_tokenizer import KeyeImageTokenizer


from recovlm.utils.ds_utils import print_input_info

#import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

from recovlm.models.qwen_2_5_vl import Qwen2_5_VLForConditionalGeneration
from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_moonvit,Qwen2_5_VLForConditionalGeneration_siglip,Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration_siglip_navit
from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit,Qwen2_5_VLProcessor_siglip
from recovlm.models.qwen3siglip.modeling_qwen3siglip import Qwen3SiglipForConditionalGeneration_navit
from recovlm.models.keye.modeling_keye import KeyeForConditionalGeneration, KeyeDecoderLayer
from recovlm.models.keye.modeling_keye import SiglipEncoderLayer as KeyeSiglipEncoderLayer
from recovlm.models.keye_vitrope.modeling_keye import KeyeForConditionalGeneration as KeyeForConditionalGeneration_vitrope, KeyeDecoderLayer as KeyeDecoderLayer_vitrope
from recovlm.models.keye_vitrope.modeling_keye import SiglipEncoderLayer as KeyeSiglipEncoderLayer_vitrope

from recovlm.models.keye_vitrope_slowfast_tatok.modeling_keye import KeyeForConditionalGeneration as KeyeForConditionalGeneration_vitrope_slowfast_tatok
from recovlm.models.keye_vitrope_slowfast_tatok.modeling_keye import  KeyeDecoderLayer as KeyeDecoderLayer_vitrope_slowfast_tatok
from recovlm.models.keye_vitrope_slowfast_tatok.modeling_keye import SiglipEncoderLayer as KeyeSiglipEncoderLayer_vitrope_slowfast_tatok

from recovlm.models.keye_vitrope_slowfast.modeling_keye import KeyeForConditionalGeneration as KeyeForConditionalGeneration_vitrope_slowfast
from recovlm.models.keye_vitrope_slowfast.modeling_keye import  KeyeDecoderLayer as KeyeDecoderLayer_vitrope_slowfast
from recovlm.models.keye_vitrope_slowfast.modeling_keye import SiglipEncoderLayer as KeyeSiglipEncoderLayer_vitrope_slowfast

from recovlm.models.keye_vitrope_slowfast_v2.modeling_keye import KeyeForConditionalGeneration as KeyeForConditionalGeneration_vitrope_slowfast_v2
from recovlm.models.keye_vitrope_slowfast_v2.modeling_keye import  KeyeDecoderLayer as KeyeDecoderLayer_vitrope_slowfast_v2
from recovlm.models.keye_vitrope_slowfast_v2.modeling_keye import SiglipEncoderLayer as KeyeSiglipEncoderLayer_vitrope_slowfast_v2

from recovlm.models.keye_vitrope_slowfast_v3.modeling_keye import KeyeForConditionalGeneration as KeyeForConditionalGeneration_vitrope_slowfast_v3
from recovlm.models.keye_vitrope_slowfast_v3.modeling_keye import  KeyeDecoderLayer as KeyeDecoderLayer_vitrope_slowfast_v3
from recovlm.models.keye_vitrope_slowfast_v3.modeling_keye import SiglipEncoderLayer as KeyeSiglipEncoderLayer_vitrope_slowfast_v3

from recovlm.models.keye_vitrope_slowfast_v4.modeling_keye import KeyeForConditionalGeneration as KeyeForConditionalGeneration_vitrope_slowfast_v4
from recovlm.models.keye_vitrope_slowfast_v4.modeling_keye import  KeyeDecoderLayer as KeyeDecoderLayer_vitrope_slowfast_v4
from recovlm.models.keye_vitrope_slowfast_v4.modeling_keye import SiglipEncoderLayer as KeyeSiglipEncoderLayer_vitrope_slowfast_v4
from recovlm.models.tokenizer.keye_tokenizer import SiglipEncoderLayer, KeyeImageTokenizer

from recovlm.models.internvl import InternVLChatModel
from recovlm.models.qwen2 import Qwen2DecoderLayer
from recovlm.models.internvl import InternVisionEncoderLayer

from recovlm.data.dataloaders_v2 import get_dataloader as get_dataloader_v2
from recovlm.data.dataloaders import get_dataloader

from recovlm.utils.merge_checkpoints import convert_zero_checkpoint_to_state_dict
from recovlm.utils.numa_bind import get_numa_bind_info
from recovlm.losses import CrossEntropyLoss
from recovlm.utils.common import set_random_seed, to_cuda, to_device, print_rank_0, \
  get_optimizer_grouped_parameters, dist_reduce_dict, Timer, heart_beat
from recovlm.training.lr_schedulers import get_scheduler

from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_rank, get_sequence_parallel_world_size, \
  get_local_sequence_boundary, initialize_model_parallel, gather_by_group, \
  get_local_sequence, get_data_parallel_group, get_data_parallel_world_size, \
  get_data_parallel_rank, gather_batches

from torch.distributed.device_mesh import init_device_mesh, DeviceMesh

from recovlm.training.distributed import shard_model, get_shard_conditions, \
  load_from_full_model_state_dict
from recovlm.training.checkpoint import load_hf_checkpoint

from recovlm.training.activations import set_activation_checkpointing

from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value, compute_fsdp_zero2_grad_norm

from recovlm.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer, Qwen2VLVisionBlock
from recovlm.models.qwen_2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock
from recovlm.models.qwen3siglip.modeling_qwen3siglip import Qwen3SiglipDecoderLayer
from recipes.ViT.training.models.MoonVision.modeling_kimi_vl import MoonVitEncoderLayer
from recipes.ViT.training.models.siglip.modeling_siglip import SiglipEncoderLayer
from recovlm.utils.time_tracker import TimeTracker
from recovlm.utils.ds_utils import format_dict_or_list
from recovlm.models.qwen3siglip.processing_qwen3siglip import Qwen3SiglipProcessor


# Logger 初始化
#logging.basicConfig(level=logging.INFO)  # 设置日志级别
#logger = logging.getLogger(__name__)  # 创建 logger 实例

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
  
  parser.add_argument("--fp32_weight", action="store_true",
                      help="Whether use fp32 for model weight updating")



  parser.add_argument("--fp32_reduce", action="store_true",
                      help="Whether use fp32 for model gradient reduction")

  parser.add_argument("--reshard_after_forward", action="store_true",
                      help="enable reshard_after_forward to enable Zero3 (default)")

  parser.add_argument("--save_checkpoint_per_step", type=int, default=1000,
                      help="The number of steps to save a checkpoint")

  parser.add_argument("--save_checkpoint_every_epoch", action="store_true",
                      help="Save checkpoint at the end of every epoch")
  
  parser.add_argument("--load_weights_only", action="store_true",
                      help="Only load model weights.")

  parser.add_argument("--compile", action="store_true",
                      help="compile model.")

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

  parser.add_argument("--model_class", type=str, default="Qwen2_5_VLForConditionalGeneration_moonvit",
                      help="The model class, one of 'Qwen2VLForConditionalGeneration' or 'Qwen2_5_VLForConditionalGeneration','Qwen2_5_VLForConditionalGeneration_moonvit','Qwen2_5_VLForConditionalGeneration_siglip', 'Qwen2_5_VLForConditionalGeneration_siglip_navit', 'KeyeForConditionalGeneration', 'KeyeForConditionalGeneration_vitrope', 'KeyeForConditionalGeneration_vitrope_slowfast', 'KeyeForConditionalGeneration_vitrope_slowfast_tatok', 'KeyeForConditionalGeneration_vitrope_slowfast_v2', 'KeyeForConditionalGeneration_vitrope_slowfast_v3', 'KeyeForConditionalGeneration_vitrope_slowfast_v4', 'InternVLChatModel'",)
  
  parser.add_argument("--model_processor", type=str, default="Qwen2_5_VLProcessor_moonvit",
                      help="The model processor class, one of 'Qwen2VLProcessor' or 'Qwen2_5_VLProcessor' or 'Qwen2_5_VLProcessor_moonvit' or 'Qwen3SiglipProcessor' or 'KeyeProcessor' or 'KeyeProcessor_vitrope'")

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

  parser.add_argument("--clip_range", type=float, default=1.0,
                      help="The gradient clip range.")

  parser.add_argument("--freeze_projector", action="store_true",
                      help="Freeze all LLM parameters (language model weights will not be updated during training).")

  parser.add_argument("--freeze_vit", action="store_true",
                      help="Freeze all visual encoder parameters except visual projector layers.")
  
  parser.add_argument("--update_vision_tower", action="store_true",
                      help="Update all vision_tower parameters.")


  parser.add_argument("--use_flash_attention_2", action="store_true",
                      help="Whether to use flash attention 2")

  parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                      help="Enable gradient checkpointing during training")

  parser.add_argument("--prefetch_parameters", action="store_true",
                      help="prefetch fsdp parameters")

  parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Gradient accumulation steps.")

  parser.add_argument("--allow_random_init_params", type=str, default='',
                      help="-")
  
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

  parser.add_argument("--monitor_image_tokens", action="store_true",
                      help="Whether to monitor image tokens. Note that this involves with an gather operation, which is time-consuming")
  
  return parser



def _init_profiler(output_dir) -> None:
    import torch.distributed as D
    import os
    if not os.path.exists(output_dir):
        if D.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)

    def trace_handler(prof):
        # if D.get_rank() == 0:
        prof.export_chrome_trace(
            os.path.join(output_dir, str(prof.step_num) + f"_w{dist.get_rank()}" + ".json")
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
        save_dir: 保存目录
        tag: checkpoint标签，如果不指定则使用时间戳
        client_state: 需要保存的额外状态信息
        dataloader: 可选的dataloader，用于保存数据加载状态
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
    )
    
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    # 生成checkpoint标签
    if tag is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = f"checkpoint_{timestamp}"
    
    ckpt_path = os.path.join(save_dir, tag)
    if dist.get_rank() == 0:
        os.makedirs(ckpt_path, exist_ok=True)
        
        # 更新latest文件
        with open(os.path.join(save_dir, "latest"), "w") as f:
            f.write(tag)
    
    # 配置FSDP state_dict
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    
    try:
        dist_checkpointer.save_checkpoint(
                    state_dict={"app": app_state},
                    output_dir=ckpt_path,           
                    tag=str(global_step)
                )

        # 保存dataloader状态（如果有）
        if dataloader is not None:
            try:
                dataloader_state = {
                    "dataloader_state_dict": dataloader.state_dict()
                }
                dataloader_path = os.path.join(ckpt_path, "dataloader_ckpt")
                if dist.get_rank() == 0:
                    os.makedirs(dataloader_path, exist_ok=True)
                dist.barrier()
                
                # 每个rank保存自己的dataloader状态
                torch.save(
                    dataloader_state,
                    os.path.join(dataloader_path, f"rank{dist.get_rank()}.pt")
                )
                print_rank_0(f"Saved dataloader state to {dataloader_path}")
            except:
                import traceback
                logging.error(f"Failed to save dataloader state! dataloader({type(dataloader)})={dataloader} \ntraceback:{traceback.format_exc()}")

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
        # 确保所有进程同步
        dist.barrier()


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


def freeze_params(args, model):

  if args.freeze_vit:
    print_rank_0("Freeze vit parameters.")
    for name, param in model.named_parameters():
      if name.startswith("visual"):
        print_rank_0(f"Disable LLM grad: {name}")
        param.requires_grad = False
    print_rank_0("=" * 50)

  if args.freeze_projector:
    print_rank_0("Freeze visual.merger parameters.")
    for name, param in model.named_parameters():
      if name.startswith("mlp_AR"):
        print_rank_0(f"Disable projector grad: {name}")
        param.requires_grad = False
    print_rank_0("=" * 50)

class TokenStats:
  def __init__(self, args, _type='image'):
    self.max_image_tokens = []
    self.min_image_tokens = []
    self.mean_image_tokens = []
    self.std_image_tokens = []
    self.args = args
    self._type = _type

  def collect_image_token_stats(self, num_image_tokens):
      # 收集所有rank的image tokens统计信息
      world_size = dist.get_world_size()
      rank = dist.get_rank()
      input_tensor = torch.tensor([num_image_tokens], dtype=torch.long).cuda()
      all_image_tokens = list(torch.zeros(world_size, dtype=torch.long).cuda().chunk(world_size) ) if rank == 0 else None
      dist.gather(input_tensor, gather_list=all_image_tokens, dst=0)

      if rank == 0:
          all_image_tokens = [x.item() for x in all_image_tokens]
          # 计算统计指标
          max_image_tokens = max(all_image_tokens)
          min_image_tokens = min(all_image_tokens)
          mean_image_tokens = sum(all_image_tokens) / world_size
          std_image_tokens = (sum((x - mean_image_tokens)**2 for x in all_image_tokens) / world_size)**0.5
      else:
          max_image_tokens = 0
          min_image_tokens = 0
          mean_image_tokens = 0
          std_image_tokens = 0

      self.max_image_tokens.append(max_image_tokens)
      self.min_image_tokens.append(min_image_tokens)
      self.mean_image_tokens.append(mean_image_tokens)
      self.std_image_tokens.append(std_image_tokens)
      return max_image_tokens, min_image_tokens, mean_image_tokens, std_image_tokens

  def stats(self):
      res = np.max(self.max_image_tokens), np.min(self.min_image_tokens),\
             np.mean(self.mean_image_tokens), np.mean(self.std_image_tokens)
      res = {
        f"perf/max_{self._type}_tokens": res[0],
        f"perf/min_{self._type}_tokens": res[1],
        f"perf/mean_{self._type}_tokens": res[2],
        f"perf/std_{self._type}_tokens": res[3]
      }
      self.max_image_tokens.clear()
      self.min_image_tokens.clear()
      self.mean_image_tokens.clear()
      self.std_image_tokens.clear()
      return res


def data_prefetch_fn(data_iter, batch_queue, sp_size):
  while True:
    try:
      t1 = time.perf_counter()
      batch = next(data_iter)
      t2 = time.perf_counter()
      if sp_size > 1:
        batches = gather_batches([batch], get_sequence_parallel_group())
        t3 = time.perf_counter()
        # print(f"rank={dist.get_rank()} get_one_batch: {t2-t1}, all_gather={t3-t2}")
        for b in batches:
          batch_queue.put(b)
      else:
        batch_queue.put(batch)
        t3 = time.perf_counter()
        # print(f"rank={dist.get_rank()}, get_one_batch: {t2-t1}")
    except Exception as e:
      traceback.print_exc()
      batch_queue.put(None)

class FakeConverter:
  def __init__(self, model_path_or_name: str = None):
    self.model_path_or_name = model_path_or_name

  def __call__(self, state_dict):
     return self.convert(state_dict)

  def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return state_dict

  def revert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return state_dict


def train():
  arg_parser = get_argument_parser()
  args = arg_parser.parse_args()

  resume_from, ckpt_id, rewrite_resume_flag = get_resume_info(args)
  
  if rewrite_resume_flag:
    args.resume_dataloader = True
    args.load_weights_only = False
    print_rank_0(f"WARN: --resume_dataloader is rewrited to True \n" \
                 f"WARN: --load_weights_only is rewrited to False \n")

  # check vision_lr
  assert args.learning_rate > 0.0

  assert all([args.commit_id, args.seed, args.comment]), \
    "Git commit, seed, and comment is required for reproducibility"

  assert any([args.save_checkpoint_per_step, args.save_checkpoint_every_epoch]), \
      "The checkpoint saving frequency is not set, save_checkpoint_per_step or " \
      "save_checkpoint_every_epoch should be set."

  rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
  world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
  local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
  local_world_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 0))
  cpu_bind = get_numa_bind_info(local_rank, local_world_size)

  ##############
  with open(args.dataset_config, encoding="utf-8") as f:
    dataset_config = json.loads(f.read())
  dataset = dataset_config.pop("name")
  dataset_config["model_class"] = args.model_class
  if args.max_length:
    dataset_config["max_length"] = args.max_length
  use_flops_balance = dataset_config.get("use_flops_balance", False)


  # torch init
  torch.cuda.set_device(local_rank)
  torch.distributed.init_process_group(rank=rank, world_size=world_size, timeout=process_group_timeout)
  device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))

  ### initialize model parallel group
  initialize_model_parallel(args.sequence_parallel_size)
  print_rank_0(f"Sequence parallel size: {get_sequence_parallel_world_size()}")

  set_random_seed(args.seed)

  ####### for pdb debug #######
  print("args.model_class:", args.model_class)


  state_dict = None

  converter = FakeConverter()

  if dist.get_rank() == 0:
    with set_default_dtype(torch.bfloat16):
      print("load_hf_checkpoint--------------:", args.model_dir)
      state_dict = load_hf_checkpoint(args.model_dir)
      state_dict = converter(state_dict)

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


  with set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = KeyeImageTokenizer.from_pretrained(
      args.model_dir, _attn_implementation="flash_attention_2",
      use_cache = False,
      ignore_mismatched_sizes=True, 
    )

  # if args.enable_gradient_checkpointing:
  #   print_rank_0("Enable gradient checkpointing")

  #   auto_wrap_policy_mapping = {
  #     "Qwen2VLForConditionalGeneration": {Qwen2VLDecoderLayer, Qwen2VLVisionBlock},
  #     "Qwen2_5_VLForConditionalGeneration": {Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock},
  #     "Qwen2_5_VLForConditionalGeneration_moonvit": {Qwen2_5_VLDecoderLayer, MoonVitEncoderLayer},
  #     "Qwen2_5_VLForConditionalGeneration_siglip": {Qwen2_5_VLDecoderLayer, SiglipEncoderLayer},
  #     "Qwen3SiglipForConditionalGeneration_navit": {Qwen3SiglipDecoderLayer, SiglipEncoderLayer},
  #     "Qwen2_5_VLForConditionalGeneration_siglip_navit": {Qwen2_5_VLDecoderLayer, SiglipEncoderLayer},
  #     "KeyeForConditionalGeneration":  {KeyeDecoderLayer, KeyeSiglipEncoderLayer},
  #     "KeyeForConditionalGeneration_vitrope":  {KeyeDecoderLayer_vitrope, KeyeSiglipEncoderLayer_vitrope},
  #     "KeyeForConditionalGeneration_vitrope_slowfast": {KeyeDecoderLayer_vitrope_slowfast, KeyeSiglipEncoderLayer_vitrope_slowfast},
  #     "KeyeForConditionalGeneration_vitrope_slowfast_tatok": {KeyeDecoderLayer_vitrope_slowfast_tatok, KeyeSiglipEncoderLayer_vitrope_slowfast_tatok},
  #     "KeyeForConditionalGeneration_vitrope_slowfast_v2": {KeyeDecoderLayer_vitrope_slowfast_v2, KeyeSiglipEncoderLayer_vitrope_slowfast_v2},
  #     "KeyeForConditionalGeneration_vitrope_slowfast_v3": {KeyeDecoderLayer_vitrope_slowfast_v3, KeyeSiglipEncoderLayer_vitrope_slowfast_v3},
  #     "KeyeForConditionalGeneration_vitrope_slowfast_v4": {KeyeDecoderLayer_vitrope_slowfast_v4, KeyeSiglipEncoderLayer_vitrope_slowfast_v4},
  #     "InternVLChatModel":{Qwen2DecoderLayer,InternVisionEncoderLayer}
  #   }
  #   set_activation_checkpointing(
  #     model, auto_wrap_policy=auto_wrap_policy_mapping[args.model_class]
  #   )
  if args.enable_gradient_checkpointing:
    print_rank_0("Enable gradient checkpointing")
    set_activation_checkpointing(
      model, auto_wrap_policy={SiglipEncoderLayer}
    )

    
  if args.fp32_weight: model = model.float()
  shard_model(
    model=model,
    shard_conditions=[partial(get_shard_conditions, model_class=args.model_class)],
    cpu_offload=False,
    reshard_after_forward=args.reshard_after_forward,
    dp_mesh=device_mesh,
    fp32_weight=args.fp32_weight,
    prefetch_parameters=args.prefetch_parameters,
    model_class=args.model_class,
    fp32_reduce=args.fp32_reduce
  )
  print("rank ", dist.get_rank())
  dist.barrier()

  with Timer("Load state dict"):
    load_from_full_model_state_dict(
      model=model, full_sd=state_dict,
      allow_random_init_params=args.allow_random_init_params) # 这里应该全部转成CUDA了, meta -> CUDA

  with torch.device(torch.cuda.current_device()):
    for m in model.modules():
      # RoPE is not covered in state dict
      if hasattr(m, "rope_init"):
        print_rank_0("Initialize RoPE")
        m.rope_init()

  print(model)
  for name, param in model.named_parameters():
    print(name, param.device)

  if state_dict is not None:
    del state_dict
  
  freeze_params(args=args, model=model)
  
  # print train params log
  for name, param in model.named_parameters():
    if param.requires_grad:
      print_rank_0(f"params not freeze: {name}")
  print_rank_0("=" * 50)

  # Split weights in two groups, one with weight decay and the other not.
  optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model,
    learning_rate=args.learning_rate,
    vision_learning_rate=args.learning_rate,
    weight_decay=args.weight_decay
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
  global_step = 0
  app_state = AppState(model=model)
  dist_checkpointer = DistributedCheckpointer()

  if ckpt_id:
    ckpt_path = os.path.join(resume_from, ckpt_id)
    global_step = global_step0 = int(ckpt_id.split("step")[-1])
    print_rank_0(
      f"Resume from checkpoint: {ckpt_path}, global_step={global_step}"
      f"load_weights_only={args.load_weights_only}")

    if not os.path.exists(ckpt_path):
      raise ValueError(f"Checkpoint path {ckpt_path} does not exist")

    # 只加载模型参数，不考虑优化器状态
    client_state = {}

    # 获取state_dict用于加载
    state_dict = {"app": app_state.set_call_back(converter.convert)}
              
    # 使用DCP API加载分片数据
    dist_checkpointer.load_checkpoint(
        state_dict=state_dict,  # 提供state_dict参数
        checkpoint_dir=resume_from,
        tag=ckpt_id
    )
    
    print_rank_0(f"Successfully loaded model using distributed checkpoint")

    if args.resume_dataloader: # and not use_flops_balance:
      print_rank_0(f"resume_from={resume_from}, len={len(resume_from)}")
      dataloader_resume_path = os.path.join(resume_from, "dataloader_ckpt", f"rank{dist.get_rank()}.pt")
      optimizer_state_dict_path = os.path.join(resume_from, "optimizer_ckpt", f"rank{dist.get_rank()}.pt")
      optimizer_state_dict = torch.load(optimizer_state_dict_path)
      lr_scheduler.load_state_dict(optimizer_state_dict["scheduler_state_dict"])
      optimizer.load_state_dict(optimizer_state_dict["optimizer_state_dict"])
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
  torch_profiler = _init_profiler(output_dir=os.path.join(args.output_dir, "torch_profile"))

  # loss_fn = CrossEntropyLoss(
  #   ignore_index=-100, return_token_loss=True, shift_labels=False)

  start_time = time.time()
  start_time0 = start_time
  show_cnt = 1
  if not args.resume_dataloader:

    global_step = 0
    global_step0 = 0


  # Metrics, acc_ account for gradient accumulation
  # TODO: use mestrics manager
  acc_avg_loss = 0.0
  acc_num_tokens = 0
  acc_num_samples = 0
  acc_valid_num_tokens = 0
  acc_num_image_tokens = 0
  total_num_image_tokens = 0
  total_num_video_tokens = 0
  acc_num_video_tokens = 0
  mfu_stats = MFUStats(args)
  batch_data_source_loss = collections.defaultdict(float)
  batch_data_source_tokens = collections.defaultdict(int)
  valid_data_source_tokens = collections.defaultdict(int)
  grad_norm = 0.0

  # get_sequence_parallel_group("gloo")

  micro_step = 0
  ticker = TimeTracker(n=args.logging_per_step)
  iter_ticker = TimeTracker(n=args.logging_per_step)
  token_stasts = TokenStats(args, _type='image')
  vid_token_stasts = TokenStats(args, _type='video')

  gpu_batch_q = queue.Queue(maxsize=2)

  prefetch_t = None
  def prefetch_to_gpu(input_fn, output_q, dev):
    while True:
      try:
        batch = input_fn()
        to_device(batch, dev, True)
        output_q.put(batch)
      except StopIteration:
        break

  if use_flops_balance:
    input_fn = lambda: batch_queue.get()
  else:
    data_iter = iter(gather_by_group(dataloader, get_sequence_parallel_group()))
    prefetch_t = threading.Thread(target=prefetch_to_gpu, args=(lambda : next(data_iter), gpu_batch_q, torch.cuda.current_device()))
    prefetch_t.start()
    input_fn =  lambda: gpu_batch_q.get()
    # input_fn = lambda : next(data_iter)

  # prefetch_t = threading.Thread(target=prefetch_to_gpu, args=(input_fn, gpu_batch_q, torch.cuda.current_device()))
  # prefetch_t.start()

  tb_metrics_q = queue.Queue(maxsize=8)
  def write_tb_async(tb_writer, metrics_queue, grad_acc_steps):
    while True:
      # metrics = metrics_queue.get()
      global_step, log_dict, ticker_stats, ds_loss, ds_tokens, ds_samples = metrics_queue.get()
      total_num_samples = log_dict["perf/total_num_samples"]
      total_num_valid_tokens = log_dict["perf/valid_total_num_tokens"]
      for name, data in log_dict.items():
        if data is not None and tb_writer:
          # print(f"add_data_{global_step}", global_step, log_dict, ticker_stats, ds_loss, ds_tokens, ds_samples)

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
              global_step=total_num_valid_tokens / grad_acc_steps,
              new_style=True
            )

      for name, data in ticker_stats.items():
        tb_writer.add_scalar(f"ticker/{name}", data, global_step=global_step, new_style=True)

      if args.monitor_datasource_loss and tb_writer:
        for key, loss_sum in ds_loss.items():
          tb_writer.add_scalar(
                f"data_source_loss/{key}",
                loss_sum / (ds_tokens[key] + 1e-6) ,
                global_step=global_step,
                new_style=True)

      if args.monitor_datasource_cnt and tb_writer:
        total_samples = sum([x for _, x in ds_samples.items()])
        for key, samples in ds_samples.items():
          tb_writer.add_scalar(
              f"data_source_sample_ratio/{key}",
              1.0 * samples / total_samples,
              global_step=global_step,
              new_style=True)

        for key, num_tokens in ds_tokens.items():
          total_tokens = sum([x for _, x in ds_tokens.items()])
          tb_writer.add_scalar(
              f"data_source_token_ratio/{key}",
              1.0 * num_tokens / total_tokens,
              global_step=global_step,
              new_style=True)

  if dist.get_rank() == 0:
    tb_writer_t = threading.Thread(target=write_tb_async,args=(tb_writer, tb_metrics_q, args.gradient_accumulation_steps))
    tb_writer_t.start()

  total_data_source_samples = 0
  total_num_valid_tokens = 0
  total_num_tokens = 0 
  total_num_samples = 0

  while True:
    # print(f"rank={torch.distributed.get_rank()}, pid={os.getpid()}, begin_training")
    ticker.tick("while_True")

    for batch in gather_by_group(dataloader, get_sequence_parallel_group()):

      micro_step += 1
      # continue
      data_source = batch.pop("data_source", None) # dataset source list cur batch

      to_cuda(batch)
      ticker.tick("to_cuda(batch)")
      

      ###### dataset checker ######

      input_ids = batch["input_ids"]
      loss_mask = batch["loss_mask"]
      attention_mask = batch.get("attention_mask", None)
      pixel_values = batch.get("pixel_values", None)
      pixel_values_videos = batch.get("pixel_values_videos", None)
      image_grid_thw = batch.get("image_grid_thw", None)
      video_grid_thw = batch.get("video_grid_thw", None)
      cu_seqlens = batch.get("cu_seqlens", None)
      sample_idx = batch["sample_idx"]
      second_per_grid_ts = batch.get("second_per_grid_ts", None)
      position_ids = batch.get("position_ids", None)
      image_flags = batch.get("image_flags", None)
      epoch_idx = batch.get("epoch_idx", torch.tensor([0])).cpu().item()
      #####slowfast######
      ###################
      # 打印 token 数量

      token_count = input_ids.numel() / args.sequence_parallel_size  # 计算 token 数量
      print_rank_0(f"Iteration {micro_step}: Token count = {token_count}")
      num_tokens = token_count

      num_samples = (sample_idx.max() + 1).sum()  / args.sequence_parallel_size

      total_num_samples += num_samples
      total_num_tokens += num_tokens

      acc_num_samples += num_samples
      acc_num_tokens += num_tokens
      ticker.tick("acc_valid_num_tokens+=num_valid_tokens")
      

      print("########################### decode ###########################")
      # batch['input_ids'][0]:  tensor([151644,   8948,    198,  ..., 151643, 151643, 151643], device='cuda:3')
      # <|im_start|>system .......

      with Timer("Fwd"):
        print(pixel_values.shape)
        print(image_grid_thw.shape)
        output = model(
          pixel_values=pixel_values,
          image_grid_thw=image_grid_thw
        )    

        # TODO: codebook_loss && reconstruction_loss
        total_loss = output["loss"]
        codebook_loss = output["codebook_loss"]
        reconstruction_loss = output["reconstruction_loss"]
        perplexity = output["perplexity"]

        print(f"loss: {total_loss}, codebook_loss: {codebook_loss}, reconstruction_loss: {reconstruction_loss}, perplexity: {perplexity}")
        # TODO: support global perplexity

        ############ NOTE: add global batchsize ############ 

      with Timer("bwd"):
        print("loss.backward() begin----------")
        loss.backward()

        clip_grad_by_value(model, args.clip_range)
        ticker.tick("loss.backward")

        if (micro_step + 1) % args.gradient_accumulation_steps == 0:
          grad_norm = compute_fsdp_zero2_grad_norm(model)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          global_step += 1

          ticker.tick(f"optimizer.step*{args.gradient_accumulation_steps}")
          print("loss.backward() end----------")

      ########## dataset source monitor ###############

      #########################################
      avg_loss = loss.detach()
      codebook_loss = codebook_loss.detach()
      reconstruction_loss = reconstruction_loss.detach()

      avg_loss = avg_loss.item() / dist.get_world_size()
      codebook_loss = codebook_loss.item() / dist.get_world_size()
      reconstruction_loss = reconstruction_loss.item() / dist.get_world_size()

      ticker.tick("reduce_acc_avg_loss")
      log_acc_step = args.logging_per_step * args.gradient_accumulation_steps

      if global_step % args.logging_per_step == 0 and \
              (micro_step + 1) % args.gradient_accumulation_steps == 0:
    
        ticker.tick(f"token_stasts*{log_acc_step}")

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
          end_time = time.time()
          sec_per_step = (end_time - start_time) / args.logging_per_step # / args.gradient_accumulation_steps
          tokens_per_sec_per_gpu = \
            acc_num_tokens  / (end_time - start_time) / dist.get_world_size()
          samples_per_sec_per_gpu = \
            acc_num_samples  / (end_time - start_time) / dist.get_world_size()
          samples_per_step_per_gpu = \
            acc_num_samples  / dist.get_world_size()

          samples_per_step_per_gpu_v2 = total_num_samples / dist.get_world_size() / global_step
          samples_per_sec_per_gpu_v2 = total_num_samples / dist.get_world_size() / (end_time - start_time0)


          tokens_per_step_per_gpu_v2 = total_num_tokens / dist.get_world_size() / global_step
          tokens_per_sec_per_gpu_v2 = total_num_tokens / dist.get_world_size() / (end_time - start_time0)


          # avg_loss = acc_avg_loss / args.gradient_accumulation_steps / args.logging_per_step
          log_dict = {
            # max_image_tokens, min_image_tokens, mean_image_tokens, std_image_tokens
            "training/loss": avg_loss,
            "loss/total": avg_loss,
            "loss/codebook": codebook_loss,
            "loss/reconstruction": reconstruction_loss,
            "perplexity": perplexity,
            "nonzero_count": nonzero_count,
            "training/grad_norm": grad_norm,
            "training/learning_rate": learning_rate,
            "perf/sec_per_step": sec_per_step,
            "perf/tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
            "perf/samples_per_sec_per_gpu": samples_per_sec_per_gpu,
            "perf/total_num_tokens": total_num_tokens,
            "perf/total_num_samples": total_num_samples,
            "perf/num_sample_per_gpu": total_num_samples / dist.get_world_size(),
            "perf/samples_per_step_per_gpu": samples_per_step_per_gpu,
            "perf/num_sample_per_sec_per_gpu": total_num_samples / (end_time - start_time) / dist.get_world_size(),

            **mfu_stats.mfu(end_time - start_time, global_step - global_step0),
            "perf/samples_per_step_per_gpu_v2": samples_per_step_per_gpu_v2,
            "perf/samples_per_sec_per_gpu_v2": samples_per_sec_per_gpu_v2,
            "perf/tokens_per_step_per_gpu_v2": tokens_per_step_per_gpu_v2,
            "perf/tokens_per_sec_per_gpu_v2": tokens_per_sec_per_gpu_v2,
            "perf/epoch_idx": epoch_idx,
          }
          start_time = end_time
          if args.monitor_image_tokens: 
            log_dict.update(colleced_token_stasts)
          ticker.tick(f"log_dict*{log_acc_step}")

          ticker_stats = {}
          for t in [ticker, iter_ticker]:
              ticker_stats.update(t.stat())
          metrics_info = (global_step, log_dict, ticker_stats, batch_data_source_loss, batch_data_source_tokens, total_data_source_samples)

          tb_metrics_q.put(metrics_info)

          ticker.tick(f"tb_metrics_q.put")
              
          ticker.tick(f"tb_writer.add_scalar*{log_acc_step}")
          print_rank_0(
            f"Step: {global_step}, Loss: {avg_loss}, "
            f"Learning Rate: {learning_rate}, "
            f"Grad Norm: {grad_norm}, "
            f"Sec per Step: {sec_per_step}",
            format_dict_or_list(log_dict),
            "\n", format_dict_or_list({"mfu_stats": mfu_stats.mfu_per_step_per_gpu, "ticker": ticker.stat()})
          )
          # upload heart_beat to remote
          if args.heartbeat_monitor:
            heart_beat(int(acc_num_tokens))

        valid_data_source_tokens = collections.defaultdict(int)
      
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
                      client_state={
                          "total_num_tokens": total_num_tokens,
                          "total_num_samples": total_num_samples,
                          "total_data_source_samples": total_data_source_samples,
                          "total_data_source_tokens": total_data_source_tokens,
                          "optimizer": optimizer.state_dict(),
                      },
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      dataloader=data_iter if use_flops_balance else  dataloader,
                      app_state=app_state.set_call_back(converter.revert), # app_state.set_call_back(state_dict), # no need to convert 
                      dist_checkpointer=dist_checkpointer
                  )
        ticker.tick(f"save_ckpt*{args.save_checkpoint_per_step * args.gradient_accumulation_steps}") 
      if torch_profiler: torch_profiler.step()
  save_model_checkpoint(
                      model=model,
                      save_dir=args.output_dir,
                      tag=f"step{global_step}",
                      global_step=global_step,
                      client_state={
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





