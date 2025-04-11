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
  get_optimizer_grouped_parameters, dist_reduce_dict, Timer, heart_beat
from recovlm.training.lr_schedulers import get_scheduler

from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_rank, get_sequence_parallel_world_size, \
  get_local_sequence_boundary, initialize_model_parallel, gather_by_group, \
  get_local_sequence, get_data_parallel_group, get_data_parallel_world_size, \
  get_data_parallel_rank

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
  arg_parser = deepspeed.add_config_arguments(arg_parser)
  args = arg_parser.parse_args()

  # 初始化分布式环境
  os.environ["KML_ID"] = args.kml_id
  os.environ["KML_TASK_ID"] = args.kml_task_id
  
  deepspeed.init_distributed()

  # 初始化模型并行组
  initialize_model_parallel(args.sequence_parallel_size)
  print_rank_0(f"Sequence parallel size: {get_sequence_parallel_world_size()}")

  set_random_seed(args.seed)
  dist.barrier()

  # 打印基本信息
  if dist.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    print_rank_0("开始统计token数量...")

  processor = Qwen2VLProcessor.from_pretrained(args.model_dir)

  # 加载数据集配置
  with open(args.dataset_config, encoding="utf-8") as f:
    dataset_config = json.loads(f.read())
  dataset = dataset_config.pop("name")
  if args.max_length:
    print_rank_0(
      f"覆盖数据集配置中的max_length: "
      f"{dataset_config['max_length']} -> {args.max_length}")
    dataset_config["max_length"] = args.max_length

  # 构建数据加载器
  with Timer("构建数据加载器"):
    dataloader = get_dataloader(name=dataset, **dataset_config)

  # 统计变量初始化
  total_num_tokens = 0
  total_num_samples = 0
  total_num_valid_tokens = 0
  
  acc_step = 0
  acc_num_tokens = 0
  
  show_cnt = 1  # 显示样本数量

  # 迭代数据加载器
  for batch in gather_by_group(dataloader, get_sequence_parallel_group()):
    if show_cnt > 0 and dist.get_rank() == 0:
      with Timer("显示样本数据"):
        input_text = processor.tokenizer.decode(batch['input_ids'][0])
        print_rank_0(
            f"输入文本示例:\n\n{input_text}\n" + "=" * 100 + "\n\n")
        show_cnt -= 1

    # 获取输入数据
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    sample_idx = batch["sample_idx"]

    # 计算token数量
    num_tokens = input_ids.numel()
    num_samples = (sample_idx.max() + 1).sum()
    num_valid_tokens = num_tokens - (sample_idx == -1).sum()

    # 在数据并行组上汇总指标
    token_metrics = torch.tensor(
      [num_tokens, num_samples, num_valid_tokens]).cuda()
    dist.all_reduce(
      token_metrics, op=dist.ReduceOp.SUM, group=get_data_parallel_group())

    num_tokens = token_metrics[0].item()
    num_samples = token_metrics[1].item()
    num_valid_tokens = token_metrics[2].item()

    # 更新总计数
    total_num_samples += num_samples
    total_num_tokens += num_tokens
    total_num_valid_tokens += num_valid_tokens

    # 更新累积计数
    acc_num_tokens += num_tokens
        
    # 增加迭代计数
    acc_step += 1

    # 定期打印统计信息
    if acc_step % args.logging_per_step == 0 and dist.get_rank() == 0:
      print_rank_0(
        f"步骤: {acc_step}, "
        f"当前总token数: {total_num_tokens}, "
        f"当前总样本数: {total_num_samples}, "
        f"当前有效token数: {total_num_valid_tokens}, "
        f"有效token比例: {1.0 * total_num_valid_tokens / total_num_tokens:.4f}"
      )

  # 输出最终统计结果
  if dist.get_rank() == 0:
    print_rank_0("统计完成!")
    print_rank_0(f"总token数: {total_num_tokens}")
    print_rank_0(f"总样本数: {total_num_samples}")
    print_rank_0(f"有效token数: {total_num_valid_tokens}")
    print_rank_0(f"有效token比例: {1.0 * total_num_valid_tokens / total_num_tokens:.4f}")
    
    # 保存统计结果到文件
    stats = {
      "total_num_tokens": total_num_tokens,
      "total_num_samples": total_num_samples,
      "total_num_valid_tokens": total_num_valid_tokens,
      "valid_token_ratio": 1.0 * total_num_valid_tokens / total_num_tokens
    }
    
    with open(os.path.join(args.output_dir, "token_stats.json"), 'w', encoding="utf-8") as f:
      json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logging.info("token统计信息已保存到token_stats.json")

if __name__ == "__main__":
  train()