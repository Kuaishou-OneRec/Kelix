from typing import Dict, Any, Union, Optional
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import datetime
import contextlib
import gc
gc.disable()
import argparse
import time
import collections
import json #noqa: F401
import multiprocessing as mp #noqa: F401
import psutil #noqa: F401
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict

torch.autograd.set_detect_anomaly(True)

process_group_timeout = datetime.timedelta(minutes=60*24)


from muse.training.checkpoint import AppState, DistributedCheckpointer #noqa: F401

def get_argument_parser():
  parser = argparse.ArgumentParser()

  ############ Checkpoint args ############
  parser.add_argument("--model_dir", type=str, default=None,
                      help="The directory of the pretrained model.")

  parser.add_argument("--resume_from", type=str, default=None,
                      help="Specify the checkpoint directory to resume from.")
  
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

  parser.add_argument("--model_class", type=str, default="Qwen3Model",
                      help="The model class name.",)

  ############ Dataset args ############
  parser.add_argument("--dataset", type=str, default=None,
                      help="The comma seperated path of indexed json file.")
  parser.add_argument("--dataset_config", type=str, default=None,
                      help="The comma seperated path of indexed json file.")

  parser.add_argument("--max_length", type=int, default=None,
                      help="Max tokens per sentence in corpus")

  parser.add_argument("--shuffle_buffer_size", type=int, default=0,
                      help="Size of shuffle buffer for local data shuffling (0 to disable)")

  parser.add_argument("--enable_dataset_checkpointing", action="store_true",
                      help="Enable dataset checkpoint recovery")

  parser.add_argument("--dataset_checkpoint_interval", type=int, default=1000,
                      help="Interval for saving dataset checkpoints (in samples)")

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

  parser.add_argument("--clip_range", type=float, default=1.0,
                      help="The gradient clip range.")

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

  ############ System Vars ############

  parser.add_argument("--enable_profile", action="store_true",
                      help="init torch profile")
  
  return parser

# TODO: move to muse.utils
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
  params_to_freeze = args.freeze_params.split(",")

  print_rank_0("Freeze parameters.")
  for name, param in model.named_parameters():
    if re.match(params_to_freeze, name):
      print_rank_0(f"Freeze: {name}")
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
      all_image_tokens = list(
        torch.zeros(world_size, dtype=torch.long).cuda().chunk(world_size) ) \
          if rank == 0 else None
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
  dataset_config["checkpoint_interval"] = args.dataset_checkpoint_interval
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
  print_rank_0("args.model_class:", args.model_class)


  state_dict = None

  converter = FakeConverter()
  if args.model_class in ['Qwen2VLForConditionalGeneration','Qwen2_5_VLForConditionalGeneration']:
      converter = Qwen2VLCheckpointConverter(args.model_dir)
  elif args.model_class == 'Qwen2_5_VLForConditionalGeneration_moonvit':
      converter = Qwen2_5_VL_moonvitCheckpointConverter(args.model_dir)
  elif args.model_class == 'Qwen2_5_VLForConditionalGeneration_siglip':
      converter = Qwen2_5_VL_siglipCheckpointConverter(args.model_dir)
  elif args.model_class == "Qwen2_5_VLForConditionalGeneration_siglip_navit":
      converter = Qwen2_5_VL_siglipCheckpointConverter(args.model_dir)
  elif args.model_class == 'InternVLChatModel':
      converter = InternVLCheckpointConverter(args.model_dir)

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

  # TODO: get_model from model class
  with set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = eval(args.model_class).from_pretrained(args.model_dir)
  
  if args.enable_gradient_checkpointing:
    print_rank_0("Enable gradient checkpointing")

    set_activation_checkpointing(
      model, auto_wrap_policy=model.get_checkpointable_module_classes()
    )

    
  if args.fp32_weight: model = model.float() # upcast fp32 to maintain master weight.
  shard_model(
    model=model,
    cpu_offload=False,
    reshard_after_forward=args.reshard_after_forward,
    dp_mesh=device_mesh,
    fp32_weight=args.fp32_weight,
    prefetch_parameters=args.prefetch_parameters,
    model_class=args.model_class,
    fp32_reduce=args.fp32_reduce
  )
  dist.barrier()

  with Timer("Load state dict"):
    load_from_full_model_state_dict(model=model, full_sd=state_dict, allow_random_init_params=args.allow_random_init_params) # 这里应该全部转成CUDA了, meta -> CUDA

  with torch.device(torch.cuda.current_device()):
    for m in model.modules():
      # RoPE is not covered in state dict
      if hasattr(m, "rope_init"):
        print_rank_0("Initialize RoPE")
        m.rope_init()

  # 暂时注释（caojiangxia）
  # # 确保任何参数都被正确初始化
  # for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
  #   if name != "visual.vision_model.embeddings.position_ids":
  #     assert not tensor.device == torch.device("meta"), \
  #       f"{name} not initialized, device={tensor.device}"

  # model = torch.compile(model)

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
    vision_learning_rate=args.vision_learning_rate,
    weight_decay=args.weight_decay,
    # no_decay_name_list=
    # [
    #   "bias", "norm1", "norm2", "visual.merger.ln_q",
    #   "input_layernorm", "post_attention_layernorm", "model.norm"
    # ] if args.model_class in ['Qwen2VLForConditionalGeneration','Qwen2_5_VLForConditionalGeneration','Qwen2_5_VLForConditionalGeneration_moonvit'] else
    # [
    #   "bias", "norm1", "norm2", "mlp1.0.weight",
    #   "input_layernorm", "post_attention_layernorm", "model.norm"
    # ]
    # ,
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

  tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
  image_token_id = tokenizer.encode('<im_patch>')[0]  if args.model_class == 'InternVLChatModel' else tokenizer.encode('<|image_pad|>')[0]
  video_token_id = tokenizer.encode('<vi_patch>')[0]  if args.model_class == 'InternVLChatModel' else tokenizer.encode('<|video_pad|>')[0]
  fast_video_token_id = -9999 if len(tokenizer.encode('<|fast_video_pad|>'))  > 1 else tokenizer.encode('<|fast_video_pad|>')[0] 
  image_start_id = tokenizer.encode("</image>")[0] if args.model_class == 'InternVLChatModel' else tokenizer.encode('<|vision_start|>')[0]
  frame_id = -9999 if len(tokenizer.encode('<|frame|>'))  > 1 else tokenizer.encode('<|frame|>')[0] 

  if dist.get_rank() == 0:
    with open(os.path.join(args.output_dir,
        f"dataset-{args.commit_id}-{timestamp}.json"), 'w',
        encoding="utf-8") as f:
      f.write(json.dumps(
        dataset_config, ensure_ascii=False, indent=2) + "\n")

  if not use_flops_balance:
    with Timer("Build dataloader"):
      try:  dataloader = get_dataloader_v2(name=dataset, **dataset_config)
      except: 
        import traceback
        print_rank_0(f"get_dataloader_v2 error: {traceback.format_exc()}")
        print_rank_0(f"get_dataloader_v2 retry for get_dataloader")
        traceback.print_exc()
        dataloader = get_dataloader(name=dataset, **dataset_config)
      if args.resume_dataloader and dataloader_state_dict is not None:
        dataloader.load_state_dict(dataloader_state_dict)


  ##############
  torch_profiler = _init_profiler(output_dir=os.path.join(args.output_dir, "torch_profile"))

  loss_fn = CrossEntropyLoss(
    ignore_index=-100, return_token_loss=True, shift_labels=False)

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

    


          
    with contextlib.ExitStack() as ctx:

      if torch_profiler: ctx.enter_context(torch_profiler)

      ticker.tick("enter_context(torch_profiler)")

      try: 
        batch = input_fn()
      except StopIteration: break
      ticker.tick("next_batch")

      micro_step += 1



      if show_cnt > 0 and dist.get_rank() <= 8:
        with Timer("Show data"):
          print("########################### decode ###########################")
          print("batch['input_ids'][0]: ", batch['input_ids'][0])
          input_text = tokenizer.decode(batch['input_ids'][0])
          time.sleep(float(dist.get_rank()) * 0.3)
          print(
              f"Input Text:\n\n{input_text}\n" + "=" * 100 + "\n\n")
          print_input_info(batch, f"rank{dist.get_rank()}")
          show_cnt -= 1
      # continue
      data_source = batch.pop("data_source", None) # dataset source list cur batch

      #print(f"X=0, rank={dist.get_rank()} current_gpu_memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
      to_cuda(batch)
      ticker.tick("to_cuda(batch)")
      #rint(f"X=1, rank={dist.get_rank()} current_gpu_memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
      

      ###### dataset checker ######
      # print("micro_step{}, data_source{}.".format(micro_step, data_source))
      # dist.barrier()
      # continue

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
      fast_pixel_values_videos = batch.get("fast_pixel_values_videos", None)
      fast_video_grid_thw = batch.get("fast_video_grid_thw", None)
      ###################
      # 打印 token 数量
      if not use_flops_balance or True:
        token_count = input_ids.numel() / args.sequence_parallel_size  # 计算 token 数量
        print_rank_0(f"Iteration {micro_step}: Token count = {token_count}")
        num_tokens = token_count

        num_samples = (sample_idx.max() + 1).sum()  / args.sequence_parallel_size

        image_tokens_ids = input_ids == image_token_id
        num_image_tokens = image_tokens_ids.sum().item() / args.sequence_parallel_size
        num_image_tokens2 = num_image_tokens

        video_tokens_ids = (input_ids == video_token_id) | (input_ids == fast_video_token_id)
        num_video_tokens = video_tokens_ids.sum().item() / args.sequence_parallel_size
        num_video_tokens2 = num_video_tokens
        num_images = round(num_image_tokens / 256) if args.model_class == "InternVLChatModel" else -1 # (((input_ids == image_start_id) | (input_ids == frame_id)).sum().item() / args.sequence_parallel_size)
        if num_images == -1: 
          num_images = sum([v.shape[0] for k, v in batch.items() if 'grid_thw' in k], 0)

        # num_image_tokens, num_tokens, num_samples, num_images
        mfu_stats.set(
          max(num_image_tokens, 1) + max(num_video_tokens, 1), 
          max(num_tokens, 1), 
          max(1, num_samples.detach().item()), 
          max(num_images, 1)
        )

        # num_tokens - (sample_idx == -1).sum()
        num_valid_tokens = torch.nonzero(loss_mask[0] == 1)[-1].item() / args.sequence_parallel_size + 1 # 我们可以采取补全的方式packing最后一个样本，所以需要按照最后一个loss是位置计算有效样本数量 
        token_metrics = torch.tensor(
          [num_tokens, num_samples, num_valid_tokens, num_image_tokens, num_video_tokens]).cuda(non_blocking=True)

        ticker.tick("token_metrics_init")
        
        dist.all_reduce(
          token_metrics, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
        ticker.tick("token_metrics_reduce")

        # print(f"rank={torch.distributed.get_rank()}, pid={os.getpid()}, token_metrics.detach")
        num_tokens, num_samples, num_valid_tokens, num_image_tokens, num_video_tokens = token_metrics.detach().cpu().numpy() * args.sequence_parallel_size
        ticker.tick("token_metrics.detach().cpu().numpy()")
      else:
        num_image_tokens2 = (input_ids == 151667).sum().item()
        num_tokens = batch.get("num_tokens", 0)
        num_samples = batch.get("num_samples", 0)
        num_valid_tokens = batch.get("num_valid_tokens", num_tokens)
        num_image_tokens = batch.get("num_image_tokens", 0)
        print_rank_0(f"Iteration {micro_step}: Token count = {num_tokens}")

      total_num_samples += num_samples
      total_num_tokens += num_tokens
      total_num_valid_tokens += num_valid_tokens
      total_num_image_tokens += num_image_tokens
      total_num_video_tokens += num_video_tokens

      acc_num_samples += num_samples
      acc_num_tokens += num_tokens
      acc_valid_num_tokens += num_valid_tokens
      acc_num_image_tokens += num_image_tokens
      acc_num_video_tokens += num_video_tokens
      ticker.tick("acc_valid_num_tokens+=num_valid_tokens")

      input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)
      labels = input_ids * loss_mask + loss_fn.ignore_index * (1 - loss_mask) # loss_mask需要保证图片的token不会被预测
      ticker.tick("labels=...")
      

      print("########################### decode ###########################")
      print("batch['input_ids'][0]: ", batch['input_ids'][0])
      # batch['input_ids'][0]:  tensor([151644,   8948,    198,  ..., 151643, 151643, 151643], device='cuda:3')
      input_text = tokenizer.decode(batch['input_ids'][0])
      print("input_text:", input_text)
      # <|im_start|>system .......

      print("######################### Check params requires_grad Begin before model(): #########################")
      for name, param in model.named_parameters():
          if param.requires_grad:
              print(f"{name}: requires_grad=True, shape={param.shape}")
          # else:
          #     print(f"{name}: requires_grad=False, shape={param.shape}")
      print("######################### Check params requires_grad End before model(): #########################")
        
      with Timer("Fwd"):
        if args.model_class == "InternVLChatModel":
            output = model(
              input_ids = input_ids, attention_mask=attention_mask,
              pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
              image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
              image_flags = image_flags,
              cu_seqlens=cu_seqlens
            )
        else:
            image_position_ids = batch.get("image_position_ids", None)
            image_grid_hws = batch.get("image_grid_hws", None)
            image_sample_indices = batch.get("image_sample_indices", None)
            image_cu_seqlens = batch.get("image_cu_seqlens", None)
            second_per_grid_ts = batch.get("second_per_grid_ts", None)
            output = model(
              input_ids = input_ids, attention_mask=attention_mask,
              pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
              image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
              cu_seqlens=cu_seqlens, image_position_ids=image_position_ids,
              image_grid_hws=image_grid_hws, image_sample_indices=image_sample_indices,
              image_cu_seqlens=image_cu_seqlens,
              max_seqlen_q=batch.get("max_seqlen_q", None),
              image_max_seqlen_q=batch.get("image_max_seqlen_q", None),
              image_max_seqlen_k=batch.get("image_max_seqlen_k", None),
              fast_pixel_values_videos=fast_pixel_values_videos,
              fast_video_grid_thw=fast_video_grid_thw, 
              position_ids=position_ids
            )
        ticker.tick("model.forward")

        # (b, N/P, V)
        # logits = output.logits
        print("######################### Check params requires_grad Begin after model(): #########################")
        for name, param in model.named_parameters():
          if param.requires_grad:
              print(f"{name}: requires_grad=True, shape={param.shape}")
          # else:
          #     print(f"{name}: requires_grad=False, shape={param.shape}")
        print("######################### Check params requires_grad End after model(): #########################")
        

        # # 提前shift logits & labels
        pad = torch.full((labels.shape[0], 1), loss_fn.ignore_index,
            dtype=labels.dtype).to(device=labels.device, non_blocking=True)
        labels = torch.cat([labels[:, 1:], pad], dim=-1) # shift
        local_labels = get_local_sequence(labels, seq_idx=1)

        # loss, per_token_loss = loss_fn(logits=logits, labels=local_labels)

        # TODO: codebook_loss && reconstruction_loss
        codebook_loss = output.loss
        print('***codebook_loss***:', codebook_loss)
        loss_reconstruction = output.loss_reconstruction
        print('***loss_reconstruction***:', loss_reconstruction)
        # NOTE: using this code for common loss calulate
        # loss = codebook_loss + loss_reconstruction
        loss = codebook_loss # NOTE: only using codebook_loss
        print('***loss_all***:', loss)

        ############ NOTE: add global batchsize ############ 
        token_frequency = output.token_frequency

        global_token_frequency = token_frequency.clone()

        dist.all_reduce(global_token_frequency, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
        if dist.get_rank() == 0:
          nonzero_count = (token_frequency > 0).sum().item()
          topk_vals, topk_idx = torch.topk(token_frequency, 10)
          codebook_size = len(token_frequency)       
          print("global used_codes:", nonzero_count, "topk_counts:", topk_vals.tolist(), "topk_idx:", topk_idx.tolist())
          token_util = nonzero_count / codebook_size
          print("global token_util:", token_util)

        # print("########################### decode ###########################")
        # print("topk_idx id: ", torch.tensor(topk_idx).to(codebook_loss))
        # tensor([5920.,  964., 1464., 2304., 2416.,  198.,  976., 2128., 2640., 1808.],device='cuda:7', dtype=torch.bfloat16)
        # topk_idx: [5364, 2303, 5448, 5924, 1777, 1640, 1924, 2102, 811, 1173]
        # topk_idx_token = tokenizer.decode(torch.tensor(topk_idx).to(codebook_loss.device))
        # print("topk_idx token:", topk_idx_token)
        # <|im_start|>system .......

        

        
        # 计算需要检查的位置：所有 loss_mask 为1的位置的前一个位置
        # 因为我们需要检查 input_ids[i+1] == labels[i]
        check_mask = torch.zeros_like(loss_mask)
        check_mask[:, :-1] = loss_mask[:, 1:]  # 将 loss_mask 右移一位

        # 提取需要检查的标签位置
        masked_labels = labels[check_mask.bool()]

        # 提取对应的 input_ids（右移后应匹配）
        shifted_input_ids = input_ids[:, 1:][loss_mask[:, 1:].bool()]

        # 断言：在 loss_mask 为1的位置之前，input_ids[i+1] == labels[i]
        assert torch.equal(masked_labels, shifted_input_ids), \
            f"标签与输入不匹配：\n" \
            f"标签位置: {masked_labels}\n" \
            f"输入位置: {shifted_input_ids}\n" \
            f"差异位置: {torch.nonzero(masked_labels != shifted_input_ids, as_tuple=True)}"
        ################# label check #################
        ticker.tick("loss_fn")

      # print(f"X=111, rank={dist.get_rank()} current_gpu_memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
      with Timer("bwd"):
        print("loss.backward() begin----------")
        loss.backward()
        
        # grad_logger(model)

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
      '''
      if args.monitor_datasource_loss:
        # WARN: assume batch_size = 1
        local_sample_idx = get_local_sequence(sample_idx).squeeze()

        unique_sample_idx = local_sample_idx.unique()
        for s_idx in unique_sample_idx:
          if s_idx < 0:
            continue
          
          local_mask = get_local_sequence(loss_mask)[0]
          mask = (local_sample_idx == s_idx) * local_mask


          per_token_loss2 = per_token_loss[:-1]
          mask = mask[1:]
          sum_loss = per_token_loss2[mask>0].sum()
          key = data_source[int(s_idx.item())]
          batch_data_source_loss[key] += sum_loss.item()
          batch_data_source_tokens[key] += mask.sum().item()

        ticker.tick("monitor_datasource_loss")
      
      if args.monitor_datasource_cnt:
        for data_source_name in data_source:
          local_acc_data_source_samples[data_source_name] += 1
        ticker.tick("monitor_datasource_cnt")
      '''
      #########################################
      avg_loss = loss.detach()
      codebook_loss = codebook_loss.detach()
      loss_reconstruction = loss_reconstruction.detach()

      # total_loss = per_token_loss2.sum()
      # dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
      # total_mask = local_mask.sum()
      # dist.all_reduce(total_mask, op=dist.ReduceOp.SUM)
      avg_loss = avg_loss.item() / dist.get_world_size()
      codebook_loss = codebook_loss.item() / dist.get_world_size()
      loss_reconstruction = loss_reconstruction.item() / dist.get_world_size()
      # avg_loss = total_loss / total_mask.sum()
      # acc_avg_loss += avg_loss

      ticker.tick("reduce_acc_avg_loss")
      log_acc_step = args.logging_per_step * args.gradient_accumulation_steps

      if global_step % args.logging_per_step == 0 and \
              (micro_step + 1) % args.gradient_accumulation_steps == 0:

        if args.monitor_image_tokens: 
          token_stasts.collect_image_token_stats(num_image_tokens2)
          vid_token_stasts.collect_image_token_stats(num_video_tokens2)
          colleced_token_stasts = token_stasts.stats()  
          vid_colleced_token_stasts = vid_token_stasts.stats()       
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
          if len(model_lrs) > 2:
            vision_learning_rate = lr_scheduler.get_lr()[2]
          else:
            vision_learning_rate = lr_scheduler.get_lr()[1]
          end_time = time.time()
          sec_per_step = (end_time - start_time) / args.logging_per_step # / args.gradient_accumulation_steps
          tokens_per_sec_per_gpu = \
            acc_num_tokens  / (end_time - start_time) / dist.get_world_size()
          samples_per_sec_per_gpu = \
            acc_num_samples  / (end_time - start_time) / dist.get_world_size()
          samples_per_step_per_gpu = \
            acc_num_samples  / dist.get_world_size()
          valid_tokens_per_sec_per_gpu = \
            acc_valid_num_tokens / (end_time - start_time) / dist.get_world_size()
          image_tokens_per_sec_per_gpu = \
            acc_num_image_tokens / (end_time - start_time) / dist.get_world_size()
          video_tokens_per_sec_per_gpu = \
            acc_num_video_tokens / (end_time - start_time) / dist.get_world_size()

          samples_per_step_per_gpu_v2 = total_num_samples / dist.get_world_size() / global_step
          samples_per_sec_per_gpu_v2 = total_num_samples / dist.get_world_size() / (end_time - start_time0)

          image_tokens_per_step_per_gpu_v2 = total_num_image_tokens / dist.get_world_size() / global_step
          image_tokens_per_sec_per_gpu_v2 = total_num_image_tokens / dist.get_world_size() / (end_time - start_time0)

          video_tokens_per_step_per_gpu_v2 = total_num_video_tokens / dist.get_world_size() / global_step
          video_tokens_per_sec_per_gpu_v2 = total_num_video_tokens / dist.get_world_size() / (end_time - start_time0)


          tokens_per_step_per_gpu_v2 = total_num_tokens / dist.get_world_size() / global_step
          tokens_per_sec_per_gpu_v2 = total_num_tokens / dist.get_world_size() / (end_time - start_time0)


          # avg_loss = acc_avg_loss / args.gradient_accumulation_steps / args.logging_per_step
          log_dict = {
            # max_image_tokens, min_image_tokens, mean_image_tokens, std_image_tokens
            "training/loss": avg_loss,
            "loss_all": avg_loss,
            "loss_codebook": codebook_loss,
            "loss_reconstruction": loss_reconstruction,
            "token_util": token_util,
            "nonzero_count": nonzero_count,
            f"training/grad_norm": grad_norm,
            "training/learning_rate": learning_rate,
            "training/vision_learning_rate": vision_learning_rate,
            "perf/sec_per_step": sec_per_step,
            "perf/tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
            "perf/samples_per_sec_per_gpu": samples_per_sec_per_gpu,
            "perf/total_num_tokens": total_num_tokens,
            "perf/total_num_samples": total_num_samples,
            "perf/num_sample_per_gpu": total_num_samples / dist.get_world_size(),
            "perf/samples_per_step_per_gpu": samples_per_step_per_gpu,
            "perf/num_sample_per_sec_per_gpu": total_num_samples / (end_time - start_time) / dist.get_world_size(),
            "perf/valid_total_num_tokens": total_num_valid_tokens ,
            "perf/valid_tokens_per_sec_per_gpu": valid_tokens_per_sec_per_gpu ,
            "perf/image_tokens_per_sec_per_gpu": image_tokens_per_sec_per_gpu,
            "perf/video_tokens_per_sec_per_gpu": video_tokens_per_sec_per_gpu,

            "perf/image_token_ratio_by_valid": image_tokens_per_sec_per_gpu / valid_tokens_per_sec_per_gpu,

            "perf/video_token_ratio_by_valid": video_tokens_per_sec_per_gpu / valid_tokens_per_sec_per_gpu,

            "perf/valid_token_ratio": total_num_valid_tokens / total_num_tokens,
            "perf/image_token_per_sample_per_gpu":total_num_image_tokens / total_num_samples,
            "perf/video_token_per_sample_per_gpu":total_num_video_tokens / total_num_samples,
            **mfu_stats.mfu(end_time - start_time, global_step - global_step0),
            "perf/samples_per_step_per_gpu_v2": samples_per_step_per_gpu_v2,
            "perf/samples_per_sec_per_gpu_v2": samples_per_sec_per_gpu_v2,
            "perf/image_tokens_per_step_per_gpu_v2": image_tokens_per_step_per_gpu_v2,
            "perf/image_tokens_per_sec_per_gpu_v2": image_tokens_per_sec_per_gpu_v2,
            "perf/video_tokens_per_step_per_gpu_v2": video_tokens_per_step_per_gpu_v2,
            "perf/video_tokens_per_sec_per_gpu_v2": video_tokens_per_sec_per_gpu_v2,
            "perf/tokens_per_step_per_gpu_v2": tokens_per_step_per_gpu_v2,
            "perf/tokens_per_sec_per_gpu_v2": tokens_per_sec_per_gpu_v2,
            "perf/epoch_idx": epoch_idx,
          }
          start_time = end_time
          if args.monitor_image_tokens: 
            log_dict.update(colleced_token_stasts)
            log_dict.update(vid_colleced_token_stasts)
          ticker.tick(f"log_dict*{log_acc_step}")

          ticker_stats = {}
          for t in [ticker, iter_ticker]:
              ticker_stats.update(t.stat())
          metrics_info = (global_step, log_dict, ticker_stats, batch_data_source_loss, batch_data_source_tokens, total_data_source_samples)

          tb_metrics_q.put(metrics_info)

          ticker.tick(f"tb_metrics_q.put")

          # if args.monitor_datasource_loss and tb_writer:
          #   for key, loss_sum in batch_data_source_loss.items():
          #     tb_writer.add_scalar(
          #           f"data_source_loss/{key}",
          #           loss_sum / (valid_data_source_tokens[key] + 1e-6) ,
          #           global_step=global_step,
          #           new_style=True)

          # if args.monitor_datasource_cnt and tb_writer:
          #   for key, samples in total_data_source_samples.items():
          #     tb_writer.add_scalar(
          #         f"data_source_sample_ratio/{key}",
          #         1.0 * samples / total_num_samples,
          #         global_step=global_step,
          #         new_style=True)

          #   for key, num_tokens in total_data_source_tokens.items():
          #     tb_writer.add_scalar(
          #         f"data_source_token_ratio/{key}",
          #         1.0 * num_tokens / total_num_valid_tokens,
          #         global_step=global_step,
          #         new_style=True)
              
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

        acc_avg_loss = 0.0
        acc_num_samples = 0
        acc_num_tokens = 0
        acc_valid_num_tokens = 0
        acc_num_image_tokens = 0

        batch_data_source_loss = collections.defaultdict(float)
        batch_data_source_tokens = collections.defaultdict(int)
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
                          "total_num_valid_tokens": total_num_valid_tokens,
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

      # print_rank_0(f"ticker_info: { format ticker.stat()}")
      iter_ticker.tick("iter_ticker")
      if torch_profiler: torch_profiler.step()


  # Save final dataset checkpoint if enabled
  if args.enable_dataset_checkpointing and hasattr(dataloader, 'dataset'):
    try:
      worker_id, _ = get_worker_info()
      dataloader.dataset.save_checkpoint(worker_id, global_step)
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





