# TODO: clean utils
from rich import print
import time
import torch
import random
import numpy as np
from transformers import set_seed as set_transformers_seed
import torch.distributed as dist
import pickle
import traceback
import subprocess
import os
from infra.perflog import create_perf_context

def print_rank_n(*msg, rank=0):
  if dist.get_rank() == rank:
    print(*msg)

def print_rank_0(*msg):
  print_rank_n(*msg, rank=0)

def get_optimizer_grouped_parameters(model,
                                     learning_rate: float,
                                     vision_learning_rate: float,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ],
                                     vision_learning_rate_layer_dacay=1.0):
  optimizer_grouped_parameters = []

  llm_wd_params_group = []
  llm_nowd_params_group = []
  vit_wd_params_group = []
  vit_nowd_params_group = []

  for n, p in model.named_parameters():
    if any(nd in n for nd in no_decay_name_list) and p.requires_grad:
      # no weight decay params
      if n.startswith("visual"):
        vit_nowd_params_group.append((n, p))
      else:
        llm_nowd_params_group.append((n, p))
    else:
      # weight decay params
      if n.startswith("visual"):
        vit_wd_params_group.append((n, p))
      else:
        llm_wd_params_group.append((n, p))
  
  # for LLM
  optimizer_grouped_parameters.append({
    "params": [p for n, p in llm_wd_params_group],
    "weight_decay": 0.0,
    "lr": learning_rate,
  })
  optimizer_grouped_parameters.append({
    "params": [p for n, p in llm_nowd_params_group],
    "weight_decay": weight_decay,
    "lr": learning_rate,
  })

  # for vit
  if vision_learning_rate_layer_dacay == 1.0:
    optimizer_grouped_parameters.append({
      "params": [p for n, p in vit_wd_params_group],
      "weight_decay": weight_decay,
      "lr": vision_learning_rate,
    })
    optimizer_grouped_parameters.append({
      "params": [p for n, p in vit_nowd_params_group],
      "weight_decay": 0.0,
      "lr": vision_learning_rate,
    })
  else:
    # decay by layer
    vit_opt_groups = []

    # get all vit layers
    layer_ids = set()
    for n, _ in vit_wd_params_group + vit_nowd_params_group:
      if n.startswith("visual.blocks."):
        layer_ids.add(int(n.split(".")[2]))
    layer_ids = list(layer_ids)
    layer_ids.sort()
    layer_ids.reverse()
    # cal layer lr
    layers_params = []
    for idx, lid in enumerate(layer_ids):
      layer_lr = vision_learning_rate * (vision_learning_rate_layer_dacay ** idx)
      print_rank_0(f"visual.blocks.{lid}. {layer_lr=}")

      layers_params.extend([n for n, p in vit_wd_params_group if n.startswith(f"visual.blocks.{lid}.")])
      layers_params.extend([n for n, p in vit_nowd_params_group if n.startswith(f"visual.blocks.{lid}.")])
      
      vit_opt_groups.append({
        "params": [p for n, p in vit_wd_params_group if n.startswith(f"visual.blocks.{lid}.")],
        "weight_decay": weight_decay,
        "lr": layer_lr,
      })
      vit_opt_groups.append({
        "params": [p for n, p in vit_nowd_params_group if n.startswith(f"visual.blocks.{lid}.")],
        "weight_decay": 0.0,
        "lr": layer_lr,
      })

    optimizer_grouped_parameters.append({
        "params": [p for n, p in vit_wd_params_group if n not in layers_params],
        "weight_decay": weight_decay,
        "lr": vision_learning_rate,
    })
    optimizer_grouped_parameters.append({
        "params": [p for n, p in vit_nowd_params_group if n not in layers_params],
        "weight_decay": 0.0,
        "lr": vision_learning_rate,
    })
    optimizer_grouped_parameters.extend(vit_opt_groups)
    
  for n, _ in llm_wd_params_group:
    print_rank_0(f"llm_weight_decay_params: {n}")
  for n, _ in llm_nowd_params_group:
    print_rank_0(f"llm_no_weight_decay_params: {n}")
  for n, _ in vit_wd_params_group:
    print_rank_0(f"vit_weight_decay_params: {n}")
  for n, _ in vit_nowd_params_group:
    print_rank_0(f"vit_no_weight_decay_params: {n}")

  return optimizer_grouped_parameters

def to_device(batch, device):
  for key in list(batch.keys()):
    if isinstance(batch[key], torch.Tensor):
      batch[key] = batch[key].to(device=device)

def to_cuda(batch):
  to_device(batch, device=torch.cuda.current_device())

def set_random_seed(seed):
  if seed is not None:
    set_transformers_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def increment_version(version):
  major, minor, patch = map(int, version.split('.'))
  patch += 1
  return f"{major}.{minor}.{patch}"

def dist_reduce_dict(local_dict, group=None):
  gather_list = [None for _ in range(dist.get_world_size(group=group))]

  dist.all_gather_object(
    object_list=gather_list, obj=local_dict, group=group)

  def reduce_dicts(dicts):
    def _reduce(d1, d2):
      for key, value in d2.items():
        if isinstance(value, dict):
          if key not in d1:
            d1[key] = {}
          _reduce(d1[key], value)
        else:
          if key in d1:
            d1[key] += value
          else:
            d1[key] = value
      return d1

    result = {}
    for d in dicts:
        result = _reduce(result, d)
    return result

  return reduce_dicts(gather_list)

class Timer:
  def __init__(self, desc: str = ""):
    self.desc = desc

  def __enter__(self):
    print_rank_0(f"Start... {self.desc}")
    self.start = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.end = time.time()
    self.elapsed = self.end - self.start
    print_rank_0(f"End... {self.desc} elapsed: {self.elapsed:.3f} ")

def shell_hdfs_ls(source_dir):
  try:
    command = f"hdfs dfs -ls {source_dir}"
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    files = []
    for line in result.stdout.splitlines():
      parts = line.split()
      if len(parts) > 0 and parts[-1].startswith('viewfs://'):
        files.append(parts[-1])
    return files

  except subprocess.CalledProcessError as e:
    print(f"Error occurred: {traceback.format_exc()}")
    return []

def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
  """Return node and worker info for PyTorch and some distributed environments.

  Args:
      group (optional): The process group for distributed environments. Defaults to None.

  Returns:
      tuple: A tuple containing (rank, world_size, worker, num_workers).
  """
  rank = 0
  world_size = 1
  worker = 0
  num_workers = 1
  if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
  else:
    try:
      import torch.distributed

      if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = group or torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
    except ModuleNotFoundError:
      pass
  if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
    worker = int(os.environ["WORKER"])
    num_workers = int(os.environ["NUM_WORKERS"])
  else:
    try:
      import torch.utils.data

      worker_info = torch.utils.data.get_worker_info()
      if worker_info is not None:
        worker = worker_info.id
        num_workers = worker_info.num_workers
    except ModuleNotFoundError:
      pass

  return rank, world_size, worker, num_workers

def get_task_tag():
    kml_id = os.environ.get("KML_ID", "")
    kml_task_id = os.environ.get("KML_TASK_ID", "")
    if kml_id.strip() == "" or kml_task_id.strip() == "":
        raise ValueError(f"env not set!!!! KML_ID={kml_id} KML_TASK={kml_task_id}")
    task_tag = f"kml-task-{kml_task_id}-record-{kml_id}"
    return task_tag

def heart_beat(num_tokens):
    current_file = os.path.abspath(__file__)
    log_ctx = create_perf_context('reco_vllm.pretrain', get_task_tag(), biz_def='infra', extra1=current_file)
    log_ctx.logstash_only(count=num_tokens)
    log_ctx.persist_data()