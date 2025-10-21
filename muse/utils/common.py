# TODO: clean utils
from typing import Tuple
from rich import print
import time
import torch
import random
import numpy as np
from pathlib import Path
from transformers import set_seed as set_transformers_seed
import torch.distributed as dist
import pickle
import traceback
import subprocess
import os


def print_rank_n(*msg, rank=0):
  if dist.get_rank() == rank:
    print(*msg)

def print_rank_0(*msg):
  print_rank_n(*msg, rank=0)

def to_device(batch, device, non_blocking=True):
  for key in list(batch.keys()):
    if isinstance(batch[key], torch.Tensor):
      batch[key] = batch[key].to(device=device, non_blocking=non_blocking)
  return batch

def to_cuda(batch, non_blocking=True):
  to_device(batch, device=torch.cuda.current_device(), non_blocking=non_blocking)

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

def get_root_dir():
  current_file_path = Path(__file__).resolve()
  # recovlm/recovlm/utils/common.py
  root_dir = current_file_path.parent.parent.parent
  return root_dir

def load_env():
  env_path = get_root_dir() / ".env"
  env = {}
  with open(env_path, encoding="utf-8") as f:
    for line in f:
      key, value = line.strip().split("=")
      env[key] = str(value)
  return env

