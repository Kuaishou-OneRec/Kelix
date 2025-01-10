# TODO: clean utils
from rich import print
import torch
import random
import numpy as np
from transformers import set_seed as set_transformers_seed
import torch.distributed as dist

def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
  optimizer_grouped_parameters = [
    {
      "params": [
        p for n, p in model.named_parameters()
        if (not any(nd in n
                    for nd in no_decay_name_list) and p.requires_grad)
      ],
      "weight_decay":
      weight_decay,
    },
    {
      "params": [
        p for n, p in model.named_parameters()
        if (any(nd in n
                for nd in no_decay_name_list) and p.requires_grad)
      ],
      "weight_decay":
      0.0,
    },
  ]
  return optimizer_grouped_parameters

def print_rank_n(*msg, rank=0):
  if dist.get_rank() == rank:
    print(*msg)

def print_rank_0(*msg):
  print_rank_n(*msg, rank=0)

def to_device(batch, device):
  for key in list(batch.keys()):
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
