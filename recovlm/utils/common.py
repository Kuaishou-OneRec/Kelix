# TODO: clean utils
from rich import print
import torch
import random
import numpy as np
from transformers import set_seed as set_transformers_seed
import torch.distributed as dist
import pickle

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

def dist_reduce_dict(local_dict, dict_reduce_func=None):
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  serialized_dict = pickle.dumps(local_dict)
  tensor = torch.ByteTensor(list(serialized_dict)).cuda()
  tensor_size = torch.tensor([tensor.size(0)], dtype=torch.int64).cuda()
  all_sizes = [torch.tensor([0], dtype=torch.int64).cuda() for _ in range(world_size)]
  dist.all_gather(all_sizes, tensor_size)

  max_size = max([t.item() for t in all_sizes])
  buffer_tensor = torch.zeros(max_size, dtype=torch.uint8).cuda()
  buffer_tensor[:tensor.size(0)] = tensor
  gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8).cuda() for _ in range(world_size)]
  dist.all_gather(gathered_tensors, buffer_tensor)

  gathered_dicts = [pickle.loads(bytes(t.cpu().tolist())) for t in gathered_tensors]
  if dict_reduce_func is not None:
    return dict_reduce_func(gathered_dicts)
  else:
    reduce_dict = {}
    for tmp_dict in gathered_dicts:
      for k, v in tmp_dict.items():
        if k not in reduce_dict:
          reduce_dict[k] = v
        else:
          reduce_dict[k] += v
    return reduce_dict


