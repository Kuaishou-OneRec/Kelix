# TODO: clean utils
import torch
import torch.distributed as dist

def print_rank_n(*msg, rank=0):
  if dist.get_rank() == rank:
    print(*msg)

def print_rank_0(*msg):
  print_rank_n(*msg, rank=0)

def move_to_cuda(batch):
  for key in list(batch.keys()):
    batch[key] = batch[key].cuda(
        torch.cuda.current_device())