import torch.distributed as dist

def get_world_size():
    return dist.get_world_size()

def get_rank():
    return dist.get_rank()

def is_rank_0():
    return get_rank() == 0