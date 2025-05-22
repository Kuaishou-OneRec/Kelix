
import os
import torch
import torch.distributed as dist

def init_processes(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '1236'
        dist.init_process_group(backend, rank=rank, world_size=size)
