import torch
import torch.distributed as dist

_N_GPUS_PER_NDOE = 8

def get_sequence_parallel_group():
    rank = dist.get_rank()
    
    node_id = rank // _N_GPUS_PER_NDOE
    
    ranks = list(
        range(node_id * _N_GPUS_PER_NDOE, (node_id + 1) * _N_GPUS_PER_NDOE))

    group = dist.new_group(ranks, backend="nccl")

    return group
