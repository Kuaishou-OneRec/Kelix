from typing import Any, Tuple, List, Iterable
import os
import time
import torch
import torch.distributed as dist

from muse.utils.common import print_rank_0, Timer
import datetime

process_group_timeout = datetime.timedelta(minutes=60 * 24)

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GROUP_GLOO = None
_DATA_PARALLEL_GROUP = None


def initialize_model_parallel(sequence_parallel_size: int):
    world_size = dist.get_world_size()
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    num_data_parallel_groups: int = sequence_parallel_size
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GROUP
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        print_rank_0(f"Sequence Parallel Group: {i}, Ranks: {ranks}")
        group = torch.distributed.new_group(ranks)
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        rank = dist.get_rank()
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_GROUP_GLOO = group_gloo
    for i in range(num_data_parallel_groups):
        ranks = [r for r in range(world_size) if r % sequence_parallel_size == i]
        print_rank_0(f"Data Parallel Group: {i}, Ranks: {ranks}")
        group = torch.distributed.new_group(ranks)
        rank = dist.get_rank()
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

def worker_init_fn(worker_id):
    if os.getenv("WORKER_MASTER_PORT", None) is None:
        os.environ["WORKER_MASTER_PORT"] = \
            str(int(os.environ["MASTER_PORT"]) + worker_id + 1)
    os.environ["MASTER_PORT"] = os.environ["WORKER_MASTER_PORT"]
    dist.init_process_group(
        "gloo", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
        timeout=process_group_timeout
    )

def get_sequence_parallel_group(backend="nccl"):
    """Get the sequence parallel group the caller rank belongs to."""
    if backend == "nccl":
        return _SEQUENCE_PARALLEL_GROUP
    elif backend == "gloo":
        return _SEQUENCE_PARALLEL_GROUP_GLOO
    else:
        raise NotImplementedError(f"Unsupport sequence parallel backend: {backend}")

def get_sequence_parallel_world_size():
    """Get the sequence parallel world size."""
    return dist.get_world_size(group=get_sequence_parallel_group())

def get_sequence_parallel_rank():
    """Get the sequence parallel rank."""
    return dist.get_rank(group=get_sequence_parallel_group())

def get_local_sequence_boundary(seq_len: int) -> Tuple[int, int]:
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    local_seqlen = seq_len // sp_size
    start, end = sp_rank * local_seqlen, (sp_rank + 1) * local_seqlen
    return start, end

def get_local_sequence(sequence: torch.Tensor, seq_idx: int = 1) -> torch.Tensor:
    if get_sequence_parallel_world_size() > 1:
        seq_len = sequence.shape[seq_idx]
        start, end = get_local_sequence_boundary(seq_len)
        # Create a slice object for the specified dimension
        slices = [slice(None)] * sequence.dim()
        slices[seq_idx] = slice(start, end)
        # Use the slice object to index the tensor
        local_sequence = sequence[tuple(slices)]
        return local_sequence
    return sequence

def get_data_parallel_group() -> dist.ProcessGroup:
    return _DATA_PARALLEL_GROUP

def get_data_parallel_rank() -> int:
    if dist.is_initialized() and get_data_parallel_group() is not None:
        return dist.get_rank(group=get_data_parallel_group())
    else:
        return 0

def get_data_parallel_world_size() -> int:
    if dist.is_initialized() and get_data_parallel_group() is not None:
        return dist.get_world_size(group=get_data_parallel_group())
    else:
        return 1

def all_to_all_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group: dist.ProcessGroup = None,
    use_sync: bool = False
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.Tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.Tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.Tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs)
        # output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.Tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        # output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")

class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
            ),
            None,
            None,
            None,
        )

def all_gather(
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
    gather_idx: int = 0,
    use_sync: bool = False) -> torch.Tensor:
    """
    all-gather for Sequence

    Args:
        inputs (torch.Tensor): a tensor to gather, with shape (bs, seqlen/P, h)
        group : torch process group
        use_sync (bool): whether to synchronize after all-gather

    Returns:
        torch.Tensor: gathered tensor (bs, seqlen, h)
    """

    seq_world_size = dist.get_world_size(group)

    if seq_world_size > 1:
        output = [torch.empty_like(input_tensor) for _ in range(seq_world_size)]
        dist.all_gather(
            tensor_list=output, tensor=input_tensor.contiguous(), group=group)
        if use_sync:
            torch.cuda.synchronize()

        return torch.cat(output, dim=gather_idx)
    else:
        return input_tensor

def shard(input_tensor, group, shard_idx):
    world_size = dist.get_world_size(group)
    if world_size > 1:
        rank = dist.get_rank(group)
        local_tensor = torch.chunk(
            input_tensor, world_size, dim=shard_idx)[rank]
        return local_tensor
    return input_tensor

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                inputs: torch.Tensor,
                group: dist.ProcessGroup,
                gather_idx: int = 0,
                use_sync: bool = False) -> torch.Tensor:
        ctx.group = group
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_gather(
            inputs, group=group, gather_idx=gather_idx,
            use_sync=use_sync)

    @staticmethod
    def backward(ctx: Any,
                 *grad_output: torch.Tensor
        ) -> Tuple[None, torch.Tensor, None, None]:
        return (
            shard(
                *grad_output,
                ctx.group,
                ctx.gather_idx
            ),
            None,
            None,
            None,
        )

def gather_batches(buffer: List[Any], group: dist.ProcessGroup):
    """
    Gather batches from all ranks in the group.
    """
    world_size = dist.get_world_size(group)
    if world_size > 1:
      with Timer("Gather batches"):
        gathered_batches = [None for _ in range(world_size)]
        start = time.time()
        dist.all_gather_object(
            object_list=gathered_batches, obj=buffer,
            group=group
        )
      gathered_batches = sum(gathered_batches, [])
    else:
      gathered_batches = buffer
    print_rank_0(f"Num batches: {len(gathered_batches)}")
    return gathered_batches

def gather_by_group(dataloader: Iterable[Any],
                    group: dist.ProcessGroup, buffer_size: int = 1) -> List[Any]:
    """
    Gather batches from all ranks in the group.
    """
    buffer = []
    for batch in dataloader:
        buffer.append(batch)
        if len(buffer) >= buffer_size:
            yield from gather_batches(buffer, group)
            buffer = []
    if len(buffer) > 0:
        yield from gather_batches(buffer, group)
