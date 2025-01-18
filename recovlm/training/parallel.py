from typing import Any, Tuple
import os
import time
import torch
import torch.distributed as dist

from flash_attn import flash_attn_varlen_func

from recovlm.utils.common import print_rank_0, Timer

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GROUP_GLOO = None
_DATA_PARALLEL_GROUP = None

def initialize_model_parallel(sequence_parallel_size):
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
        "gloo", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"])
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

def get_local_sequence_boundary(seq_len):
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    local_seqlen = seq_len // sp_size
    start, end = sp_rank * local_seqlen, (sp_rank + 1) * local_seqlen
    return start, end

def get_local_sequence(sequence: torch.Tensor, seq_idx: int = 1):
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

def get_data_parallel_group():
    return _DATA_PARALLEL_GROUP

def get_data_parallel_rank():
    return dist.get_rank(group=get_data_parallel_group())

def get_data_parallel_world_size():
    return dist.get_world_size(group=get_data_parallel_group())

def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
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
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
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
    input_tensor: torch.tensor,
    group: dist.ProcessGroup = None,
    gather_idx: int = 0,
    use_sync: bool = False) -> torch.tensor:
    """
    all-gather for Sequence

    Args:
        inputs (torch.tensor): a tensor to gather, with shape (bs, seqlen/P, h)
        group : torch process group
        use_sync (bool): whether to synchronize after all-gather

    Returns:
        torch.tensor: gathered tensor (bs, seqlen, h)
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

def scatter(
    input_tensor: torch.tensor,
    group=None,
    scatter_idx: int = 0,
    use_sync: bool = False) -> torch.tensor:
    """
    scatter for Sequence

    Args:
        input_tensor (torch.tensor): a tensor to gather, default with shape (bs, seqlen, h)
        group : torch process group
        scatter_idx: the dim to scatter
        use_sync (bool): whether to synchronize after all-gather

    Returns:
        torch.tensor: scattered tensor (bs, seqlen/P, h)
    """
    seq_world_size = dist.get_world_size(group)

    if seq_world_size > 1:
        scatter_list = torch.split(input_tensor, seq_world_size, dim=scatter_idx)
        output_tensor = torch.empty_like(scatter_list[0])
        dist.scatter(
            tensor=output_tensor, scatter_list=scatter_list, group=group)
        if use_sync:
            torch.cuda.synchronize()

        return output_tensor
    else:
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
        return all_gather(inputs, group=group, gather_idx=gather_idx)

    @staticmethod
    def backward(ctx: Any,
                 *grad_output: torch.Tensor
        ) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            scatter(
                *grad_output,
                group=ctx.group,
                scatter_idx=ctx.gather_idx,
                use_sync=ctx.use_sync),
            None,
            None,
            None,
        )

class UlyssesAttention(torch.nn.Module):
    """UlyssesAttention, current support FA2 with packing only.

        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1) -> None:

        super(UlyssesAttention, self).__init__()
        self.spg = get_sequence_parallel_group()
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        # (b, N/P, h, d) -> (b, N, h/P, d)
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)
        # print_rank_0(kwargs)
        dropout_p = kwargs.get("dropout_p", 0.0)
        causal = kwargs.get("causal", False)
        sliding_window = kwargs.get("sliding_window", -1)
        
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        cu_seqlens = cu_seqlens.to(torch.int32)

        attn_output = flash_attn_varlen_func(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=dropout_p,
            window_size=(sliding_window, sliding_window),
            causal=causal
        )

        # if isinstance(attn_output, tuple):
        #     attn_output = attn_output[0]

        output = SeqAllToAll4D.apply(
            self.spg, attn_output.unsqueeze(0), self.gather_idx, self.scatter_idx
        )

        return output

def gather_batches(buffer, group):
    world_size = dist.get_world_size(group)
    if world_size > 1:
      with Timer("Gather batches"):
        gathered_batches = [None for _ in range(world_size)]
        dist.all_gather_object(
            object_list=gathered_batches, obj=buffer,
            group=group
        )

      gathered_batches = sum(gathered_batches, [])
    else:
      gathered_batches = buffer
    print_rank_0(f"Num batches: {len(gathered_batches)}")
    return gathered_batches

def gather_by_group(dataloader, group, buffer_size=1):
    buffer = []
    for batch in dataloader:
        buffer.append(batch)
        if len(buffer) >= buffer_size:
            yield from gather_batches(buffer, group)
            buffer = []
    if len(buffer) > 0:
        yield from gather_batches(buffer, group)
