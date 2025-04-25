import torch
from torch import Tensor
from typing import List


__all__ = [
    "varlen_sparse_segment_sum",
    "varlen_sparse_segment_sum_grad",
    "prefetch_pull",
    "prefetch",
    "remote_pull_sparse",
    "pull_sparse",
    "push_sparse",
]


def varlen_sparse_segment_sum(
    embed: Tensor,
    embed_offset: Tensor,
    segment_ids: Tensor,
    block_dims: List[int],
    num_slots: List[int],
    batch_size: int,
) -> List[Tensor]:
    return torch.ops.kai_torch.varlen_sparse_segment_sum.default(
        embed, embed_offset, segment_ids, block_dims, num_slots, batch_size
    )


def varlen_sparse_segment_sum_grad(
    embed: Tensor,
    embed_offset: Tensor,
    segment_ids: Tensor,
    grads: List[Tensor],
    block_dims: List[int],
    num_slots: List[int],
    batch_size: int,
) -> Tensor:
    return torch.ops.kai_torch.varlen_sparse_segment_sum_grad.default(
        embed, embed_offset, segment_ids, grads, block_dims, num_slots, batch_size
    )


@torch.library.register_fake("kai_torch::varlen_sparse_segment_sum")
def _(embed, embed_offset, segment_ids, block_dims, num_slots, batch_size):
    torch._check(embed_offset.shape == segment_ids.shape)
    torch._check(len(block_dims) == len(num_slots))

    total_dim = 0
    result = []
    for d, n in zip(block_dims, num_slots):
        result.append(torch.empty(batch_size * d * n), dtype=embed.dtype)
    return result


def _varlen_sparse_segment_sum_backward(ctx, grads):
    embed, embed_offset, segment_ids = ctx.saved_tensors
    grad_embed = None
    if ctx.needs_input_grad[0]:
        grad_embed = torch.ops.kai_torch.varlen_sparse_segment_sum_grad.default(
            embed,
            embed_offset,
            segment_ids,
            grads,
            ctx.block_dims,
            ctx.num_slots,
            ctx.batch_size,
        )
    return grad_embed, None, None, None, None, None


def _varlen_sparse_segment_sum_setup_context(ctx, inputs, output):
    embed, embed_offset, segment_ids, block_dims, num_slots, batch_size = inputs
    ctx.block_dims = block_dims
    ctx.num_slots = num_slots
    ctx.batch_size = batch_size
    if ctx.needs_input_grad[0]:
        ctx.save_for_backward(embed, embed_offset, segment_ids)
    else:
        ctx.save_for_backward(None, None, None)


torch.library.register_autograd(
    "kai_torch::varlen_sparse_segment_sum",
    _varlen_sparse_segment_sum_backward,
    setup_context=_varlen_sparse_segment_sum_setup_context,
)


@torch.library.register_fake("kai_torch::varlen_sparse_segment_sum_grad")
def _(embed, embed_offset, segment_ids, grads, block_dims, num_slots, batch_size):
    return torch.empty_like(embed)


def prefetch_pull(
    batch_id: Tensor,
    uid: Tensor,
    uslot: Tensor,
    udim: Tensor,
    num_by_shard: Tensor,
    show: Tensor,
    score: Tensor,
    dim_by_shard: Tensor,
) -> None:
    return torch.ops.kai_torch.prefetch_pull.default(
        batch_id, uid, uslot, udim, num_by_shard, show, score, dim_by_shard
    )


@torch.library.register_fake("kai_torch::prefetch_pull")
def _(batch_id, uid, uslot, udim, num_by_shard, show, score, dim_by_shard):
    return None


def prefetch(
    batch_id: Tensor,
    uid: Tensor,
    uslot: Tensor,
    udim: Tensor,
    num_by_shard: Tensor,
    show: Tensor,
    score: Tensor,
) -> None:
    return torch.ops.kai_torch.prefetch.default(
        batch_id, uid, uslot, udim, num_by_shard, show, score
    )


@torch.library.register_fake("kai_torch::prefetch")
def _(batch_id, uid, uslot, udim, num_by_shard, show, score):
    return None


def remote_pull_sparse(
    batch_id: Tensor,
    dim_by_shard: Tensor,
    uslot: Tensor,
    num_by_shard: Tensor,
    show: Tensor,
) -> None:
    return torch.ops.kai_torch.remote_pull_sparse.default(
        batch_id, dim_by_shard, uslot, num_by_shard, show
    )


@torch.library.register_fake("kai_torch::remote_pull_sparse")
def _(batch_id, dim_by_shard, uslot, num_by_shard, show):
    return None


def pull_sparse(batch_id: Tensor, emb_size: int) -> Tensor:
    return torch.ops.kai_torch.pull_sparse.default(batch_id, emb_size)


@torch.library.register_fake("kai_torch::pull_sparse")
def _(batch_id, emb_size):
    return torch.empty([emb_size], dtype=torch.float32)


def push_sparse(
    batch_id: Tensor,
    udim: Tensor,
    dim_by_shard: Tensor,
    num_by_shard: Tensor,
    max_ts: Tensor,
    grad: Tensor,
) -> None:
    return torch.ops.kai_torch.push_sparse.default(
        batch_id, udim, dim_by_shard, num_by_shard, max_ts, grad
    )


@torch.library.register_fake("kai_torch::push_sparse")
def _(batch_id, udim, dim_by_shard, num_by_shard, max_ts, grad):
    return None
