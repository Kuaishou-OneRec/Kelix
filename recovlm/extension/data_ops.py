import os
import torch
from torch import Tensor
from typing import List, Optional


__all__ = [
    "uniq_shard_op",
    "make_embed_block_op",
    "filter_op",
    "reco_remap",
]


def uniq_shard_op(
    batch: List[Tensor],
    block_dims_str: str,
    slot_ids_str: str,
    ps_num: int,
    engine_num: int,
    ps_seed: Optional[int] = None,
    dedup_showclk: bool = False,
    show_coeff: float = 0.2,
    clk_coeff: float = 1.0,
    use_nonclk: bool = False,
    nonclk_coeff: float = -1.0,
    block_indicators_str: str = "",
    indicator_idx_str: str = "",
    label_index: int = 0,
    input_offset: int = 1,
    trainables: List[bool] = [],
) -> List[torch.Tensor]:
    """
    uniq_shard_op

    returns:
        [emb_off, uid, uslot, udim, show, click, score, udim_off, shard, total_dim]
    """
    if ps_seed is None:
        ps_seed = int(os.environ.get("KML_ID", "0"))
    return torch.ops.kai_torch.uniq_shard_op(
        label_index,
        input_offset,
        block_dims_str,
        slot_ids_str,
        trainables,
        ps_seed,
        ps_num,
        engine_num,
        dedup_showclk,
        show_coeff,
        clk_coeff,
        use_nonclk,
        nonclk_coeff,
        block_indicators_str,
        indicator_idx_str,
        batch,
    )


@torch.library.register_fake("kai_torch::uniq_shard_op")
def _(
    label_index: int,
    input_offset: int,
    block_dims_str: str,
    slot_ids_str: str,
    trainables: List[bool],
    ps_seed: int,
    ps_num: int,
    engine_num: int,
    dedup_showclk: float,
    show_coeff: float,
    clk_coeff: float,
    use_nonclk: bool,
    nonclk_coeff: float,
    block_indicators_str: str,
    indicator_idx_str: str,
    batch: List[Tensor],
):
    return []


def make_embed_block_op(
    block_list: List[str], block_width: List[int], batch: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    make_embed_block_op

    returns:
        [block_ids, block_cumsum]
    """
    return torch.ops.kai_torch.make_embed_block_op(block_list, block_width, batch)


@torch.library.register_fake("kai_torch::make_embed_block_op")
def _(block_list: List[str], block_width: List[int], batch: List[torch.Tensor]):
    return []


def filter_op(inputs: List[Tensor], mask: Tensor) -> List[Tensor]:
    """
    Args:
        inputs (List[Tensor]): flatten batch data
        mask (Tensor): the mask of which row need to filter

    Examples:
    >>> filter_op([[torch.tensor([1,2,3,4,5])]], torch.tensor([False, True, False, False, False]))
    [[torch.tensor([1,3,4,5])]]
    """
    return torch.ops.kai_torch.filter_op(inputs, mask)


@torch.library.register_fake("kai_torch::filter_op")
def _(inputs: List[Tensor], mask: Tensor) -> List[Tensor]:
    return []


def reco_remap(
    remap_inputs: List[Tensor],
    remap_slots: List[int],
    inplace: bool,
) -> List[Tensor]:
    """
    Args:
        remap_inputs (List[Tensor]): batch of data to remap
        remap_slots (List[int]): remap to slots
        inplace (bool): data will not be copied if inplace is True

    Examples:
    >>> # input number is (20<<48) + 1
    >>> reco_remap([torch.tensor([5629499534213121])], [40], True)
    [tensor([6755399441055745])]
    >>> # output number is (40<<48) + 1
    """
    return torch.ops.kai_torch.reco_remap(
        remap_inputs, remap_slots, inplace
    )


@torch.library.register_fake("kai_torch::reco_remap")
def _(
    remap_inputs: List[Tensor],
    remap_slots: List[int],
    inplace: bool,
) -> List[Tensor]:
    return []
