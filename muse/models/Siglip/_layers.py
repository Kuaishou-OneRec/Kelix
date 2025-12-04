# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional, Tuple, Literal

import torch
from torch import nn
from muse.layers.attention_utils import get_attention_function
from muse.layers.feed_forward import FeedForward
from muse.layers.rms_norm import RMSNorm
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.kv_cache import KVCache

logger = logging.getLogger(__name__)

def SiglipMLP(dim: int, hidden_dim: int, activation_fn: Optional[nn.Module] = None) -> FeedForward:
    fc1 = nn.Linear(dim, hidden_dim, bias=False)
    fc2 = nn.Linear(hidden_dim, dim, bias=False)
    return FeedForward(gate_proj=fc1, down_proj=fc2, up_proj=None, activation=activation_fn) if activation_fn is not None else FeedForward(gate_proj=fc1, down_proj=fc2, up_proj=None)


class SiglipAxialRotaryEmbedding(nn.Module):
    """把二维 (h, w) RoPE 封装成 attention 可直接调用的模块。"""

    def __init__(self, head_dim: int, *, max_grid_size: int, base: int = 10_000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim 必须能被 2 整除，才能按 h/w 分半。")
        axis_dim = head_dim // 2
        self.height_rope = RotaryPositionalEmbeddings(
            dim=axis_dim, max_seq_len=max_grid_size, base=base
        )
        self.width_rope = RotaryPositionalEmbeddings(
            dim=axis_dim, max_seq_len=max_grid_size, base=base
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _lookup(self, rope: RotaryPositionalEmbeddings, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_ids.dtype != torch.long:
            pos_ids = pos_ids.long()
        cache = rope.cache  # [max_seq_len, dim, 2]
        gathered = cache[pos_ids]  # [..., dim, 2]
        cos = gathered[..., 0].unsqueeze(-2)  # [..., 1, dim]
        sin = gathered[..., 1].unsqueeze(-2)
        if cos.dim() == 3:  # 处理 [seq, 1, dim] 的情况
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return cos, sin

    def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
        if input_pos is None:
            return x
        if isinstance(input_pos, dict):
            height_ids = input_pos["height"]
            width_ids = input_pos["width"]
        else:
            height_ids, width_ids = input_pos  # 形状可为 [seq] 或 [batch, seq]

        cos_h, sin_h = self._lookup(self.height_rope, height_ids)
        cos_w, sin_w = self._lookup(self.width_rope, width_ids)
        cos = torch.cat([cos_h, cos_w], dim=-1).to(dtype=x.dtype)
        sin = torch.cat([sin_h, sin_w], dim=-1).to(dtype=x.dtype)
        # x 需 reshape 为 [batch, seq, num_heads, head_dim] 再传进来
        return (x * cos) + (self._rotate_half(x) * sin)