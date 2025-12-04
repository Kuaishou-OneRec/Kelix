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
    fc1 = nn.Linear(dim, hidden_dim, bias=True)
    fc2 = nn.Linear(hidden_dim, dim, bias=True)
    return FeedForward(gate_proj=fc1, down_proj=fc2, up_proj=None, activation=activation_fn) if activation_fn is not None else FeedForward(gate_proj=fc1, down_proj=fc2, up_proj=None)


class SiglipAttention(nn.Module):
    """Siglip multi-head attention with optional Flash Attention 2 support."""

    def __init__(self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = False,
        attn_dropout: float = 0.0,
        attention_function: Literal["eager", "flash_attention_2"] = "eager",
        
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.attn_dropout = attn_dropout
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal
        self.head_dim = head_dim

        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings
        self.q_norm = q_norm
        self.k_norm = k_norm
        self._attention_function = get_attention_function(attention_function)
        self.cache_enabled = False


    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.
        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            max_seq_len (int): maximum sequence length model will be run with.
        """
        # Don't overwrite user defined kv_cache from init
        if self.kv_cache is not None:
            logger.warning(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    def reset_cache(self):
        """Reset the key value caches."""
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()


    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        q = self.q_proj(x)

        q_per_kv = self.num_heads // self.num_kv_heads
        if q_per_kv != 1:
            q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)
        q = q.transpose(1, 2)
        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            k = self.k_proj(y)
            v = self.v_proj(y)

            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)

            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        if self.num_heads != self.num_kv_heads:
            expand_shape = (b, self.num_kv_heads, q_per_kv, -1, self.head_dim)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        output = self._attention_function(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
            **kwargs,
        )

        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)



class SiglipAxialRotaryEmbedding(nn.Module):
    """把二维 (h, w) RoPE 封装成 attention 可直接调用的模块。"""

    def __init__(self, head_dim: int, *, max_grid_size: int, base: int = 10_000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim 必须能被 2 整除，才能按 h/w 分半。")
        self.axis_dim = head_dim // 2
        self.height_rope = RotaryPositionalEmbeddings(
            dim=self.axis_dim, max_seq_len=max_grid_size, base=base
        )
        self.width_rope = RotaryPositionalEmbeddings(
            dim=self.axis_dim, max_seq_len=max_grid_size, base=base
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Standard rotary position embedding rotation."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """标准 RoPE 计算公式: x * cos + rotate_half(x) * sin"""
        # 确保 cos/sin 和 x 在同一个 device 和 dtype
        # 注意：这里 _rotate_half 是在 axis_dim 内部进行的，不会混合 H 和 W
        return (x * cos) + (self._rotate_half(x) * sin)

    def _lookup(self, rope: RotaryPositionalEmbeddings, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_ids.dtype != torch.long:
            pos_ids = pos_ids.long()
        cache = rope.cache  # [max_seq_len, dim, 2]
        
        # 增加边界保护，防止索引越界 (Muse 的 NaViT 模式下 seq_len 可能会变)
        # 如果 pos_ids 超出 cache 范围，需要报错或者动态计算（视 RotaryPositionalEmbeddings 实现而定）
        
        gathered = cache[pos_ids]  # [..., dim, 2]
        cos = gathered[..., 0]     # [..., dim]
        sin = gathered[..., 1]     # [..., dim]
        
        # 调整维度以匹配 x: [batch, seq, 1, axis_dim]
        # 假设 x 是 [batch, seq, num_heads, head_dim]
        # pos_ids 是 [batch, seq] -> gathered 是 [batch, seq, dim, 2]
        # 我们需要 unsqueeze 让它能在 num_heads 维度广播
        return cos.unsqueeze(-2), sin.unsqueeze(-2)

    def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
        # x shape: [batch, seq, num_heads, head_dim]
        if input_pos is None:
            return x
        
        if isinstance(input_pos, dict):
            height_ids = input_pos["height"]
            width_ids = input_pos["width"]
        else:
            height_ids, width_ids = input_pos

        # 1. 把输入拆成两半
        # x_h: [..., axis_dim], x_w: [..., axis_dim]
        x_h, x_w = x.chunk(2, dim=-1)

        # 2. 分别查找 Embedding
        # cos_h, sin_h shape: [batch, seq, 1, axis_dim]
        cos_h, sin_h = self._lookup(self.height_rope, height_ids)
        cos_w, sin_w = self._lookup(self.width_rope, width_ids)

        # 3. 确保类型匹配 (bfloat16/float32)
        cos_h, sin_h = cos_h.to(x.dtype), sin_h.to(x.dtype)
        cos_w, sin_w = cos_w.to(x.dtype), sin_w.to(x.dtype)

        # 4. 分别应用 RoPE (关键修改！)
        # 在各自的子空间内旋转，互不干扰
        out_h = self._apply_rope(x_h, cos_h, sin_h)
        out_w = self._apply_rope(x_w, cos_w, sin_w)

        # 5. 拼回原状
        return torch.cat([out_h, out_w], dim=-1)