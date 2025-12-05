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

def KeyeMLP(dim: int, hidden_dim: int, activation_fn: Optional[nn.Module] = None) -> FeedForward:
    fc1 = nn.Linear(dim, hidden_dim, bias=True)
    fc2 = nn.Linear(hidden_dim, dim, bias=True)
    if activation_fn is None:
        activation_fn = nn.GELU(approximate="tanh")
    return FeedForward(gate_proj=fc1, down_proj=fc2, up_proj=None, activation=activation_fn) if activation_fn is not None else FeedForward(gate_proj=fc1, down_proj=fc2, up_proj=None)



class MultiHeadAttention(nn.Module):
    """Multi-headed attention layer with support for grouped query
    attention (GQA) introduced in https://arxiv.org/abs/2305.13245v1.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    `litgpt.Config <https://github.com/Lightning-AI/litgpt/blob/eda1aaaf391fd689664f95487ab03dc137e213fd/litgpt/config.py>`_).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings or LlamaRotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query, e.g. RMSNorm. For decoding, this is applied
            before updating from kv_cache. This means it will only support token wide normalization and not
            batch or sequence wide normalization.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.
            Default value is 0.0.

    Raises:
        ValueError:
            If ``num_heads % num_kv_heads != 0``, **or**
            if ``embed_dim % num_heads != 0``, **or**
            if ``attn_dropout < 0`` or ``attn_dropout > 1``, **or**
            if q_norm is defined without k_norm or vice versa
    """

    def __init__(
        self,
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
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        attention_function: str = "eager",
    ) -> None:
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

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        self._attention_function = get_attention_function(attention_function)

        # this flag indicates whether to update the kv-cache during forward
        # passes. when disabled, we can have the cache setup but still
        # perform normal forward passes
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
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. Optional only with kv_cache enabled.
            mask (Optional[torch.Tensor]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.decoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs: Additional keyword arguments to pass to the attention function. Common kwargs include:
                - cu_seqlens (torch.Tensor): cu_seqlens for the query and key, required if sample packing is used.
                  A int tensor with shape (N+1,) where N is the number of samples in a sequence.
                  e.g., [0, 10, 15, 40] means 3 samples with length 10, 5, 25 respectively.
                - window_size (int): sliding window size for local attention, default is -1 (no window).
                - cu_seqlens_q (torch.Tensor): cu_seqlens for the query specifically.
                - cu_seqlens_k (torch.Tensor): cu_seqlens for the key specifically.
                Note: Available kwargs depend on the attention_function being used.

        Raises:
            ValueError: If no ``y`` input and ``kv_cache`` is not enabled.

        Returns:
            torch.Tensor: output tensor with attention applied

        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Update k and v shape, positional embeddings, and normalization

            # k,v shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            # Apply positional embeddings
            # k,v shape: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Update key-value cache
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        # If needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        # k,v shape: [b, n_kv, s, h_d] -> [b, n_h, s, h_d]
        if self.num_heads != self.num_kv_heads:
            # For cross attention, we need to handle different sequence lengths
            # k,v have shape [b, n_kv, s_y, h_d], q has shape [b, n_h, s_x, h_d]
            # We need to expand k,v to [b, n_h, s_y, h_d]
            expand_shape = (b, self.num_kv_heads, q_per_kv, k.size(2), self.head_dim)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        # if get_context_parallel_world_size() > 1:
        #     cpg = get_context_parallel_group()
        #     # If context parallel is enabled, the input is sharded along
        #     # the sequence length dimension. We need to recover the original 
        #     # sequence length before the attention function.
        #     # q, k, v: [b, s_x, n_h, h_d] -> [b, s_x * P, n_h // P, h_d]
        #     q = SeqAllToAll4D.apply(cpg, q, 2, 1)
        #     k = SeqAllToAll4D.apply(cpg, k, 2, 1)
        #     v = SeqAllToAll4D.apply(cpg, v, 2, 1)
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = self._attention_function(
            q=q,
            k=k,
            v=v,
            is_causal=self.is_causal,
            attn_dropout=self.attn_dropout,
            **kwargs
        )

        # if get_context_parallel_world_size() > 1:
        #     cpg = get_context_parallel_group()
        #     # output: [b, s_x * P, n_h // P, h_d] -> [b, s_x, n_h, h_d]
        #     output = SeqAllToAll4D.apply(cpg, output, 1, 2)
        # # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)


# class KeyeAxialRotaryEmbedding(nn.Module):
#     """把二维 (h, w) RoPE 封装成 attention 可直接调用的模块。"""

#     def __init__(self, head_dim: int, *, max_grid_size: int, base: int = 10_000) -> None:
#         super().__init__()
#         if head_dim % 2 != 0:
#             raise ValueError("head_dim 必须能被 2 整除，才能按 h/w 分半。")
#         axis_dim = head_dim // 2
#         self.height_rope = RotaryPositionalEmbeddings(
#             dim=axis_dim, max_seq_len=max_grid_size, base=base
#         )
#         self.width_rope = RotaryPositionalEmbeddings(
#             dim=axis_dim, max_seq_len=max_grid_size, base=base
#         )

#     @staticmethod
#     def _rotate_half(x: torch.Tensor) -> torch.Tensor:
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat([-x2, x1], dim=-1)

#     def _lookup(self, rope: RotaryPositionalEmbeddings, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         if pos_ids.dtype != torch.long:
#             pos_ids = pos_ids.long()
#         cache = rope.cache  # [max_seq_len, dim, 2]
#         gathered = cache[pos_ids]  # [..., dim, 2]
#         cos = gathered[..., 0].unsqueeze(-2)  # [..., 1, dim]
#         sin = gathered[..., 1].unsqueeze(-2)
#         if cos.dim() == 3:  # 处理 [seq, 1, dim] 的情况
#             cos = cos.unsqueeze(0)
#             sin = sin.unsqueeze(0)
#         return cos, sin

#     def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
#         if input_pos is None:
#             return x
#         if isinstance(input_pos, dict):
#             height_ids = input_pos["height"]
#             width_ids = input_pos["width"]
#         else:
#             height_ids, width_ids = input_pos  # 形状可为 [seq] 或 [batch, seq]

#         cos_h, sin_h = self._lookup(self.height_rope, height_ids)
#         cos_w, sin_w = self._lookup(self.width_rope, width_ids)
#         cos = torch.cat([cos_h, cos_w], dim=-1).to(dtype=x.dtype)
#         sin = torch.cat([sin_h, sin_w], dim=-1).to(dtype=x.dtype)
#         # x 需 reshape 为 [batch, seq, num_heads, head_dim] 再传进来
#         return (x * cos) + (self._rotate_half(x) * sin)



class KeyeAxialRotaryEmbedding(nn.Module):
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
            height_ids, width_ids = input_pos

        # [B, S, 36]
        cos_h, sin_h = self._lookup(self.height_rope, height_ids)
        cos_w, sin_w = self._lookup(self.width_rope, width_ids)

        # === 调试打印 ===
        # 我们假设 dim=36 是由 [18个H, 18个H] 组成的
        # 检查 cos_h 的前半段和后半段是否相等
        h_part1, h_part2 = cos_h.chunk(2, dim=-1)
        w_part1, w_part2 = cos_w.chunk(2, dim=-1)
        
        # 只打印第一个 token 的前 3 维
        if torch.rand(1) < 0.01: # 偶尔打印防止刷屏，或者在单测里必定打印
            print(f"\n[RoPE Debug Internal]")
            print(f"H_Part1 (first 3): {h_part1.flatten()[:3].detach().cpu().tolist()}")
            print(f"H_Part2 (first 3): {h_part2.flatten()[:3].detach().cpu().tolist()}")
            print(f"Equal? {(h_part1 - h_part2).abs().max().item() < 1e-5}")
        # ================

        # 你的修正逻辑
        cos = torch.cat([h_part1, w_part1, h_part2, w_part2], dim=-1).to(dtype=x.dtype)
        sin = torch.cat([sin_h.chunk(2, dim=-1)[0], 
                         sin_w.chunk(2, dim=-1)[0], 
                         sin_h.chunk(2, dim=-1)[1], 
                         sin_w.chunk(2, dim=-1)[1]], dim=-1).to(dtype=x.dtype)
        
        return (x * cos) + (self._rotate_half(x) * sin)



    # def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
    #     print("DEBUG: Running KeyeAxialRotaryEmbedding Forward...")  # <--- 加入这行
    #     if input_pos is None:
    #         return x
    #     if isinstance(input_pos, dict):
    #         height_ids = input_pos["height"]
    #         width_ids = input_pos["width"]
    #     else:
    #         height_ids, width_ids = input_pos  # 形状可为 [seq] 或 [batch, seq]

    #     # cos_h, sin_h 维度为 [..., axis_dim] (即 36)
    #     # 内部结构为 [Part1(18), Part2(18)]
    #     cos_h, sin_h = self._lookup(self.height_rope, height_ids)
    #     cos_w, sin_w = self._lookup(self.width_rope, width_ids)
        
    #     # [FIX] 对齐 Origin 模型的 RoPE 排列逻辑
    #     # Origin 逻辑: repeat(1, 2) -> [H, W, H, W]
    #     # Muse 原逻辑: cat([H, W]) -> [H, H, W, W] (H包含part1/2)
    #     # 修正：将 H 和 W 分别切开，然后交错拼接
        
    #     # 1. Split parts
    #     ch1, ch2 = cos_h.chunk(2, dim=-1)
    #     cw1, cw2 = cos_w.chunk(2, dim=-1)
        
    #     sh1, sh2 = sin_h.chunk(2, dim=-1)
    #     sw1, sw2 = sin_w.chunk(2, dim=-1)
        
    #     # 2. Interleave: [H1, W1, H2, W2]
    #     cos = torch.cat([ch1, cw1, ch2, cw2], dim=-1).to(dtype=x.dtype)
    #     sin = torch.cat([sh1, sw1, sh2, sw2], dim=-1).to(dtype=x.dtype)
        
    #     # x 需 reshape 为 [batch, seq, num_heads, head_dim] 再传进来
    #     return (x * cos) + (self._rotate_half(x) * sin)