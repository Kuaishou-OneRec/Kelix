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
    # 显式 bias=True，虽然默认就是 True，但写出来更保险
    fc1 = nn.Linear(dim, hidden_dim, bias=True)
    fc2 = nn.Linear(hidden_dim, dim, bias=True)
    
    # SigLIP 默认使用 GELU(approximate='tanh')
    if activation_fn is None:
        activation_fn = nn.GELU(approximate="tanh")
        
    return FeedForward(
        gate_proj=fc1, 
        down_proj=fc2, 
        up_proj=None, 
        activation=activation_fn
    )

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
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # 1. Projection & Reshape -> [B, S, H, D]
        q = self.q_proj(x)
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # 2. Q Norm & RoPE (通常在 S, H 维度操作比较方便，保持现状)
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)
        if self.q_norm is not None:
            q = self.q_norm(q)

        # Handle K/V
        if y is None:
            # Self Attention Path (No Cache for test)
            if self.kv_cache is None:
                k = self.k_proj(x).view(b, s_x, -1, self.head_dim)
                v = self.v_proj(x).view(b, s_x, -1, self.head_dim)
                if self.pos_embeddings is not None:
                    k = self.pos_embeddings(k, input_pos=input_pos)
                if self.k_norm is not None:
                    k = self.k_norm(k)
            else:
                # ... Cache logic ...
                pass
        else:
            # Cross Attention ...
            pass

        # ---------------------------------------------------------------------
        # [关键修复] Transpose: [B, S, H, D] -> [B, H, S, D]
        # 这是为了匹配 SDPA (scaled_dot_product_attention) 的标准输入要求
        # HF 源码对应逻辑: query_states.view(batch, q_len, heads, dim).transpose(1, 2)
        # ---------------------------------------------------------------------
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. Attention Calculation
        # 输入形状现在是 [B, H, S, D]，计算的是 Sequence 维度的 Attention
        output = self._attention_function(
            q=q,
            k=k,
            v=v,
            is_causal=self.is_causal,
            attn_dropout=self.attn_dropout,
            **kwargs
        )

        # 4. Transpose Back: [B, H, S, D] -> [B, S, H, D]
        output = output.transpose(1, 2).contiguous()
        
        # 5. Flatten: [B, S, H*D]
        output = output.view(b, s_x, -1)
        
        return self.output_proj(output)

class SiglipAxialRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, *, max_grid_size: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.axis_dim = head_dim // 2
        self.base = base
        
        # 计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        
        #用于控制打印次数
        self.debug_counter = 0

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self.max_seq_len_cached or self.cos_cached is None or self.cos_cached.device != device or self.cos_cached.dtype != dtype:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)

    def _lookup_freqs(self, pos_ids: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_pos = pos_ids.max().item() + 1
        self._update_cos_sin_cache(max_pos, pos_ids.device, torch.float32 if pos_ids.device.type == "cpu" else torch.bfloat16)
        
        cos = self.cos_cached[pos_ids]
        sin = self.sin_cached[pos_ids]
        
        # Reshape [B, S, 1, AxisDim]
        cos = cos.view(batch_size, seq_len, 1, -1)
        sin = sin.view(batch_size, seq_len, 1, -1)
        
        return cos, sin

    def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
        # x: [Batch, Seq, NumHeads, HeadDim]
        if input_pos is None:
            return x
        
        batch_size, seq_len, _, head_dim = x.shape
        
        if isinstance(input_pos, dict):
            height_ids = input_pos["height"]
            width_ids = input_pos["width"]
        else:
            height_ids, width_ids = input_pos

        # 1. 获取 H 和 W 的频率
        cos_h, sin_h = self._lookup_freqs(height_ids, batch_size, seq_len)
        cos_w, sin_w = self._lookup_freqs(width_ids, batch_size, seq_len)
        
        # 2. 拼接频率 (Global RoPE logic)
        cos = torch.cat([cos_h, cos_w], dim=-1).to(x.dtype)
        sin = torch.cat([sin_h, sin_w], dim=-1).to(x.dtype)

        # ================= [MUSE DEBUG START] =================
        if self.debug_counter < 3:
            print(f"\n[MUSE DEBUG] SiglipAxialRotaryEmbedding inputs:")
            print(f"  x (q) shape: {x.shape}")
            print(f"  cos shape (concat): {cos.shape}")
            
            # 打印 Cos 样本 (取出第一个 Batch, 第一个 Seq 的向量)
            # cos shape is [B, S, 1, HeadDim] -> flatten to compare with HF
            cos_sample = cos[0, 0, 0, :].flatten() 
            mid = head_dim // 2
            
            print(f"  cos sample (Head Start): {cos_sample[:5].detach().cpu().numpy()}")
            print(f"  cos sample (Head Mid - Boundary): {cos_sample[mid-2:mid+3].detach().cpu().numpy()}")
            
            # 打印 RoPE 前的 Q
            print(f"  x (pre-rope) sample: {x[0, 0, 0, :5].detach().cpu().numpy()}")
        # ======================================================

        # 3. Apply
        out = self._apply_rope(x, cos, sin)

        # ================= [MUSE DEBUG RESULT] =================
        if self.debug_counter < 3:
            print(f"  x output sample: {out[0, 0, 0, :5].detach().cpu().numpy()}")
            print("-" * 50)
            self.debug_counter += 1
        # =======================================================

        return out