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

    def __init__(self, head_dim: int, *, max_grid_size: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for Axial RoPE.")
        
        self.axis_dim = head_dim // 2
        self.base = base
        self.max_grid_size = max_grid_size
        
        # Precompute inverse frequencies (Standard RoPE logic)
        # theta_i = 1.0 / (base ** (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Caches
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Dynamically update cache if sequence length exceeds current cache."""
        if seq_len > self.max_seq_len_cached or self.cos_cached is None or self.cos_cached.device != device or self.cos_cached.dtype != dtype:
            self.max_seq_len_cached = seq_len
            
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            # Concatenate to match rotate_half logic: [cos(theta_0), cos(theta_1)..., cos(theta_0)...]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Standard rotary position embedding rotation."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """标准 RoPE 计算公式: x * cos + rotate_half(x) * sin"""
        return (x * cos) + (self._rotate_half(x) * sin)

    def _lookup(self, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pos_ids: [batch, seq] or [seq]
        seq_len = pos_ids.max().item() + 1
        self._update_cos_sin_cache(seq_len, pos_ids.device, pos_ids.dtype if pos_ids.is_floating_point() else torch.float32)
        
        # Select from cache
        # Note: We cast to input dtype (handled outside or here)
        cos = self.cos_cached[:seq_len].to(pos_ids.device)[pos_ids]
        sin = self.sin_cached[:seq_len].to(pos_ids.device)[pos_ids]
        
        # Reshape for broadcasting: [..., 1, axis_dim]
        # Input x is [batch, seq, heads, dim] -> split -> [batch, seq, heads, axis_dim]
        # We need cos/sin to be [batch, seq, 1, axis_dim]
        if cos.ndim == 2: # [seq, dim]
             cos = cos.unsqueeze(0).unsqueeze(2)
             sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.ndim == 3: # [batch, seq, dim]
             cos = cos.unsqueeze(2)
             sin = sin.unsqueeze(2)
             
        return cos, sin

    def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
        # x shape: [batch, seq, num_heads, head_dim]
        if input_pos is None:
            return x
        
        if isinstance(input_pos, dict):
            height_ids = input_pos["height"]
            width_ids = input_pos["width"]
        else:
            height_ids, width_ids = input_pos

        # 1. Split (Axial Split)
        x_h, x_w = x.chunk(2, dim=-1)

        # 2. Lookup
        cos_h, sin_h = self._lookup(height_ids)
        cos_w, sin_w = self._lookup(width_ids)
        
        # Ensure dtype match
        cos_h, sin_h = cos_h.to(x.dtype), sin_h.to(x.dtype)
        cos_w, sin_w = cos_w.to(x.dtype), sin_w.to(x.dtype)

        # 3. Apply Independently
        out_h = self._apply_rope(x_h, cos_h, sin_h)
        out_w = self._apply_rope(x_w, cos_w, sin_w)

        # 4. Concat
        return torch.cat([out_h, out_w], dim=-1)