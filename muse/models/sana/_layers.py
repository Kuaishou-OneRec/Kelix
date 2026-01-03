# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sana DiT model layers.

This module implements the core building blocks for the Sana diffusion transformer,
following the exact logic from the original Sana codebase.

Reference: https://github.com/NVlabs/Sana
"""

import math
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from muse.layers.position_embeddings import Roraty2DPositionalEmbeddings


def get_same_padding(kernel_size: int) -> int:
    """Calculate same padding for given kernel size."""
    return kernel_size // 2


def val2tuple(x, n: int):
    """Convert value to tuple of length n."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return tuple([x] * n)


def t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN-Zero modulation.
    
    Args:
        x: Input tensor [B, N, C]
        shift: Shift tensor [B, 1, C]
        scale: Scale tensor [B, 1, C]
    
    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Reference: Sana/diffusion/model/norms.py Lines 183-232
    """
    
    def __init__(self, dim: int, scale_factor: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.
    
    Reference: Sana/diffusion/model/nets/sana_blocks.py Lines 1058-1103
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.
        
        Args:
            t: 1-D Tensor of N indices, one per batch element.
            dim: Dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        
        Returns:
            Tensor of shape [N, dim] of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype


class PatchEmbedMS(nn.Module):
    """2D Image to Patch Embedding with multi-scale support.
    
    Reference: Sana/diffusion/model/nets/sana_blocks.py Lines 1326-1359
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        kernel_size: Optional[int] = None,
        padding: int = 0,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = kernel_size or patch_size
        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]
        if isinstance(patch_size, (tuple, list)):
            self.patch_size = tuple(patch_size)
        else:
            self.patch_size = (patch_size, patch_size)
        self.flatten = flatten
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer.
    
    Reference: Sana/diffusion/model/nets/basic_modules.py Lines 555-574
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor, HW: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self.fc1(x.to(self.fc1.weight.dtype))
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvLayer(nn.Module):
    """Convolutional layer with optional normalization and activation.
    
    Reference: Sana/diffusion/model/nets/basic_modules.py Lines 29-96
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: Optional[int] = None,
        use_bias: bool = False,
        norm: Optional[str] = None,
        act: Optional[str] = None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        
        # Build normalization
        if norm == "bn2d":
            self.norm = nn.BatchNorm2d(out_dim)
        elif norm == "ln2d":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        
        # Build activation
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    """GLU MBConv FFN layer.
    
    Reference: Sana/diffusion/model/nets/basic_modules.py Lines 99-174
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        use_bias: Tuple[bool, bool, bool] = (True, True, False),
        norm: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
        act: Tuple[Optional[str], Optional[str], Optional[str]] = ("silu", "silu", None),
        dilation: int = 1,
    ):
        super().__init__()
        out_features = out_features or in_features
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)
        
        # GLU activation
        if act[1] == "silu":
            self.glu_act = nn.SiLU(inplace=False)
        elif act[1] == "gelu":
            self.glu_act = nn.GELU()
        else:
            self.glu_act = nn.Identity()
        
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_features,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )
    
    def forward(self, x: torch.Tensor, HW: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N ** 0.5)
        else:
            H, W = HW
        
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        
        x = self.point_conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        
        return x


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention for text conditioning.
    
    Reference: Sana/diffusion/model/nets/sana_blocks.py Lines 48-98
    
    This implementation supports both:
    - xformers-style packed sequences with y_lens list (for exact alignment with official)
    - Standard attention with tensor mask (fallback)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if qk_norm:
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-5)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        # 2D RoPE for cross attention (when use_rope=True)
        if use_rope:
            self.pos_embeddings = Roraty2DPositionalEmbeddings(self.head_dim)
        else:
            self.pos_embeddings = None
        
        # Check for xformers availability
        self._xformers_available = False
        try:
            import xformers.ops
            self._xformers_available = True
            self._xformers = xformers
        except ImportError:
            pass
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask=None,
        x_input_pos=None,
        cond_input_pos=None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Query tensor [B, N, C] (image tokens)
            cond: Key/Value tensor [1, L*B, C] (packed) or [B, L, C] (text tokens)
            mask: List of y_lens (xformers style) or attention mask tensor
            x_input_pos: 2D position ids for query, dict {"height": h_ids, "width": w_ids}
                or tuple (height_ids, width_ids)
            cond_input_pos: 2D position ids for key/value, same format as x_input_pos
        
        Returns:
            Output tensor [B, N, C]
        """
        B, N, C = x.shape
        
        q = self.q_linear(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        
        q = self.q_norm(q).view(B, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)
        
        # Apply 2D RoPE using Roraty2DPositionalEmbeddings
        if self.pos_embeddings is not None and x_input_pos is not None:
            print(f"Apply 2D RoPE, x_input_pos={x_input_pos}, q={q.shape}")
            q = self.pos_embeddings(q, input_pos=x_input_pos)
        if self.pos_embeddings is not None and cond_input_pos is not None:
            print(f"Apply 2D RoPE, cond_input_pos={cond_input_pos}, k={k.shape}")
            k = self.pos_embeddings(k, input_pos=cond_input_pos)
        
        if self._xformers_available:
            # Use xformers memory efficient attention with block diagonal mask
            attn_bias = None
            if mask is not None:
                # mask is list of y_lens for each batch item
                attn_bias = self._xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            x = self._xformers.ops.memory_efficient_attention(
                q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
            )
        else:
            # Fallback to standard attention
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            
            attn_mask = None
            if mask is not None and not isinstance(mask, list):
                if mask.ndim == 2:
                    # Check if mask is already in additive format (values <= 0)
                    if mask.max() <= 0:
                        # Already additive mask format
                        attn_mask = mask.to(q.dtype)
                    else:
                        # Binary mask format, convert to additive
                        attn_mask = (1 - mask.to(q.dtype)) * -10000.0
                    attn_mask = attn_mask[:, None, None].repeat(1, self.num_heads, 1, 1)
                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2)
        
        x = x.reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    


class FlashAttention(nn.Module):
    """Multi-head Flash Attention block with QK norm.
    
    Reference: Sana/diffusion/model/nets/sana_blocks.py Lines 859-939
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        HW: Optional[Tuple[int, int]] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        dtype = q.dtype
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        
        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            q = self._apply_rotary_emb(q, rotary_emb)
            k = self._apply_rotary_emb(k, rotary_emb)
        
        # Transpose for attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Attention mask
        attn_mask = None
        if mask is not None and mask.ndim == 2:
            attn_mask = (1 - mask.to(q.dtype)) * -10000.0
            attn_mask = attn_mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)
        
        x = x.reshape(B, N, C).to(dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def _apply_rotary_emb(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings."""
        x_rotated = torch.view_as_complex(hidden_states.transpose(1, 2).to(torch.float64).unflatten(3, (-1, 2)))
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).transpose(1, 2)
        return x_out.type_as(hidden_states)


class LiteLA(nn.Module):
    """Lightweight Linear Attention.
    
    Reference: Sana/diffusion/model/nets/sana_blocks.py Lines 211-301
    """
    
    PAD_VAL = 1
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-8, # diffusers uses 1e-15
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        heads = heads or int(out_dim // dim * heads_ratio)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        
        self.to_q = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.to_k = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.to_v = nn.Linear(in_dim, in_dim, bias=use_bias)
        self.proj = nn.Linear(in_dim, out_dim)
        
        self.kernel_func = nn.ReLU(inplace=False)
        
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
    
    def attn_matmul(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Lightweight linear attention computation.
        
        Always computes in float32 for numerical stability, matching diffusers' behavior.
        """
        # Convert to float32 BEFORE matmul operations (matches diffusers)
        q = self.kernel_func(q).float()  # B, h, h_d, N
        k = self.kernel_func(k).float()
        v = v.float()
        
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=self.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)
        
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        HW: Optional[Tuple[int, int]] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        dtype = q.dtype
        
        q = self.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)
        v = v.transpose(-1, -2)
        
        q = q.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N)
        v = v.reshape(B, C // self.dim, self.dim, N)
        
        if rotary_emb is not None:
            q = self._apply_rotary_emb(q, rotary_emb)
            k = self._apply_rotary_emb(k, rotary_emb)
        
        out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)
        
        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)
        
        return out
    
    def _apply_rotary_emb(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings for linear attention."""
        # For LiteLA, shape is (B, h, h_d, N), need different handling
        x_rotated = torch.view_as_complex(
            hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2))
        )
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return x_out.type_as(hidden_states)


class CaptionEmbedder(nn.Module):
    """Embeds text captions into vector representations with dropout for CFG.
    
    Reference: Sana/diffusion/model/nets/sana_blocks.py Lines 1174-1229
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        uncond_prob: float,
        act_layer: nn.Module = nn.GELU,
        token_num: int = 120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5)
        )
        self.uncond_prob = uncond_prob
    
    def token_drop(
        self,
        caption: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
        y_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], y_embedding, caption)
        return caption
    
    def forward(
        self,
        caption: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_embedding = self.y_embedding
        if train:
            if caption.shape[-2] < self.y_embedding.shape[-2]:
                y_embedding = self.y_embedding[:caption.shape[-2], :]
        
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids, y_embedding)
        
        caption = self.y_proj(caption)
        return caption


class T2IFinalLayer(nn.Module):
    """Final layer for text-to-image generation.
    
    Reference: Sana/diffusion/model/nets/sana_blocks.py Lines 984-1016
    """
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SanaMSBlock(nn.Module):
    """Sana transformer block with adaLN-Zero conditioning.
    
    Reference: Sana/diffusion/model/nets/sana_multi_scale.py Lines 53-148
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        qk_norm: bool = False,
        attn_type: str = "flash",
        ffn_type: str = "mlp",
        mlp_acts: Tuple[str, str, Optional[str]] = ("silu", "silu", None),
        linear_head_dim: int = 32,
        cross_norm: bool = False,
        cross_attn_type: str = "flash",
        use_cross_attn_rope: bool = False,
        cross_attn_x_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads  # Store for RoPE computation
        self.use_cross_attn_rope = use_cross_attn_rope
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention
        if attn_type == "flash":
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
            )
        elif attn_type == "linear":
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(
                hidden_size,
                hidden_size,
                heads=self_num_heads,
                eps=1e-8,
                qk_norm=qk_norm,
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        # Cross-attention
        self.cross_attn = MultiHeadCrossAttention(
            hidden_size,
            num_heads,
            qk_norm=cross_norm,
            use_rope=use_cross_attn_rope,
        )
        
        # Optional norm before cross attention
        if cross_attn_x_norm:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_cross = None

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # FFN
        if ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu,
                drop=0,
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        HW: Optional[Tuple[int, int]] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        # Get modulation parameters from timestep embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        
        # Self-attention with modulation
        x = x + self.drop_path(
            gate_msa * self.attn(
                t2i_modulate(self.norm1(x), shift_msa, scale_msa),
                HW=HW,
                rotary_emb=image_rotary_emb,
            )
        )
        
        # Cross-attention with optional x norm and 2D RoPE
        # x_input_pos and cond_input_pos are passed from model forward
        x_for_cross = self.norm_cross(x) if self.norm_cross is not None else x
        x = x + self.cross_attn(
            x_for_cross, y, mask,
            x_input_pos=kwargs.get("x_input_pos", None),
            cond_input_pos=kwargs.get("cond_input_pos", None))
        
        # FFN with modulation
        x = x + self.drop_path(
            gate_mlp * self.mlp(
                t2i_modulate(self.norm2(x), shift_mlp, scale_mlp),
                HW=HW,
            )
        )
        
        return x.contiguous()
