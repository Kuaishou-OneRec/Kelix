# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Literal

import torch
from torch import nn
from muse.layers.attention_utils import get_attention_function
from muse.layers.feed_forward import FeedForward
from muse.layers.kv_cache import KVCache

from muse.training.parallel import get_context_parallel_world_size, \
    get_context_parallel_group, SeqAllToAll4D

logger = logging.getLogger(__name__)

def qwen3_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Qwen2 model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)

class Qwen3Attention(nn.Module):
    """
    Basically, it is standard multihead attention, but with QK-norm applied before
    the RoPE. It is unusual for most of the models, but Qwen3 became an exception to the rule.

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
        attention_function: Literal["eager", "flash_attention_2"] = "eager",
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
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. Optional only with kv_cache enabled.
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
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

        # Qwen3 applies QK-norm before the RoPE, which is different from most of the models.
        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        # Apply positional embeddings after q-norm
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

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

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Apply positional embeddings after k-norm
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # Update key-value cache
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        # If needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        # k,v shape: [b, n_kv, s, h_d] -> [b, n_h, s, h_d]
        if self.num_heads != self.num_kv_heads:
            expand_shape = (b, -1, self.num_kv_heads, q_per_kv, self.head_dim)
            k = k.unsqueeze(3).expand(expand_shape).flatten(2, 3)
            v = v.unsqueeze(3).expand(expand_shape).flatten(2, 3)

        if get_context_parallel_world_size() > 1:
            cpg = get_context_parallel_group()
            # If context parallel is enabled, the input is sharded along
            # the sequence length dimension. We need to recover the original 
            # sequence length before the attention function.
            # q, k, v: [b, s_x, n_h, h_d] -> [b, s_x * P, n_h // P, h_d]
            q = SeqAllToAll4D.apply(cpg, q, 2, 1)
            k = SeqAllToAll4D.apply(cpg, k, 2, 1)
            v = SeqAllToAll4D.apply(cpg, v, 2, 1)

        output = self._attention_function(
            q=q,
            k=k,
            v=v,
            mask=mask,
            attention_dropout=self.attn_dropout,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
            training=self.training,
            **kwargs,
        )
        if get_context_parallel_world_size() > 1:
            cpg = get_context_parallel_group()
            # output: [b, s_x * P, n_h // P, h_d] -> [b, s_x, n_h, h_d]
            output = SeqAllToAll4D.apply(cpg, output, 1, 2)
        # reshape the output to be the same shape as the input
        output = output.contiguous().view(b, s_x, -1)
        return self.output_proj(output)




class MultimodalRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base: float = 1_000_000.0,
        mrope_section: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Default mrope_section if not provided
        self.mrope_section = mrope_section or [16, 24, 24]
        self.rope_init()
    
    def rope_init(self):
        """Initialize inverse frequency buffer."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For default rope type, attention_scaling is 1.0
        self.attention_scaling = 1.0
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input.
        
        Args:
            x: Input tensor with shape [..., head_dim]
            
        Returns:
            Rotated tensor with same shape
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        *,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multimodal 3D rotary position embedding.

        Args:
            x: Input tensor with shape [batch_size, seq_len, num_heads, head_dim]
            input_pos: Position indices. Can be:
                       - [batch_size, seq_len]: Standard 1D position ids, will be expanded to 3D
                       - [3, batch_size, seq_len]: 3D multimodal position ids where 3 represents (temporal, height, width)
                       If None, returns x unchanged (no RoPE applied).

        Returns:
            Tensor with rotary position embedding applied, same shape as input
        """
        if input_pos is None:
            return x

        # x shape: [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Handle different input_pos formats
        if input_pos.dim() == 2:  # [batch_size, seq_len] -> expand to 3D
            # For 1D position ids, use same values for all 3 dimensions
            position_ids = input_pos.unsqueeze(0).expand(3, -1, -1)  # [3, batch_size, seq_len]
        elif input_pos.dim() == 3 and input_pos.shape[0] == 3:  # [3, batch_size, seq_len]
            position_ids = input_pos
        else:
            raise ValueError(f"Unsupported input_pos shape: {input_pos.shape}. Expected [batch_size, seq_len] or [3, batch_size, seq_len]")
        
        # Core RoPE block. Keye has different position ids for 3 dimensions
        # So we expand the inv_freq to shape (3, ...)
        # inv_freq shape: [dim // 2]
        # Expand to: [3, batch_size, dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        
        # position_ids shape: [3, batch_size, seq_len]
        # Expand to: [3, batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, :, None, :].float()
        
        # Force float32 for precision
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        with torch.autocast(device_type=device_type, enabled=False):
            # Compute frequencies: [3, batch_size, dim // 2, seq_len]
            # Then transpose to: [3, batch_size, seq_len, dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            
            # Concatenate to get full embedding: [3, batch_size, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Compute cos and sin: [3, batch_size, seq_len, dim]
            cos = emb.cos()
            sin = emb.sin()

        # Apply attention scaling (for compatibility with advanced RoPE types)
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        # Apply multimodal section splitting
        # mrope_section * 2 for cos/sin concatenation
        # e.g., [16, 24, 24] -> [16, 24, 24, 16, 24, 24]
        mrope_section_doubled = self.mrope_section * 2
        
        # Split cos/sin by sections: each is [3, batch_size, seq_len, section_size]
        cos_sections = cos.split(mrope_section_doubled, dim=-1)
        sin_sections = sin.split(mrope_section_doubled, dim=-1)
        
        # Debug store (raw cos/sin before chunk)
        # 注意：Origin 在 apply_multimodal_rotary_pos_emb 中存储的 cos/sin 已经是 bfloat16
        # 所以这里也需要先转换 dtype 再存储，保持一致
        if not hasattr(self, "_debug_rope_intermediates"):
            self._debug_rope_intermediates = {
                "inv_freq": None,
                "position_ids": None,
                "cos_before_chunk": None,
                "sin_before_chunk": None,
                "cos_after_chunk": None,
                "sin_after_chunk": None,
                "mrope_section": None,
            }
        self._debug_rope_intermediates["inv_freq"] = self.inv_freq.to(dtype=x.dtype).detach()
        self._debug_rope_intermediates["position_ids"] = position_ids.detach()
        # Origin 存储的 cos/sin 是 bfloat16（因为在 KeyeRotaryEmbedding.forward 返回时已转换）
        # 所以这里也需要先转换为 x.dtype 再存储
        self._debug_rope_intermediates["cos_before_chunk"] = cos.detach()
        self._debug_rope_intermediates["sin_before_chunk"] = sin.detach()
        self._debug_rope_intermediates["mrope_section"] = torch.tensor(
            self.mrope_section, device=x.device
        )

        # Select appropriate dimension for each section (cycling through 0, 1, 2)
        # section[i % 3] selects temporal(0), height(1), or width(2)
        # Result shape: [batch_size, seq_len, dim]
        cos_combined = torch.cat(
            [section[i % 3] for i, section in enumerate(cos_sections)],
            dim=-1
        )
        sin_combined = torch.cat(
            [section[i % 3] for i, section in enumerate(sin_sections)],
            dim=-1
        )

        # Store combined cos/sin (after chunk) 
        # Origin 存储的 shape 是 [1, 1, 209, 128]，但这是因为 Origin 的 q/k 是 [b, h, s, d]
        # Muse 的 q/k 是 [b, s, h, d]，所以 unsqueeze 位置不同
        # 为了对比，我们存储 unsqueeze 到 dim=1 的版本，与 Origin 保持一致
        self._debug_rope_intermediates["cos_after_chunk"] = cos_combined.unsqueeze(1).to(dtype=x.dtype).detach()
        self._debug_rope_intermediates["sin_after_chunk"] = sin_combined.unsqueeze(1).to(dtype=x.dtype).detach()

        # Add head dimension for broadcasting: [batch_size, seq_len, 1, dim]
        cos_combined = cos_combined.unsqueeze(2).to(dtype=x.dtype)
        sin_combined = sin_combined.unsqueeze(2).to(dtype=x.dtype)
        
        # Apply RoPE: x_embed = (x * cos) + (rotate_half(x) * sin)
        x_rotated = self.rotate_half(x)
        x_out = (x * cos_combined) + (x_rotated * sin_combined)

        # Store outputs for debugging
        # Transpose to match Origin's [batch, heads, seq_len, head_dim] format
        # Muse uses [batch, seq_len, heads, head_dim], so we need transpose(1, 2)
        if not hasattr(self, "_debug_rope_outputs"):
            self._debug_rope_outputs = []
        self._debug_rope_outputs.append(x_out.transpose(1, 2).detach())
        
        return x_out
    
    def apply_rotary_pos_emb_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Apply multimodal 3D rotary position embedding to both query and key tensors.
        
        This is a convenience method that applies the same position embedding to both
        q and k tensors, which is the common use case in attention.
        
        Args:
            q: Query tensor with shape [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor with shape [batch_size, seq_len, num_kv_heads, head_dim]
            input_pos: Position indices with shape [3, batch_size, seq_len]
        
        Returns:
            Tuple of (q_embed, k_embed) with rotary position embedding applied
        """
        q_embed = self.forward(q, input_pos=input_pos)
        k_embed = self.forward(k, input_pos=input_pos)
        return q_embed, k_embed
