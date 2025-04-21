import math
from copy import deepcopy
from typing import Union, Tuple, Sequence, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import PytorchGELUTanh
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available

from .configuration_moonvit import MoonViTConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
):
    """Multi-head attention using flash attention 2.
    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].
    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert (
        k_cu_seqlens[-1] == k.shape[0] == v.shape[0]
    ), "k_cu_seqlens must sum to k.shape[0]"
    assert q.dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
    )
    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SDPA attention.
    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
    """
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "sdpa": sdpa_attention,
    "eager": eager_attention,
}


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == self.weight.shape[:-1]:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=shape,
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )
        out = x + torch.cat(pos_embs)
        return out


class MoonVisionPatchEmbed(nn.Module):

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
    ):
        super().__init__()
        assert isinstance(
            patch_size, (int, Sequence)
        ), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert (
            len(patch_size) == 2
        ), f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_emb = Learnable2DInterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, dim=out_dim
        )

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 2): grid height and width
        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_hws)
        return x


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis = None

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_hws (torch.Tensor): grid height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(grid_hws.device)

        shapes = grid_hws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonVitEncoderLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = "eager",
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set
        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(
            hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MoonVitEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
    ) -> None:
        super().__init__()

        self.rope_2d = Rope2DPosEmb(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [MoonVitEncoderLayer(**block_cfg) for _ in range(num_layers)]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, grid_hws: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_hws=grid_hws)

        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_hws.dtype),
                grid_hws[:, 0] * grid_hws[:, 1],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def patch_merger(
    x: torch.Tensor,
    grid_hws: torch.Tensor,
    merge_kernel_size: list[int, int] = (2, 2),
) -> List[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for x_shape in grid_hws.tolist():
        height, width = x_shape[0], x_shape[1]
        # Get the current sequence
        seq = x[pre_sum : pre_sum + height * width]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.view(
            new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = reshaped_seq.permute(0, 2, 1, 3, 4).contiguous()
        padded_seq = reshaped_seq.view(
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)
        pre_sum += height * width

    return outputs


class MoonVitQFormerAttention(nn.Module):
    """Multi-head attention module for QFormer"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L_q, D)
            encoder_hidden_states: (B, L_kv, D)
            attention_mask: (B, L_q, L_kv)
        Returns:
            torch.Tensor: (B, L_q, D)
        """
        batch_size, query_length, _ = hidden_states.shape
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key_length = encoder_hidden_states.shape[1]
        
        # Project query, key, value
        query = self.query(hidden_states).reshape(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(encoder_hidden_states).reshape(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(encoder_hidden_states).reshape(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute context
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).reshape(batch_size, query_length, self.hidden_size)
        
        output = self.output(context)
        
        return output


class MoonVitQFormerLayer(nn.Module):
    """QFormer layer with self-attention and cross-attention"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        activation_function=F.gelu,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MoonVitQFormerAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Cross-attention
        self.cross_attn = MoonVitQFormerAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed forward
        self.mlp = MLP2(
            dims=[hidden_size, intermediate_size, hidden_size],
            activation=activation_function,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L_q, D)
            encoder_hidden_states: (B, L_kv, D)
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Cross-attention
        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MoonVitQFormer(nn.Module):
    """QFormer module for extracting image embeddings"""
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        num_queries: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_size))
        nn.init.normal_(self.query_tokens, mean=0, std=0.02)
        
        # QFormer layers
        self.layers = nn.ModuleList(
            [
                MoonVitQFormerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Pooler for creating a single embedding
        self.pooler = nn.Linear(hidden_size * num_queries, hidden_size)
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: Image features from vision encoder (B, L, D)
            encoder_attention_mask: Optional mask for encoder states
        Returns:
            torch.Tensor: Query features (B, num_queries, D)
        """
        batch_size = encoder_hidden_states.shape[0]
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Create cross-attention mask if needed
        if encoder_attention_mask is not None:
            # Expand attention mask to (B, num_queries, encoder_seq_len)
            attention_mask = encoder_attention_mask.unsqueeze(1).expand(-1, self.num_queries, -1)
        else:
            attention_mask = None
        
        # Process through QFormer layers
        hidden_states = query_tokens
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states
    
    def get_image_embedding(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Get a single embedding vector for each image from query features.
        
        Args:
            query_features: Query features from QFormer (B, num_queries, D)
        Returns:
            torch.Tensor: Single embedding per image (B, D)
        """
        # Flatten the query features and project to embedding dimension
        batch_size = query_features.shape[0]
        flattened = query_features.reshape(batch_size, -1)
        embedding = self.pooler(flattened)
        
        return embedding


class MoonVitPretrainedModel(PreTrainedModel):
    config_class = MoonViTConfig
    model_type = "moonvit"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: MoonViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )

        self.encoder = MoonVitEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PytorchGELUTanh(),
                "attn_bias": True,
                "attn_implementation": config._attn_implementation,
            },
        )
        
        # Add QFormer for extracting image embeddings
        self.qformer = MoonVitQFormer(
            hidden_size=config.hidden_size,
            num_layers=2,  # Can be configured as needed
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_queries=32,  # Can be configured as needed
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_hws: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_hws (torch.Tensor): The grid height and width.
        Returns:
            torch.Tensor: The output tokens.
        """
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        hidden_states = patch_merger(
            hidden_states, grid_hws, merge_kernel_size=self.merge_kernel_size
        )
        return hidden_states
        
    def get_image_embeddings(
        self, pixel_values: torch.Tensor, grid_hws: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract a single embedding vector for each image using QFormer.
        
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_hws (torch.Tensor): The grid height and width.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, hidden_size) containing 
                          one embedding vector per image.
        """
        # Get patch features from the vision encoder
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        
        # Process image patches with QFormer
        batch_image_indices = []
        start_idx = 0
        
        # Create a batch of image features
        image_features = []
        for i, hw in enumerate(grid_hws.tolist()):
            height, width = hw[0], hw[1]
            num_patches = height * width
            # Extract patches for this image
            img_patches = hidden_states[start_idx:start_idx + num_patches]
            # Add batch dimension for this single image
            img_patches = img_patches.unsqueeze(0)
            image_features.append(img_patches)
            start_idx += num_patches
        
        # Process each image with QFormer
        embeddings = []
        for img_features in image_features:
            # Process through QFormer
            query_features = self.qformer(img_features)
            # Get single embedding
            embedding = self.qformer.get_image_embedding(query_features)
            embeddings.append(embedding)
        
        # Stack all embeddings
        return torch.cat(embeddings, dim=0)