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
Sana DiT Model Definition.

This module implements the Sana diffusion transformer model for text-to-image generation,
following the exact architecture from the original Sana codebase.

Reference: https://github.com/NVlabs/Sana
- Model: Sana/diffusion/model/nets/sana_multi_scale.py Lines 155-413
"""

import math
import logging
from typing import Dict, Optional, Tuple, Any, List, Callable

import numpy as np
import torch
import torch.nn as nn

from muse.models.base import Model
from muse.config.model_config import SanaConfig
from muse.models.sana._layers import (
    TimestepEmbedder,
    CaptionEmbedder,
    PatchEmbedMS,
    SanaMSBlock,
    T2IFinalLayer,
    RMSNorm,
)

logger = logging.getLogger(__name__)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
    pe_interpolation: float = 1.0,
    base_size: int = 16,
) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings.
    
    Reference: Sana/diffusion/model/nets/sana.py Lines 416-433
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Grid height and width
        cls_token: Whether to include class token
        extra_tokens: Number of extra tokens
        pe_interpolation: Position embedding interpolation factor
        base_size: Base size for interpolation
    
    Returns:
        Position embeddings array of shape [grid_size*grid_size, embed_dim]
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size
    
    grid_h_arr = np.arange(grid_h, dtype=np.float32) / (grid_h / base_size) / pe_interpolation
    grid_w_arr = np.arange(grid_w, dtype=np.float32) / (grid_w / base_size) / pe_interpolation
    grid = np.meshgrid(grid_w_arr, grid_h_arr)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h, grid_w])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D sincos pos embed from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sincos pos embed from positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class SanaModel(Model):
    """Sana Diffusion Transformer for text-to-image generation.
    
    This implements the SanaMS architecture with support for both
    flash attention and linear attention.
    
    Reference: Sana/diffusion/model/nets/sana_multi_scale.py Lines 155-413
    """
    
    def __init__(self, config: SanaConfig):
        super().__init__(config)
        self.config = config
        
        # Model attributes
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.use_pe = config.use_pe
        self.pe_interpolation = config.pe_interpolation
        self.y_norm = config.y_norm
        
        # Store h, w for unpatchify
        self.h = self.w = 0
        
        # Patch embedding
        kernel_size = config.patch_size
        self.x_embedder = PatchEmbedMS(
            config.patch_size,
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            bias=True,
        )
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        
        # Caption embedding
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )
        
        # Y normalization
        if self.y_norm:
            self.attention_y_norm = RMSNorm(
                config.hidden_size,
                scale_factor=config.y_norm_scale_factor,
                eps=config.norm_eps,
            )
        
        # Position embedding buffer
        base_size = config.input_size // config.patch_size
        self.base_size = base_size
        num_patches = base_size * base_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, config.hidden_size))
        self.pos_embed_ms = None
        
        # Transformer blocks
        drop_path = torch.linspace(0, config.drop_path, config.depth, device='cpu').tolist()
        self.blocks = nn.ModuleList([
            SanaMSBlock(
                config.hidden_size,
                config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop_path=drop_path[i],
                qk_norm=config.qk_norm,
                attn_type=config.attn_type,
                ffn_type=config.ffn_type,
                mlp_acts=config.mlp_acts,
                linear_head_dim=config.linear_head_dim,
                cross_norm=config.cross_norm,
                cross_attn_type=config.cross_attn_type,
            )
            for i in range(config.depth)
        ])
        
        # Final layer
        self.final_layer = T2IFinalLayer(
            config.hidden_size,
            config.patch_size,
            self.out_channels,
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights following Sana's initialization scheme.
        
        Reference: Sana/diffusion/model/nets/sana_multi_scale.py Lines 414-441
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        
        # Initialize caption embedding MLP
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        
        # Initialize position embeddings if using sincos
        if self.use_pe:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.pos_embed.shape[1] ** 0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        data_info: Optional[Dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of Sana.
        
        Args:
            x: Latent tensor [N, C, H, W]
            timestep: Timestep tensor [N,]
            y: Text embedding tensor [N, 1, L, C] or [N, L, C]
            mask: Attention mask for text [N, 1, 1, L] or [N, L]
            data_info: Optional dict with additional info
        
        Returns:
            Output tensor [N, out_C, H, W]
        
        Reference: Sana/diffusion/model/nets/sana_multi_scale.py Lines 314-383
        """
        bs = x.shape[0]
        print(f"inputs x: {x.shape}")
        x = x.to(self.dtype)
        # Match official: timestep.long().to(torch.float32)
        timestep = timestep.long().to(torch.float32)
        y = y.to(self.dtype)
        
        # Get spatial dimensions
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        
        # Patch embedding
        x = self.x_embedder(x)  # [N, T, D] where T = H * W / patch_size^2
        
        # Apply position embedding if enabled
        if self.use_pe:
            if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x.shape[1:]:
                self.pos_embed_ms = (
                    torch.from_numpy(
                        get_2d_sincos_pos_embed(
                            self.pos_embed.shape[-1],
                            (self.h, self.w),
                            pe_interpolation=self.pe_interpolation,
                            base_size=self.base_size,
                        )
                    )
                    .unsqueeze(0)
                    .to(x.device)
                    .to(self.dtype)
                )
            x = x + self.pos_embed_ms
        
        # Timestep embedding
        t = self.t_embedder(timestep)  # [N, D]
        t0 = self.t_block(t)  # [N, 6*D]
        
        # Caption embedding
        print("before shape of y: {}".format(y.shape))
        y = self.y_embedder(y, self.training, mask=mask)  # [N, 1, L, D] or [N, L, D]
        print("shape of y: {}".format(y.shape))
        if self.y_norm:
            y = self.attention_y_norm(y)
        
        # Check for xformers availability (same logic as official)
        _xformers_available = False
        try:
            import xformers.ops
            _xformers_available = True
        except ImportError:
            pass
        
        # Process mask and y for cross-attention (matching official exactly)
        if mask is not None:
            mask = mask.to(torch.int16)
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1) if mask.ndim > 2 else mask
            if _xformers_available:
                # Pack y based on mask for xformers
                y = y.squeeze(1) if y.ndim == 4 else y
                y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = mask  # For non-xformers, pass mask as-is
                y = y.squeeze(1) if y.ndim == 4 else y
        elif _xformers_available:
            y_lens = [y.shape[2]] * y.shape[0] if y.ndim == 4 else [y.shape[1]] * y.shape[0]
            y = y.squeeze(1) if y.ndim == 4 else y
            y = y.view(1, -1, x.shape[-1])
        else:
            raise ValueError("xformers is required for Sana cross-attention without mask")
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, y, t0, y_lens, (self.h, self.w))
        
        # Final layer
        x = self.final_layer(x, t)  # [N, T, patch_size^2 * out_channels]
        
        # Unpatchify
        x = self.unpatchify(x)  # [N, out_channels, H, W]
        
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to image.
        
        Args:
            x: Tensor [N, T, patch_size^2 * C]
        
        Returns:
            Image tensor [N, C, H, W]
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs
    
    def forward_with_dpmsolver(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        data_info: Optional[Dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for DPM-Solver (no variance prediction).
        
        Reference: Sana/diffusion/model/nets/sana_multi_scale.py Lines 392-398
        """
        model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out
    
    @property
    def dtype(self):
        # Use compute dtype if set by FSDP (for mixed precision training)
        if hasattr(self, '_compute_dtype') and self._compute_dtype is not None:
            return self._compute_dtype
        return next(self.parameters()).dtype
    
    def get_layers_to_shard(self):
        """Return layers for FSDP sharding."""
        return self.blocks
    
    def get_checkpointable_module_classes(self):
        """Return module classes for gradient checkpointing."""
        return {SanaMSBlock}
    
    def get_optimizer_grouped_parameters(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        """Get optimizer grouped parameters."""
        return [
            {
                "params": [p for n, p in self.named_parameters() if p.requires_grad],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            },
        ]
    
    def convert_sana_state_dict(
        self,
        sana_state_dict: Dict[str, torch.Tensor],
        null_embed_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Convert Sana checkpoint state dict to muse format.
        
        The Sana checkpoint format is mostly compatible, with a few exceptions:
        - pos_embed is regenerated and should be removed
        - y_embedder.y_embedding should be loaded from null_embed file
        
        Reference: Sana/diffusion/utils/checkpoint.py Lines 251-331
        
        Args:
            sana_state_dict: State dict from Sana checkpoint
            null_embed_path: Path to null embed file
        
        Returns:
            Converted state dict for muse SanaModel
        """
        converted_state_dict = {}
        remove_keys = ["pos_embed", "pos_embed_ms"]
        
        for key, tensor in sana_state_dict.items():
            # Skip position embeddings (will be regenerated)
            if any(rk in key for rk in remove_keys):
                logger.info(f"Skipping key: {key} (will be regenerated)")
                continue
            
            # Auto reshape if needed (e.g., [dim,dim2,1,1] -> [dim,dim2,1,1,1])
            converted_state_dict[key] = tensor
        
        # Load null embedding if provided
        if null_embed_path is not None:
            try:
                null_embed = torch.load(null_embed_path, map_location="cpu")
                converted_state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
                logger.info(f"Loaded null embed from {null_embed_path}")
            except Exception as e:
                logger.warning(f"Failed to load null embed from {null_embed_path}: {e}")
        
        return converted_state_dict
    
    def convert_hf_state_dict(
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Convert Diffusers SanaTransformer2DModel state dict to muse format.
        
        Key mappings:
        - patch_embed -> x_embedder
        - time_embed -> t_embedder + t_block
        - caption_projection -> y_embedder.y_proj
        - caption_norm -> attention_y_norm
        - transformer_blocks -> blocks
        - norm_out + proj_out + scale_shift_table -> final_layer
        
        Args:
            hf_state_dict: State dict from Hugging Face model
        
        Returns:
            Converted state dict for muse SanaModel
        """
        muse_state_dict = {}
        
        for key, value in hf_state_dict.items():
            
            new_key = key
            
            # Patch embedding: patch_embed -> x_embedder
            if key.startswith("patch_embed."):
                new_key = key.replace("patch_embed.", "x_embedder.")
            
            # Time embedding (AdaLayerNormSingle)
            # diffusers: time_embed.emb.timestep_embedder.linear_1/2 -> muse: t_embedder.mlp.0/2
            # diffusers: time_embed.linear.* -> muse: t_block.1.*
            elif key.startswith("time_embed."):
                if "emb.timestep_embedder.linear_1." in key:
                    new_key = key.replace("time_embed.emb.timestep_embedder.linear_1.", "t_embedder.mlp.0.")
                elif "emb.timestep_embedder.linear_2." in key:
                    new_key = key.replace("time_embed.emb.timestep_embedder.linear_2.", "t_embedder.mlp.2.")
                elif "linear." in key:
                    new_key = key.replace("time_embed.linear.", "t_block.1.")
                else:
                    # Skip other time_embed keys that don't map directly
                    continue
            
            # Caption projection: caption_projection -> y_embedder.y_proj
            elif key.startswith("caption_projection."):
                new_key = key.replace("caption_projection.", "y_embedder.y_proj.")
                # PixArtAlphaTextProjection: linear_1, linear_2 -> Mlp: fc1, fc2
                new_key = new_key.replace("linear_1.", "fc1.")
                new_key = new_key.replace("linear_2.", "fc2.")
            
            # Caption norm: caption_norm -> attention_y_norm
            elif key.startswith("caption_norm."):
                new_key = key.replace("caption_norm.", "attention_y_norm.")
            
            # Transformer blocks: transformer_blocks -> blocks
            elif key.startswith("transformer_blocks."):
                new_key = key.replace("transformer_blocks.", "blocks.")
                
                # Self attention: attn1 -> attn (for LiteLA)
                new_key = new_key.replace(".attn1.", ".attn.")
                # Cross attention: attn2 -> cross_attn
                new_key = new_key.replace(".attn2.", ".cross_attn.")
                
                # FFN: ff -> mlp
                new_key = new_key.replace(".ff.", ".mlp.")
                
                # GLUMBConv mappings
                # diffusers uses nn.Conv2d directly, muse uses ConvLayer wrapper
                # ff.conv_inverted -> mlp.inverted_conv.conv
                new_key = new_key.replace(".mlp.conv_inverted.", ".mlp.inverted_conv.conv.")
                new_key = new_key.replace(".mlp.conv_depth.", ".mlp.depth_conv.conv.")
                new_key = new_key.replace(".mlp.conv_point.", ".mlp.point_conv.conv.")
                # ff.nonlinearity has no parameters, skip
                # ff.norm -> mlp.norm (if exists) - but muse GLUMBConv doesn't have separate norm
                
                # Self-attention output projection: to_out.0 -> proj
                new_key = new_key.replace(".attn.to_out.0.", ".attn.proj.")
                
                # Self-attention norms: norm_q -> q_norm, norm_k -> k_norm
                new_key = new_key.replace(".attn.norm_q.", ".attn.q_norm.")
                new_key = new_key.replace(".attn.norm_k.", ".attn.k_norm.")
                
                # Cross-attention: to_q -> q_linear, to_out.0 -> proj
                new_key = new_key.replace(".cross_attn.to_q.", ".cross_attn.q_linear.")
                new_key = new_key.replace(".cross_attn.to_out.0.", ".cross_attn.proj.")
                
                # Cross-attention norms
                new_key = new_key.replace(".cross_attn.norm_q.", ".cross_attn.q_norm.")
                new_key = new_key.replace(".cross_attn.norm_k.", ".cross_attn.k_norm.")
            
            # Final layer: norm_out -> final_layer.norm_final
            elif key.startswith("norm_out."):
                new_key = key.replace("norm_out.norm.", "final_layer.norm_final.")
            
            # proj_out -> final_layer.linear
            elif key.startswith("proj_out."):
                new_key = key.replace("proj_out.", "final_layer.linear.")
            
            # scale_shift_table -> final_layer.scale_shift_table
            elif key == "scale_shift_table":
                new_key = "final_layer.scale_shift_table"
            
            muse_state_dict[new_key] = value
        
        return muse_state_dict
    
    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        """Return initializer function for given parameter name."""
        def default_init(tensor: torch.Tensor):
            if tensor.ndim >= 2:
                nn.init.xavier_uniform_(tensor)
            else:
                nn.init.zeros_(tensor)
        return default_init
