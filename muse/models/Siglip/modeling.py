from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from research.muse.config.model_config import SiglipVisionConfig
from research.muse.models.Siglip._layers import SiglipAttention, SiglipMLP
from research.muse.models.base import Model
from research.muse.layers.rms_norm import RMSNorm

logger = logging.getLogger(__name__)


def lecun_normal_(tensor: torch.Tensor) -> None:
    if tensor.dim() < 2:
        std = 0.01
    elif tensor.dim() == 2:
        fan_in = tensor.size(1)
        std = math.sqrt(1.0 / fan_in)
    else:
        fan_in = tensor.size(1)
        for size in tensor.size()[2:]:
            fan_in *= size
        std = math.sqrt(1.0 / fan_in)
    nn.init.normal_(tensor, mean=0.0, std=std)


def default_flax_embed_init(tensor: torch.Tensor) -> None:
    nn.init.normal_(tensor, mean=0.0, std=1.0)


class SigLIPRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self) -> None:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.has_learnable_position_embedding = config.has_learnable_position_embedding

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_patches).expand((1, -1)), persistent=False
        )

    @staticmethod
    def _flatten_image_grid(
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]]
    ) -> List[Tuple[int, int, int]]:
        if image_grid_thw is None:
            return []
        flattened: List[Tuple[int, int, int]] = []
        for grid in image_grid_thw:
            if isinstance(grid, list):
                flattened.extend(grid)
            else:
                flattened.append(grid)
        return flattened

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int, is_after_patchify: bool = False
    ) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        if pixel_values.dim() != 5:
            raise ValueError(
                f"Expected pixel_values with 5 dims (batch, seq, c, h, w). Got: {pixel_values.shape}"
            )

        from einops import rearrange

        batch_size, seq_len, channel, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = rearrange(embeddings, "(b l) d -> b l d", b=batch_size, l=seq_len)

        if self.has_learnable_position_embedding:
            if interpolate_pos_encoding and image_grid_thw is not None:
                flattened = self._flatten_image_grid(image_grid_thw)
                if flattened:
                    assert batch_size == 1, "Interpolated positional encoding assumes batch=1"
                    start = 0
                    accum: List[torch.Tensor] = []
                    embeddings = embeddings.squeeze(0)
                    for t, h, w in flattened:
                        end = start + t * h * w
                        slice_embed = embeddings[start:end, :]
                        pos_embed = (
                            self.interpolate_pos_encoding(slice_embed, h, w, True)
                            .squeeze(0)
                            .repeat(t, 1)
                        )
                        accum.append(slice_embed + pos_embed)
                        start = end
                    embeddings = torch.cat(accum, dim=0).unsqueeze(0)
            elif position_ids is not None:
                embeddings = embeddings + self.packing_position_embedding(position_ids)

        return embeddings


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        head_dim = self.hidden_size // config.num_attention_heads
        q_norm = RMSNorm(head_dim, eps=config.qk_norm_eps) if config.use_qk_norm else None
        k_norm = RMSNorm(head_dim, eps=config.qk_norm_eps) if config.use_qk_norm else None
        self.self_attn = SiglipAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            attn_dropout=config.attention_dropout,
            attention_function=config.attention_function,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        self.mlp = SiglipMLP(
            dim=self.hidden_size,
            hidden_dim=config.intermediate_size,
            activation=nn.GELU(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        cu_seqlens: Optional[torch.Tensor] = None,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            rope_emb=rope_emb,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: Tuple[torch.Tensor, Optional[torch.Tensor]] = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2, config.rope_theta)

    @staticmethod
    def _flatten_image_grid(
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]]
    ) -> List[Tuple[int, int, int]]:
        if image_grid_thw is None:
            return []
        flattened: List[Tuple[int, int, int]] = []
        for grid in image_grid_thw:
            if isinstance(grid, list):
                flattened.extend(grid)
            else:
                flattened.append(grid)
        return flattened

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        height_position_ids: Optional[torch.Tensor] = None,
        width_position_ids: Optional[torch.Tensor] = None,
        use_rope: bool = False,
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states: Optional[Tuple[torch.Tensor, ...]] = () if output_hidden_states else None
        all_attentions: Optional[Tuple[torch.Tensor, ...]] = () if output_attentions else None

        hidden_states = inputs_embeds
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if use_rope and image_grid_thw is not None:
            flattened = self._flatten_image_grid(image_grid_thw)
            if width_position_ids is None or height_position_ids is None:
                split_hids: List[torch.Tensor] = []
                split_wids: List[torch.Tensor] = []
                device = inputs_embeds.device
                for t, h, w in flattened:
                    image_pids = torch.arange(t * h * w, device=device) % (h * w)
                    sample_hids = image_pids // w
                    sample_wids = image_pids % w
                    split_hids.append(sample_hids)
                    split_wids.append(sample_wids)
                height_position_ids = torch.concat(split_hids, dim=0)
                width_position_ids = torch.concat(split_wids, dim=0)
            pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
            max_grid = int(pids.max().item()) + 1
            rope_freqs = self.rotary_pos_emb(max_grid)
            rope_emb = rope_freqs[pids].flatten(1)
            rope_emb = rope_emb.repeat(1, 2)
            rope_emb = (rope_emb.cos(), rope_emb.sin())

        for layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)  # type: ignore
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                rope_emb=rope_emb,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # type: ignore

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)  # type: ignore

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config.hidden_size, config.intermediate_size, activation=nn.GELU())

    def forward(
        self, hidden_state: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state, key_padding_mask=key_padding_mask)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_head = config.vision_use_head
        self.pooler = SiglipMultiheadAttentionPoolingHead(config) if self.use_head else None

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        attention_mask: Optional[torch.Tensor] = None,
        sample_indices: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        use_rope: bool = False,
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(
            pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            use_rope=use_rope,
        )
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)

        pooler_output = None
        if self.use_head and self.pooler is not None:
            pooler_output = self.pooler(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def _build_position_metadata(
    image_grid_thw: Optional[List[Tuple[int, int, int]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if image_grid_thw is None:
        raise ValueError("image_grid_thw must be provided for packed inputs")
    total_tokens = sum(int(np.prod(grid)) for grid in image_grid_thw)
    siglip_position_ids = []
    sample_indices = []
    cu_seqlens = [0]
    for idx, grid in enumerate(image_grid_thw):
        numel = int(np.prod(grid))
        siglip_position_ids.append(torch.arange(numel) % (grid[1] * grid[2]))
        sample_indices.append(torch.full((numel,), idx, dtype=torch.long))
        cu_seqlens.append(cu_seqlens[-1] + numel)
    position_ids = torch.cat(siglip_position_ids, dim=0)
    sample_indices = torch.cat(sample_indices, dim=0)
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)
    return position_ids, sample_indices, cu_seqlens_tensor


class SiglipVisionModel(Model):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.vision_model = SiglipVisionTransformer(config)
        self.post_init()

    def forward(
        self,
        pixel_values,
        image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        use_rope: bool = False,
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("pixel_values must be provided")

        if cu_seqlens is None and image_grid_thw is not None:
            position_ids, sample_indices, cu_seqlens = _build_position_metadata(image_grid_thw)
        else:
            sample_indices = None

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            use_rope=use_rope,
        )
        return vision_outputs

    def get_initializer(self, name: str):
        module_name = name.rsplit(".", 1)[0]
        param_suffix = name.rsplit(".", 1)[1] if "." in name else ""
        module = dict(self.named_modules()).get(module_name)

        if isinstance(module, nn.Linear):
            def _init(t: torch.Tensor):
                if param_suffix == "weight":
                    lecun_normal_(t)
                elif param_suffix == "bias":
                    nn.init.zeros_(t)
            return _init
        if isinstance(module, nn.Conv2d):
            def _init_conv(t: torch.Tensor):
                if param_suffix == "weight":
                    lecun_normal_(t)
                else:
                    nn.init.zeros_(t)
            return _init_conv
        if isinstance(module, nn.Embedding):
            def _init_embed(t: torch.Tensor):
                if param_suffix == "weight":
                    default_flax_embed_init(t)
            return _init_embed
        if isinstance(module, (nn.LayerNorm, RMSNorm)):
            def _init_norm(t: torch.Tensor):
                if param_suffix in ("weight", "scale"):
                    nn.init.ones_(t)
                elif param_suffix == "bias":
                    nn.init.zeros_(t)
            return _init_norm
        return lecun_normal_

    def get_layers_to_shard(self):
        return self.vision_model.encoder.layers

    def get_checkpointable_module_classes(self):
        return {SiglipEncoderLayer}
