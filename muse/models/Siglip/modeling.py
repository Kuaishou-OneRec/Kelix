from typing import Dict, List, Optional, Tuple, Union
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from muse.config.model_config import SiglipVisionConfig
from muse.models.Siglip._layers import (
    SiglipAttention,
    SiglipMLP,
    SiglipAxialRotaryEmbedding,
)
from muse.layers.transformer import TransformerSelfAttentionLayer
from muse.layers.rms_norm import RMSNorm

# Import will be done when muse.models is imported, avoiding circular import
# The actual registration happens in __init__.py after import

logger = logging.getLogger(__name__)


def lecun_normal_(tensor: torch.Tensor) -> None:
    """LeCun normal initialization.
    
    LeCun normal initialization: std = sqrt(1 / fan_in)
    This is similar to Kaiming normal but uses fan_in instead of fan_out.
    
    For Linear layers: fan_in = in_features
    For Conv2d layers: fan_in = in_channels * kernel_size[0] * kernel_size[1]
    """
    if tensor.dim() < 2:
        # For 1D tensors (bias, etc.), use a small std
        std = 0.01
    elif tensor.dim() == 2:
        # Linear layer: (out_features, in_features)
        fan_in = tensor.size(1)
        std = math.sqrt(1.0 / fan_in)
    else:
        # Convolutional layer: (out_channels, in_channels, kernel_h, kernel_w, ...)
        # fan_in = in_channels * product of kernel sizes
        fan_in = tensor.size(1)
        for s in tensor.size()[2:]:
            fan_in *= s
        std = math.sqrt(1.0 / fan_in)
    init.normal_(tensor, mean=0.0, std=std)


def default_flax_embed_init(tensor: torch.Tensor) -> None:
    """Default Flax embedding initialization.
    
    Uses normal distribution with std = 1.0
    """
    init.normal_(tensor, mean=0.0, std=1.0)



class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.has_learnable_position_embedding = config.has_learnable_position_embedding if hasattr(config, "has_learnable_position_embedding") else False
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.cache_position_embedding = dict()
        self.cache_position_count = dict()
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int, is_after_patchify: bool = False) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
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

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def fetch_position_embedding_lfu_cache(self, embeddings, h, w, max_cache=20):
        grid = (h, w)
        if grid in self.cache_position_embedding:
            self.cache_position_count[grid] += 1
            return self.cache_position_embedding[grid]
        
        if len(self.cache_position_embedding) >= max_cache:
            min_hit_grid = min(self.cache_position_count, key=self.cache_position_count.get)
            self.cache_position_count.pop(min_hit_grid)
            self.cache_position_embedding.pop(min_hit_grid)
        
        position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)
        self.cache_position_count[grid] = 1
        self.cache_position_embedding[grid] = position_embedding
        return position_embedding

    def forward(
        self, 
        pixel_values: torch.FloatTensor, 
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        interpolate_pos_encoding=False,
        has_learnable_position_embedding=False
    ) -> torch.Tensor:
        has_learnable_position_embedding = self.has_learnable_position_embedding if hasattr(
            self.config, "has_learnable_position_embedding"
        ) else has_learnable_position_embedding
        target_dtype = self.patch_embedding.weight.dtype


        if pixel_values.dim() == 5:
            if position_ids is None:
                raise ValueError(
                    "position_ids must be provided when pixel_values has 5 dimensions."
                )
            from einops import rearrange

            batch_size, sequence_len, channel, height, width = pixel_values.shape
            pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
            embeddings = patch_embeds.flatten(-2).squeeze(-1)
            embeddings = rearrange(embeddings, "(b l) d -> b l d", b=batch_size, l=squence_len)

            # todo: not dubug
            if has_learnable_position_embedding:
                if interpolate_pos_encoding and image_grid_thw is not None:
                    flatten_image_grid_thw = self.flatten_list(image_grid_thw)
                    assert batch_size == 1
                    start = 0
                    image_embedding_list = list()
                    assert sum([np.prod(x) for x in flatten_image_grid_thw]) == embeddings.shape[1], (flatten_image_grid_thw, embeddings.shape)
                    embeddings = embeddings.squeeze(0)
                    tmp_embeddings = list()
                    for image_grid in image_grid_thw:
                        t, h, w = image_grid
                        end = start + t * h * w
                        image_embeddings = embeddings[start: end, :]
                        position_embedding = self.interpolate_pos_encoding(image_embeddings, h, w, True).squeeze(0).repeat(
                            t, 1)
                        image_embeddings = image_embeddings + position_embedding
                        tmp_embeddings.append(image_embeddings)
                        start = end
                    embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)
                else:
                    embeddings = embeddings + self.packing_position_embedding(position_ids)
            return embeddings

        raise NotImplementedError(str(pixel_values.shape))


class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipVisionConfig
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        num_kv_heads = num_heads
        self.rope = SiglipAxialRotaryEmbedding(head_dim, max_grid_size=4096, base=config.rope_theta)

        attn_dropout = getattr(config, "attention_dropout", 0.0)
        intermediate_dim = getattr(config, "intermediate_size", embed_dim * 4)
        use_qk_norm = getattr(config, "use_qk_norm", False)
        qk_norm_eps = getattr(config, "qk_norm_eps", 1e-6)
        norm_eps = getattr(config, "layer_norm_eps", 1e-5)

        layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            q_norm = RMSNorm(dim=head_dim, eps=qk_norm_eps) if use_qk_norm else None
            k_norm = RMSNorm(dim=head_dim, eps=qk_norm_eps) if use_qk_norm else None
            self_attn = SiglipAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=True),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
                output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=True),
                pos_embeddings=self.rope,
                q_norm=q_norm,
                k_norm=k_norm,
                kv_cache=None,
                max_seq_len=self.max_seq_len,
                is_causal=False,
                attn_dropout=attn_dropout,
                attention_function=config.attention_function,
            )
            mlp = SiglipMLP(dim=embed_dim, hidden_dim=intermediate_dim)
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps),
                mlp_norm=nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps),
            )
            layers.append(layer)
        self.layers = layers


    # Ignore copy
    # @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        image_grid_thw: Optional[
            List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]
        ] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        device = inputs_embeds.device
        hidden_states = inputs_embeds
        attention_mask = (
            attention_mask.to(inputs_embeds.dtype) if attention_mask is not None else None
        )
        attn_kwargs = {}
        if cu_seqlens is not None:
            attn_kwargs["cu_seqlens"] = cu_seqlens


        flatten_image_grid_thw = self.flatten_list(image_grid_thw)
        split_hids = list()
        split_wids = list()
        for t,h,w in flatten_image_grid_thw:
            image_pids = torch.arange(t * h * w, device = device) % (h * w)
            sample_hids = image_pids // w
            sample_wids = image_pids % w
            split_hids.append(sample_hids)
            split_wids.append(sample_wids)
        width_position_ids = torch.concat(split_wids, dim=0)
        height_position_ids = torch.concat(split_hids, dim=0)



        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_kwargs = {"mask": attention_mask, **attn_kwargs}
            layer_kwargs["input_pos"] = {"height": height_position_ids, "width": width_position_ids}

            hidden_states = encoder_layer(hidden_states, **layer_kwargs)

            if output_attentions:
                all_attentions = all_attentions + (None,)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": encoder_states,
            "attentions": all_attentions,
        }



class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.ln_post = nn.LayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.FloatTensor,
        position_ids: Optional[torch.Tensor] = None, 
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None, 
        interpolate_pos_encoding: bool = False, 
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        has_learnable_position_embedding: bool = False) -> Dict[str, torch.Tensor]:
        embeddings = self.embeddings(
            pixel_values,
            position_ids,
            image_grid_thw,
            interpolate_pos_encoding,
            has_learnable_position_embedding,
        )
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
        )
        hidden_states = encoder_outputs["last_hidden_state"]
        hidden_states = self.ln_post(hidden_states) 
        encoder_outputs["last_hidden_state"] = hidden_states
        return encoder_outputs