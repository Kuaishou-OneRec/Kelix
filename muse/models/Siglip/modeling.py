from typing import Dict, Callable
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import logging

from muse.models.base import Model
from muse.models.Siglip._layers import SiglipAttention, SiglipMLP, SiglipAxialRotaryEmbedding
from muse.layers.transformer import TransformerDecoder, TransformerSelfAttentionLayer
from muse.layers.rms_norm import RMSNorm
from muse.layers.linear import TiedLinear
from muse.layers.feed_forward import FeedForward

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
    def __init__(self, config: KeyeVisionConfig):
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

        sqrt_num_positions = torch_int(num_positions**0.5)
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
        has_learnable_position_embedding = self.has_learnable_position_embedding
        if pixel_values.dim() == 5:
            assert position_ids is not None
            from einops import rearrange
            batch_size, squence_len, channel, height, width = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
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
        else:
            raise NotImplementedError(str(pixel_values.shape))


class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: KeyeConfig
    """

    def __init__(self, config: KeyeConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        num_kv_heads = num_heads
        self.rope = SiglipAxialRotaryEmbedding(head_dim, max_grid_size=4096, base=config.rope_theta)

        layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
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
                q_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.q_norm else None,
                k_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.k_norm else None,
                kv_cache=None,
                max_seq_len=config.max_seq_len,
                attn_dropout=config.attn_dropout,
                attention_function=config.attention_function
            )
            mlp = SiglipMLP(dim=config.embed_dim, hidden_dim=config.intermediate_dim)
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=nn.LayerNorm(dim=config.embed_dim, eps=config.norm_eps),
                mlp_norm=nn.LayerNorm(dim=config.embed_dim, eps=config.norm_eps),
            )
            layers.append(layer)
        self.layers = layers
        # self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2, 
        #     config.rope_theta if hasattr(config, "rope_theta") else 10000
        # )
        # self.rope = RotaryPositionalEmbeddings(
        #     dim=head_dim,base=config.rope_theta
        # )


    def _validate_inputs(
        self,
        tokens: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Validates inputs for ``forward``.
        Args:
            tokens (Optional[torch.Tensor]): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Attention mask used for inference and for sequence packing.
            encoder_input (Optional[torch.Tensor]): Encoder input for cross-attention.
            encoder_mask (Optional[torch.Tensor]): Encoder attention mask for cross-embedding attention.
            input_pos (Optional[torch.Tensor]): Input tensor position IDs.
            input_embeds (Optional[torch.Tensor]): Input tensor embeddings (if short-circuiting token embeddings).

        Raises:
            ValueError:
                If neither tokens nor input_embeds are passed **or**
                If seq_len of x is bigger than max_seq_len, **or**
                if the model has caches which have been setup with self-attention layers and ``mask`` is not provided, **or**
                if the model has caches which have been setup with encoder layers and ``encoder_mask`` is not provided, **or**
                if the model has caches which have been setup ``input_pos`` is not provided.
        """

        if tokens is None and input_embeds is None:
            raise ValueError(
                "Either tokens or input_embeds must be provided to the decoder."
            )

        # input tensor of shape [b, s]
        seq_len = tokens.shape[1] if tokens is not None else input_embeds.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        if self.caches_are_enabled():
            if mask is None:
                raise ValueError(
                    "KV-caches for self-attention layers are setup for inference mode, causal masks must be provided!"
                    " Use the `mask` arg to provide a causal mask."
                )

            if encoder_input is not None and encoder_mask is None:
                raise ValueError(
                    "KV-caches for cross-attention/fusion layers are setup for inference mode and you seem to be using"
                    " encoder_input, causal masks must be provided! Use the `encoder_mask` arg to provide a causal mask."
                )

            if input_pos is None:
                raise ValueError(
                    "KV-caches are setup for inference mode, input positions must be provided!"
                )

    # Ignore copy
    # @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
    ) -> BaseModelOutput:
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
        
        vision_or_text = "vision"

        # use_rope = (use_rope is True) and (vision_or_text == "vision")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        device = inputs_embeds.device
        hidden_states = inputs_embeds
        attention_mask = attention_mask.to(inputs_embeds.dtype) if attention_mask is not None else None
        # if use_rope is True:
        #     flatten_image_grid_thw = self.flatten_list(image_grid_thw)
        #     # assert sum([np.prod(x) for x in flatten_image_grid_thw]) == hidden_states.shape[1], (flatten_image_grid_thw, hidden_states.shape)

        #     if width_position_ids is None or height_position_ids is None:
        #         split_hids = list()
        #         split_wids = list()
        #         for t, h, w in flatten_image_grid_thw:
        #             image_pids = torch.arange(t * h * w, device=device) % (h * w)
        #             sample_hids = image_pids // w
        #             sample_wids = image_pids % w
        #             split_hids.append(sample_hids)
        #             split_wids.append(sample_wids)
        #         width_position_ids = torch.concat(split_wids, dim=0)
        #         height_position_ids = torch.concat(split_hids, dim=0)
            

        #     pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
        #     max_grid_size = pids.max() + 1
        #     rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
        #     rope_emb = rope_emb_max_grid[pids].flatten(1)
        #     rope_emb = rope_emb.repeat(1, 2)
        #     rope_emb = (rope_emb.cos(), rope_emb.sin())


        # else:
        #     rope_emb = None


        
            
        attn_cu_seqlens = cu_seqlens


        attention_mask = attention_mask.to(inputs_embeds.dtype) if attention_mask is not None else None

        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    cu_seqlens=attn_cu_seqlens,
                    rope_emb=rope_emb,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": encoder_states,
            "attentions": all_attentions,
        }



class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: KeyeConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.ln_post = nn.LayerNorm(config.hidden_size) 

    def forward(self, pixel_values: torch.FloatTensor,
        position_ids: Optional[torch.Tensor] = None, 
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None, 
        interpolate_pos_encoding: bool = False, 
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        has_learnable_position_embedding: bool = False) -> Dict[str, torch.Tensor]:
        embeddings = self.embeddings(pixel_values, position_ids, image_grid_thw, interpolate_pos_encoding, has_learnable_position_embedding)
        encoder_outputs = self.encoder(embeddings, attention_mask, cu_seqlens)
        hidden_states = encoder_outputs["last_hidden_state"]
        hidden_states = self.ln_post(hidden_states) 
        encoder_outputs["last_hidden_state"] = hidden_states
        return encoder_outputs