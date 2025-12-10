from typing import Dict, List, Optional, Tuple, Union, Callable
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit._layers import (
    KeyeMLP,
)
from muse.layers.attention import MultiHeadAttention
from muse.layers.position_embeddings import TwoD_RotaryEmbedding
from muse.layers.transformer import TransformerSelfAttentionLayer
from muse.layers.rms_norm import RMSNorm
from muse.models.base import Model
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



class KeyeVisionEmbeddings(nn.Module):
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

    def forward(
        self, 
        pixel_values: torch.FloatTensor, 
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None,
        interpolate_pos_encoding=False,
        has_learnable_position_embedding=True
    ) -> torch.Tensor:
        has_learnable_position_embedding = self.has_learnable_position_embedding if hasattr(
            self.config, "has_learnable_position_embedding"
        ) else has_learnable_position_embedding
        target_dtype = self.patch_embedding.weight.dtype
        if pixel_values.dim() == 5:
            if position_ids is None:
                for thw_tuple in image_grid_thw:
                    numel = np.prod(thw_tuple)
                    position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                raise ValueError(
                    "position_ids must be provided when pixel_values has 5 dimensions."
                )
            from einops import rearrange

            batch_size, sequence_len, channel, height, width = pixel_values.shape
            pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
            embeddings = patch_embeds.flatten(-2).squeeze(-1)
            embeddings = rearrange(embeddings, "(b l) d -> b l d", b=batch_size, l=sequence_len)

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
                    raise NotImplementedError(str(pixel_values.shape))
            return embeddings

        raise NotImplementedError(str(pixel_values.shape))


class KeyeVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: KeyeVisionConfig
    """

    def __init__(self, config: KeyeVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        num_kv_heads = num_heads
        self.rope = TwoD_RotaryEmbedding(head_dim, max_grid_size=4096, base=config.rope_theta)

        attn_dropout = getattr(config, "attention_dropout", 0.0)
        intermediate_dim = getattr(config, "intermediate_size", embed_dim * 4)
        use_qk_norm = getattr(config, "use_qk_norm", False)
        qk_norm_eps = getattr(config, "qk_norm_eps", 1e-6)
        norm_eps = getattr(config, "layer_norm_eps", 1e-6)
        max_seq_len = getattr(config, "max_seq_len", 4096)

        layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            q_norm = RMSNorm(dim=head_dim, eps=qk_norm_eps) if use_qk_norm else None
            k_norm = RMSNorm(dim=head_dim, eps=qk_norm_eps) if use_qk_norm else None
            self_attn = MultiHeadAttention(
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
                max_seq_len=max_seq_len,
                is_causal=False,
                attn_dropout=attn_dropout,
                attention_function=config.attention_function,
            )
            mlp = KeyeMLP(dim=embed_dim, hidden_dim=intermediate_dim, activation_fn=nn.SiLU())
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps),
                mlp_norm=nn.LayerNorm(normalized_shape=embed_dim, eps=norm_eps),
            )
            layers.append(layer)
        self.layers = layers



    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

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
        # Debug: track H/W position id ranges to ensure Muse matches Origin ordering
        try:
            print(f"[DEBUG rope muse] height_position_ids={height_position_ids.tolist()}")
            print(f"[DEBUG rope muse] width_position_ids ={width_position_ids.tolist()}")
            print(
                f"[DEBUG rope muse] hids[min,max]={height_position_ids.min().item()},{height_position_ids.max().item()} "
                f"wids[min,max]={width_position_ids.min().item()},{width_position_ids.max().item()}"
            )
        except Exception as e:
            print(f"[DEBUG rope muse] print failed: {e}")



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



class KeyeVisionTransformer(Model):
    def __init__(self, config: KeyeVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = KeyeVisionEmbeddings(config)
        self.encoder = KeyeVisionEncoder(config)
        self.ln_post = nn.LayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps
        )


    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        """Return an initializer function for the given parameter name.
        
        This function implements Keye-VL-1_5 style initialization:
        - Attention modules (Qwen3Attention): xavier_uniform_ for weights, zeros_ for bias
        - MLP modules (FeedForward): xavier_uniform_ for weights, normal_(std=1e-6) for bias
        - Linear/Conv2d layers: lecun_normal_ for weights, zeros_ for bias
        - Embedding layers: default_flax_embed_init (normal with std=1.0)
        - LayerNorm/RMSNorm: weight=1, bias=0
        
        Reference: https://huggingface.co/Kwai-Keye/Keye-VL-1_5-8B/blob/main/modeling_keye_vl_1_5.py
        
        Args:
            name: Parameter name (e.g., "model.layers.0.attn.q_proj.weight")
            
        Returns:
            A callable function that takes a tensor and initializes it
        """
        # Find the module corresponding to this parameter name
        # Remove the parameter suffix (e.g., ".weight", ".bias", ".scale")
        module_name = name.rsplit(".", 1)[0]
        param_suffix = name.rsplit(".", 1)[1] if "." in name else ""
        
        # Get the module and its parent
        module = None
        parent_module = None
        for mod_name, mod in self.named_modules():
            if mod_name == module_name:
                module = mod
                # Get parent module name
                if "." in mod_name:
                    parent_name = ".".join(mod_name.rsplit(".", 1)[:-1])
                    for p_name, p_mod in self.named_modules():
                        if p_name == parent_name:
                            parent_module = p_mod
                            break
                break
        
        if module is None:
            parts = module_name.split(".")
            try:
                current = self
                for part in parts:
                    current = getattr(current, part)
                if isinstance(current, TiedLinear):
                    module = current.tied_module
                else:
                    module = current if isinstance(current, nn.Module) else None
            except AttributeError:
                module = None
        
        if module is None:
            return lecun_normal_
        
        is_attention = isinstance(parent_module, MultiHeadAttention)
        
        is_mlp = isinstance(parent_module, FeedForward)

        if isinstance(module, nn.Linear):
            def linear_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    if is_attention:
                        init.xavier_uniform_(tensor)
                    elif is_mlp:
                        init.xavier_uniform_(tensor)
                    else:
                        lecun_normal_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    if is_mlp:
                        init.normal_(tensor, mean=0.0, std=1e-6)
                    else:
                        init.zeros_(tensor)
            return linear_init
            
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                                  nn.ConvTranspose1d, nn.ConvTranspose2d)):
            def conv_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    lecun_normal_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    init.zeros_(tensor)
            return conv_init
            
        elif isinstance(module, nn.Embedding):
            # TODO: use better embedding initialization
            def embedding_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    # Use default_flax_embed_init (normal with std=1.0)
                    default_flax_embed_init(tensor)
            return embedding_init
            
        elif isinstance(module, nn.MultiheadAttention):
            def mha_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    init.xavier_uniform_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    init.zeros_(tensor)
            return mha_init
            
        elif (isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
              or "LayerNorm" in module.__class__.__name__
              or "RMSNorm" in module.__class__.__name__
              or isinstance(module, RMSNorm)):
            def norm_init(tensor: torch.Tensor):
                if param_suffix in ["weight", "scale"] and tensor is not None:
                    init.ones_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    init.zeros_(tensor)
            return norm_init
        
        return lecun_normal_

    def convert_hf_state_dict(self,
                              hf_state_dict: Dict[str, torch.Tensor],
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to a model state dictionary.
        
        Only converts vision_model weights. Skips text_model, logit_scale, logit_bias,
        and vision_model.head (pooling head) since they are not part of this model.
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
        converted_state_dict = {}
        skipped_keys = []
        
        # Prefix to look for in HF state dict
        vision_prefix = "siglip.vision_model."
        
        for hf_key, tensor in hf_state_dict.items():
            # Skip non-vision_model keys (text_model, logit_scale, logit_bias, etc.)
            if not hf_key.startswith(vision_prefix):
                skipped_keys.append(hf_key)
                continue
            
            # Remove the prefix to get the relative key
            rel_key = hf_key[len(vision_prefix):]
            
            # Skip pooling head (vision_model.head.*)
            if rel_key.startswith("head."):
                skipped_keys.append(hf_key)
                continue
            
            # Handle embeddings
            if rel_key.startswith("embeddings."):
                # Embeddings map directly: patch_embedding, position_embedding, packing_position_embedding
                converted_key = rel_key
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle post_layernorm -> ln_post
            if rel_key.startswith("post_layernorm."):
                # post_layernorm.weight -> ln_post.weight
                # post_layernorm.bias -> ln_post.bias
                suffix = rel_key.replace("post_layernorm.", "")
                converted_key = f"ln_post.{suffix}"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle encoder layers
            if rel_key.startswith("encoder.layers."):
                parts = rel_key.split(".", 3)  # ["encoder", "layers", "{i}", "rest"]
                if len(parts) < 4:
                    skipped_keys.append(hf_key)
                    continue
                
                layer_idx = parts[2]
                rest_key = parts[3]
                
                if rest_key.startswith("layer_norm1."):
                    suffix = rest_key.replace("layer_norm1.", "")
                    converted_key = f"encoder.layers.{layer_idx}.sa_norm.{suffix}"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                if rest_key.startswith("layer_norm2."):
                    suffix = rest_key.replace("layer_norm2.", "")
                    converted_key = f"encoder.layers.{layer_idx}.mlp_norm.{suffix}"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                if rest_key.startswith("self_attn."):
                    attn_key = rest_key.replace("self_attn.", "attn.")
                    attn_key = attn_key.replace("out_proj.", "output_proj.")
                    converted_key = f"encoder.layers.{layer_idx}.{attn_key}"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                if rest_key.startswith("mlp."):
                    new_rest_key = rest_key.replace("fc1", "w1").replace("fc2", "w2")
                    converted_key = f"encoder.layers.{layer_idx}.{new_rest_key}"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                skipped_keys.append(hf_key)
                continue
            
            skipped_keys.append(hf_key)
        
        if skipped_keys:
            interesting_skips = [k for k in skipped_keys if "text_model" not in k]
            if interesting_skips:
                logger.warning(
                    f"Skipped {len(interesting_skips)} keys during conversion (excluding text_model). "
                    f"First few: {interesting_skips[:5]}"
                )
        
        logger.info(
            f"Converted {len(converted_state_dict)} keys from "
            f"{len(hf_state_dict)} Hugging Face keys"
        )
        
        return converted_state_dict

    def get_layers_to_shard(self):
        return self.encoder.layers

    def get_checkpointable_module_classes(self):
        return {TransformerSelfAttentionLayer}


    def forward(self, pixel_values: torch.FloatTensor,
        position_ids: Optional[torch.Tensor] = None, 
        image_grid_thw: Optional[List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]] = None, 
        interpolate_pos_encoding: bool = True, 
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        has_learnable_position_embedding: bool = False,
        **kwargs) -> Dict[str, torch.Tensor]:
        # 兼容来自旧调用的多余参数（如 vision_return_embed_list / use_rope / window_size / sample_indices）
        # 这些参数在当前实现中未使用，安全忽略。
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


# Alias for backward compatibility and registry
KeyeVisionModel = KeyeVisionTransformer

