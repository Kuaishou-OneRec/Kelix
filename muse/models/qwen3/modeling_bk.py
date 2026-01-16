from typing import Dict, Callable, Optional, List
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import logging

from muse.models.base import Model
from muse.config import Qwen3Config
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.models.qwen3._layers import MultimodalRotaryEmbedding
from muse.layers.transformer import TransformerDecoder, TransformerSelfAttentionLayer
from muse.layers.rms_norm import RMSNorm
from muse.layers.linear import TiedLinear
from muse.models.qwen3._layers import qwen3_mlp, Qwen3Attention,KeyeFlashAttention2
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


class Qwen3Model(Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        head_dim = config.head_dim or config.embed_dim // config.num_heads
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads if config.num_kv_heads else num_heads

        # Select RoPE implementation based on config
        if config.use_multimodal_rope:
            # Use 3D multimodal RoPE for vision-language models (Keye-VL style)
            # Get mrope_section from config or rope_scaling
            mrope_section = config.mrope_section
            if mrope_section is None and config.rope_scaling is not None:
                mrope_section = config.rope_scaling.get("mrope_section")
            if mrope_section is None:
                # Default mrope_section if not specified
                mrope_section = [16, 24, 24]
                logger.warning(
                    f"use_multimodal_rope=True but mrope_section not specified. "
                    f"Using default: {mrope_section}"
                )
            self.rope = MultimodalRotaryEmbedding(
                dim=head_dim,
                max_seq_len=config.max_seq_len,
                base=config.rope_base,
                mrope_section=mrope_section,
            )
            logger.info(
                f"Using MultimodalRotaryEmbedding with mrope_section={mrope_section}"
            )
        else:
            # Use standard 1D RoPE
            self.rope = RotaryPositionalEmbeddings(
                dim=head_dim, max_seq_len=config.max_seq_len, base=config.rope_base
            )

        layers = nn.ModuleList()
        if config.use_multimodal_rope == False:
            for _ in range(config.num_layers):
                self_attn = Qwen3Attention(
                    embed_dim=config.embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_proj=nn.Linear(config.embed_dim, num_heads * head_dim, bias=config.q_proj_bias),
                    k_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.k_proj_bias),
                    v_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.v_proj_bias),
                    output_proj=nn.Linear(num_heads * head_dim, config.embed_dim, bias=False),
                    pos_embeddings=self.rope,
                    q_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.q_norm else None, # norm on head_dim
                    k_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.k_norm else None,
                    kv_cache=None,
                    max_seq_len=config.max_seq_len,
                    attn_dropout=config.attn_dropout,
                    attention_function=config.attention_function
                )
                mlp = qwen3_mlp(dim=config.embed_dim, hidden_dim=config.intermediate_dim)
                layer = TransformerSelfAttentionLayer(
                    attn=self_attn,
                    mlp=mlp,
                    sa_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                    mlp_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                )
                layers.append(layer)
        else:
            for _ in range(config.num_layers):
                self_attn = KeyeFlashAttention2(
                    embed_dim=config.embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_proj=nn.Linear(config.embed_dim, num_heads * head_dim, bias=config.q_proj_bias),
                    k_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.k_proj_bias),
                    v_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.v_proj_bias),
                    output_proj=nn.Linear(num_heads * head_dim, config.embed_dim, bias=False),
                    pos_embeddings=self.rope,
                    q_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.q_norm else None, # norm on head_dim
                    k_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.k_norm else None,
                    kv_cache=None,
                    max_seq_len=config.max_seq_len,
                    attn_dropout=config.attn_dropout,
                    attention_function=config.attention_function
                )
                mlp = qwen3_mlp(dim=config.embed_dim, hidden_dim=config.intermediate_dim)
                layer = TransformerSelfAttentionLayer(
                    attn=self_attn,
                    mlp=mlp,
                    sa_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                    mlp_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                )
                layers.append(layer)

        tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        if config.tie_word_embeddings:
            output_proj = TiedLinear(tok_embeddings)
        else:
            output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.model = TransformerDecoder(
            tok_embeddings=tok_embeddings,
            layers=layers,
            max_seq_len=config.max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
            output=output_proj,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
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
        
        # Handle TiedLinear: its weight is actually the tied_module's weight
        # TiedLinear is not an nn.Module, so it won't appear in named_modules()
        if module is None:
            # Try to get the module by attribute access
            parts = module_name.split(".")
            try:
                current = self
                for part in parts:
                    current = getattr(current, part)
                # Check if it's a TiedLinear
                if isinstance(current, TiedLinear):
                    # TiedLinear's weight is the tied_module's weight
                    module = current.tied_module
                else:
                    module = current if isinstance(current, nn.Module) else None
            except AttributeError:
                module = None
        
        if module is None:
            # If module not found, return a default lecun_normal initializer
            return lecun_normal_
        
        # Check if this is part of an Attention module
        is_attention = isinstance(parent_module, Qwen3Attention)
        
        # Check if this is part of an MLP/FeedForward module
        is_mlp = isinstance(parent_module, FeedForward)
        
        # Define initializer based on module type
        if isinstance(module, nn.Linear):
            def linear_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    if is_attention:
                        # Attention layers use xavier_uniform_
                        init.xavier_uniform_(tensor)
                    elif is_mlp:
                        # MLP layers use xavier_uniform_ for weights
                        init.xavier_uniform_(tensor)
                    else:
                        # Other Linear layers use lecun_normal_
                        lecun_normal_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    if is_mlp:
                        # MLP bias uses normal with std=1e-6
                        init.normal_(tensor, mean=0.0, std=1e-6)
                    else:
                        # Other biases use zeros
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
        
        # Default: lecun_normal initialization
        return lecun_normal_

    def get_layers_to_shard(self):
        # Return the ModuleList directly - it's iterable and supports reversed()
        # fully_shard will be applied to each individual layer, not the ModuleList itself
        return self.model.layers

    def get_checkpointable_module_classes(self):
        return {TransformerSelfAttentionLayer}

    def convert_hf_state_dict(self,
                              hf_state_dict: Dict[str, torch.Tensor],
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to a model state dictionary
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
        converted_state_dict = {}
        tie_word_embeddings = self.config.tie_word_embeddings
        skipped_keys = []
        
        for hf_key, tensor in hf_state_dict.items():
            # Skip lm_head if tie_word_embeddings is True
            if tie_word_embeddings and hf_key == "lm_head.weight":
                skipped_keys.append(hf_key)
                continue
            
            # Handle embedding layer
            if hf_key == "model.embed_tokens.weight":
                converted_key = "model.tok_embeddings.weight"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle final norm (RMSNorm uses 'scale' not 'weight')
            if hf_key == "model.norm.weight":
                converted_key = "model.norm.scale"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle output layer (lm_head)
            if hf_key == "lm_head.weight":
                converted_key = "model.output.weight"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle transformer layers
            if hf_key.startswith("model.layers."):
                # Extract layer index and remaining key
                parts = hf_key.split(".", 3)  # ["model", "layers", "{i}", "rest"]
                if len(parts) < 4:
                    # Skip keys that don't match expected format
                    skipped_keys.append(hf_key)
                    continue
                
                layer_idx = parts[2]
                rest_key = parts[3]
                
                # Handle attention weights
                if rest_key.startswith("self_attn."):
                    attn_key = rest_key.replace("self_attn.", "attn.")
                    # Map o_proj to output_proj
                    attn_key = attn_key.replace("o_proj", "output_proj")
                    # Handle q_norm and k_norm (they are RMSNorm, use 'scale' not 'weight')
                    # Hugging Face uses .weight, Muse uses .scale
                    attn_key = attn_key.replace("q_norm.weight", "q_norm.scale")
                    attn_key = attn_key.replace("k_norm.weight", "k_norm.scale")
                    converted_key = f"model.layers.{layer_idx}.{attn_key}"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                # Handle MLP weights
                # Hugging Face uses gate_proj/up_proj/down_proj
                # Muse uses w1/w3/w2 (gate_proj->w1, up_proj->w3, down_proj->w2)
                if rest_key.startswith("mlp."):
                    mlp_key = rest_key.replace("mlp.", "")
                    if mlp_key == "gate_proj.weight":
                        converted_key = f"model.layers.{layer_idx}.mlp.w1.weight"
                    elif mlp_key == "up_proj.weight":
                        converted_key = f"model.layers.{layer_idx}.mlp.w3.weight"
                    elif mlp_key == "down_proj.weight":
                        converted_key = f"model.layers.{layer_idx}.mlp.w2.weight"
                    else:
                        # Skip unknown MLP keys
                        skipped_keys.append(hf_key)
                        continue
                    converted_state_dict[converted_key] = tensor
                    continue
                
                # Handle layer norms (RMSNorm uses 'scale' not 'weight')
                if rest_key == "input_layernorm.weight":
                    converted_key = f"model.layers.{layer_idx}.sa_norm.scale"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                if rest_key == "post_attention_layernorm.weight":
                    converted_key = f"model.layers.{layer_idx}.mlp_norm.scale"
                    converted_state_dict[converted_key] = tensor
                    continue
            
            # If key doesn't match any pattern, skip it
            # This handles keys like "model.config" or other non-weight keys
            skipped_keys.append(hf_key)
        
        if skipped_keys:
            logger.warning(
                f"Skipped {len(skipped_keys)} keys during conversion. "
                f"First few: {skipped_keys[:10]}"
            )
        
        logger.info(
            f"Converted {len(converted_state_dict)} keys from "
            f"{len(hf_state_dict)} Hugging Face keys"
        )
        
        return converted_state_dict
