from typing import Dict
import torch
import logging

from muse.models.base import Model
from muse.config import Qwen3Config
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.transformer import TransformerDecoder, TransformerSelfAttentionLayer
from muse.layers.rms_norm import RMSNorm
from muse.layers.linear import TiedLinear
import torch.nn as nn
from muse.models.qwen3._layers import qwen3_mlp, Qwen3Attention

# Import will be done when muse.models is imported, avoiding circular import
# The actual registration happens in __init__.py after import

logger = logging.getLogger(__name__)

class Qwen3Model(Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        head_dim = config.head_dim or config.embed_dim // config.num_heads
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads if config.num_kv_heads else num_heads

        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=config.max_seq_len, base=config.rope_base)

        layers = nn.ModuleList()
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
    
    def get_layers_to_shard(self):
        return [self.model.layers]

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
