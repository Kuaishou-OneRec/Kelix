from typing import Dict
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

class Qwen3Model(Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        head_dim = config.head_dim or config.embed_dim // config.num_heads
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads if config.num_kv_heads else num_heads

        rope = RotaryPositionalEmbeddings(
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
                pos_embeddings=rope,
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
            A dictionary of model state.
        """
        return hf_state_dict
