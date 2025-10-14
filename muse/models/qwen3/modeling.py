from muse.models.base import Model
from muse.config import Qwen3Config
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.attention import Qwen3Attention
from muse.layers.transformer import TransformerDecoder
from muse.layers.norm import RMSNorm
from muse.layers.linear import TiedLinear
import torch.nn as nn
from muse.models.qwen3._layers import qwen3_mlp

class Qwen3Model(Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads if config.num_key_value_heads else config.num_attention_heads

        rope = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta)

        layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self_attn = Qwen3Attention(
                embed_dim=config.hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(config.hidden_size, num_heads * head_dim, bias=q_proj_bias),
                k_proj=nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=k_proj_bias),
                v_proj=nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=v_proj_bias),
                output_proj=nn.Linear(num_heads * head_dim, config.hidden_size, bias=False),
                pos_embeddings=rope,
                q_norm=RMSNorm(dim=head_dim, eps=config.rms_norm_eps), # norm on head_dim
                k_norm=RMSNorm(dim=head_dim, eps=config.rms_norm_eps),
                kv_cache=None,
                max_seq_len=config.max_position_embeddings,
                attn_dropout=attn_dropout,
            )
            mlp = qwen3_mlp(dim=config.hidden_size, hidden_dim=config.intermediate_size)
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps),
                mlp_norm=RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps),
            )
            layers.append(layer)    

        tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        if tie_word_embeddings:
            output_proj = TiedLinear(tok_embeddings)
        else:
            output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.model = TransformerDecoder(
            tok_embeddings=tok_embeddings,
            layers=layers,
            max_seq_len=config.max_position_embeddings,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            output=output_proj,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)