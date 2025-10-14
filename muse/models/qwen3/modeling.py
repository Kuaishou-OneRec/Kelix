from muse.models.base import Model
from muse.config import Qwen3Config

class Qwen3Model(Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        head_dim = head_dim or embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads

        rope = Qwen2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

        layers = nn.ModuleList()
        for _ in range(num_layers):
            self_attn = Qwen3Attention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=q_proj_bias),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=k_proj_bias),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=v_proj_bias),
                output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
                pos_embeddings=rope,
                q_norm=RMSNorm(dim=head_dim, eps=config.rms_norm_eps) if q_norm else None, # norm on head_dim
                k_norm=RMSNorm(dim=head_dim, eps=config.rms_norm_eps) if k_norm else None,
                kv_cache=None,
                max_seq_len=config.max_position_embeddings,
                attn_dropout=attn_dropout,
            )
            mlp = qwen3_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
                mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            )
            layers.append(layer)

        tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        if tie_word_embeddings:
            output_proj = TiedLinear(tok_embeddings)
        else:
            output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.model = TransformerDecoder(
            tok_embeddings=tok_embeddings,
            layers=layers,
            max_seq_len=config.max_position_embeddings,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=RMSNorm(embed_dim, eps=config.rms_norm_eps),
            output=output_proj,
        )

    def forward(self, *args, **kwargs):
        pass