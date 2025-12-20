import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


# 代码1：Qwen3RotaryEmbedding
class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.device = device
        self.config.rope_theta = 1000000
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    def rope_init(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

# 代码2：RotaryPositionalEmbeddings
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 1000_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cache = torch.stack([cos, sin], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, seq_len, n_h, h_d = x.shape
        if input_pos is None:
            rope_cache = self.cache[:seq_len]
            rope_cache = rope_cache.unsqueeze(0).unsqueeze(2)
        else:
            rope_cache = self.cache[input_pos]
            rope_cache = rope_cache.unsqueeze(2)
        cos = rope_cache[..., 0].to(dtype=x.dtype)
        sin = rope_cache[..., 1].to(dtype=x.dtype)
        x_rotated = self.rotate_half(x)
        x_out = (x * cos) + (x_rotated * sin)
        return x_out

# 测试逻辑
if __name__ == "__main__":
    # 模拟配置
    class MockConfig:
        def __init__(self):
            self.rope_scaling = None
            self.max_position_embeddings = 4096

    # 初始化参数
    config = MockConfig()
    head_dim = 128
    batch_size = 2
    seq_len = 10

    # 实例化两个RoPE
    qwen_rope = Qwen3RotaryEmbedding(config, device="cpu")
    custom_rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=4096)

    # 生成测试数据
    position_ids = torch.arange(seq_len).expand(batch_size, -1)
    x_dummy = torch.randn(batch_size, seq_len, head_dim, device="cpu")

    # 获取cos/sin
    cos1, sin1 = qwen_rope.forward(x_dummy, position_ids)
    custom_cache = custom_rope.cache[position_ids]
    cos2, sin2 = custom_cache[..., 0], custom_cache[..., 1]

    # 比较（允许1e-6浮点误差）
    is_match = torch.allclose(cos1, cos2, atol=1e-6) and torch.allclose(sin1, sin2, atol=1e-6)
    print("有误差" if not is_match else "无误差")