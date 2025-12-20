import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import os


# 会漏fp32的被转成bf16精度

# 代码1：Qwen3RotaryEmbedding（保存中间结果）
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
        self.config.hidden_size = 4096
        self.config.num_attention_heads = 32
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        # 初始化inv_freq
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids) -> tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        
        # Step 1: 扩展inv_freq
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        inv_freq_expanded0 = inv_freq_expanded
        
        if os.environ.get("Qwen3RMSNorm_fp32", "1") == "1":
            inv_freq_expanded = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64)[: (self.head_dim // 2)].float() / self.head_dim)).to(x.device)
            self.inv_freq = inv_freq_expanded #  这里暂时对齐custom的实现，后续可以去掉
        inv_freq_expanded = inv_freq_expanded[None,:,None]
        # Step 2: 扩展position_ids
        position_ids_expanded = position_ids[:, None, :].float()

        # Step 3: 计算freqs（矩阵乘法+转置）
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float())
            
            # Step 4: 拼接生成emb
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Step 5: 计算cos/sin并缩放
            if os.environ.get("Qwen3RMSNorm_fp32", "1") == "1":
                cos = emb.cos().bfloat16() * self.attention_scaling
                sin = emb.sin().bfloat16() * self.attention_scaling
            else:
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
        
        cos_out = cos.to(dtype=x.dtype)
        sin_out = sin.to(dtype=x.dtype)
        
        return cos_out, sin_out
    
    def rope_init(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

# 代码2：RotaryPositionalEmbeddings（逐步骤对比Qwen的中间结果）
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 1000_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.intermediates = {}  # 保存自己的中间结果
        self.rope_init()

    def rope_init(self):
        # Step 1: 计算theta（对应Qwen的inv_freq）
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64)[: (self.dim // 2)].float() / self.dim))
        self.intermediates["theta"] = theta.clone()
        self.register_buffer("theta", theta, persistent=False)
        
        self.build_rope_cache(self.max_seq_len)
        print(f"self.cache_ddd000", self.cache.dtype)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Step 2: 生成seq_idx（对应Qwen的position_ids）
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        self.intermediates["seq_idx"] = seq_idx.clone()
        
        # Step 3: 计算idx_theta（外积，对应Qwen的freqs_before_trans）
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        self.intermediates["idx_theta"] = idx_theta.clone()
        
        # Step 4: 拼接生成freqs（对应Qwen的emb）
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        self.intermediates["freqs_concat"] = freqs.clone()
        
        # Step 5: 计算cos/sin（无缩放）
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.intermediates["cos_final"] = cos.clone()
        self.intermediates["sin_final"] = sin.clone()
        
        # Step 6: 构建cache
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

    def compare_with_qwen(self, qwen_intermediates: Dict, seq_len: int, device: str):
        """核心对比函数：逐步骤对比和Qwen的中间结果"""
        print("\n" + "="*80)
        print("开始逐步骤对比 Qwen3Rotary vs RotaryPositionalEmbeddings")
        print("="*80)
        
        # 对比步骤1: inv_freq (Qwen) vs theta (Custom)
        print("\n【步骤1】Qwen.inv_freq vs Custom.theta")
        q_inv_freq = qwen_intermediates["inv_freq_original"].cpu()
        c_theta = self.intermediates["theta"].cpu()
        diff = torch.abs(q_inv_freq - c_theta).mean().item()
        is_same = diff < 1e-10
        print(f"  形状: Qwen={q_inv_freq.shape}, Custom={c_theta.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致: {'✅' if is_same else '❌'}")
        if not is_same:
            print(f"  Qwen前5值: {q_inv_freq[:5].float().numpy()}")
            print(f"  Custom前5值: {c_theta[:5].float().numpy()}")
        
        # 对比步骤2: freqs_before_trans (Qwen) vs idx_theta (Custom)
        print("\n【步骤2】Qwen.freqs_before_trans vs Custom.idx_theta")
        # Qwen: [1,64,66] → 转置为[66,64]；Custom: [4096,64] → 取前66行
        q_freqs_bt = qwen_intermediates["freqs_before_trans"][0, :, :seq_len].T.cpu()  # [66,64]
        c_idx_theta = self.intermediates["idx_theta"][:seq_len, :].cpu()  # [66,64]
        diff = torch.abs(q_freqs_bt - c_idx_theta).mean().item()
        is_same = diff < 1e-10
        print(f"  形状: Qwen={q_freqs_bt.shape}, Custom={c_idx_theta.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致: {'✅' if is_same else '❌'}")
        
        # 对比步骤3: emb (Qwen) vs freqs_concat (Custom)
        print("\n【步骤3】Qwen.emb vs Custom.freqs_concat")
        q_emb = qwen_intermediates["emb"][0, :seq_len, :].cpu()  # [66,128]
        c_freqs_concat = self.intermediates["freqs_concat"][:seq_len, :].cpu()  # [66,128]
        diff = torch.abs(q_emb - c_freqs_concat).mean().item()
        is_same = diff < 1e-10
        print(f"  形状: Qwen={q_emb.shape}, Custom={c_freqs_concat.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致: {'✅' if is_same else '❌'}")
        
        # 对比步骤4: cos_before_scaling (Qwen) vs cos_final (Custom)
        print("\n【步骤4】Qwen.cos_before_scaling (无缩放) vs Custom.cos_final")
        q_cos_bs = qwen_intermediates["cos_before_scaling"][0, :seq_len, :].cpu()  # [66,128]
        c_cos = self.intermediates["cos_final"][:seq_len, :].cpu()  # [66,128]
        diff = torch.abs(q_cos_bs - c_cos).mean().item()
        is_same = diff < 1e-10
        print(f"  形状: Qwen={q_cos_bs.shape}, Custom={c_cos.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致: {'✅' if is_same else '❌'}")
        
        # 对比步骤5: cos_final (Qwen，带缩放) vs Custom.cos_final
        print("\n【步骤5】Qwen.cos_final (带缩放) vs Custom.cos_final")
        q_cos_final = qwen_intermediates["cos_final"][0, :seq_len, :].cpu()  # [66,128]
        c_cos_final = self.intermediates["cos_final"][:seq_len, :].cpu()  # [66,128]
        diff = torch.abs(q_cos_final - c_cos_final).mean().item()
        is_same = diff < 1e-6
        print(f"  Qwen的缩放因子: {qwen_intermediates['attention_scaling']}")
        print(f"  形状: Qwen={q_cos_final.shape}, Custom={c_cos_final.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致 (允许1e-6误差): {'✅' if is_same else '❌'}")
        
        # 最终提取Custom的cos/sin（和测试逻辑一致）
        custom_cache = self.cache[:seq_len]
        print(f"self.cache_ddd", self.cache.dtype)
        cos2 = custom_cache[..., 0].cpu()
        sin2 = custom_cache[..., 1].cpu()
        q_cos_out = qwen_intermediates["cos_final"][0].cpu()
        q_sin_out = qwen_intermediates["sin_final"][0].cpu()
        
        # 最终对比
        print("\n【最终结果】测试逻辑中的cos/sin对比")
        cos_diff = torch.abs(q_cos_out - cos2).mean().item()
        sin_diff = torch.abs(q_sin_out - sin2).mean().item()
        print(f"  cos均值误差: {cos_diff:.10f}")
        print(f"  sin均值误差: {sin_diff:.10f}")
        print(f"  整体是否一致 (1e-6): {'✅' if (cos_diff < 1e-6 and sin_diff < 1e-6) else '❌'}")
        print("="*80 + "\n")
        import IPython
        IPython.embed()

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
    batch_size = 1
    seq_len = 66

    # 设备设置（统一用CPU避免cuda精度问题干扰）
    device = "cpu"
    print(f"使用设备: {device} (CPU避免cuda精度干扰)")

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        # 1. 实例化QwenRoPE并运行，保存所有中间结果
        qwen_rope = Qwen3RotaryEmbedding(config, device=device).to(device).bfloat16()
        position_ids = torch.arange(seq_len).expand(batch_size, -1).to(device)
        x_dummy = torch.randn(batch_size, seq_len, head_dim, device=device)
        # 运行forward并保存中间结果
        cos1, sin1, qwen_intermediates = qwen_rope.forward(x_dummy, position_ids, save_intermediates=True)

        # 2. 实例化CustomRoPE并逐步骤对比Qwen的中间结果
        custom_rope = RotaryPositionalEmbeddings(
            dim=head_dim, 
            max_seq_len=4096, 
            base=1000000  # 和Qwen的theta对齐
        ).to(device).bfloat16()
        # 核心：调用对比函数，逐步骤校验
        custom_rope.compare_with_qwen(qwen_intermediates, seq_len, device)

        # 3. 原测试逻辑验证
        custom_cache = custom_rope.cache[position_ids]
        cos2, sin2 = custom_cache[..., 0], custom_cache[..., 1]
        # cos1=torch.float32, cos2=torch.bfloat16

        print(f"cos1={cos1.dtype}, cos2={cos2.dtype}")
        is_match = torch.allclose(cos1.bfloat16(), cos2.bfloat16(), atol=1e-6) and torch.allclose(sin1.float(), sin2.float(), atol=1e-6)
        print(f"\n原测试逻辑最终结果: {'有误差' if not is_match else '无误差'}")