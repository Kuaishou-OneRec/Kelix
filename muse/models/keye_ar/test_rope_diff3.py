import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


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
        self.attention_scaling = 1.0  # 强制缩放因子为1
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids, save_intermediates: bool = False) -> tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        intermediates = {} if save_intermediates else None
        
        # Step 1: 扩展inv_freq
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        inv_freq_expanded0 = inv_freq_expanded
        
        inv_freq_expanded = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64)[: (self.head_dim // 2)].float() / self.head_dim))
        self.inv_freq = inv_freq_expanded
        inv_freq_expanded = inv_freq_expanded[None,:,None]
        if save_intermediates:
            intermediates["inv_freq_expanded"] = inv_freq_expanded.clone()
            intermediates["inv_freq_original"] = self.inv_freq.clone()
            intermediates["attention_scaling"] = self.attention_scaling
        
        # Step 2: 扩展position_ids
        position_ids_expanded = position_ids[:, None, :].float()
        if save_intermediates:
            intermediates["position_ids_expanded"] = position_ids_expanded.clone()
        print(f"self.attention_scaling={self.attention_scaling}")
        
        # Step 3: 计算freqs（矩阵乘法+转置）
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs_before_trans = (inv_freq_expanded.float() @ position_ids_expanded.float())
            freqs = freqs_before_trans.transpose(1, 2)
            if save_intermediates:
                intermediates["freqs_before_trans"] = freqs_before_trans.clone()
                intermediates["freqs"] = freqs.clone()
            
            # Step 4: 拼接生成emb
            emb = torch.cat((freqs, freqs), dim=-1)
            if save_intermediates:
                intermediates["emb"] = emb.clone()
            
            # Step 5: 计算cos/sin并缩放
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            if save_intermediates:
                intermediates["cos_before_scaling"] = emb.cos().clone()
                intermediates["sin_before_scaling"] = emb.sin().clone()
                intermediates["cos_final"] = cos.clone()
                intermediates["sin_final"] = sin.clone()
        
        cos_out = cos.to(dtype=x.dtype)
        sin_out = sin.to(dtype=x.dtype)
        
        if save_intermediates:
            return cos_out, sin_out, intermediates
        return cos_out, sin_out
    
    def rope_init(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

# 代码2：RotaryPositionalEmbeddings（逐步骤对比Qwen的中间结果）
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 1000_000, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.intermediates = {}  # 保存自己的中间结果
        self.dtype = dtype  # 新增：控制cache的存储类型
        self.rope_init()

    def rope_init(self):
        # Step 1: 计算theta（对应Qwen的inv_freq）
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64)[: (self.dim // 2)].float() / self.dim))
        self.intermediates["theta"] = theta.clone()
        self.register_buffer("theta", theta.to(dtype=self.dtype), persistent=False)
        
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Step 2: 生成seq_idx（对应Qwen的position_ids）
        seq_idx = torch.arange(max_seq_len, dtype=torch.float32, device=self.theta.device)
        self.intermediates["seq_idx"] = seq_idx.clone()
        
        # Step 3: 计算idx_theta（外积，对应Qwen的freqs_before_trans）
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta.float()).float()
        self.intermediates["idx_theta"] = idx_theta.clone()
        
        # Step 4: 拼接生成freqs（对应Qwen的emb）
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        self.intermediates["freqs_concat"] = freqs.clone()
        
        # Step 5: 计算cos/sin（无缩放）
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.intermediates["cos_final"] = cos.clone()
        self.intermediates["sin_final"] = sin.clone()
        
        # Step 6: 构建cache（按指定类型存储）
        cache = torch.stack([cos.to(dtype=self.dtype), sin.to(dtype=self.dtype)], dim=-1)
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
            print(f"  Qwen前5值: {q_inv_freq[:5].float()}")
            print(f"  Custom前5值: {c_theta[:5].float()}")
        
        # 对比步骤2: freqs_before_trans (Qwen) vs idx_theta (Custom)
        print("\n【步骤2】Qwen.freqs_before_trans vs Custom.idx_theta")
        q_freqs_bt = qwen_intermediates["freqs_before_trans"][0, :, :seq_len].T.cpu()
        c_idx_theta = self.intermediates["idx_theta"][:seq_len, :].cpu()
        diff = torch.abs(q_freqs_bt - c_idx_theta).mean().item()
        is_same = diff < 1e-10
        print(f"  形状: Qwen={q_freqs_bt.shape}, Custom={c_idx_theta.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致: {'✅' if is_same else '❌'}")
        
        # 对比步骤3: Qwen.emb vs Custom.freqs_concat
        print("\n【步骤3】Qwen.emb vs Custom.freqs_concat")
        q_emb = qwen_intermediates["emb"][0, :seq_len, :].cpu()
        c_freqs_concat = self.intermediates["freqs_concat"][:seq_len, :].cpu()
        diff = torch.abs(q_emb - c_freqs_concat).mean().item()
        is_same = diff < 1e-10
        print(f"  形状: Qwen={q_emb.shape}, Custom={c_freqs_concat.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致: {'✅' if is_same else '❌'}")
        
        # 对比步骤4: Qwen.cos_before_scaling vs Custom.cos_final
        print("\n【步骤4】Qwen.cos_before_scaling (无缩放) vs Custom.cos_final")
        q_cos_bs = qwen_intermediates["cos_before_scaling"][0, :seq_len, :].cpu()
        c_cos = self.intermediates["cos_final"][:seq_len, :].cpu()
        diff = torch.abs(q_cos_bs - c_cos).mean().item()
        is_same = diff < 1e-10
        print(f"  形状: Qwen={q_cos_bs.shape}, Custom={c_cos.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致: {'✅' if is_same else '❌'}")
        
        # 对比步骤5: Qwen.cos_final (带缩放) vs Custom.cos_final
        print("\n【步骤5】Qwen.cos_final (带缩放) vs Custom.cos_final")
        q_cos_final = qwen_intermediates["cos_final"][0, :seq_len, :].cpu()
        c_cos_final = self.intermediates["cos_final"][:seq_len, :].cpu()
        diff = torch.abs(q_cos_final - c_cos_final).mean().item()
        is_same = diff < 1e-6
        print(f"  Qwen的缩放因子: {qwen_intermediates['attention_scaling']}")
        print(f"  形状: Qwen={q_cos_final.shape}, Custom={c_cos_final.shape}")
        print(f"  均值误差: {diff:.10f}")
        print(f"  是否一致 (允许1e-6误差): {'✅' if is_same else '❌'}")
        
        # 最终提取Custom的cos/sin
        custom_cache = self.cache[:seq_len]
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

def debug_dtype_impact(qwen_cos, qwen_sin, custom_cos, custom_sin, seq_len):
    """新增：调试数据类型对误差的影响"""
    print("\n" + "="*80)
    print("数据类型精度影响分析")
    print("="*80)
    
    # 1. 对比float32下的结果（无精度损失）
    print("\n【1】float32 精度对比（无转换损失）")
    cos1_float32 = qwen_cos.cpu().float()
    cos2_float32 = custom_cos.cpu().float()
    sin1_float32 = qwen_sin.cpu().float()
    sin2_float32 = custom_sin.cpu().float()
    
    cos_diff_float32 = torch.abs(cos1_float32 - cos2_float32).mean().item()
    sin_diff_float32 = torch.abs(sin1_float32 - sin2_float32).mean().item()
    is_match_float32 = torch.allclose(cos1_float32, cos2_float32, atol=1e-6) and torch.allclose(sin1_float32, sin2_float32, atol=1e-6)
    
    print(f"  cos均值误差: {cos_diff_float32:.10f}")
    print(f"  sin均值误差: {sin_diff_float32:.10f}")
    print(f"  float32下是否一致: {'✅' if is_match_float32 else '❌'}")
    
    # 2. 对比bfloat16下的结果（有精度损失）
    print("\n【2】bfloat16 精度对比（模拟转换损失）")
    cos1_bf16 = cos1_float32.to(torch.bfloat16)
    cos2_bf16 = cos2_float32.to(torch.bfloat16)
    sin1_bf16 = sin1_float32.to(torch.bfloat16)
    sin2_bf16 = sin2_float32.to(torch.bfloat16)
    
    cos_diff_bf16 = torch.abs(cos1_bf16 - cos2_bf16).mean().item()
    sin_diff_bf16 = torch.abs(sin1_bf16 - sin2_bf16).mean().item()
    is_match_bf16 = torch.allclose(cos1_bf16, cos2_bf16, atol=1e-6) and torch.allclose(sin1_bf16, sin2_bf16, atol=1e-6)
    
    print(f"  cos均值误差: {cos_diff_bf16:.10f}")
    print(f"  sin均值误差: {sin_diff_bf16:.10f}")
    print(f"  bfloat16下是否一致: {'✅' if is_match_bf16 else '❌'}")
    
    # 3. 打印具体数值差异（直观展示bfloat16精度损失）
    print("\n【3】具体数值对比（第0个位置，前5个维度）")
    print(f"  float32 cos1: {cos1_float32[0,0,:5]}")
    print(f"  float32 cos2: {cos2_float32[0,0,:5]}")
    print(f"  bfloat16 cos1: {cos1_bf16[0,0,:5]}")
    print(f"  bfloat16 cos2: {cos2_bf16[0,0,:5]}")
    print(f"  bfloat16转换后的差值: {torch.abs(cos1_bf16[0,0,:5] - cos2_bf16[0,0,:5])}")
    
    # 4. 验证bfloat16的精度极限
    print("\n【4】bfloat16 精度极限验证")
    bf16_eps = torch.finfo(torch.bfloat16).eps  # bfloat16的最小精度
    print(f"  bfloat16最小可表示差值 (eps): {bf16_eps:.10f}")
    print(f"  你的误差阈值 (1e-6): {1e-6:.10f}")
    print(f"  结论: bfloat16精度({bf16_eps:.10f}) < 阈值(1e-6)，无法满足1e-6的误差要求")

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

    # 设备设置（统一用CPU）
    device = "cpu"
    print(f"使用设备: {device} (CPU避免cuda精度干扰)")

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
        # 1. 实例化QwenRoPE（float32计算）
        qwen_rope = Qwen3RotaryEmbedding(config, device=device).to(device)  # 不转bfloat16，保持float32
        position_ids = torch.arange(seq_len).expand(batch_size, -1).to(device)
        x_dummy = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
        cos1, sin1, qwen_intermediates = qwen_rope.forward(x_dummy, position_ids, save_intermediates=True)

        # 2. 实例化CustomRoPE（先以float32存储cache，避免提前精度损失）
        custom_rope = RotaryPositionalEmbeddings(
            dim=head_dim, 
            max_seq_len=4096, 
            base=1000000,
            dtype=torch.float32  # 新增：cache用float32存储
        ).to(device)
        
        # 逐步骤对比
        custom_rope.compare_with_qwen(qwen_intermediates, seq_len, device)

        # 3. 提取Custom的cos/sin
        custom_cache = custom_rope.cache[position_ids]
        cos2, sin2 = custom_cache[..., 0], custom_cache[..., 1]

        # 4. 新增：调试数据类型对误差的影响
        debug_dtype_impact(cos1, sin1, cos2, sin2, seq_len)

        # 5. 原测试逻辑验证（分别测试float32和bfloat16）
        print("\n" + "="*80)
        print("原测试逻辑验证")
        print("="*80)
        print(f"cos1={cos1.dtype}, cos2={cos2.dtype}")
        
        # float32下的对比
        is_match_float32 = torch.allclose(cos1.float(), cos2.float(), atol=1e-6) and torch.allclose(sin1.float(), sin2.float(), atol=1e-6)
        # bfloat16下的对比
        is_match_bf16 = torch.allclose(cos1.bfloat16(), cos2.bfloat16(), atol=1e-6) and torch.allclose(sin1.bfloat16(), sin2.bfloat16(), atol=1e-6)
        
        print(f"float32下结果: {'无误差' if is_match_float32 else '有误差'}")
        print(f"bfloat16下结果: {'无误差' if is_match_bf16 else '有误差'}")