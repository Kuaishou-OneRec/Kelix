import torch
import torch.nn as nn
import numpy as np

# 定义第一个 Qwen3RMSNorm（修正了明显的类型转换错误）
class Qwen3RMSNorm_V1(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # 转换为 float32 计算
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 修正错误：原代码 to(hidden_states) 无效，应转换回输入 dtype
        hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# 定义第二个 Qwen3RMSNorm
class Qwen3RMSNorm_V2(nn.Module):
    """
    Root Mean Square Normalization in fp32.

    See: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        dim = hidden_size
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.weight

# 测试函数
def test_rmsnorm_bf16():
    # 1. 配置测试参数
    hidden_size = 4096  # 模拟大模型的隐藏层维度
    batch_size = 8
    seq_len = 128
    eps = 1e-6
    
    # 设置设备（优先GPU，CPU也可运行但bf16仅部分CPU支持）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")
    
    # 2. 创建两个RMSNorm实例，并将参数转换为bf16
    norm_v1 = Qwen3RMSNorm_V1(hidden_size, eps).to(device, dtype=torch.bfloat16)
    norm_v2 = Qwen3RMSNorm_V2(hidden_size, eps).to(device, dtype=torch.bfloat16)
    
    # 确保两个实例的weight参数完全一致（消除参数初始化差异）
    norm_v1.weight.data = norm_v2.weight.data.clone()
    
    # 3. 生成bf16类型的测试输入（模拟模型中间层输出）
    # 随机输入，均值0，方差1，符合模型输入的常见分布
    torch.manual_seed(42)  # 固定随机种子，保证可复现
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, 
        device=device, dtype=torch.bfloat16
    )
    
    # 4. 前向传播（关闭梯度计算，加速测试）
    with torch.no_grad():
        output_v1 = norm_v1(input_tensor)
        output_v2 = norm_v2(input_tensor)
    
    # 5. 计算数值差异
    # 绝对误差（逐元素）
    abs_error = torch.abs(output_v1 - output_v2)
    # 相对误差（避免除零，加小值）
    rel_error = abs_error / (torch.abs(output_v2) + 1e-10)
    
    # 统计指标
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    # 6. 打印结果
    print("\n=== 数值差异统计（bf16 dtype）===")
    print(f"最大绝对误差: {max_abs_error:.8f}")
    print(f"平均绝对误差: {mean_abs_error:.8f}")
    print(f"最大相对误差: {max_rel_error:.8f}")
    print(f"平均相对误差: {mean_rel_error:.8f}")
    
    # 打印部分元素对比（直观查看）
    print("\n=== 前3个元素对比 ===")
    print(f"V1 输出[0,0,0]: {output_v1[0,0,0].item():.8f}")
    print(f"V2 输出[0,0,0]: {output_v2[0,0,0].item():.8f}")
    print(f"误差[0,0,0]: {abs_error[0,0,0].item():.8f}")
    
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error
    }

# 执行测试
if __name__ == "__main__":
    # 检查bf16支持
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        if not torch.cuda.is_available() and not torch.backends.cpu.bf16:
            print("警告：当前设备不支持bf16，将使用float16模拟测试！")
            torch.set_default_dtype(torch.float16)
        test_rmsnorm_bf16()