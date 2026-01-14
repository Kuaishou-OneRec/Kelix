from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from easydict import EasyDict as edict

# = a================================================================
# Block 1: Your High-Quality, Optimized Loss Function
# ===================================================================

class CrossEntropyLoss(nn.Module):
    """
    An efficient CrossEntropyLoss module that avoids redundant calculations.
    It first computes per-token losses and then manually applies the reduction.
    (Based on the user-provided, superior implementation).
    """
    def __init__(self,
                 ignore_index: int = -100,
                 return_token_loss: bool = False,
                 shift_labels: bool = True,
                 reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.return_token_loss = return_token_loss
        self.reduction = reduction
        self.shift_labels = shift_labels

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits (torch.Tensor): A single tensor of shape (..., vocab_size).
            labels (torch.Tensor): Ground truth labels.
        """
        vocab_size = logits.shape[-1]
        
        if self.shift_labels:
          logits = logits[:, :-1, :]
          labels = labels[:, 1:]
          
        # Reshape for cross-entropy calculation
        logits_flat = logits.float().reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Step 1: Compute per-token loss. This is the base for all other calculations.
        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        
        # Step 2: Manually apply reduction to get the final loss.
        loss = per_token_loss.sum()
        if self.reduction == "mean":
            # Ensure we divide by the number of valid (non-ignored) tokens
            total_elements = (labels_flat != self.ignore_index).sum()
            if total_elements > 0:
                loss /= total_elements
            else: # Handle case where all tokens are ignored
                loss.zero_()

        # Return what's requested
        if self.return_token_loss:
            return loss, per_token_loss
        
        return loss


# ===================================================================
# Block 2: The Controller Class for Memory-Efficient Training
# ===================================================================

class ChunkedLossComputer:
    """
    内存高效的两阶段反向传播控制器

    【核心功能】
    解决大型语言模型(LLM)中lm_head过大导致的显存不足问题，通过分块计算实现内存优化。

    【实现原理】
    采用两阶段计算策略：
    1. 将输入序列分成多个小批次(minibatch)，对每个批次单独计算logits和损失。
    2. 手动计算每个批次的梯度并累加，而不是一次性计算所有梯度。
    3. 最后将累加的梯度应用到模型参数和输入上。
    
    【内存优化效果】
    通过分块处理，避免了一次性为整个序列分配巨大的中间张量，显著减少GPU内存使用峰值。

    注意：
    返回的两个loss都是bp过+detach过的
    请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!!
    请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!!
    请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!!
    """
    def __init__(self, lm_head: nn.Module, loss_fn: nn.Module, minibatch_size: int, shift_labels: bool = True):
        """
        初始化两阶段梯度计算器
        
        参数:
            lm_head: 语言模型的输出层，通常是nn.Linear。也可以是任意与loss_fn适配的nn.Module。
            loss_fn: 损失函数。该函数必须返回一个元组 (avg_loss, per_token_loss)。
            minibatch_size: 每个分块的大小，用于控制内存使用。
            shift_labels: 是否偏移标签(用于自回归模型)。
        """
        if not isinstance(lm_head, nn.Module) or not isinstance(loss_fn, nn.Module):
            raise TypeError("lm_head和loss_fn必须是nn.Module的实例")
            
        self.lm_head = lm_head
        self.loss_fn = loss_fn
        self.minibatch_size = minibatch_size
        self.shift_labels = shift_labels
        self.loss_info = {}

    def forward_and_backward(self, input: torch.Tensor, labels: torch.Tensor, loss_fn_args: dict = {}, tokenwise_loss_weight=None):
        """
        执行两阶段的前向和反向传播过程
        
        参数:
            input: 输入张量，形状通常为[batch_size, seq_len, hidden_dim]。
            labels: 标签张量，形状通常为[batch_size, seq_len]。
        
        返回:
            tuple[torch.Tensor, torch.Tensor]:
                - final_avg_loss: 整个输入的平均损失值。
                - per_token_loss: 整个输入的per-token损失。

        注意：
        返回的两个loss都是bp过+detach过的
        请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!! 若有必要，请你把loss计算逻辑写到loss_fn中
        请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!! 若有必要，请你把loss计算逻辑写到loss_fn中
        请不要直接使用forward_and_backward返回的两个loss进行任何需要bp的操作，任何的需要bp的操作都是无效的!!!! 若有必要，请你把loss计算逻辑写到loss_fn中
        """
        self.ticker.tick("lm_head")
        params = list(self.lm_head.parameters())
        grad_accs = [torch.zeros_like(p) for p in params]
        grad_input_full = torch.zeros_like(input)

        total_loss_sum_for_reporting = torch.tensor(0.0, device=input.device)

        # if tokenwise_loss_weight is not None:
        #     tokenwise_loss_weight = tokenwise_loss_weight * tokenwise_loss_weight.numel() / tokenwise_loss_weight.sum()

        all_per_token_losses = []

        seq_len = input.size(1)
        
        # 计算总有效元素数量
        labels_to_count = labels[:, 1:] if self.shift_labels else labels
        total_elements = (labels_to_count != getattr(self.loss_fn, 'ignore_index', -100)).sum()
        
        if total_elements.item() == 0:
            return torch.tensor(0.0, device=input.device), None

        # 第一阶段: 分块计算前向和梯度累加
        for i in range(0, seq_len, self.minibatch_size):
            start, end = i, min(i + self.minibatch_size, seq_len)
            input_chunk = input[:, start:end, :].detach().requires_grad_()

            if tokenwise_loss_weight is not None:
                assert tokenwise_loss_weight.shape == labels.shape, f"tokenwise_loss_weight.shape={tokenwise_loss_weight.shape}, labels.shape={labels.shape}"
                loss_weight_chunk = tokenwise_loss_weight[:, start:end]
                loss_weight_chunk_flat = loss_weight_chunk.reshape(-1)
            else:
                loss_weight_chunk_flat = 1
            
            logits_chunk = self.lm_head(input_chunk)

            if self.shift_labels:
                label_start, label_end = start + 1, end + 1
                labels_chunk = labels[:, label_start:label_end]
                # 确保logits和labels长度匹配
                if logits_chunk.size(1) > labels_chunk.size(1):
                    logits_chunk = logits_chunk[:, :labels_chunk.size(1), :]
            else:
                labels_chunk = labels[:, start:end]

            if labels_chunk.numel() == 0:
                continue

            logits_flat = logits_chunk.reshape(-1, self.lm_head.out_features)
            labels_flat = labels_chunk.reshape(-1)            

            # === 核心改动: 一次调用获取avg_loss和per_token_loss ===
            loss_chunk_avg, per_token_loss_chunk = self.loss_fn(logits_flat, labels_flat, loss_weight_chunk=loss_weight_chunk_flat, **loss_fn_args)


            # 为了反向传播，我们需要损失的和 (sum)，而不是平均值 (avg)
            # 因此我们用 avg_loss * 有效token数 来重构 sum_loss
            valid_tokens_in_chunk = (labels_flat != getattr(self.loss_fn, 'ignore_index', -100)).sum()
            
            if valid_tokens_in_chunk.item() == 0:
                all_per_token_losses.append(per_token_loss_chunk.detach())
                continue # 如果当前块没有有效token，则跳过
            

            loss_chunk_sum = loss_chunk_avg * valid_tokens_in_chunk

            # 手动计算梯度
            # 只对requires_grad=True的参数计算梯度
            tensors_to_grad = [p for p in params if p.requires_grad] + [input_chunk]
            grads = torch.autograd.grad(outputs=loss_chunk_sum, inputs=tensors_to_grad, retain_graph=False)
        
            # 累加梯度 - 只更新需要梯度的参数
            grad_idx = 0
            for j in range(len(params)):
                if params[j].requires_grad:
                    grad_accs[j] += grads[grad_idx]
                    grad_idx += 1
            grad_input_full[:, start:end, :] = grads[grad_idx]  # input_chunk的梯度在最后

            # 累加损失总和，用于最终计算总平均损失
            total_loss_sum_for_reporting += loss_chunk_sum.detach()
            
            # 存储每个token的损失 (移至CPU以节省GPU内存)
            all_per_token_losses.append(per_token_loss_chunk.detach())
        
        # 第二阶段: 应用累加的梯度
        for j, p in enumerate(params):
            if p.requires_grad:
                p.grad = grad_accs[j] / total_elements

        input.backward(gradient=grad_input_full / total_elements)
        # 计算最终的平均损失
        final_avg_loss = (total_loss_sum_for_reporting / total_elements).detach()
        per_token_loss = torch.cat(all_per_token_losses) if all_per_token_losses else None
        final_avg_loss.requires_grad = True

        self.loss_info = {
            'loss': final_avg_loss,
            'per_token_loss': per_token_loss
        }
        return final_avg_loss, per_token_loss

# ===================================================================
# Block 3: The Full Demonstration
# ===================================================================

def format_mem(b):
    return f"{b / 1024**3:.3f} GB"

def _run_single_test_case(device, config, shift_labels):
    """Helper function to run a full validation for a given configuration."""
    
    # Unpack config
    batch_size, seq_len, in_dim, vocab_size, minibatch_size = \
        config['batch_size'], config['seq_len'], config['in_dim'], config['vocab_size'], config['minibatch_size']

    print("\n" + "#"*60)
    print(f"###   Testing with shift_labels = {shift_labels}   ###")
    print("#"*60)
    
    # --- 1. Baseline: Standard Full Tensor Approach ---
    print("\n--- 1. Baseline (Standard nn.Linear) ---")
    torch.manual_seed(42)
    base_model = nn.Linear(in_dim, vocab_size, bias=True).to(device)
    
    torch.manual_seed(42)
    input_base = torch.randn(batch_size, seq_len, in_dim, requires_grad=True, device=device)
    labels_base = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    start_time_base = time.time()
    logits_base = base_model(input_base)
    
    if shift_labels:
        logits_flat = logits_base[:, :-1, :].contiguous().view(-1, vocab_size)
        labels_flat = labels_base[:, 1:].contiguous().view(-1)
    else:
        logits_flat = logits_base.contiguous().view(-1, vocab_size)
        labels_flat = labels_base.contiguous().view(-1)
        
    loss_base = F.cross_entropy(logits_flat, labels_flat)
    loss_base.backward()
    duration_base = time.time() - start_time_base
    
    peak_mem_base = torch.cuda.max_memory_allocated(device)

    print(f"Loss: {loss_base.item():.6f}")
    print(f"Execution Time: {duration_base:.4f} seconds")
    print(f"Peak Memory:    {format_mem(peak_mem_base)}")
    
    base_results = {
        'loss': loss_base.clone(),
        'input_grad': input_base.grad.clone(),
        'weight_grad': base_model.weight.grad.clone(),
        'bias_grad': base_model.bias.grad.clone(),
    }
    del input_base, labels_base, logits_base, logits_flat, labels_flat, loss_base
    torch.cuda.empty_cache()

    # --- 2. New Method: Using the ChunkedLossComputer Controller ---
    print("\n--- 2. Efficient (Using ChunkedLossComputer) ---")
    
    torch.manual_seed(42)
    efficient_lm_head = nn.Linear(in_dim, vocab_size, bias=True).to(device)
    efficient_lm_head.weight.data.copy_(base_model.weight.data)
    efficient_lm_head.bias.data.copy_(base_model.bias.data)
    
    # === 更新初始化方式 ===
    grad_computer = ChunkedLossComputer(
        lm_head=efficient_lm_head,
        loss_fn=CrossEntropyLoss(return_token_loss=True, shift_labels=False, reduction='mean'), # 使用新的loss类
        minibatch_size=minibatch_size,
        shift_labels=shift_labels
    )
    
    torch.manual_seed(42)
    input_efficient = torch.randn(batch_size, seq_len, in_dim, requires_grad=True, device=device)
    labels_efficient = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    start_time_efficient = time.time()
    # === 更新调用方式，接收元组的第一个元素 ===
    loss_efficient, pl = grad_computer.forward_and_backward(input_efficient, labels_efficient)
    print(f"pl:{pl.shape} input_efficient:{input_efficient.shape} labels_efficient:{labels_efficient.shape}")
    duration_efficient = time.time() - start_time_efficient
    
    peak_mem_efficient = torch.cuda.max_memory_allocated(device)
    
    print(f"Loss: {loss_efficient.item():.6f}")
    print(f"Execution Time: {duration_efficient:.4f} seconds")
    print(f"Peak Memory:    {format_mem(peak_mem_efficient)}")

    # --- 3. Comparison ---
    print("\n--- 3. Numerical Correctness Verification ---")
    
    atol = 1e-5 # 稍微放宽容忍度，以应对浮点数累加可能带来的微小误差
    loss_is_close = torch.allclose(base_results['loss'], loss_efficient, atol=atol)
    input_grad_is_close = torch.allclose(base_results['input_grad'], input_efficient.grad, atol=atol)
    weight_grad_is_close = torch.allclose(base_results['weight_grad'], efficient_lm_head.weight.grad, atol=atol)
    bias_grad_is_close = torch.allclose(base_results['bias_grad'], efficient_lm_head.bias.grad, atol=atol)

    print(f"[*] Final Average Loss is close: { '✅' if loss_is_close else '❌' }")
    print(f"[*] Input Gradients are close:   { '✅' if input_grad_is_close else '❌' }")
    print(f"[*] Weight Gradients are close:  { '✅' if weight_grad_is_close else '❌' }")
    print(f"[*] Bias Gradients are close:    { '✅' if bias_grad_is_close else '❌' }")

    if all([loss_is_close, input_grad_is_close, weight_grad_is_close, bias_grad_is_close]):
        print("\n✅ SUCCESS: Test case passed.")
    else:
        print("\n❌ FAILURE: Test case failed.")
        print(f"Loss diff: {(base_results['loss'] - loss_efficient).abs().item()}")
        print(f"Input grad diff: {(base_results['input_grad'] - input_efficient.grad).abs().max().item()}")
        print(f"Weight grad diff: {(base_results['weight_grad'] - efficient_lm_head.weight.grad).abs().max().item()}")
        print(f"Bias grad diff: {(base_results['bias_grad'] - efficient_lm_head.bias.grad).abs().max().item()}")
def run_full_validation_demo():
    """
    Runs a comprehensive validation suite for the controller-based method.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return

    device = "cuda"
    config = {
        'batch_size': 1,
        'seq_len': 8192,
        'in_dim': 1024,
        'vocab_size': 200000,
        'minibatch_size': 2048
    }

    print("\n" + "="*60)
    print("--- Starting Full Validation Suite ---")
    print("="*60)
    print(f"Params: Batch={config['batch_size']}, SeqLen={config['seq_len']}, Dim={config['in_dim']}, Vocab={config['vocab_size']}")
    print(f"Controller Chunk Size: {config['minibatch_size']}")

    _run_single_test_case(device, config, shift_labels=True)
    _run_single_test_case(device, config, shift_labels=False)

    print("\n" + "="*60)
    print("--- Full Validation Complete ---")
    print("="*60)


# ===================================================================
# Block 4: New Demo for ignore_index Validation
# ===================================================================

def _run_ignore_index_test_case(device, config):
    """
    一个专门的测试用例，用于验证在标签中包含ignore_index时，
    ChunkedLossComputer的行为是否与标准方法一致。
    """
    # 解包配置
    batch_size, seq_len, in_dim, vocab_size, minibatch_size = \
        config['batch_size'], config['seq_len'], config['in_dim'], config['vocab_size'], config['minibatch_size']
    
    ignore_index = -100 # PyTorch默认的忽略索引

    print("\n" + "#"*70)
    print(f"###   Testing with ignore_index = {ignore_index}   ###")
    print("#"*70)
    
    # --- 1. 基准方法: 标准 nn.Linear + F.cross_entropy with ignore_index ---
    print("\n--- 1. Baseline (Standard with ignore_index) ---")
    torch.manual_seed(123)
    base_model = nn.Linear(in_dim, vocab_size, bias=True).to(device)
    
    torch.manual_seed(123)
    input_base = torch.randn(batch_size, seq_len, in_dim, requires_grad=True, device=device)
    
    # 生成带有一些 ignore_index 的标签
    labels_base = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # 将大约10%的标签设置为 ignore_index
    mask = torch.rand(labels_base.shape) < 0.1
    labels_base[mask] = ignore_index
    
    print(f"Generated {mask.sum().item()} ignored labels out of {labels_base.numel()}.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    start_time_base = time.time()
    
    # 标准前向传播
    logits_base = base_model(input_base)
    
    # 注意：这里我们明确传递 ignore_index 参数
    loss_base = F.cross_entropy(
        logits_base.view(-1, vocab_size),
        labels_base.view(-1),
        ignore_index=ignore_index
    )
    
    # 标准反向传播
    loss_base.backward()
    duration_base = time.time() - start_time_base
    
    peak_mem_base = torch.cuda.max_memory_allocated(device)

    print(f"Loss: {loss_base.item():.6f}")
    print(f"Execution Time: {duration_base:.4f} seconds")
    print(f"Peak Memory:    {format_mem(peak_mem_base)}")
    
    # 保存基准结果用于对比
    base_results = {
        'loss': loss_base.clone(),
        'input_grad': input_base.grad.clone(),
        'weight_grad': base_model.weight.grad.clone(),
        'bias_grad': base_model.bias.grad.clone(),
    }
    del input_base, labels_base, logits_base, loss_base
    torch.cuda.empty_cache()

    # --- 2. 高效方法: 使用 ChunkedLossComputer ---
    print("\n--- 2. Efficient (ChunkedLossComputer with ignore_index) ---")
    
    torch.manual_seed(123)
    efficient_lm_head = nn.Linear(in_dim, vocab_size, bias=True).to(device)
    # 确保模型权重与基准完全一致
    efficient_lm_head.weight.data.copy_(base_model.weight.data)
    efficient_lm_head.bias.data.copy_(base_model.bias.data)
    
    # 初始化控制器，内部的loss_fn已经配置了ignore_index
    grad_computer = ChunkedLossComputer(
        lm_head=efficient_lm_head,
        loss_fn=CrossEntropyLoss(ignore_index=ignore_index, return_token_loss=True, shift_labels=False, reduction='mean'),
        minibatch_size=minibatch_size,
        shift_labels=False # 在这个测试中不使用标签偏移，以直接对比
    )
    
    torch.manual_seed(123)
    input_efficient = torch.randn(batch_size, seq_len, in_dim, requires_grad=True, device=device)
    
    # 生成与基准完全相同的带 ignore_index 的标签
    labels_efficient = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels_efficient[mask] = ignore_index

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    start_time_efficient = time.time()
    loss_efficient, _ = grad_computer.forward_and_backward(input_efficient, labels_efficient)
    duration_efficient = time.time() - start_time_efficient
    
    peak_mem_efficient = torch.cuda.max_memory_allocated(device)
    
    print(f"Loss: {loss_efficient.item():.6f}")
    print(f"Execution Time: {duration_efficient:.4f} seconds")
    print(f"Peak Memory:    {format_mem(peak_mem_efficient)}")

    # --- 3. 对比验证 ---
    print("\n--- 3. Numerical Correctness Verification (with ignore_index) ---")
    
    atol = 1e-5 # 容忍度
    loss_is_close = torch.allclose(base_results['loss'], loss_efficient, atol=atol)
    input_grad_is_close = torch.allclose(base_results['input_grad'], input_efficient.grad, atol=atol)
    weight_grad_is_close = torch.allclose(base_results['weight_grad'], efficient_lm_head.weight.grad, atol=atol)
    bias_grad_is_close = torch.allclose(base_results['bias_grad'], efficient_lm_head.bias.grad, atol=atol)

    print(f"[*] Final Average Loss is close: { '✅' if loss_is_close else '❌' }")
    print(f"[*] Input Gradients are close:   { '✅' if input_grad_is_close else '❌' }")
    print(f"[*] Weight Gradients are close:  { '✅' if weight_grad_is_close else '❌' }")
    print(f"[*] Bias Gradients are close:    { '✅' if bias_grad_is_close else '❌' }")

    if all([loss_is_close, input_grad_is_close, weight_grad_is_close, bias_grad_is_close]):
        print("\n✅ SUCCESS: Test case with ignore_index passed.")
    else:
        print("\n❌ FAILURE: Test case with ignore_index failed.")
        print(f"Loss diff: {(base_results['loss'] - loss_efficient).abs().item()}")
        print(f"Input grad diff: {(base_results['input_grad'] - input_efficient.grad).abs().max().item()}")
        print(f"Weight grad diff: {(base_results['weight_grad'] - efficient_lm_head.weight.grad).abs().max().item()}")
        print(f"Bias grad diff: {(base_results['bias_grad'] - efficient_lm_head.bias.grad).abs().max().item()}")

def run_ignore_index_validation_demo():
    """
    运行一个专门的验证，以测试在存在ignore_index时控制器的正确性。
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return

    device = "cuda"
    # 使用与之前相同的配置
    config = {
        'batch_size': 1,
        'seq_len': 8192,
        'in_dim': 1024,
        'vocab_size': 200000,
        'minibatch_size': 2048
    }

    print("\n" + "="*70)
    print("--- Starting ignore_index Validation Suite ---")
    print("="*70)
    print(f"Params: Batch={config['batch_size']}, SeqLen={config['seq_len']}, Dim={config['in_dim']}, Vocab={config['vocab_size']}")
    print(f"Controller Chunk Size: {config['minibatch_size']}")

    _run_ignore_index_test_case(device, config)

    print("\n" + "="*70)
    print("--- ignore_index Validation Complete ---")
    print("="*70)


# ===================================================================
# Block 5: Demo for Frozen lm_head (requires_grad=False)
# ===================================================================

def run_frozen_lm_head_demo():
    """
    测试当 lm_head 的 requires_grad 为 False 时，ChunkedLossComputer 是否能够正常工作
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 配置参数
    config = {
        'batch_size': 2,
        'seq_len': 128,
        'in_dim': 512,
        'vocab_size': 10000,
        'minibatch_size': 32
    }
    
    print(f"\n" + "="*60)
    print(f"--- Testing Frozen lm_head (requires_grad=False) ---" )
    print(f"="*60)
    print(f"Params: Batch={config['batch_size']}, SeqLen={config['seq_len']}, Dim={config['in_dim']}, Vocab={config['vocab_size']}")
    
    # 创建一个简单的模型，包含一个 lm_head
    class SimpleModel(nn.Module):
        def __init__(self, in_dim, vocab_size):
            super().__init__()
            # 模拟模型的主体部分
            self.transformer = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, in_dim)
            )
            # 创建 lm_head，但后续会设置 requires_grad=False
            self.lm_head = nn.Linear(in_dim, vocab_size)
        
        def forward(self, x):
            # 只返回中间特征，不应用 lm_head，因为我们要使用 ChunkedLossComputer 来处理 lm_head
            return self.transformer(x)
    
    # 初始化模型
    torch.manual_seed(42)
    model = SimpleModel(config['in_dim'], config['vocab_size']).to(device)
    
    # 设置 lm_head.requires_grad = False，冻结输出层
    print(f"Original lm_head.requires_grad for weight: {model.lm_head.weight.requires_grad}")
    print(f"Original lm_head.requires_grad for bias: {model.lm_head.bias.requires_grad}")
    model.lm_head.weight.requires_grad = False
    model.lm_head.bias.requires_grad = False
    print(f"After freezing - lm_head.requires_grad for weight: {model.lm_head.weight.requires_grad}")
    print(f"After freezing - lm_head.requires_grad for bias: {model.lm_head.bias.requires_grad}")
    
    # 创建 ChunkedLossComputer
    loss_fn = CrossEntropyLoss(return_token_loss=True, shift_labels=False, reduction='mean')
    grad_computer = ChunkedLossComputer(
        lm_head=model.lm_head,
        loss_fn=loss_fn,
        minibatch_size=config['minibatch_size'],
        shift_labels=False
    )
    
    # 准备输入和标签
    input_tensor = torch.randn(
        config['batch_size'], 
        config['seq_len'], 
        config['in_dim'], 
        requires_grad=True,
        device=device
    )
    labels = torch.randint(
        0, 
        config['vocab_size'], 
        (config['batch_size'], config['seq_len']), 
        device=device
    )
    
    # 记录模型参数的梯度状态
    print("\n--- Initial Gradient Status --- ")
    for name, param in model.named_parameters():
        print(f"{name}.grad: {param.grad is not None}")
    
    # 前向传播获取中间特征
    print("\n--- Running Forward Pass --- ")
    hidden_states = model(input_tensor)
    
    # 使用 ChunkedLossComputer 计算损失并反向传播
    print("\n--- Running ChunkedLossComputer --- ")
    try:
        loss, per_token_loss = grad_computer.forward_and_backward(hidden_states, labels)
        print(f"Loss computed successfully: {loss.item():.6f}")
        print(f"Per token loss shape: {per_token_loss.shape if per_token_loss is not None else None}")
        
        # 检查梯度
        print("\n--- Gradient Status After Backward --- ")
        for name, param in model.named_parameters():
            print(f"{name}.grad: {param.grad is not None}")
            if param.grad is not None:
                print(f"  - Gradient norm: {param.grad.norm().item():.6f}")
        
        # 验证模型主体部分（transformer）是否接收到了梯度
        # 因为 lm_head 被冻结了，所以梯度应该正确地传递到前面的层
        transformer_has_grads = all(param.grad is not None for param in model.transformer.parameters())
        
        # 验证 lm_head 没有梯度（因为它被冻结了）
        lm_head_has_no_grads = (model.lm_head.weight.grad is None) and (model.lm_head.bias.grad is None)
        
        if transformer_has_grads and lm_head_has_no_grads:
            print("\n✅ SUCCESS: Gradients were correctly propagated to transformer layers and not to frozen lm_head!")
        else:
            print("\n❌ FAILURE: Gradient propagation issue detected.")
            if not transformer_has_grads:
                print("  - Transformer layers did not receive gradients")
            if not lm_head_has_no_grads:
                print("  - Frozen lm_head received gradients unexpectedly")
                
    except Exception as e:
        print(f"\n❌ ERROR: Exception occurred during ChunkedLossComputer execution: {str(e)}")
        import traceback
        traceback.print_exc()

# 在主函数中添加新的演示调用
if __name__ == "__main__":
    # # 运行原始的验证
    # run_full_validation_demo()
    
    # # 运行新的、针对 ignore_index 的验证
    # run_ignore_index_validation_demo()
    
    # 运行新的、针对冻结 lm_head 的验证
    run_frozen_lm_head_demo()
