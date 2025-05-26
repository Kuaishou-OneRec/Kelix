from typing import List

import torch
import torch.nn.functional as F
import torch.distributed as dist

from recovlm.training.parallel import get_sequence_parallel_world_size, \
  get_sequence_parallel_group

class CrossEntropyLoss(torch.nn.Module):
  """
  Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

  Whenever the model is trained with bf16, before running CE, we have to upcast
  it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
  Models like llama3 have large vocabulary size and, therefore, have a large output
  tensor of shape ``(bsz, num_tokens, vocab_size)``. If we chunk on the token level, you can still compute
  the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

  The CE and upcasting have to be compiled together for better performance.
  When using this class, we recommend using :func:`torch.compile` only on the method ``compute_cross_entropy``.
  The gains from chunking won't be realized if you compile the entire class.

  For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
  """

  def __init__(self,
               ignore_index: int = -100,
               return_token_loss: bool = False,
               shift_labels: bool = True,
               reduction: str = "mean"):
    super().__init__()
    self.ignore_index = ignore_index
    self.return_token_loss = return_token_loss
    self.shift_labels = shift_labels
    self.reduction = reduction

  def forward(self, logits: torch.Tensor,
              labels: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): List of chunked logits of length
            ``self.num_output_chunks``, where each chunk has shape
            ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
        labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.

    Returns:
        torch.Tensor: Cross entropy loss of shape (1,).

    Example:
        >>> loss_fn = ChunkedCrossEntropyLoss()
        >>>
        >>> h = torch.tensor([bsz, num_tokens, dim])
        >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
        >>>
        >>> labels = torch.tensor([bsz, num_tokens])
        >>> loss = loss_fn(output_chunks, labels)
    """
    total_elements = (labels != self.ignore_index).sum().cuda()
    # if get_sequence_parallel_world_size() > 1:
    #   dist.all_reduce(
    #     total_elements, op=dist.ReduceOp.SUM,
    #     group=get_sequence_parallel_group())
    vocab_size = logits.shape[-1]

    if self.shift_labels:
      logits = logits[:, :-1, :]
      labels = labels[:, 1:]
    if dist.get_rank() == 0:
        print(f"logits: {logits.shape}, labels: {labels.shape}")
    per_token_loss = F.cross_entropy(
      logits.float().reshape(-1, vocab_size),
      labels.reshape(-1), #ignore_index=self.ignore_index,
      reduction="none"
    )
    #per_token_loss = logits.sum() / logits.numel()
    loss = per_token_loss.sum()
    if self.reduction == "mean" and total_elements > 0:
      loss /= total_elements
    if self.return_token_loss:
      return loss, per_token_loss
    return loss


import torch
import torch.nn.functional as F

class CrossEntropyLossReweight(torch.nn.Module):
  """
  Cross-entropy损失函数，支持多种重加权方案: token, sample, 和 square。
  
  这个类完全兼容标准的CrossEntropyLoss，可以作为直接替代品使用。
  与InternVL项目中的损失计算方式保持一致。
  """

  def __init__(self,
               ignore_index: int = -100,
               return_token_loss: bool = False,
               shift_labels: bool = True,
               reduction: str = "mean",
               loss_reduction: str = "token"):
    super().__init__()
    self.ignore_index = ignore_index
    self.return_token_loss = return_token_loss
    self.shift_labels = shift_labels
    self.reduction = reduction
    
    valid_loss_reductions = ["token", "sample", "square"]
    if loss_reduction not in valid_loss_reductions:
      raise ValueError(f"loss_reduction must be one of {valid_loss_reductions}, but got {loss_reduction}")
    self.loss_reduction = loss_reduction

  def compute_weights(self, counts):
    """
    根据有效标记数量计算每个样本的权重
    与InternVL项目中的len2weight函数对齐
    """
    nonzero_mask = counts > 0
    weights = torch.zeros_like(counts, dtype=torch.float).to(counts.device)
    
    if self.loss_reduction == 'token':
      weights[nonzero_mask] = 1.0
    elif self.loss_reduction == 'sample':
      weights[nonzero_mask] = 1.0 / counts[nonzero_mask].float()
    elif self.loss_reduction == 'square':
      # 按照InternVL中的实现使用平方根
      weights[nonzero_mask] = 1.0 / (counts[nonzero_mask].float() ** 0.5)
    
    return weights

  def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算加权交叉熵损失
    
    Args:
        logits (torch.Tensor): 形状为 (batch_size, num_tokens, vocab_size) 的模型输出。
        labels (torch.Tensor): 形状为 (batch_size, num_tokens) 的真实标签。

    Returns:
        torch.Tensor: 形状为 (1,) 的交叉熵损失，或者如果return_token_loss=True，返回(loss, per_token_loss)的元组。
    """
    # 获取词汇表大小
    vocab_size = logits.shape[-1]

    # 处理标签移位 - 与InternVL项目保持一致
    if self.shift_labels:
      shift_logits = logits[:, :-1, :].contiguous()
      shift_labels = labels[:, 1:].contiguous()
    else:
      shift_logits = logits
      shift_labels = labels
    
    # 计算每个样本中有效(非忽略)的标记数量
    not_ignored = shift_labels.ne(self.ignore_index)
    x_counts = not_ignored.sum(dim=1)
    
    # 展平logits和labels以用于计算每个标记的损失
    shift_logits_flat = shift_logits.reshape(-1, vocab_size)
    shift_labels_flat = shift_labels.reshape(-1)
    
    # 计算每个标记的损失，使用reduction='none'以便后续加权
    per_token_loss = F.cross_entropy(
      shift_logits_flat.float(),
      shift_labels_flat,
      ignore_index=self.ignore_index,
      reduction="none"
    )
    
    # 将token loss重塑为原始形状
    per_token_loss_reshaped = per_token_loss.view(shift_labels.shape)
    
    # 计算样本权重
    weights = self.compute_weights(x_counts)
    
    # 创建掩码来处理无效标记
    mask = not_ignored.float()
    
    # 将权重扩展为与每个标记相同的维度
    weights_expanded = weights.unsqueeze(1).expand_as(per_token_loss_reshaped)
    
    # 应用权重和掩码 - 与InternVL实现一致
    weighted_loss = per_token_loss_reshaped * weights_expanded * mask
    
    # 计算权重总和 - 与InternVL中的shift_weights_sum对应
    weights_sum = (weights_expanded * mask).sum()
    
    # 计算最终损失 - 与InternVL一致，加权和除以权重总和
    loss = weighted_loss.sum() / weights_sum
    
    # 根据需要返回结果
    if self.return_token_loss:
      return loss, per_token_loss
    return loss

# class CrossEntropyLossReweight(torch.nn.Module):
#   """
#   Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.
#   Now supports multiple reweighting schemes: token, sample, and square.
  
#   This class is fully compatible with CrossEntropyLoss and can be used as a drop-in replacement.
#   """

#   def __init__(self,
#                ignore_index: int = -100,
#                return_token_loss: bool = False,
#                shift_labels: bool = True,
#                reduction: str = "mean",
#                loss_reduction: str = "token"):
#     super().__init__()
#     self.ignore_index = ignore_index
#     self.return_token_loss = return_token_loss
#     self.shift_labels = shift_labels
#     self.reduction = reduction
    
#     valid_loss_reductions = ["token", "sample", "square"]
#     if loss_reduction not in valid_loss_reductions:
#       raise ValueError(f"loss_reduction must be one of {valid_loss_reductions}, but got {loss_reduction}")
#     self.loss_reduction = loss_reduction

#   def compute_weights(self, counts):
#     """Vectorized weight computation for each sample based on token counts."""
#     nonzero_mask = counts > 0
#     weights = torch.zeros_like(counts, dtype=torch.float).to(counts.device)
    
#     if self.loss_reduction == 'token':
#       weights[nonzero_mask] = 1.0
#     elif self.loss_reduction == 'sample':
#       weights[nonzero_mask] = 1.0 / counts[nonzero_mask].float()
#     elif self.loss_reduction == 'square':
#       weights[nonzero_mask] = 1.0 / (counts[nonzero_mask].float() ** 0.5)
    
#     # 对于token方式，我们不需要归一化，保持与原始CrossEntropyLoss一致
#     if self.loss_reduction != 'token' and weights.sum() > 0:
#       weights = weights / weights.sum()
      
#     return weights

#   def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     """
#     与CrossEntropyLoss完全兼容的forward实现。
    
#     Args:
#         logits (torch.Tensor): Logits of shape (batch_size, num_tokens, vocab_size).
#         labels (torch.Tensor): Ground truth labels of shape (batch_size, num_tokens).

#     Returns:
#         torch.Tensor: Cross entropy loss of shape (1,), or a tuple of (loss, per_token_loss)
#     """
#     # 计算有效元素（与原CrossEntropyLoss保持一致）
#     total_elements = (labels != self.ignore_index).sum().cuda()
#     vocab_size = logits.shape[-1]

#     # 处理标签移位（如果需要）- 保持与原实现一致
#     if self.shift_labels:
#       shift_logits = logits[:, :-1, :]
#       shift_labels = labels[:, 1:]
#     else:
#       shift_logits = logits
#       shift_labels = labels
    
#     # 计算每个样本的有效标记数
#     not_ignored = shift_labels.ne(self.ignore_index)
#     x_counts = not_ignored.sum(dim=1)
    
#     # 计算每个标记的损失
#     shift_logits_flat = shift_logits.reshape(-1, vocab_size)
#     shift_labels_flat = shift_labels.reshape(-1)
    
#     per_token_loss = F.cross_entropy(
#       shift_logits_flat.float(),
#       shift_labels_flat,
#       ignore_index=self.ignore_index,
#       reduction="none"
#     )
    
#     # 如果是token-level加权（即标准交叉熵），直接使用原始实现
#     if self.loss_reduction == 'token':
#       loss = per_token_loss.sum()
#       if self.reduction == "mean" and total_elements > 0:
#         loss /= total_elements
#     else:
#       # 对于其他加权策略，应用权重 - 使用向量化操作代替循环，保留梯度
#       per_token_loss_reshaped = per_token_loss.view(shift_labels.shape)
      
#       # 计算样本权重
#       weights = self.compute_weights(x_counts)
      
#       # 创建掩码来处理无效标记
#       mask = not_ignored.float()
      
#       # 将权重扩展为与每个标记相同的维度 [batch_size, 1] -> [batch_size, seq_len]
#       weights_expanded = weights.unsqueeze(1).expand_as(per_token_loss_reshaped)
      
#       # 应用权重和掩码 - 这样可以保留梯度
#       weighted_loss = per_token_loss_reshaped * weights_expanded * mask
      
#       # 合计加权损失
#       loss = weighted_loss.sum()
    
#     # 保持与原实现一致的返回值
#     if self.return_token_loss:
#       return loss, per_token_loss
#     return loss
