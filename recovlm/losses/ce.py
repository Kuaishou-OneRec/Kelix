from typing import List

import torch
import torch.nn.functional as F
import torch.distributed as dist

from recovlm.training.parallel import get_sequence_parallel_world_size, \
  get_sequence_parallel_group

# # TODO need check by other
# class CrossEntropyLoss(torch.nn.Module):
#   """
#   Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.
#   Now supports multiple reweighting schemes: token, sample, and square.

#   Whenever the model is trained with bf16, before running CE, we have to upcast
#   it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
#   Models like llama3 have large vocabulary size and, therefore, have a large output
#   tensor of shape ``(bsz, num_tokens, vocab_size)``. 
  
#   Args:
#       ignore_index (int): Token index to ignore in loss calculation (default: -100)
#       return_token_loss (bool): Whether to return per-token loss alongside the total loss (default: False)
#       shift_labels (bool): Whether to shift labels for language modeling tasks (default: True)
#       reduction (str): Reduction method to apply - either "mean" or "sum" (default: "mean")
#       loss_reduction (str): Reweighting strategy to use - one of "token", "sample", or "square" (default: "token")
#           - token: Each token contributes equally (standard CE)
#           - sample: Sample weight is 1/length (inversely proportional to valid token count)
#           - square: Sample weight is 1/length² (inversely proportional to square of valid token count)
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
    
#     # Validate loss_reduction parameter
#     valid_loss_reductions = ["token", "sample", "square"]
#     if loss_reduction not in valid_loss_reductions:
#       raise ValueError(f"loss_reduction must be one of {valid_loss_reductions}, but got {loss_reduction}")
#     self.loss_reduction = loss_reduction

#   def compute_weights(self, counts):
#     """
#     Vectorized weight computation for each sample based on token counts.
    
#     Args:
#         counts: Tensor of valid token counts per sample.
        
#     Returns:
#         Tensor of weights for each sample.
#     """
#     # Create a mask for non-zero counts to avoid division by zero
#     nonzero_mask = counts > 0
#     weights = torch.zeros_like(counts, dtype=torch.float)
    
#     if self.loss_reduction == 'token':
#       # For token-level weighting, all samples get equal weight
#       weights[nonzero_mask] = 1.0
#     elif self.loss_reduction == 'sample':
#       # For sample-level weighting, weight is inversely proportional to count
#       weights[nonzero_mask] = 1.0 / counts[nonzero_mask].float()
#     elif self.loss_reduction == 'square':
#       # For square weighting, weight is inversely proportional to count squared
#       weights[nonzero_mask] = 1.0 / (counts[nonzero_mask].float() ** 2)
    
#     # Normalize weights to sum to 1
#     weight_sum = weights.sum()
#     if weight_sum > 0:
#       weights = weights / weight_sum
      
#     return weights

#   def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     """
#     Args:
#         logits (torch.Tensor): Logits of shape (batch_size, num_tokens, vocab_size).
#         labels (torch.Tensor): Ground truth labels of shape (batch_size, num_tokens).

#     Returns:
#         torch.Tensor: Cross entropy loss of shape (1,).
#     """
#     batch_size = labels.shape[0]
#     vocab_size = logits.shape[-1]
    
#     # Handle label shifting if needed
#     if self.shift_labels:
#       shift_logits = logits[:, :-1, :]
#       shift_labels = labels[:, 1:]
#     else:
#       shift_logits = logits
#       shift_labels = labels
    
#     # Calculate valid tokens per sample (tokens not equal to ignore_index)
#     not_ignored = shift_labels.ne(self.ignore_index)
#     x_counts = not_ignored.sum(dim=1)
    
#     # Compute sample weights based on token counts
#     weights = self.compute_weights(x_counts)
    
#     # Calculate per-token loss
#     shift_logits_flat = shift_logits.reshape(-1, vocab_size)
#     shift_labels_flat = shift_labels.reshape(-1)
    
#     per_token_loss = F.cross_entropy(
#       shift_logits_flat.float(),
#       shift_labels_flat,
#       ignore_index=self.ignore_index,
#       reduction="none"
#     )
    
#     # Reshape loss to (batch_size, seq_len) and apply sample weights
#     per_token_loss = per_token_loss.view(batch_size, -1)
#     weighted_loss = per_token_loss * weights.unsqueeze(1)
    
#     # Sum the weighted losses
#     loss = weighted_loss.sum()
    
#     if self.return_token_loss:
#       return loss, per_token_loss
#     return loss

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
    per_token_loss = F.cross_entropy(
      logits.float().reshape(-1, vocab_size),
      labels.reshape(-1), ignore_index=self.ignore_index,
      reduction="none"
    )
    loss = per_token_loss.sum()
    if self.reduction == "mean" and total_elements > 0:
      loss /= total_elements
    if self.return_token_loss:
      return loss, per_token_loss
    return loss
