from typing import List

import torch
import torch.nn.functional as F

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

  def __init__(self, ignore_index: int = -100):
    super().__init__()
    self.ignore_index = ignore_index

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
    total_elements = (labels != self.ignore_index).sum()
    vocab_size = logits.shape[-1]
    # TODO: 暂时修改，labels外部shirft
    # loss = F.cross_entropy(
    #   logits.float()[:,:-1,:].reshape(-1, vocab_size),
    #   labels[:,1:].reshape(-1), ignore_index=self.ignore_index,
    #   reduction="sum"
    # )
    loss = F.cross_entropy(
      logits.float().reshape(-1, vocab_size),
      labels.reshape(-1), ignore_index=self.ignore_index,
      reduction="sum"
    )
    if total_elements > 0:
      loss /= total_elements
    return loss
