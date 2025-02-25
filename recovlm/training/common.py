from typing import Dict, Any, Union, Optional, Generator

import torch
import contextlib

@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16

    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

def clip_grad_by_value(model, clip_range=None):
    if clip_range is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_range)

def get_global_grad_norm(model):
    grads = [
        param.grad.data for param in model.parameters() \
            if param.grad is not None]
    return torch.nn.utils.get_total_norm(grads, norm_type=2.0)
