"""
Learning Rate Schedulers for Training.

This module provides learning rate scheduling utilities for PyTorch training,
with a focus on cosine annealing with linear warmup. The schedulers support:

- Linear warmup phase
- Cosine annealing decay
- Configurable minimum learning rate
- Optional stop steps for delayed start
- Multiple decay cycles

The implementation is based on HuggingFace Transformers and adapted for
flexible training workflows.

Functions:
    get_cosine_scheduler: Create cosine annealing with warmup scheduler
    get_scheduler: Factory function for creating schedulers by name

Example:
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = get_cosine_scheduler(
    ...     optimizer,
    ...     num_warmup_steps=1000,
    ...     num_training_steps=10000,
    ...     min_lr_rate=0.1
    ... )
    >>> 
    >>> for epoch in range(num_epochs):
    ...     for batch in dataloader:
    ...         loss = model(batch)
    ...         loss.backward()
    ...         optimizer.step()
    ...         scheduler.step()
"""
# Copy from torchtune
import math
from typing import Union, Optional
from functools import partial

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
        num_stop_steps: int = 0,
        min_lr_rate: float = 0.0) -> float:
    """
    Compute learning rate multiplier for cosine schedule with warmup.
    
    This is an internal helper function that implements the learning rate
    schedule logic. It returns a multiplier (0.0 to 1.0) that will be applied
    to the base learning rate.
    
    Schedule phases:
    1. Stop phase (0 to num_stop_steps): LR = 0
    2. Warmup phase (num_stop_steps to num_warmup_steps): Linear increase
    3. Cosine phase (num_warmup_steps to num_training_steps): Cosine decay
    4. Post-training (after num_training_steps): min_lr_rate
    
    Args:
        current_step (int): Current training step
        num_warmup_steps (int): Number of warmup steps
        num_training_steps (int): Total number of training steps
        num_cycles (float): Number of cosine cycles (0.5 = half cosine)
        num_stop_steps (int): Number of steps to keep LR at 0. Defaults to 0.
        min_lr_rate (float): Minimum LR as fraction of base LR. Defaults to 0.0.
        
    Returns:
        float: Learning rate multiplier in range [min_lr_rate, 1.0]
    """
        
    if num_stop_steps > 0 and current_step < num_stop_steps:
        return 0.0
    else:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return min_lr_rate
        progress = float(current_step - num_warmup_steps) /\
            float(max(1, num_training_steps - num_warmup_steps))
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1 - min_lr_rate) + min_lr_rate
        return max(0, factor)

def get_cosine_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    num_stop_steps: int = 0,
    last_epoch: int = -1,
    min_lr: float = None,
    min_lr_rate: float = None,
    **kwargs) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to 0.0 on a cosine schedule over
    the remaining ``num_training_steps-num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError(
            "One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`"
        )

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        num_stop_steps=num_stop_steps,
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs) -> LambdaLR:
    """
    Factory function to create learning rate schedulers by name.
    
    Currently supports:
    - "cosine": Cosine annealing with warmup
    
    Args:
        name (str): Name of the scheduler ("cosine")
        optimizer (Optimizer): PyTorch optimizer to schedule
        num_warmup_steps (int, optional): Number of warmup steps
        num_training_steps (int, optional): Total training steps
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        LambdaLR: Configured learning rate scheduler
        
    Raises:
        NotImplementedError: If scheduler name is not supported
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = get_scheduler(
        ...     name="cosine",
        ...     optimizer=optimizer,
        ...     num_warmup_steps=500,
        ...     num_training_steps=10000,
        ...     min_lr_rate=0.1
        ... )
    """
    if name == "cosine":
        return get_cosine_scheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported LR schduler `{name}`")

