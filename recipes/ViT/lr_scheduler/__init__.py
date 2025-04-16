# Copy from torchtune
import math
from typing import Union, Optional
from functools import partial

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from recipes.ViT.common import filter_function_arguments


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float,
        num_stop_steps: int = 0,
        min_lr_rate: float = 0.0):
        
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


def build_scheduler(config, optimizer):
    name = config.type
    kwargs = filter_function_arguments(get_cosine_scheduler, config, new_obj=True, exclude_keys=["name", "type", "optimizer"])
    if name == "cosine":
        return get_cosine_scheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported LR schduler `{name}`")


__all__ = [
    "build_scheduler"
]
