# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Exponential Moving Average (EMA) for model weights.

This module provides utilities for maintaining EMA of model parameters,
commonly used in diffusion model training for improved generation quality.

Reference: Sana/train_scripts/train.py ema_update function
"""

from typing import Dict, Optional
from copy import deepcopy
import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def ema_update(
    model_dest: nn.Module,
    model_src: nn.Module,
    rate: float,
) -> None:
    """Update EMA model parameters.
    
    Performs exponential moving average update:
    dest_param = rate * dest_param + (1 - rate) * src_param
    
    Reference: Sana/train_scripts/train.py Lines 88-93
    
    Args:
        model_dest: EMA model to update (destination)
        model_src: Training model (source)
        rate: EMA decay rate (typically 0.9999)
    """
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest, f"EMA source and destination should be different: {p_name}"
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class EMAModel:
    """Wrapper for maintaining EMA of model parameters.
    
    This class provides a convenient interface for:
    - Creating an EMA copy of the model
    - Updating EMA weights
    - Saving/loading EMA checkpoints
    - Temporarily swapping EMA weights for inference
    
    Note: EMA is NOT supported with FSDP (Fully Sharded Data Parallel).
    With FSDP, model weights are sharded across GPUs, making EMA updates
    complex and memory-intensive.
    
    Args:
        model: The training model to track
        decay: EMA decay rate (default: 0.9999)
        device: Device to store EMA model on
        
    Example:
        >>> model = MyModel()
        >>> ema = EMAModel(model, decay=0.9999)
        >>> 
        >>> # During training
        >>> for batch in dataloader:
        >>>     loss = model(batch)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     ema.update()  # Update EMA after each step
        >>> 
        >>> # For inference with EMA weights
        >>> with ema.apply_ema():
        >>>     output = model(input)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.device = device
        self.training_model = model
        
        # Create EMA copy
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        if device is not None:
            self.ema_model.to(device)
        
        logger.info(f"EMA model created with decay={decay}")
    
    def update(self, decay: Optional[float] = None) -> None:
        """Update EMA parameters.
        
        Args:
            decay: Optional override for decay rate
        """
        rate = decay if decay is not None else self.decay
        ema_update(self.ema_model, self.training_model, rate)
    
    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights to another model.
        
        Args:
            model: Target model to copy weights to
        """
        model.load_state_dict(self.ema_model.state_dict())
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA model state dict."""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)
    
    def to(self, device: torch.device) -> "EMAModel":
        """Move EMA model to device."""
        self.ema_model.to(device)
        self.device = device
        return self
    
    def eval(self) -> nn.Module:
        """Get EMA model in eval mode."""
        return self.ema_model.eval()
    
    @torch.no_grad()
    def apply_ema(self):
        """Context manager to temporarily apply EMA weights to training model.
        
        Usage:
            with ema.apply_ema():
                # model now has EMA weights
                output = model(input)
            # model weights restored
        """
        return _EMAContext(self.training_model, self.ema_model)


class _EMAContext:
    """Context manager for temporarily swapping EMA weights."""
    
    def __init__(self, training_model: nn.Module, ema_model: nn.Module):
        self.training_model = training_model
        self.ema_model = ema_model
        self.backup_state = None
    
    def __enter__(self):
        # Backup training weights
        self.backup_state = {
            name: param.data.clone() 
            for name, param in self.training_model.named_parameters()
        }
        # Apply EMA weights
        self.training_model.load_state_dict(self.ema_model.state_dict())
        return self.training_model
    
    def __exit__(self, *args):
        # Restore training weights
        for name, param in self.training_model.named_parameters():
            param.data.copy_(self.backup_state[name])
