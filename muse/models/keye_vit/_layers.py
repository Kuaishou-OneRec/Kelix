# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional, Tuple, Literal

import torch
from torch import nn
from muse.layers.attention_utils import get_attention_function
from muse.layers.feed_forward import FeedForward
from muse.layers.rms_norm import RMSNorm
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.kv_cache import KVCache

from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb


logger = logging.getLogger(__name__)

def KeyeMLP(dim: int, hidden_dim: int, activation_fn: Optional[nn.Module] = None) -> FeedForward:
    # 显式 bias=True，虽然默认就是 True，但写出来更保险
    fc1 = nn.Linear(dim, hidden_dim, bias=True)
    fc2 = nn.Linear(hidden_dim, dim, bias=True)
    
    # SigLIP 默认使用 GELU(approximate='tanh')
    if activation_fn is None:
        activation_fn = nn.GELU(approximate="tanh")
        
    return FeedForward(
        gate_proj=fc1, 
        down_proj=fc2, 
        up_proj=None, 
        activation=activation_fn
    )

