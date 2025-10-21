# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Tuple


class KVCache:
    """
    Key-Value cache for attention layers.
    
    This cache stores key and value tensors for efficient inference,
    allowing the model to avoid recomputing past tokens.
    
    Args:
        batch_size (int): Batch size for the cache.
        max_seq_len (int): Maximum sequence length to cache.
        num_kv_heads (int): Number of key/value heads.
        head_dim (int): Dimension of each head.
        dtype (torch.dtype): Data type for cache tensors.
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # Initialize cache tensors
        self.k_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype
        )
        self.cache_pos = 0
    
    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key and value tensors.
        
        Args:
            k (torch.Tensor): Key tensor with shape [b, n_kv, s, h_d]
            v (torch.Tensor): Value tensor with shape [b, n_kv, s, h_d]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value caches
        """
        seq_len = k.size(2)
        
        # Move cache to same device as input if needed
        if self.k_cache.device != k.device:
            self.k_cache = self.k_cache.to(k.device)
            self.v_cache = self.v_cache.to(v.device)
        
        # Update cache
        self.k_cache[:, :, self.cache_pos:self.cache_pos + seq_len] = k
        self.v_cache[:, :, self.cache_pos:self.cache_pos + seq_len] = v
        self.cache_pos += seq_len
        
        # Return the full cache up to current position
        return (
            self.k_cache[:, :, :self.cache_pos],
            self.v_cache[:, :, :self.cache_pos]
        )
    
    def reset(self):
        """Reset cache position to 0."""
        self.cache_pos = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
