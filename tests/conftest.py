#!/usr/bin/env python3
"""Shared pytest fixtures and utilities for layer testing."""

import pytest
import torch
import torch.nn as nn
from typing import Optional, Tuple

from muse.layers.kv_cache import KVCache


# Fixtures for common test configurations
@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def seq_len():
    """Default sequence length for tests."""
    return 8


@pytest.fixture
def embed_dim():
    """Default embedding dimension for tests."""
    return 64


@pytest.fixture
def num_heads():
    """Default number of attention heads."""
    return 4


@pytest.fixture
def head_dim():
    """Default head dimension."""
    return 16


@pytest.fixture
def hidden_dim():
    """Default hidden dimension for feedforward."""
    return 256


@pytest.fixture
def device():
    """Device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float32


# Fixtures for creating common test tensors
@pytest.fixture
def random_tensor(batch_size, seq_len, embed_dim, dtype, device):
    """Create a random tensor with common dimensions."""
    def _make_tensor(b=None, s=None, d=None, dt=None, dev=None):
        b = b or batch_size
        s = s or seq_len
        d = d or embed_dim
        dt = dt or dtype
        dev = dev or device
        return torch.randn(b, s, d, dtype=dt, device=dev)
    return _make_tensor


@pytest.fixture
def simple_linear_layer(embed_dim, device):
    """Create a simple linear layer for testing."""
    def _make_layer(in_dim=None, out_dim=None):
        in_dim = in_dim or embed_dim
        out_dim = out_dim or embed_dim
        return nn.Linear(in_dim, out_dim, bias=False).to(device)
    return _make_layer


# Utility functions
def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = ""
):
    """Assert two tensors are close with custom tolerances."""
    assert actual.shape == expected.shape, (
        f"Shape mismatch: {actual.shape} vs {expected.shape}. {msg}"
    )
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Tensors not close. Max diff: {(actual - expected).abs().max():.6f}. {msg}"
    )


def check_gradient_flow(module: nn.Module, input_tensor: torch.Tensor):
    """Check that gradients flow through the module."""
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    output = module(input_tensor) if isinstance(module, nn.Module) else module
    
    # Create a simple loss
    loss = output.sum()
    loss.backward()
    
    # Check input has gradients
    assert input_tensor.grad is not None, "Input gradients are None"
    assert not torch.all(input_tensor.grad == 0), "All input gradients are zero"
    
    # Check module parameters have gradients
    for name, param in module.named_parameters() if isinstance(module, nn.Module) else []:
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def get_kv_cache(
    batch_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> KVCache:
    """Factory function to create KV cache."""
    return KVCache(batch_size, max_seq_len, num_kv_heads, head_dim, dtype)

