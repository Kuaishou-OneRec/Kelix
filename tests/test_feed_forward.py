#!/usr/bin/env python3
"""Unit tests for FeedForward layer."""

import pytest
import torch
import torch.nn as nn

from muse.layers.feed_forward import FeedForward
from tests.conftest import assert_tensors_close, check_gradient_flow


class TestFeedForward:
    """Test suite for FeedForward module."""
    
    def test_initialization_with_up_proj(self, device):
        """Test FeedForward initialization with up_proj (w3)."""
        gate_proj = nn.Linear(64, 256, bias=False).to(device)
        down_proj = nn.Linear(256, 64, bias=False).to(device)
        up_proj = nn.Linear(64, 256, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            up_proj=up_proj,
            activation=nn.SiLU()
        )
        
        assert ff.w1 is gate_proj
        assert ff.w2 is down_proj
        assert ff.w3 is up_proj
        assert isinstance(ff.activation, nn.SiLU)
    
    def test_initialization_without_up_proj(self, device):
        """Test FeedForward initialization without up_proj."""
        gate_proj = nn.Linear(64, 256, bias=False).to(device)
        down_proj = nn.Linear(256, 64, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            activation=nn.SiLU()
        )
        
        assert ff.w1 is gate_proj
        assert ff.w2 is down_proj
        assert ff.w3 is None
    
    def test_forward_with_up_proj(self, device):
        """Test forward pass with up_proj (gated variant)."""
        in_dim = 64
        hidden_dim = 256
        
        gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device)
        up_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            up_proj=up_proj,
            activation=nn.SiLU()
        )
        
        x = torch.randn(2, 8, in_dim, device=device)
        output = ff(x)
        
        assert output.shape == (2, 8, in_dim)
        assert output.device == device
        
        # Manually compute to verify
        h_gate = nn.SiLU()(gate_proj(x))
        h_up = up_proj(x)
        h = h_gate * h_up
        expected = down_proj(h)
        
        assert_tensors_close(output, expected, rtol=1e-5, atol=1e-6)
    
    def test_forward_without_up_proj(self, device):
        """Test forward pass without up_proj (simple variant)."""
        in_dim = 64
        hidden_dim = 256
        
        gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            activation=nn.SiLU()
        )
        
        x = torch.randn(2, 8, in_dim, device=device)
        output = ff(x)
        
        assert output.shape == (2, 8, in_dim)
        
        # Manually compute to verify
        h = nn.SiLU()(gate_proj(x))
        expected = down_proj(h)
        
        assert_tensors_close(output, expected, rtol=1e-5, atol=1e-6)
    
    def test_different_activations(self, device):
        """Test with different activation functions."""
        in_dim = 32
        hidden_dim = 128
        
        activations = [nn.SiLU(), nn.ReLU(), nn.GELU(), nn.Tanh()]
        
        for activation in activations:
            gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
            down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device)
            
            ff = FeedForward(
                gate_proj=gate_proj,
                down_proj=down_proj,
                activation=activation
            )
            
            x = torch.randn(2, 4, in_dim, device=device)
            output = ff(x)
            
            assert output.shape == (2, 4, in_dim)
            assert not torch.isnan(output).any()
    
    def test_gradient_flow_with_up_proj(self, device):
        """Test gradients flow through all paths with up_proj."""
        in_dim = 32
        hidden_dim = 64
        
        gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device)
        up_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            up_proj=up_proj,
            activation=nn.SiLU()
        )
        
        x = torch.randn(2, 4, in_dim, device=device, requires_grad=True)
        
        output = ff(x)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
        # Check all projection gradients
        assert gate_proj.weight.grad is not None
        assert down_proj.weight.grad is not None
        assert up_proj.weight.grad is not None
        
        assert not torch.all(gate_proj.weight.grad == 0)
        assert not torch.all(down_proj.weight.grad == 0)
        assert not torch.all(up_proj.weight.grad == 0)
    
    def test_gradient_flow_without_up_proj(self, device):
        """Test gradients flow correctly without up_proj."""
        in_dim = 32
        hidden_dim = 64
        
        gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            activation=nn.SiLU()
        )
        
        x = torch.randn(2, 4, in_dim, device=device, requires_grad=True)
        
        output = ff(x)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
        # Check projection gradients
        assert gate_proj.weight.grad is not None
        assert down_proj.weight.grad is not None
        
        assert not torch.all(gate_proj.weight.grad == 0)
        assert not torch.all(down_proj.weight.grad == 0)
    
    def test_different_dimensions(self, device):
        """Test with various dimension sizes."""
        test_configs = [
            (16, 64),
            (64, 256),
            (128, 512),
            (256, 1024),
        ]
        
        for in_dim, hidden_dim in test_configs:
            gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
            down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device)
            up_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
            
            ff = FeedForward(
                gate_proj=gate_proj,
                down_proj=down_proj,
                up_proj=up_proj
            )
            
            x = torch.randn(2, 4, in_dim, device=device)
            output = ff(x)
            
            assert output.shape == (2, 4, in_dim)
    
    def test_different_batch_and_seq_lengths(self, device):
        """Test with various batch sizes and sequence lengths."""
        in_dim = 64
        hidden_dim = 256
        
        gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device)
        up_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            up_proj=up_proj
        )
        
        test_shapes = [
            (1, 1, in_dim),
            (1, 10, in_dim),
            (4, 8, in_dim),
            (8, 128, in_dim),
        ]
        
        for shape in test_shapes:
            x = torch.randn(*shape, device=device)
            output = ff(x)
            assert output.shape == shape
    
    def test_dtype_preservation(self, device):
        """Test that dtype is preserved through forward pass."""
        in_dim = 32
        hidden_dim = 128
        
        dtypes = [torch.float32]
        if torch.cuda.is_available():
            dtypes.extend([torch.float16, torch.bfloat16])
        
        for dtype in dtypes:
            gate_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device=device, dtype=dtype)
            down_proj = nn.Linear(hidden_dim, in_dim, bias=False).to(device=device, dtype=dtype)
            up_proj = nn.Linear(in_dim, hidden_dim, bias=False).to(device=device, dtype=dtype)
            
            ff = FeedForward(
                gate_proj=gate_proj,
                down_proj=down_proj,
                up_proj=up_proj
            )
            
            x = torch.randn(2, 4, in_dim, device=device, dtype=dtype)
            output = ff(x)
            
            assert output.dtype == dtype
    
    def test_default_activation(self, device):
        """Test that default activation is SiLU."""
        gate_proj = nn.Linear(32, 128, bias=False).to(device)
        down_proj = nn.Linear(128, 32, bias=False).to(device)
        
        # Create without specifying activation (uses default)
        ff = FeedForward(gate_proj=gate_proj, down_proj=down_proj)
        
        assert isinstance(ff.activation, nn.SiLU)
    
    def test_output_not_nan_or_inf(self, device):
        """Test that output doesn't contain NaN or Inf values."""
        gate_proj = nn.Linear(64, 256, bias=False).to(device)
        down_proj = nn.Linear(256, 64, bias=False).to(device)
        up_proj = nn.Linear(64, 256, bias=False).to(device)
        
        ff = FeedForward(
            gate_proj=gate_proj,
            down_proj=down_proj,
            up_proj=up_proj
        )
        
        x = torch.randn(2, 8, 64, device=device)
        output = ff(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

