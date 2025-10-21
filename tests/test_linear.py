#!/usr/bin/env python3
"""Unit tests for linear layers."""

import pytest
import torch
import torch.nn as nn

from muse.layers.linear import Linear, TiedLinear
from tests.conftest import assert_tensors_close, check_gradient_flow


class TestLinear:
    """Test suite for Linear module."""
    
    def test_initialization(self):
        """Test Linear module can be initialized."""
        linear = Linear()
        assert isinstance(linear, nn.Module)
    
    def test_forward_basic(self, device):
        """Test basic forward pass."""
        linear = Linear()
        x = torch.randn(2, 4, 8, device=device)
        weight = torch.randn(16, 8, device=device)
        
        output = linear(x, weight)
        
        assert output.shape == (2, 4, 16)
        assert output.device.type == device.type
    
    def test_forward_matches_functional(self, device):
        """Test that output matches torch.nn.functional.linear."""
        linear = Linear()
        x = torch.randn(3, 5, 10, device=device)
        weight = torch.randn(20, 10, device=device)
        
        output = linear(x, weight)
        expected = torch.nn.functional.linear(x, weight)
        
        assert_tensors_close(output, expected)
    
    def test_gradient_flow(self, device):
        """Test gradients flow correctly."""
        linear = Linear()
        x = torch.randn(2, 3, 4, device=device, requires_grad=True)
        weight = torch.randn(5, 4, device=device, requires_grad=True)
        
        output = linear(x, weight)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert weight.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.all(weight.grad == 0)
    
    def test_different_dtypes(self, device):
        """Test with different data types."""
        linear = Linear()
        
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                continue  # bfloat16 may not be supported on CPU
            
            x = torch.randn(2, 3, 4, device=device, dtype=dtype)
            weight = torch.randn(5, 4, device=device, dtype=dtype)
            
            output = linear(x, weight)
            assert output.dtype == dtype


class TestTiedLinear:
    """Test suite for TiedLinear module."""
    
    def test_initialization_with_valid_module(self, device):
        """Test TiedLinear initializes with a module that has weight."""
        tied_module = nn.Linear(10, 20, bias=False).to(device)
        tied_linear = TiedLinear(tied_module)
        
        assert tied_linear.tied_module is tied_module
        assert isinstance(tied_linear.linear, Linear)
    
    def test_initialization_without_weight_raises_error(self):
        """Test TiedLinear raises error when module has no weight attribute."""
        # Create a module without weight attribute
        class NoWeightModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.some_param = nn.Parameter(torch.randn(5, 5))
        
        module = NoWeightModule()
        
        with pytest.raises(AttributeError, match="does not have attribute 'weight'"):
            TiedLinear(module)
    
    def test_forward_basic(self, device):
        """Test basic forward pass."""
        tied_module = nn.Linear(8, 16, bias=False).to(device)
        tied_linear = TiedLinear(tied_module)
        
        x = torch.randn(2, 4, 8, device=device)
        output = tied_linear(x)
        
        assert output.shape == (2, 4, 16)
        assert output.device.type == device.type
    
    def test_weight_sharing(self, device):
        """Test that weights are actually shared with tied module."""
        tied_module = nn.Linear(10, 20, bias=False).to(device)
        tied_linear = TiedLinear(tied_module)
        
        x = torch.randn(3, 10, device=device)
        
        # Compute output using TiedLinear
        output_tied = tied_linear(x)
        
        # Compute output using the tied module directly
        output_direct = tied_module(x)
        
        # Should be identical since they share weights
        assert_tensors_close(output_tied, output_direct)
    
    def test_weight_updates_propagate(self, device):
        """Test that weight updates in tied module affect TiedLinear."""
        tied_module = nn.Linear(5, 10, bias=False).to(device)
        tied_linear = TiedLinear(tied_module)
        
        x = torch.randn(2, 5, device=device)
        
        # Get initial output
        output_before = tied_linear(x).clone()
        
        # Modify the tied module's weight
        with torch.no_grad():
            tied_module.weight.add_(1.0)
        
        # Get new output
        output_after = tied_linear(x)
        
        # Outputs should be different
        assert not torch.allclose(output_before, output_after)
    
    def test_gradient_flow(self, device):
        """Test gradients flow back to tied module weights."""
        tied_module = nn.Linear(8, 12, bias=False).to(device)
        tied_linear = TiedLinear(tied_module)
        
        x = torch.randn(2, 3, 8, device=device, requires_grad=True)
        
        output = tied_linear(x)
        loss = output.sum()
        loss.backward()
        
        # Check input has gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
        # Check tied module's weight has gradients
        assert tied_module.weight.grad is not None
        assert not torch.all(tied_module.weight.grad == 0)
    
    def test_multiple_tied_linear_share_weights(self, device):
        """Test multiple TiedLinear instances sharing the same module."""
        tied_module = nn.Linear(6, 8, bias=False).to(device)
        tied_linear1 = TiedLinear(tied_module)
        tied_linear2 = TiedLinear(tied_module)
        
        x = torch.randn(2, 6, device=device)
        
        output1 = tied_linear1(x)
        output2 = tied_linear2(x)
        
        # Both should produce identical outputs
        assert_tensors_close(output1, output2)
    
    def test_callable_interface(self, device):
        """Test that TiedLinear works as a callable."""
        tied_module = nn.Linear(4, 6, bias=False).to(device)
        tied_linear = TiedLinear(tied_module)
        
        x = torch.randn(2, 3, 4, device=device)
        
        # Should be callable directly
        output = tied_linear(x)
        assert output.shape == (2, 3, 6)
    
    def test_with_bias_ignored(self, device):
        """Test that bias in tied module is ignored."""
        # Create module with bias
        tied_module = nn.Linear(5, 10, bias=True).to(device)
        tied_module.bias.data.fill_(5.0)  # Set bias to non-zero
        
        tied_linear = TiedLinear(tied_module)
        
        x = torch.randn(2, 5, device=device)
        
        # TiedLinear should use only weight, not bias
        output_tied = tied_linear(x)
        expected = torch.nn.functional.linear(x, tied_module.weight)  # No bias
        
        assert_tensors_close(output_tied, expected)
        
        # Should be different from using the module directly (which includes bias)
        output_with_bias = tied_module(x)
        assert not torch.allclose(output_tied, output_with_bias)

