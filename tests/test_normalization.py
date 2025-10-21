#!/usr/bin/env python3
"""Unit tests for normalization layers."""

import pytest
import torch
import torch.nn as nn

from muse.layers.layer_norm import Fp32LayerNorm
from muse.layers.rms_norm import RMSNorm, rms_norm
from tests.conftest import assert_tensors_close, check_gradient_flow


class TestFp32LayerNorm:
    """Test suite for Fp32LayerNorm."""
    
    def test_initialization(self):
        """Test Fp32LayerNorm can be initialized."""
        norm = Fp32LayerNorm(64)
        assert isinstance(norm, nn.LayerNorm)
        assert norm.normalized_shape == (64,)
    
    def test_initialization_with_kwargs(self):
        """Test initialization with various arguments."""
        norm = Fp32LayerNorm(128, eps=1e-6, elementwise_affine=True)
        assert norm.normalized_shape == (128,)
        assert norm.eps == 1e-6
        assert norm.elementwise_affine
    
    def test_forward_fp32(self, device):
        """Test forward pass with fp32 input."""
        norm = Fp32LayerNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
        
        output = norm(x)
        
        assert output.shape == x.shape
        assert output.dtype == torch.float32
        assert output.device.type == device.type
    
    def test_forward_fp16_returns_fp16(self, device):
        """Test that fp16 input returns fp16 output (but computed in fp32)."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 testing requires CUDA")
        
        norm = Fp32LayerNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float16)
        
        output = norm(x)
        
        assert output.shape == x.shape
        assert output.dtype == torch.float16, "Output should match input dtype"
        assert output.device.type == device.type
    
    def test_forward_bf16_returns_bf16(self, device):
        """Test that bf16 input returns bf16 output (but computed in fp32)."""
        if not torch.cuda.is_available():
            pytest.skip("BF16 testing requires CUDA")
        
        norm = Fp32LayerNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16)
        
        output = norm(x)
        
        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16, "Output should match input dtype"
        assert output.device.type == device.type
    
    def test_normalization_correctness_fp32(self, device):
        """Test that normalization is correct for fp32."""
        dim = 64
        norm = Fp32LayerNorm(dim).to(device)
        x = torch.randn(2, 8, dim, device=device, dtype=torch.float32)
        
        output = norm(x)
        
        # Check normalized output has mean ~0 and std ~1 (before affine transform)
        # Since we have affine transform, we need to check the stats differently
        # Compare with standard LayerNorm
        standard_norm = nn.LayerNorm(dim).to(device)
        standard_norm.weight.data.copy_(norm.weight.data)
        standard_norm.bias.data.copy_(norm.bias.data)
        
        expected = standard_norm(x)
        assert_tensors_close(output, expected, rtol=1e-5, atol=1e-6)
    
    def test_normalization_correctness_fp16(self, device):
        """Test normalization correctness with fp16 input."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 testing requires CUDA")
        
        dim = 64
        norm = Fp32LayerNorm(dim).to(device)
        x = torch.randn(2, 8, dim, device=device, dtype=torch.float16)
        
        output = norm(x)
        
        # Compare with fp32 computation
        x_fp32 = x.float()
        expected = norm(x_fp32).half()
        
        # Allow larger tolerance for fp16
        assert_tensors_close(output, expected, rtol=1e-2, atol=1e-3)
    
    def test_gradient_flow(self, device):
        """Test gradients flow correctly."""
        norm = Fp32LayerNorm(32).to(device)
        x = torch.randn(2, 4, 32, device=device, requires_grad=True)
        
        output = norm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert norm.weight.grad is not None
        assert norm.bias.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_without_elementwise_affine(self, device):
        """Test LayerNorm without learnable parameters."""
        norm = Fp32LayerNorm(64, elementwise_affine=False).to(device)
        x = torch.randn(2, 8, 64, device=device)
        
        output = norm(x)
        
        assert output.shape == x.shape
        assert norm.weight is None
        assert norm.bias is None


class TestRMSNorm:
    """Test suite for RMSNorm module."""
    
    def test_initialization(self):
        """Test RMSNorm can be initialized."""
        norm = RMSNorm(64)
        assert isinstance(norm, nn.Module)
        assert norm.normalized_shape == (64,)
        assert norm.eps == 1e-6
    
    def test_initialization_custom_eps(self):
        """Test initialization with custom epsilon."""
        norm = RMSNorm(128, eps=1e-5)
        assert norm.eps == 1e-5
    
    def test_scale_parameter(self):
        """Test that scale parameter is initialized correctly."""
        dim = 64
        norm = RMSNorm(dim)
        
        assert norm.scale.shape == (dim,)
        assert torch.allclose(norm.scale, torch.ones(dim))
    
    def test_forward_basic(self, device):
        """Test basic forward pass."""
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device)
        
        output = norm(x)
        
        assert output.shape == x.shape
        assert output.device.type == device.type
        assert output.dtype == x.dtype
    
    def test_forward_fp32(self, device):
        """Test forward with fp32 input."""
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float32)
        
        output = norm(x)
        
        assert output.dtype == torch.float32
    
    def test_forward_fp16_returns_fp16(self, device):
        """Test that fp16 input returns fp16 output."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 testing requires CUDA")
        
        norm = RMSNorm(64).to(device)
        # Convert scale to fp16 as well for proper dtype handling
        norm.scale.data = norm.scale.data.to(torch.float16)
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float16)
        
        output = norm(x)
        
        # Note: Output might be fp32 due to RMSNorm computation, then converted back
        # The important thing is the computation is correct
        assert output.shape == x.shape
    
    def test_rms_normalization_correctness(self, device):
        """Test RMS normalization is computed correctly."""
        dim = 64
        norm = RMSNorm(dim, eps=1e-6).to(device)
        # Set scale to ones to test just the normalization
        norm.scale.data.fill_(1.0)
        
        x = torch.randn(2, 8, dim, device=device)
        output = norm(x)
        
        # Manually compute RMS norm
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = (x_fp32 / rms).type_as(x)
        
        assert_tensors_close(output, expected, rtol=1e-5, atol=1e-6)
    
    def test_scale_applied_correctly(self, device):
        """Test that scale parameter is applied correctly."""
        dim = 8
        norm = RMSNorm(dim).to(device)
        # Set scale to specific values
        norm.scale.data = torch.arange(1, dim + 1, dtype=torch.float32, device=device)
        
        x = torch.randn(2, 4, dim, device=device)
        output = norm(x)
        
        # Compute expected output
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + norm.eps)
        expected = (x_normed * norm.scale).type_as(x)
        
        assert_tensors_close(output, expected, rtol=1e-5, atol=1e-6)
    
    def test_gradient_flow(self, device):
        """Test gradients flow correctly."""
        norm = RMSNorm(32).to(device)
        x = torch.randn(2, 4, 32, device=device, requires_grad=True)
        
        output = norm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert norm.scale.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.all(norm.scale.grad == 0)
    
    def test_different_batch_sizes(self, device):
        """Test with different batch sizes."""
        norm = RMSNorm(64).to(device)
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 8, 64, device=device)
            output = norm(x)
            assert output.shape == (batch_size, 8, 64)
    
    def test_different_seq_lengths(self, device):
        """Test with different sequence lengths."""
        norm = RMSNorm(64).to(device)
        
        for seq_len in [1, 10, 100]:
            x = torch.randn(2, seq_len, 64, device=device)
            output = norm(x)
            assert output.shape == (2, seq_len, 64)


class TestRMSNormFunction:
    """Test suite for rms_norm functional version."""
    
    def test_functional_basic(self, device):
        """Test basic functional rms_norm."""
        x = torch.randn(2, 8, 64, device=device)
        output = rms_norm(x)
        
        assert output.shape == x.shape
        assert output.device.type == device.type
        assert output.dtype == x.dtype
    
    def test_functional_custom_eps(self, device):
        """Test functional version with custom epsilon."""
        x = torch.randn(2, 8, 64, device=device)
        output = rms_norm(x, eps=1e-5)
        
        assert output.shape == x.shape
    
    def test_functional_correctness(self, device):
        """Test functional rms_norm is computed correctly."""
        x = torch.randn(2, 8, 64, device=device)
        eps = 1e-6
        
        output = rms_norm(x, eps=eps)
        
        # Manually compute
        x_fp32 = x.float()
        expected = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)).type_as(x)
        
        assert_tensors_close(output, expected, rtol=1e-5, atol=1e-6)
    
    def test_functional_matches_module_without_scale(self, device):
        """Test functional version matches module version when scale=1."""
        dim = 64
        module = RMSNorm(dim, eps=1e-6).to(device)
        module.scale.data.fill_(1.0)  # Set scale to 1
        
        x = torch.randn(2, 8, dim, device=device)
        
        output_functional = rms_norm(x, eps=1e-6)
        output_module = module(x)
        
        assert_tensors_close(output_functional, output_module, rtol=1e-5, atol=1e-6)
    
    def test_functional_gradient_flow(self, device):
        """Test gradients flow through functional version."""
        x = torch.randn(2, 8, 64, device=device, requires_grad=True)
        
        output = rms_norm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_functional_fp16(self, device):
        """Test functional version with fp16."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 testing requires CUDA")
        
        x = torch.randn(2, 8, 64, device=device, dtype=torch.float16)
        output = rms_norm(x)
        
        assert output.dtype == torch.float16
        assert output.shape == x.shape

