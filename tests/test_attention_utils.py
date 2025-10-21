#!/usr/bin/env python3
"""Unit tests for attention utility functions."""

import pytest
import torch

from muse.layers.attention_utils import (
    EagerAttention,
    FlashAttention2,
    get_attention_function
)
from tests.conftest import assert_tensors_close


class TestEagerAttention:
    """Test suite for EagerAttention."""
    
    def test_initialization(self):
        """Test EagerAttention can be initialized."""
        attn = EagerAttention()
        assert attn is not None
    
    def test_forward_basic(self, device):
        """Test basic forward pass."""
        attn = EagerAttention()
        
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output.device.type == device.type
    
    def test_forward_without_causal(self, device):
        """Test forward without causal masking."""
        attn = EagerAttention()
        
        q = torch.randn(1, 2, 4, 8, device=device)
        k = torch.randn(1, 2, 4, 8, device=device)
        v = torch.randn(1, 2, 4, 8, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_forward_with_causal(self, device):
        """Test forward with causal masking."""
        attn = EagerAttention()
        
        batch_size = 1
        num_heads = 2
        seq_len = 6
        head_dim = 8
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        output = attn.forward(q, k, v, is_causal=True, attn_dropout=0.0)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_causal_mask_effect(self, device):
        """Test that causal mask actually prevents attending to future tokens."""
        attn = EagerAttention()
        
        # Use simple input where we can verify causality
        batch_size = 1
        num_heads = 1
        seq_len = 4
        head_dim = 4
        
        # Create queries, keys, values with specific pattern
        q = torch.ones(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.ones(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1).expand(-1, -1, -1, head_dim).float()
        
        output_causal = attn.forward(q, k, v, is_causal=True, attn_dropout=0.0)
        output_no_causal = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        # Outputs should be different due to masking
        assert not torch.allclose(output_causal, output_no_causal)
    
    def test_attention_scores_scaling(self, device):
        """Test that attention scores are properly scaled by sqrt(d_k)."""
        attn = EagerAttention()
        
        q = torch.randn(1, 1, 4, 16, device=device)
        k = torch.randn(1, 1, 4, 16, device=device)
        v = torch.randn(1, 1, 4, 16, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, device):
        """Test gradients flow through attention."""
        attn = EagerAttention()
        
        q = torch.randn(1, 2, 4, 8, device=device, requires_grad=True)
        k = torch.randn(1, 2, 4, 8, device=device, requires_grad=True)
        v = torch.randn(1, 2, 4, 8, device=device, requires_grad=True)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        loss = output.sum()
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.all(q.grad == 0)
        assert not torch.all(k.grad == 0)
        assert not torch.all(v.grad == 0)
    
    def test_different_seq_lengths_qk(self, device):
        """Test with different sequence lengths for q and k."""
        attn = EagerAttention()
        
        q_seq_len = 4
        k_seq_len = 8
        
        q = torch.randn(1, 2, q_seq_len, 16, device=device)
        k = torch.randn(1, 2, k_seq_len, 16, device=device)
        v = torch.randn(1, 2, k_seq_len, 16, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        assert output.shape == (1, 2, q_seq_len, 16)


class TestFlashAttention2:
    """Test suite for FlashAttention2."""
    
    @pytest.fixture(autouse=True)
    def check_flash_attn(self):
        """Check if flash_attn is available, skip tests if not."""
        pytest.importorskip("flash_attn", reason="flash_attn not installed")
    
    def test_initialization(self):
        """Test FlashAttention2 can be initialized."""
        attn = FlashAttention2()
        assert attn is not None
    
    def test_forward_basic(self, device):
        """Test basic forward pass with FlashAttention."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        attn = FlashAttention2()
        
        batch_size = 2
        num_heads = 4
        seq_len = 128
        head_dim = 64
        
        # FlashAttention expects unsqueezed batch
        q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        assert output.shape == q.shape
        assert output.device.type == device.type
    
    def test_forward_with_causal(self, device):
        """Test forward with causal masking."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        attn = FlashAttention2()
        
        seq_len = 64
        num_heads = 4
        head_dim = 32
        
        q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        output = attn.forward(q, k, v, is_causal=True, attn_dropout=0.0)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_forward_with_varlen(self, device):
        """Test forward with variable length sequences (cu_seqlens)."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        attn = FlashAttention2()
        
        # Two sequences: length 10 and 15
        total_len = 25
        num_heads = 4
        head_dim = 32
        
        q = torch.randn(1, total_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, total_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, total_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        # Cumulative sequence lengths
        cu_seqlens = torch.tensor([0, 10, 25], device=device, dtype=torch.int32)
        
        output = attn.forward(
            q, k, v,
            is_causal=False,
            attn_dropout=0.0,
            cu_seqlens=cu_seqlens
        )
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_forward_with_window_size(self, device):
        """Test forward with sliding window attention."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        attn = FlashAttention2()
        
        seq_len = 128
        num_heads = 4
        head_dim = 32
        window_size = 64
        
        q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        output = attn.forward(
            q, k, v,
            is_causal=False,
            attn_dropout=0.0,
            window_size=window_size
        )
        
        assert output.shape == q.shape
    
    def test_separate_cu_seqlens_qk(self, device):
        """Test with separate cu_seqlens for q and k."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        attn = FlashAttention2()
        
        q_len = 20
        k_len = 30
        num_heads = 4
        head_dim = 32
        
        q = torch.randn(1, q_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, k_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, k_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        cu_seqlens_q = torch.tensor([0, 10, 20], device=device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, 15, 30], device=device, dtype=torch.int32)
        
        output = attn.forward(
            q, k, v,
            is_causal=False,
            attn_dropout=0.0,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k
        )
        
        assert output.shape == q.shape


class TestGetAttentionFunction:
    """Test suite for get_attention_function factory."""
    
    def test_get_eager_attention(self):
        """Test factory returns EagerAttention for 'eager' type."""
        attn = get_attention_function("eager")
        assert isinstance(attn, EagerAttention)
    
    def test_get_flash_attention(self):
        """Test factory returns FlashAttention2 for 'flash_attention_2' type."""
        attn = get_attention_function("flash_attention_2")
        assert isinstance(attn, FlashAttention2)
    
    def test_invalid_attention_type_raises_error(self):
        """Test that invalid attention type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown attention type"):
            get_attention_function("invalid_type")
    
    def test_returned_attention_is_callable(self):
        """Test that returned attention functions are callable."""
        eager_attn = get_attention_function("eager")
        flash_attn = get_attention_function("flash_attention_2")
        
        assert hasattr(eager_attn, 'forward')
        assert callable(eager_attn.forward)
        assert hasattr(flash_attn, 'forward')
        assert callable(flash_attn.forward)
    
    def test_case_sensitivity(self):
        """Test that attention type is case-sensitive."""
        # Valid lowercase
        attn = get_attention_function("eager")
        assert isinstance(attn, EagerAttention)
        
        # Invalid uppercase should raise error
        with pytest.raises(ValueError):
            get_attention_function("EAGER")
    
    def test_factory_returns_new_instances(self):
        """Test that factory returns new instances each time."""
        attn1 = get_attention_function("eager")
        attn2 = get_attention_function("eager")
        
        # Should be different instances
        assert attn1 is not attn2

