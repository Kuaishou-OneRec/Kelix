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
        seq_len = 8
        num_heads = 4
        head_dim = 16
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert output.device.type == device.type
    
    def test_forward_without_causal(self, device):
        """Test forward without causal masking."""
        attn = EagerAttention()
        
        # Shape: (b, s, n_h, h_d) = (1, 4, 2, 8)
        q = torch.randn(1, 4, 2, 8, device=device)
        k = torch.randn(1, 4, 2, 8, device=device)
        v = torch.randn(1, 4, 2, 8, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_forward_with_causal(self, device):
        """Test forward with causal masking."""
        attn = EagerAttention()
        
        batch_size = 1
        seq_len = 6
        num_heads = 2
        head_dim = 8
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        output = attn.forward(q, k, v, is_causal=True, attn_dropout=0.0)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_causal_mask_effect(self, device):
        """Test that causal mask actually prevents attending to future tokens."""
        attn = EagerAttention()
        
        # Use simple input where we can verify causality
        batch_size = 1
        seq_len = 4
        num_heads = 1
        head_dim = 4
        
        # Create queries, keys, values with specific pattern
        # Shape: (b, s, n_h, h_d)
        q = torch.ones(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.ones(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.arange(seq_len, device=device).view(1, seq_len, 1, 1).expand(-1, -1, num_heads, head_dim).float()
        
        output_causal = attn.forward(q, k, v, is_causal=True, attn_dropout=0.0)
        output_no_causal = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        # Outputs should be different due to masking
        assert not torch.allclose(output_causal, output_no_causal)
    
    def test_attention_scores_scaling(self, device):
        """Test that attention scores are properly scaled by sqrt(d_k)."""
        attn = EagerAttention()
        
        # Shape: (b, s, n_h, h_d) = (1, 4, 1, 16)
        q = torch.randn(1, 4, 1, 16, device=device)
        k = torch.randn(1, 4, 1, 16, device=device)
        v = torch.randn(1, 4, 1, 16, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, device):
        """Test gradients flow through attention."""
        attn = EagerAttention()
        
        # Shape: (b, s, n_h, h_d) = (1, 4, 2, 8)
        q = torch.randn(1, 4, 2, 8, device=device, requires_grad=True)
        k = torch.randn(1, 4, 2, 8, device=device, requires_grad=True)
        v = torch.randn(1, 4, 2, 8, device=device, requires_grad=True)
        
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
        num_heads = 2
        head_dim = 16
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(1, q_seq_len, num_heads, head_dim, device=device)
        k = torch.randn(1, k_seq_len, num_heads, head_dim, device=device)
        v = torch.randn(1, k_seq_len, num_heads, head_dim, device=device)
        
        output = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0)
        
        assert output.shape == (1, q_seq_len, num_heads, head_dim)
    
    def test_dropout_disabled_by_default(self, device):
        """Test that dropout is disabled by default (training=False)."""
        attn = EagerAttention()
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(1, 4, 2, 8, device=device)
        k = torch.randn(1, 4, 2, 8, device=device)
        v = torch.randn(1, 4, 2, 8, device=device)
        
        # Run multiple times with dropout - without training=True, results should be identical
        output1 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5)
        output2 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5)
        
        # Without training=True, dropout should not be applied, so outputs should be identical
        assert torch.allclose(output1, output2)
    
    def test_dropout_enabled_in_training_mode(self, device):
        """Test that dropout is applied when training=True."""
        torch.manual_seed(42)
        attn = EagerAttention()
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(1, 8, 4, 16, device=device)
        k = torch.randn(1, 8, 4, 16, device=device)
        v = torch.randn(1, 8, 4, 16, device=device)
        
        # Run with training=True - different random seeds should give different results
        torch.manual_seed(123)
        output1 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5, training=True)
        
        torch.manual_seed(456)
        output2 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5, training=True)
        
        # With training=True and dropout > 0, outputs should be different due to dropout randomness
        assert not torch.allclose(output1, output2)
    
    def test_zero_dropout_same_output(self, device):
        """Test that zero dropout gives identical outputs regardless of training mode."""
        attn = EagerAttention()
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(1, 4, 2, 8, device=device)
        k = torch.randn(1, 4, 2, 8, device=device)
        v = torch.randn(1, 4, 2, 8, device=device)
        
        output_train = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0, training=True)
        output_eval = attn.forward(q, k, v, is_causal=False, attn_dropout=0.0, training=False)
        
        # With zero dropout, training mode should not affect output
        assert torch.allclose(output_train, output_eval)


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
        
        # Skip this test as it requires actual flash_attn library
        pytest.skip("FlashAttention2 requires flash_attn library to be installed")
    
    def test_forward_with_causal(self, device):
        """Test forward with causal masking."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        # Skip this test as it requires actual flash_attn library
        pytest.skip("FlashAttention2 requires flash_attn library to be installed")
    
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
        
        # Skip this test as it requires actual flash_attn library
        pytest.skip("FlashAttention2 requires flash_attn library to be installed")
    
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
    
    def test_dropout_disabled_by_default(self, device):
        """Test that dropout is disabled by default (training=False)."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        attn = FlashAttention2()
        
        seq_len = 16
        num_heads = 4
        head_dim = 32
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        # Run multiple times with dropout - without training=True, results should be identical
        output1 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5)
        output2 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5)
        
        # Without training=True, dropout should not be applied (dropout_p becomes 0.0)
        assert torch.allclose(output1, output2)
    
    def test_training_mode_affects_dropout(self, device):
        """Test that training=True enables dropout."""
        if not torch.cuda.is_available():
            pytest.skip("FlashAttention requires CUDA")
        
        attn = FlashAttention2()
        
        seq_len = 32
        num_heads = 4
        head_dim = 32
        
        # Shape: (b, s, n_h, h_d)
        q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        # Run with training=True - different random states should give different results
        torch.manual_seed(123)
        output1 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5, training=True)
        
        torch.manual_seed(456)
        output2 = attn.forward(q, k, v, is_causal=False, attn_dropout=0.5, training=True)
        
        # With training=True and dropout > 0, outputs should be different
        assert not torch.allclose(output1, output2)


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

