#!/usr/bin/env python3
"""Unit tests for MultiHeadAttention layer."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from muse.layers.attention import MultiHeadAttention
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.rms_norm import RMSNorm
from tests.conftest import get_kv_cache, assert_tensors_close


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention."""
    
    def test_initialization_mha(self, device):
        """Test MHA initialization (num_heads == num_kv_heads)."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        assert attn.num_heads == num_heads
        assert attn.num_kv_heads == num_heads
        assert attn.head_dim == head_dim
        assert attn.embed_dim == embed_dim
    
    def test_initialization_gqa(self, device):
        """Test GQA initialization (num_heads > num_kv_heads > 1)."""
        embed_dim = 64
        num_heads = 8
        num_kv_heads = 2
        head_dim = 8
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        assert attn.num_heads == num_heads
        assert attn.num_kv_heads == num_kv_heads
    
    def test_initialization_mqa(self, device):
        """Test MQA initialization (num_kv_heads == 1)."""
        embed_dim = 64
        num_heads = 4
        num_kv_heads = 1
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        assert attn.num_heads == num_heads
        assert attn.num_kv_heads == 1
    
    def test_invalid_num_heads_raises_error(self, device):
        """Test that invalid num_heads raises ValueError."""
        embed_dim = 64
        num_heads = 5
        num_kv_heads = 2  # 5 % 2 != 0
        head_dim = 16
        
        with pytest.raises(ValueError, match="num_heads .* must be divisible by"):
            MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
                output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            )
    
    def test_invalid_embed_dim_raises_error(self, device):
        """Test that embed_dim not divisible by num_heads raises ValueError."""
        embed_dim = 65  # Not divisible by 4
        num_heads = 4
        head_dim = 16
        
        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
            MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            )
    
    def test_invalid_attn_dropout_raises_error(self, device):
        """Test that invalid attn_dropout raises ValueError."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        with pytest.raises(ValueError, match="attn_dropout .* must be between"):
            MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
                attn_dropout=1.5,  # Invalid
            )
    
    def test_qk_norm_must_be_set_together(self, device):
        """Test that q_norm and k_norm must be set together."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        # Only q_norm set
        with pytest.raises(ValueError, match="q and k norm must be set together"):
            MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
                output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
                q_norm=RMSNorm(head_dim),
                k_norm=None,
            )
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_self_attention(self, mock_sp, device):
        """Test forward pass for self-attention."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        output = attn(x, x)
        
        assert output.shape == (2, 8, embed_dim)
        assert output.device == device
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_with_positional_embeddings(self, mock_sp, device):
        """Test forward with positional embeddings."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=128).to(device)
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            pos_embeddings=rope,
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        output = attn(x, x)
        
        assert output.shape == (2, 8, embed_dim)
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_with_qk_norm(self, mock_sp, device):
        """Test forward with query and key normalization."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            q_norm=RMSNorm(head_dim).to(device),
            k_norm=RMSNorm(head_dim).to(device),
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        output = attn(x, x)
        
        assert output.shape == (2, 8, embed_dim)
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_gqa(self, mock_sp, device):
        """Test forward pass with GQA."""
        embed_dim = 64
        num_heads = 8
        num_kv_heads = 2
        head_dim = 8
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        output = attn(x, x)
        
        assert output.shape == (2, 8, embed_dim)
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_gradient_flow(self, mock_sp, device):
        """Test gradients flow through attention."""
        embed_dim = 32
        num_heads = 4
        head_dim = 8
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        x = torch.randn(2, 4, embed_dim, device=device, requires_grad=True)
        output = attn(x, x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_setup_cache(self, device):
        """Test KV cache setup."""
        embed_dim = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        assert attn.kv_cache is None
        assert not attn.cache_enabled
        
        # Setup cache
        attn.setup_cache(batch_size=2, dtype=torch.float32, max_seq_len=128)
        
        assert attn.kv_cache is not None
        assert attn.cache_enabled
    
    def test_reset_cache(self, device):
        """Test KV cache reset."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        attn.setup_cache(batch_size=2, dtype=torch.float32, max_seq_len=128)
        
        # Reset should work
        attn.reset_cache()
        assert attn.kv_cache.cache_pos == 0
    
    def test_reset_cache_without_setup_raises_error(self, device):
        """Test that reset_cache raises error if cache not setup."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        with pytest.raises(RuntimeError, match="Key value caches are not setup"):
            attn.reset_cache()
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_without_y_and_no_cache_raises_error(self, mock_sp, device):
        """Test that forward without y and no cache raises ValueError."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        
        with pytest.raises(ValueError, match="Must provide y input or use kv_cache"):
            attn(x, y=None)
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_is_causal_parameter(self, mock_sp, device):
        """Test is_causal parameter is properly stored."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        # Test with is_causal=True (default)
        attn_causal = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            is_causal=True,
        )
        assert attn_causal.is_causal is True
        
        # Test with is_causal=False
        attn_no_causal = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            is_causal=False,
        )
        assert attn_no_causal.is_causal is False

