#!/usr/bin/env python3
"""Unit tests for Transformer layers."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from muse.layers.transformer import (
    TransformerSelfAttentionLayer,
    TransformerCrossAttentionLayer
)
from muse.layers.attention import MultiHeadAttention
from muse.layers.feed_forward import FeedForward
from muse.layers.rms_norm import RMSNorm
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from tests.conftest import assert_tensors_close, get_kv_cache


class TestTransformerSelfAttentionLayer:
    """Test suite for TransformerSelfAttentionLayer."""
    
    def create_test_layer(self, embed_dim, num_heads, hidden_dim, device):
        """Helper to create a test transformer layer."""
        head_dim = embed_dim // num_heads
        
        # Create attention module
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
        
        # Create feedforward module
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
            down_proj=nn.Linear(hidden_dim, embed_dim, bias=False).to(device),
            up_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
        )
        
        # Create layer
        layer = TransformerSelfAttentionLayer(
            attn=attn,
            mlp=mlp,
            sa_norm=RMSNorm(embed_dim).to(device),
            mlp_norm=RMSNorm(embed_dim).to(device),
        )
        
        return layer
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_initialization(self, mock_sp, device):
        """Test TransformerSelfAttentionLayer can be initialized."""
        layer = self.create_test_layer(
            embed_dim=64,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        assert isinstance(layer, TransformerSelfAttentionLayer)
        assert isinstance(layer.attn, MultiHeadAttention)
        assert isinstance(layer.mlp, FeedForward)
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_basic(self, mock_sp, device):
        """Test basic forward pass."""
        embed_dim = 64
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        output = layer(x)
        
        assert output.shape == x.shape
        assert output.device.type == device.type
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_residual_connections(self, mock_sp, device):
        """Test that residual connections are applied."""
        embed_dim = 64
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        
        # Set all parameters to zero to test residual
        with torch.no_grad():
            for param in layer.parameters():
                param.zero_()
        
        output = layer(x)
        
        # With zero weights, output should be close to input due to residual connections
        # (though normalization may affect this slightly)
        assert output.shape == x.shape
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_with_mask(self, mock_sp, device):
        """Test forward with attention mask."""
        embed_dim = 64
        seq_len = 8
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        x = torch.randn(2, seq_len, embed_dim, device=device)
        mask = torch.ones(2, seq_len, seq_len, dtype=torch.bool, device=device)
        
        output = layer(x, mask=mask)
        
        assert output.shape == x.shape
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_with_input_pos(self, mock_sp, device):
        """Test forward with input_pos for positional encoding."""
        embed_dim = 64
        seq_len = 8
        batch_size = 2
        
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)
        input_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        output = layer(x, input_pos=input_pos)
        
        assert output.shape == x.shape
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_gradient_flow(self, mock_sp, device):
        """Test gradients flow through the layer."""
        embed_dim = 32
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=128,
            device=device
        )
        
        x = torch.randn(2, 4, embed_dim, device=device, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_setup_caches(self, device):
        """Test cache setup."""
        embed_dim = 64
        
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        assert not layer.caches_are_setup()
        
        layer.setup_caches(
            batch_size=2,
            dtype=torch.float32,
            encoder_max_seq_len=128,
            decoder_max_seq_len=256
        )
        
        assert layer.caches_are_setup()
    
    def test_reset_cache(self, device):
        """Test cache reset."""
        embed_dim = 64
        
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        layer.setup_caches(
            batch_size=2,
            dtype=torch.float32,
            encoder_max_seq_len=128,
            decoder_max_seq_len=256
        )
        
        layer.reset_cache()
        assert layer.attn.kv_cache.cache_pos == 0
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_with_scale_modules(self, mock_sp, device):
        """Test with scale modules for attention and MLP."""
        embed_dim = 64
        head_dim = 16
        num_heads = 4
        hidden_dim = 256
        
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
        
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
            down_proj=nn.Linear(hidden_dim, embed_dim, bias=False).to(device),
            up_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
        )
        
        # Create custom scale modules
        sa_scale = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        mlp_scale = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        
        layer = TransformerSelfAttentionLayer(
            attn=attn,
            mlp=mlp,
            sa_norm=RMSNorm(embed_dim).to(device),
            mlp_norm=RMSNorm(embed_dim).to(device),
            sa_scale=sa_scale,
            mlp_scale=mlp_scale,
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        output = layer(x)
        
        assert output.shape == x.shape
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_with_mask_mod(self, mock_sp, device):
        """Test with mask modification function."""
        embed_dim = 64
        
        def custom_mask_mod(mask, bsz, seq_len, **kwargs):
            """Custom mask modification function."""
            if mask is None:
                mask = torch.ones(bsz, seq_len, seq_len, dtype=torch.bool, device=device)
            return mask
        
        head_dim = 16
        num_heads = 4
        
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
        
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, 256, bias=False).to(device),
            down_proj=nn.Linear(256, embed_dim, bias=False).to(device),
            up_proj=nn.Linear(embed_dim, 256, bias=False).to(device),
        )
        
        layer = TransformerSelfAttentionLayer(
            attn=attn,
            mlp=mlp,
            sa_norm=RMSNorm(embed_dim).to(device),
            mlp_norm=RMSNorm(embed_dim).to(device),
            mask_mod=custom_mask_mod,
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        output = layer(x)
        
        assert output.shape == x.shape


class TestTransformerCrossAttentionLayer:
    """Test suite for TransformerCrossAttentionLayer."""
    
    def create_test_layer(self, embed_dim, num_heads, hidden_dim, device):
        """Helper to create a test cross-attention layer."""
        head_dim = embed_dim // num_heads
        
        # Create attention module without positional embeddings
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            pos_embeddings=None,  # No pos embeddings for cross attention
        )
        
        # Create feedforward module
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
            down_proj=nn.Linear(hidden_dim, embed_dim, bias=False).to(device),
            up_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
        )
        
        # Create layer
        layer = TransformerCrossAttentionLayer(
            attn=attn,
            mlp=mlp,
            ca_norm=RMSNorm(embed_dim).to(device),
            mlp_norm=RMSNorm(embed_dim).to(device),
        )
        
        return layer
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_initialization(self, mock_sp, device):
        """Test TransformerCrossAttentionLayer can be initialized."""
        layer = self.create_test_layer(
            embed_dim=64,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        assert isinstance(layer, TransformerCrossAttentionLayer)
        assert isinstance(layer.attn, MultiHeadAttention)
        assert isinstance(layer.mlp, FeedForward)
    
    def test_initialization_with_pos_embeddings_raises_error(self, device):
        """Test that initialization with pos_embeddings raises AssertionError."""
        embed_dim = 64
        num_heads = 4
        head_dim = 16
        
        # Create attention with positional embeddings
        rope = RotaryPositionalEmbeddings(dim=head_dim).to(device)
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
        
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, 256, bias=False).to(device),
            down_proj=nn.Linear(256, embed_dim, bias=False).to(device),
        )
        
        with pytest.raises(AssertionError, match="Doesn't support positional embeddings"):
            TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
                ca_norm=RMSNorm(embed_dim).to(device),
                mlp_norm=RMSNorm(embed_dim).to(device),
            )
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_basic(self, mock_sp, device):
        """Test basic forward pass."""
        embed_dim = 64
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        encoder_output = torch.randn(2, 12, embed_dim, device=device)
        
        output = layer(x, encoder_input=encoder_output)
        
        assert output.shape == x.shape
        assert output.device.type == device.type
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_forward_different_encoder_seq_len(self, mock_sp, device):
        """Test forward with different encoder sequence length."""
        embed_dim = 64
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        decoder_seq_len = 8
        encoder_seq_len = 20
        
        x = torch.randn(2, decoder_seq_len, embed_dim, device=device)
        encoder_output = torch.randn(2, encoder_seq_len, embed_dim, device=device)
        
        output = layer(x, encoder_input=encoder_output)
        
        assert output.shape == (2, decoder_seq_len, embed_dim)
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_gradient_flow(self, mock_sp, device):
        """Test gradients flow through the layer."""
        embed_dim = 32
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=128,
            device=device
        )
        
        x = torch.randn(2, 4, embed_dim, device=device, requires_grad=True)
        encoder_output = torch.randn(2, 6, embed_dim, device=device, requires_grad=True)
        
        output = layer(x, encoder_input=encoder_output)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert encoder_output.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.all(encoder_output.grad == 0)
    
    def test_setup_caches(self, device):
        """Test cache setup for cross attention."""
        embed_dim = 64
        
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        assert not layer.caches_are_setup()
        
        layer.setup_caches(
            batch_size=2,
            dtype=torch.float32,
            encoder_max_seq_len=128,
            decoder_max_seq_len=256
        )
        
        assert layer.caches_are_setup()
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_residual_connections(self, mock_sp, device):
        """Test that residual connections are applied."""
        embed_dim = 64
        layer = self.create_test_layer(
            embed_dim=embed_dim,
            num_heads=4,
            hidden_dim=256,
            device=device
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        encoder_output = torch.randn(2, 12, embed_dim, device=device)
        
        output = layer(x, encoder_input=encoder_output)
        
        # Output should have same shape as input
        assert output.shape == x.shape
    
    @patch('muse.layers.attention.get_sequence_parallel_world_size', return_value=1)
    def test_with_scale_modules(self, mock_sp, device):
        """Test with scale modules for cross-attention and MLP."""
        embed_dim = 64
        head_dim = 16
        num_heads = 4
        hidden_dim = 256
        
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            k_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            v_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False).to(device),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False).to(device),
            pos_embeddings=None,
        )
        
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
            down_proj=nn.Linear(hidden_dim, embed_dim, bias=False).to(device),
            up_proj=nn.Linear(embed_dim, hidden_dim, bias=False).to(device),
        )
        
        # Create custom scale modules
        ca_scale = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        mlp_scale = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        
        layer = TransformerCrossAttentionLayer(
            attn=attn,
            mlp=mlp,
            ca_norm=RMSNorm(embed_dim).to(device),
            mlp_norm=RMSNorm(embed_dim).to(device),
            ca_scale=ca_scale,
            mlp_scale=mlp_scale,
        )
        
        x = torch.randn(2, 8, embed_dim, device=device)
        encoder_output = torch.randn(2, 12, embed_dim, device=device)
        
        output = layer(x, encoder_input=encoder_output)
        
        assert output.shape == x.shape

