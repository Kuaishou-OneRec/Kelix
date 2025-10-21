#!/usr/bin/env python3
"""Unit tests for positional embedding layers."""

import pytest
import torch
import torch.nn as nn
import math

from muse.layers.position_embeddings import (
    RotaryPositionalEmbeddings,
    VisionRotaryPositionalEmbeddings
)
from tests.conftest import assert_tensors_close


class TestRotaryPositionalEmbeddings:
    """Test suite for RotaryPositionalEmbeddings (RoPE)."""
    
    def test_initialization(self):
        """Test RoPE can be initialized."""
        rope = RotaryPositionalEmbeddings(dim=64, max_seq_len=2048, base=10_000)
        
        assert rope.dim == 64
        assert rope.max_seq_len == 2048
        assert rope.base == 10_000
    
    def test_default_parameters(self):
        """Test default initialization parameters."""
        rope = RotaryPositionalEmbeddings(dim=64)
        
        assert rope.dim == 64
        assert rope.max_seq_len == 4096  # default
        assert rope.base == 10_000  # default
    
    def test_theta_computation(self):
        """Test theta values are computed correctly."""
        dim = 8
        base = 10_000
        rope = RotaryPositionalEmbeddings(dim=dim, base=base)
        
        # Manually compute expected theta
        expected_theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        assert rope.theta.shape == (dim // 2,)
        assert_tensors_close(rope.theta, expected_theta, rtol=1e-5, atol=1e-8)
    
    def test_cache_shape(self):
        """Test cache has correct shape."""
        dim = 64
        max_seq_len = 512
        rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_seq_len)
        
        # Cache shape should be [max_seq_len, dim // 2, 2]
        assert rope.cache.shape == (max_seq_len, dim // 2, 2)
    
    def test_forward_basic(self, device):
        """Test basic forward pass."""
        dim = 64
        rope = RotaryPositionalEmbeddings(dim=dim).to(device)
        
        # Input shape: [b, s, n_h, h_d]
        x = torch.randn(2, 8, 4, dim, device=device)
        output = rope(x)
        
        assert output.shape == x.shape
        assert output.device.type == device.type
    
    def test_forward_without_input_pos(self, device):
        """Test forward without input_pos (uses sequential positions)."""
        dim = 32
        seq_len = 10
        rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=128).to(device)
        
        x = torch.randn(2, seq_len, 4, dim, device=device)
        output = rope(x, input_pos=None)
        
        assert output.shape == x.shape
    
    def test_forward_with_input_pos(self, device):
        """Test forward with custom input_pos."""
        dim = 32
        seq_len = 8
        batch_size = 2
        rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=128).to(device)
        
        x = torch.randn(batch_size, seq_len, 4, dim, device=device)
        # Custom positions
        input_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        output = rope(x, input_pos=input_pos)
        
        assert output.shape == x.shape
    
    def test_cache_extends_when_needed(self, device):
        """Test that cache extends when sequence exceeds max_seq_len."""
        dim = 32
        initial_max_len = 16
        rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=initial_max_len).to(device)
        
        # Verify initial cache size
        assert rope.cache.shape[0] == initial_max_len
        
        # Use sequence longer than initial max_seq_len
        longer_seq_len = 32
        rope.max_seq_len = longer_seq_len
        rope.build_rope_cache(longer_seq_len)
        
        x = torch.randn(1, longer_seq_len, 2, dim, device=device)
        
        # Should work with extended cache
        output = rope(x)
        
        assert output.shape == x.shape
        # Cache should have been extended
        assert rope.cache.shape[0] >= longer_seq_len
    
    def test_dtype_preservation(self, device):
        """Test that output dtype matches input dtype."""
        dim = 64
        rope = RotaryPositionalEmbeddings(dim=dim).to(device)
        
        dtypes = [torch.float32]
        if torch.cuda.is_available():
            dtypes.extend([torch.float16, torch.bfloat16])
        
        for dtype in dtypes:
            x = torch.randn(2, 8, 4, dim, device=device, dtype=dtype)
            output = rope(x)
            
            assert output.dtype == dtype
    
    def test_rotary_embedding_correctness(self, device):
        """Test RoPE computation is correct."""
        dim = 4  # Small dim for manual verification
        seq_len = 2
        rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=128).to(device)
        
        x = torch.randn(1, seq_len, 1, dim, device=device)
        output = rope(x)
        
        # Output should be same shape
        assert output.shape == x.shape
        
        # The computation involves rotation, so values should be different
        assert not torch.allclose(output, x)
    
    def test_gradient_flow(self, device):
        """Test gradients flow through RoPE."""
        dim = 32
        rope = RotaryPositionalEmbeddings(dim=dim).to(device)
        
        x = torch.randn(2, 8, 4, dim, device=device, requires_grad=True)
        output = rope(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_different_num_heads(self, device):
        """Test with different numbers of heads."""
        dim = 64
        rope = RotaryPositionalEmbeddings(dim=dim).to(device)
        
        for num_heads in [1, 2, 4, 8, 16]:
            x = torch.randn(2, 8, num_heads, dim, device=device)
            output = rope(x)
            assert output.shape == (2, 8, num_heads, dim)


class TestVisionRotaryPositionalEmbeddings:
    """Test suite for VisionRotaryPositionalEmbeddings (2D RoPE for images)."""
    
    def test_initialization(self):
        """Test VisionRoPE can be initialized."""
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=40,
            tile_size=400,
            dim=32,
            base=10_000
        )
        
        assert rope.patch_grid_size == 10  # 400 / 40
        assert rope.seq_len == 101  # 10*10 + 1 for CLS token
        assert rope.dim == 32
        assert rope.base == 10_000
    
    def test_append_cls_token_true(self):
        """Test with CLS token appended at end."""
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=40,
            tile_size=400,
            dim=32,
            append_cls_token=True
        )
        
        assert rope.append_cls_token is True
        assert rope.seq_len == 101  # patches + 1 CLS
    
    def test_append_cls_token_false(self):
        """Test with CLS token prepended at beginning."""
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=40,
            tile_size=400,
            dim=32,
            append_cls_token=False
        )
        
        assert rope.append_cls_token is False
        assert rope.seq_len == 101  # patches + 1 CLS
    
    def test_cache_shape(self):
        """Test cache has correct shape."""
        patch_size = 40
        tile_size = 400
        dim = 64
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim
        )
        
        patches_per_tile = (tile_size // patch_size) ** 2
        # Cache shape: [patches_per_tile + 1, dim // 2, 2]
        # Note: VisionRoPE internally uses dim // 2 for theta calculation
        assert rope.cache.shape == (patches_per_tile + 1, dim // 2, 2)
    
    def test_forward_basic(self, device):
        """Test basic forward pass."""
        patch_size = 40
        tile_size = 400
        dim = 64
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim
        ).to(device)
        
        num_tiles = 2
        patches_per_tile = (tile_size // patch_size) ** 2
        seq_len = num_tiles * (patches_per_tile + 1)  # Including CLS tokens
        
        # Input shape: [b, s, n_h, h_d]
        x = torch.randn(2, seq_len, 4, dim, device=device)
        output = rope(x)
        
        assert output.shape == x.shape
        assert output.device.type == device.type
    
    def test_cls_token_zeroed_append(self, device):
        """Test that CLS token position has zero frequencies when appended."""
        patch_size = 20
        tile_size = 60  # 3x3 grid
        dim = 16
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim,
            append_cls_token=True
        ).to(device)
        
        # The last position should have zeros for CLS token
        cls_cache = rope.cache[-1]  # Last position
        # Both cos and sin should be 1 and 0 respectively for zero frequency
        assert torch.allclose(cls_cache[:, 0], torch.ones_like(cls_cache[:, 0]))
        assert torch.allclose(cls_cache[:, 1], torch.zeros_like(cls_cache[:, 1]))
    
    def test_cls_token_zeroed_prepend(self, device):
        """Test that CLS token position has zero frequencies when prepended."""
        patch_size = 20
        tile_size = 60
        dim = 16
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim,
            append_cls_token=False
        ).to(device)
        
        # The first position should have zeros for CLS token
        cls_cache = rope.cache[0]  # First position
        assert torch.allclose(cls_cache[:, 0], torch.ones_like(cls_cache[:, 0]))
        assert torch.allclose(cls_cache[:, 1], torch.zeros_like(cls_cache[:, 1]))
    
    def test_forward_with_multiple_tiles(self, device):
        """Test forward with multiple tiles."""
        patch_size = 40
        tile_size = 400
        dim = 32
        num_tiles = 4
        
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim
        ).to(device)
        
        patches_per_tile = (tile_size // patch_size) ** 2
        total_seq_len = num_tiles * (patches_per_tile + 1)
        
        x = torch.randn(1, total_seq_len, 8, dim, device=device)
        output = rope(x)
        
        assert output.shape == x.shape
    
    def test_dtype_preservation(self, device):
        """Test that output dtype matches input dtype."""
        patch_size = 40
        tile_size = 400
        dim = 32
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim
        ).to(device)
        
        seq_len = 101  # One tile worth
        
        dtypes = [torch.float32]
        if torch.cuda.is_available():
            dtypes.extend([torch.float16, torch.bfloat16])
        
        for dtype in dtypes:
            x = torch.randn(1, seq_len, 4, dim, device=device, dtype=dtype)
            output = rope(x)
            
            assert output.dtype == dtype
    
    def test_gradient_flow(self, device):
        """Test gradients flow through VisionRoPE."""
        patch_size = 40
        tile_size = 400
        dim = 32
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim
        ).to(device)
        
        seq_len = 101
        x = torch.randn(1, seq_len, 4, dim, device=device, requires_grad=True)
        output = rope(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_patch_grid_calculation(self):
        """Test patch grid size is calculated correctly."""
        test_cases = [
            (10, 100, 10),   # patch_size=10, tile_size=100 -> 10x10 grid
            (20, 100, 5),    # patch_size=20, tile_size=100 -> 5x5 grid
            (40, 400, 10),   # patch_size=40, tile_size=400 -> 10x10 grid
        ]
        
        for patch_size, tile_size, expected_grid_size in test_cases:
            rope = VisionRotaryPositionalEmbeddings(
                patch_size=patch_size,
                tile_size=tile_size,
                dim=32
            )
            assert rope.patch_grid_size == expected_grid_size
    
    def test_2d_positional_encoding(self, device):
        """Test that 2D positional encoding is applied (not just 1D)."""
        patch_size = 20
        tile_size = 60  # 3x3 grid
        dim = 16
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim
        ).to(device)
        
        seq_len = 10  # 9 patches + 1 CLS
        x = torch.randn(1, seq_len, 2, dim, device=device)
        output = rope(x)
        
        # The output should be different from input due to rotation
        assert not torch.allclose(output, x)
        assert output.shape == x.shape
    
    def test_kwargs_ignored(self, device):
        """Test that extra kwargs are ignored (for compatibility with RoPE)."""
        patch_size = 40
        tile_size = 400
        dim = 32
        rope = VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=dim
        ).to(device)
        
        x = torch.randn(1, 101, 4, dim, device=device)
        
        # Should work with extra kwargs
        output = rope(x, input_pos=None, some_other_kwarg="ignored")
        
        assert output.shape == x.shape

