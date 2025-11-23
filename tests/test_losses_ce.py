#!/usr/bin/env python3
"""Unit tests for CrossEntropyLoss module."""

import pytest
import torch
import torch.nn as nn

from muse.losses.ce import CrossEntropyLoss


class TestCrossEntropyLoss:
    """Test suite for CrossEntropyLoss."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = CrossEntropyLoss()
        assert loss_fn.ignore_index == -100
        assert loss_fn.return_token_loss is False
        assert loss_fn.reduction == "mean"
        assert loss_fn.shift_labels is True

    def test_initialization_custom(self):
        """Test custom initialization."""
        loss_fn = CrossEntropyLoss(
            ignore_index=-1,
            return_token_loss=True,
            shift_labels=False,
            reduction="mean"
        )
        assert loss_fn.ignore_index == -1
        assert loss_fn.return_token_loss is True
        assert loss_fn.shift_labels is False
        assert loss_fn.reduction == "mean"

    def test_forward_basic(self, device):
        """Test basic forward pass."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(shift_labels=False)
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_forward_with_shift_labels(self, device):
        """Test forward pass with shift_labels=True."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn_shift = CrossEntropyLoss(shift_labels=True)
        loss_fn_no_shift = CrossEntropyLoss(shift_labels=False)

        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        loss_shift = loss_fn_shift(logits, labels)
        # For no_shift, we need to manually shift
        logits_no_shift = logits[:, :-1, :]
        labels_no_shift = labels[:, 1:]
        loss_no_shift = loss_fn_no_shift(logits_no_shift, labels_no_shift)

        # Results should be similar (within numerical precision)
        assert torch.allclose(loss_shift, loss_no_shift, atol=1e-6)

    def test_forward_with_ignore_index(self, device):
        """Test forward pass with ignore_index."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(ignore_index=-100, shift_labels=False)
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        # Set some labels to ignore_index
        labels[0, 2] = -100
        labels[1, 3] = -100

        loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_forward_all_ignored(self, device):
        """Test forward pass when all tokens are ignored."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(ignore_index=-100, shift_labels=False)
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.full((batch_size, seq_len), -100).to(device)

        loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert loss.item() == 0.0

    def test_forward_return_token_loss(self, device):
        """Test forward pass with return_token_loss=True."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(
            return_token_loss=True,
            shift_labels=False
        )
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        loss, per_token_loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert per_token_loss.shape == (batch_size * seq_len,)
        assert torch.all(per_token_loss >= 0)

    def test_forward_return_token_loss_with_shift(self, device):
        """Test return_token_loss with shift_labels=True."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(
            return_token_loss=True,
            shift_labels=True
        )
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        loss, per_token_loss = loss_fn(logits, labels)
        assert loss.shape == ()
        # After shifting, seq_len becomes seq_len - 1
        expected_shape = (batch_size * (seq_len - 1),)
        assert per_token_loss.shape == expected_shape

    def test_forward_return_token_loss_with_ignore(self, device):
        """Test return_token_loss with ignore_index."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(
            return_token_loss=True,
            ignore_index=-100,
            shift_labels=False
        )
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        # Set some labels to ignore_index
        labels[0, 2] = -100
        labels[1, 3] = -100

        loss, per_token_loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert per_token_loss.shape == (batch_size * seq_len,)
        # Ignored tokens should have zero loss
        assert per_token_loss[0 * seq_len + 2].item() == 0.0
        assert per_token_loss[1 * seq_len + 3].item() == 0.0

    def test_compare_with_pytorch_ce(self, device):
        """Test that our loss matches PyTorch CrossEntropyLoss."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(shift_labels=False, ignore_index=-100)
        pytorch_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        our_loss = loss_fn(logits, labels)
        # PyTorch expects (N, C) and (N,) for 2D case
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        pytorch_loss = pytorch_loss_fn(logits_flat, labels_flat)

        assert torch.allclose(our_loss, pytorch_loss, atol=1e-6)

    def test_compare_with_pytorch_ce_with_ignore(self, device):
        """Test that our loss matches PyTorch with ignore_index."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(shift_labels=False, ignore_index=-100)
        pytorch_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        labels[0, 2] = -100
        labels[1, 3] = -100

        our_loss = loss_fn(logits, labels)
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        pytorch_loss = pytorch_loss_fn(logits_flat, labels_flat)

        assert torch.allclose(our_loss, pytorch_loss, atol=1e-6)

    def test_different_batch_sizes(self, device):
        """Test with different batch sizes."""
        vocab_size = 10
        loss_fn = CrossEntropyLoss(shift_labels=False)

        for batch_size in [1, 2, 4, 8]:
            seq_len = 5
            logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

            loss = loss_fn(logits, labels)
            assert loss.shape == ()
            assert loss.item() >= 0

    def test_different_seq_lengths(self, device):
        """Test with different sequence lengths."""
        batch_size = 2
        vocab_size = 10
        loss_fn = CrossEntropyLoss(shift_labels=False)

        for seq_len in [1, 5, 10, 50, 100]:
            logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

            loss = loss_fn(logits, labels)
            assert loss.shape == ()
            assert loss.item() >= 0

    def test_different_vocab_sizes(self, device):
        """Test with different vocabulary sizes."""
        batch_size = 2
        seq_len = 5
        loss_fn = CrossEntropyLoss(shift_labels=False)

        for vocab_size in [10, 100, 1000, 50000]:
            logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

            loss = loss_fn(logits, labels)
            assert loss.shape == ()
            assert loss.item() >= 0

    def test_gradient_flow(self, device):
        """Test that gradients flow correctly."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(shift_labels=False)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_shift_labels_consistency(self, device):
        """Test that shift_labels produces consistent results."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10

        loss_fn_shift = CrossEntropyLoss(shift_labels=True)
        loss_fn_no_shift = CrossEntropyLoss(shift_labels=False)

        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        loss_shift = loss_fn_shift(logits, labels)
        # Manually shift for comparison
        logits_manual = logits[:, :-1, :]
        labels_manual = labels[:, 1:]
        loss_no_shift = loss_fn_no_shift(logits_manual, labels_manual)

        assert torch.allclose(loss_shift, loss_no_shift, atol=1e-6)

    def test_reduction_mean(self, device):
        """Test that reduction='mean' works correctly."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10

        loss_fn = CrossEntropyLoss(reduction="mean", shift_labels=False)
        loss_fn_with_token = CrossEntropyLoss(
            return_token_loss=True,
            shift_labels=False
        )
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        loss = loss_fn(logits, labels)
        # Get per-token loss to verify mean
        _, per_token_loss = loss_fn_with_token(logits, labels)

        # Count non-ignored tokens
        valid_tokens = (labels != -100).sum().item()
        expected_loss = per_token_loss.sum() / valid_tokens

        assert torch.allclose(loss, expected_loss, atol=1e-6)

