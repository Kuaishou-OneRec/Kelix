"""
Unit tests for muse.data.datasets.text module.
"""
import pytest
import json
import torch
import tempfile
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from pathlib import Path

from muse.data.datasets.text import TextDataset


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.vocab = {
            '<|im_start|>': 1,
            '<|im_end|>': 2,
            'system': 3,
            'user': 4,
            'assistant': 5,
            'hello': 6,
            'world': 7,
            'test': 8,
        }

    def encode(self, text):
        """Simple encoding that splits by space and maps to vocab"""
        tokens = []
        for word in text.split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Unknown token
                tokens.append(999)
        return tokens


class TestTextDataset:
    """Test TextDataset class"""

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_text_dataset_init(self, mock_tokenizer_class):
        """Test TextDataset initialization"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer",
                system_prompt="default",
                chat_template="chat"
            )

            assert dataset.tokenizer is not None
            assert dataset.max_length_per_sample == 2048
            assert dataset.pad_to_multiple_of == 1

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_text_dataset_init_no_tokenizer(self, mock_tokenizer_class):
        """Test TextDataset initialization without tokenizer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path=None
            )

            assert dataset.tokenizer is None

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_get_content(self, mock_tokenizer_class):
        """Test get_content method"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            # Test with valid JSON
            sample = {"messages": '[{"role": "user", "content": "hello"}]'}
            content = dataset.get_content(sample, "messages")
            assert len(content) == 1
            assert content[0]["role"] == "user"

            # Test with invalid JSON
            sample = {"messages": "invalid json"}
            content = dataset.get_content(sample, "messages")
            assert content == []

            # Test with missing key
            sample = {}
            content = dataset.get_content(sample, "messages")
            assert content == []

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_messages_single_turn(self, mock_tokenizer_class):
        """Test process_messages with single turn conversation"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer",
                add_system_prompt=True,
                add_prompt_loss=False
            )

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"}
            ]

            result = dataset.process_messages(messages)
            assert result is not None
            assert "input_ids" in result
            assert "loss_mask" in result
            assert "position_ids" in result
            assert result["input_ids"].shape[0] == 1  # batch dimension

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_messages_with_system_prompt(self, mock_tokenizer_class):
        """Test process_messages with system prompt"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer",
                add_system_prompt=True,
                system_prompt="You are a helpful assistant"
            )

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"}
            ]

            result = dataset.process_messages(messages)
            assert result is not None
            # System prompt should be added
            assert len(messages) == 3  # system, user, assistant

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_messages_with_prompt_loss(self, mock_tokenizer_class):
        """Test process_messages with prompt loss enabled"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer",
                add_prompt_loss=True
            )

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"}
            ]

            result = dataset.process_messages(messages)
            assert result is not None
            # Check that loss_mask has non-zero values for prompts
            loss_mask = result["loss_mask"]

            print("loss_mask: ", loss_mask)
            print("input_ids: ", result["input_ids"])
            assert torch.sum(loss_mask) < 0

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_messages_no_response(self, mock_tokenizer_class):
        """Test process_messages with only prompt (no response)"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer",
                add_prompt_loss=False
            )

            messages = [
                {"role": "user", "content": "hello"}
            ]

            result = dataset.process_messages(messages)
            # Should return None if all loss_mask is 0
            assert result is None

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_messages_max_length_truncation(self, mock_tokenizer_class):
        """Test process_messages with max_length truncation"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer",
                max_length_per_sample=10
            )

            # Create a long message
            long_content = " ".join(["word"] * 100)
            messages = [
                {"role": "user", "content": long_content},
                {"role": "assistant", "content": "response"}
            ]

            result = dataset.process_messages(messages)
            assert result is not None
            # Should be truncated to max_length
            assert result["input_ids"].shape[1] <= 10

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_messages_padding(self, mock_tokenizer_class):
        """Test process_messages with padding"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer",
                pad_to_multiple_of=4
            )

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"}
            ]

            result = dataset.process_messages(messages)
            assert result is not None
            # Length should be multiple of 4
            length = result["input_ids"].shape[1]
            assert length % 4 == 0

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_segments(self, mock_tokenizer_class):
        """Test process_segments method"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            segments = [
                {"type": "text", "text": "hello world"}
            ]

            result = dataset.process_segments(segments)
            assert result is not None
            assert "input_ids" in result
            assert "loss_mask" in result
            assert "position_ids" in result

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_segments_empty(self, mock_tokenizer_class):
        """Test process_segments with empty segments"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            segments = []
            result = dataset.process_segments(segments)
            assert result is None

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_prioritizes_messages(self, mock_tokenizer_class):
        """Test that process method prioritizes messages over segments"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            sample = {
                "messages": '[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]',
                "segments": '[{"type": "text", "text": "other"}]'
            }

            result = dataset.process(sample)
            assert result is not None
            # Should process messages, not segments

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_falls_back_to_segments(self, mock_tokenizer_class):
        """Test that process falls back to segments when messages are empty"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            sample = {
                "messages": "[]",
                "segments": '[{"type": "text", "text": "hello"}]'
            }

            result = dataset.process(sample)
            assert result is not None
            # Should process segments

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_process_returns_none_when_empty(self, mock_tokenizer_class):
        """Test that process returns None when both messages and segments are empty"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            sample = {
                "messages": "[]",
                "segments": "[]"
            }

            result = dataset.process(sample)
            assert result is None

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_pack_sample(self, mock_tokenizer_class):
        """Test pack_sample method"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            inputs = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "loss_mask": torch.tensor([[1, 1, 1]]),
                "position_ids": torch.tensor([[0, 1, 2]])
            }

            new_inputs = {
                "input_ids": torch.tensor([[4, 5]]),
                "loss_mask": torch.tensor([[1, 1]]),
                "position_ids": torch.tensor([[0, 1]])
            }

            packed = dataset.pack_sample([inputs, new_inputs])
            assert packed["input_ids"].shape[1] == 5  # 3 + 2
            assert packed["loss_mask"].shape[1] == 5
            assert packed["position_ids"].shape[1] == 5
            assert packed["cu_seqlen"] == [0, 3, 5]

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_get_sample_length(self, mock_tokenizer_class):
        """Test get_sample_length method"""
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'uuid': ['1'],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = Path(tmpdir) / "test.parquet"
            df.to_parquet(parquet_path)

            dataset = TextDataset(
                sources=[str(parquet_path)],
                tokenizer_path="test/tokenizer"
            )

            sample = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]])
            }

            length = dataset.get_sample_length(sample)
            assert length == 5

