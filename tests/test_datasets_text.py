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
            'You': 9,
            'are': 10,
            'a': 11,
            'helpful': 12,
            ".": 13,
            " ": 14,
            "\n": 15
        }
        # Sort vocab keys by length (longest first) for max prefix matching
        self._sorted_vocab_keys = sorted(self.vocab.keys(), key=len, reverse=True)
        # Create reverse mapping: token_id -> word for decode
        self._id_to_word = {v: k for k, v in self.vocab.items()}
        # Unknown token id
        self.unk_token_id = 999

    def encode(self, text):
        """
        Encode text using max prefix matching.
        Directly match the entire text against vocab using longest prefix match,
        preserving whitespace characters if they are in the vocab.
        """
        tokens = []
        remaining_text = text
        
        while remaining_text:
            matched = False
            # Try to find the longest matching prefix in vocab
            for vocab_key in self._sorted_vocab_keys:
                if remaining_text.startswith(vocab_key):
                    tokens.append(self.vocab[vocab_key])
                    remaining_text = remaining_text[len(vocab_key):]
                    matched = True
                    break
            
            if not matched:
                # No match found, use unknown token for the first character
                if remaining_text:
                    tokens.append(self.unk_token_id)
                    remaining_text = remaining_text[1:]  # Skip one character
                else:
                    break
        
        return tokens
    

    def decode(self, token_ids):
        """
        Decode token ids back to text.
        Maps token ids to words and concatenates them directly (whitespace is preserved as tokens).
        
        Args:
            token_ids: List of token ids or torch.Tensor
        
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        
        text_parts = []
        for token_id in token_ids:
            # Skip pad token
            if token_id == self.pad_token_id:
                continue
            # Map token id to word
            if token_id in self._id_to_word:
                text_parts.append(self._id_to_word[token_id])
            elif token_id == self.unk_token_id:
                # Unknown token - use placeholder
                text_parts.append('<unk>')
            else:
                # Unknown token id
                text_parts.append(f'<unk_{token_id}>')
        
        # Directly concatenate (whitespace is already in vocab as tokens)
        return ''.join(text_parts)


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
                add_system_prompt=False,
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
            # Check that loss_mask has non-zero values for prompts
            loss_mask = result["loss_mask"]
            input_ids = result["input_ids"][0].tolist()
            assert input_ids == [1, 3, 15, 9, 14, 10, 14, 11, 14, 12, 14, 5, 13, 2, 15, 1, 4, 15, 6, 2, 15, 1, 5, 15, 7, 2, 15]
            assert sum(loss_mask) == len(input_ids)

            dataset.add_prompt_loss = False
            result = dataset.process_messages(messages)
            
            loss_mask = result["loss_mask"]
            input_ids = result["input_ids"][0].tolist()
            assert input_ids == [1, 3, 15, 9, 14, 10, 14, 11, 14, 12, 14, 5, 13, 2, 15, 1, 4, 15, 6, 2, 15, 1, 5, 15, 7, 2, 15]
            print(sum(loss_mask))
            assert sum(loss_mask) == len(input_ids)


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


class TestChatTemplateRendering:
    """Test chat jinja template rendering"""

    def test_chat_template_system_message(self):
        """Test chat template rendering for system message"""
        from muse.data.templates import TemplateLoader
        from jinja2 import Template

        template_loader = TemplateLoader()
        template_content = template_loader.load("chat")
        template = Template(template_content, trim_blocks=True, lstrip_blocks=True)

        # Test system message rendering
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        rendered = template.render(messages=messages)
        
        assert rendered == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n"

        # Test system message rendering without default system
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        rendered_default_system = template.render(messages=messages, add_default_system=True)

        assert rendered_default_system == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n"


        # Test system message rendering without default system
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        rendered_no_system = template.render(messages=messages)

        assert rendered_no_system == "<|im_start|>user\nHello, how are you?<|im_end|>\n"

    def test_chat_template_user_message(self):
        """Test chat template rendering for user message"""
        from muse.data.templates import TemplateLoader
        from jinja2 import Template

        template_loader = TemplateLoader()
        template_content = template_loader.load("chat")
        template = Template(template_content, trim_blocks=True, lstrip_blocks=True)

        # Test user message with default settings
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        rendered = template.render(messages=messages)
        
        assert rendered == "<|im_start|>user\nHello, how are you?<|im_end|>\n"

        # Test user message with add_generation_prompt
        rendered_with_prompt = template.render(
            messages=messages,
            add_generation_prompt=True
        )
    
        assert rendered_with_prompt == "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"

    def test_chat_template_assistant_message(self):
        """Test chat template rendering for assistant message"""
        from muse.data.templates import TemplateLoader
        from jinja2 import Template

        template_loader = TemplateLoader()
        template_content = template_loader.load("chat")
        template = Template(template_content, trim_blocks=True, lstrip_blocks=True)

        # Test assistant message with default settings
        messages = [{"role": "assistant", "content": "I'm doing well, thank you!"}]
        rendered = template.render(messages=messages)
        
        assert rendered == "<|im_start|>assistant\nI'm doing well, thank you!<|im_end|>\n"

        # Test assistant message without prefix
        rendered_no_prefix = template.render(
            messages=messages,
            add_prefix=False
        )
        assert rendered_no_prefix == "I'm doing well, thank you!<|im_end|>\n"

    def test_chat_template_multi_turn_conversation(self):
        """Test chat template rendering for multi-turn conversation"""
        from muse.data.templates import TemplateLoader
        from jinja2 import Template

        template_loader = TemplateLoader()
        template_content = template_loader.load("chat")
        template = Template(template_content, trim_blocks=True, lstrip_blocks=True)

        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"}
        ]
        
        rendered = template.render(messages=messages)
        
        assert rendered == "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\nB<|im_end|>\n<|im_start|>user\nC<|im_end|>\n<|im_start|>assistant\nD<|im_end|>\n"
