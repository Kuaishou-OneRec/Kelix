"""
Unit tests for muse.data.datasets.image module.
"""
import pytest
import json
import torch
import tempfile
import base64
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from muse.data.datasets.image import Text2ImageDataset


class MockTokenizerOutput:
    """Mock output from tokenizer __call__"""
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class MockTokenizer:
    """Mock tokenizer for testing, compatible with HuggingFace tokenizer interface"""
    def __init__(self, max_length=300):
        self.pad_token_id = 0
        self.default_max_length = max_length

    def encode(self, text):
        """Simple encode: return list of integers based on text length"""
        if not text:
            return []
        # Simple encoding: each character maps to an integer
        return [ord(c) % 100 for c in text[:10]]

    def __call__(self, text, max_length=None, padding=None, truncation=True, return_tensors=None):
        """Tokenizer call interface compatible with HuggingFace tokenizers.
        
        Args:
            text: Input text to tokenize
            max_length: Maximum sequence length
            padding: Padding strategy ("max_length" for fixed length padding)
            truncation: Whether to truncate
            return_tensors: Return format ("pt" for PyTorch tensors)
        
        Returns:
            MockTokenizerOutput with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.default_max_length
            
        # Encode text
        tokens = self.encode(text)
        
        # Truncate if needed
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Create attention mask (1 for real tokens)
        attention_mask = [1] * len(tokens)
        
        # Pad if needed
        if padding == "max_length":
            pad_length = max_length - len(tokens)
            tokens = tokens + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            input_ids = torch.tensor([tokens])  # [1, L]
            attention_mask = torch.tensor([attention_mask])  # [1, L]
        else:
            input_ids = tokens
            attention_mask = attention_mask
        
        return MockTokenizerOutput(input_ids, attention_mask)


def create_test_image(width=100, height=100, color='red', mode='RGB'):
    """Create a test PIL Image"""
    return Image.new(mode, (width, height), color=color)


def create_test_parquet(tmp_path, data=None):
    """Create a test parquet file"""
    if data is None:
        data = {
            'uuid': ['1'],
            'source': ['test'],
            'image': [None],
            'text': ['test caption']
        }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path)
    return str(parquet_path)


class TestText2ImageDatasetInit:
    """Test Text2ImageDataset initialization"""

    def test_init_basic(self, tmp_path):
        """Test basic initialization with default parameters"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=512,
            tokenizer=tokenizer,
            num_workers=1
        )
        
        assert dataset.image_size == (512, 512)
        assert dataset.tokenizer is tokenizer
        assert dataset.max_text_length == 300
        assert dataset.center_crop is True

    def test_init_with_tuple_size(self, tmp_path):
        """Test initialization with tuple image_size"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=(256, 512),
            tokenizer=tokenizer,
            num_workers=1
        )
        
        assert dataset.image_size == (256, 512)

    def test_init_center_crop_false(self, tmp_path):
        """Test initialization with center_crop=False"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            tokenizer=tokenizer,
            center_crop=False,
            num_workers=1
        )
        
        assert dataset.center_crop is False


class TestBuildTransform:
    """Test _build_transform method"""

    def test_build_transform_center_crop(self, tmp_path):
        """Test transform building with center crop enabled"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            tokenizer=tokenizer,
            center_crop=True,
            num_workers=1
        )
        
        # Test that transform works
        img = create_test_image(300, 200)
        transformed = dataset.transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 256, 256)
        # Check normalization: values should be in [-1, 1] range
        assert transformed.min() >= -1.0
        assert transformed.max() <= 1.0

    def test_build_transform_no_center_crop(self, tmp_path):
        """Test transform building without center crop"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=(128, 256),
            tokenizer=tokenizer,
            center_crop=False,
            num_workers=1
        )
        
        img = create_test_image(300, 200)
        transformed = dataset.transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 128, 256)


class TestLoadImage:
    """Test _load_image method"""

    def test_load_image_from_path(self, tmp_path):
        """Test loading image from file path"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create and save a test image
        img_path = tmp_path / "test.jpg"
        img = create_test_image()
        img.save(img_path)
        
        loaded = dataset._load_image(str(img_path))
        
        assert loaded is not None
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (100, 100)

    def test_load_image_from_bytes(self, tmp_path):
        """Test loading image from bytes"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create image bytes
        img = create_test_image(50, 50)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        loaded = dataset._load_image(img_bytes)
        
        assert loaded is not None
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (50, 50)

    def test_load_image_from_base64(self, tmp_path):
        """Test loading image from base64 string"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create base64 encoded image
        img = create_test_image(60, 60, color='blue')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        loaded = dataset._load_image(img_base64)
        
        assert loaded is not None
        assert isinstance(loaded, Image.Image)

    def test_load_image_from_pil(self, tmp_path):
        """Test loading image from PIL Image (passthrough)"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img = create_test_image(80, 80)
        loaded = dataset._load_image(img)
        
        assert loaded is img  # Should be the same object

    def test_load_image_from_numpy(self, tmp_path):
        """Test loading image from numpy array"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create numpy array (RGB image)
        np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        loaded = dataset._load_image(np_img)
        
        assert loaded is not None
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (100, 100)

    def test_load_image_invalid(self, tmp_path):
        """Test loading invalid image returns None"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Test with non-existent path
        loaded = dataset._load_image("/nonexistent/path/image.jpg")
        assert loaded is None
        
        # Test with invalid base64
        loaded = dataset._load_image("not_valid_base64_or_path!!!")
        assert loaded is None


class TestGetContent:
    """Test get_content method"""

    def test_get_content_valid_json(self, tmp_path):
        """Test get_content with valid JSON"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        sample = {"messages": '[{"role": "user", "content": "hello"}]'}
        content = dataset.get_content(sample, "messages")
        
        assert len(content) == 1
        assert content[0]["role"] == "user"
        assert content[0]["content"] == "hello"

    def test_get_content_invalid_json(self, tmp_path):
        """Test get_content with invalid JSON returns empty list"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        sample = {"messages": "invalid json"}
        content = dataset.get_content(sample, "messages")
        
        assert content == []

    def test_get_content_missing_key(self, tmp_path):
        """Test get_content with missing key returns empty list"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        sample = {}
        content = dataset.get_content(sample, "messages")
        
        assert content == []


class TestExtractImageText:
    """Test extract_image_text method"""

    def test_extract_direct_fields(self, tmp_path):
        """Test extraction from direct image/text fields"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img = create_test_image()
        sample = {"image": img, "text": "A beautiful sunset"}
        
        result = dataset.extract_image_text(sample)
        
        assert result["image"] is img
        assert result["text"] == "A beautiful sunset"

    def test_extract_from_messages_string_content(self, tmp_path):
        """Test extraction from messages format with string content"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img_data = "base64_image_data"
        messages = [
            {"role": "user", "content": "Generate an image of a cat"},
            {"role": "assistant", "content": [{"type": "image", "image": img_data}]}
        ]
        sample = {"messages": json.dumps(messages)}
        
        result = dataset.extract_image_text(sample)
        
        assert result["text"] == "Generate an image of a cat"
        assert result["image"] == img_data

    def test_extract_from_messages_list_content(self, tmp_path):
        """Test extraction from messages format with list content"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img_data = "base64_image_data"
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Generate a dog"}]},
            {"role": "assistant", "content": [{"type": "image", "image": img_data}]}
        ]
        sample = {"messages": json.dumps(messages)}
        
        result = dataset.extract_image_text(sample)
        
        assert result["text"] == "Generate a dog"
        assert result["image"] == img_data

    def test_extract_from_messages_image_gen_type(self, tmp_path):
        """Test extraction from messages with image_gen type (legacy format)"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img_data = "base64_image_data"
        messages = [
            {"role": "user", "content": "Generate something"},
            {"role": "assistant", "content": [{"type": "image_gen", "image_gen": img_data}]}
        ]
        sample = {"messages": json.dumps(messages)}
        
        result = dataset.extract_image_text(sample)
        
        assert result["image"] == img_data

    def test_extract_from_segments(self, tmp_path):
        """Test extraction from segments format"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img_data = "segment_image_data"
        segments = [
            {"type": "text", "text": "Segment text"},
            {"type": "image", "image": img_data}
        ]
        sample = {"segments": json.dumps(segments)}
        
        result = dataset.extract_image_text(sample)
        
        assert result["text"] == "Segment text"
        assert result["image"] == img_data

    def test_extract_empty_sample(self, tmp_path):
        """Test extraction from empty sample"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        sample = {}
        result = dataset.extract_image_text(sample)
        
        assert result["image"] is None
        assert result["text"] is None


class TestValidateMessages:
    """Test _validate_messages method"""

    def test_validate_messages_valid_string_content(self, tmp_path):
        """Test validation passes for valid messages with string content"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": "Generate an image"},
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]}
        ]
        
        # Should not raise
        dataset._validate_messages(messages)

    def test_validate_messages_valid_list_content(self, tmp_path):
        """Test validation passes for valid messages with list content"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Generate"}]},
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]}
        ]
        
        # Should not raise
        dataset._validate_messages(messages)

    def test_validate_messages_with_system(self, tmp_path):
        """Test validation passes for messages with system role (system is skipped)"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # With system message at the beginning
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate an image"},
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]}
        ]
        
        # Should not raise - system message is skipped
        dataset._validate_messages(messages)
        
        # With system message in different position
        messages = [
            {"role": "user", "content": "Generate an image"},
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]}
        ]
        
        # Should not raise
        dataset._validate_messages(messages)

    def test_validate_messages_invalid_count(self, tmp_path):
        """Test validation fails for wrong message count"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Too few messages
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ValueError, match="exactly 2 non-system messages"):
            dataset._validate_messages(messages)
        
        # Too many non-system messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]},
            {"role": "user", "content": "More"}
        ]
        with pytest.raises(ValueError, match="exactly 2 non-system messages"):
            dataset._validate_messages(messages)

    def test_validate_messages_missing_user(self, tmp_path):
        """Test validation fails when user message is missing"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]},
            {"role": "assistant", "content": [{"type": "image", "image": "data2"}]}
        ]
        with pytest.raises(ValueError, match="1 user message"):
            dataset._validate_messages(messages)

    def test_validate_messages_missing_assistant(self, tmp_path):
        """Test validation fails when assistant message is missing"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"}
        ]
        with pytest.raises(ValueError, match="1 assistant message"):
            dataset._validate_messages(messages)

    def test_validate_messages_multiple_text_blocks(self, tmp_path):
        """Test validation fails when user has multiple text blocks"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"}
            ]},
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]}
        ]
        with pytest.raises(ValueError, match="exactly 1 text block"):
            dataset._validate_messages(messages)

    def test_validate_messages_no_text_block(self, tmp_path):
        """Test validation fails when user has no text block"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": [{"type": "other", "data": "stuff"}]},
            {"role": "assistant", "content": [{"type": "image", "image": "data"}]}
        ]
        with pytest.raises(ValueError, match="exactly 1 text block"):
            dataset._validate_messages(messages)

    def test_validate_messages_multiple_image_blocks(self, tmp_path):
        """Test validation fails when assistant has multiple image blocks"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": "Generate"},
            {"role": "assistant", "content": [
                {"type": "image", "image": "data1"},
                {"type": "image", "image": "data2"}
            ]}
        ]
        with pytest.raises(ValueError, match="exactly 1 image block"):
            dataset._validate_messages(messages)

    def test_validate_messages_no_image_block(self, tmp_path):
        """Test validation fails when assistant has no image block"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": "Generate"},
            {"role": "assistant", "content": [{"type": "text", "text": "response"}]}
        ]
        with pytest.raises(ValueError, match="exactly 1 image block"):
            dataset._validate_messages(messages)

    def test_validate_messages_assistant_content_not_list(self, tmp_path):
        """Test validation fails when assistant content is not a list"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        messages = [
            {"role": "user", "content": "Generate"},
            {"role": "assistant", "content": "just a string"}
        ]
        with pytest.raises(ValueError, match="must be list"):
            dataset._validate_messages(messages)


class TestValidateSegments:
    """Test _validate_segments method"""

    def test_validate_segments_valid(self, tmp_path):
        """Test validation passes for valid segments"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        segments = [
            {"type": "text", "text": "Caption"},
            {"type": "image", "image": "data"}
        ]
        
        # Should not raise
        dataset._validate_segments(segments)

    def test_validate_segments_invalid_count(self, tmp_path):
        """Test validation fails for wrong segment count"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Too few
        segments = [{"type": "text", "text": "Only one"}]
        with pytest.raises(ValueError, match="exactly 2 items"):
            dataset._validate_segments(segments)
        
        # Too many
        segments = [
            {"type": "text", "text": "One"},
            {"type": "image", "image": "Two"},
            {"type": "text", "text": "Three"}
        ]
        with pytest.raises(ValueError, match="exactly 2 items"):
            dataset._validate_segments(segments)

    def test_validate_segments_wrong_first_type(self, tmp_path):
        """Test validation fails when first segment is not text"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        segments = [
            {"type": "image", "image": "data"},
            {"type": "text", "text": "Caption"}
        ]
        with pytest.raises(ValueError, match="First segment must be type='text'"):
            dataset._validate_segments(segments)

    def test_validate_segments_wrong_second_type(self, tmp_path):
        """Test validation fails when second segment is not image"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        segments = [
            {"type": "text", "text": "Caption"},
            {"type": "text", "text": "Another text"}
        ]
        with pytest.raises(ValueError, match="Second segment must be type='image'"): 
            dataset._validate_segments(segments)


class TestProcessPair:
    """Test _process_pair method"""

    def test_process_pair_basic(self, tmp_path):
        """Test basic processing of image-text pair"""
        parquet_path = create_test_parquet(tmp_path)
        max_text_length = 20
        tokenizer = MockTokenizer(max_length=max_text_length)
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=64,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
            num_workers=1
        )
        
        img = create_test_image()
        sample = {"image": img, "text": "A test caption"}
        
        result = dataset._process_pair(sample)
        
        assert result is not None
        assert "image" in result
        assert "text" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (3, 64, 64)
        assert result["text"] == "A test caption"
        # Check input_ids and attention_mask are tensors with fixed length
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert result["input_ids"].shape == (max_text_length,)
        assert result["attention_mask"].shape == (max_text_length,)
        # Check attention_mask has 1s for real tokens and 0s for padding
        assert result["attention_mask"].sum() > 0  # Some real tokens
        assert (result["attention_mask"] <= 1).all()  # All values are 0 or 1

    def test_process_pair_image_none(self, tmp_path):
        """Test processing returns None when image is None"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        sample = {"image": None, "text": "Some text"}
        result = dataset._process_pair(sample)
        
        assert result is None

    def test_process_pair_text_none(self, tmp_path):
        """Test processing returns None when text is None"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img = create_test_image()
        sample = {"image": img, "text": None}
        result = dataset._process_pair(sample)
        
        assert result is None

    def test_process_pair_rgb_conversion(self, tmp_path):
        """Test that non-RGB images are converted to RGB"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=64,
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create RGBA image
        img = create_test_image(mode='RGBA')
        sample = {"image": img, "text": "RGBA image"}
        
        result = dataset._process_pair(sample)
        
        assert result is not None
        assert result["image"].shape == (3, 64, 64)  # RGB has 3 channels

    def test_process_pair_grayscale_conversion(self, tmp_path):
        """Test that grayscale images are converted to RGB"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=64,
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create grayscale image
        img = create_test_image(mode='L')
        sample = {"image": img, "text": "Grayscale image"}
        
        result = dataset._process_pair(sample)
        
        assert result is not None
        assert result["image"].shape == (3, 64, 64)  # Converted to RGB


class TestProcess:
    """Test process method (main entry point)"""

    def test_process_direct_fields(self, tmp_path):
        """Test process with direct image/text fields"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=64,
            tokenizer=tokenizer,
            num_workers=1
        )
        
        img = create_test_image()
        sample = {"image": img, "text": "Direct text"}
        
        result = dataset.process(sample)
        
        assert result is not None
        assert result["text"] == "Direct text"

    def test_process_from_messages(self, tmp_path):
        """Test process with messages format"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=64,
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create base64 image
        img = create_test_image()
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        messages = [
            {"role": "user", "content": "Generate a flower"},
            {"role": "assistant", "content": [{"type": "image", "image": img_base64}]}
        ]
        sample = {"messages": json.dumps(messages)}
        
        result = dataset.process(sample)
        
        assert result is not None
        assert result["text"] == "Generate a flower"

    def test_process_returns_none_for_invalid(self, tmp_path):
        """Test process returns None for invalid samples"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Empty sample
        sample = {}
        result = dataset.process(sample)
        
        assert result is None


class TestCollateFn:
    """Test collate_fn method"""

    def test_collate_basic(self, tmp_path):
        """Test basic batch collation"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            image_size=64,
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Create batch of processed samples
        batch = [
            {
                "image": torch.randn(3, 64, 64),
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1])
            },
            {
                "image": torch.randn(3, 64, 64),
                "input_ids": torch.tensor([4, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1])
            }
        ]
        
        result = dataset.collate_fn(batch)
        
        assert "image" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["image"].shape == (2, 3, 64, 64)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

    def test_collate_missing_keys(self, tmp_path):
        """Test collation with missing keys"""
        parquet_path = create_test_parquet(tmp_path)
        tokenizer = MockTokenizer()
        
        dataset = Text2ImageDataset(
            sources=[parquet_path],
            tokenizer=tokenizer,
            num_workers=1
        )
        
        # Batch with only image key
        batch = [
            {"image": torch.randn(3, 64, 64)},
            {"image": torch.randn(3, 64, 64)}
        ]
        
        result = dataset.collate_fn(batch)
        
        assert "image" in result
        assert "input_ids" not in result
        assert "attention_mask" not in result


# =============================================================================
# Tests for Dynamic Multi-Scale Training
# =============================================================================

from muse.data.utils import (
    ResolutionBudget,
    ResolutionBudgetConfig,
    DEFAULT_RESOLUTION_BUDGETS,
    parse_resolution_budgets,
)
from muse.data.datasets.image import ResolutionBudgetSampler


class TestResolutionBudgetConfig:
    """Test ResolutionBudgetConfig class"""
    
    def test_init_normalizes_weights(self):
        """Test that weights are normalized on init"""
        config = ResolutionBudgetConfig(
            budgets=[
                ResolutionBudget(512, 32),
                ResolutionBudget(1024, 8),
            ],
            start_weights=[2.0, 1.0],  # Will be normalized to [0.667, 0.333]
            end_weights=[1.0, 2.0],    # Will be normalized to [0.333, 0.667]
        )
        
        assert abs(sum(config.start_weights) - 1.0) < 1e-6
        assert abs(sum(config.end_weights) - 1.0) < 1e-6
    
    def test_get_weights_at_start(self):
        """Test get_weights returns start_weights at progress=0"""
        config = ResolutionBudgetConfig(
            budgets=[
                ResolutionBudget(512, 32),
                ResolutionBudget(1024, 8),
            ],
            start_weights=[0.8, 0.2],
            end_weights=[0.2, 0.8],
        )
        
        weights = config.get_weights(0.0)
        
        assert abs(weights[0] - 0.8) < 1e-6
        assert abs(weights[1] - 0.2) < 1e-6
    
    def test_get_weights_at_end(self):
        """Test get_weights returns end_weights at progress=1"""
        config = ResolutionBudgetConfig(
            budgets=[
                ResolutionBudget(512, 32),
                ResolutionBudget(1024, 8),
            ],
            start_weights=[0.8, 0.2],
            end_weights=[0.2, 0.8],
        )
        
        weights = config.get_weights(1.0)
        
        assert abs(weights[0] - 0.2) < 1e-6
        assert abs(weights[1] - 0.8) < 1e-6
    
    def test_get_weights_at_midpoint(self):
        """Test get_weights returns interpolated weights at progress=0.5"""
        config = ResolutionBudgetConfig(
            budgets=[
                ResolutionBudget(512, 32),
                ResolutionBudget(1024, 8),
            ],
            start_weights=[0.8, 0.2],
            end_weights=[0.2, 0.8],
        )
        
        weights = config.get_weights(0.5)
        
        # At midpoint, should be [0.5, 0.5]
        assert abs(weights[0] - 0.5) < 1e-6
        assert abs(weights[1] - 0.5) < 1e-6
    
    def test_get_weights_clamps_progress(self):
        """Test get_weights clamps progress to [0, 1]"""
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 32)],
            start_weights=[1.0],
            end_weights=[1.0],
        )
        
        # Should not raise for out-of-range progress
        weights_neg = config.get_weights(-0.5)
        weights_over = config.get_weights(1.5)
        
        assert len(weights_neg) == 1
        assert len(weights_over) == 1


class TestParseResolutionBudgets:
    """Test parse_resolution_budgets function"""
    
    def test_parse_basic(self):
        """Test parsing basic budget string"""
        config = parse_resolution_budgets("512:32,1024:8")
        
        assert len(config.budgets) == 2
        assert config.budgets[0].size == 512
        assert config.budgets[0].batch_size == 32
        assert config.budgets[1].size == 1024
        assert config.budgets[1].batch_size == 8
    
    def test_parse_with_weights(self):
        """Test parsing with explicit weights"""
        config = parse_resolution_budgets(
            "512:32,768:16,1024:8",
            "0.7,0.2,0.1",
            "0.1,0.2,0.7"
        )
        
        assert len(config.budgets) == 3
        assert abs(config.start_weights[0] - 0.7) < 1e-6
        assert abs(config.end_weights[2] - 0.7) < 1e-6
    
    def test_parse_defaults_weights(self):
        """Test that weights default correctly when not specified"""
        config = parse_resolution_budgets("512:32,1024:8")
        
        # Default: start favors low-res (0.7 for first)
        assert config.start_weights[0] > config.start_weights[1]
        # Default: end favors high-res (0.7 for last)
        assert config.end_weights[1] > config.end_weights[0]


class TestResolutionBudgetSampler:
    """Test ResolutionBudgetSampler class"""
    
    def test_init(self):
        """Test sampler initialization"""
        config = DEFAULT_RESOLUTION_BUDGETS
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        
        assert len(sampler.budgets) == 3
        assert sampler.total_steps == 1000
        assert sampler._current_step == 0
    
    def test_set_step(self):
        """Test set_step updates current step"""
        config = DEFAULT_RESOLUTION_BUDGETS
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        
        sampler.set_step(500)
        
        assert sampler._current_step == 500
        assert abs(sampler.progress - 0.5) < 1e-6
    
    def test_progress_capped_at_one(self):
        """Test progress is capped at 1.0"""
        config = DEFAULT_RESOLUTION_BUDGETS
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        
        sampler.set_step(2000)  # Beyond total_steps
        
        assert sampler.progress == 1.0
    
    def test_sample_returns_budget(self):
        """Test sample returns a ResolutionBudget"""
        config = DEFAULT_RESOLUTION_BUDGETS
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        
        budget = sampler.sample()
        
        assert isinstance(budget, ResolutionBudget)
        assert budget.size in [512, 768, 1024]
    
    def test_sample_distribution_at_start(self):
        """Test sample distribution favors low-res at start"""
        config = ResolutionBudgetConfig(
            budgets=[
                ResolutionBudget(512, 32),
                ResolutionBudget(1024, 8),
            ],
            start_weights=[0.9, 0.1],
            end_weights=[0.1, 0.9],
        )
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        sampler.set_step(0)
        
        # Sample many times and count
        counts = {512: 0, 1024: 0}
        for _ in range(1000):
            budget = sampler.sample()
            counts[budget.size] += 1
        
        # At start, 512 should be much more common (90% expected)
        assert counts[512] > counts[1024] * 2  # At least 2x more
    
    def test_sample_distribution_at_end(self):
        """Test sample distribution favors high-res at end"""
        config = ResolutionBudgetConfig(
            budgets=[
                ResolutionBudget(512, 32),
                ResolutionBudget(1024, 8),
            ],
            start_weights=[0.9, 0.1],
            end_weights=[0.1, 0.9],
        )
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        sampler.set_step(1000)
        
        # Sample many times and count
        counts = {512: 0, 1024: 0}
        for _ in range(1000):
            budget = sampler.sample()
            counts[budget.size] += 1
        
        # At end, 1024 should be much more common (90% expected)
        assert counts[1024] > counts[512] * 2  # At least 2x more
    
    def test_get_aspect_ratios(self):
        """Test get_aspect_ratios returns correct dict"""
        config = DEFAULT_RESOLUTION_BUDGETS
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        
        ratios_512 = sampler.get_aspect_ratios(512)
        ratios_1024 = sampler.get_aspect_ratios(1024)
        
        # Both should have aspect ratios
        assert len(ratios_512) > 0
        assert len(ratios_1024) > 0
        # 1024 sizes should be larger
        assert ratios_1024["1.0"][0] > ratios_512["1.0"][0]
    
    def test_get_stats(self):
        """Test get_stats returns correct info"""
        config = DEFAULT_RESOLUTION_BUDGETS
        sampler = ResolutionBudgetSampler(config, total_steps=1000)
        sampler.set_step(250)
        
        stats = sampler.get_stats()
        
        assert stats["step"] == 250
        assert abs(stats["progress"] - 0.25) < 1e-6
        assert 512 in stats["weights"]
        assert 768 in stats["weights"]
        assert 1024 in stats["weights"]
