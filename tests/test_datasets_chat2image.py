"""
Unit tests for Chat2ImageDataset.
"""
import pytest
import json
import torch
import tempfile
import base64
import os
from io import BytesIO
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Optional

from muse.data.datasets.image import (
    Chat2ImageDataset,
    MultiScaleDatasetWrapper,
    ResolutionBudgetScheduler,
)
from muse.data.utils import (
    ResolutionBudget,
    ResolutionBudgetConfig,
    get_aspect_ratio_dict,
    get_closest_ratio,
)


# Real processor path for testing
PROCESSOR_PATH = "/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/"
PROCESSOR_AVAILABLE = os.path.exists(PROCESSOR_PATH)

# Skip marker for tests that require the real processor
requires_processor = pytest.mark.skipif(
    not PROCESSOR_AVAILABLE,
    reason=f"Processor not found at {PROCESSOR_PATH}"
)


def create_test_image(width=100, height=100, color='red', mode='RGB'):
    """Create a test PIL Image."""
    return Image.new(mode, (width, height), color=color)


def create_test_parquet(tmp_path, data=None):
    """Create a test parquet file."""
    if data is None:
        data = {
            'uuid': ['1'],
            'source': ['test'],
            'image': [None],
            'message': [json.dumps([
                {"role": "user", "content": "Generate an image of a cat"},
                {"role": "assistant", "content": [{"type": "image", "image": "test_image"}]}
            ])]
        }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path)
    return str(parquet_path)


class MockDataset:
    """Mock iterable dataset for testing MultiScaleDatasetWrapper."""
    
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self._build_multiscale_transform = self._mock_transform
    
    def _mock_transform(self, target_size):
        """Mock transform that returns a composed transform."""
        def transform(img):
            return torch.randn(3, target_size[0], target_size[1])
        return transform
    
    def __iter__(self):
        return iter(self.samples)


# =============================================================================
# Tests for Chat2ImageDataset
# =============================================================================

@requires_processor
class TestChat2ImageDatasetInit:
    """Test Chat2ImageDataset initialization."""

    def test_init_basic(self, tmp_path):
        """Test basic initialization with default parameters."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=512,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        assert dataset.image_size == (512, 512)
        assert dataset.max_condition_length == 384
        assert dataset.center_crop is True
        assert dataset.multi_scale is False

    def test_init_with_tuple_size(self, tmp_path):
        """Test initialization with tuple image_size."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=(256, 512),
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        assert dataset.image_size == (256, 512)

    def test_init_multi_scale(self, tmp_path):
        """Test initialization with multi_scale enabled."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=1024,
            processor_path=PROCESSOR_PATH,
            multi_scale=True,
            num_workers=1
        )
        
        assert dataset.multi_scale is True


@requires_processor
class TestChat2ImageDatasetTransform:
    """Test Chat2ImageDataset transform methods."""

    def test_build_transform(self, tmp_path):
        """Test _build_transform creates valid transform pipeline."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            processor_path=PROCESSOR_PATH,
            center_crop=True,
            num_workers=1
        )
        
        img = create_test_image(300, 200)
        transformed = dataset.transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 256, 256)
        # Check normalization: values should be in [-1, 1] range
        assert transformed.min() >= -1.0
        assert transformed.max() <= 1.0

    def test_build_multiscale_transform(self, tmp_path):
        """Test _build_multiscale_transform creates correct size transform."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=512,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        img = create_test_image(400, 300)
        transform = dataset._build_multiscale_transform((768, 512))
        transformed = transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 768, 512)


@requires_processor
class TestChat2ImageDatasetExtract:
    """Test Chat2ImageDataset extract_image_text method."""

    def test_extract_from_message_field(self, tmp_path):
        """Test extraction from message field."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        img = create_test_image()
        message = [
            {"role": "user", "content": "Generate an image of a cat"},
            {"role": "assistant", "content": [{"type": "image", "image": "test_image"}]}
        ]
        sample = {"message": json.dumps(message), "image": img}
        
        result = dataset.extract_image_text(sample)
        
        assert result["image"] is img
        assert result["message"] == json.dumps(message)

    def test_extract_returns_none_for_missing_message(self, tmp_path):
        """Test extraction returns None for missing message field."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        sample = {"text": "test caption", "image": create_test_image()}
        result = dataset.extract_image_text(sample)
        
        assert result is None


@requires_processor
class TestChat2ImageDatasetProcessPair:
    """Test Chat2ImageDataset _process_pair method."""

    def test_process_pair_with_message(self, tmp_path):
        """Test _process_pair with valid message field."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        img = create_test_image(256, 256)
        message = [
            {"role": "user", "content": "Generate an image of a cat"},
            {"role": "assistant", "content": [{"type": "image", "image": "test_image"}]}
        ]
        sample = {"message": json.dumps(message), "image": img}
        
        result = dataset._process_pair(sample)
        
        # Should contain processor outputs
        assert result is not None
        assert "pixel_values" in result
        assert "image_grid_thw" in result
        # Should contain all processor fields
        processor_fields = ["input_ids", "attention_mask", "position_ids"]
        for field in processor_fields:
            if field in result:
                assert isinstance(result[field], torch.Tensor)

    def test_process_pair_with_images_dict(self, tmp_path):
        """Test _process_pair with images dictionary."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        img = create_test_image(256, 256)
        message = [
            {"role": "user", "content": "Generate an image of a cat"},
            {"role": "assistant", "content": [{"type": "image", "image": "img1"}]}
        ]
        images = {"img1": img}
        sample = {
            "message": json.dumps(message), 
            "image": "img1",
            "images": json.dumps(images)
        }
        
        result = dataset._process_pair(sample)
        
        assert result is not None
        assert "pixel_values" in result
        assert "image_grid_thw" in result


@requires_processor
class TestChat2ImageDatasetCollateFn:
    """Test Chat2ImageDataset collate_fn method."""

    def test_collate_concatenates_processor_outputs(self, tmp_path):
        """Test that collate_fn concatenates processor outputs correctly."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        # Create mock batch with processor outputs
        batch = [
            {
                "pixel_values": torch.randn(3, 256, 256),
                "image_grid_thw": torch.tensor([3, 256, 256]),
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            },
            {
                "pixel_values": torch.randn(3, 256, 256),
                "image_grid_thw": torch.tensor([3, 256, 256]),
                "input_ids": torch.tensor([[5, 6, 7, 8]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            }
        ]
        
        result = dataset.collate_fn(batch)
        
        # Check that all processor outputs are concatenated
        assert "pixel_values" in result
        assert "image_grid_thw" in result
        assert "input_ids" in result
        assert "attention_mask" in result
        
        # pixel_values should be stacked (batch dimension)
        assert result["pixel_values"].shape == (2, 3, 256, 256)
        # image_grid_thw should be stacked (batch dimension)
        assert result["image_grid_thw"].shape == (2, 3)
        # input_ids should be concatenated along sequence dimension
        assert result["input_ids"].shape == (2, 4)
        # attention_mask should be concatenated along sequence dimension
        assert result["attention_mask"].shape == (2, 4)

    def test_collate_handles_missing_fields(self, tmp_path):
        """Test collate_fn handles samples with missing fields gracefully."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        # Create batch with different fields
        batch = [
            {
                "pixel_values": torch.randn(3, 256, 256),
                "image_grid_thw": torch.tensor([3, 256, 256]),
                "input_ids": torch.tensor([[1, 2, 3]]),
            },
            {
                "pixel_values": torch.randn(3, 256, 256),
                "image_grid_thw": torch.tensor([3, 256, 256]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        ]
        
        result = dataset.collate_fn(batch)
        
        # Should contain common fields
        assert "pixel_values" in result
        assert "image_grid_thw" in result
        # Should not contain fields that are not in all samples
        assert "input_ids" not in result
        assert "attention_mask" not in result


@requires_processor
class TestChat2ImageDatasetIntegration:
    """Integration tests for Chat2ImageDataset."""

    def test_end_to_end_processing(self, tmp_path):
        """Test complete dataset processing pipeline."""
        # Create test parquet with message data
        message_data = [
            {"role": "user", "content": "Generate an image of a sunset"},
            {"role": "assistant", "content": [{"type": "image", "image": "img1"}]}
        ]
        
        parquet_path = create_test_parquet(tmp_path, data={
            'uuid': ['1', '2'],
            'source': ['test', 'test'],
            'image': ['img1', 'img2'],
            'message': [json.dumps(message_data), json.dumps(message_data)],
            'images': [json.dumps({"img1": "base64_img1", "img2": "base64_img2"})] * 2
        })
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        # Test dataset length
        assert len(dataset) == 2
        
        # Test sample retrieval
        sample = dataset[0]
        assert sample is not None
        assert "pixel_values" in sample
        assert "image_grid_thw" in sample
        
        # Test batch collation
        batch = [dataset[0], dataset[1]]
        collated = dataset.collate_fn(batch)
        
        assert "pixel_values" in collated
        assert "image_grid_thw" in collated
        assert collated["pixel_values"].shape[0] == 2


# =============================================================================
# Tests for MultiScaleDatasetWrapper with Chat2ImageDataset
# =============================================================================

@requires_processor
class TestMultiScaleWrapperWithChat2Image:
    """Test MultiScaleDatasetWrapper with Chat2ImageDataset."""

    def test_wrapper_initialization(self, tmp_path):
        """Test wrapper initialization with Chat2ImageDataset."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Chat2ImageDataset(
            sources=[parquet_path],
            image_size=512,
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, batch_size=16)],
            start_weights=[1.0],
            end_weights=[1.0],
        )
        
        wrapper = MultiScaleDatasetWrapper(dataset, config)
        
        assert wrapper.dataset is dataset
        assert wrapper.config is config

    def test_wrapper_iteration(self, tmp_path):
        """Test wrapper iteration with mock data."""
        # Create mock samples with height/width for multi-scale processing
        mock_samples = [
            {
                "image": create_test_image(300, 200),
                "height": 300,
                "width": 200,
                "message": json.dumps([
                    {"role": "user", "content": "Test message 1"},
                    {"role": "assistant", "content": [{"type": "image", "image": "img1"}]}
                ])
            },
            {
                "image": create_test_image(400, 300), 
                "height": 400,
                "width": 300,
                "message": json.dumps([
                    {"role": "user", "content": "Test message 2"},
                    {"role": "assistant", "content": [{"type": "image", "image": "img2"}]}
                ])
            }
        ]
        
        # Mock the dataset to return our samples
        dataset = MockDataset(mock_samples)
        dataset.rng = np.random.default_rng(42)  # Add rng for compatibility
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, batch_size=2)],
            start_weights=[1.0],
            end_weights=[1.0],
        )
        
        wrapper = MultiScaleDatasetWrapper(dataset, config, max_bucket_size=10)
        
        # Test iteration
        batches = list(wrapper)
        assert len(batches) > 0
        for batch in batches:
            assert len(batch) == 2
            for sample in batch:
                assert "target_height" in sample
                assert "target_width" in sample


if __name__ == "__main__":
    pytest.main([__file__])