"""
Unit tests for Token2ImageDataset and MultiScaleDatasetWrapper.
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
    Token2ImageDataset,
    MultiScaleDatasetWrapper,
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
            'text': ['test caption']
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
# Tests for Token2ImageDataset
# =============================================================================

@requires_processor
class TestToken2ImageDatasetInit:
    """Test Token2ImageDataset initialization."""

    def test_init_basic(self, tmp_path):
        """Test basic initialization with default parameters."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
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
        
        dataset = Token2ImageDataset(
            sources=[parquet_path],
            image_size=(256, 512),
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        assert dataset.image_size == (256, 512)

    def test_init_multi_scale(self, tmp_path):
        """Test initialization with multi_scale enabled."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
            sources=[parquet_path],
            image_size=1024,
            processor_path=PROCESSOR_PATH,
            multi_scale=True,
            num_workers=1
        )
        
        assert dataset.multi_scale is True


@requires_processor
class TestToken2ImageDatasetTransform:
    """Test Token2ImageDataset transform methods."""

    def test_build_transform(self, tmp_path):
        """Test _build_transform creates valid transform pipeline."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
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
        
        dataset = Token2ImageDataset(
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
class TestToken2ImageDatasetExtract:
    """Test Token2ImageDataset extract_image_text method."""

    def test_extract_direct_fields(self, tmp_path):
        """Test extraction from direct image/text fields."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
            sources=[parquet_path],
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        img = create_test_image()
        sample = {"image": img, "text": "A beautiful sunset"}
        
        result = dataset.extract_image_text(sample)
        
        assert result["image"] is img
        assert result["text"] == "A beautiful sunset"

    def test_extract_from_messages(self, tmp_path):
        """Test extraction from messages format."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
            sources=[parquet_path],
            processor_path=PROCESSOR_PATH,
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

    def test_extract_from_segments(self, tmp_path):
        """Test extraction from segments format."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
            sources=[parquet_path],
            processor_path=PROCESSOR_PATH,
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

    def test_extract_returns_none_for_missing(self, tmp_path):
        """Test extraction returns None for missing data."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
            sources=[parquet_path],
            processor_path=PROCESSOR_PATH,
            num_workers=1
        )
        
        sample = {}
        result = dataset.extract_image_text(sample)
        
        assert result is None


@requires_processor
class TestToken2ImageDatasetCollateFn:
    """Test Token2ImageDataset collate_fn method."""

    def test_collate_concatenates_pixel_values(self, tmp_path):
        """Test that collate_fn concatenates pixel_values correctly."""
        parquet_path = create_test_parquet(tmp_path)
        
        dataset = Token2ImageDataset(
            sources=[parquet_path],
            image_size=256,
            processor_path=PROCESSOR_PATH,
            num_workers=1,
            multi_scale=True
        )
        
        # Create batch with raw samples (image paths) and target dimensions for multi_scale
        # Save test images to temp files
        img1_path = tmp_path / "test_img1.png"
        img2_path = tmp_path / "test_img2.png"
        create_test_image(256, 256).save(img1_path)
        create_test_image(256, 256).save(img2_path)
        
        batch = [
            {
                "image": str(img1_path),
                "target_height": 224,
                "target_width": 252,
            },
            {
                "image": str(img2_path),
                "target_height": 224,
                "target_width": 252,
            }
        ]
        
        result = dataset.collate_fn(batch)
        
        assert "pixel_values" in result
        assert "image_grid_thw" in result
        assert "image" in result
        # pixel_values should be concatenated along dim 0 (2 samples)
        assert result["pixel_values"].dim() == 4
        # image_grid_thw should be concatenated (2 samples)
        assert result["image_grid_thw"].shape[0] == 2
        # image should be stacked (batch_size=2)
        assert result["image"].shape[0] == 2
        assert result["image"].shape[1] == 3  # RGB channels
        assert result["image"].shape[2] == 224
        assert result["image"].shape[3] == 252

        torch.testing.assert_close(
            result["image_grid_thw"],
            torch.tensor([[1, 16, 18], [1, 16, 18]])
        )


# =============================================================================
# Tests for MultiScaleDatasetWrapper
# =============================================================================

class TestMultiScaleDatasetWrapperInit:
    """Test MultiScaleDatasetWrapper initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        mock_dataset = MockDataset([])
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 4)],
        )
        
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
        )
        
        assert wrapper.config is config
        assert wrapper.drop_last is False
        assert wrapper.max_bucket_size == 10000
    
    def test_init_with_options(self):
        """Test initialization with custom options."""
        mock_dataset = MockDataset([])
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 4)],
        )
        
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
            drop_last=True,
            max_bucket_size=5000,
        )
        
        assert wrapper.drop_last is True
        assert wrapper.max_bucket_size == 5000


class TestMultiScaleDatasetWrapperIteration:
    """Test MultiScaleDatasetWrapper iteration behavior."""
    
    def test_yields_batches(self):
        """Test wrapper yields batches when bucket is full."""
        # Create samples with height/width info
        samples = [
            {"height": 512, "width": 512, "data": f"sample_{i}"}
            for i in range(10)
        ]
        mock_dataset = MockDataset(samples)
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 4)],  # batch_size=4
        )
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
        )
        
        batches = list(wrapper)
        
        # With 10 samples and batch_size=4, should yield 2 batches
        assert len(batches) == 2
        # Each batch should have 4 samples
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
    
    def test_adds_target_size_to_samples(self):
        """Test that wrapper adds target_height/target_width to samples."""
        samples = [
            {"height": 512, "width": 512, "data": f"sample_{i}"}
            for i in range(8)
        ]
        mock_dataset = MockDataset(samples)
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 4)],
        )
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
        )
        
        batches = list(wrapper)
        
        # Each sample in batch should have target_height and target_width
        for batch in batches:
            for sample in batch:
                assert "target_height" in sample
                assert "target_width" in sample
    
    def test_groups_by_aspect_ratio(self):
        """Test that samples are grouped by aspect ratio."""
        # Create samples with different aspect ratios
        samples = [
            {"height": 512, "width": 512, "data": "square_1"},
            {"height": 512, "width": 512, "data": "square_2"},
            {"height": 512, "width": 512, "data": "square_3"},
            {"height": 512, "width": 512, "data": "square_4"},
            {"height": 768, "width": 384, "data": "tall_1"},  # Different ratio
            {"height": 768, "width": 384, "data": "tall_2"},
        ]
        mock_dataset = MockDataset(samples)
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 4)],
        )
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
        )
        
        batches = list(wrapper)
        
        # Should yield 1 batch of 4 square samples
        # The 2 tall samples won't form a full batch
        assert len(batches) == 1
        assert len(batches[0]) == 4
        # All samples should be square (1.0 aspect ratio)
        for sample in batches[0]:
            assert sample["data"].startswith("square")
    
    def test_skips_none_samples(self):
        """Test that None samples are skipped."""
        samples = [
            {"height": 512, "width": 512, "data": "sample_1"},
            None,
            {"height": 512, "width": 512, "data": "sample_2"},
            None,
            {"height": 512, "width": 512, "data": "sample_3"},
            {"height": 512, "width": 512, "data": "sample_4"},
        ]
        mock_dataset = MockDataset(samples)
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 4)],
        )
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
        )
        
        batches = list(wrapper)
        
        # Should yield 1 batch of 4 valid samples
        assert len(batches) == 1
        assert len(batches[0]) == 4
    
    def test_respects_max_bucket_size(self):
        """Test that bucket size is limited."""
        # Create many samples with same aspect ratio
        samples = [
            {"height": 512, "width": 512, "data": f"sample_{i}"}
            for i in range(100)
        ]
        mock_dataset = MockDataset(samples)
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 50)],  # Large batch size
        )
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
            max_bucket_size=10,  # Small bucket limit
        )
        
        batches = list(wrapper)
        
        # With max_bucket_size=10 and batch_size=50, no batches should be yielded
        # because bucket never reaches 50 samples before being truncated to 10
        assert len(batches) == 0
    
    def test_empty_dataset(self):
        """Test behavior with empty dataset."""
        mock_dataset = MockDataset([])
        
        config = ResolutionBudgetConfig(
            budgets=[ResolutionBudget(512, 4)],
        )
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
        )
        
        batches = list(wrapper)
        
        assert len(batches) == 0


class TestMultiScaleDatasetWrapperMultiResolution:
    """Test MultiScaleDatasetWrapper with multiple resolutions."""
    
    def test_multi_resolution_config(self):
        """Test with multiple resolution budgets."""
        samples = [
            {"height": 512, "width": 512, "data": f"sample_{i}"}
            for i in range(20)
        ]
        mock_dataset = MockDataset(samples)
        
        config = ResolutionBudgetConfig(
            budgets=[
                ResolutionBudget(512, 8),
                ResolutionBudget(1024, 4),
            ],
        )
        wrapper = MultiScaleDatasetWrapper(
            dataset=mock_dataset,
            config=config,
        )
        
        # Wrapper should have aspect ratio dicts for both resolutions
        assert 512 in wrapper._aspect_ratios
        assert 1024 in wrapper._aspect_ratios
