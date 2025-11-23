"""
Unit tests for muse.data.datasets.base module.
"""
import pytest
import os
import tempfile
import json
import hashlib
import base64
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO

from muse.data.datasets.base import (
    is_hdfs,
    get_worker_info,
    is_image_exist,
    load_image,
    calculate_text_hash,
    load_parquet,
    ParquetReader,
    DistributedDataset,
    PARQUET_CACHE_DIR,
)


class TestUtilityFunctions:
    """Test utility functions in base.py"""

    def test_is_hdfs(self):
        """Test HDFS path detection"""
        assert is_hdfs("hdfs://path/to/file") is True
        assert is_hdfs("viewfs://path/to/file") is True
        assert is_hdfs("/local/path/to/file") is False
        assert is_hdfs("file://path/to/file") is False
        assert is_hdfs("s3://bucket/path") is False

    def test_get_worker_info_no_worker(self):
        """Test get_worker_info when no worker context exists"""
        worker, num_workers = get_worker_info()
        assert worker == 0
        assert num_workers == 1

    @patch('torch.utils.data.get_worker_info')
    def test_get_worker_info_with_worker(self, mock_get_worker_info):
        """Test get_worker_info when worker context exists"""
        mock_worker_info = Mock()
        mock_worker_info.id = 2
        mock_worker_info.num_workers = 4
        mock_get_worker_info.return_value = mock_worker_info

        worker, num_workers = get_worker_info()
        assert worker == 2
        assert num_workers == 4

    def test_is_image_exist(self, tmp_path):
        """Test image existence check"""
        # Non-existent file
        assert is_image_exist("/nonexistent/path.jpg") is False
        assert is_image_exist("") is False
        assert is_image_exist(None) is False

        # Create a test image file
        img_path = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        assert is_image_exist(str(img_path)) is True

        # Empty file
        empty_path = tmp_path / "empty.jpg"
        empty_path.touch()
        assert is_image_exist(str(empty_path)) is False

    def test_load_image_from_path(self, tmp_path):
        """Test loading image from file path"""
        img_path = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(img_path)

        loaded_img = load_image(str(img_path))
        assert loaded_img is not None
        assert isinstance(loaded_img, Image.Image)
        assert loaded_img.size == (100, 100)

    def test_load_image_from_base64(self):
        """Test loading image from base64 string"""
        # Create a simple image and encode to base64
        img = Image.new('RGB', (50, 50), color='green')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        loaded_img = load_image(img_base64)
        assert loaded_img is not None
        assert isinstance(loaded_img, Image.Image)

    def test_load_image_nonexistent(self):
        """Test loading non-existent image"""
        result = load_image("/nonexistent/path.jpg")
        assert result is None

    def test_calculate_text_hash(self):
        """Test text hash calculation"""
        text1 = "test text"
        text2 = "test text"
        text3 = "different text"

        hash1 = calculate_text_hash(text1)
        hash2 = calculate_text_hash(text2)
        hash3 = calculate_text_hash(text3)

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_load_parquet_local_file(self, tmp_path):
        """Test loading parquet file from local path"""
        # Create a test parquet file
        df = pd.DataFrame({
            'uuid': ['1', '2', '3'],
            'source': ['test', 'test', 'test'],
            'data': ['a', 'b', 'c']
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        parquet_file = load_parquet(str(parquet_path))
        assert parquet_file is not None
        loaded_df = parquet_file.to_pandas()
        assert len(loaded_df) == 3

    @patch('os.system')
    @patch('os.path.exists')
    def test_load_parquet_hdfs_with_cache(self, mock_exists, mock_system):
        """Test loading parquet file from HDFS with cache"""
        hdfs_path = "hdfs://path/to/file.parquet"
        cache_dir = f'/code/dataset_cache/0_0'
        cache_fn = os.path.join(
            cache_dir, str(calculate_text_hash(hdfs_path)) + '_file.parquet')

        # Mock cache file exists
        def exists_side_effect(path):
            if path == hdfs_path:
                return False
            if path == cache_fn:
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        # Create a test parquet file in cache
        os.makedirs(cache_dir, exist_ok=True)
        df = pd.DataFrame({'uuid': ['1'], 'source': ['test']})
        df.to_parquet(cache_fn)

        parquet_file = load_parquet(hdfs_path)
        assert parquet_file is not None


class TestParquetReader:
    """Test ParquetReader class"""

    def test_parquet_reader_init(self):
        """Test ParquetReader initialization"""
        sources = ["file1.parquet", "file2.parquet"]
        reader = ParquetReader(sources)
        assert reader.sources == sources

    def test_parquet_reader_parser(self):
        """Test ParquetReader._parser method"""
        reader = ParquetReader([])
        row = {
            "uuid": "test-uuid",
            "source": "test-source",
            "data": "test-data"
        }
        filename = "test.parquet"

        sample = reader._parser(row, filename, 0, 1)
        assert sample is not None
        assert sample["__file__"] == "test.parquet"
        assert sample["__index__"] == 0
        assert sample["__total__"] == 1

    def test_parquet_reader_iter(self, tmp_path):
        """Test ParquetReader iteration"""
        # Create test parquet files
        df1 = pd.DataFrame({
            'uuid': ['1', '2'],
            'source': ['test1', 'test1'],
            'messages': ['[]', '[]']
        })
        df2 = pd.DataFrame({
            'uuid': ['3'],
            'source': ['test2'],
            'messages': ['[]']
        })

        parquet_path1 = tmp_path / "test1.parquet"
        parquet_path2 = tmp_path / "test2.parquet"
        df1.to_parquet(parquet_path1)
        df2.to_parquet(parquet_path2)

        reader = ParquetReader([str(parquet_path1), str(parquet_path2)])

        samples = list(reader)
        assert len(samples) == 3
        assert samples[0]["__index__"] == 0
        assert samples[1]["__index__"] == 1
        assert samples[2]["__index__"] == 0
        assert samples[2]["__total__"] == 1
        assert samples[1]["__total__"] == 2

    @patch('muse.data.datasets.base.load_parquet')
    def test_parquet_reader_error_handling(self, mock_load_parquet):
        """Test ParquetReader error handling"""
        mock_load_parquet.side_effect = Exception("File not found")
        reader = ParquetReader(["nonexistent.parquet"])

        # Should not raise, but skip the file
        samples = list(reader)
        assert len(samples) == 0


class TestDistributedDataset:
    """Test DistributedDataset class"""

    def test_distributed_dataset_init_with_list(self, tmp_path):
        """Test DistributedDataset initialization with file list"""
        # Create test parquet files
        df = pd.DataFrame({'uuid': ['1'], 'source': ['test']})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        # DistributedDataset accepts list as source (even though type hint says str)
        dataset = DistributedDataset(
            sources=[str(parquet_path)],
            num_workers=1,
            seed=42
        )
        assert isinstance(dataset.sources, list) or len(dataset._files) > 0
        assert dataset.num_workers == 1
        assert dataset.seed == 42

    def test_distributed_dataset_init_with_json(self, tmp_path):
        """Test DistributedDataset initialization with JSON file list"""
        # Create test parquet files
        df = pd.DataFrame({'uuid': ['1'], 'source': ['test']})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        # Create JSON file list
        json_path = tmp_path / "file_list.json"
        with open(json_path, 'w') as f:
            json.dump([str(parquet_path)], f)

        dataset = DistributedDataset(
            sources=str(json_path),
            num_workers=1
        )
        assert len(dataset._files) > 0 or dataset.dataset is not None

    def test_distributed_dataset_init_with_directory(self, tmp_path):
        """Test DistributedDataset initialization with directory"""
        # Create test parquet files in subdirectory
        subdir = tmp_path / "data"
        subdir.mkdir()
        df = pd.DataFrame({'uuid': ['1'], 'source': ['test']})
        parquet_path = subdir / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = DistributedDataset(
            sources=str(tmp_path),
            num_workers=1
        )
        # Should find the parquet file
        assert dataset.dataset is not None or len(dataset._files) > 0

    def test_distributed_dataset_shard_by_auto_files(self, tmp_path):
        """Test auto sharding mode selecting files mode"""
        # Create multiple parquet files
        files = []
        for i in range(10):
            df = pd.DataFrame({'uuid': [str(i)], 'source': ['test']})
            parquet_path = tmp_path / f"test_{i}.parquet"
            df.to_parquet(parquet_path)
            files.append(str(parquet_path))

        # Use list as source (supported by _load_file_list)
        dataset = DistributedDataset(
            sources=files,  # type: ignore
            num_workers=1,
            shard_by="auto"
        )
        # With 10 files and 1 worker, should select files mode
        assert dataset._actual_shard_by == "files"

    def test_distributed_dataset_shard_by_auto_samples(self, tmp_path):
        """Test auto sharding mode selecting samples mode"""
        # Create few parquet files
        df = pd.DataFrame({'uuid': ['1'], 'source': ['test']})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = DistributedDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=10,  # More workers than files
            shard_by="auto"
        )
        # With 1 file and 10 workers, should select samples mode
        assert dataset._actual_shard_by == "samples"

    def test_distributed_dataset_shard_by_files(self, tmp_path):
        """Test files sharding mode"""
        files = []
        for i in range(5):
            df = pd.DataFrame({'uuid': [str(i)], 'source': ['test']})
            parquet_path = tmp_path / f"test_{i}.parquet"
            df.to_parquet(parquet_path)
            files.append(str(parquet_path))

        dataset = DistributedDataset(
            sources=files,  # type: ignore
            num_workers=2,
            shard_by="files"
        )
        assert dataset._actual_shard_by == "files"
        assert len(dataset._files) == 5

    def test_distributed_dataset_shard_by_samples(self, tmp_path):
        """Test samples sharding mode"""
        df = pd.DataFrame({'uuid': ['1', '2', '3'], 'source': ['test'] * 3})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = DistributedDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=2,
            shard_by="samples"
        )
        assert dataset._actual_shard_by == "samples"
        assert dataset.dataset is not None

    def test_distributed_dataset_invalid_shard_by(self, tmp_path):
        """Test invalid shard_by parameter"""
        df = pd.DataFrame({'uuid': ['1'], 'source': ['test']})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        with pytest.raises(AssertionError, match="shard_by must be"):
            DistributedDataset(
                sources=[str(parquet_path)],  # type: ignore
                shard_by="invalid"
            )

    def test_distributed_dataset_iter_no_packing(self, tmp_path):
        """Test DistributedDataset iteration without packing"""
        # Create a simple dataset that implements process
        class TestDataset(DistributedDataset):
            def process(self, sample):
                return {"data": sample["__index__"]}

        df = pd.DataFrame({
            'uuid': ['1', '2'],
            'source': ['test'] * 2,
            'messages': ['[]', '[]']
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TestDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=1,
            packing=False
        )

        samples = list(dataset)
        assert len(samples) == 2
        assert samples[0]["data"] == 0
        assert samples[1]["data"] == 1

    def test_distributed_dataset_multiple_epochs(self, tmp_path):
        """Test DistributedDataset with multiple epochs"""
        class TestDataset(DistributedDataset):
            def process(self, sample):
                return {"data": sample["__index__"]}

        df = pd.DataFrame({
            'uuid': ['1'],
            'source': ['test'],
            'messages': ['[]']
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TestDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=1,
            num_epochs=3
        )

        samples = list(dataset)
        assert len(samples) == 3  # Should repeat 3 times

    def test_distributed_dataset_rank_world_size(self, tmp_path):
        """Test DistributedDataset uses correct rank and world_size from parameters"""
        df = pd.DataFrame({'uuid': ['1'], 'source': ['test']})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        # Test with explicit rank and world_size
        dataset = DistributedDataset(
            sources=[str(parquet_path)],
            rank=1,
            world_size=4,
            num_workers=1
        )
        assert dataset.rank == 1
        assert dataset.world_size == 4

        # Test with default values
        dataset_default = DistributedDataset(
            sources=[str(parquet_path)],
            num_workers=1
        )
        assert dataset_default.rank == 0
        assert dataset_default.world_size == 1

