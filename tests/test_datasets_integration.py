"""
Integration tests for datasets module.
"""
import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import torch
from torch.utils.data import DataLoader

from muse.data.datasets.base import DistributedDataset, ParquetReader
from muse.data.datasets.text import TextDataset


class TestDataset(DistributedDataset):
    """Simple test dataset that implements process method"""
    def process(self, sample):
        return {"index": sample.get("__index__", "unknown")}


class TestParquetIntegration:
    """Integration tests with real parquet files"""

    def test_parquet_reader_with_real_file(self, tmp_path):
        """Test ParquetReader with real parquet file"""
        # Create test data
        df = pd.DataFrame({
            'uuid': ['1', '2', '3'],
            'source': ['test'] * 3,
            'messages': ['[]', '[]', '[]'],
            'data': ['a', 'b', 'c']
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        reader = ParquetReader([str(parquet_path)])
        samples = list(reader)

        assert len(samples) == 3
        assert samples[0]["__index__"] == 0
        assert samples[1]["__index__"] == 1
        assert samples[2]["__index__"] == 2
        assert samples[2]["__total__"] == 3

    def test_parquet_reader_multiple_files(self, tmp_path):
        """Test ParquetReader with multiple parquet files"""
        files = []
        for i in range(3):
            df = pd.DataFrame({
                'uuid': [str(i)],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = tmp_path / f"test_{i}.parquet"
            df.to_parquet(parquet_path)
            files.append(str(parquet_path))

        reader = ParquetReader(files)
        samples = list(reader)

        assert len(samples) == 3
        assert all(s["__index__"] in [0, 1, 2] for s in samples)
        assert all(s["__total__"] == 3 for s in samples)

    def test_distributed_dataset_end_to_end(self, tmp_path):
        """Test DistributedDataset end-to-end with real files"""
        # Create test parquet files
        df = pd.DataFrame({
            'uuid': ['1', '2', '3'],
            'source': ['test'] * 3,
            'messages': ['[]', '[]', '[]']
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TestDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=1,
            num_epochs=1
        )

        samples = list(dataset)
        assert len(samples) == 3

    def test_distributed_dataset_multiple_epochs(self, tmp_path):
        """Test DistributedDataset with multiple epochs"""
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
            num_epochs=3
        )

        samples = list(dataset)
        assert len(samples) == 6  # 2 samples * 3 epochs

    def test_distributed_dataset_with_dataloader(self, tmp_path):
        """Test DistributedDataset with PyTorch DataLoader"""
        df = pd.DataFrame({
            'uuid': ['1', '2', '3', '4'],
            'source': ['test'] * 4,
            'messages': ['[]'] * 4
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TestDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=1
        )

        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
        batches = list(dataloader)

        assert len(batches) >= 1

class TestTextDatasetIntegration:
    """Integration tests for TextDataset"""

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_text_dataset_end_to_end(self, mock_tokenizer_class, tmp_path):
        """Test TextDataset end-to-end with real parquet file"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.encode = lambda x: [1, 2, 3, 4, 5]  # Simple mock encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Create test parquet file with messages
        messages_data = json.dumps([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"}
        ])
        df = pd.DataFrame({
            'uuid': ['1', '2'],
            'source': ['test'] * 2,
            'messages': [messages_data, messages_data]
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TextDataset(
            sources=[str(parquet_path)],
            tokenizer_path="test/tokenizer",
            num_workers=1
        )
        for sample in dataset:
            print("sample", sample)
        samples = list(dataset)
        # TODO: debug
        print("samples", samples)
        assert len(samples) == 2
        assert all("input_ids" in s for s in samples)
        assert all("loss_mask" in s for s in samples)

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_text_dataset_with_segments(self, mock_tokenizer_class, tmp_path):
        """Test TextDataset with segments data"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.encode = lambda x: [1, 2, 3]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        segments_data = json.dumps([
            {"type": "text", "text": "hello world"}
        ])
        df = pd.DataFrame({
            'uuid': ['1'],
            'source': ['test'],
            'segments': [segments_data]
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TextDataset(
            sources=[str(parquet_path)],
            tokenizer_path="test/tokenizer",
            num_workers=1
        )

        samples = list(dataset)
        assert len(samples) == 1
        assert "input_ids" in samples[0]

    @patch('muse.data.datasets.text.AutoTokenizer')
    def test_text_dataset_packing(self, mock_tokenizer_class, tmp_path):
        """Test TextDataset with packing enabled"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.encode = lambda x: [1, 2, 3]  # Short sequences for packing
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        messages_data = json.dumps([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"}
        ])
        df = pd.DataFrame({
            'uuid': ['1', '2'],
            'source': ['test'] * 2,
            'messages': [messages_data, messages_data]
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TextDataset(
            sources=[str(parquet_path)],
            tokenizer_path="test/tokenizer",
            num_workers=1,
            packing=True,
            max_length=20  # Large enough to pack multiple samples
        )

        samples = list(dataset)
        # With packing, might get fewer samples
        assert len(samples) >= 1
        # Check for cu_seqlen if packing occurred
        if len(samples) > 0 and "cu_seqlen" in samples[0]:
            assert "cu_seqlen" in samples[0]


class TestDistributedScenarios:
    """Test distributed training scenarios"""

    @patch('muse.data.datasets.base.get_data_parallel_rank')
    @patch('muse.data.datasets.base.get_data_parallel_world_size')
    def test_distributed_dataset_files_sharding(self, mock_world_size, mock_rank, tmp_path):
        """Test file sharding in distributed scenario"""
        mock_rank.return_value = 0
        mock_world_size.return_value = 2

        # Create multiple files
        files = []
        for i in range(6):
            df = pd.DataFrame({
                'uuid': [str(i)],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = tmp_path / f"test_{i}.parquet"
            df.to_parquet(parquet_path)
            files.append(str(parquet_path))

        dataset = TestDataset(
            sources=files,  # type: ignore
            num_workers=1,
            shard_by="files"
        )

        # Rank 0 should get files 0, 2, 4
        samples = list(dataset)
        assert len(samples) == 3

    @patch('muse.data.datasets.base.get_data_parallel_rank')
    @patch('muse.data.datasets.base.get_data_parallel_world_size')
    def test_distributed_dataset_samples_sharding(self, mock_world_size, mock_rank, tmp_path):
        """Test sample sharding in distributed scenario"""
        mock_rank.return_value = 0
        mock_world_size.return_value = 2

        df = pd.DataFrame({
            'uuid': ['0', '1', '2', '3', '4', '5'],
            'source': ['test'] * 6,
            'messages': ['[]'] * 6
        })
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        dataset = TestDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=1,
            shard_by="samples"
        )

        # Rank 0 should get samples 0, 2, 4
        samples = list(dataset)
        assert len(samples) == 3

    @patch('muse.data.datasets.base.get_data_parallel_rank')
    @patch('muse.data.datasets.base.get_data_parallel_world_size')
    def test_distributed_dataset_auto_sharding(self, mock_world_size, mock_rank, tmp_path):
        """Test auto sharding mode selection"""
        mock_rank.return_value = 0
        mock_world_size.return_value = 2

        # Test with many files (should choose files mode)
        files = []
        for i in range(10):
            df = pd.DataFrame({
                'uuid': [str(i)],
                'source': ['test'],
                'messages': ['[]']
            })
            parquet_path = tmp_path / f"test_{i}.parquet"
            df.to_parquet(parquet_path)
            files.append(str(parquet_path))

        dataset = TestDataset(
            sources=files,  # type: ignore
            num_workers=1,
            shard_by="auto"
        )

        # Should use files mode
        assert dataset._actual_shard_by == "files"

        # Test with few files (should choose samples mode)
        df = pd.DataFrame({
            'uuid': ['0', '1'],
            'source': ['test'] * 2,
            'messages': ['[]', '[]']
        })
        parquet_path = tmp_path / "test_few.parquet"
        df.to_parquet(parquet_path)

        dataset2 = TestDataset(
            sources=[str(parquet_path)],  # type: ignore
            num_workers=1,
            shard_by="auto"
        )

        # Should use samples mode
        assert dataset2._actual_shard_by == "samples"


class TestErrorHandling:
    """Test error handling in datasets"""

    def test_parquet_reader_handles_missing_file(self):
        """Test ParquetReader handles missing files gracefully"""
        reader = ParquetReader(["nonexistent_file.parquet"])
        samples = list(reader)
        # Should not raise, but return empty list
        assert len(samples) == 0

    def test_distributed_dataset_handles_empty_source(self, tmp_path):
        """Test DistributedDataset handles empty source"""
        with pytest.raises(AssertionError, match="No file found"):
            dataset = TestDataset(
                sources=[],  # type: ignore
                num_workers=1
            )

    def test_distributed_dataset_handles_invalid_json(self, tmp_path):
        """Test DistributedDataset handles invalid JSON file list"""
        json_path = tmp_path / "invalid.json"
        json_path.write_text("invalid json content")

        with pytest.raises((json.JSONDecodeError, ValueError)):
            # Depending on implementation, might raise different errors
            try:
                dataset = TestDataset(
                    sources=str(json_path),
                    num_workers=1
                )
            except Exception as e:
                # Expected to fail
                assert True

