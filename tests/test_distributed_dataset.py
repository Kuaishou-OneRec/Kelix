#!/usr/bin/env python3

import os
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from torch.utils.data import DataLoader
import pandas as pd
import random
import shutil

# Import the classes we want to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from muse.data.datasets.base import DistributedDataset, ParquetDataset, get_worker_info
from muse.data.datasets.recoverable import RecoverableDistributedDataset
from muse.data.datasets.factory import create_dataset


class MockDataset(DistributedDataset):
    """Mock dataset for testing purposes"""
    
    def process(self, sample):
        """Simple process function that just returns the sample with a processed flag"""
        return {
            **sample,
            "processed": True,
            "sample_id": sample.get("__key__", "unknown")
        }


class MockRecoverableDataset(RecoverableDistributedDataset):
    """Mock recoverable dataset for testing purposes"""
    
    def __init__(self, sources, **kwargs):
        base_dataset = MockDataset(sources, **kwargs)
        shuffle_buffer_size = kwargs.pop('shuffle_buffer_size', 0)
        enable_checkpointing = kwargs.pop('enable_checkpointing', False)
        checkpoint_dir = kwargs.pop('checkpoint_dir', None)
        checkpoint_interval = kwargs.pop('checkpoint_interval', 1000)
        
        super().__init__(
            dataset=base_dataset,
            shuffle_buffer_size=shuffle_buffer_size,
            enable_checkpointing=enable_checkpointing,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval
        )


class TestDistributedDataset(unittest.TestCase):
    """Test suite for DistributedDataset with shuffle buffer and checkpoint functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create mock parquet files for testing
        self.mock_files = [
            os.path.join(self.temp_dir, f"test_file_{i}.parquet") 
            for i in range(3)
        ]
        
        # Create some test data
        self.test_samples = [
            {
                "__key__": f"sample_{i}",
                "__url__": "test_file.parquet",
                "messages": json.dumps([{"role": "user", "content": f"Test message {i}"}]),
                "images": "{}",
                "videos": "{}",
                "source": "test",
                "uuid": f"uuid_{i}"
            }
            for i in range(10)
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_backward_compatibility(self):
        """Test that default parameters maintain backward compatibility"""
        # Test base dataset remains clean and simple
        base_dataset = MockDataset(
            sources=self.mock_files,
            rank=0,
            world_size=1,
            num_workers=1,
            seed=42
        )
        
        # Base dataset should not have advanced features
        self.assertFalse(hasattr(base_dataset, 'shuffle_buffer_size'))
        self.assertFalse(hasattr(base_dataset, 'enable_checkpointing'))
        self.assertFalse(hasattr(base_dataset, 'save_checkpoint'))
        
        # Factory function should return base dataset when no advanced features requested
        factory_dataset = create_dataset(
            sources=self.mock_files,
            rank=0,
            world_size=1
        )
        self.assertIsInstance(factory_dataset, DistributedDataset)
        self.assertNotIsInstance(factory_dataset, RecoverableDistributedDataset)
        
        # RecoverableDataset with default settings should be disabled
        recoverable_dataset = MockRecoverableDataset(
            sources=self.mock_files,
            shuffle_buffer_size=0,
            enable_checkpointing=False,
            rank=0,
            world_size=1
        )
        
        # Should have attributes but disabled
        self.assertEqual(recoverable_dataset.shuffle_buffer_size, 0)
        self.assertFalse(recoverable_dataset.enable_checkpointing)
        self.assertFalse(recoverable_dataset._should_checkpoint())
    
    def test_shuffle_buffer_initialization(self):
        """Test shuffle buffer initialization and configuration"""
        buffer_size = 5
        
        # Test clean DistributedDataset (no shuffle/checkpoint logic)
        base_dataset = MockDataset(
            sources=self.mock_files,
            rank=0,
            world_size=1,
            seed=42
        )
        # Should not have any buffer attributes
        self.assertFalse(hasattr(base_dataset, 'shuffle_buffer_size'))
        self.assertFalse(hasattr(base_dataset, 'buffer_a'))
        
        # Test RecoverableDistributedDataset with buffering enabled
        recoverable_dataset = MockRecoverableDataset(
            sources=self.mock_files,
            shuffle_buffer_size=buffer_size,
            enable_checkpointing=True,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_interval=3,
            rank=0,
            world_size=1,
            seed=42
        )
        
        self.assertEqual(recoverable_dataset.shuffle_buffer_size, buffer_size)
        self.assertTrue(recoverable_dataset.enable_checkpointing)
        self.assertEqual(recoverable_dataset.checkpoint_interval, 3)
        self.assertEqual(recoverable_dataset.checkpoint_dir, self.checkpoint_dir)
        
        # Test double buffer initialization
        self.assertEqual(len(recoverable_dataset.buffer_a), 0)
        self.assertEqual(len(recoverable_dataset.buffer_b), 0)
        self.assertFalse(recoverable_dataset.buffer_a_ready)
        self.assertFalse(recoverable_dataset.buffer_b_ready)
        self.assertIsNone(recoverable_dataset.current_consume_buffer)
        self.assertIsNone(recoverable_dataset.current_fill_buffer)
    
    def test_checkpoint_path_generation(self):
        """Test checkpoint path generation"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            enable_checkpointing=True,
            checkpoint_dir=self.checkpoint_dir,
            rank=2,
            world_size=4
        )
        
        worker_id = 3
        expected_path = os.path.join(
            self.checkpoint_dir, 
            f"recoverable_dataset_checkpoint_rank2_worker{worker_id}.json"
        )
        actual_path = dataset._get_checkpoint_path(worker_id)
        self.assertEqual(actual_path, expected_path)
    
    def test_checkpoint_save_load_cycle(self):
        """Test saving and loading checkpoints"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            enable_checkpointing=True,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_interval=2,
            rank=1,
            world_size=2,
            seed=42
        )
        
        # Set some state to save
        dataset.current_file_idx = 2
        dataset.current_row_idx = 15
        dataset.total_samples_processed = 25
        dataset.samples_since_checkpoint = 2
        
        # Save checkpoint
        worker_id = 0
        global_step = 100
        dataset.save_checkpoint(worker_id, global_step)
        
        # Verify checkpoint file was created
        checkpoint_path = dataset._get_checkpoint_path(worker_id)
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Create new dataset instance and load checkpoint
        new_dataset = MockRecoverableDataset(
            sources=self.mock_files,
            enable_checkpointing=True,
            checkpoint_dir=self.checkpoint_dir,
            rank=1,
            world_size=2,
            seed=42
        )
        
        # Load checkpoint
        success = new_dataset.load_checkpoint(worker_id)
        self.assertTrue(success)
        
        # Verify state was restored
        self.assertEqual(new_dataset.current_file_idx, 2)
        self.assertEqual(new_dataset.current_row_idx, 15)
        self.assertEqual(new_dataset.total_samples_processed, 25)
        self.assertEqual(new_dataset.samples_since_checkpoint, 0)  # Reset after load
    
    def test_checkpoint_validation(self):
        """Test checkpoint validation for rank/worker mismatch"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            enable_checkpointing=True,
            checkpoint_dir=self.checkpoint_dir,
            rank=0,
            world_size=1
        )
        
        # Save checkpoint
        dataset.save_checkpoint(0, 100)
        
        # Try to load with different rank
        wrong_rank_dataset = MockRecoverableDataset(
            sources=self.mock_files,
            enable_checkpointing=True,
            checkpoint_dir=self.checkpoint_dir,
            rank=1,  # Different rank
            world_size=1
        )
        
        # Should fail validation
        success = wrong_rank_dataset.load_checkpoint(0)
        self.assertFalse(success)
    
    def test_should_checkpoint_logic(self):
        """Test checkpoint interval logic"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            enable_checkpointing=True,
            checkpoint_interval=5
        )
        
        # Initially should not checkpoint
        self.assertFalse(dataset._should_checkpoint())
        
        # After reaching interval, should checkpoint
        dataset.samples_since_checkpoint = 5
        self.assertTrue(dataset._should_checkpoint())
        
        # After exceeding interval, should still checkpoint
        dataset.samples_since_checkpoint = 10
        self.assertTrue(dataset._should_checkpoint())
    
    def test_buffer_filling_and_shuffling(self):
        """Test double buffer filling and shuffling functionality"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            shuffle_buffer_size=3,
            seed=42
        )
        
        # Create mock iterator
        mock_samples = [{"id": i, "__key__": f"key_{i}"} for i in range(5)]
        mock_iter = iter(mock_samples)
        
        # Test double buffer filling
        samples_added = dataset._fill_buffer_completely(dataset.buffer_a, mock_iter)
        self.assertEqual(samples_added, 3)  # Should fill to buffer size
        self.assertEqual(len(dataset.buffer_a), 3)
        
        # Test shuffling
        original_buffer = dataset.buffer_a.copy()
        dataset._shuffle_buffer(dataset.buffer_a)
        
        # Should have same samples but potentially different order
        shuffled_ids = [s["id"] for s in dataset.buffer_a]
        original_ids = [s["id"] for s in original_buffer]
        self.assertEqual(sorted(shuffled_ids), sorted(original_ids))
        
        # Test buffer management methods
        buffer_by_name = dataset._get_buffer_by_name('a')
        self.assertIs(buffer_by_name, dataset.buffer_a)
        
        dataset._set_buffer_ready_flag('a', True)
        self.assertTrue(dataset._get_buffer_ready_flag('a'))
        
        # Test buffer switching
        dataset.current_consume_buffer = 'a'
        dataset.current_fill_buffer = 'b'
        dataset._switch_buffers()
        self.assertEqual(dataset.current_consume_buffer, 'b')
        self.assertEqual(dataset.current_fill_buffer, 'a')
    
    @patch('muse.data.datasets.base.get_worker_info')
    def test_iteration_with_shuffle_disabled(self, mock_get_worker_info):
        """Test iteration behavior when shuffle is disabled (backward compatibility)"""
        mock_get_worker_info.return_value = (0, 1)  # worker_id=0, num_workers=1
        
        dataset = MockDataset(
            sources=self.mock_files,
            shuffle_buffer_size=0,  # Disabled
            enable_checkpointing=False,
            rank=0,
            world_size=1,
            shard_by="samples"
        )
        
        # Mock the underlying dataset to return test samples
        mock_base_dataset = Mock()
        mock_base_dataset.__iter__ = Mock(return_value=iter(self.test_samples[:3]))
        dataset.dataset = mock_base_dataset
        
        # Iterate and collect samples
        processed_samples = list(dataset)
        
        # Should have processed all samples without shuffling
        self.assertEqual(len(processed_samples), 3)
        for sample in processed_samples:
            self.assertTrue(sample["processed"])
    
    def test_file_list_loading(self):
        """Test different ways of loading file lists"""
        # Test with list of files
        dataset = MockDataset(sources=self.mock_files)
        files = dataset._load_file_list()
        self.assertEqual(files, self.mock_files)
        
        # Test with JSON file
        json_file = os.path.join(self.temp_dir, "files.json")
        with open(json_file, 'w') as f:
            json.dump([f"{f}.parquet" for f in self.mock_files], f)
        
        dataset = MockDataset(sources=json_file)
        files = dataset._load_file_list()
        expected = [f"{f}.parquet" for f in self.mock_files]
        self.assertEqual(sorted(files), sorted(expected))
    
    def test_shard_by_modes(self):
        """Test different sharding modes"""
        # Test auto mode with sufficient files
        dataset = MockDataset(
            sources=self.mock_files * 5,  # 15 files
            num_workers=2,
            world_size=2,  # Total 4 workers
            shard_by="auto"
        )
        dataset._build()
        self.assertEqual(dataset._actual_shard_by, "files")
        
        # Test auto mode with insufficient files
        dataset = MockDataset(
            sources=self.mock_files[:2],  # 2 files
            num_workers=2,
            world_size=2,  # Total 4 workers
            shard_by="auto"
        )
        dataset._build()
        self.assertEqual(dataset._actual_shard_by, "samples")
        
        # Test forced files mode with sufficient files
        dataset = MockDataset(
            sources=self.mock_files * 3,  # 9 files for 8 workers - sufficient
            num_workers=4,
            world_size=2,  # Total 8 workers  
            shard_by="files"
        )
        dataset._build()
        self.assertEqual(dataset._actual_shard_by, "files")
        
        # Test forced files mode with insufficient files (should auto-switch)
        dataset = MockDataset(
            sources=self.mock_files,  # 3 files for 8 workers - insufficient
            shard_by="files"
        )
        dataset._build()
        self.assertEqual(dataset._actual_shard_by, "samples")  # Should auto-switch
    
    def test_double_buffer_streaming(self):
        """Test that double buffer system maintains continuous flow"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            shuffle_buffer_size=4,
            enable_checkpointing=False,
            rank=0,
            world_size=1,
            seed=42
        )
        
        # Create a longer sequence of samples to test streaming behavior
        test_samples = [
            {"__key__": f"sample_{i}", "value": i} 
            for i in range(10)
        ]
        mock_iter = iter(test_samples)
        
        # Test the double buffer behavior
        worker_id = 0
        samples_yielded = []
        
        # Simulate the double buffer iterator
        for sample in dataset._iter_with_double_buffer(mock_iter, worker_id):
            samples_yielded.append(sample)
            
            # Break after getting a few samples to test partial consumption
            if len(samples_yielded) >= 6:
                break
        
        # Should have yielded samples
        self.assertEqual(len(samples_yielded), 6)
        
        # All samples should be processed
        for sample in samples_yielded:
            self.assertTrue(sample["processed"])
        
        # Check that we got valid samples from the original set
        all_original_keys = [f"sample_{i}" for i in range(10)]
        yielded_keys = [sample["sample_id"] for sample in samples_yielded]
        
        # All yielded samples should be from the original set
        for key in yielded_keys:
            self.assertIn(key, all_original_keys)
        
        # Should have yielded exactly 6 unique samples
        self.assertEqual(len(set(yielded_keys)), 6)
    
    def test_double_buffer_shuffle_quality(self):
        """Test that double buffer shuffle produces good randomness"""
        buffer_size = 5
        test_samples = [{"__key__": f"sample_{i}", "value": i} for i in range(20)]
        
        # Test double buffer shuffle
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            shuffle_buffer_size=buffer_size,
            rank=0,
            world_size=1,
            seed=12345  # Fixed seed for reproducibility
        )
        
        double_buffer_results = []
        mock_iter = iter(test_samples.copy())
        for sample in dataset._iter_with_double_buffer(mock_iter, 0):
            double_buffer_results.append(sample["sample_id"])
        
        # Should produce the same number of samples
        self.assertEqual(len(double_buffer_results), len(test_samples))
        
        # Should contain all original samples
        original_keys = [f"sample_{i}" for i in range(20)]
        self.assertEqual(sorted(double_buffer_results), sorted(original_keys))
        
        # Order should be different from original (shuffled)
        self.assertNotEqual(double_buffer_results, original_keys)
    
    def test_double_buffer_switching(self):
        """Test that double buffer switching works correctly"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            shuffle_buffer_size=3,  # Small buffer for easy testing
            rank=0,
            world_size=1,
            seed=42
        )
        
        # Create test samples - enough for multiple buffer switches
        test_samples = [{"__key__": f"sample_{i}", "value": i} for i in range(10)]
        mock_iter = iter(test_samples)
        
        # Track buffer switches by mocking the switch method
        original_switch = dataset._switch_buffers
        switch_calls = []
        
        def tracking_switch():
            switch_calls.append({
                'before_consume': dataset.current_consume_buffer,
                'before_fill': dataset.current_fill_buffer
            })
            return original_switch()
        
        dataset._switch_buffers = tracking_switch
        
        # Iterate through samples to trigger buffer switches
        samples_consumed = 0
        for sample in dataset._iter_with_double_buffer(mock_iter, 0):
            samples_consumed += 1
            if samples_consumed >= 8:  # Consume enough to trigger switches
                break
        
        # Should have made at least one buffer switch
        self.assertGreater(len(switch_calls), 0)
        
        # Verify that samples were consumed
        self.assertEqual(samples_consumed, 8)
        
        # Wait for any background threads to complete
        if dataset.fill_thread and dataset.fill_thread.is_alive():
            dataset.fill_thread.join(timeout=1.0)

    def test_double_buffer_complete_functionality(self):
        """Test complete double buffer functionality including async filling and shuffle quality"""
        dataset = MockRecoverableDataset(
            sources=self.mock_files,
            shuffle_buffer_size=4,
            rank=0,
            world_size=1,
            seed=42
        )
        
        # Create test samples
        test_samples = [{"__key__": f"sample_{i}", "value": i} for i in range(12)]
        mock_iter = iter(test_samples)
        
        # Test complete iteration through double buffer
        consumed_samples = []
        for sample in dataset._iter_with_double_buffer(mock_iter, 0):
            consumed_samples.append(sample)
        
        # Should have consumed all samples
        self.assertEqual(len(consumed_samples), 12)
        
        # All samples should be processed
        for sample in consumed_samples:
            self.assertTrue(sample["processed"])
        
        # Verify all original samples were consumed (though in different order due to shuffle)
        consumed_keys = [sample["sample_id"] for sample in consumed_samples]
        expected_keys = [f"sample_{i}" for i in range(12)]
        self.assertEqual(sorted(consumed_keys), sorted(expected_keys))
        
        # Order should be different from original due to shuffle
        self.assertNotEqual(consumed_keys, expected_keys)
        
        # Verify buffer state after completion
        self.assertEqual(len(dataset.buffer_a), 0)
        self.assertEqual(len(dataset.buffer_b), 0)
        
        # Wait for any remaining threads
        if dataset.fill_thread and dataset.fill_thread.is_alive():
            dataset.fill_thread.join(timeout=1.0)

    def test_factory_function(self):
        """Test factory function creates appropriate dataset type"""
        # Test factory returns base dataset when no advanced features needed
        basic_dataset = create_dataset(
            sources=self.mock_files,
            rank=0,
            world_size=1
        )
        self.assertIsInstance(basic_dataset, DistributedDataset)
        self.assertNotIsInstance(basic_dataset, RecoverableDistributedDataset)
        
        # Test factory returns recoverable dataset when buffering enabled
        buffered_dataset = create_dataset(
            sources=self.mock_files,
            shuffle_buffer_size=100,
            rank=0,
            world_size=1
        )
        self.assertIsInstance(buffered_dataset, RecoverableDistributedDataset)
        self.assertIsInstance(buffered_dataset.dataset, DistributedDataset)
        
        # Test factory returns recoverable dataset when checkpointing enabled
        checkpoint_dataset = create_dataset(
            sources=self.mock_files,
            enable_checkpointing=True,
            rank=0,
            world_size=1
        )
        self.assertIsInstance(checkpoint_dataset, RecoverableDistributedDataset)
        
        # Test factory returns recoverable dataset when both enabled
        full_dataset = create_dataset(
            sources=self.mock_files,
            shuffle_buffer_size=50,
            enable_checkpointing=True,
            checkpoint_dir=self.checkpoint_dir,
            rank=0,
            world_size=1
        )
        self.assertIsInstance(full_dataset, RecoverableDistributedDataset)
        self.assertEqual(full_dataset.shuffle_buffer_size, 50)
        self.assertTrue(full_dataset.enable_checkpointing)

    def test_architecture_separation(self):
        """Test that the new architecture properly separates concerns"""
        # Base dataset should only handle data distribution
        base_dataset = MockDataset(
            sources=self.mock_files,
            rank=0,
            world_size=1
        )
        
        # Should not have buffer or checkpoint attributes
        self.assertFalse(hasattr(base_dataset, 'shuffle_buffer_size'))
        self.assertFalse(hasattr(base_dataset, 'enable_checkpointing'))
        self.assertFalse(hasattr(base_dataset, 'buffer_a'))
        self.assertFalse(hasattr(base_dataset, 'save_checkpoint'))
        
        # Recoverable dataset should handle advanced features
        recoverable_dataset = RecoverableDistributedDataset(
            dataset=base_dataset,
            shuffle_buffer_size=10,
            enable_checkpointing=True
        )
        
        # Should have all advanced features
        self.assertEqual(recoverable_dataset.shuffle_buffer_size, 10)
        self.assertTrue(recoverable_dataset.enable_checkpointing)
        self.assertTrue(hasattr(recoverable_dataset, 'buffer_a'))
        self.assertTrue(hasattr(recoverable_dataset, 'save_checkpoint'))
        
        # Should properly delegate to underlying dataset
        self.assertIs(recoverable_dataset.dataset, base_dataset)

    def test_integration_with_dataloader(self):
        """Test integration with PyTorch DataLoader using factory function"""
        # Test basic dataset integration
        basic_dataset = create_dataset(
            sources=self.mock_files,
            rank=0,
            world_size=1
        )
        
        # Mock the underlying dataset iteration
        with patch.object(basic_dataset, 'dataset') as mock_dataset:
            mock_dataset.__iter__ = Mock(return_value=iter(self.test_samples[:5]))
            
            with patch('muse.data.datasets.base.get_worker_info', return_value=(0, 1)):
                dataloader = DataLoader(basic_dataset, batch_size=2, num_workers=0)
                batches = list(dataloader)
                self.assertGreater(len(batches), 0)
        
        # Test recoverable dataset integration
        recoverable_dataset = create_dataset(
            sources=self.mock_files,
            shuffle_buffer_size=2,
            rank=0,
            world_size=1
        )
        
        self.assertIsInstance(recoverable_dataset, RecoverableDistributedDataset)
        
        # Should be able to create dataloader without errors
        with patch.object(recoverable_dataset.dataset, 'dataset') as mock_dataset:
            mock_dataset.__iter__ = Mock(return_value=iter(self.test_samples[:3]))
            
            with patch('muse.data.datasets.base.get_worker_info', return_value=(0, 1)):
                dataloader = DataLoader(recoverable_dataset, batch_size=1, num_workers=0)
                # Should not error on creation
                self.assertIsNotNone(dataloader)


if __name__ == '__main__':
    # Set up basic test environment
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
