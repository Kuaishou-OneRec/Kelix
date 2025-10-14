from typing import Dict, Any, List, Optional, Union
import random
import json
import os
import time
import threading
from torch.utils.data import IterableDataset

from .base import DistributedDataset, get_worker_info, PARQUET_CACHE_DIR


class RecoverableDatasetWrapper(IterableDataset):
    """
    A wrapper around DistributedDataset that adds shuffle buffering and checkpoint recovery.
    
    Uses composition to separate concerns:
    - DistributedDataset: handles data distribution and sharding
    - RecoverableDistributedDataset: handles buffering, shuffling, and recovery
    """
    
    def __init__(self,
                 dataset: Dataset,
                 shuffle_buffer_size: int = 0,
                 enable_checkpointing: bool = False,
                 checkpoint_dir: str = None,
                 checkpoint_interval: int = 1000):
        """
        Args:
            dataset: The underlying DistributedDataset to wrap
            shuffle_buffer_size: Size of shuffle buffer (0 to disable)
            enable_checkpointing: Whether to enable checkpoint recovery
            checkpoint_dir: Directory for checkpoint files
            checkpoint_interval: Interval for saving checkpoints (in samples)
        """
        self.dataset = dataset
        self.shuffle_buffer_size = shuffle_buffer_size
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = checkpoint_dir or PARQUET_CACHE_DIR
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize shuffle buffer system if enabled
        if self.shuffle_buffer_size > 0:
            # Double buffer system for optimal performance
            self.buffer_a = []
            self.buffer_b = []
            self.current_consume_buffer = None  # Currently consuming buffer
            self.current_fill_buffer = None     # Currently filling buffer
            self.buffer_a_ready = False         # Buffer A ready for consumption
            self.buffer_b_ready = False         # Buffer B ready for consumption
            self.fill_thread = None             # Background fill thread
            self.stop_filling = False           # Signal to stop background filling
        
        # Initialize RNG for shuffling
        self.buffer_rng = random.Random(getattr(dataset, 'rng', random.Random()).getstate()[1][0] + 1)
        
        # Initialize checkpoint state
        self.current_file_idx = 0
        self.current_row_idx = 0
        self.total_samples_processed = 0
        self.samples_since_checkpoint = 0

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process sample through the underlying dataset"""
        return self.dataset.process(sample)

    def _get_checkpoint_path(self, worker_id: int) -> str:
        """Generate checkpoint file path for current rank and worker"""
        checkpoint_filename = f"recoverable_dataset_checkpoint_rank{self.dataset.rank}_worker{worker_id}.json"
        return os.path.join(self.checkpoint_dir, checkpoint_filename)

    def _fill_buffer_completely(self, target_buffer: List, dataset_iter) -> int:
        """Completely fill target buffer with samples from dataset iterator"""
        target_buffer.clear()
        samples_added = 0
        
        try:
            while len(target_buffer) < self.shuffle_buffer_size:
                sample = next(dataset_iter)
                if sample is not None:
                    target_buffer.append(sample)
                    samples_added += 1
                else:
                    continue
        except StopIteration:
            pass
            
        return samples_added

    def _shuffle_buffer(self, buffer: List):
        """Shuffle the contents of specified buffer"""
        if buffer:
            self.buffer_rng.shuffle(buffer)

    def _get_buffer_by_name(self, buffer_name: str) -> List:
        """Get buffer by name ('a' or 'b')"""
        if buffer_name == 'a':
            return self.buffer_a
        elif buffer_name == 'b':
            return self.buffer_b
        else:
            raise ValueError(f"Invalid buffer name: {buffer_name}")

    def _get_buffer_ready_flag(self, buffer_name: str) -> bool:
        """Get ready flag for specified buffer"""
        if buffer_name == 'a':
            return self.buffer_a_ready
        elif buffer_name == 'b':
            return self.buffer_b_ready
        else:
            raise ValueError(f"Invalid buffer name: {buffer_name}")

    def _set_buffer_ready_flag(self, buffer_name: str, ready: bool):
        """Set ready flag for specified buffer"""
        if buffer_name == 'a':
            self.buffer_a_ready = ready
        elif buffer_name == 'b':
            self.buffer_b_ready = ready
        else:
            raise ValueError(f"Invalid buffer name: {buffer_name}")

    def _switch_buffers(self):
        """Switch consume and fill buffers"""
        if self.current_consume_buffer == 'a':
            self.current_consume_buffer = 'b'
            self.current_fill_buffer = 'a'
        else:
            self.current_consume_buffer = 'a'
            self.current_fill_buffer = 'b'

    def _async_fill_buffer(self, buffer_name: str, dataset_iter):
        """Fill buffer asynchronously in background thread"""
        try:
            target_buffer = self._get_buffer_by_name(buffer_name)
            samples_added = self._fill_buffer_completely(target_buffer, dataset_iter)
            
            if samples_added > 0:
                # Shuffle the filled buffer
                self._shuffle_buffer(target_buffer)
                # Mark buffer as ready
                self._set_buffer_ready_flag(buffer_name, True)
            else:
                # No more data available
                self._set_buffer_ready_flag(buffer_name, False)
                
        except Exception as e:
            print(f"Error in async buffer fill: {e}")
            self._set_buffer_ready_flag(buffer_name, False)

    def _start_async_fill(self, buffer_name: str, dataset_iter):
        """Start asynchronous filling of specified buffer"""
        if self.fill_thread and self.fill_thread.is_alive():
            # Previous fill still running, wait for it to complete
            self.fill_thread.join(timeout=0.1)
        
        # Start new fill thread
        self.fill_thread = threading.Thread(
            target=self._async_fill_buffer,
            args=(buffer_name, dataset_iter),
            daemon=True
        )
        self.fill_thread.start()

    def _should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint"""
        return (self.enable_checkpointing and 
                self.samples_since_checkpoint >= self.checkpoint_interval)

    def save_checkpoint(self, worker_id: int, global_step: int = 0):
        """Save current dataset state to checkpoint file"""
        if not self.enable_checkpointing:
            return
            
        checkpoint_path = self._get_checkpoint_path(worker_id)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Serialize RNG state for JSON compatibility
        def serialize_rng_state(state):
            """Convert RNG state tuple to JSON-serializable format"""
            if isinstance(state, tuple):
                return [serialize_rng_state(item) for item in state]
            elif isinstance(state, list):
                return [serialize_rng_state(item) for item in state]
            else:
                return state

        # Prepare checkpoint state
        checkpoint_state = {
            "current_file_idx": self.current_file_idx,
            "current_row_idx": self.current_row_idx,
            "total_samples_processed": self.total_samples_processed,
            "buffer_rng_state": serialize_rng_state(self.buffer_rng.getstate()),
            "global_step": global_step,
            "rank": self.dataset.rank,
            "worker_id": worker_id,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "timestamp": time.time()
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_state, f, indent=2)
            
            self.samples_since_checkpoint = 0
            print(f"Saved recoverable dataset checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save recoverable dataset checkpoint: {e}")

    def load_checkpoint(self, worker_id: int) -> bool:
        """Load dataset state from checkpoint file"""
        if not self.enable_checkpointing:
            return False
            
        checkpoint_path = self._get_checkpoint_path(worker_id)
        
        if not os.path.exists(checkpoint_path):
            return False
            
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_state = json.load(f)
            
            # Validate checkpoint matches current configuration
            if (checkpoint_state.get("rank") != self.dataset.rank or 
                checkpoint_state.get("worker_id") != worker_id):
                print(f"Warning: Checkpoint mismatch for rank {self.dataset.rank}, worker {worker_id}")
                return False
            
            # Restore state
            self.current_file_idx = checkpoint_state.get("current_file_idx", 0)
            self.current_row_idx = checkpoint_state.get("current_row_idx", 0)
            self.total_samples_processed = checkpoint_state.get("total_samples_processed", 0)
            self.samples_since_checkpoint = 0
            
            # Restore RNG state
            if "buffer_rng_state" in checkpoint_state:
                def deserialize_rng_state(state):
                    """Convert JSON-deserialized state back to tuple format"""
                    if isinstance(state, list):
                        return tuple(deserialize_rng_state(item) for item in state)
                    else:
                        return state
                
                rng_state = deserialize_rng_state(checkpoint_state["buffer_rng_state"])
                self.buffer_rng.setstate(rng_state)
            
            print(f"Loaded recoverable dataset checkpoint from {checkpoint_path}")
            print(f"Resuming from file_idx={self.current_file_idx}, row_idx={self.current_row_idx}")
            return True
            
        except Exception as e:
            print(f"Failed to load recoverable dataset checkpoint: {e}")
            return False

    def _iter_passthrough(self, dataset_iter, worker_id: int):
        """Pass-through iteration without buffering (when shuffle_buffer_size=0)"""
        for sample in dataset_iter:
            inputs = self.process(sample)
            self.total_samples_processed += 1
            self.samples_since_checkpoint += 1
            
            # Save checkpoint if needed
            if self._should_checkpoint():
                self.save_checkpoint(worker_id)
                
            yield inputs

    def _iter_with_double_buffer(self, dataset_iter, worker_id: int):
        """Iterate using double buffer system for optimal shuffle performance"""
        print(f"  Initializing RecoverableDistributedDataset double buffer system (buffer_size={self.shuffle_buffer_size})")
        
        # Fill initial buffer A
        samples_in_a = self._fill_buffer_completely(self.buffer_a, dataset_iter)
        if samples_in_a == 0:
            print("  No samples available, exiting")
            return
        
        # Shuffle buffer A and mark it ready
        self._shuffle_buffer(self.buffer_a)
        self.buffer_a_ready = True
        
        # Set initial state: consume A, fill B
        self.current_consume_buffer = 'a'
        self.current_fill_buffer = 'b'
        
        # Start async filling of buffer B
        self._start_async_fill('b', dataset_iter)
        
        buffer_switch_count = 0
        
        while True:
            # Get current consume buffer
            consume_buffer = self._get_buffer_by_name(self.current_consume_buffer)
            consume_ready = self._get_buffer_ready_flag(self.current_consume_buffer)
            
            if not consume_ready or len(consume_buffer) == 0:
                # Current buffer is empty or not ready, try to switch
                fill_buffer_name = self.current_fill_buffer
                fill_buffer_ready = self._get_buffer_ready_flag(fill_buffer_name)
                
                if fill_buffer_ready:
                    # Switch to the filled buffer
                    print(f"  RecoverableDataset buffer switch #{buffer_switch_count + 1}: {self.current_consume_buffer} -> {fill_buffer_name}")
                    self._switch_buffers()
                    buffer_switch_count += 1
                    
                    # Mark old consume buffer as not ready and start filling it
                    self._set_buffer_ready_flag(self.current_fill_buffer, False)
                    self._start_async_fill(self.current_fill_buffer, dataset_iter)
                    
                    # Update consume buffer reference
                    consume_buffer = self._get_buffer_by_name(self.current_consume_buffer)
                else:
                    # Both buffers exhausted, end iteration
                    if self.fill_thread:
                        self.fill_thread.join(timeout=1.0)
                    print(f"  RecoverableDataset double buffer iteration complete after {buffer_switch_count} switches")  
                    break
            
            # Consume samples from current buffer
            while len(consume_buffer) > 0:
                sample = consume_buffer.pop(0)
                
                inputs = self.process(sample)
                self.total_samples_processed += 1
                self.samples_since_checkpoint += 1
                
                # Save checkpoint if needed
                if self._should_checkpoint():
                    self.save_checkpoint(worker_id)
                    
                yield inputs

    def __iter__(self):
        """Main iteration entry point"""
        worker_id, num_workers = get_worker_info()
        
        # Try to load from checkpoint first
        checkpoint_loaded = self.load_checkpoint(worker_id)
        
        # Get iterator from underlying dataset
        dataset_iter = iter(self.dataset)
        
        # Skip samples if resuming from checkpoint
        if checkpoint_loaded and self.total_samples_processed > 0:
            print(f"RecoverableDataset: Skipping {self.total_samples_processed} samples to resume from checkpoint")
            for _ in range(self.total_samples_processed):
                try:
                    next(dataset_iter)
                except StopIteration:
                    break
        
        # Choose iteration strategy based on buffer configuration
        if self.shuffle_buffer_size > 0:
            # Use double buffer system
            yield from self._iter_with_double_buffer(dataset_iter, worker_id)
        else:
            # Pass-through mode
            yield from self._iter_passthrough(dataset_iter, worker_id)



ParquetReader
FileBasedDataset
- 使用两个buffer
- buffer填满后，记录对应的file_idx和row_idx
- buffer切换后，说明这个buffer已经消费完，更新file_idx和row_idx到checkpoint中，用于下次恢复
- 从checkpoint中恢复后，文件跳到file_idx，跳过对应的row_idx，然后开始重建buffer，继续消费

Dataset:
- FileBasedDataset
- SampleProcessor
- RedistributeSample

RpcDataset

SampleProcessor
- TextSampleProcessor
- Qwen2VLSampleProcessor
- Qwen2_5_VLSampleProcessor
