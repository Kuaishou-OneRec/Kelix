from typing import Any


from logging import Logger


import traceback
import threading
from sortedcontainers import SortedDict
import hashlib
from torch.utils.data import IterableDataset
import traceback
from collections import defaultdict
import logging
logger: Logger = logging.getLogger(__name__)


class LocalShuffleBuffer(object):
    """
    A buffer class to implement local data shuffling.
    Maintains a fixed-size buffer to randomize the order of data samples during iteration.
    """
    
    def __init__(self, buffer_size: int = 2048, random_fetch: float = 0.01) -> None:
        """
        Initialize the LocalShuffleBuffer.
        
        Args:
            buffer_size: Maximum capacity of the buffer
            random_fetch: Probability to randomly fetch a sample before buffer is full (0.0-1.0)
        """
        self.random_fetch = random_fetch  # Probability for random extraction
        self.buffer_size = buffer_size    # Maximum number of samples to hold
        self.buffer = SortedDict()        # Sorted dictionary to store samples (key: hash, value: sample)
        self.count = defaultdict(int)     # Counter for statistics (adds, conflicts, epochs, etc.)
        self.count["buffer_epoch"] = 0    # Epoch counter for the buffer
        # Large multiplier to avoid hash collisions across epochs (0xffffffffffffffff in hex)
        self.buffer_multiply = int('f' * 16, 16)
        self.lock = threading.Lock()
        self.sample_bit_mapping: dict[Any, Any] = {}

    def __len__(self):
        return len(self.buffer)

    def _calc_sample_hash(self, obj: dict, buffer_epoch: int = None, index=0) -> int:
        """
        Calculate a unique hash for a sample to use as buffer key.
        Maps sample identifier to integer with random-like distribution.
        
        Args:
            obj: Sample object containing "uuid" and "source" keys
            
        Returns:
            Integer hash value
        """
        # Create unique string from sample identifiers
        unique_str = obj["uuid"] + obj["source"] + f"@ep{buffer_epoch}" + f"@idx{index}"
        
        # Generate MD5 hash and convert to integer
        hash_obj = hashlib.md5(unique_str.encode('utf-8'))
        hex_str = hash_obj.hexdigest()[:16]  # Take first 16 hex characters
        base_hash = int(hex_str, 16)
        
        if buffer_epoch is None: 
            buffer_epoch = self.count["buffer_epoch"]

        # Add epoch-based offset to prevent cross-epoch collisions
        return base_hash + self.buffer_multiply * buffer_epoch

    def add(self, obj: dict, fn: str = None, epoch: int = None, log_info: str = '', index=0) -> bool:
        """
        Add a sample to the buffer. Returns whether to continue adding (True) or fetch a sample (False).
        
        Args:
            obj: Sample object to add to buffer
            fn: Optional filename/identifier for logging
            epoch: Optional epoch index
            
        Returns:
            True if sample was added and buffer isn't ready for extraction, 
            False if extraction should occur (buffer full or random fetch triggered)
        """
        try:
            # Calculate hash for the sample
            obj_hash = self._calc_sample_hash(obj, buffer_epoch=epoch, index=index)
            self.count["add"] += 1  # Increment total addition counter
            
            # Update buffer epoch every buffer_size additions
            if self.count["add"] % self.buffer_size == 0:
                self.count["buffer_epoch"] += 1

            # Handle hash collisions (duplicate unique identifiers)
            if obj_hash in self.buffer:
                self.count["conflict"] += 1
                # Log warning periodically for collision rate
                if self.count["conflict"] % 100 == 0:
                    conflict_rate = self.count["conflict"] / self.count["add"]
                    logger.warning(
                        '=' * 30 + 
                        f"\n{log_info} Potential duplicate samples with same uuid/source! "
                        f"uuid={obj['uuid']}, source={obj['source']}, fn={fn}, "
                        f"conflict_rate={conflict_rate:.4f}, add_count={self.count['add']}\n"
                    )
            
            with self.lock:
                # Add sample to buffer
                self.buffer[obj_hash] = obj

            # Random fetch trigger (small probability to extract before buffer is full). It prevents downstream timeout error.
            if (obj_hash % 10000) < 10000 * self.random_fetch:
                return False  # Trigger extraction
            
            # Check if buffer has reached capacity
            if len(self.buffer) < self.buffer_size:
                return True   # Continue adding (buffer not full)
            else:
                return False  # Trigger extraction (buffer full)
                
        except Exception as e:
            logger.error(f"Error in LocalShuffleBuffer.add(): {traceback.format_exc()}")
            raise e

    def get(self) -> dict:
        """
        Extract a sample from the buffer.
        
        Returns:
            A sample object from the buffer
            
        Raises:
            ValueError: If buffer is empty
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot get sample from empty buffer")

        with self.lock:
            # Pop last item from SortedDict (provides random-like access due to hashing)
            return self.buffer.popitem(0)[1]

    def preprocess_df(self, df, epoch_idx, fn_index, fn):
        """Preprocess the dataframe to add necessary columns"""
        df['__epoch_idx__'] = epoch_idx
        df['__fn_idx__'] = fn_index
        df['__fn__'] = fn
        df['__sample_index__'] = range(len(df))
        return df

    

    def __len__(self) -> int:
        """Return current number of samples in the buffer"""
        return len(self.buffer)
