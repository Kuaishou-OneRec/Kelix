from .base import DistributedDataset, ParquetDataset, get_worker_info
from .recoverable import RecoverableDistributedDataset
from .factory import create_dataset, get_recoverable_dataset

# Backward compatibility - expose all classes
__all__ = [
    'DistributedDataset', 
    'ParquetDataset',
    'RecoverableDistributedDataset', 
    'create_dataset',
    'get_recoverable_dataset',
    'get_worker_info'
]
