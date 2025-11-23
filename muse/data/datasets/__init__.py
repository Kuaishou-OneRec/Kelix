from .base import DistributedDataset
from .text import TextDataset

# Backward compatibility - expose all classes
__all__ = [
    'DistributedDataset',
    'TextDataset',
]
