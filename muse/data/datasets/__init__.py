from .base import DistributedDataset
from .text import TextDataset
from .image import Text2ImageDataset, Token2ImageDataset, \
    MultiScaleDatasetWrapper

# Backward compatibility - expose all classes
__all__ = [
    'DistributedDataset',
    'TextDataset',
    'Text2ImageDataset',
    'Token2ImageDataset',
    'MultiScaleDatasetWrapper'
]
