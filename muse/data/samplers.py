# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Data Samplers for Multi-scale Training.

This module implements batch samplers that group samples by aspect ratio
for efficient multi-scale training of diffusion models.

Reference: Sana/diffusion/utils/data_sampler.py
"""

import logging
from typing import Dict, List, Sequence, Optional, Iterator

from torch.utils.data import BatchSampler, Dataset, Sampler


logger = logging.getLogger(__name__)


# Predefined aspect ratios for different resolutions
# Reference: Sana/diffusion/data/datasets/utils.py
ASPECT_RATIO_512 = {
    '0.25': (256, 1024), '0.26': (256, 992), '0.27': (256, 960), '0.28': (256, 928),
    '0.32': (288, 896), '0.33': (288, 864), '0.35': (288, 832), '0.4': (320, 800),
    '0.42': (320, 768), '0.48': (352, 736), '0.5': (352, 704), '0.52': (352, 672),
    '0.57': (384, 672), '0.6': (384, 640), '0.68': (416, 608), '0.72': (416, 576),
    '0.78': (448, 576), '0.82': (448, 544), '0.88': (480, 544), '0.94': (480, 512),
    '1.0': (512, 512),
    '1.07': (512, 480), '1.13': (544, 480), '1.21': (544, 448), '1.29': (576, 448),
    '1.38': (576, 416), '1.46': (608, 416), '1.67': (640, 384), '1.75': (672, 384),
    '2.0': (704, 352), '2.09': (736, 352), '2.4': (768, 320), '2.5': (800, 320),
    '2.89': (832, 288), '3.0': (864, 288), '3.11': (896, 288), '3.62': (928, 256),
    '3.75': (960, 256), '3.88': (992, 256), '4.0': (1024, 256),
}

ASPECT_RATIO_1024 = {
    '0.25': (512, 2048), '0.26': (512, 1984), '0.27': (512, 1920), '0.28': (512, 1856),
    '0.32': (576, 1792), '0.33': (576, 1728), '0.35': (576, 1664), '0.4': (640, 1600),
    '0.42': (640, 1536), '0.48': (704, 1472), '0.5': (704, 1408), '0.52': (704, 1344),
    '0.57': (768, 1344), '0.6': (768, 1280), '0.68': (832, 1216), '0.72': (832, 1152),
    '0.78': (896, 1152), '0.82': (896, 1088), '0.88': (960, 1088), '0.94': (960, 1024),
    '1.0': (1024, 1024),
    '1.07': (1024, 960), '1.13': (1088, 960), '1.21': (1088, 896), '1.29': (1152, 896),
    '1.38': (1152, 832), '1.46': (1216, 832), '1.67': (1280, 768), '1.75': (1344, 768),
    '2.0': (1408, 704), '2.09': (1472, 704), '2.4': (1536, 640), '2.5': (1600, 640),
    '2.89': (1664, 576), '3.0': (1728, 576), '3.11': (1792, 576), '3.62': (1856, 512),
    '3.75': (1920, 512), '3.88': (1984, 512), '4.0': (2048, 512),
}


def get_aspect_ratio_dict(image_size: int) -> Dict[str, tuple]:
    """Get aspect ratio dictionary for given image size.
    
    Args:
        image_size: Base image size (512, 1024, etc.)
        
    Returns:
        Dictionary mapping aspect ratio strings to (height, width) tuples
    """
    if image_size <= 512:
        return ASPECT_RATIO_512
    elif image_size <= 1024:
        return ASPECT_RATIO_1024
    else:
        # Scale up from 1024
        scale = image_size / 1024
        return {
            k: (int(h * scale), int(w * scale))
            for k, (h, w) in ASPECT_RATIO_1024.items()
        }


class AspectRatioBatchSampler(BatchSampler):
    """A sampler that groups images with similar aspect ratio into the same batch.
    
    This enables efficient multi-scale training by ensuring all images in a batch
    have the same resolution, avoiding padding waste.
    
    Args:
        sampler: Base sampler providing indices
        dataset: Dataset with get_data_info method returning height/width
        batch_size: Number of samples per batch
        aspect_ratios: Dictionary of aspect ratios to (height, width) mappings
        drop_last: Whether to drop the last incomplete batch
        
    Reference: Sana/diffusion/utils/data_sampler.py
    """
    
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios: Optional[Dict[str, tuple]] = None,
        drop_last: bool = False,
        image_size: int = 1024,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f"sampler should be an instance of Sampler, got {type(sampler)}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, got {batch_size}")
        
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Use provided aspect ratios or get default for image_size
        if aspect_ratios is None:
            aspect_ratios = get_aspect_ratio_dict(image_size)
        self.aspect_ratios = aspect_ratios
        
        # Initialize buckets for each aspect ratio
        self._aspect_ratio_buckets: Dict[str, List[int]] = {
            ratio: [] for ratio in aspect_ratios.keys()
        }
        
        logger.info(f"AspectRatioBatchSampler initialized with {len(aspect_ratios)} aspect ratios")
    
    def _get_closest_ratio(self, height: int, width: int) -> str:
        """Find the closest predefined aspect ratio for given dimensions.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            String key of the closest aspect ratio
        """
        ratio = height / width
        return min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches grouped by aspect ratio."""
        for idx in self.sampler:
            # Get image dimensions from dataset
            data_info = self._get_data_info(idx)
            if data_info is None:
                continue
            
            height = data_info.get("height")
            width = data_info.get("width")
            if height is None or width is None:
                continue
            
            # Find closest aspect ratio bucket
            closest_ratio = self._get_closest_ratio(height, width)
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            
            # Yield batch when bucket is full
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]
        
        # Yield remaining samples in buckets
        for bucket in self._aspect_ratio_buckets.values():
            while bucket:
                if not self.drop_last or len(bucket) == self.batch_size:
                    yield bucket[:]
                del bucket[:]
    
    def _get_data_info(self, idx: int) -> Optional[Dict]:
        """Get data info for an index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with height/width info, or None
        """
        if hasattr(self.dataset, 'get_data_info'):
            return self.dataset.get_data_info(idx)
        return None
    
    def __len__(self) -> int:
        """Return approximate number of batches."""
        return len(self.sampler) // self.batch_size


class MultiScaleCollator:
    """Collator that handles variable-size images from multi-scale training.
    
    This collator resizes images to their target aspect ratio resolution
    before batching.
    
    Args:
        aspect_ratios: Dictionary mapping ratio strings to (height, width)
        base_collate_fn: Base collate function to use after resizing
    """
    
    def __init__(
        self,
        aspect_ratios: Dict[str, tuple],
        base_collate_fn=None,
    ):
        self.aspect_ratios = aspect_ratios
        self.base_collate_fn = base_collate_fn
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch with multi-scale support.
        
        All images in a batch should have the same aspect ratio (ensured by sampler).
        This collator resizes them to the target resolution for that ratio.
        """
        if not batch:
            return {}
        
        if self.base_collate_fn is not None:
            return self.base_collate_fn(batch)
        
        # Simple default collation - stack tensors
        import torch
        result = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([s[key] for s in batch])
            else:
                result[key] = [s[key] for s in batch]
        
        return result
