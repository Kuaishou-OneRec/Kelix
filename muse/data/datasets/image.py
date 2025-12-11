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
Image-Text Dataset for Diffusion Training.

This module implements datasets for text-to-image diffusion model training.
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import os
import random
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T

from muse.data.datasets.base import DistributedDataset, load_image


logger = logging.getLogger(__name__)


class TextToImageDataset(DistributedDataset):
    """Dataset for text-to-image pairs.
    
    This dataset loads image-text pairs and processes them for diffusion training.
    It supports:
    - Online image loading and transformation
    - Multi-resolution support
    
    Args:
        sources: Path(s) to parquet files or directory
        image_size: Target image size (int or tuple)
        vae: Optional VAE model for encoding images
        tokenizer: Optional tokenizer for text
        text_encoder: Optional text encoder model
        max_text_length: Maximum text sequence length
        center_crop: Whether to center crop images
        random_flip: Whether to apply random horizontal flip
        **kwargs: Additional args passed to DistributedDataset
    """
    
    def __init__(
        self,
        sources: Union[List[str], str],
        image_size: Union[int, Tuple[int, int]] = 1024,
        vae: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        max_text_length: int = 300,
        center_crop: bool = True,
        **kwargs,
    ):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.center_crop = center_crop
        
        # Build transforms
        self.transform = self._build_transform()
        
        super().__init__(sources, **kwargs)
    
    def _build_transform(self) -> Callable:
        """Build image transformation pipeline."""
        transform_list = []
        
        if self.center_crop:
            # 先按短边 Resize（保持宽高比），再 CenterCrop
            target_size = min(self.image_size)
            transform_list.extend([
                T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),  # 短边缩放到 target_size
                T.CenterCrop(target_size),  # 裁剪成正方形
            ])
        
        transform_list.extend([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        
        return T.Compose(transform_list)
    
    def _load_image(self, image_data: Any) -> Optional[Image.Image]:
        """Load image from various formats.
        
        Args:
            image_data: Image path, base64 string, or bytes
        
        Returns:
            PIL Image or None if loading fails
        """
        try:
            if isinstance(image_data, str):
                return load_image(image_data)
            elif isinstance(image_data, bytes):
                from io import BytesIO
                return Image.open(BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                return image_data
            elif hasattr(image_data, 'tobytes'):
                # numpy array
                return Image.fromarray(image_data)
            else:
                return load_image(str(image_data))
        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            return None
    
    def process(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample.
        
        Args:
            sample: Raw sample dict from parquet
        
        Returns:
            Processed sample dict or None if processing fails
        """
        result = {}

        # Load and process image
        image_data = sample["image"]
        if image_data is None:
            return None
        
        image = self._load_image(image_data)
        if image is None:
            return None
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms
        image = self.transform(image)
        result["image"] = image

        # Tokenize text
        text = sample["text"]


        result["text"] = text
        result["input_ids"] = self.tokenizer.encode(text)
        result["attention_mask"] = [1] * len(result["input_ids"])
        
        return result
    
      def get_content(self,
                  sample: Dict[str, Any],
                  key: str) -> List[Dict[str, Any]]:
        """Get content from sample"""
        content = sample.get(key, "[]")
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return []
            return content

    def get_content(self,
                    sample: Dict[str, Any],
                    key: str) -> List[Dict[str, Any]]:
        """Get content from sample"""
        content = sample.get(key, "[]")
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            return []
            return content

    def extract_image_text(sample: Dict[str, Any]) -> Tuple[Optional[Image.Image], Optional[str]]:
        image = sample.get("image", None)
        text = sample.get("text", None)
        if image and text:
            return {
                "image": image,
                "text": text
            }
        # 需要保证messages和segments都是单轮，否则逻辑会有问题
        messages = self.get_content(sample, "messages")
        segments = self.get_content(sample, "segments")
        if messages:
            for turn in messages:
                if turn["role"] == "user":
                    content = turn["content"]
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        for block in content:
                            if block["type"] == "text":
                                text = block["text"]
                                break
                elif turn["role"] == "assistant":
                    content = turn["content"]
                    for block in content:
                        if block["type"] == "image":
                            image = block["image"]
                            break
                        # TODO: 兼容有问题的数据格式，后续得下掉
                        if block["type"] == "image_gen":
                            image = block["image_gen"]
                            break
                else:
                    continue

        if segments:
            for segment in segments:
                if segment["type"] == "image":
                    image = segment["image"]
                    break
                elif segment["type"] == "text":
                    text = segment["text"]
                    break
                else:
                    continue
        return {
            "image": image,
            "text": text
        }


    def process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pair = self.extract_image_text(sample)
        return self.process(pair)

    def collate_fn(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Collate batch samples.
        
        Args:
            batch: List of processed samples
        
        Returns:
            Collated batch dict
        """
        result = {}
        
        # Collate tensors
        for key in ["image", "input_ids", "attention_mask"]:
            if key in batch[0]:
                result[key] = torch.stack([s[key] for s in batch])
        
        return result
