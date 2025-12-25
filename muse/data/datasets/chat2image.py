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

from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Iterator
import os
import random
import json
import logging
import collections
from PIL import Image, ImageDraw, ImageFont


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T

from transformers import AutoTokenizer, AutoProcessor
from keye_vl_utils import process_vision_info


from muse.data.datasets.base import DistributedDataset, load_image
from muse.data.utils import (
    get_aspect_ratio_dict, 
    get_closest_ratio,
    get_resolution_level,
    ResolutionBudget,
    ResolutionBudgetConfig,
)

from torch.utils.data import IterableDataset

from muse.utils.common import print_rank_0
from muse.data.datasets.image import Token2ImageDataset

logger = logging.getLogger(__name__)

import os




class Chat2ImageDataset(Token2ImageDataset):
    """Dataset for chat-style image generation with message-based processing.
    
    This dataset extends Token2ImageDataset to support chat-style message processing.
    It uses the 'message' field from samples for apply_chat_template and includes
    all processor output fields in the result.
    
    Args:
        sources: Path(s) to parquet files or directory
        image_size: Target image size (int or tuple)
        processor_path: Path to processor
        max_condition_length: Maximum condition sequence length
        **kwargs: Additional args passed to DistributedDataset
    
    each sample shows as
    {
        "__key__": 000000000, 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Draw a cat."},
                ]
            },
            {"role": "assistant", "content": [
                {"type": "image", "image": "0.jpg"}
            ]},
        ],
        "images": {"0.jpg": "/path/to/image/0.jpg"},
        "source": "kwai_video",
        ...
    }
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _process_pair(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single image-text pair using chat-style message processing.
        
        Args:
            sample: Sample dict with 'image' and 'text' keys
        
        Returns:
            Processed sample dict or None if processing fails
        """
        result = {}

        # Load and process image
        image_data = sample.get("image")
        if image_data is None:
            return None
        
        image = load_image(image_data)
        if image is None:
            return None
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.multi_scale:
            target_h, target_w = sample["target_height"], sample["target_width"]
            target_image = self._build_multiscale_transform((target_h, target_w))(image)
        else:
            target_image = self.transform(image)

        # Get message from sample for chat template processing
        messages = sample["message"]

        # Apply chat template using the message from sample
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False
        )

        image_inputs, _, _ = process_vision_info(messages)

        # Process with processor and include ALL output fields
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        # Include all processor output fields in result
        for key, value in inputs.items():
            result[key] = value
        
        # Add the target image
        result["image"] = target_image
        
        return result


    def process(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample with message-based chat processing.
        
        Extracts image-text pair from sample and processes it using chat-style messages.
        
        Args:
            sample: Raw sample dict from parquet
        
        Returns:
            Processed sample dict or None if processing fails
        """
        def recursive_traverse(obj, call_back_function):
            """
            递归遍历dict/list对象，对每个对象（包括子对象）先执行回调函数
            
            参数:
                obj: 待遍历的对象，仅支持dict或list类型
                call_back_function: 回调函数，接收当前遍历的对象作为参数
            """
            # 第一步：调用回调函数处理当前对象
            call_back_function(obj)
            
            # 判断类型并递归遍历内部成员
            if isinstance(obj, list):
                # 遍历列表的每个元素
                for item in obj:
                    recursive_traverse(item, call_back_function)
            elif isinstance(obj, dict):
                # 遍历字典的每个值（key一般为不可变类型，无需递归）
                for value in obj.values():
                    recursive_traverse(value, call_back_function)

        pair = self.extract_image_text(sample)

        if pair:
            images = json.loads(sample.get("images", '{}'))
            image = pair["image"]
            if image in images:
                pair["image"] = images[image]
            
            metadata = json.loads(sample.get("metadata", '{}'))
            images_info = metadata.get("images_info", {})
            image_info = images_info.get(image, {})
            height = image_info.get("height", None)
            width = image_info.get("width", None)
            if height is not None and width is not None:
                pair["height"] = height
                pair["width"] = width

            messages = json.loads(sample["messages"])
            image_dict = json.loads(sample["images"])

            def call_back(x):
                if not isinstance(x, dict): return
                if x.get("type") in ("image_gen", "image"):
                    x["image"] = image_dict[x["image"]] if x["image"] in image_dict else x["image"]

            recursive_traverse(messages, call_back)
            pair["message"] = messages
            return pair
        return None

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Collate batch samples.
        
        Args:
            batch: List of processed samples
        
        Returns:
            Collated batch dict
            
        Note:
            For multi-scale training without AspectRatioBatchSampler,
            images in a batch may have different sizes. We resize all
            images to the first image's size for consistent batching.
        """
        # Here is the real process of batch.
        result = {}
        batch = [self._process_pair(sample) for sample in batch]

        # Concatenate pixel_values: [s, d] -> [S, d] where S is sum of all s
        result["pixel_values"] = torch.concat([s["pixel_values"] for s in batch], dim=0)
        result["image_grid_thw"] = torch.concat([s["image_grid_thw"] for s in batch], dim=0)

        for key in ["input_ids", "attention_mask"]:
            if key in batch[0]:
                result[key] = torch.concat([s[key] for s in batch], dim=1)       

        # Add cu_seqlens for flash_attention
        if "input_ids" in batch[0]:
            # Calculate sequence lengths for each sample
            seq_lens = [s["input_ids"].shape[1] for s in batch]
            # Create cumulative sequence lengths: [0, seq_len1, seq_len1+seq_len2, ...]
            cu_seqlens = torch.tensor([0] + seq_lens, dtype=torch.int32).cumsum(dim=0)
            result["cu_seqlens"] = cu_seqlens

        if "image" in batch[0]:
            result["image"] = torch.stack([s["image"] for s in batch])
            
        return result
