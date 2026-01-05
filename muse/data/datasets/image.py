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


logger = logging.getLogger(__name__)


class Text2ImageDataset(DistributedDataset):
    """Dataset for text-to-image pairs.
    
    This dataset loads image-text pairs and processes them for diffusion training.
    It supports:
    - Online image loading and transformation
    - Multi-resolution support
    - Complex Human Instruction (CHI) for enhanced text-image alignment
    
    Args:
        sources: Path(s) to parquet files or directory
        image_size: Target image size (int or tuple)
        vae: Optional VAE model for encoding images
        tokenizer: Optional tokenizer for text
        text_encoder: Optional text encoder model
        max_text_length: Maximum text sequence length
        center_crop: Whether to center crop images
        random_flip: Whether to apply random horizontal flip
        chi_prompt: Complex Human Instruction prompt lines (list of strings)
        **kwargs: Additional args passed to DistributedDataset
    """
    
    # Default CHI prompt from Sana paper
    DEFAULT_CHI_PROMPT = [
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
        '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.',
        '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.',
        'Here are examples of how to transform or refine prompts:',
        '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
        '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
        'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
        'User Prompt: ',
    ]
    
    def __init__(
        self,
        sources: Union[List[str], str],
        image_size: Union[int, Tuple[int, int]] = 1024,
        tokenizer_path: Optional[Any] = None,
        max_text_length: int = 300,
        center_crop: bool = True,
        chi_prompt: Optional[List[str]] = None,
        use_chi: bool = False,
        multi_scale: bool = False,
        padding_side: str = "right",
        **kwargs,
    ):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.base_image_size = image_size if isinstance(image_size, int) else image_size[0]
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = padding_side
        print_rank_0(f"Tokenizer padding side: {self.tokenizer.padding_side}")
        self.max_text_length = max_text_length
        self.center_crop = center_crop
        
        # CHI (Complex Human Instruction) support
        self.use_chi = use_chi
        if use_chi:
            chi_lines = chi_prompt if chi_prompt else self.DEFAULT_CHI_PROMPT
            self.chi_prompt = "\n".join(chi_lines)
            # Compute number of system prompt tokens for token selection
            self.num_chi_tokens = len(self.tokenizer.encode(self.chi_prompt))
            logger.info(f"CHI enabled with {self.num_chi_tokens} system prompt tokens")
        else:
            self.chi_prompt = None
            self.num_chi_tokens = 0
        
        # Multi-scale training support
        self.multi_scale = multi_scale
        if multi_scale:
            self.aspect_ratios = get_aspect_ratio_dict(self.base_image_size)
            logger.info(f"Multi-scale training enabled with {len(self.aspect_ratios)} aspect ratios")
        else:
            self.aspect_ratios = None
        
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
    
    def get_content(self,
                    sample: Dict[str, Any],
                    key: str) -> List[Dict[str, Any]]:
        """Get content from sample.
        
        Args:
            sample: Sample dict
            key: Key to get content from
            
        Returns:
            Parsed JSON content or empty list if parsing fails
        """
        content = sample.get(key, "[]")
        try:
            content = json.loads(content)
            return content
        except (json.JSONDecodeError, TypeError):
            return []

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Validate messages format.
        
        Messages must be single-turn: exactly 2 non-system messages (1 user + 1 assistant).
        System messages are allowed and will be skipped during validation and processing.
        - User message content: str or list with exactly 1 text block
        - Assistant message content: list with exactly 1 image/image_gen block
        
        Args:
            messages: List of message dicts
            
        Raises:
            ValueError: If messages format is invalid
        """
        # Filter out system messages for validation
        non_system_messages = [m for m in messages if m.get("role") != "system"]
        
        if len(non_system_messages) != 2:
            raise ValueError(
                f"Messages must have exactly 2 non-system messages (1 user + 1 assistant), "
                f"got {len(non_system_messages)}"
            )
        
        user_msg = None
        assistant_msg = None
        
        for msg in non_system_messages:
            role = msg.get("role")
            if role == "user":
                user_msg = msg
            elif role == "assistant":
                assistant_msg = msg
        
        if user_msg is None:
            raise ValueError("Messages must contain exactly 1 user message")
        if assistant_msg is None:
            raise ValueError("Messages must contain exactly 1 assistant message")
        
        # Validate user message content
        user_content = user_msg.get("content")
        if isinstance(user_content, list):
            text_blocks = [b for b in user_content if b.get("type") == "text"]
            if len(text_blocks) != 1:
                raise ValueError(
                    f"User message must contain exactly 1 text block, "
                    f"got {len(text_blocks)}"
                )
        elif not isinstance(user_content, str):
            raise ValueError(
                f"User message content must be str or list, got {type(user_content)}"
            )
        
        # Validate assistant message content
        assistant_content = assistant_msg.get("content")
        if not isinstance(assistant_content, list):
            raise ValueError(
                f"Assistant message content must be list, got {type(assistant_content)}"
            )
        
        image_blocks = [
            b for b in assistant_content 
            if b.get("type") in ("image", "image_gen")
        ]
        if len(image_blocks) != 1:
            raise ValueError(
                f"Assistant message must contain exactly 1 image block, "
                f"got {len(image_blocks)}"
            )

    def _validate_segments(self, segments: List[Dict[str, Any]]) -> None:
        """Validate segments format.
        
        Segments must have exactly 2 items:
        - First segment: type="text"
        - Second segment: type="image"
        
        Args:
            segments: List of segment dicts
            
        Raises:
            ValueError: If segments format is invalid
        """
        if len(segments) != 2:
            raise ValueError(
                f"Segments must have exactly 2 items, got {len(segments)}"
            )
        
        if segments[0].get("type") != "text":
            raise ValueError(
                f"First segment must be type='text', "
                f"got type='{segments[0].get('type')}'"
            )
        
        if segments[1].get("type") != "image":
            raise ValueError(
                f"Second segment must be type='image', "
                f"got type='{segments[1].get('type')}'"
            )

    def extract_image_text(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract image and text from sample.
        
        Supports multiple formats:
        - Direct image/text fields
        - Messages format (chat-style, single-turn only)
        - Segments format (exactly 2 segments: text + image)
        - Multiple captions with random selection
        
        Args:
            sample: Raw sample dict
            
        Returns:
            Dict with 'image' and 'text' keys
            
        Raises:
            ValueError: If messages or segments format is invalid
        """
        image = sample.get("image", None)
        text = sample.get("text", None)
        
        # Check for multiple captions
        captions = sample.get("captions", None)

        if captions and isinstance(captions, dict) and len(captions) > 1:
            # Multiple captions - random selection
            text = random.choice(list(captions.values()))
        elif captions and isinstance(captions, dict):
            # Single caption in dict format
            text = next(iter(captions.values()))
        
        if image and text:
            return {
                "image": image,
                "text": text
            }
        
        messages = self.get_content(sample, "messages")
        segments = self.get_content(sample, "segments")
        
        if messages:
            # Validate messages format
            self._validate_messages(messages)
            
            for turn in messages:
                if turn["role"] == "user":
                    content = turn["content"]
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        # Already validated: exactly 1 text block
                        for block in content:
                            if block["type"] == "text":
                                text = block["text"]
                                break
                elif turn["role"] == "assistant":
                    content = turn["content"]
                    # Already validated: exactly 1 image block
                    for block in content:
                        if block["type"] == "image":
                            image = block["image"]
                            break
                        if block["type"] == "image_gen":
                            # 兼容错误的格式，后面记得修复
                            # Gen_BLIP3o-Pretrain-Long-Caption/0.0.1
                            # type=image_gen, image: 
                            if "image" in block:
                                image = block["image"]
                            elif "image_gen" in block:
                                image = block["image_gen"]
                            break

        if segments:
            # Validate segments format
            self._validate_segments(segments)
            
            # First segment is text, second is image (validated)
            text = segments[0]["text"]
            image = segments[1]["image"]
        
        if image is None or text is None:
            return None

        return {
            "image": image,
            "text": text
        }

    def _build_multiscale_transform(self, target_size: Tuple[int, int]) -> Callable:
        """Build transform for a specific target size in multi-scale training.
        
        Args:
            target_size: Target (height, width) for this aspect ratio
            
        Returns:
            Transform pipeline for this size
        """
        transform_list = [
            T.Resize(target_size, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
        return T.Compose(transform_list)
    
    def _process_pair(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single image-text pair.
        
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
        
        # Apply transforms (with multi-scale support)
        if self.multi_scale and self.aspect_ratios:
            # Get original image size and find closest aspect ratio
            orig_w, orig_h = image.size
            closest_ratio = get_closest_ratio(orig_h, orig_w, self.aspect_ratios)
            target_h, target_w = self.aspect_ratios[closest_ratio]
            
            # Build and apply transform for this specific target size
            transform = self._build_multiscale_transform((target_h, target_w))
            image = transform(image)
            
            # Store aspect ratio info for potential use in collation
            result["aspect_ratio"] = closest_ratio
            result["target_size"] = (target_h, target_w)
        else:
            # Standard fixed-size transform
            image = self.transform(image)
        
        result["image"] = image

        # Tokenize text with padding
        text = sample.get("text")
        if text is None:
            return None

        result["text"] = text
        
        if self.use_chi and self.chi_prompt:
            # CHI tokenization: prepend system prompt and select specific tokens
            # Reference: Sana/train_scripts/train.py Lines 362-382
            full_prompt = self.chi_prompt + text
            # magic number 2: [bos], [_]
            max_length_all = self.num_chi_tokens + self.max_text_length - 2
            
            tokens = self.tokenizer(
                full_prompt,
                padding="max_length",
                max_length=max_length_all,
                truncation=True,
                return_tensors="pt",
            )
            
            # Select: first BOS token + last (max_text_length - 1) tokens
            # This keeps the BOS and the user prompt tokens, discarding most of chi_prompt
            select_index = [0] + list(range(-self.max_text_length + 1, 0))
            result["input_ids"] = tokens.input_ids[:, select_index].squeeze(0)  # [L]
            result["attention_mask"] = tokens.attention_mask[:, select_index].squeeze(0)  # [L]
        else:
            # Standard tokenization without CHI
            tokens = self.tokenizer(
                text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = tokens.input_ids.squeeze(0)  # [L]
            result["attention_mask"] = tokens.attention_mask.squeeze(0)  # [L]
        
        return result

    def process(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample.
        
        Extracts image-text pair from sample and processes it.
        
        Args:
            sample: Raw sample dict from parquet
        
        Returns:
            Processed sample dict or None if processing fails
        """
        pair = self.extract_image_text(sample)
        if pair:
            images = json.loads(sample.get("images", '{}'))
            image = pair["image"]
            if image in images:
                pair["image"] = images[image]
            
            metadata = json.loads(sample.get("metadata", '{}'))
            images_info = metadata.get("images", {})
            if image in images_info:
                pair["height"] = images_info[image]["height"]
                pair["width"] = images_info[image]["width"]
            return self._process_pair(pair)
        return None

    def collate_fn(
        self,
        batch: List[Dict[str, torch.Tensor]],
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
        result = {}
        # Standard mode: all images have same size
        if "image" in batch[0]:
            result["image"] = torch.stack([s["image"] for s in batch])
        # Collate text tensors
        for key in ["input_ids", "attention_mask"]:
            if key in batch[0]:
                result[key] = torch.stack([s[key] for s in batch])
        
        return result

class Token2ImageDataset(DistributedDataset):
    """Dataset for visual token-to-image pairs.
    
    This dataset loads image dataset and processes them for diffusion training.
    It supports:
    - Online image loading and transformation
    - Multi-resolution support
    
    Args:
        sources: Path(s) to parquet files or directory
        image_size: Target image size (int or tuple)
        tokenizer_path: Path to tokenizer
        max_condition_length: Maximum condition sequence length
        **kwargs: Additional args passed to DistributedDataset
    """
    
    def __init__(
        self,
        sources: Union[List[str], str],
        image_size: Union[int, Tuple[int, int]] = 1024,
        processor_path: Optional[Any] = None,
        max_condition_length: int = 384,
        center_crop: bool = True,
        multi_scale: bool = False,
        **kwargs,
    ):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.processor_path = processor_path
        self.processor = AutoProcessor.from_pretrained(
            self.processor_path, trust_remote_code=True)
        self.max_condition_length = max_condition_length
        self.center_crop = center_crop
        
        # Multi-scale training support
        self.multi_scale = multi_scale
        
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
                T.Resize(target_size, interpolation=T.InterpolationMode.LANCZOS),  # 短边缩放到 target_size
                T.CenterCrop(target_size),  # 裁剪成正方形
            ])
        
        transform_list.extend([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        
        return T.Compose(transform_list)
    
    def _build_multiscale_transform(self, target_size: Tuple[int, int]) -> Callable:
        """Build transform for a specific target size in multi-scale training.
        
        Args:
            target_size: Target (height, width) for this aspect ratio
            
        Returns:
            Transform pipeline for this size
        """
        transform_list = [
            T.Resize(target_size, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
        return T.Compose(transform_list)
    
    def get_content(self,
                    sample: Dict[str, Any],
                    key: str) -> List[Dict[str, Any]]:
        """Get content from sample.
        
        Args:
            sample: Sample dict
            key: Key to get content from
            
        Returns:
            Parsed JSON content or empty list if parsing fails
        """
        content = sample.get(key, "[]")
        try:
            content = json.loads(content)
            return content
        except (json.JSONDecodeError, TypeError):
            return []

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Validate messages format.
        
        Messages must be single-turn: exactly 2 non-system messages (1 user + 1 assistant).
        System messages are allowed and will be skipped during validation and processing.
        - User message content: str or list with exactly 1 text block
        - Assistant message content: list with exactly 1 image/image_gen block
        
        Args:
            messages: List of message dicts
            
        Raises:
            ValueError: If messages format is invalid
        """
        # Filter out system messages for validation
        non_system_messages = [m for m in messages if m.get("role") != "system"]
        
        if len(non_system_messages) != 2:
            raise ValueError(
                f"Messages must have exactly 2 non-system messages (1 user + 1 assistant), "
                f"got {len(non_system_messages)}"
            )
        
        user_msg = None
        assistant_msg = None
        
        for msg in non_system_messages:
            role = msg.get("role")
            if role == "user":
                user_msg = msg
            elif role == "assistant":
                assistant_msg = msg
        
        if user_msg is None:
            raise ValueError("Messages must contain exactly 1 user message")
        if assistant_msg is None:
            raise ValueError("Messages must contain exactly 1 assistant message")
        
        # Validate user message content
        user_content = user_msg.get("content")
        if isinstance(user_content, list):
            text_blocks = [b for b in user_content if b.get("type") == "text"]
            if len(text_blocks) != 1:
                raise ValueError(
                    f"User message must contain exactly 1 text block, "
                    f"got {len(text_blocks)}"
                )
        elif not isinstance(user_content, str):
            raise ValueError(
                f"User message content must be str or list, got {type(user_content)}"
            )
        
        # Validate assistant message content
        assistant_content = assistant_msg.get("content")
        if not isinstance(assistant_content, list):
            raise ValueError(
                f"Assistant message content must be list, got {type(assistant_content)}"
            )
        
        image_blocks = [
            b for b in assistant_content 
            if b.get("type") in ("image", "image_gen")
        ]
        if len(image_blocks) != 1:
            raise ValueError(
                f"Assistant message must contain exactly 1 image block, "
                f"got {len(image_blocks)}"
            )

    def _validate_segments(self, segments: List[Dict[str, Any]]) -> None:
        """Validate segments format.
        
        Segments must have exactly 2 items:
        - First segment: type="text"
        - Second segment: type="image"
        
        Args:
            segments: List of segment dicts
            
        Raises:
            ValueError: If segments format is invalid
        """
        if len(segments) != 2:
            raise ValueError(
                f"Segments must have exactly 2 items, got {len(segments)}"
            )
        
        if segments[0].get("type") != "text":
            raise ValueError(
                f"First segment must be type='text', "
                f"got type='{segments[0].get('type')}'"
            )
        
        if segments[1].get("type") != "image":
            raise ValueError(
                f"Second segment must be type='image', "
                f"got type='{segments[1].get('type')}'"
            )

    def extract_image_text(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract image and text from sample.
        
        Supports multiple formats:
        - Direct image/text fields
        - Messages format (chat-style, single-turn only)
        - Segments format (exactly 2 segments: text + image)
        - Multiple captions with random selection
        
        Args:
            sample: Raw sample dict
            
        Returns:
            Dict with 'image' and 'text' keys
            
        Raises:
            ValueError: If messages or segments format is invalid
        """
        image = sample.get("image", None)
        text = sample.get("text", None)
        source = sample.get("source", None)
        
        if image and text:
            return {
                "image": image,
                "text": text,
                "source": source
            }
        
        messages = self.get_content(sample, "messages")
        segments = self.get_content(sample, "segments")
        
        if messages:
            # Validate messages format
            self._validate_messages(messages)
            
            for turn in messages:
                if turn["role"] == "user":
                    content = turn["content"]
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        # Already validated: exactly 1 text block
                        for block in content:
                            if block["type"] == "text":
                                text = block["text"]
                                break
                elif turn["role"] == "assistant":
                    content = turn["content"]
                    # Already validated: exactly 1 image block
                    for block in content:
                        # TODO: 暂时兼容错误的格式，后面记得修复
                        # Gen_BLIP3o-Pretrain-Long-Caption/0.0.1
                        # type=image_gen 
                        if block["type"] == "image" or block["type"] == "image_gen":
                            image = block["image"]
                            break

        if segments:
            # Validate segments format
            self._validate_segments(segments)
            
            # First segment is text, second is image (validated)
            text = segments[0]["text"]
            image = segments[1]["image"]
        
        if image is None or text is None:
            return None

        return {"image": image, "text": text, "source": source}


    def _process_pair(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single image-text pair.
        
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

        target_h, target_w = self.image_size
        if self.multi_scale:
            target_h, target_w = sample["target_height"], sample["target_width"]

        if not self.multi_scale and self.center_crop:
            assert target_h == target_w, "center_crop should only paired with square image."
            image = T.Compose(
                [
                    T.Resize(target_h, interpolation=T.InterpolationMode.LANCZOS),  # first resize to target_size
                    T.CenterCrop(target_h),  # 裁剪成正方形
                ]
            )(image)
        else:
            image = T.Resize((target_h, target_w), interpolation=T.InterpolationMode.LANCZOS)(image)
        
        image_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )(image)

        assert image.size == (target_w, target_h)

        fake_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 4 * 28 * 28,
                        "max_pixels": self.max_condition_length * 28 * 28
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            fake_message, 
            tokenize=False
        )

        image_inputs, _, _ = process_vision_info(fake_message)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        result["image"] = image_tensor
        result["pixel_values"] = inputs["pixel_values"]
        result["image_grid_thw"] = inputs["image_grid_thw"]
        
        return result

    def is_valid_scores(self, path, source):
        white_tables = [
            "Ketu_Generation",
            "Gen_text_image",
            "Gen_BLIP3o-60k",
            "Gen_qwig_trans",
            "TextAtlas5M",
        ]
        if any([name in source for name in white_tables]):
            return True
        try:
            infos = json.load(open(path + ".image.json", "r"))
        except:
            return True

        if "aesthetic" in infos:
            if infos["aesthetic"] <= 3.75:
                return False
        if "watermark" in infos:
            if infos["watermark"] >= 0.8:
                return False
        if "nsfw" in infos:
            if infos["nsfw"] > 0.7:
                return False

        if "iqa" in infos:
            if infos["iqa"].get("Good_image", 1.0) < 0.15:
                return False
            if infos["iqa"].get("Sharp_image", 1.0) < 0.1:
                return False
    
            if infos["iqa"].get("sharp_edges", 1.0) < 0.1:
                return False
    
            if infos["iqa"].get("Noise-free_image", 1.0) < 0.1:
                return False

        if "over_exposure" in infos:
            if infos["over_exposure"] > 0.95:
                return False

        if "under_exposure" in infos:
            if infos["under_exposure"] > 0.1:
                return False

        if "rms" in infos:
            if infos["rms"] > 100 or infos["rms"] < 5:
                return False
        return True


    def process(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample.
        
        Extracts image-text pair from sample and processes it.
        
        Args:
            sample: Raw sample dict from parquet
        
        Returns:
            Processed sample dict or None if processing fails
        """
        pair = self.extract_image_text(sample)
        if pair:
            images = json.loads(sample.get("images", '{}'))
            image = pair["image"]
            if image in images:
                pair["image"] = images[image]
            if self.is_valid_scores(pair["image"], sample["source"]) is False:
                return None

            metadata = json.loads(sample.get("metadata", '{}'))
            images_info = metadata.get("images_info", {})
            image_info = images_info.get(image, {})
            height = image_info.get("height", None)
            width = image_info.get("width", None)
            if height is not None and width is not None:
                pair["height"] = height
                pair["width"] = width
            else:
                image = load_image(pair["image"])
                if image is None:
                    return None

                pair["height"] = image.height
                pair["width"] = image.width
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
        result["image"] = torch.stack([s["image"] for s in batch])

        return result

class MultiScaleDatasetWrapper(IterableDataset):
    """Multi-scale dataset wrapper with global buckets.
    
    Wraps a X2ImageDataset and groups samples by (resolution, aspect_ratio) into 
    global buckets. When any bucket has enough samples, yields a batch.
    
    Args:
        dataset: The X2ImageDataset to wrap (must have _build_multiscale_transform).
        config: ResolutionBudgetConfig defining resolutions and batch sizes.
        drop_last: Whether to drop incomplete batches at the end.
        max_bucket_size: Maximum samples per bucket to avoid memory overflow.
        
    Example:
        >>> config = ResolutionBudgetConfig(
        ...     budgets=[
        ...         ResolutionBudget(512, batch_size=32),
        ...         ResolutionBudget(1024, batch_size=8),
        ...     ],
        ... )
        >>> wrapper = MultiScaleDatasetWrapper(dataset, config)
        >>> dataloader = DataLoader(wrapper, batch_size=1, collate_fn=...)
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        config: ResolutionBudgetConfig,
        drop_last: bool = False,
        max_bucket_size: int = 10000
    ):
        self.dataset = dataset
        self.config = config
        self.drop_last = drop_last
        self.max_bucket_size = max_bucket_size
        
        # Batch sizes per resolution
        self._batch_sizes = {b.size: b.batch_size for b in config.budgets}
        
        # Pre-compute aspect ratio dicts for each resolution
        self._aspect_ratios = {
            b.size: get_aspect_ratio_dict(b.size) for b in config.budgets
        }
        
        # Supported resolution levels
        self._resolutions = set(b.size for b in config.budgets)
        
        # Log config
        logger.info(
            f"MultiScaleDatasetWrapper initialized with {len(config.budgets)} resolutions, "
            f"drop_last={drop_last}"
        )
        for b in config.budgets:
            logger.info(f"  {b.size}px: batch_size={b.batch_size}")

    def __iter__(self) -> Iterator[List[Dict]]:
        """Iterate and yield batches grouped by (resolution, aspect_ratio).
        
        Algorithm:
        1. Read sample from dataset
        2. Add sample to bucket based on its natural resolution/aspect_ratio
        3. Check if this bucket is ready (has enough samples)
        4. If yes, yield batch
        5. Repeat until dataset exhausted
        """
        # Global buckets: (resolution, aspect_ratio) -> [samples]
        buckets: Dict[int, Dict[str, List[Dict]]] = {}
        source_count = collections.Counter()
        batch_count = 0

        # Initialize buckets for each resolution
        for res in self._resolutions:
            buckets[res] = {}
            for aspect_ratio in self._aspect_ratios[res]:
                buckets[res][aspect_ratio] = []

        for sample in self.dataset:
            if sample is None:
                continue
            if not ("height" in sample and "width" in sample):
                continue
            orig_h, orig_w = sample["height"], sample["width"]

            # Determine resolution level based on sample's natural dimensions
            res = get_resolution_level(orig_h, orig_w)

            if res not in self._resolutions:
                continue
            
            # Find closest aspect ratio for this resolution
            aspect_ratio = get_closest_ratio(orig_h, orig_w, self._aspect_ratios[res])

            # Add sample to bucket
            buckets[res][aspect_ratio].append(sample)
            
            # Limit bucket size to avoid memory overflow
            if len(buckets[res][aspect_ratio]) > self.max_bucket_size:
                buckets[res][aspect_ratio] = buckets[res][aspect_ratio][-self.max_bucket_size:]

            # Check all buckets for any that are ready
            for check_res in self._resolutions:
                batch_size = self._batch_sizes[check_res]
                for check_ratio, bucket in buckets[check_res].items():
                    if len(bucket) >= batch_size:
                        # Extract batch
                        batch = bucket[:batch_size]
                        buckets[check_res][check_ratio] = bucket[batch_size:]
                        
                        # Set target dimensions for transform
                        tgt_h, tgt_w = self._aspect_ratios[check_res][check_ratio]
                        for s in batch:
                            s["target_height"] = tgt_h
                            s["target_width"] = tgt_w
                        
                        # Update stats
                        source_count.update([s.get("source", "unknown") for s in batch])
                        batch_count += 1
                        
                        if batch_count % 10 == 0:
                            print(
                                f"Batch: {batch_count}, "
                                f"Resolution: {check_res}, "
                                f"Target size: ({tgt_h}, {tgt_w}), "
                                f"Aspect ratio: {check_ratio}, "
                                f"Batch size: {len(batch)}, "
                                f"Source count: {source_count}"
                            )
                        
                        yield batch


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

        # Handle multi-scale: check if target dimensions are provided (from MultiScaleDatasetWrapper)
        # If not provided, use default transform
        if self.multi_scale and "target_height" in sample and "target_width" in sample:
            target_h, target_w = sample["target_height"], sample["target_width"]
            target_image = self._build_multiscale_transform((target_h, target_w))(image)
        else:
            # Default transform for fixed resolution or when target dimensions not available
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
                    x["max_pixels"] = self.max_condition_length * \
                        (self.processor.image_processor.patch_size * self.processor.image_processor.merge_size) ** 2

            # 这里是把所有'image'字段替换成路径, recursive_traverse是为了满足不同的格式
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


class GenEvalInferenceDataset(Chat2ImageDataset):
    """
    gen_eval_csv_path的内容如下:
    index   tag     include_class   include_count   include_color   include_position        exclude_class   exclude_count   question
    1       single_object   bench   1                                       a photo of a bench
    2       single_object   cow     1                                       a photo of a cow

    a resulted sample is like 
    {'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {'role': 'user', 'content': [{'type': 'text', 'text': 'a photo of a bench'}]}], 
        'metadata': {'index': 1, 'tag': 'single_object', 'include_class': 'bench', 
        'include_count': '1', 'include_color': None, 'include_position': None, 'exclude_class': None, 
        'exclude_count': None, 'question': 'a photo of a bench'}}
    """
    def __init__(self, 
                 processor_path,
                 gen_eval_csv_path="/llm_reco/lingzhixin/recovlm_data/generation_data/GenEval.tsv", 
                 template='{}', 
                 systemp_prompt="You are a helpful assistant.",
                 infer_repeats=4,
                 **kwargs
                 ):
        self.gen_eval_csv_path = gen_eval_csv_path
        self.packing = False
        self.template = template
        self.system_prompt = systemp_prompt
        self.all_data = self._load_all_data()
        self.processor_path = processor_path
        self.processor = AutoProcessor.from_pretrained(
            self.processor_path, trust_remote_code=True)
        self.infer_repeats = infer_repeats
    
    def _load_all_data(self):
        """
        _load_all_data 的 Docstring

        return a list of dict:
            the dict keeps detailed information of each sample
        """
        import csv
        
        all_data = []
        
        # 打开TSV文件并读取内容
        with open(self.gen_eval_csv_path, 'r', encoding='utf-8') as f:
            # 使用csv.reader，设置分隔符为制表符
            reader = csv.DictReader(f, delimiter='\t')
            
            # 遍历每一行，将其转换为字典并添加到all_data列表中
            for row in reader:
                # 转换数据类型
                sample_dict = {
                    'index': int(row['index']),
                    'tag': row['tag'],
                    'include_class': row['include_class'],
                    'include_count': row['include_count'] if row['include_count'] else None,
                    'include_color': row['include_color'] or None,
                    'include_position': row['include_position'] or None,
                    'exclude_class': row['exclude_class'] or None,
                    'exclude_count': row['exclude_count'] if row['exclude_count'] else None,
                    'question': row['question']
                }
                all_data.append(sample_dict)
        from muse.training.parallel import get_data_parallel_rank, get_data_parallel_world_size
        from muse.data.datasets.base import get_worker_info
        worker_id, num_workers = get_worker_info()
        rank, world_size = get_data_parallel_rank(), get_data_parallel_world_size()
        all_data = all_data[rank::world_size]
        all_data = all_data[worker_id::num_workers]
        return all_data

    def extract_image_text(self, sample):
        return sample

    def process(self, sample):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": self.template.format(sample["question"])},
        ]})
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"],
            "messages": messages,
            "metadata": sample
        }

    def __iter__(self):
        """Iterate through the dataset, processing samples and handling epochs."""
        import copy
        for sample in self.all_data:
            sample = self.process(sample)
            for _ in range(self.infer_repeats):
                yield copy.deepcopy(sample)
