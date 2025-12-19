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
    ResolutionBudget,
    ResolutionBudgetConfig,
)

from torch.utils.data import IterableDataset

from muse.utils.common import print_rank_0


logger = logging.getLogger(__name__)


# =============================================================================
# Resolution Budget Sampler for Dynamic Multi-Scale Training
# =============================================================================

class ResolutionBudgetSampler:
    """Samples resolution budgets with curriculum weight scheduling.
    
    Weights interpolate linearly from start_weights to end_weights
    based on training progress (current_step / total_steps).
    
    Args:
        config: ResolutionBudgetConfig with budgets and start/end weights
        total_steps: Total training steps for progress calculation
        
    Example:
        >>> config = ResolutionBudgetConfig(
        ...     budgets=[ResolutionBudget(512, 32), ResolutionBudget(1024, 8)],
        ...     start_weights=[0.8, 0.2],
        ...     end_weights=[0.2, 0.8],
        ... )
        >>> sampler = ResolutionBudgetSampler(config, total_steps=100000)
        >>> sampler.set_step(0)       # weights = [0.8, 0.2]
        >>> sampler.set_step(50000)   # weights = [0.5, 0.5]
        >>> sampler.set_step(100000)  # weights = [0.2, 0.8]
    """
    
    def __init__(self, config: ResolutionBudgetConfig, total_steps: int):
        self.config = config
        self.budgets = config.budgets
        self.total_steps = max(1, total_steps)
        self._current_step = 0
        
        # Pre-compute aspect ratio dicts for each resolution
        self.aspect_ratio_dicts = {
            b.size: get_aspect_ratio_dict(b.size) for b in self.budgets
        }
        
        logger.info(
            f"ResolutionBudgetSampler initialized with {len(self.budgets)} resolutions, "
            f"total_steps={total_steps}"
        )
        logger.info(f"  Start weights: {config.start_weights}")
        logger.info(f"  End weights: {config.end_weights}")
    
    def set_step(self, step: int):
        """Update current training step for weight interpolation."""
        self._current_step = step
    
    @property
    def progress(self) -> float:
        """Current training progress in [0, 1]."""
        return min(1.0, self._current_step / self.total_steps)
    
    @property
    def current_weights(self) -> List[float]:
        """Get current interpolated weights."""
        return self.config.get_weights(self.progress)
    
    def sample(self) -> ResolutionBudget:
        """Sample a resolution budget using current interpolated weights."""
        weights = self.current_weights
        return random.choices(self.budgets, weights=weights, k=1)[0]
    
    def get_aspect_ratios(self, size: int) -> Dict[str, Tuple[int, int]]:
        """Get aspect ratio dict for given resolution."""
        return self.aspect_ratio_dicts[size]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current sampling statistics for logging."""
        weights = self.current_weights
        return {
            "step": self._current_step,
            "progress": self.progress,
            "weights": {b.size: w for b, w in zip(self.budgets, weights)},
        }


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
            T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
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
        self.base_image_size = image_size if isinstance(image_size, int) else image_size[0]
        self.processor_path = processor_path
        self.processor = AutoProcessor.from_pretrained(
            self.processor_path, trust_remote_code=True)
        self.max_condition_length = max_condition_length
        self.center_crop = center_crop
        
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
            T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
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
        
        # Store raw image data for dynamic multi-scale re-processing
        result["_raw_image"] = image_data
        
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
            target_image = transform(image)
            
            # Store aspect ratio info for potential use in collation
            result["aspect_ratio"] = closest_ratio
            result["target_size"] = (target_h, target_w)
        else:
            target_h, target_w = self.image_size
            target_image = self.transform(image)
        
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

        result["image"] = target_image
        result["pixel_values"] = inputs["pixel_values"]
        result["image_grid_thw"] = inputs["image_grid_thw"]
        
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

        # Concatenate pixel_values: [s, d] -> [S, d] where S is sum of all s
        result["pixel_values"] = torch.concat([s["pixel_values"] for s in batch], dim=0)
        result["image_grid_thw"] = torch.concat([s["image_grid_thw"] for s in batch], dim=0)
        result["image"] = torch.stack([s["image"] for s in batch])
        
        # Include resolution metadata for logging (all samples in batch have same resolution)
        if "resolution" in batch[0]:
            result["resolution"] = batch[0]["resolution"]
        if "target_size" in batch[0]:
            result["target_size"] = batch[0]["target_size"]
        if "aspect_ratio" in batch[0]:
            result["aspect_ratio"] = batch[0]["aspect_ratio"]

        return result

class MultiScaleDatasetWrapper(IterableDataset):
    """Multi-scale dataset wrapper.
    
    Wraps a X2ImageDataset and groups samples by aspect ratio into buckets.
    Args:
        dataset: The X2ImageDataset to wrap.
    Example:
        >>> dataset = X2ImageDataset(sources=..., multi_scale=True)
        >>> bucket_dataset = MultiScaleDatasetWrapper(
        ...     dataset=dataset,
        ...     batch_size=16,
        ...     aspect_ratios=get_aspect_ratio_dict(1024),
        ... )
        >>> dataloader = DataLoader(
        ...     bucket_dataset,
        ...     batch_size=1,
        ...     collate_fn=lambda x: dataset.collate_fn(x[0]),
        ... )
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int,
        bucket_key: str = "aspect_ratio",
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = dataset.aspect_ratios
        assert self.aspect_ratios is not None, "`aspect_ratios` is None in dataset"
        self.bucket_key = bucket_key
        self.drop_last = drop_last
        
        logger.info(
            f"Text2ImageMultiScaleDataset initialized with batch_size={batch_size}, "
            f"{len(self.aspect_ratios)} aspect ratios, drop_last={drop_last}"
        )
    
    def __iter__(self) -> Iterator[List[Dict]]:
        """Iterate and yield batches grouped by aspect ratio."""
        # Create a bucket for each aspect ratio
        buckets: Dict[str, List] = {
            ratio: [] for ratio in self.aspect_ratios.keys()}
        
        for sample in self.dataset:
            # Get the sample's aspect ratio
            ratio = sample.get(self.bucket_key)
            if ratio is None or ratio not in buckets:
                # Default to square if ratio not found
                ratio = "1.0"
            
            # Add sample to the corresponding bucket
            buckets[ratio].append(sample)
            
            # Yield batch when bucket is full
            if len(buckets[ratio]) >= self.batch_size:
                yield buckets[ratio][:self.batch_size]
                buckets[ratio] = buckets[ratio][self.batch_size:]
        
        # Handle remaining samples after iteration ends
        if not self.drop_last:
            for ratio, bucket in buckets.items():
                # Yield any complete batches remaining
                while len(bucket) >= self.batch_size:
                    yield bucket[:self.batch_size]
                    bucket = bucket[self.batch_size:]
                # Yield incomplete batch if any samples remain
                if bucket:
                    yield bucket


class DynamicMultiScaleDatasetWrapper(IterableDataset):
    """Dynamic multi-scale dataset wrapper with curriculum resolution sampling.
    
    For each micro-step (batch yield):
    1. Sample a resolution budget by current interpolated weight
    2. Use that resolution's aspect ratios and batch size
    3. Fill a bucket until batch_size samples with same aspect ratio
    
    The sampler's weights are updated externally via `set_step()` to implement
    curriculum learning (low-res heavy early, high-res heavy late).
    
    Args:
        dataset: Base Token2ImageDataset
        config: ResolutionBudgetConfig with budgets and weight schedule
        total_steps: Total training steps for weight interpolation
        drop_last: Whether to drop incomplete batches
        
    Example:
        >>> from muse.data.utils import DEFAULT_RESOLUTION_BUDGETS
        >>> config = DEFAULT_RESOLUTION_BUDGETS
        >>> wrapper = DynamicMultiScaleDatasetWrapper(dataset, config, total_steps=100000)
        >>> for step, batch in enumerate(DataLoader(wrapper, batch_size=1)):
        ...     wrapper.set_step(step)  # Update weights based on progress
        ...     print(batch["image"].shape, batch["resolution"])
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        config: ResolutionBudgetConfig,
        total_steps: int,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.config = config
        self.sampler = ResolutionBudgetSampler(config, total_steps)
        self.drop_last = drop_last
        
        # Store reference to dataset's transform builder
        self._build_multiscale_transform = dataset._build_multiscale_transform
        
        logger.info(
            f"DynamicMultiScaleDatasetWrapper initialized with "
            f"{len(config.budgets)} resolution budgets, total_steps={total_steps}"
        )
        for b, sw, ew in zip(config.budgets, config.start_weights, config.end_weights):
            logger.info(f"  {b.size}x{b.size}: batch_size={b.batch_size}, "
                       f"weight {sw:.2f} -> {ew:.2f}")
    
    def set_step(self, step: int):
        """Update current training step for curriculum weight scheduling.
        
        Call this each training step to update the resolution sampling weights.
        """
        self.sampler.set_step(step)
    
    def get_sampler_stats(self) -> Dict[str, Any]:
        """Get current sampler statistics for logging."""
        return self.sampler.get_stats()
    
    def _transform_for_resolution(
        self,
        image: Image.Image,
        resolution: int,
        aspect_ratios: Dict[str, Tuple[int, int]]
    ) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        """Transform image to target resolution with closest aspect ratio.
        
        Args:
            image: PIL Image (already RGB)
            resolution: Target resolution budget
            aspect_ratios: Aspect ratio dict for this resolution
            
        Returns:
            (transformed_tensor, aspect_ratio_key, target_size)
        """
        orig_w, orig_h = image.size
        closest_ratio = get_closest_ratio(orig_h, orig_w, aspect_ratios)
        target_h, target_w = aspect_ratios[closest_ratio]
        
        transform = self._build_multiscale_transform((target_h, target_w))
        tensor = transform(image)
        
        return tensor, closest_ratio, (target_h, target_w)
    
    def __iter__(self) -> Iterator[List[Dict]]:
        """Iterate and yield batches with dynamic resolution.
        
        Algorithm:
        1. Sample a resolution budget for current batch
        2. Create empty buckets for each aspect ratio at that resolution
        3. Pull samples from dataset, transform them, bucket by aspect ratio
        4. When any bucket reaches batch_size, yield it and start new resolution
        5. Leftover samples are re-queued for next iteration
        """
        # Queue of raw samples for re-processing at different resolutions
        sample_queue: List[Dict] = []
        
        # Iterator over base dataset
        base_iter = iter(self.dataset)
        exhausted = False
        
        while True:
            # Sample resolution for this micro-step
            budget = self.sampler.sample()
            aspect_ratios = self.sampler.get_aspect_ratios(budget.size)
            batch_size = budget.batch_size
            
            # Buckets for this resolution: aspect_ratio -> list of processed samples
            buckets: Dict[str, List[Dict]] = {r: [] for r in aspect_ratios.keys()}
            
            batch_ready = False
            
            while not batch_ready:
                # Try to get a sample
                sample = None
                
                # First try queue
                if sample_queue:
                    sample = sample_queue.pop(0)
                elif not exhausted:
                    try:
                        sample = next(base_iter)
                    except StopIteration:
                        exhausted = True
                
                if sample is None:
                    # No more samples available
                    break
                
                # Extract image for processing
                # The sample from dataset already has processed fields,
                # but we need the raw image to re-transform for different resolution
                raw_image = sample.get("_raw_image")
                if raw_image is None:
                    # Sample doesn't have raw image preserved - skip
                    continue
                
                image = load_image(raw_image)
                if image is None:
                    continue
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Transform for current resolution
                tensor, ratio, target_size = self._transform_for_resolution(
                    image, budget.size, aspect_ratios
                )
                
                # Build processed sample
                processed = {
                    "image": tensor,
                    "aspect_ratio": ratio,
                    "target_size": target_size,
                    "resolution": budget.size,
                    "_raw_image": raw_image,  # Keep for potential re-queue
                }
                
                # Copy other fields from original sample
                for key in ["pixel_values", "image_grid_thw"]:
                    if key in sample:
                        processed[key] = sample[key]
                
                # Add to bucket
                buckets[ratio].append(processed)
                
                # Check if bucket is full
                if len(buckets[ratio]) >= batch_size:
                    # Yield the batch
                    batch = buckets[ratio][:batch_size]
                    remaining = buckets[ratio][batch_size:]
                    buckets[ratio] = []
                    
                    # Re-queue remaining and other buckets for next micro-step
                    for r, bucket in buckets.items():
                        sample_queue.extend(bucket)
                    sample_queue.extend(remaining)
                    
                    yield batch
                    batch_ready = True
            
            # If we're here without batch_ready, we've exhausted samples
            if not batch_ready:
                # Handle remaining samples
                if not self.drop_last:
                    for ratio, bucket in buckets.items():
                        if bucket:
                            yield bucket
                break
