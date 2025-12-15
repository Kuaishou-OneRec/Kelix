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


class ImageTextDataset(DistributedDataset):
    """Dataset for image-text pairs with online VAE/text encoding.
    
    This dataset loads image-text pairs and processes them for diffusion training.
    It supports:
    - Online image loading and transformation
    - Optional VAE encoding
    - Optional text encoding
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
        text_encoder: Optional[nn.Module] = None,
        max_text_length: int = 300,
        center_crop: bool = True,
        random_flip: bool = True,
        image_column: str = "image",
        text_column: str = "text",
        latent_column: Optional[str] = None,
        text_embed_column: Optional[str] = None,
        sample_posterior: bool = True,
        **kwargs,
    ):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.max_text_length = max_text_length
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.image_column = image_column
        self.text_column = text_column
        self.latent_column = latent_column
        self.text_embed_column = text_embed_column
        self.sample_posterior = sample_posterior
        
        # Build transforms
        self.transform = self._build_transform()
        
        super().__init__(sources, **kwargs)
    
    def _build_transform(self) -> Callable:
        """Build image transformation pipeline."""
        transform_list = []
        
        if self.center_crop:
            transform_list.append(T.CenterCrop(min(self.image_size)))
        
        transform_list.extend([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
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
    
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space using VAE.
        
        Args:
            image: Image tensor [C, H, W] or [B, C, H, W]
        
        Returns:
            Latent tensor
        """
        if self.vae is None:
            return image
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        device = next(self.vae.parameters()).device
        dtype = next(self.vae.parameters()).dtype
        image = image.to(device=device, dtype=dtype)
        
        with torch.no_grad():
            posterior = self.vae.encode(image)
            if hasattr(posterior, 'latent_dist'):
                posterior = posterior.latent_dist
            if self.sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
            z = z * self.vae.config.scaling_factor
        
        return z.squeeze(0).cpu()
    
    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text to embeddings using text encoder.
        
        Args:
            text: Input text string
        
        Returns:
            Dict with 'text_embeds' and 'attention_mask'
        """
        if self.tokenizer is None or self.text_encoder is None:
            return {"text": text}
        
        device = next(self.text_encoder.parameters()).device
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        
        # Encode
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
            )
            # Get hidden states
            if hasattr(text_outputs, 'last_hidden_state'):
                text_embeds = text_outputs.last_hidden_state
            elif isinstance(text_outputs, tuple):
                text_embeds = text_outputs[0]
            else:
                text_embeds = text_outputs
        
        return {
            "text_embeds": text_embeds.squeeze(0).cpu(),
            "attention_mask": attention_mask.squeeze(0).cpu(),
        }
    
    def process(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample.
        
        Args:
            sample: Raw sample dict from parquet
        
        Returns:
            Processed sample dict or None if processing fails
        """
        result = {}
        
        # Handle pre-encoded latents
        if self.latent_column and self.latent_column in sample:
            latent = sample[self.latent_column]
            if isinstance(latent, (np.ndarray, list)):
                latent = torch.tensor(latent)
            result["latents"] = latent
        else:
            # Load and process image
            image_data = sample.get(self.image_column)
            if image_data is None:
                return None
            
            image = self._load_image(image_data)
            if image is None:
                return None
            
            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Apply random flip
            if self.random_flip and random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Apply transforms
            image = self.transform(image)
            
            # Encode to latent space
            if self.vae is not None:
                latent = self._encode_image(image)
                result["latents"] = latent
            else:
                result["image"] = image
        
        # Handle pre-encoded text embeddings
        if self.text_embed_column and self.text_embed_column in sample:
            text_embed = sample[self.text_embed_column]
            if isinstance(text_embed, (np.ndarray, list)):
                text_embed = torch.tensor(text_embed)
            result["text_embeds"] = text_embed
            
            # Get mask if available
            mask_column = self.text_embed_column.replace("embed", "mask")
            if mask_column in sample:
                mask = sample[mask_column]
                if isinstance(mask, (np.ndarray, list)):
                    mask = torch.tensor(mask)
                result["attention_mask"] = mask
        else:
            # Load and encode text
            text = sample.get(self.text_column, "")
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            text = str(text)
            
            text_result = self._encode_text(text)
            result.update(text_result)
        
        # Add metadata
        result["__file__"] = sample.get("__file__", "")
        result["__index__"] = sample.get("__index__", -1)
        
        return result
    
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
        for key in ["latents", "image", "text_embeds", "attention_mask"]:
            if key in batch[0]:
                result[key] = torch.stack([s[key] for s in batch])
        
        # Collate text strings
        if "text" in batch[0]:
            result["text"] = [s["text"] for s in batch]
        
        return result


class ImageFolderDataset(torch.utils.data.Dataset):
    """Simple dataset for loading images from a folder.
    
    This is useful for inference/evaluation where you just want to
    load images without text pairs.
    """
    
    def __init__(
        self,
        folder: str,
        image_size: Union[int, Tuple[int, int]] = 1024,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        self.folder = folder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        
        # Find all images
        self.image_paths = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, f))
        self.image_paths.sort()
        
        # Build transform
        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "path": path,
        }
