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
VAE and Text Encoder Utilities for Sana.

This module provides helper functions to load and use the VAE and text encoder
models for Sana training and inference.

Reference: Sana/diffusion/model/builder.py
"""

from typing import Optional, Tuple, Any
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_vae(
    vae_type: str = "AutoencoderDC",
    vae_pretrained: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Load VAE model.
    
    Args:
        vae_type: Type of VAE to load (currently only "AutoencoderDC" supported)
        vae_pretrained: Pretrained model path or HuggingFace repo
        device: Device to load model on
        dtype: Data type for model
    
    Returns:
        Loaded VAE model
    
    Reference: Sana/diffusion/model/builder.py Lines 23-24
    """
    if vae_type == "AutoencoderDC":
        try:
            from diffusers import AutoencoderDC
        except ImportError:
            raise ImportError(
                "diffusers is required for AutoencoderDC. "
                "Install it with: pip install diffusers"
            )
        
        logger.info(f"Loading AutoencoderDC from {vae_pretrained}")
        vae = AutoencoderDC.from_pretrained(
            vae_pretrained,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported VAE type: {vae_type}")
    
    if device is not None:
        vae = vae.to(device)
    
    vae.eval()
    vae.requires_grad_(False)
    
    return vae


def vae_encode(
    vae: nn.Module,
    images: torch.Tensor,
    sample_posterior: bool = True,
) -> torch.Tensor:
    """Encode images to latent space using VAE.
    
    Args:
        vae: VAE model
        images: Input images tensor [B, C, H, W] in range [-1, 1]
        sample_posterior: Whether to sample from posterior (True) or use mode (False)
    
    Returns:
        Latent tensor [B, C', H', W']
    
    Reference: Sana/train_scripts/train.py Lines 100-110
    """
    with torch.no_grad():
        # Get latent distribution
        posterior = vae.encode(images)
        
        # Handle different output formats
        if hasattr(posterior, 'latent_dist'):
            posterior = posterior.latent_dist
        
        # Sample or get mode
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        # Apply scaling factor
        if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
            z = z * vae.config.scaling_factor
        
    return z


def vae_decode(
    vae: nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Decode latents to images using VAE.
    
    Args:
        vae: VAE model
        latents: Latent tensor [B, C', H', W']
    
    Returns:
        Decoded images tensor [B, C, H, W] in range [-1, 1]
    """
    with torch.no_grad():
        # Apply inverse scaling factor
        if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
            latents = latents / vae.config.scaling_factor
        
        images = vae.decode(latents).sample
    
    return images


def get_text_encoder(
    name: str = "google/gemma-2-2b-it",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, nn.Module]:
    """Load text encoder and tokenizer.
    
    Args:
        name: Model name or path
        device: Device to load model on
        dtype: Data type for model
    
    Returns:
        Tuple of (tokenizer, text_encoder)
    
    Reference: Sana/diffusion/model/builder.py Lines 53-89
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "transformers is required for text encoder. "
            "Install it with: pip install transformers"
        )
    
    logger.info(f"Loading text encoder from {name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
    )
    
    # Get decoder if available (for encoder-decoder models)
    if hasattr(model, 'get_decoder'):
        text_encoder = model.get_decoder()
    else:
        text_encoder = model
    
    if device is not None:
        text_encoder = text_encoder.to(device)
    
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    
    return tokenizer, text_encoder


def encode_text(
    tokenizer: Any,
    text_encoder: nn.Module,
    texts: list,
    max_length: int = 300,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode text to embeddings.
    
    Args:
        tokenizer: Tokenizer instance
        text_encoder: Text encoder model
        texts: List of text strings to encode
        max_length: Maximum sequence length
        device: Device for output tensors
    
    Returns:
        Tuple of (text_embeds, attention_mask)
        - text_embeds: [B, 1, L, D] tensor
        - attention_mask: [B, 1, 1, L] tensor
    
    Reference: Sana/train_scripts/train.py Lines 300-310
    """
    if device is None:
        device = next(text_encoder.parameters()).device
    
    # Tokenize
    tokens = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    
    # Encode
    with torch.no_grad():
        outputs = text_encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        
        # Get hidden states
        if hasattr(outputs, 'last_hidden_state'):
            text_embeds = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            text_embeds = outputs[0]
        else:
            text_embeds = outputs
    
    # Add dimension for cross attention: [B, 1, L, D]
    text_embeds = text_embeds[:, None]
    attention_mask = attention_mask[:, None, None]
    
    return text_embeds, attention_mask
