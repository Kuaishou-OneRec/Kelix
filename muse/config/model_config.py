"""
Model Configuration Classes.

This module defines configuration classes for model architectures, using Pydantic
for validation and type checking. All model configurations inherit from BaseConfig
and follow a structured, type-safe approach to defining model hyperparameters.

The module provides:
- Base ModelConfig class for common model properties
- Qwen3Config for Qwen3 architecture
- Automatic validation of configuration values
- Serialization to/from JSON

Classes:
    ModelConfig: Base configuration for all models
    Qwen3Config: Configuration for Qwen3 transformer models

Example:
    >>> from muse.config.model_config import Qwen3Config
    >>> 
    >>> # Create configuration
    >>> config = Qwen3Config(
    ...     vocab_size=151936,
    ...     embed_dim=4096,
    ...     num_layers=32,
    ...     num_heads=32,
    ...     num_kv_heads=32,
    ...     intermediate_dim=11008
    ... )
    >>> 
    >>> # Save to file
    >>> config.save("model_config.json")
    >>> 
    >>> # Load from file
    >>> loaded_config = Qwen3Config.from_json_file("model_config.json")
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model configuration classes."""

from typing import Optional, Literal
from pydantic import Field, field_validator, model_validator

from muse.config.base import BaseConfig


class ModelConfig(BaseConfig):
    """Base model configuration.
    
    This serves as the base class for all model-specific configurations.
    """
    
    # Model identification
    model_class: str = Field(
        description="Model class name (e.g., 'Qwen3Model')"
    )


class Qwen3Config(ModelConfig):
    """Configuration for Qwen3 model architecture.
    
    This configuration is specific to the Qwen3 model family.
    """
    
    # Architecture dimensions
    vocab_size: int = Field(
        default=151936,
        description="Vocabulary size"
    )
    embed_dim: int = Field(
        default=4096,
        description="Hidden dimension size"
    )
    num_layers: int = Field(
        default=32,
        description="Number of transformer layers"
    )
    tie_word_embeddings: bool = Field(
        default=True,
        description="Whether to tie the word embeddings"
    )
    # Attention configuration
    num_heads: int = Field(
        default=32,
        description="Number of attention heads"
    )
    num_kv_heads: int = Field(
        default=32,
        description="Number of key-value heads for GQA/MQA"
    )
    head_dim: int = Field(
        default=128,
        description="Dimension of each attention head"
    )
    attn_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attention dropout probability"
    )
    attention_function: Literal["eager", "flash_attention_2"] = Field(
        default="eager",
        description="Attention implementation to use"
    )
    q_proj_bias: bool = Field(
        default=False,
        description="Whether to use bias in the q_proj layer"
    )
    k_proj_bias: bool = Field(
        default=False,
        description="Whether to use bias in the k_proj layer"
    )
    v_proj_bias: bool = Field(
        default=False,
        description="Whether to use bias in the v_proj layer"
    )
    
    # Feed-forward configuration
    intermediate_dim: int = Field(
        default=11008,
        description="Intermediate size in FFN"
    )
    
    # Position embeddings
    max_seq_len: int = Field(
        default=32768,
        description="Maximum sequence length"
    )
    rope_base: float = Field(
        default=10000.0,
        description="RoPE theta parameter"
    )
    rope_impl: Literal["llama", "hf"] = Field(
        default="llama",
        description="RoPE implementation style: 'llama' (interleaved cos/sin) or 'hf' (separated cos/sin)"
    )
    
    # Normalization
    norm_eps: float = Field(
        default=1e-6,
        description="RMS normalization epsilon"
    )
    q_norm: bool = Field(
        default=True,
        description="Whether to use normalization in the q_proj layer"
    )
    k_norm: bool = Field(
        default=True,
        description="Whether to use normalization in the k_proj layer"
    )

    @field_validator("num_heads")
    @classmethod
    def validate_num_heads(cls, v, info):
        """Validate that num_heads is divisible by num_kv_heads."""
        if "num_kv_heads" in info.data:
            num_kv_heads = info.data["num_kv_heads"]
            if v % num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({v}) must be divisible by "
                    f"num_kv_heads ({num_kv_heads})"
                )
        return v
    
    @field_validator("head_dim")
    @classmethod
    def validate_head_dim(cls, v, info):
        """Validate that embed_dim equals num_heads * head_dim."""
        if "embed_dim" in info.data and "num_heads" in info.data:
            embed_dim = info.data["embed_dim"]
            num_heads = info.data["num_heads"]
            expected_embed_dim = num_heads * v
            if embed_dim != expected_embed_dim:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must equal "
                    f"num_heads ({num_heads}) * head_dim ({v}) = {expected_embed_dim}"
                )
        return v

    @model_validator(mode="after")
    def validate_head_relationships(cls, values: "Qwen3Config") -> "Qwen3Config":
        """Ensure head-related fields stay consistent after initialization."""
        if values.num_heads % values.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({values.num_heads}) must be divisible by "
                f"num_kv_heads ({values.num_kv_heads})"
            )

        expected_embed_dim = values.num_heads * values.head_dim
        if values.embed_dim != expected_embed_dim:
            raise ValueError(
                f"embed_dim ({values.embed_dim}) must equal "
                f"num_heads ({values.num_heads}) * head_dim ({values.head_dim}) "
                f"= {expected_embed_dim}"
            )
        return values


class SiglipVisionConfig(ModelConfig):
    """Configuration for the SigLIP vision transformer encoder."""

    model_class: str = Field(
        default="SiglipVisionTransformer",
        description="Model class name used for registry lookup.",
    )
    image_size: int = Field(default=384, description="Input image resolution.")
    patch_size: int = Field(default=14, description="Patch size of the stem conv.")
    num_channels: int = Field(default=3, description="Number of input channels.")
    hidden_size: int = Field(default=1152, description="Transformer hidden dimension.")
    num_hidden_layers: int = Field(default=27, description="Number of encoder blocks.")
    num_attention_heads: int = Field(default=16, description="Attention heads.")
    intermediate_size: int = Field(default=4304, description="MLP hidden dimension.")
    max_seq_len: int = Field(
        default=4096,
        description="Maximum sequence length for attention. Typically (image_size/patch_size)^2.",
    )
    layer_norm_eps: float = Field(default=1e-6, description="Layer norm epsilon.")
    attention_dropout: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Attention dropout probability."
    )
    has_learnable_position_embedding: bool = Field(
        default=True,
        description="Use learnable packing position embeddings for vision tokens.",
    )
    use_qk_norm: bool = Field(default=False, description="Apply RMSNorm to Q/K projections.")
    qk_norm_eps: float = Field(default=1e-6, description="Epsilon for Q/K RMSNorm layers.")
    rope_theta: float = Field(default=10000.0, description="RoPE base frequency.")
    attention_function: Literal["eager", "flash_attention_2"] = Field(
        default="flash_attention_2",
        description="Attention backend implementation.",
    )
    output_attentions: bool = Field(
        default=False,
        description="Whether the encoder returns attention probabilities.",
    )
    output_hidden_states: bool = Field(
        default=False,
        description="Whether the encoder returns hidden states from all layers.",
    )


class KeyeVisionConfig(ModelConfig):
    """Configuration for the SigLIP vision transformer encoder."""

    model_class: str = Field(
        default="SiglipVisionTransformer",
        description="Model class name used for registry lookup.",
    )
    image_size: int = Field(default=384, description="Input image resolution.")
    patch_size: int = Field(default=14, description="Patch size of the stem conv.")
    num_channels: int = Field(default=3, description="Number of input channels.")
    hidden_size: int = Field(default=1152, description="Transformer hidden dimension.")
    num_hidden_layers: int = Field(default=27, description="Number of encoder blocks.")
    num_attention_heads: int = Field(default=16, description="Attention heads.")
    intermediate_size: int = Field(default=4304, description="MLP hidden dimension.")
    max_seq_len: int = Field(
        default=4096,
        description="Maximum sequence length for attention. Typically (image_size/patch_size)^2.",
    )
    layer_norm_eps: float = Field(default=1e-6, description="Layer norm epsilon.")
    attention_dropout: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Attention dropout probability."
    )
    has_learnable_position_embedding: bool = Field(
        default=True,
        description="Use learnable packing position embeddings for vision tokens.",
    )
    use_qk_norm: bool = Field(default=False, description="Apply RMSNorm to Q/K projections.")
    qk_norm_eps: float = Field(default=1e-6, description="Epsilon for Q/K RMSNorm layers.")
    rope_theta: float = Field(default=10000.0, description="RoPE base frequency.")
    attention_function: Literal["eager", "flash_attention_2"] = Field(
        default="flash_attention_2",
        description="Attention backend implementation.",
    )
    output_attentions: bool = Field(
        default=False,
        description="Whether the encoder returns attention probabilities.",
    )
    output_hidden_states: bool = Field(
        default=False,
        description="Whether the encoder returns hidden states from all layers.",
    )


    
