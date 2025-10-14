# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model configuration classes."""

from typing import Optional, Literal
from pydantic import Field, field_validator

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
    hidden_size: int = Field(
        default=4096,
        description="Hidden dimension size"
    )
    num_hidden_layers: int = Field(
        default=32,
        description="Number of transformer layers"
    )

    # Attention configuration
    num_attention_heads: int = Field(
        default=32,
        description="Number of attention heads"
    )
    num_key_value_heads: int = Field(
        default=32,
        description="Number of key-value heads for GQA/MQA"
    )
    head_dim: int = Field(
        default=128,
        description="Dimension of each attention head"
    )
    
    # Feed-forward configuration
    intermediate_size: int = Field(
        default=11008,
        description="Intermediate size in FFN"
    )
    
    # Position embeddings
    max_position_embeddings: int = Field(
        default=32768,
        description="Maximum sequence length"
    )
    rope_theta: float = Field(
        default=10000.0,
        description="RoPE theta parameter"
    )
    
    # Attention settings
    attention_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Attention dropout probability"
    )
    attention_function: Literal["eager", "flash_attention2"] = Field(
        default="eager",
        description="Attention implementation to use"
    )
    
    # Normalization
    rms_norm_eps: float = Field(
        default=1e-6,
        description="RMS normalization epsilon"
    )
    
    @field_validator("num_attention_heads")
    @classmethod
    def validate_num_heads(cls, v, info):
        """Validate that num_attention_heads is divisible by num_key_value_heads."""
        if "num_key_value_heads" in info.data:
            num_kv_heads = info.data["num_key_value_heads"]
            if v % num_kv_heads != 0:
                raise ValueError(
                    f"num_attention_heads ({v}) must be divisible by "
                    f"num_key_value_heads ({num_kv_heads})"
                )
        return v
    
    @field_validator("head_dim")
    @classmethod
    def validate_head_dim(cls, v, info):
        """Validate that hidden_size equals num_attention_heads * head_dim."""
        if "hidden_size" in info.data and "num_attention_heads" in info.data:
            hidden_size = info.data["hidden_size"]
            num_heads = info.data["num_attention_heads"]
            expected_hidden_size = num_heads * v
            if hidden_size != expected_hidden_size:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must equal "
                    f"num_attention_heads ({num_heads}) * head_dim ({v}) = {expected_hidden_size}"
                )
        return v

