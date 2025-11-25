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

