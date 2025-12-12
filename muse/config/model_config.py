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

from typing import Optional, Literal, List
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
    hidden_act: str = Field(
        default="silu",
        description="Activation function for MLP (e.g., silu, gelu, gelu_pytorch_tanh)"
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
    attention_bias: bool = Field(
        default=False,
        description="Whether to use bias terms in attention projections"
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
    rope_theta: float = Field(
        default=10000.0,
        description="RoPE theta parameter (alias for rope_base)"
    )
    rope_scaling: Optional[dict] = Field(
        default=None,
        description="RoPE scaling config (e.g., {'rope_type': 'default', 'mrope_section': [...]})"
    )
    use_sliding_window: bool = Field(
        default=False,
        description="Enable sliding-window attention"
    )
    sliding_window: Optional[int] = Field(
        default=None,
        description="Sliding window size when use_sliding_window is enabled"
    )
    
    # Normalization
    norm_eps: float = Field(
        default=1e-6,
        description="RMS normalization epsilon"
    )
    rms_norm_eps: float = Field(
        default=1e-6,
        description="Alias for RMS normalization epsilon"
    )
    q_norm: bool = Field(
        default=True,
        description="Whether to use normalization in the q_proj layer"
    )
    k_norm: bool = Field(
        default=True,
        description="Whether to use normalization in the k_proj layer"
    )
    
    # Multimodal RoPE (3D RoPE for vision-language models)
    use_multimodal_rope: bool = Field(
        default=True,
        description="Whether to use 3D multimodal RoPE instead of standard 1D RoPE. "
                    "Required for models like Keye-VL that use temporal/height/width position encoding."
    )
    mrope_section: Optional[List[int]] = Field(
        default=None,
        description="Multimodal RoPE section sizes for [temporal, height, width]. "
                    "E.g., [16, 24, 24] means 16 dims for temporal, 24 for height, 24 for width. "
                    "Only used when use_multimodal_rope=True. If None, will try to read from rope_scaling."
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
            # Some checkpoints (e.g., Keye) use head_dim overriding embed_dim/num_heads,
            # allowing q_proj out_dim != embed_dim. Skip strict check here.
        return v

    @model_validator(mode="after")
    def validate_head_relationships(cls, values: "Qwen3Config") -> "Qwen3Config":
        """Ensure head-related fields stay consistent after initialization."""
        if values.num_heads % values.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({values.num_heads}) must be divisible by "
                f"num_kv_heads ({values.num_kv_heads})"
            )

        # Relax embed_dim vs num_heads * head_dim to support checkpoints where head_dim is overridden
        # and q_proj out_dim != embed_dim (e.g., Keye).
        return values

class SiglipVisionConfig(ModelConfig):
    """Configuration for the SigLIP vision transformer encoder."""

    model_class: str = Field(
        default="KeyeVisionModel",
        description="Model class name used for registry lookup.",
    )
    image_size: int = Field(default=384, description="Input image resolution.")
    patch_size: int = Field(default=14, description="Patch size of the stem conv.")
    num_channels: int = Field(default=3, description="Number of input channels.")
    hidden_size: int = Field(default=1152, description="Transformer hidden dimension.")
    num_hidden_layers: int = Field(default=27, description="Number of encoder blocks.")
    num_attention_heads: int = Field(default=16, description="Attention heads.")
    intermediate_size: int = Field(default=4304, description="MLP hidden dimension.")
    hidden_act: str = Field(
        default="silu",
        description="Activation function used in the vision MLP (e.g., gelu_pytorch_tanh, silu, gelu)."
    )
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
        default="KeyeVisionModel",
        description="Model class name used for registry lookup.",
    )
    image_size: int = Field(default=384, description="Input image resolution.")
    patch_size: int = Field(default=14, description="Patch size of the stem conv.")
    num_channels: int = Field(default=3, description="Number of input channels.")
    hidden_size: int = Field(default=1152, description="Transformer hidden dimension.")
    num_hidden_layers: int = Field(default=27, description="Number of encoder blocks.")
    num_attention_heads: int = Field(default=16, description="Attention heads.")
    intermediate_size: int = Field(default=4304, description="MLP hidden dimension.")
    hidden_act: str = Field(
        default="silu",
        description="Activation function used in the vision MLP (e.g., gelu_pytorch_tanh, silu, gelu)."
    )
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


class KeyeTokenizerConfig(ModelConfig):
    """视觉Tokenizer配置，供 KeyeImageTokenizer 使用。"""

    model_class: str = Field(
        default="KeyeImageTokenizer",
        description="模型注册名，对应 KeyeImageTokenizer",
    )
    vision_config: KeyeVisionConfig = Field(
        default_factory=KeyeVisionConfig,
        description="视觉编码器配置，默认与 KeyeVisionConfig 一致",
    )
    codebook_size: int = Field(default=8192, description="码本大小")
    embedding_dim: int = Field(default=128, description="量化后维度")
    init_embedding_dim: int = Field(default=4096, description="初始码本维度")
    llm_hidden_size: int = Field(default=4096, description="对齐到LLM的维度")
    n_q_tokens: int = Field(default=8, description="每个位置量化token数量")
    split_dim: bool = Field(default=False, description="是否按维度切分码本")
    split_voc: int = Field(default=1, description="词表切分数量")
    add_voc_reducer: bool = Field(default=False, description="是否使用voc reducer")
    vq_sampling_mode: str = Field(default="argmin", description="VQ采样方式 argmin/softmax")
    vq_temperature: float = Field(default=1.0, description="softmax温度")
    vq_temperature_decay: float = Field(default=0.999, description="温度衰减")
    vq_min_temperature: float = Field(default=0.1, description="最低温度")
    pre_llm_align: bool = Field(default=False, description="是否先线性对齐到LLM维度")
