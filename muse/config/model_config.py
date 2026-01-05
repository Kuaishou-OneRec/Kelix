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

from typing import Optional, Literal, List, Tuple
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
    eos_token_id: Optional[int] = Field(
        default=151645,
        description="End-of-sequence token ID"
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
    """Configuration for KeyeImageTokenizer visual tokenizer."""

    model_class: str = Field(
        default="KeyeImageTokenizer",
        description="Model registry name for KeyeImageTokenizer",
    )
    vision_config: KeyeVisionConfig = Field(
        default_factory=KeyeVisionConfig,
        description="Vision encoder configuration",
    )
    codebook_size: int = Field(default=8192, description="Codebook size for vector quantization")
    embedding_dim: int = Field(default=128, description="Embedding dimension after quantization")
    init_embedding_dim: int = Field(default=4096, description="Initial codebook embedding dimension")
    llm_hidden_size: int = Field(default=4096, description="Hidden size for LLM alignment")
    n_q_tokens: int = Field(default=8, description="Number of quantized tokens per position")
    split_dim: bool = Field(default=False, description="Whether to split codebook by dimension")
    split_voc: int = Field(default=1, description="Number of vocabulary splits")
    add_voc_reducer: bool = Field(default=False, description="Whether to use vocabulary reducer")
    vq_sampling_mode: str = Field(default="argmin", description="VQ sampling mode: argmin or softmax")
    vq_temperature: float = Field(default=1.0, description="Temperature for softmax sampling")
    vq_temperature_decay: float = Field(default=0.999, description="Temperature decay rate")
    vq_min_temperature: float = Field(default=0.1, description="Minimum temperature")
    pre_llm_align: bool = Field(default=False, description="Whether to apply linear alignment before LLM projection")
    output_dim: int = Field(default=1024, description="Output embedding dimension")
    fusion_type: Literal["mean", "sum"] = Field(default="sum", description="Token fusion method: mean or sum")

class SanaConfig(ModelConfig):
    """Configuration for Sana DiT model architecture.
    
    Sana is a diffusion transformer for text-to-image generation using
    Flow Matching with linear attention support.
    
    Reference: https://github.com/NVlabs/Sana
    """
    
    # Image/Latent dimensions
    input_size: int = Field(
        default=32,
        description="Input latent size (e.g., 32 for 1024px with 32x downsample)"
    )
    patch_size: int = Field(
        default=1,
        description="Patch size for patch embedding"
    )
    in_channels: int = Field(
        default=32,
        description="Number of input channels (VAE latent channels)"
    )
    
    # Architecture dimensions
    hidden_size: int = Field(
        default=2240,
        description="Hidden dimension size"
    )
    depth: int = Field(
        default=20,
        description="Number of transformer blocks"
    )
    num_heads: int = Field(
        default=20,
        description="Number of attention heads"
    )
    mlp_ratio: float = Field(
        default=2.5,
        description="MLP hidden dimension ratio"
    )
    
    # Text encoder configuration
    caption_channels: int = Field(
        default=2304,
        description="Text embedding dimension from text encoder"
    )
    model_max_length: int = Field(
        default=300,
        description="Maximum text sequence length"
    )
    
    # Attention configuration
    attn_type: Literal["flash", "linear"] = Field(
        default="linear",
        description="Self-attention type: 'flash' for flash attention, 'linear' for LiteLA"
    )
    cross_attn_type: Literal["flash", "linear"] = Field(
        default="flash",
        description="Cross-attention type"
    )
    linear_head_dim: int = Field(
        default=32,
        description="Head dimension for linear attention"
    )
    qk_norm: bool = Field(
        default=False,
        description="Whether to use QK normalization in self-attention"
    )
    cross_norm: bool = Field(
        default=False,
        description="Whether to use QK normalization in cross-attention"
    )
    use_cross_attn_rope: bool = Field(
        default=False,
        description="Whether to apply 2D RoPE to query in cross-attention"
    )
    use_position_scale: bool = Field(
        default=False,
        description="Whether to use position scale in cross-attention"
    )
    cross_attn_x_norm: bool = Field(
        default=False,
        description="Whether to apply normalization to x before cross-attention"
    )
    
    # FFN configuration
    ffn_type: Literal["mlp", "glumbconv"] = Field(
        default="glumbconv",
        description="FFN type: 'mlp' for standard MLP, 'glumbconv' for GLU MBConv"
    )
    mlp_acts: Tuple[str, str, Optional[str]] = Field(
        default=("silu", "silu", None),
        description="Activation functions for GLUMBConv"
    )
    
    # Output configuration
    pred_sigma: bool = Field(
        default=False,
        description="Whether to predict sigma (variance)"
    )
    learn_sigma: bool = Field(
        default=False,
        description="Whether to learn sigma"
    )
    
    # Position embedding
    use_pe: bool = Field(
        default=False,
        description="Whether to use positional embedding"
    )
    pe_interpolation: float = Field(
        default=1.0,
        description="Position embedding interpolation factor"
    )
    
    # Normalization
    y_norm: bool = Field(
        default=True,
        description="Whether to normalize text embeddings"
    )
    y_norm_scale_factor: float = Field(
        default=0.01,
        description="Scale factor for y normalization"
    )
    norm_eps: float = Field(
        default=1e-5,
        description="Normalization epsilon"
    )
    
    # Training
    class_dropout_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Classifier-free guidance dropout probability"
    )
    drop_path: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Drop path rate for stochastic depth"
    )
    
    # VAE configuration
    vae_type: str = Field(
        default="AutoencoderDC",
        description="VAE model type"
    )
    vae_pretrained: str = Field(
        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        description="Pretrained VAE model path"
    )
    vae_downsample_rate: int = Field(
        default=32,
        description="VAE spatial downsample rate"
    )
    
    # Text encoder configuration
    text_encoder_name: str = Field(
        default="google/gemma-2-2b-it",
        description="Text encoder model name"
    )

    y_embedding_init_method: str = Field(
        default="randn",
        description="Method for initializing y_embeddings"
    )

    use_connector: bool = Field(
        default=False,
        description="Whether to use diffusion connector. If False, there will be only y_embedder with the two linear layers randomly initialized."
    )



class UnifiedQwen3Config(Qwen3Config):
    """Configuration for UnifiedQwen3 model architecture.
    
    This configuration extends Qwen3Config with additional fields for unified autoregressive vision-language models.
    """
    
    # Pre-embedding configuration for vision tokens
    pre_embedding_size: Optional[int] = Field(
        default=None,
        description="Size of the pre-embedding layer for vision tokens. "
                    "If None, uses direct embedding lookup."
    )
    pre_embedding_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in the pre-embedding vocabulary. "
                    "Only used when pre_embedding_size is not None."
    )

    # Token IDs for special tokens
    image_token_id: Optional[int] = Field(
        default=None,
        description="Token ID used to represent image placeholders in the text sequence."
    )
    pad_token_id: Optional[int] = Field(
        default=None,
        description="Token ID used for padding."
    )
    q_eos_token: Optional[int] = Field(
        default=None,
        description="Token ID used to represent quantization end token."
    )

    vision_start_token_id: Optional[int] = Field(
        default=151652,
        description="Token ID used to represent the start of an image token sequence."
    )

    vision_end_token_id: Optional[int] = Field(
        default=151653,
        description="Token ID used to represent the end of an image token sequence."
    )

    output_last_hidden_states_only: bool = Field(
        default=False,
        description="Whether to output only the last hidden states of the model."
    )

    token_decoder_with_teacher_forcing: bool = Field(
        default=True,
        description="Whether to use teacher forcing during training. Disable it during generation."
    )

    # # Tokenizer configuration
    codebook_size: int = Field(default=8192, description="码本大小")
    n_q_tokens: int = Field(default=8, description="每个位置量化token数量")

    @model_validator(mode="after")
    def validate_pre_embedding_fields(cls, values: "UnifiedQwen3Config") -> "UnifiedQwen3Config":
        """Validate that pre_embedding_tokens is provided when pre_embedding_size is set."""
        if values.pre_embedding_size is not None and values.pre_embedding_tokens is None:
            raise ValueError(
                "pre_embedding_tokens must be provided when pre_embedding_size is set."
            )
        return values


class UnifiedTokenDecoderConfig(ModelConfig):
    """Configuration for UnifiedTokenDecoder model architecture.
    
    This configuration defines the parameters for the UnifiedTokenDecoder,
    which is a transformer-based decoder for autoregressive token generation.
    """
    
    # Model identification
    model_class: str = Field(
        default="UnifiedTokenDecoder",
        description="Model class name (e.g., 'UnifiedTokenDecoder')"
    )
    
    # Core model dimensions
    vocab_size: int = Field(
        default=8192,
        description="Vocabulary size for the token decoder"
    )
    max_pos_length: int = Field(
        default=65537,
        description="Maximum sequence length"
    )
    max_length: int = Field(
        default=9,
        description="Maximum sequence length"
    )
    d_model: int = Field(
        default=512,
        description="Hidden dimension size"
    )
    eos_token: int = Field(
        default=151681,
        description="End-of-sequence token ID"
    )
    
    # Transformer architecture
    nhead: int = Field(
        default=4,
        description="Number of attention heads"
    )
    num_layers: int = Field(
        default=1,
        description="Number of transformer layers"
    )
    dim_feedforward: int = Field(
        default=1024,
        description="Dimension of the feedforward network"
    )
    
    # Additional configuration
    use_gradient_checkpointing: bool = Field(
        default=True,
        description="Whether to use gradient checkpointing"
    )
    input_dim: Optional[int] = Field(
        default=None,
        description="Input dimension (used when reduce=True)"
    )
    reduce: bool = Field(
        default=True,
        description="Whether to apply dimensionality reduction"
    )
    attention_function: Literal["eager", "flash_attention_2"] = Field(
        default="eager",
        description="Attention implementation to use"
    )


class KeyeARConfig(ModelConfig):
    """Configuration for KeyeAR model architecture.
    
    This configuration defines the parameters for the KeyeAR model,
    which is an autoregressive vision-language model based on Qwen3 architecture.
    """
    
    # Model identification
    model_class: str = Field(
        default="KeyeARModel",
        description="Model class name (e.g., 'KeyeARModel')"
    )
    
    # Core model configurations
    qwen_config: UnifiedQwen3Config = Field(
        default_factory=UnifiedQwen3Config,
        description="Configuration for the Qwen3 model component"
    )
    tokenizer_config: KeyeTokenizerConfig = Field(
        default_factory=KeyeTokenizerConfig,
        description="Configuration for the visual tokenizer component"
    )
    token_decoder_config: UnifiedTokenDecoderConfig = Field(
        default_factory=UnifiedTokenDecoderConfig,
        description="Configuration for the token decoder component"
    )


