"""
Convert HuggingFace Keye-VL checkpoint to Muse KeyeTokenizerEnd2EndImage format.

This script converts a full Keye-VL multimodal model checkpoint (visual tokenizer + LLM)
from HuggingFace format to Muse format.

Usage:
    python convert_checkpoint.py \
        --hf-dir /path/to/hf_checkpoint \
        --output-dir /path/to/output \
        --dtype bfloat16
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import json
import argparse
import torch
import logging

from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig, KeyeTokenizerEnd2EndImageConfig
from muse.models.keye_tokenizer_end2end_image import KeyeTokenizerEnd2EndImage
from muse.training.common import set_default_dtype
from muse.training.checkpoint import load_hf_checkpoint

logger = logging.getLogger(__name__)


def _build_qwen3_config(hf_config: Dict[str, Any]) -> Qwen3Config:
    """Build Muse Qwen3Config from HuggingFace config dictionary.
    
    Args:
        hf_config: Raw HuggingFace config.json content
        
    Returns:
        Qwen3Config instance for the LLM
    """
    # For Keye-VL models, LLM config is at top level (not nested under 'text_config')
    # But we fall back to text_config for compatibility with other model formats
    text_cfg = hf_config.get("text_config", hf_config)
    
    # Handle rope_scaling configuration and extract mrope_section
    rope_scaling = text_cfg.get("rope_scaling", None)
    mrope_section = None
    if rope_scaling is not None:
        mrope_section = rope_scaling.get("mrope_section", None)
    
    qwen_cfg = Qwen3Config(
        model_class="Qwen3Model",
        vocab_size=text_cfg.get("vocab_size", 151936),
        embed_dim=text_cfg.get("hidden_size", 4096),
        num_layers=text_cfg.get("num_hidden_layers", 32),
        num_heads=text_cfg.get("num_attention_heads", 32),
        num_kv_heads=text_cfg.get("num_key_value_heads", text_cfg.get("num_attention_heads", 32)),
        head_dim=text_cfg.get("head_dim", 128),
        intermediate_dim=text_cfg.get("intermediate_size", 11008),
        max_seq_len=text_cfg.get("max_position_embeddings", 32768),
        rope_base=float(text_cfg.get("rope_theta", 1_000_000)),
        rope_theta=float(text_cfg.get("rope_theta", 1_000_000)),
        rope_scaling=rope_scaling,
        norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        hidden_act=text_cfg.get("hidden_act", "silu"),
        tie_word_embeddings=text_cfg.get("tie_word_embeddings", True),
        attention_bias=text_cfg.get("attention_bias", False),
        q_norm=text_cfg.get("use_qk_norm", True),
        k_norm=text_cfg.get("use_qk_norm", True),
        attention_function=hf_config.get("_attn_implementation", "flash_attention_2"),
        use_sliding_window=text_cfg.get("use_sliding_window", False),
        sliding_window=text_cfg.get("sliding_window", None),
        use_multimodal_rope=hf_config.get("use_multimodal_rope", True),
        mrope_section=mrope_section,
    )
    return qwen_cfg


def _build_vision_config(hf_config: Dict[str, Any]) -> KeyeVisionConfig:
    """Build Muse KeyeVisionConfig from HuggingFace config dictionary.
    
    Args:
        hf_config: Raw HuggingFace config.json content
        
    Returns:
        KeyeVisionConfig instance for the vision encoder
    """
    # Navigate to inner vision config
    outer_vcfg = hf_config.get("vision_config", {})
    inner_vcfg = outer_vcfg.get("vision_config", outer_vcfg)
    
    vision_cfg = KeyeVisionConfig(
        hidden_size=inner_vcfg.get("hidden_size", 1152),
        num_hidden_layers=inner_vcfg.get("num_hidden_layers", 27),
        num_attention_heads=inner_vcfg.get("num_attention_heads", 16),
        image_size=inner_vcfg.get("image_size", 384),
        patch_size=inner_vcfg.get("patch_size", 14),
        intermediate_size=inner_vcfg.get("intermediate_size", 4304),
        hidden_act=inner_vcfg.get("hidden_act", "gelu_pytorch_tanh"),
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_dropout=inner_vcfg.get("attention_dropout", 0.0),
        rope_theta=inner_vcfg.get("rope_theta", 10000.0),
        use_qk_norm=inner_vcfg.get("use_qk_norm", False),
        qk_norm_eps=inner_vcfg.get("qk_norm_eps", 1e-6),
        attention_function=hf_config.get("_attn_implementation", "flash_attention_2"),
    )
    return vision_cfg


def _build_tokenizer_config(
    hf_config: Dict[str, Any],
    vision_cfg: KeyeVisionConfig
) -> KeyeTokenizerConfig:
    """Build Muse KeyeTokenizerConfig from HuggingFace config dictionary.
    
    Args:
        hf_config: Raw HuggingFace config.json content
        vision_cfg: Already-built KeyeVisionConfig
        
    Returns:
        KeyeTokenizerConfig instance for the visual tokenizer
    """
    outer_vcfg = hf_config.get("vision_config", {})
    
    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=outer_vcfg.get("llm_hidden_size", 4096),
        embedding_dim=outer_vcfg.get("embedding_dim", 128),
        init_embedding_dim=outer_vcfg.get("init_embedding_dim", 4096),
        codebook_size=outer_vcfg.get("codebook_size", 65536),
        n_q_tokens=outer_vcfg.get("n_q_tokens", 8),
        split_voc=outer_vcfg.get("split_voc", 1),
        add_voc_reducer=outer_vcfg.get("add_voc_reducer", False),
        split_dim=outer_vcfg.get("split_dim", False),
        vq_sampling_mode="argmin",
        vq_temperature=1.0,
        vq_temperature_decay=0.999,
        vq_min_temperature=0.1,
        output_dim=outer_vcfg.get("output_dim", 1024),
    )
    return tokenizer_cfg


def build_configs(hf_config: Dict[str, Any]) -> Tuple[Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig]:
    """Build all required configs from HuggingFace config.
    
    Args:
        hf_config: Raw HuggingFace config.json content
        
    Returns:
        Tuple of (qwen_config, vision_config, tokenizer_config)
    """
    qwen_cfg = _build_qwen3_config(hf_config)
    vision_cfg = _build_vision_config(hf_config)
    tokenizer_cfg = _build_tokenizer_config(hf_config, vision_cfg)
    
    return qwen_cfg, vision_cfg, tokenizer_cfg


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Keye-VL checkpoint to Muse format"
    )
    parser.add_argument(
        "--hf-dir", 
        type=str, 
        required=True,
        help="Path to HuggingFace checkpoint directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Output directory for converted Muse checkpoint"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights"
    )
    parser.add_argument(
        "--image-token-id",
        type=int,
        default=151655,
        help="Token ID for image placeholder"
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="avg",
        choices=["avg", "sum"],
        help="Pooling method for visual token projection"
    )
    parser.add_argument(
        "--amplifier",
        type=float,
        default=1.0,
        help="Amplifier for visual token projection"
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    
    # Load HuggingFace config
    hf_config_path = Path(args.hf_dir) / "config.json"
    logger.info(f"Loading config from {hf_config_path}")
    with open(hf_config_path, encoding="utf-8") as f:
        hf_config = json.loads(f.read())
    
    # Build configs
    qwen_cfg, vision_cfg, tokenizer_cfg = build_configs(hf_config)
    
    # Get image_token_id from config if available
    image_token_id = hf_config.get("image_token_id", args.image_token_id)
    
    logger.info(f"Qwen3 config: embed_dim={qwen_cfg.embed_dim}, num_layers={qwen_cfg.num_layers}")
    logger.info(f"Vision config: hidden_size={vision_cfg.hidden_size}, num_layers={vision_cfg.num_hidden_layers}")
    logger.info(f"Tokenizer config: n_q_tokens={tokenizer_cfg.n_q_tokens}, codebook_size={tokenizer_cfg.codebook_size}")
    
    # Load HuggingFace state dict
    logger.info(f"Loading HuggingFace checkpoint from {args.hf_dir}")
    hf_state_dict = load_hf_checkpoint(args.hf_dir)
    logger.info(f"Loaded {len(hf_state_dict)} keys from HuggingFace checkpoint")
    
    # Convert state dict using model's built-in converter
    tie_word_embeddings = hf_config.get("text_config", hf_config).get("tie_word_embeddings", True)
    state_dict = KeyeTokenizerEnd2EndImage.convert_hf_state_dict(
        hf_state_dict,
        tie_word_embeddings=tie_word_embeddings
    )
    logger.info(f"Converted to {len(state_dict)} Muse keys")
    
    # Create model
    logger.info(f"Creating model with dtype={args.dtype}")
    
    # Create unified config
    unified_config = KeyeTokenizerEnd2EndImageConfig(
        qwen_config=qwen_cfg,
        vision_config=vision_cfg,
        tokenizer_config=tokenizer_cfg,
        image_token_id=image_token_id,
        pool=args.pool,
        amplifier=args.amplifier
    )
    
    with set_default_dtype(args.dtype), torch.device("cpu"):
        model = KeyeTokenizerEnd2EndImage(unified_config)
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
        for k in missing[:20]:
            logger.warning(f"  - {k}")
        if len(missing) > 20:
            logger.warning(f"  ... and {len(missing) - 20} more")
    
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:20]:
            logger.warning(f"  - {k}")
        if len(unexpected) > 20:
            logger.warning(f"  ... and {len(unexpected) - 20} more")
    
    if not missing and not unexpected:
        logger.info("All keys matched perfectly!")
    
    # Save converted checkpoint
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving converted checkpoint to {output_path}")
    # Force updating the config in the model before saving
    model.config = unified_config
    model.save_pretrained(str(output_path))
    
    # Save unified config as muse_config.json for reference
    with open(output_path / "muse_config.json", "w", encoding="utf-8") as f:
        f.write(unified_config.model_dump_json(indent=2))
    
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
