from typing import Dict, Any
from pathlib import Path
import json
import argparse
import torch
from muse.config import KeyeVisionConfig, KeyeTokenizerConfig
from muse.models.keye_tokenizer import KeyeImageTokenizer as MuseKeyeImageTokenizer
from muse.training.common import set_default_dtype

def _build_muse_tokenizer_config(hf_config: Dict[str, Any]) -> KeyeTokenizerConfig:
    """Build Muse KeyeTokenizerConfig from raw config dictionary."""
    outer_vcfg = hf_config["vision_config"]
    inner_vcfg = outer_vcfg["vision_config"]
    
    vision_cfg = KeyeVisionConfig(
        hidden_size=inner_vcfg["hidden_size"],
        num_hidden_layers=inner_vcfg["num_hidden_layers"],
        num_attention_heads=inner_vcfg["num_attention_heads"],
        image_size=inner_vcfg["image_size"],
        patch_size=inner_vcfg["patch_size"],
        intermediate_size=inner_vcfg["intermediate_size"],
        hidden_act=inner_vcfg.get("hidden_act", "gelu_pytorch_tanh"),
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_dropout=inner_vcfg.get("attention_dropout", 0.0),
        rope_theta=inner_vcfg.get("rope_theta", 10000.0),
        use_qk_norm=inner_vcfg.get("use_qk_norm", False),
        qk_norm_eps=inner_vcfg.get("qk_norm_eps", 1e-6),
        attention_function=raw_cfg.get("_attn_implementation", "flash_attention_2"),
    )
    
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
    )
    return tokenizer_cfg

def extract_and_save_tokenizer_weights(
    full_state_dict: Dict[str, torch.Tensor],
    save_path: str,
    raw_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Extract visual_tokenizer weights from full Keye-VL checkpoint,
    convert to Muse format, and save to local path.
    
    Returns the converted Muse-style state dict.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract visual_tokenizer.* keys from full state dict
    origin_tokenizer_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith("visual_tokenizer."):
            new_k = k[len("visual_tokenizer."):]
            origin_tokenizer_state_dict[new_k] = v

    muse_state_dict = {}
    for k, v in origin_tokenizer_state_dict.items():
        new_k = k
        
        # Convert visual.vision_model.* to visual.*
        if k.startswith("visual.vision_model."):
            new_k = "visual." + k[len("visual.vision_model."):]
        
        # Convert encoder.layers.X.layer_norm1 -> encoder.layers.X.sa_norm
        new_k = new_k.replace(".layer_norm1.", ".sa_norm.")
        # Convert encoder.layers.X.layer_norm2 -> encoder.layers.X.mlp_norm
        new_k = new_k.replace(".layer_norm2.", ".mlp_norm.")
        # Convert self_attn -> attn
        new_k = new_k.replace(".self_attn.", ".attn.")
        # Convert out_proj -> output_proj
        new_k = new_k.replace(".out_proj.", ".output_proj.")
        # Convert mlp.fc1 -> mlp.w1
        new_k = new_k.replace(".mlp.fc1.", ".mlp.w1.")
        # Convert mlp.fc2 -> mlp.w2
        new_k = new_k.replace(".mlp.fc2.", ".mlp.w2.")
        # Convert post_layernorm -> ln_post
        new_k = new_k.replace(".post_layernorm.", ".ln_post.")
        
        muse_state_dict[new_k] = v
    
    
    # 3. Save the converted weights
    weights_path = save_dir / "pytorch_model.bin"
    torch.save(muse_state_dict, weights_path)
    
    # 4. Save the config as JSON
    config_path = save_dir / "config.json"
    tokenizer_cfg = _build_muse_tokenizer_config(raw_cfg)
    with open(config_path, "w") as f:
        json.dump(tokenizer_cfg.dict(), f, indent=2)
    
    # 5. Copy preprocessor_config.json if exists in original checkpoint
    original_ckpt_dir = Path(DEFAULT_CKPT)
    preprocessor_config_src = original_ckpt_dir / "preprocessor_config.json"
    if preprocessor_config_src.exists():
        import shutil
        shutil.copy(preprocessor_config_src, save_dir / "preprocessor_config.json")
    
    return muse_state_dict


def load_muse_tokenizer_from_saved(
    save_path: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[MuseKeyeImageTokenizer, KeyeTokenizerConfig]:
    """
    Load Muse KeyeImageTokenizer from saved weights.
    """
    save_dir = Path(save_path)
    
    # Load config
    config_path = save_dir / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Build KeyeTokenizerConfig from saved config
    vision_cfg_dict = config_dict.get("vision_config", {})
    vision_cfg = KeyeVisionConfig(**vision_cfg_dict)
    
    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=config_dict.get("llm_hidden_size", 4096),
        embedding_dim=config_dict.get("embedding_dim", 128),
        init_embedding_dim=config_dict.get("init_embedding_dim", 4096),
        codebook_size=config_dict.get("codebook_size", 65536),
        n_q_tokens=config_dict.get("n_q_tokens", 8),
        split_voc=config_dict.get("split_voc", 1),
        add_voc_reducer=config_dict.get("add_voc_reducer", False),
        split_dim=config_dict.get("split_dim", False),
        vq_sampling_mode=config_dict.get("vq_sampling_mode", "argmin"),
        vq_temperature=config_dict.get("vq_temperature", 1.0),
        vq_temperature_decay=config_dict.get("vq_temperature_decay", 0.999),
        vq_min_temperature=config_dict.get("vq_min_temperature", 0.1),
    )
    
    # Initialize model
    with set_default_dtype(dtype):
        muse_tokenizer = MuseKeyeImageTokenizer(tokenizer_cfg).to(device)
    
    # Load weights
    weights_path = save_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location="cpu")
    
    missing, unexpected = muse_tokenizer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Muse tokenizer missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"Muse tokenizer unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
    
    muse_tokenizer.eval()
    print(f"Loaded Muse tokenizer from: {save_path}")
    
    return muse_tokenizer, tokenizer_cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=True)
    parser.add_argument("--outpur-dir", type=str, required=True)
    parser.add_argument("--processor-dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()

    muse_config = _build_muse_tokenizer_config(Path(args.hf_dir) / "config.json")
    print(muse_config)

def main():
    args = get_args()

if __name__ == "__main__":
    main()
