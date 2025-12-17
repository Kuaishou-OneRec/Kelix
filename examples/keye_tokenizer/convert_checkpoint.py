from typing import Dict, Any, Tuple
from pathlib import Path
import json
import argparse
import torch
from muse.config import KeyeVisionConfig, KeyeTokenizerConfig
from muse.models.keye_tokenizer import KeyeImageTokenizer
from muse.training.common import set_default_dtype
from muse.training.checkpoint import load_hf_checkpoint

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
        attention_function=hf_config.get("_attn_implementation", "flash_attention_2"),
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
        output_dim=outer_vcfg.get("output_dim", 1024),
    )
    return tokenizer_cfg

def convert_hf_checkpoint(hf_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract visual_tokenizer weights from full Keye-VL checkpoint,
    convert to Muse format, and save to local path.
    
    Returns the converted Muse-style state dict.
    """
    
    # 1. Extract visual_tokenizer.* keys from full state dict
    origin_tokenizer_state_dict = {}
    for k, v in hf_state_dict.items():
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
        new_k = new_k.replace("quant_projector", "up_projectors")
        
        muse_state_dict[new_k] = v
    
    return muse_state_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--processor-dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()

def main():
    args = get_args()
    hf_config_path = Path(args.hf_dir) / "config.json"
    with open(hf_config_path) as f:
        hf_config = json.loads(f.read())
    config = _build_muse_tokenizer_config(hf_config)
    hf_state_dict = load_hf_checkpoint(args.hf_dir)
    state_dict = convert_hf_checkpoint(hf_state_dict)

    with set_default_dtype(args.dtype), torch.device("cpu"):
        tokenizer = KeyeImageTokenizer(config)

    missing, unexpected = tokenizer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Tokenizer missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"Tokenizer unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
    
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
