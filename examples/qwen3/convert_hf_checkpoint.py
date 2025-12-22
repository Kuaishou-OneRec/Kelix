from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import torch
from muse.config import Qwen3Config
from muse.models.qwen3 import Qwen3Model
from muse.training.common import set_default_dtype


def _build_qwen3_config(hf_cfg: Dict[str, Any]) -> Qwen3Config:
    """Map Hugging Face config to Muse Qwen3Config."""
    embed_dim = hf_cfg.get("hidden_size") or hf_cfg.get("dim")
    num_heads = hf_cfg.get("num_attention_heads") or hf_cfg.get("n_head")
    num_layers = hf_cfg.get("num_hidden_layers") or hf_cfg.get("n_layer")
    num_kv_heads = (
        hf_cfg.get("num_key_value_heads")
        or hf_cfg.get("n_kv_head")
        or num_heads
    
    head_dim = hf_cfg.get("head_dim") or (embed_dim // num_heads)
    intermediate_dim = hf_cfg.get("intermediate_size") or hf_cfg.get(
        "ffn_hidden_size", 4 * embed_dim
    )
    max_seq_len = (
        hf_cfg.get("max_position_embeddings")
        or hf_cfg.get("max_seq_len")
        or 32768
    )
    rope_base = hf_cfg.get("rope_theta", hf_cfg.get("rotary_emb_base", 10000.0))
    attn_dropout = hf_cfg.get(
        "attention_dropout",
        hf_cfg.get("attention_dropout_prob", 0.0),
    )
    qkv_bias = hf_cfg.get("use_qkv_bias")
    q_norm_flag = hf_cfg.get("use_qk_norm", hf_cfg.get("qk_norm", True))

    attention_function = (
        "flash_attention_2" if hf_cfg.get("use_flash_attn", False) else "eager"
    )

    return Qwen3Config(
        model_class="Qwen3Model",
        vocab_size=hf_cfg["vocab_size"],
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_dim,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
        norm_eps=hf_cfg.get("rms_norm_eps", 1e-6),
        attn_dropout=attn_dropout,
        tie_word_embeddings=hf_cfg.get("tie_word_embeddings", True),
        q_proj_bias=hf_cfg.get("q_proj_bias", qkv_bias or False),
        k_proj_bias=hf_cfg.get("k_proj_bias", qkv_bias or False),
        v_proj_bias=hf_cfg.get("v_proj_bias", qkv_bias or False),
        attention_function=attention_function,
        q_norm=q_norm_flag,
        k_norm=q_norm_flag,
    )

def convert_hf_checkpoint(hf_checkpoint_path: str,
                          new_model_dir: str):
    """Convert a Hugging Face checkpoint to a Muse checkpoint"""

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto")
    hf_config_dict = hf_model.config.to_dict()
    config = _build_qwen3_config(hf_config_dict)
    with set_default_dtype(torch.bfloat16):
        model = Qwen3Model(config)

    state_dict = model.convert_hf_state_dict(hf_model.state_dict())

    model.load_state_dict(state_dict)
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype

    model = model.to(device=device, dtype=dtype)
    
    # Double-check that all parameters are in the correct dtype
    for name, param in model.named_parameters():
        if param.dtype != dtype:
            print(f"Warning: Parameter {name} has dtype {param.dtype}, expected {dtype}")
            param.data = param.data.to(dtype=dtype)
    
    for name, buffer in model.named_buffers():
        if buffer.dtype != dtype:
            print(f"Warning: Buffer {name} has dtype {buffer.dtype}, expected {dtype}")
            buffer.data = buffer.data.to(dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_path)
    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(hf_model.device)


    # Ensure eager attention is used
    hf_model.config._attn_implementation = "eager"
    
    # Ensure Muse model uses eager attention
    model.config.attention_function = "eager"

    model.to(device="cuda")

    with torch.no_grad():
        # HF forward
        print("Running HF model forward pass...")
        hf_outputs = hf_model(**model_inputs)
        hf_logits = hf_outputs.logits
        
        # Muse forward - Muse model expects 'tokens' instead of 'input_ids'
        print("Running Muse model forward pass...")
        inputs = {"tokens": model_inputs["input_ids"]}
        logits = model(**inputs)

        # Calculate differences
        logits_diff = (hf_logits - logits).abs()
        max_diff = logits_diff.max().item()
        mean_diff = logits_diff.mean().item()
        median_diff = logits_diff.median().item()
        
        # Calculate relative differences
        hf_abs = hf_logits.abs()
        relative_diff = logits_diff / (hf_abs + 1e-8)  # Add small epsilon to avoid division by zero
        max_relative_diff = relative_diff.max().item()
        mean_relative_diff = relative_diff.mean().item()

        print(f"Max diff: {max_diff:.6e}")
        print(f"Mean diff: {mean_diff:.6e}")
        print(f"Median diff: {median_diff:.6e}")
        print(f"Max relative diff: {max_relative_diff:.6e}")
        print(f"Mean relative diff: {mean_relative_diff:.6e}")

        print(f"{'='*60}\n")

        # Summary
        if max_diff < 1e-5:
            print("✓✓✓ SUCCESS: Logits match perfectly!")
        else:
            print("✗ FAILURE: Logits differ beyond tolerance")

    model.save_pretrained(new_model_dir)
    tokenizer.save_pretrained(new_model_dir)

if __name__ == "__main__":
    convert_hf_checkpoint("/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base", "/llm_reco_ssd/zhouyang12/models/muse/Qwen3-8B-Base")