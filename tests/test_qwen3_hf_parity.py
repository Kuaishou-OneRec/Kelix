"""
Integration test to ensure Muse Qwen3 matches Hugging Face logits.
"""

import os
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    )
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


def test_qwen3_logits_align_with_hf_checkpoint():
    """Ensure Muse Qwen3 logits match the Hugging Face reference model."""
    # checkpoint_dir = os.environ.get(CHECKPOINT_ENV)
    # if not checkpoint_dir:
    #     pytest.skip(f"{CHECKPOINT_ENV} environment variable is not set.")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"
    )

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

    # conduct text completion
    generated_ids = hf_model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 


    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()

    muse_config = _build_qwen3_config(hf_config_dict)
    with set_default_dtype(torch.bfloat16):
        muse_model = Qwen3Model(muse_config)

    # Convert and load state dict
    state_dict = muse_model.convert_hf_state_dict(hf_state_dict)
    
    # Handle missing keys (e.g., if tie_word_embeddings=True, lm_head is skipped)
    missing_keys, unexpected_keys = muse_model.load_state_dict(
        state_dict, strict=False
    )
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:10]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:10]}...")

    # Move Muse model to same device and dtype as HF model
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    muse_model = muse_model.to(device=device, dtype=dtype)
    muse_model.eval()
    hf_model.eval()

    # Get HF model logits
    with torch.no_grad():
        hf_outputs = hf_model(**model_inputs)
        hf_logits = hf_outputs.logits

    # Get Muse model logits
    # Muse model expects 'tokens' instead of 'input_ids'
    # For simple cases without padding, we can omit mask and use default causal mask
    muse_tokens = model_inputs["input_ids"]
    
    # Note: Muse model uses default causal mask if mask is not provided
    # If attention_mask is needed (e.g., for padding), it should be converted
    # from HF format [batch, seq_len] to Muse format [batch, seq_len, seq_len]
    # For this test, we assume no padding, so we omit the mask
    with torch.no_grad():
        muse_logits = muse_model(tokens=muse_tokens)

    # Compare logits
    print(f"\n{'='*60}")
    print("Logits Comparison")
    print(f"{'='*60}")
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Muse logits shape: {muse_logits.shape}")
    print(f"HF logits dtype: {hf_logits.dtype}")
    print(f"Muse logits dtype: {muse_logits.dtype}")
    
    # Ensure same dtype for comparison
    if hf_logits.dtype != muse_logits.dtype:
        muse_logits = muse_logits.to(dtype=hf_logits.dtype)
    
    # Calculate differences
    diff = (hf_logits - muse_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    max_diff_per_token = diff.max(dim=-1)[0].max().item()
    
    print(f"\nMax absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Max difference per token: {max_diff_per_token:.6e}")
    
    # Check if logits are close
    # Use reasonable tolerance for floating point comparison
    rtol = 1e-4
    atol = 1e-4
    
    is_close = torch.allclose(hf_logits, muse_logits, rtol=rtol, atol=atol)
    print(f"\nLogits match (rtol={rtol}, atol={atol}): {is_close}")
    
    if not is_close:
        # Find positions with largest differences
        max_diff_pos = diff.argmax()
        max_diff_pos_3d = torch.unravel_index(max_diff_pos, diff.shape)
        print(f"\nLargest difference at position: {max_diff_pos_3d}")
        print(f"HF value: {hf_logits[max_diff_pos_3d].item():.6f}")
        print(f"Muse value: {muse_logits[max_diff_pos_3d].item():.6f}")
        print(f"Difference: {diff[max_diff_pos_3d].item():.6e}")
    
    # Assert that logits are close (with reasonable tolerance)
    assert torch.allclose(
        hf_logits, muse_logits, rtol=rtol, atol=atol
    ), (
        f"Logits do not match! "
        f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
    )
    
    print(f"\n{'='*60}")
    print("✓ Logits comparison passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_qwen3_logits_align_with_hf_checkpoint()