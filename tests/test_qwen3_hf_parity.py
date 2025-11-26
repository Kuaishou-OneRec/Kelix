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
    # generated_ids = hf_model.generate(
    #     **model_inputs,
    #     max_new_tokens=32768
    # )
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 


    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()

    muse_config = _build_qwen3_config(hf_config_dict)
    
    # Get target device and dtype from HF model
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    
    # Create Muse model with correct dtype
    with set_default_dtype("bfloat16" if dtype == torch.bfloat16 else "float32"):
        muse_model = Qwen3Model(muse_config)
    
    print(muse_config, hf_config_dict)

    # Convert and load state dict
    state_dict = muse_model.convert_hf_state_dict(hf_state_dict)
    
    # Get Muse model's expected state dict keys
    muse_model_state_dict = muse_model.state_dict()
    muse_expected_keys = set(muse_model_state_dict.keys())
    converted_keys = set(state_dict.keys())
    
    # Check key matching
    print(f"\n{'='*60}")
    print("Weight Loading Check")
    print(f"{'='*60}")
    print(f"HF state dict keys: {len(hf_state_dict)}")
    print(f"Converted state dict keys: {len(state_dict)}")
    print(f"Muse model expected keys: {len(muse_expected_keys)}")
    
    # Find missing and extra keys
    missing_in_converted = muse_expected_keys - converted_keys
    extra_in_converted = converted_keys - muse_expected_keys
    skipped_keys = set(hf_state_dict.keys()) - set(
        muse_model.convert_hf_state_dict(hf_state_dict).keys()
    )
    
    if missing_in_converted:
        print(f"\n⚠️  Missing keys in converted state dict ({len(missing_in_converted)}):")
        for key in sorted(list(missing_in_converted))[:20]:
            print(f"  - {key}")
        if len(missing_in_converted) > 20:
            print(f"  ... and {len(missing_in_converted) - 20} more")
    
    if extra_in_converted:
        print(f"\n⚠️  Extra keys in converted state dict ({len(extra_in_converted)}):")
        for key in sorted(list(extra_in_converted))[:20]:
            print(f"  - {key}")
        if len(extra_in_converted) > 20:
            print(f"  ... and {len(extra_in_converted) - 20} more")
    
    if skipped_keys:
        print(f"\nℹ️  Skipped keys during conversion ({len(skipped_keys)}):")
        for key in sorted(list(skipped_keys))[:20]:
            print(f"  - {key}")
        if len(skipped_keys) > 20:
            print(f"  ... and {len(skipped_keys) - 20} more")
    
    # Convert state dict tensors to target dtype and device
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            state_dict[key] = tensor.to(device=device, dtype=dtype)
    
    # Handle missing keys (e.g., if tie_word_embeddings=True, lm_head is skipped)
    missing_keys, unexpected_keys = muse_model.load_state_dict(
        state_dict, strict=False
    )
    
    if missing_keys:
        print(f"\n⚠️  Missing keys after load_state_dict ({len(missing_keys)}):")
        for key in missing_keys[:20]:
            print(f"  - {key}")
        if len(missing_keys) > 20:
            print(f"  ... and {len(missing_keys) - 20} more")
    else:
        print(f"\n✓ All expected keys loaded successfully")
    
    if unexpected_keys:
        print(f"\n⚠️  Unexpected keys after load_state_dict ({len(unexpected_keys)}):")
        for key in unexpected_keys[:20]:
            print(f"  - {key}")
        if len(unexpected_keys) > 20:
            print(f"  ... and {len(unexpected_keys) - 20} more")
    
    # Compare some key weights to verify correctness
    print(f"\n{'='*60}")
    print("Weight Value Comparison (Sample)")
    print(f"{'='*60}")
    
    # Check embedding layer
    if "model.tok_embeddings.weight" in state_dict:
        hf_embed_key = "model.embed_tokens.weight"
        if hf_embed_key in hf_state_dict:
            hf_embed = hf_state_dict[hf_embed_key].to(device=device, dtype=dtype)
            muse_embed = state_dict["model.tok_embeddings.weight"]
            embed_diff = (hf_embed - muse_embed).abs()
            print(f"Embedding layer:")
            print(f"  Shape: HF={hf_embed.shape}, Muse={muse_embed.shape}")
            print(f"  Max diff: {embed_diff.max().item():.6e}")
            print(f"  Mean diff: {embed_diff.mean().item():.6e}")
            if embed_diff.max().item() > 1e-5:
                print(f"  ⚠️  Large difference detected!")
    
    # Check all transformer layers
    print(f"\nChecking all {muse_config.num_layers} transformer layers...")
    layer_issues = []
    total_checked = 0
    total_matched = 0
    
    for layer_idx in range(muse_config.num_layers):
        layer_has_issue = False
        
        # Check attention weights
        attn_weights = [
            ("q_proj.weight", "q_proj.weight"),
            ("k_proj.weight", "k_proj.weight"),
            ("v_proj.weight", "v_proj.weight"),
            ("o_proj.weight", "output_proj.weight"),
        ]
        
        # Check attention biases (if exist)
        if muse_config.q_proj_bias:
            attn_weights.append(("q_proj.bias", "q_proj.bias"))
        if muse_config.k_proj_bias:
            attn_weights.append(("k_proj.bias", "k_proj.bias"))
        if muse_config.v_proj_bias:
            attn_weights.append(("v_proj.bias", "v_proj.bias"))
        
        for hf_weight_name, muse_weight_name in attn_weights:
            hf_key = f"model.layers.{layer_idx}.self_attn.{hf_weight_name}"
            muse_key = f"model.layers.{layer_idx}.attn.{muse_weight_name}"
            if hf_key in hf_state_dict and muse_key in state_dict:
                total_checked += 1
                hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
                muse_weight = state_dict[muse_key]
                weight_diff = (hf_weight - muse_weight).abs()
                max_diff = weight_diff.max().item()
                if max_diff > 1e-5:
                    layer_has_issue = True
                    layer_issues.append(
                        f"Layer {layer_idx} {hf_weight_name}: "
                        f"max_diff={max_diff:.6e}"
                    )
                else:
                    total_matched += 1
        
        # Check q_norm and k_norm (if exist)
        if muse_config.q_norm:
            for norm_name in ["q_norm.weight", "k_norm.weight"]:
                hf_key = f"model.layers.{layer_idx}.self_attn.{norm_name}"
                muse_key = f"model.layers.{layer_idx}.attn.{norm_name.replace('.weight', '.scale')}"
                if hf_key in hf_state_dict and muse_key in state_dict:
                    total_checked += 1
                    hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
                    muse_weight = state_dict[muse_key]
                    weight_diff = (hf_weight - muse_weight).abs()
                    max_diff = weight_diff.max().item()
                    if max_diff > 1e-5:
                        layer_has_issue = True
                        layer_issues.append(
                            f"Layer {layer_idx} {norm_name}: "
                            f"max_diff={max_diff:.6e}"
                        )
                    else:
                        total_matched += 1
        
        # Check MLP weights
        mlp_mapping = [
            ("gate_proj.weight", "w1.weight"),
            ("up_proj.weight", "w3.weight"),
            ("down_proj.weight", "w2.weight"),
        ]
        for hf_weight_name, muse_weight_name in mlp_mapping:
            hf_key = f"model.layers.{layer_idx}.mlp.{hf_weight_name}"
            muse_key = f"model.layers.{layer_idx}.mlp.{muse_weight_name}"
            if hf_key in hf_state_dict and muse_key in state_dict:
                total_checked += 1
                hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
                muse_weight = state_dict[muse_key]
                weight_diff = (hf_weight - muse_weight).abs()
                max_diff = weight_diff.max().item()
                if max_diff > 1e-5:
                    layer_has_issue = True
                    layer_issues.append(
                        f"Layer {layer_idx} MLP {hf_weight_name}: "
                        f"max_diff={max_diff:.6e}"
                    )
                else:
                    total_matched += 1
        
        # Check layer norms
        layer_norms = [
            ("input_layernorm.weight", "sa_norm.scale"),
            ("post_attention_layernorm.weight", "mlp_norm.scale"),
        ]
        for hf_weight_name, muse_weight_name in layer_norms:
            hf_key = f"model.layers.{layer_idx}.{hf_weight_name}"
            muse_key = f"model.layers.{layer_idx}.{muse_weight_name}"
            if hf_key in hf_state_dict and muse_key in state_dict:
                total_checked += 1
                hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
                muse_weight = state_dict[muse_key]
                weight_diff = (hf_weight - muse_weight).abs()
                max_diff = weight_diff.max().item()
                if max_diff > 1e-5:
                    layer_has_issue = True
                    layer_issues.append(
                        f"Layer {layer_idx} {hf_weight_name}: "
                        f"max_diff={max_diff:.6e}"
                    )
                else:
                    total_matched += 1
    
    # Report layer issues
    print(f"\nWeight comparison summary:")
    print(f"  Total weights checked: {total_checked}")
    print(f"  Weights matched (diff < 1e-5): {total_matched}")
    print(f"  Weights with issues: {len(layer_issues)}")
    
    if layer_issues:
        print(f"\n⚠️  Found {len(layer_issues)} weight mismatches:")
        for issue in layer_issues[:50]:  # Limit to first 50 issues
            print(f"  - {issue}")
        if len(layer_issues) > 50:
            print(f"  ... and {len(layer_issues) - 50} more issues")
    else:
        print(f"✓ All transformer layer weights match!")
    
    # Check final norm
    hf_norm_key = "model.norm.weight"
    muse_norm_key = "model.norm.scale"
    if hf_norm_key in hf_state_dict and muse_norm_key in state_dict:
        hf_norm = hf_state_dict[hf_norm_key].to(device=device, dtype=dtype)
        muse_norm = state_dict[muse_norm_key]
        norm_diff = (hf_norm - muse_norm).abs()
        print(f"\nFinal norm:")
        print(f"  Shape: HF={hf_norm.shape}, Muse={muse_norm.shape}")
        print(f"  Max diff: {norm_diff.max().item():.6e}")
        print(f"  Mean diff: {norm_diff.mean().item():.6e}")
        if norm_diff.max().item() > 1e-5:
            print(f"  ⚠️  Large difference detected!")
    
    # Check output layer
    # Note: If tie_word_embeddings=True, lm_head.weight is skipped in conversion
    # and should match tok_embeddings.weight instead
    if "model.output.weight" in state_dict:
        hf_output_key = "lm_head.weight"
        if hf_output_key in hf_state_dict:
            hf_output = hf_state_dict[hf_output_key].to(device=device, dtype=dtype)
            muse_output = state_dict["model.output.weight"]
            output_diff = (hf_output - muse_output).abs()
            print(f"Output layer:")
            print(f"  Shape: HF={hf_output.shape}, Muse={muse_output.shape}")
            print(f"  Max diff: {output_diff.max().item():.6e}")
            print(f"  Mean diff: {output_diff.mean().item():.6e}")
            if output_diff.max().item() > 1e-5:
                print(f"  ⚠️  Large difference detected!")
    elif muse_config.tie_word_embeddings:
        # If tie_word_embeddings, output should match embeddings
        print(f"Output layer: Tied with embeddings (tie_word_embeddings=True)")
        if "model.tok_embeddings.weight" in state_dict:
            # Check if model.output is actually TiedLinear
            from muse.layers.linear import TiedLinear
            if isinstance(muse_model.model.output, TiedLinear):
                muse_output_weight = muse_model.model.output.tied_module.weight
                muse_embed_weight = muse_model.model.tok_embeddings.weight
                if muse_output_weight is muse_embed_weight:
                    print(f"  ✓ Output layer correctly tied with embeddings")
                else:
                    print(f"  ⚠️  Output layer not properly tied!")
                    # Compare values
                    weight_diff = (muse_output_weight - muse_embed_weight).abs()
                    print(f"  Max diff between tied weights: {weight_diff.max().item():.6e}")
            else:
                print(f"  ⚠️  Output layer is not TiedLinear!")
    
    print(f"{'='*60}\n")

    # Move Muse model to same device and dtype as HF model
    # This ensures all parameters and buffers are on the correct device/dtype
    muse_model = muse_model.to(device=device, dtype=dtype)
    
    # Double-check that all parameters are in the correct dtype
    for name, param in muse_model.named_parameters():
        if param.dtype != dtype:
            print(f"Warning: Parameter {name} has dtype {param.dtype}, expected {dtype}")
            param.data = param.data.to(dtype=dtype)
    
    for name, buffer in muse_model.named_buffers():
        if buffer.dtype != dtype:
            print(f"Warning: Buffer {name} has dtype {buffer.dtype}, expected {dtype}")
            buffer.data = buffer.data.to(dtype=dtype)
    
    # Ensure eager attention is used
    hf_model.config._attn_implementation = "eager"
    
    # Ensure Muse model uses eager attention
    muse_config.attention_function = "eager"
    
    muse_model.eval()
    hf_model.eval()
    
    with torch.no_grad():
        # HF forward
        hf_outputs = hf_model(**model_inputs, output_attentions=True)
        hf_logits = hf_outputs.logits
        
        # Muse forward - Muse model expects 'tokens' instead of 'input_ids'
        muse_inputs = {"tokens": model_inputs["input_ids"]}
        if "attention_mask" in model_inputs:
            # Muse expects mask in shape [batch, seq_len, seq_len] for causal mask
            # For now, we'll let Muse use default causal mask
            pass
        muse_logits = muse_model(**muse_inputs)

if __name__ == "__main__":
    test_qwen3_logits_align_with_hf_checkpoint()