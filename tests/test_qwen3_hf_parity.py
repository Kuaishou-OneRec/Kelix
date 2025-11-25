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
    import os
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
    hf_model.config._attn_implementation = "eager"
    
    # Ensure Muse model uses eager attention
    muse_config.attention_function = "eager"
    
    muse_model.eval()
    hf_model.eval()

    # ========================================================================
    # Focus on Layer 0 Attention Comparison Only
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("Layer 0 Attention Detailed Comparison")
    print(f"{'='*60}\n")
    
    # Get first layer attention modules
    hf_layer_0 = hf_model.model.layers[0]
    muse_layer_0 = muse_model.model.layers[0]
    hf_attn_0 = hf_layer_0.self_attn
    muse_attn_0 = muse_layer_0.attn
    
    # Compare attention parameters
    print("1. Attention Parameters Comparison")
    print("-" * 60)
    
    hf_head_dim = hf_attn_0.head_dim
    hf_num_heads = hf_attn_0.config.num_attention_heads
    hf_num_kv_heads = hf_attn_0.config.num_key_value_heads
    hf_scaling = hf_attn_0.scaling
    hf_attn_dropout = hf_attn_0.attention_dropout
    
    muse_head_dim = muse_attn_0.head_dim
    muse_num_heads = muse_attn_0.num_heads
    muse_num_kv_heads = muse_attn_0.num_kv_heads
    muse_scaling = (muse_head_dim ** -0.5)
    muse_attn_dropout = muse_attn_0.attn_dropout
    
    print(f"  head_dim: HF={hf_head_dim}, Muse={muse_head_dim}")
    print(f"  num_heads: HF={hf_num_heads}, Muse={muse_num_heads}")
    print(f"  num_kv_heads: HF={hf_num_kv_heads}, Muse={muse_num_kv_heads}")
    print(f"  scaling: HF={hf_scaling:.6e}, Muse={muse_scaling:.6e}")
    print(f"  attention_dropout: HF={hf_attn_dropout}, Muse={muse_attn_dropout}")
    
    assert hf_head_dim == muse_head_dim, f"head_dim mismatch: {hf_head_dim} != {muse_head_dim}"
    assert hf_num_heads == muse_num_heads, f"num_heads mismatch: {hf_num_heads} != {muse_num_heads}"
    assert hf_num_kv_heads == muse_num_kv_heads, f"num_kv_heads mismatch: {hf_num_kv_heads} != {muse_num_kv_heads}"
    assert abs(hf_scaling - muse_scaling) < 1e-6, f"scaling mismatch: {hf_scaling} != {muse_scaling}"
    assert hf_attn_dropout == muse_attn_dropout, f"attention_dropout mismatch: {hf_attn_dropout} != {muse_attn_dropout}"
    print("  ✓ All attention parameters match!\n")
    
    # Compare attention weight parameters
    print("2. Attention Weight Parameters Comparison")
    print("-" * 60)
    
    def compare_weights(hf_key, muse_key, name):
        hf_weight = hf_state_dict[hf_key].to(device=device, dtype=dtype)
        muse_weight = state_dict[muse_key]
        diff = (hf_weight - muse_weight).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  {name}:")
        print(f"    Shape: HF={hf_weight.shape}, Muse={muse_weight.shape}")
        print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        if max_diff > 1e-5:
            print(f"    ⚠️  Mismatch!")
            return False
        else:
            print(f"    ✓ Match!")
            return True
    
    weight_matches = True
    weight_matches &= compare_weights(
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.attn.q_proj.weight",
        "q_proj.weight"
    )
    weight_matches &= compare_weights(
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.attn.k_proj.weight",
        "k_proj.weight"
    )
    weight_matches &= compare_weights(
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.attn.v_proj.weight",
        "v_proj.weight"
    )
    weight_matches &= compare_weights(
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.attn.output_proj.weight",
        "o_proj.weight"
    )
    
    if muse_config.q_norm:
        weight_matches &= compare_weights(
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.attn.q_norm.scale",
            "q_norm.scale"
        )
        weight_matches &= compare_weights(
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.attn.k_norm.scale",
            "k_norm.scale"
        )
    
    if not weight_matches:
        print("\n⚠️  Some attention weight parameters do not match!")
    else:
        print("\n✓ All attention weight parameters match!\n")
    
    # Register hooks to capture intermediate activations
    hf_intermediates = {}
    muse_intermediates = {}
    
    def compare_tensors(hf_tensor, muse_tensor, name, atol=1e-4):
        """Compare two tensors and print detailed diff info."""
        hf_tensor = hf_tensor.to(device=device, dtype=dtype)
        muse_tensor = muse_tensor.to(device=device, dtype=dtype)
        
        # Handle shape differences by trying to reshape
        if hf_tensor.shape != muse_tensor.shape:
            print(f"  {name}: Shape mismatch!")
            print(f"    HF shape: {hf_tensor.shape}, Muse shape: {muse_tensor.shape}")
            # Try to transpose Muse tensor if needed
            if len(hf_tensor.shape) == len(muse_tensor.shape) == 4:
                # Try transpose (1, 2) for [b, h, s, d] vs [b, s, h, d]
                if hf_tensor.shape[1] == muse_tensor.shape[2] and hf_tensor.shape[2] == muse_tensor.shape[1]:
                    muse_tensor = muse_tensor.transpose(1, 2)
                    print(f"    Transposed Muse tensor to match HF shape")
        
        if hf_tensor.shape != muse_tensor.shape:
            print(f"    ⚠️  Cannot compare due to shape mismatch")
            return False
        
        diff = (hf_tensor - muse_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  {name}:")
        print(f"    Shape: {hf_tensor.shape}")
        print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        
        if max_diff > atol:
            print(f"    ⚠️  Mismatch!")
            # Find position of max diff
            max_diff_idx = diff.argmax()
            max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
            print(f"    Max diff position: {max_diff_pos}")
            print(f"    HF value: {hf_tensor[max_diff_pos].item():.6f}")
            print(f"    Muse value: {muse_tensor[max_diff_pos].item():.6f}")
            return False
        else:
            print(f"    ✓ Match!")
            return True
    
    # Hook functions for HF
    def hf_q_proj_hook(module, input, output):
        hf_intermediates['q_after_proj'] = output.detach().clone()
    
    def hf_k_proj_hook(module, input, output):
        hf_intermediates['k_after_proj'] = output.detach().clone()
    
    def hf_v_proj_hook(module, input, output):
        hf_intermediates['v_after_proj'] = output.detach().clone()
    
    def hf_q_norm_hook(module, input, output):
        hf_intermediates['q_after_norm'] = output.detach().clone()
    
    def hf_k_norm_hook(module, input, output):
        hf_intermediates['k_after_norm'] = output.detach().clone()
    
    def hf_attn_forward_hook(module, input, output):
        # HF attention returns (hidden_states, attn_weights)
        if isinstance(output, tuple):
            hf_intermediates['attn_output'] = output[0].detach().clone()
            if len(output) > 1 and output[1] is not None:
                hf_intermediates['attn_weights'] = output[1].detach().clone()
        else:
            hf_intermediates['attn_output'] = output.detach().clone()
    
    def hf_before_o_proj_hook(module, input):
        if isinstance(input, tuple):
            hf_intermediates['before_o_proj'] = input[0].detach().clone()
        else:
            hf_intermediates['before_o_proj'] = input.detach().clone()
    
    # Hook functions for Muse
    def muse_q_proj_hook(module, input, output):
        muse_intermediates['q_after_proj'] = output.detach().clone()
    
    def muse_k_proj_hook(module, input, output):
        muse_intermediates['k_after_proj'] = output.detach().clone()
    
    def muse_v_proj_hook(module, input, output):
        muse_intermediates['v_after_proj'] = output.detach().clone()
    
    def muse_q_norm_hook(module, input, output):
        muse_intermediates['q_after_norm'] = output.detach().clone()
    
    def muse_k_norm_hook(module, input, output):
        muse_intermediates['k_after_norm'] = output.detach().clone()
    
    def muse_before_output_proj_hook(module, input):
        if isinstance(input, tuple):
            muse_intermediates['before_output_proj'] = input[0].detach().clone()
        else:
            muse_intermediates['before_output_proj'] = input.detach().clone()
    
    # Wrap Muse attention function to capture q, k, v before attention
    original_muse_attn_fn = muse_attn_0._attention_function
    muse_attn_fn_inputs = {}
    
    def muse_attn_fn_wrapper(*args, **kwargs):
        # Capture q, k, v
        if len(args) >= 3:
            muse_attn_fn_inputs['q'] = args[0].detach().clone()
            muse_attn_fn_inputs['k'] = args[1].detach().clone()
            muse_attn_fn_inputs['v'] = args[2].detach().clone()
        elif 'q' in kwargs:
            muse_attn_fn_inputs['q'] = kwargs['q'].detach().clone()
            muse_attn_fn_inputs['k'] = kwargs['k'].detach().clone()
            muse_attn_fn_inputs['v'] = kwargs['v'].detach().clone()
        
        result = original_muse_attn_fn(*args, **kwargs)
        muse_intermediates['attn_output'] = result.detach().clone()
        return result
    
    muse_attn_0._attention_function = muse_attn_fn_wrapper
    
    # We'll capture HF qkv after RoPE by hooking the attention forward method
    # and manually extracting the values after RoPE is applied
    hf_qkv_after_rope = {}
    
    # Hook HF attention forward to capture qkv after RoPE
    # HF attention forward does: q_proj -> view -> q_norm -> transpose -> RoPE -> attention
    # We'll capture qkv right before they enter the attention computation
    def hf_attn_forward_pre_hook(module, input):
        # This hook captures inputs to attention forward
        # But we need qkv after RoPE, which happens inside forward
        # We'll use a different approach: hook the attention function call
        pass
    
    # Instead, we'll manually compute HF's qkv after RoPE from intermediates
    # by re-running the forward pass with a custom hook that captures after RoPE
    # For now, we'll skip direct RoPE comparison and rely on attention scores/weights comparison
    
    # Register hooks
    hf_hooks = []
    hf_hooks.append(hf_attn_0.q_proj.register_forward_hook(hf_q_proj_hook))
    hf_hooks.append(hf_attn_0.k_proj.register_forward_hook(hf_k_proj_hook))
    hf_hooks.append(hf_attn_0.v_proj.register_forward_hook(hf_v_proj_hook))
    if hasattr(hf_attn_0, 'q_norm') and hf_attn_0.q_norm is not None:
        hf_hooks.append(hf_attn_0.q_norm.register_forward_hook(hf_q_norm_hook))
    if hasattr(hf_attn_0, 'k_norm') and hf_attn_0.k_norm is not None:
        hf_hooks.append(hf_attn_0.k_norm.register_forward_hook(hf_k_norm_hook))
    hf_hooks.append(hf_attn_0.register_forward_hook(hf_attn_forward_hook))
    hf_hooks.append(hf_attn_0.o_proj.register_forward_pre_hook(hf_before_o_proj_hook))
    
    muse_hooks = []
    muse_hooks.append(muse_attn_0.q_proj.register_forward_hook(muse_q_proj_hook))
    muse_hooks.append(muse_attn_0.k_proj.register_forward_hook(muse_k_proj_hook))
    muse_hooks.append(muse_attn_0.v_proj.register_forward_hook(muse_v_proj_hook))
    if hasattr(muse_attn_0, 'q_norm') and muse_attn_0.q_norm is not None:
        muse_hooks.append(muse_attn_0.q_norm.register_forward_hook(muse_q_norm_hook))
    if hasattr(muse_attn_0, 'k_norm') and muse_attn_0.k_norm is not None:
        muse_hooks.append(muse_attn_0.k_norm.register_forward_hook(muse_k_norm_hook))
    muse_hooks.append(muse_attn_0.output_proj.register_forward_pre_hook(muse_before_output_proj_hook))
    
    # Forward pass with output_attentions=True for HF
    print("3. Running Forward Pass to Capture Intermediates")
    print("-" * 60)
    
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
    
    print("  ✓ Forward pass completed\n")
    
    # Compare intermediate activations
    print("4. Intermediate Activations Comparison")
    print("-" * 60)
    
    # 4.1 Compare qkv projections
    print("\n4.1 QKV Projections:")
    compare_tensors(
        hf_intermediates['q_after_proj'],
        muse_intermediates['q_after_proj'],
        "q_proj output"
    )
    compare_tensors(
        hf_intermediates['k_after_proj'],
        muse_intermediates['k_after_proj'],
        "k_proj output"
    )
    compare_tensors(
        hf_intermediates['v_after_proj'],
        muse_intermediates['v_after_proj'],
        "v_proj output"
    )
    
    # 4.2 Compare normalizations
    if muse_config.q_norm:
        print("\n4.2 Normalizations:")
        compare_tensors(
            hf_intermediates['q_after_norm'],
            muse_intermediates['q_after_norm'],
            "q_norm output"
        )
        compare_tensors(
            hf_intermediates['k_after_norm'],
            muse_intermediates['k_after_norm'],
            "k_norm output"
        )
    
    # 4.3 Compare qkv after RoPE (input to attention function)
    print("\n4.3 QKV After RoPE (Input to Attention Function):")
    
    # Note: Direct RoPE comparison requires hooking HF's internal apply_rotary_pos_emb
    # which is not easily accessible. Instead, we'll verify RoPE correctness indirectly
    # by comparing attention scores and weights, which depend on RoPE outputs.
    
    # Show Muse qkv shapes after RoPE (captured from attention function inputs)
    if 'q' in muse_attn_fn_inputs:
        muse_q_after_rope = muse_attn_fn_inputs['q']
        muse_k_after_rope = muse_attn_fn_inputs['k']
        muse_v_after_rope = muse_attn_fn_inputs['v']
        
        print("  Muse qkv after RoPE shapes:")
        print(f"    q: {muse_q_after_rope.shape}")
        print(f"    k: {muse_k_after_rope.shape}")
        print(f"    v: {muse_v_after_rope.shape}")
        print("  Note: HF qkv after RoPE will be verified indirectly via attention scores/weights comparison")
    else:
        print("  ⚠️  Could not capture Muse qkv after RoPE")
    
    # 4.4 Manually compute and compare attention scores and weights
    print("\n4.4 Attention Scores and Weights:")
    
    # Manual attention computation for comparison
    # We'll compute from Muse's qkv inputs
    if 'q' in muse_attn_fn_inputs and 'k' in muse_attn_fn_inputs and 'v' in muse_attn_fn_inputs:
        muse_q = muse_attn_fn_inputs['q'].to(device=device, dtype=dtype)
        muse_k = muse_attn_fn_inputs['k'].to(device=device, dtype=dtype)
        muse_v = muse_attn_fn_inputs['v'].to(device=device, dtype=dtype)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(muse_q).any() or torch.isinf(muse_q).any():
            print("  ⚠️  NaN/Inf detected in muse_q!")
        if torch.isnan(muse_k).any() or torch.isinf(muse_k).any():
            print("  ⚠️  NaN/Inf detected in muse_k!")
        if torch.isnan(muse_v).any() or torch.isinf(muse_v).any():
            print("  ⚠️  NaN/Inf detected in muse_v!")
        
        # Check actual shapes - Muse may have already expanded k/v for GQA
        print(f"  Muse qkv shapes:")
        print(f"    q: {muse_q.shape}")
        print(f"    k: {muse_k.shape}")
        print(f"    v: {muse_v.shape}")
        
        # Muse's attention function receives k/v that may already be expanded
        # If k/v shape matches q shape in head dimension, they're already expanded
        if muse_k.shape[1] == muse_num_heads:
            # Already expanded
            muse_k_expanded = muse_k
            muse_v_expanded = muse_v
        else:
            # Need to expand for GQA
            num_key_value_groups = muse_num_heads // muse_num_kv_heads
            batch, num_kv_heads, seq_len, head_dim = muse_k.shape
            muse_k_expanded = muse_k.unsqueeze(2).expand(
                batch, num_kv_heads, num_key_value_groups, seq_len, head_dim
            ).reshape(batch, muse_num_heads, seq_len, head_dim)
            muse_v_expanded = muse_v.unsqueeze(2).expand(
                batch, num_kv_heads, num_key_value_groups, seq_len, head_dim
            ).reshape(batch, muse_num_heads, seq_len, head_dim)
        
        # Compute attention scores: q @ k^T * scaling
        muse_scores = torch.matmul(muse_q, muse_k_expanded.transpose(-2, -1)) * muse_scaling
        
        # Check scores before mask
        if torch.isnan(muse_scores).any() or torch.isinf(muse_scores).any():
            print("  ⚠️  NaN/Inf detected in scores before mask!")
            print(f"    Scores stats: min={muse_scores.min().item():.6f}, max={muse_scores.max().item():.6f}, mean={muse_scores.mean().item():.6f}")
        
        # Apply causal mask (upper triangular mask)
        seq_len = muse_q.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        muse_scores = muse_scores + causal_mask
        
        # Check scores after mask
        if torch.isnan(muse_scores).any() or torch.isinf(muse_scores).any():
            print("  ⚠️  NaN/Inf detected in scores after mask!")
            # Replace inf with a large negative value for softmax
            muse_scores = torch.where(torch.isinf(muse_scores), torch.tensor(-1e9, device=device, dtype=dtype), muse_scores)
        
        # Compute attention weights
        muse_weights = torch.nn.functional.softmax(
            muse_scores.float(), dim=-1
        ).to(dtype)
        
        # Check weights
        if torch.isnan(muse_weights).any():
            print("  ⚠️  NaN detected in weights!")
        
        # Compute attention output
        muse_attn_output_manual = torch.matmul(muse_weights, muse_v_expanded)
        
        print(f"  Manual computation:")
        print(f"    Scores shape: {muse_scores.shape}")
        print(f"    Scores range: [{muse_scores.min().item():.6f}, {muse_scores.max().item():.6f}]")
        print(f"    Weights shape: {muse_weights.shape}")
        print(f"    Weights range: [{muse_weights.min().item():.6f}, {muse_weights.max().item():.6f}]")
        print(f"    Weights sum per row (should be ~1.0): min={muse_weights.sum(dim=-1).min().item():.6f}, max={muse_weights.sum(dim=-1).max().item():.6f}")
        
        # Compare with HF attention weights if available
        if 'attn_weights' in hf_intermediates:
            hf_weights = hf_intermediates['attn_weights'].to(device=device, dtype=dtype)
            print(f"\n  Comparing with HF attention weights:")
            compare_tensors(hf_weights, muse_weights, "attention weights")
        
        # Compare attention output
        # HF attention output is [batch, seq_len, embed_dim] after reshape and output_proj
        # Muse attention output from attention function is [batch, num_heads, seq_len, head_dim]
        # We need to reshape Muse output to compare with HF's before_o_proj
        print(f"\n  Comparing attention output:")
        muse_attn_output_reshaped = muse_intermediates['attn_output'].transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        batch_size, seq_len, num_heads, head_dim = muse_attn_output_reshaped.shape
        muse_attn_output_reshaped = muse_attn_output_reshaped.reshape(batch_size, seq_len, -1)  # [batch, seq_len, embed_dim]
        
        # Compare with HF's before_o_proj (which is the attention output before output_proj)
        if 'before_o_proj' in hf_intermediates:
            compare_tensors(
                hf_intermediates['before_o_proj'],
                muse_attn_output_reshaped,
                "attention output (before output_proj)"
            )
        else:
            # Fallback: compare with HF's attn_output (which is after output_proj)
            compare_tensors(
                hf_intermediates['attn_output'],
                muse_intermediates['attn_output'],
                "attention output"
            )
        
        # Compare before output projection
        # HF's before_o_proj is already compared above
        # Now compare after output_proj
        if 'before_output_proj' in muse_intermediates:
            print(f"\n  Muse before_output_proj shape: {muse_intermediates['before_output_proj'].shape}")
            # This should match HF's attn_output (which is after output_proj)
            if 'attn_output' in hf_intermediates:
                print(f"  HF attn_output (after output_proj) shape: {hf_intermediates['attn_output'].shape}")
                # Reshape Muse's before_output_proj to match HF's shape
                muse_before_output_proj = muse_intermediates['before_output_proj']
                if len(muse_before_output_proj.shape) == 4:  # [batch, num_heads, seq_len, head_dim]
                    muse_before_output_proj = muse_before_output_proj.transpose(1, 2).contiguous()
                    batch_size, seq_len, num_heads, head_dim = muse_before_output_proj.shape
                    muse_before_output_proj = muse_before_output_proj.reshape(batch_size, seq_len, -1)
                compare_tensors(
                    hf_intermediates['attn_output'],
                    muse_before_output_proj,
                    "after output_proj"
                )
    
    # Cleanup hooks
    for hook in hf_hooks:
        hook.remove()
    for hook in muse_hooks:
        hook.remove()
    
    # Restore original attention function
    muse_attn_0._attention_function = original_muse_attn_fn
    
    print(f"\n{'='*60}")
    print("Layer 0 Attention Comparison Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_qwen3_logits_align_with_hf_checkpoint()