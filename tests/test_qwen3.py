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

from unittest.mock import patch

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
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
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

    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()

    muse_config = _build_qwen3_config(hf_config_dict)
    
    # Get target device and dtype from HF model
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    
    # Create Muse model with correct dtype
    model_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    with set_default_dtype(model_dtype):
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
    
    print(f"\n{'='*60}")
    print("Forward Pass & Logits Comparison")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # HF forward
        print("Running HF model forward pass...")
        hf_outputs = hf_model(**model_inputs)
        hf_logits = hf_outputs.logits
        
        # Muse forward - Muse model expects 'tokens' instead of 'input_ids'
        print("Running Muse model forward pass...")
        muse_inputs = {"tokens": model_inputs["input_ids"]}
        muse_logits = muse_model(**muse_inputs)
        
        # Ensure both logits are on same device and dtype for comparison
        hf_logits = hf_logits.to(device=device, dtype=dtype)
        muse_logits = muse_logits.to(device=device, dtype=dtype)
        
        # Compare shapes
        print(f"\nLogits shapes:")
        print(f"  HF:   {hf_logits.shape}")
        print(f"  Muse: {muse_logits.shape}")
        
        if hf_logits.shape != muse_logits.shape:
            print(f"  ⚠️  Shape mismatch!")
            # Try to align shapes if possible
            min_seq_len = min(hf_logits.shape[1], muse_logits.shape[1])
            hf_logits = hf_logits[:, :min_seq_len, :]
            muse_logits = muse_logits[:, :min_seq_len, :]
            print(f"  Using aligned shapes: HF={hf_logits.shape}, Muse={muse_logits.shape}")
        
        # Check for NaN/Inf
        hf_has_nan = torch.isnan(hf_logits).any().item()
        hf_has_inf = torch.isinf(hf_logits).any().item()
        muse_has_nan = torch.isnan(muse_logits).any().item()
        muse_has_inf = torch.isinf(muse_logits).any().item()
        
        print(f"\nNaN/Inf checks:")
        print(f"  HF:   NaN={hf_has_nan}, Inf={hf_has_inf}")
        print(f"  Muse: NaN={muse_has_nan}, Inf={muse_has_inf}")
        
        if hf_has_nan or hf_has_inf:
            print(f"  ⚠️  HF logits contain NaN/Inf!")
        if muse_has_nan or muse_has_inf:
            print(f"  ⚠️  Muse logits contain NaN/Inf!")
        
        # Calculate differences
        logits_diff = (hf_logits - muse_logits).abs()
        max_diff = logits_diff.max().item()
        mean_diff = logits_diff.mean().item()
        median_diff = logits_diff.median().item()
        
        # Calculate relative differences
        hf_abs = hf_logits.abs()
        relative_diff = logits_diff / (hf_abs + 1e-8)  # Add small epsilon to avoid division by zero
        max_relative_diff = relative_diff.max().item()
        mean_relative_diff = relative_diff.mean().item()
        
        # Calculate per-token statistics
        per_token_max_diff = logits_diff.max(dim=-1)[0]  # [b, s]
        per_token_mean_diff = logits_diff.mean(dim=-1)   # [b, s]
        
        print(f"\nLogits difference statistics:")
        print(f"  Max absolute diff:     {max_diff:.6e}")
        print(f"  Mean absolute diff:    {mean_diff:.6e}")
        print(f"  Median absolute diff:  {median_diff:.6e}")
        print(f"  Max relative diff:     {max_relative_diff:.6e}")
        print(f"  Mean relative diff:    {mean_relative_diff:.6e}")
        
        print(f"\nPer-token statistics:")
        print(f"  Max diff per token - Max: {per_token_max_diff.max().item():.6e}")
        print(f"  Max diff per token - Mean: {per_token_max_diff.mean().item():.6e}")
        print(f"  Mean diff per token - Max: {per_token_mean_diff.max().item():.6e}")
        print(f"  Mean diff per token - Mean: {per_token_mean_diff.mean().item():.6e}")
        
        # Check if differences are within acceptable tolerance
        # For bfloat16, we expect some numerical differences
        if dtype == torch.bfloat16:
            tolerance = 1e-2  # More lenient for bfloat16
        else:
            tolerance = 1e-5  # Stricter for float32
        
        print(f"\nTolerance check (tolerance={tolerance:.0e}):")
        within_tolerance = max_diff < tolerance
        print(f"  Max diff within tolerance: {within_tolerance}")
        
        if not within_tolerance:
            print(f"  ⚠️  Logits differ beyond tolerance!")
            
            # Find positions with largest differences
            flat_diff = logits_diff.flatten()
            top_k = min(10, len(flat_diff))
            top_k_values, top_k_indices = torch.topk(flat_diff, top_k)
            
            print(f"\n  Top {top_k} largest differences:")
            for i, (val, idx) in enumerate(zip(top_k_values, top_k_indices)):
                # Convert flat index to (batch, seq, vocab) coordinates
                batch_idx = idx // (muse_logits.shape[1] * muse_logits.shape[2])
                remainder = idx % (muse_logits.shape[1] * muse_logits.shape[2])
                seq_idx = remainder // muse_logits.shape[2]
                vocab_idx = remainder % muse_logits.shape[2]
                
                hf_val = hf_logits[batch_idx, seq_idx, vocab_idx].item()
                muse_val = muse_logits[batch_idx, seq_idx, vocab_idx].item()
                
                print(f"    [{i+1}] pos=({batch_idx}, {seq_idx}, {vocab_idx}), "
                      f"diff={val.item():.6e}, HF={hf_val:.6e}, Muse={muse_val:.6e}")
        else:
            print(f"  ✓ Logits match within tolerance!")
        
        # Compare predicted tokens (argmax)
        hf_predicted = hf_logits.argmax(dim=-1)  # [b, s]
        muse_predicted = muse_logits.argmax(dim=-1)  # [b, s]
        
        token_matches = (hf_predicted == muse_predicted).float()
        token_match_rate = token_matches.mean().item()
        
        print(f"\nPredicted token comparison:")
        print(f"  Token match rate: {token_match_rate*100:.2f}%")
        print(f"  Total tokens: {hf_predicted.numel()}")
        print(f"  Matching tokens: {token_matches.sum().item():.0f}")
        print(f"  Mismatching tokens: {(1 - token_matches).sum().item():.0f}")
        
        if token_match_rate < 1.0:
            # Find positions where predictions differ
            mismatch_mask = hf_predicted != muse_predicted
            mismatch_positions = torch.nonzero(mismatch_mask, as_tuple=False)
            
            print(f"\n  First 10 mismatched positions:")
            for i, pos in enumerate(mismatch_positions[:10]):
                b_idx, s_idx = pos[0].item(), pos[1].item()
                hf_token = hf_predicted[b_idx, s_idx].item()
                muse_token = muse_predicted[b_idx, s_idx].item()
                hf_logit_val = hf_logits[b_idx, s_idx, hf_token].item()
                muse_logit_val = muse_logits[b_idx, s_idx, muse_token].item()
                
                print(f"    [{i+1}] pos=({b_idx}, {s_idx}): "
                      f"HF_token={hf_token} (logit={hf_logit_val:.4f}), "
                      f"Muse_token={muse_token} (logit={muse_logit_val:.4f})")
        
        # Compare logits value ranges
        print(f"\nLogits value ranges:")
        print(f"  HF:   min={hf_logits.min().item():.4f}, max={hf_logits.max().item():.4f}, "
              f"mean={hf_logits.mean().item():.4f}, std={hf_logits.std().item():.4f}")
        print(f"  Muse: min={muse_logits.min().item():.4f}, max={muse_logits.max().item():.4f}, "
              f"mean={muse_logits.mean().item():.4f}, std={muse_logits.std().item():.4f}")
        
        print(f"\n{'='*60}\n")
        
        # Summary
        if within_tolerance and token_match_rate == 1.0:
            print("✓✓✓ SUCCESS: Logits match perfectly!")
        elif within_tolerance:
            print("✓ SUCCESS: Logits match within tolerance (some token predictions differ)")
        else:
            print("✗ FAILURE: Logits differ beyond tolerance")

def test_checkpint():
    hf_checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base"
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/muse/Qwen3-8B-Base"
    with set_default_dtype(torch.bfloat16):
        model = Qwen3Model.from_pretrained(
            checkpoint_dir, attention_function="flash_attention_2")
    
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint_dir,
        torch_dtype=torch.bfloat16,
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

    device = "cuda"

    model = model.to(device=device)

    # Ensure eager attention is used
    hf_model.config._attn_implementation = "flash_attention_2"

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

if __name__ == '__main__':
    #test_qwen3_logits_align_with_hf_checkpoint()
    test_checkpint()
