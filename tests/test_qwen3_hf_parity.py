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
    
    muse_model.eval()
    hf_model.eval()

    # Register hooks to capture intermediate activations
    hf_activations = {}
    muse_activations = {}
    
    def make_hf_hook(name):
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hf_activations[name] = output[0].detach().clone()
            else:
                hf_activations[name] = output.detach().clone()
        return hook
    
    def make_muse_hook(name):
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                muse_activations[name] = output[0].detach().clone()
            else:
                muse_activations[name] = output.detach().clone()
        return hook
    
    # Register hooks for HF model
    hf_hooks = []
    
    # Embedding layer
    if hasattr(hf_model.model, 'embed_tokens'):
        hf_hooks.append(
            hf_model.model.embed_tokens.register_forward_hook(
                make_hf_hook("embedding")
            )
        )
    
    # Transformer layers
    for i in range(len(hf_model.model.layers)):
        hf_hooks.append(
            hf_model.model.layers[i].register_forward_hook(
                make_hf_hook(f"layer_{i}")
            )
        )
    
    # Final norm
    if hasattr(hf_model.model, 'norm'):
        hf_hooks.append(
            hf_model.model.norm.register_forward_hook(
                make_hf_hook("final_norm")
            )
        )
    
    # Output layer (lm_head)
    if hasattr(hf_model, 'lm_head'):
        hf_hooks.append(
            hf_model.lm_head.register_forward_hook(
                make_hf_hook("output")
            )
        )
    
    # Register hooks for Muse model
    muse_hooks = []
    
    # Embedding layer
    if hasattr(muse_model.model, 'tok_embeddings'):
        muse_hooks.append(
            muse_model.model.tok_embeddings.register_forward_hook(
                make_muse_hook("embedding")
            )
        )
    
    # Transformer layers
    for i in range(len(muse_model.model.layers)):
        muse_hooks.append(
            muse_model.model.layers[i].register_forward_hook(
                make_muse_hook(f"layer_{i}")
            )
        )
    
    # Final norm
    if hasattr(muse_model.model, 'norm'):
        muse_hooks.append(
            muse_model.model.norm.register_forward_hook(
                make_muse_hook("final_norm")
            )
        )
    
    # Output layer - hook the TransformerDecoder's forward output (which includes unembed)
    # This captures the logits after unembed, matching HF's lm_head output
    # We'll compare this separately, so we hook it with a different name
    if hasattr(muse_model.model, 'output'):
        # Hook the output layer's forward call
        # Note: unembed calls self.output(h), so we hook output directly
        from muse.layers.linear import TiedLinear
        if isinstance(muse_model.model.output, TiedLinear):
            # For TiedLinear, hook the linear operation
            muse_hooks.append(
                muse_model.model.output.linear.register_forward_hook(
                    make_muse_hook("output_before_float")
                )
            )
        else:
            muse_hooks.append(
                muse_model.model.output.register_forward_hook(
                    make_muse_hook("output_before_float")
                )
            )
    
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
    
    # Remove hooks
    for hook in hf_hooks:
        hook.remove()
    for hook in muse_hooks:
        hook.remove()
    
    # Compare activations layer by layer
    print(f"\n{'='*60}")
    print("Layer-by-Layer Activation Comparison")
    print(f"{'='*60}")
    
    # Compare embedding
    if "embedding" in hf_activations and "embedding" in muse_activations:
        hf_emb = hf_activations["embedding"].to(device=device, dtype=dtype)
        muse_emb = muse_activations["embedding"].to(device=device, dtype=dtype)
        emb_diff = (hf_emb - muse_emb).abs()
        max_diff = emb_diff.max().item()
        mean_diff = emb_diff.mean().item()
        print(f"\nEmbedding output:")
        print(f"  Shape: HF={hf_emb.shape}, Muse={muse_emb.shape}")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        if max_diff > 1e-4:
            print(f"  ⚠️  Large difference detected!")
        else:
            print(f"  ✓ Match!")
    
    # Compare transformer layers
    first_mismatch_layer = None
    for i in range(muse_config.num_layers):
        hf_key = f"layer_{i}"
        muse_key = f"layer_{i}"
        if hf_key in hf_activations and muse_key in muse_activations:
            hf_act = hf_activations[hf_key].to(device=device, dtype=dtype)
            muse_act = muse_activations[muse_key].to(device=device, dtype=dtype)
            act_diff = (hf_act - muse_act).abs()
            max_diff = act_diff.max().item()
            mean_diff = act_diff.mean().item()
            relative_diff = (act_diff / (hf_act.abs() + 1e-8)).max().item()
            
            status = "✓ Match!" if max_diff < 1e-4 else "⚠️  Mismatch!"
            print(f"\nLayer {i} output:")
            print(f"  Shape: HF={hf_act.shape}, Muse={muse_act.shape}")
            print(f"  Max diff: {max_diff:.6e}")
            print(f"  Mean diff: {mean_diff:.6e}")
            print(f"  Max relative diff: {relative_diff:.6e}")
            print(f"  {status}")
            
            if max_diff > 1e-4 and first_mismatch_layer is None:
                first_mismatch_layer = i
                print(f"  ⚠️  First mismatch detected at layer {i}!")
                
                # For the first mismatched layer, do detailed debugging
                if i == 0:
                    print(f"\n{'='*60}")
                    print(f"Detailed Debugging for Layer {i}")
                    print(f"{'='*60}")
                    
                    # Check layer input (should match previous layer output or embedding)
                    if i == 0 and "embedding" in hf_activations and "embedding" in muse_activations:
                        hf_layer_input = hf_activations["embedding"].to(device=device, dtype=dtype)
                        muse_layer_input = muse_activations["embedding"].to(device=device, dtype=dtype)
                        input_diff = (hf_layer_input - muse_layer_input).abs()
                        print(f"\nLayer {i} input (embedding output):")
                        print(f"  Max diff: {input_diff.max().item():.6e}")
                        print(f"  Mean diff: {input_diff.mean().item():.6e}")
                        if input_diff.max().item() > 1e-4:
                            print(f"  ⚠️  Layer input already mismatches!")
                        else:
                            print(f"  ✓ Layer input matches")
                    
                    # Register detailed hooks for first layer submodules
                    hf_layer_0 = hf_model.model.layers[0]
                    muse_layer_0 = muse_model.model.layers[0]
                    
                    hf_layer_0_activations = {}
                    muse_layer_0_activations = {}
                    
                    def make_detailed_hf_hook(name):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                hf_layer_0_activations[name] = output[0].detach().clone()
                            else:
                                hf_layer_0_activations[name] = output.detach().clone()
                        return hook
                    
                    def make_detailed_muse_hook(name):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                muse_layer_0_activations[name] = output[0].detach().clone()
                            else:
                                muse_layer_0_activations[name] = output.detach().clone()
                        return hook
                    
                    detailed_hf_hooks = []
                    detailed_muse_hooks = []
                    
                    # Hook HF layer 0 submodules
                    if hasattr(hf_layer_0, 'input_layernorm'):
                        detailed_hf_hooks.append(
                            hf_layer_0.input_layernorm.register_forward_hook(
                                make_detailed_hf_hook("sa_norm")
                            )
                        )
                    if hasattr(hf_layer_0, 'self_attn'):
                        detailed_hf_hooks.append(
                            hf_layer_0.self_attn.register_forward_hook(
                                make_detailed_hf_hook("attention")
                            )
                        )
                    if hasattr(hf_layer_0, 'post_attention_layernorm'):
                        detailed_hf_hooks.append(
                            hf_layer_0.post_attention_layernorm.register_forward_hook(
                                make_detailed_hf_hook("mlp_norm")
                            )
                        )
                    if hasattr(hf_layer_0, 'mlp'):
                        detailed_hf_hooks.append(
                            hf_layer_0.mlp.register_forward_hook(
                                make_detailed_hf_hook("mlp")
                            )
                        )
                    
                    # Hook Muse layer 0 submodules
                    if hasattr(muse_layer_0, 'sa_norm'):
                        detailed_muse_hooks.append(
                            muse_layer_0.sa_norm.register_forward_hook(
                                make_detailed_muse_hook("sa_norm")
                            )
                        )
                    if hasattr(muse_layer_0, 'attn'):
                        detailed_muse_hooks.append(
                            muse_layer_0.attn.register_forward_hook(
                                make_detailed_muse_hook("attention")
                            )
                        )
                    if hasattr(muse_layer_0, 'mlp_norm'):
                        detailed_muse_hooks.append(
                            muse_layer_0.mlp_norm.register_forward_hook(
                                make_detailed_muse_hook("mlp_norm")
                            )
                        )
                    if hasattr(muse_layer_0, 'mlp'):
                        detailed_muse_hooks.append(
                            muse_layer_0.mlp.register_forward_hook(
                                make_detailed_muse_hook("mlp")
                            )
                        )
                    
                    # Re-run forward pass through the full model to capture submodule activations
                    # This ensures all necessary setup (like position embeddings) is done
                    with torch.no_grad():
                        # Clear previous activations
                        hf_layer_0_activations.clear()
                        muse_layer_0_activations.clear()
                        
                        # Run full forward pass - hooks will capture layer 0 submodules
                        _ = hf_model(**model_inputs)
                        _ = muse_model(tokens=muse_tokens)
                    
                    # Remove detailed hooks
                    for hook in detailed_hf_hooks:
                        hook.remove()
                    for hook in detailed_muse_hooks:
                        hook.remove()
                    
                    # Compare submodule outputs
                    submodules_to_check = ["sa_norm", "attention", "mlp_norm", "mlp"]
                    for submod_name in submodules_to_check:
                        if submod_name in hf_layer_0_activations and submod_name in muse_layer_0_activations:
                            hf_submod = hf_layer_0_activations[submod_name].to(device=device, dtype=dtype)
                            muse_submod = muse_layer_0_activations[submod_name].to(device=device, dtype=dtype)
                            submod_diff = (hf_submod - muse_submod).abs()
                            max_submod_diff = submod_diff.max().item()
                            mean_submod_diff = submod_diff.mean().item()
                            print(f"\n  Layer {i} {submod_name}:")
                            print(f"    Shape: HF={hf_submod.shape}, Muse={muse_submod.shape}")
                            print(f"    Max diff: {max_submod_diff:.6e}")
                            print(f"    Mean diff: {mean_submod_diff:.6e}")
                            if max_submod_diff > 1e-4:
                                print(f"    ⚠️  Mismatch!")
                                
                                # If attention module mismatches, debug its internal steps
                                if submod_name == "attention":
                                    print(f"\n    {'-'*50}")
                                    print(f"    Debugging Attention Module Internals")
                                    print(f"    {'-'*50}")
                                    
                                    # Hook attention submodules
                                    hf_attn_0 = hf_layer_0.self_attn
                                    muse_attn_0 = muse_layer_0.attn
                                    
                                    hf_attn_internals = {}
                                    muse_attn_internals = {}
                                    
                                    def make_attn_hf_hook(name):
                                        def hook(module, input, output):
                                            if isinstance(output, tuple):
                                                hf_attn_internals[name] = output[0].detach().clone()
                                            else:
                                                hf_attn_internals[name] = output.detach().clone()
                                        return hook
                                    
                                    def make_attn_muse_hook(name):
                                        def hook(module, input, output):
                                            if isinstance(output, tuple):
                                                muse_attn_internals[name] = output[0].detach().clone()
                                            else:
                                                muse_attn_internals[name] = output.detach().clone()
                                        return hook
                                    
                                    attn_hf_hooks = []
                                    attn_muse_hooks = []
                                    
                                    # Hook attention projections
                                    if hasattr(hf_attn_0, 'q_proj'):
                                        attn_hf_hooks.append(
                                            hf_attn_0.q_proj.register_forward_hook(
                                                make_attn_hf_hook("q_proj")
                                            )
                                        )
                                    if hasattr(hf_attn_0, 'k_proj'):
                                        attn_hf_hooks.append(
                                            hf_attn_0.k_proj.register_forward_hook(
                                                make_attn_hf_hook("k_proj")
                                            )
                                        )
                                    if hasattr(hf_attn_0, 'v_proj'):
                                        attn_hf_hooks.append(
                                            hf_attn_0.v_proj.register_forward_hook(
                                                make_attn_hf_hook("v_proj")
                                            )
                                        )
                                    if hasattr(hf_attn_0, 'o_proj'):
                                        attn_hf_hooks.append(
                                            hf_attn_0.o_proj.register_forward_hook(
                                                make_attn_hf_hook("output_proj")
                                            )
                                        )
                                    
                                    # Hook Muse attention projections
                                    if hasattr(muse_attn_0, 'q_proj'):
                                        attn_muse_hooks.append(
                                            muse_attn_0.q_proj.register_forward_hook(
                                                make_attn_muse_hook("q_proj")
                                            )
                                        )
                                    if hasattr(muse_attn_0, 'k_proj'):
                                        attn_muse_hooks.append(
                                            muse_attn_0.k_proj.register_forward_hook(
                                                make_attn_muse_hook("k_proj")
                                            )
                                        )
                                    if hasattr(muse_attn_0, 'v_proj'):
                                        attn_muse_hooks.append(
                                            muse_attn_0.v_proj.register_forward_hook(
                                                make_attn_muse_hook("v_proj")
                                            )
                                        )
                                    if hasattr(muse_attn_0, 'output_proj'):
                                        attn_muse_hooks.append(
                                            muse_attn_0.output_proj.register_forward_hook(
                                                make_attn_muse_hook("output_proj")
                                            )
                                        )
                                    
                                    # Hook q_norm and k_norm if they exist
                                    if hasattr(hf_attn_0, 'q_norm') and hf_attn_0.q_norm is not None:
                                        attn_hf_hooks.append(
                                            hf_attn_0.q_norm.register_forward_hook(
                                                make_attn_hf_hook("q_norm")
                                            )
                                        )
                                    if hasattr(hf_attn_0, 'k_norm') and hf_attn_0.k_norm is not None:
                                        attn_hf_hooks.append(
                                            hf_attn_0.k_norm.register_forward_hook(
                                                make_attn_hf_hook("k_norm")
                                            )
                                        )
                                    
                                    if hasattr(muse_attn_0, 'q_norm') and muse_attn_0.q_norm is not None:
                                        attn_muse_hooks.append(
                                            muse_attn_0.q_norm.register_forward_hook(
                                                make_attn_muse_hook("q_norm")
                                            )
                                        )
                                    if hasattr(muse_attn_0, 'k_norm') and muse_attn_0.k_norm is not None:
                                        attn_muse_hooks.append(
                                            muse_attn_0.k_norm.register_forward_hook(
                                                make_attn_muse_hook("k_norm")
                                            )
                                        )
                                    
                                    # Re-run forward pass to capture attention internals
                                    with torch.no_grad():
                                        hf_attn_internals.clear()
                                        muse_attn_internals.clear()
                                        _ = hf_model(**model_inputs)
                                        _ = muse_model(tokens=muse_tokens)
                                    
                                    # Remove hooks
                                    for hook in attn_hf_hooks:
                                        hook.remove()
                                    for hook in attn_muse_hooks:
                                        hook.remove()
                                    
                                    # Compare attention internals
                                    attn_steps = ["q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "output_proj"]
                                    for step_name in attn_steps:
                                        if step_name in hf_attn_internals and step_name in muse_attn_internals:
                                            hf_step = hf_attn_internals[step_name].to(device=device, dtype=dtype)
                                            muse_step = muse_attn_internals[step_name].to(device=device, dtype=dtype)
                                            
                                            print(f"\n      Attention {step_name}:")
                                            print(f"        Shape: HF={hf_step.shape}, Muse={muse_step.shape}")
                                            
                                            # Check if shapes match before comparing
                                            if hf_step.shape != muse_step.shape:
                                                print(f"        ⚠️  Shape mismatch! Attempting to reshape for comparison.")
                                                print(f"        Note: This may be due to different tensor layouts.")
                                                
                                                # Try to reshape to match shapes
                                                hf_reshaped = None
                                                muse_reshaped = None
                                                
                                                # For q_norm/k_norm, shapes might be:
                                                # HF: [batch, seq_len, num_heads, head_dim]
                                                # Muse: [batch, num_heads, seq_len, head_dim]
                                                if step_name in ["q_norm", "k_norm"]:
                                                    if len(hf_step.shape) == 4 and len(muse_step.shape) == 4:
                                                        # Check if it's a transpose issue
                                                        if (hf_step.shape[0] == muse_step.shape[0] and
                                                            hf_step.shape[1] == muse_step.shape[2] and
                                                            hf_step.shape[2] == muse_step.shape[1] and
                                                            hf_step.shape[3] == muse_step.shape[3]):
                                                            # HF: [b, s, h, d] -> Muse: [b, h, s, d]
                                                            hf_reshaped = hf_step.transpose(1, 2)
                                                            muse_reshaped = muse_step
                                                            print(f"        Reshaped: HF {hf_step.shape} -> {hf_reshaped.shape} (transpose(1,2))")
                                                        elif (hf_step.shape[0] == muse_step.shape[0] and
                                                              hf_step.shape[2] == muse_step.shape[1] and
                                                              hf_step.shape[1] == muse_step.shape[2] and
                                                              hf_step.shape[3] == muse_step.shape[3]):
                                                            # Muse: [b, h, s, d] -> HF: [b, s, h, d]
                                                            muse_reshaped = muse_step.transpose(1, 2)
                                                            hf_reshaped = hf_step
                                                            print(f"        Reshaped: Muse {muse_step.shape} -> {muse_reshaped.shape} (transpose(1,2))")
                                                
                                                # If reshaped successfully, compare
                                                if hf_reshaped is not None and muse_reshaped is not None:
                                                    if hf_reshaped.shape == muse_reshaped.shape:
                                                        step_diff = (hf_reshaped - muse_reshaped).abs()
                                                        max_step_diff = step_diff.max().item()
                                                        mean_step_diff = step_diff.mean().item()
                                                        print(f"        Max diff (after reshape): {max_step_diff:.6e}")
                                                        print(f"        Mean diff (after reshape): {mean_step_diff:.6e}")
                                                        if max_step_diff > 1e-4:
                                                            print(f"        ⚠️  Mismatch after reshape!")
                                                        else:
                                                            print(f"        ✓ Match after reshape!")
                                                    else:
                                                        print(f"        Could not match shapes even after transpose")
                                                else:
                                                    # Fallback: flatten and compare if total elements match
                                                    try:
                                                        hf_flat = hf_step.flatten()
                                                        muse_flat = muse_step.flatten()
                                                        if hf_flat.shape == muse_flat.shape:
                                                            step_diff = (hf_flat - muse_flat).abs()
                                                            max_step_diff = step_diff.max().item()
                                                            mean_step_diff = step_diff.mean().item()
                                                            print(f"        Flattened comparison:")
                                                            print(f"          Max diff: {max_step_diff:.6e}")
                                                            print(f"          Mean diff: {mean_step_diff:.6e}")
                                                            if max_step_diff > 1e-4:
                                                                print(f"          ⚠️  Mismatch!")
                                                        else:
                                                            print(f"        Total elements: HF={hf_flat.numel()}, Muse={muse_flat.numel()}")
                                                    except Exception as e:
                                                        print(f"        Could not reshape for comparison: {e}")
                                            else:
                                                step_diff = (hf_step - muse_step).abs()
                                                max_step_diff = step_diff.max().item()
                                                mean_step_diff = step_diff.mean().item()
                                                print(f"        Max diff: {max_step_diff:.6e}")
                                                print(f"        Mean diff: {mean_step_diff:.6e}")
                                                if max_step_diff > 1e-4:
                                                    print(f"        ⚠️  Mismatch!")
                                                else:
                                                    print(f"        ✓ Match!")
                                    
                                    # Comprehensive step-by-step attention debugging
                                    if step_name == "output_proj" and max_step_diff > 1e-4:
                                        print(f"\n      {'='*60}")
                                        print(f"      Comprehensive Step-by-Step Attention Debugging")
                                        print(f"      {'='*60}")
                                        
                                        # Store all intermediate values
                                        hf_intermediates = {}
                                        muse_intermediates = {}
                                        
                                        # Hook all intermediate steps for Muse
                                        muse_hooks = []
                                        
                                        # 1. Hook q after q_proj
                                        def muse_q_proj_hook(module, input, output):
                                            muse_intermediates['q_after_proj'] = output.detach().clone()
                                        muse_hooks.append(muse_attn_0.q_proj.register_forward_hook(muse_q_proj_hook))
                                        
                                        # 2. Hook k after k_proj
                                        def muse_k_proj_hook(module, input, output):
                                            muse_intermediates['k_after_proj'] = output.detach().clone()
                                        muse_hooks.append(muse_attn_0.k_proj.register_forward_hook(muse_k_proj_hook))
                                        
                                        # 3. Hook v after v_proj
                                        def muse_v_proj_hook(module, input, output):
                                            muse_intermediates['v_after_proj'] = output.detach().clone()
                                        muse_hooks.append(muse_attn_0.v_proj.register_forward_hook(muse_v_proj_hook))
                                        
                                        # 4. Hook q after q_norm (if exists)
                                        if hasattr(muse_attn_0, 'q_norm') and muse_attn_0.q_norm is not None:
                                            def muse_q_norm_hook(module, input, output):
                                                muse_intermediates['q_after_norm'] = output.detach().clone()
                                            muse_hooks.append(muse_attn_0.q_norm.register_forward_hook(muse_q_norm_hook))
                                        
                                        # 5. Hook k after k_norm (if exists)
                                        if hasattr(muse_attn_0, 'k_norm') and muse_attn_0.k_norm is not None:
                                            def muse_k_norm_hook(module, input, output):
                                                muse_intermediates['k_after_norm'] = output.detach().clone()
                                            muse_hooks.append(muse_attn_0.k_norm.register_forward_hook(muse_k_norm_hook))
                                        
                                        # 6. Hook RoPE (pos_embeddings) outputs
                                        if hasattr(muse_attn_0, 'pos_embeddings') and muse_attn_0.pos_embeddings is not None:
                                            def muse_rope_q_hook(module, input, output):
                                                muse_intermediates['q_after_rope'] = output.detach().clone()
                                            def muse_rope_k_hook(module, input, output):
                                                muse_intermediates['k_after_rope'] = output.detach().clone()
                                            # Note: RoPE might be called multiple times, we'll hook it
                                            # For now, we'll capture after all processing
                                        
                                        # 7. Hook attention function inputs (q, k, v before attention)
                                        # We need to wrap the attention function to capture inputs
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
                                            
                                            # Call original function
                                            result = original_muse_attn_fn(*args, **kwargs)
                                            muse_intermediates['attn_output'] = result.detach().clone()
                                            return result
                                        
                                        muse_attn_0._attention_function = muse_attn_fn_wrapper
                                        
                                        # 8. Hook output before output_proj
                                        def muse_before_output_proj_hook(module, input):
                                            # forward_pre_hook only receives (module, input), not output
                                            if isinstance(input, tuple):
                                                muse_intermediates['before_output_proj'] = input[0].detach().clone()
                                            else:
                                                muse_intermediates['before_output_proj'] = input.detach().clone()
                                        muse_hooks.append(muse_attn_0.output_proj.register_forward_pre_hook(muse_before_output_proj_hook))
                                        
                                        # For HF, hook similar steps
                                        hf_hooks = []
                                        
                                        # Hook HF attention module's forward to capture intermediates
                                        def hf_attn_forward_hook(module, input, output):
                                            # HF returns (hidden_states, attn_weights) tuple
                                            # attn_weights may be None if not requested
                                            if isinstance(output, tuple):
                                                hf_intermediates['attn_output'] = output[0].detach().clone()
                                                if len(output) > 1 and output[1] is not None:
                                                    hf_intermediates['attn_weights'] = output[1].detach().clone()
                                            else:
                                                hf_intermediates['attn_output'] = output.detach().clone()
                                        # hf_attn_0 is already the attention module, not a layer containing it
                                        hf_hooks.append(hf_attn_0.register_forward_hook(hf_attn_forward_hook))
                                        
                                        # Hook HF q_proj, k_proj, v_proj
                                        # Check if attributes exist (HF uses different naming)
                                        if hasattr(hf_attn_0, 'q_proj'):
                                            def hf_q_proj_hook(module, input, output):
                                                hf_intermediates['q_after_proj'] = output.detach().clone()
                                            hf_hooks.append(hf_attn_0.q_proj.register_forward_hook(hf_q_proj_hook))
                                        
                                        if hasattr(hf_attn_0, 'k_proj'):
                                            def hf_k_proj_hook(module, input, output):
                                                hf_intermediates['k_after_proj'] = output.detach().clone()
                                            hf_hooks.append(hf_attn_0.k_proj.register_forward_hook(hf_k_proj_hook))
                                        
                                        if hasattr(hf_attn_0, 'v_proj'):
                                            def hf_v_proj_hook(module, input, output):
                                                hf_intermediates['v_after_proj'] = output.detach().clone()
                                            hf_hooks.append(hf_attn_0.v_proj.register_forward_hook(hf_v_proj_hook))
                                        
                                        # Hook HF q_norm, k_norm
                                        if hasattr(hf_attn_0, 'q_norm') and hf_attn_0.q_norm is not None:
                                            def hf_q_norm_hook(module, input, output):
                                                hf_intermediates['q_after_norm'] = output.detach().clone()
                                            hf_hooks.append(hf_attn_0.q_norm.register_forward_hook(hf_q_norm_hook))
                                        
                                        if hasattr(hf_attn_0, 'k_norm') and hf_attn_0.k_norm is not None:
                                            def hf_k_norm_hook(module, input, output):
                                                hf_intermediates['k_after_norm'] = output.detach().clone()
                                            hf_hooks.append(hf_attn_0.k_norm.register_forward_hook(hf_k_norm_hook))
                                        
                                        # Hook HF output_proj input (HF uses o_proj)
                                        hf_output_proj_module = None
                                        if hasattr(hf_attn_0, 'o_proj'):
                                            hf_output_proj_module = hf_attn_0.o_proj
                                        elif hasattr(hf_attn_0, 'output_proj'):
                                            hf_output_proj_module = hf_attn_0.output_proj
                                        
                                        if hf_output_proj_module is not None:
                                            def hf_before_output_proj_hook(module, input):
                                                # forward_pre_hook only receives (module, input), not output
                                                if isinstance(input, tuple):
                                                    hf_intermediates['before_output_proj'] = input[0].detach().clone()
                                                else:
                                                    hf_intermediates['before_output_proj'] = input.detach().clone()
                                            hf_hooks.append(hf_output_proj_module.register_forward_pre_hook(hf_before_output_proj_hook))
                                        
                                        # Re-run forward pass
                                        with torch.no_grad():
                                            hf_intermediates.clear()
                                            muse_intermediates.clear()
                                            muse_attn_fn_inputs.clear()
                                            _ = hf_model(**model_inputs)
                                            _ = muse_model(tokens=muse_tokens)
                                        
                                        # Remove hooks
                                        for hook in muse_hooks + hf_hooks:
                                            hook.remove()
                                        muse_attn_0._attention_function = original_muse_attn_fn
                                        
                                        # Compare all intermediate steps
                                        steps_to_compare = [
                                            ('q_after_proj', 'q after q_proj'),
                                            ('k_after_proj', 'k after k_proj'),
                                            ('v_after_proj', 'v after v_proj'),
                                            ('q_after_norm', 'q after q_norm'),
                                            ('k_after_norm', 'k after k_norm'),
                                            ('before_output_proj', 'before output_proj'),
                                            ('attn_output', 'attention output'),
                                        ]
                                        
                                        print(f"\n        Step-by-Step Comparison:")
                                        for key, description in steps_to_compare:
                                            if key in hf_intermediates and key in muse_intermediates:
                                                hf_val = hf_intermediates[key].to(device=device, dtype=dtype)
                                                muse_val = muse_intermediates[key].to(device=device, dtype=dtype)
                                                
                                                print(f"\n          {description}:")
                                                print(f"            Shape: HF={hf_val.shape}, Muse={muse_val.shape}")
                                                
                                                if hf_val.shape == muse_val.shape:
                                                    diff = (hf_val - muse_val).abs()
                                                    max_diff = diff.max().item()
                                                    mean_diff = diff.mean().item()
                                                    print(f"            Max diff: {max_diff:.6e}")
                                                    print(f"            Mean diff: {mean_diff:.6e}")
                                                    if max_diff > 1e-4:
                                                        print(f"            ⚠️  Mismatch!")
                                                    else:
                                                        print(f"            ✓ Match!")
                                                else:
                                                    print(f"            ⚠️  Shape mismatch!")
                                                    # Try transpose for q_norm/k_norm
                                                    if key in ['q_after_norm', 'k_after_norm']:
                                                        if (len(hf_val.shape) == 4 and len(muse_val.shape) == 4 and
                                                            hf_val.shape[0] == muse_val.shape[0] and
                                                            hf_val.shape[1] == muse_val.shape[2] and
                                                            hf_val.shape[2] == muse_val.shape[1] and
                                                            hf_val.shape[3] == muse_val.shape[3]):
                                                            hf_val_t = hf_val.transpose(1, 2)
                                                            diff = (hf_val_t - muse_val).abs()
                                                            max_diff = diff.max().item()
                                                            mean_diff = diff.mean().item()
                                                            print(f"            After transpose: Max diff={max_diff:.6e}, Mean diff={mean_diff:.6e}")
                                        
                                        # Check attention function inputs
                                        if 'q' in muse_attn_fn_inputs:
                                            print(f"\n          Attention function inputs (q, k, v):")
                                            for tensor_name in ['q', 'k', 'v']:
                                                if tensor_name in muse_attn_fn_inputs:
                                                    muse_tensor = muse_attn_fn_inputs[tensor_name].to(device=device, dtype=dtype)
                                                    print(f"            Muse {tensor_name}: shape={muse_tensor.shape}")
                                            
                                            # Manually compute attention scores and weights for comparison
                                            if 'q' in muse_attn_fn_inputs and 'k' in muse_attn_fn_inputs:
                                                muse_q = muse_attn_fn_inputs['q'].to(device=device, dtype=dtype)
                                                muse_k = muse_attn_fn_inputs['k'].to(device=device, dtype=dtype)
                                                
                                                print(f"\n          Manual attention computation:")
                                                # Compute scores: q @ k^T / sqrt(head_dim)
                                                head_dim = muse_q.shape[-1]
                                                muse_scores = torch.matmul(muse_q, muse_k.transpose(-2, -1)) / (head_dim ** 0.5)
                                                print(f"            Scores shape: {muse_scores.shape}")
                                                print(f"            Scores range: [{muse_scores.min().item():.4f}, {muse_scores.max().item():.4f}]")
                                                
                                                # Apply causal mask
                                                seq_len = muse_scores.shape[-1]
                                                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
                                                muse_scores_masked = muse_scores.masked_fill(causal_mask, float('-inf'))
                                                
                                                # Compute weights
                                                muse_weights = torch.nn.functional.softmax(muse_scores_masked, dim=-1)
                                                print(f"            Weights shape: {muse_weights.shape}")
                                                print(f"            Weights sum (should be ~seq_len): {muse_weights.sum(dim=-1).mean().item():.4f}")
                                                
                                                # Compare with HF attn_weights if available
                                                if 'attn_weights' in hf_intermediates:
                                                    hf_weights = hf_intermediates['attn_weights'].to(device=device, dtype=dtype)
                                                    print(f"\n            Comparing attention weights:")
                                                    print(f"              HF shape: {hf_weights.shape}, Muse shape: {muse_weights.shape}")
                                                    if hf_weights.shape == muse_weights.shape:
                                                        weights_diff = (hf_weights - muse_weights).abs()
                                                        max_weights_diff = weights_diff.max().item()
                                                        mean_weights_diff = weights_diff.mean().item()
                                                        print(f"              Max diff: {max_weights_diff:.6e}")
                                                        print(f"              Mean diff: {mean_weights_diff:.6e}")
                                                        if max_weights_diff > 1e-4:
                                                            print(f"              ⚠️  Attention weights mismatch!")
                                                        else:
                                                            print(f"              ✓ Attention weights match!")
                                        
                                        print(f"\n      {'='*60}\n")
                                        
                                        # Compare attention output before output_proj
                                        if 'output' in hf_attn_before_output_proj and 'output' in muse_attn_before_output_proj:
                                            hf_attn_out = hf_attn_before_output_proj['output'].to(device=device, dtype=dtype)
                                            muse_attn_out = muse_attn_before_output_proj['output'].to(device=device, dtype=dtype)
                                            
                                            print(f"\n        Attention output (before output_proj):")
                                            print(f"          Shape: HF={hf_attn_out.shape}, Muse={muse_attn_out.shape}")
                                            
                                            if hf_attn_out.shape == muse_attn_out.shape:
                                                attn_out_diff = (hf_attn_out - muse_attn_out).abs()
                                                max_attn_out_diff = attn_out_diff.max().item()
                                                mean_attn_out_diff = attn_out_diff.mean().item()
                                                print(f"          Max diff: {max_attn_out_diff:.6e}")
                                                print(f"          Mean diff: {mean_attn_out_diff:.6e}")
                                                if max_attn_out_diff > 1e-4:
                                                    print(f"          ⚠️  Mismatch! Problem is in attention computation itself.")
                                                else:
                                                    print(f"          ✓ Match! Problem is in output_proj.")
                                                    
                                                    # Deep dive into output_proj
                                                    print(f"\n        {'='*50}")
                                                    print(f"        Deep Dive: output_proj Analysis")
                                                    print(f"        {'='*50}")
                                                    
                                                    # Get output_proj weights
                                                    hf_output_proj_weight = None
                                                    muse_output_proj_weight = None
                                                    
                                                    if hasattr(hf_attn_0, 'o_proj'):
                                                        hf_output_proj_weight = hf_attn_0.o_proj.weight.data.to(device=device, dtype=dtype)
                                                    elif hasattr(hf_attn_0, 'output_proj'):
                                                        hf_output_proj_weight = hf_attn_0.output_proj.weight.data.to(device=device, dtype=dtype)
                                                    
                                                    if hasattr(muse_attn_0, 'output_proj'):
                                                        muse_output_proj_weight = muse_attn_0.output_proj.weight.data.to(device=device, dtype=dtype)
                                                    
                                                    if hf_output_proj_weight is not None and muse_output_proj_weight is not None:
                                                        print(f"\n          output_proj weights:")
                                                        print(f"            Shape: HF={hf_output_proj_weight.shape}, Muse={muse_output_proj_weight.shape}")
                                                        
                                                        if hf_output_proj_weight.shape == muse_output_proj_weight.shape:
                                                            weight_diff = (hf_output_proj_weight - muse_output_proj_weight).abs()
                                                            max_weight_diff = weight_diff.max().item()
                                                            mean_weight_diff = weight_diff.mean().item()
                                                            print(f"            Max diff: {max_weight_diff:.6e}")
                                                            print(f"            Mean diff: {mean_weight_diff:.6e}")
                                                            if max_weight_diff > 1e-5:
                                                                print(f"            ⚠️  Weight mismatch!")
                                                            else:
                                                                print(f"            ✓ Weights match!")
                                                        
                                                        # Manually compute output_proj to verify
                                                        print(f"\n          Manual output_proj computation:")
                                                        
                                                        # HF: o_proj(input)
                                                        hf_manual_out = None
                                                        if hasattr(hf_attn_0, 'o_proj'):
                                                            hf_manual_out = torch.nn.functional.linear(
                                                                hf_attn_out, hf_output_proj_weight
                                                            )
                                                        elif hasattr(hf_attn_0, 'output_proj'):
                                                            hf_manual_out = torch.nn.functional.linear(
                                                                hf_attn_out, hf_output_proj_weight
                                                            )
                                                        
                                                        # Muse: output_proj(input)
                                                        muse_manual_out = None
                                                        if hasattr(muse_attn_0, 'output_proj'):
                                                            muse_manual_out = torch.nn.functional.linear(
                                                                muse_attn_out, muse_output_proj_weight
                                                            )
                                                        
                                                        if hf_manual_out is not None and muse_manual_out is not None:
                                                            print(f"            Manual output shape: HF={hf_manual_out.shape}, Muse={muse_manual_out.shape}")
                                                            
                                                            if hf_manual_out.shape == muse_manual_out.shape:
                                                                manual_diff = (hf_manual_out - muse_manual_out).abs()
                                                                max_manual_diff = manual_diff.max().item()
                                                                mean_manual_diff = manual_diff.mean().item()
                                                                print(f"            Max diff: {max_manual_diff:.6e}")
                                                                print(f"            Mean diff: {mean_manual_diff:.6e}")
                                                                
                                                                # Compare with actual output_proj output
                                                                if 'output_proj' in hf_attn_internals and 'output_proj' in muse_attn_internals:
                                                                    hf_actual_out = hf_attn_internals['output_proj'].to(device=device, dtype=dtype)
                                                                    muse_actual_out = muse_attn_internals['output_proj'].to(device=device, dtype=dtype)
                                                                    
                                                                    hf_vs_manual = (hf_actual_out - hf_manual_out).abs()
                                                                    muse_vs_manual = (muse_actual_out - muse_manual_out).abs()
                                                                    
                                                                    print(f"\n            Verification:")
                                                                    print(f"              HF actual vs manual: max={hf_vs_manual.max().item():.6e}")
                                                                    print(f"              Muse actual vs manual: max={muse_vs_manual.max().item():.6e}")
                                                                    
                                                                    if hf_vs_manual.max().item() < 1e-5 and muse_vs_manual.max().item() < 1e-5:
                                                                        print(f"              ✓ Manual computation matches actual output")
                                                                        print(f"              → Problem is likely in input shape/format or weight loading")
                                                                    else:
                                                                        print(f"              ⚠️  Manual computation differs from actual output")
                                                                        print(f"              → Problem is in output_proj implementation")
                                            else:
                                                print(f"          ⚠️  Shape mismatch!")
                                                # Try to reshape if possible
                                                if (hf_attn_out.numel() == muse_attn_out.numel()):
                                                    hf_flat = hf_attn_out.flatten()
                                                    muse_flat = muse_attn_out.flatten()
                                                    flat_diff = (hf_flat - muse_flat).abs()
                                                    print(f"          Flattened comparison:")
                                                    print(f"            Max diff: {flat_diff.max().item():.6e}")
                                                    print(f"            Mean diff: {flat_diff.mean().item():.6e}")
                                        
                                        print(f"      {'-'*50}\n")
                                    
                                    print(f"    {'-'*50}\n")
                            else:
                                print(f"    ✓ Match!")
                    
                    print(f"{'='*60}\n")
    
    # Compare final norm
    if "final_norm" in hf_activations and "final_norm" in muse_activations:
        hf_norm = hf_activations["final_norm"].to(device=device, dtype=dtype)
        muse_norm = muse_activations["final_norm"].to(device=device, dtype=dtype)
        norm_diff = (hf_norm - muse_norm).abs()
        max_diff = norm_diff.max().item()
        mean_diff = norm_diff.mean().item()
        print(f"\nFinal norm output:")
        print(f"  Shape: HF={hf_norm.shape}, Muse={muse_norm.shape}")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        if max_diff > 1e-4:
            print(f"  ⚠️  Large difference detected!")
        else:
            print(f"  ✓ Match!")
    
    # Compare output layer (lm_head for HF, output for Muse)
    # Note: Muse's output goes through unembed which applies norm and then output
    # HF's lm_head takes norm output and produces logits
    if "output" in hf_activations:
        # For Muse, we need to compare with the final norm output before unembed
        # since HF's lm_head takes norm output as input
        if "final_norm" in muse_activations:
            muse_norm_out = muse_activations["final_norm"].to(device=device, dtype=dtype)
            # Compare the input to output layer (norm output)
            print(f"\nOutput layer input (norm output):")
            print(f"  Shape: HF (to lm_head)={hf_model.model.norm.weight.shape if hasattr(hf_model.model, 'norm') else 'N/A'}, Muse={muse_norm_out.shape}")
            # Note: We can't directly compare HF's norm output without hooking it
            # So we skip this comparison
        
        # Compare output layer outputs if available
        if "output_before_float" in muse_activations:
            hf_out = hf_activations["output"].to(device=device, dtype=dtype)
            muse_out = muse_activations["output_before_float"].to(device=device, dtype=dtype)
            # Note: Muse's output is before .float() conversion in unembed
            # HF's output is the final logits
            out_diff = (hf_out - muse_out).abs()
            max_diff = out_diff.max().item()
            mean_diff = out_diff.mean().item()
            print(f"\nOutput layer output (logits before dtype conversion):")
            print(f"  Shape: HF={hf_out.shape}, Muse={muse_out.shape}")
            print(f"  Max diff: {max_diff:.6e}")
            print(f"  Mean diff: {mean_diff:.6e}")
            if max_diff > 1e-4:
                print(f"  ⚠️  Large difference detected!")
            else:
                print(f"  ✓ Match!")
    
    # Summary
    print(f"\n{'='*60}")
    print("Activation Comparison Summary")
    print(f"{'='*60}")
    if first_mismatch_layer is not None:
        print(f"⚠️  First mismatch detected at layer {first_mismatch_layer}")
        print(f"   All layers before layer {first_mismatch_layer} match correctly.")
    else:
        print(f"✓ All intermediate activations match!")
    print(f"{'='*60}\n")
    
    # Note: TransformerDecoder.unembed() converts output to float32 for numerical stability
    # Convert to match HF model's dtype for comparison
    muse_logits = muse_logits.to(dtype=hf_logits.dtype)

    # Compare logits
    print(f"\n{'='*60}")
    print("Logits Comparison")
    print(f"{'='*60}")
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Muse logits shape: {muse_logits.shape}")
    print(f"HF logits dtype: {hf_logits.dtype}")
    print(f"Muse logits dtype: {muse_logits.dtype} (converted for comparison)")
    
    # Calculate differences
    diff = (hf_logits - muse_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    max_diff_per_token = diff.max(dim=-1)[0].max().item()
    
    print(f"\nMax absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Max difference per token: {max_diff_per_token:.6e}")
    
    # Check if logits are close
    # For bfloat16, use more lenient tolerance due to numerical precision
    # bfloat16 has ~3-4 decimal digits of precision
    if dtype == torch.bfloat16:
        rtol = 1e-2  # 1% relative tolerance
        atol = 1.0   # Absolute tolerance of 1.0
    else:
        rtol = 1e-4
        atol = 1e-4
    
    is_close = torch.allclose(hf_logits, muse_logits, rtol=rtol, atol=atol)
    print(f"\nLogits match (rtol={rtol}, atol={atol}): {is_close}")
    
    # Calculate relative differences for better diagnostics
    hf_logits_abs = hf_logits.abs()
    relative_diff = diff / (hf_logits_abs + 1e-8)  # Add small epsilon to avoid division by zero
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()
    
    print(f"Max relative difference: {max_relative_diff:.6e}")
    print(f"Mean relative difference: {mean_relative_diff:.6e}")
    
    if not is_close:
        # Find positions with largest differences
        max_diff_pos = diff.argmax()
        max_diff_pos_3d = torch.unravel_index(max_diff_pos, diff.shape)
        print(f"\nLargest difference at position: {max_diff_pos_3d}")
        print(f"HF value: {hf_logits[max_diff_pos_3d].item():.6f}")
        print(f"Muse value: {muse_logits[max_diff_pos_3d].item():.6f}")
        print(f"Absolute difference: {diff[max_diff_pos_3d].item():.6e}")
        print(f"Relative difference: {relative_diff[max_diff_pos_3d].item():.6e}")
        
        # Check if differences are within acceptable range for bfloat16
        if dtype == torch.bfloat16 and max_diff < 10.0 and mean_diff < 2.0:
            print(f"\nNote: Differences are within acceptable range for bfloat16 precision.")
            print(f"bfloat16 has ~3-4 decimal digits of precision, so small differences are expected.")
    
    # Assert that logits are close (with reasonable tolerance)
    assert torch.allclose(
        hf_logits, muse_logits, rtol=rtol, atol=atol
    ), (
        f"Logits do not match! "
        f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, "
        f"Max relative diff: {max_relative_diff:.6e}"
    )
    
    print(f"\n{'='*60}")
    print("✓ Logits comparison passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_qwen3_logits_align_with_hf_checkpoint()