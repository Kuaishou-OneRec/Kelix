"""
Dump all intermediate activations from both HF and Muse Qwen3 Layer 0 for debugging.

This script captures all inputs/outputs at each step of the first transformer layer
for both HF and Muse models, and saves them to files for offline comparison.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from muse.config import Qwen3Config
from muse.models.qwen3 import Qwen3Model
from muse.training.common import set_default_dtype


def _build_qwen3_config(hf_cfg):
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


def dump_hf_layer0_activations(hf_model, model_inputs, output_dir):
    """Dump HF model Layer 0 activations."""
    import os
    
    hf_activations = {}
    
    def make_hf_hook(name, storage):
        def hook(module, input, output):
            if isinstance(input, tuple):
                storage[f"{name}_input"] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input]
            else:
                storage[f"{name}_input"] = input.detach().clone() if isinstance(input, torch.Tensor) else input
            
            if isinstance(output, tuple):
                storage[f"{name}_output"] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in output]
            else:
                storage[f"{name}_output"] = output.detach().clone() if isinstance(output, torch.Tensor) else output
        return hook
    
    def make_hf_pre_hook(name, storage):
        def hook(module, input):
            if isinstance(input, tuple):
                storage[f"{name}_input"] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input]
            else:
                storage[f"{name}_input"] = input.detach().clone() if isinstance(input, torch.Tensor) else input
        return hook
    
    # Get HF Layer 0
    hf_layer_0 = hf_model.model.layers[0]
    hf_attn_0 = hf_layer_0.self_attn
    
    hf_hooks = []
    
    # Hook embedding
    hf_hooks.append(hf_model.model.embed_tokens.register_forward_hook(
        make_hf_hook("embedding", hf_activations)
    ))
    
    # Hook Layer 0
    hf_hooks.append(hf_layer_0.register_forward_pre_hook(
        make_hf_pre_hook("layer0", hf_activations)
    ))
    
    # Hook input_layernorm (sa_norm)
    hf_hooks.append(hf_layer_0.input_layernorm.register_forward_hook(
        make_hf_hook("layer0_input_layernorm", hf_activations)
    ))
    
    # Hook attention module
    hf_hooks.append(hf_attn_0.register_forward_pre_hook(
        make_hf_pre_hook("attn0", hf_activations)
    ))
    
    # Hook q_proj
    hf_hooks.append(hf_attn_0.q_proj.register_forward_hook(
        make_hf_hook("attn0_q_proj", hf_activations)
    ))
    
    # Hook q_norm
    if hasattr(hf_attn_0, 'q_norm') and hf_attn_0.q_norm is not None:
        hf_hooks.append(hf_attn_0.q_norm.register_forward_hook(
            make_hf_hook("attn0_q_norm", hf_activations)
        ))
    
    # Hook k_proj
    hf_hooks.append(hf_attn_0.k_proj.register_forward_hook(
        make_hf_hook("attn0_k_proj", hf_activations)
    ))
    
    # Hook k_norm
    if hasattr(hf_attn_0, 'k_norm') and hf_attn_0.k_norm is not None:
        hf_hooks.append(hf_attn_0.k_norm.register_forward_hook(
            make_hf_hook("attn0_k_norm", hf_activations)
        ))
    
    # Hook v_proj
    hf_hooks.append(hf_attn_0.v_proj.register_forward_hook(
        make_hf_hook("attn0_v_proj", hf_activations)
    ))
    
    # Hook attention forward to capture qkv after RoPE and attention output
    def hf_attn_forward_hook(module, input, output):
        # HF attention returns (hidden_states, attn_weights)
        if isinstance(output, tuple):
            hf_activations['attn0_output'] = output[0].detach().clone()
            if len(output) > 1 and output[1] is not None:
                hf_activations['attn0_attn_weights'] = output[1].detach().clone()
        else:
            hf_activations['attn0_output'] = output.detach().clone()
    
    hf_hooks.append(hf_attn_0.register_forward_hook(hf_attn_forward_hook))
    
    # Hook o_proj (output_proj)
    hf_hooks.append(hf_attn_0.o_proj.register_forward_pre_hook(
        make_hf_pre_hook("attn0_o_proj", hf_activations)
    ))
    hf_hooks.append(hf_attn_0.o_proj.register_forward_hook(
        make_hf_hook("attn0_o_proj", hf_activations)
    ))
    
    # Hook post_attention_layernorm (mlp_norm)
    hf_hooks.append(hf_layer_0.post_attention_layernorm.register_forward_hook(
        make_hf_hook("layer0_post_attention_layernorm", hf_activations)
    ))
    
    # Hook MLP components
    hf_mlp = hf_layer_0.mlp
    hf_hooks.append(hf_mlp.gate_proj.register_forward_hook(
        make_hf_hook("layer0_mlp_gate_proj", hf_activations)
    ))
    hf_hooks.append(hf_mlp.up_proj.register_forward_hook(
        make_hf_hook("layer0_mlp_up_proj", hf_activations)
    ))
    hf_hooks.append(hf_mlp.down_proj.register_forward_hook(
        make_hf_hook("layer0_mlp_down_proj", hf_activations)
    ))
    
    # Hook MLP output
    hf_hooks.append(hf_mlp.register_forward_hook(
        make_hf_hook("layer0_mlp", hf_activations)
    ))
    
    # Hook Layer 0 output
    hf_hooks.append(hf_layer_0.register_forward_hook(
        make_hf_hook("layer0", hf_activations)
    ))
    
    # Forward pass
    print("Running HF forward pass to capture activations...")
    with torch.no_grad():
        hf_outputs = hf_model(**model_inputs, output_attentions=True)
    
    # Remove hooks
    for hook in hf_hooks:
        hook.remove()
    
    # Save HF activations
    hf_output_dir = os.path.join(output_dir, "hf")
    os.makedirs(hf_output_dir, exist_ok=True)
    
    for name, value in hf_activations.items():
        if isinstance(value, torch.Tensor):
            torch.save(value.cpu(), os.path.join(hf_output_dir, f"{name}.pt"))
        elif isinstance(value, (list, tuple)):
            saved_items = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    saved_items.append(item.cpu())
                else:
                    saved_items.append(item)
            torch.save(saved_items, os.path.join(hf_output_dir, f"{name}.pt"))
        elif isinstance(value, dict):
            saved_dict = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    saved_dict[k] = v.cpu()
                else:
                    saved_dict[k] = v
            torch.save(saved_dict, os.path.join(hf_output_dir, f"{name}.pt"))
        else:
            torch.save(value, os.path.join(hf_output_dir, f"{name}.pt"))
    
    print(f"✓ Saved {len(hf_activations)} HF activation tensors to {hf_output_dir}")
    return hf_activations


def dump_layer0_activations(checkpoint_dir, output_dir="layer0_dumps", prompt=None, dump_hf=True):
    """
    Dump all intermediate activations from both HF and Muse Qwen3 Layer 0.
    
    Args:
        checkpoint_dir: Path to HF checkpoint directory
        output_dir: Directory to save dumped activations
        prompt: Input prompt (default: simple test prompt)
        dump_hf: Whether to dump HF activations (default: True)
    """
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and HF model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Prepare input
    if prompt is None:
        prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(hf_model.device)
    
    # Get config and build Muse model
    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()
    muse_config = _build_qwen3_config(hf_config_dict)
    
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    
    # Create Muse model
    with set_default_dtype("bfloat16" if dtype == torch.bfloat16 else "float32"):
        muse_model = Qwen3Model(muse_config)
    
    # Convert and load state dict
    state_dict = muse_model.convert_hf_state_dict(hf_state_dict)
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            state_dict[key] = tensor.to(device=device, dtype=dtype)
    
    muse_model.load_state_dict(state_dict, strict=False)
    muse_model = muse_model.to(device=device, dtype=dtype)
    muse_model.eval()
    
    # Ensure eager attention
    import os
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
    hf_model.config._attn_implementation = "eager"
    hf_model.eval()
    
    # Dump HF activations first
    if dump_hf:
        print("\n" + "=" * 80)
        print("Dumping HF Model Activations")
        print("=" * 80)
        hf_activations = dump_hf_layer0_activations(hf_model, model_inputs, output_dir)
    
    # Storage for Muse activations
    activations = {}
    
    # Hook functions to capture activations
    def make_hook(name, storage):
        def hook(module, input, output):
            if isinstance(input, tuple):
                storage[f"{name}_input"] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input]
            else:
                storage[f"{name}_input"] = input.detach().clone() if isinstance(input, torch.Tensor) else input
            
            if isinstance(output, tuple):
                storage[f"{name}_output"] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in output]
            else:
                storage[f"{name}_output"] = output.detach().clone() if isinstance(output, torch.Tensor) else output
        return hook
    
    def make_pre_hook(name, storage):
        def hook(module, input):
            if isinstance(input, tuple):
                storage[f"{name}_input"] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input]
            else:
                storage[f"{name}_input"] = input.detach().clone() if isinstance(input, torch.Tensor) else input
        return hook
    
    # Get Layer 0
    layer_0 = muse_model.model.layers[0]
    attn_0 = layer_0.attn
    
    hooks = []
    
    # Hook embedding layer
    hooks.append(muse_model.model.tok_embeddings.register_forward_hook(
        make_hook("embedding", activations)
    ))
    
    # Hook Layer 0 inputs
    hooks.append(layer_0.register_forward_pre_hook(
        make_pre_hook("layer0", activations)
    ))
    
    # Hook sa_norm (input normalization)
    hooks.append(layer_0.sa_norm.register_forward_hook(
        make_hook("layer0_sa_norm", activations)
    ))
    
    # Hook attention module inputs
    hooks.append(attn_0.register_forward_pre_hook(
        make_pre_hook("attn0", activations)
    ))
    
    # Hook q_proj
    hooks.append(attn_0.q_proj.register_forward_hook(
        make_hook("attn0_q_proj", activations)
    ))
    
    # Hook q_norm
    if attn_0.q_norm is not None:
        hooks.append(attn_0.q_norm.register_forward_hook(
            make_hook("attn0_q_norm", activations)
        ))
    
    # Hook k_proj
    hooks.append(attn_0.k_proj.register_forward_hook(
        make_hook("attn0_k_proj", activations)
    ))
    
    # Hook k_norm
    if attn_0.k_norm is not None:
        hooks.append(attn_0.k_norm.register_forward_hook(
            make_hook("attn0_k_norm", activations)
        ))
    
    # Hook v_proj
    hooks.append(attn_0.v_proj.register_forward_hook(
        make_hook("attn0_v_proj", activations)
    ))
    
    # Hook RoPE (pos_embeddings)
    rope_call_count = {'q': 0, 'k': 0}
    def rope_hook(module, input, output):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        
        if rope_call_count['q'] == 0:
            rope_call_count['q'] += 1
            activations["attn0_rope_q_input"] = x.detach().clone()
            activations["attn0_rope_q_output"] = output.detach().clone()
        elif rope_call_count['k'] == 0:
            rope_call_count['k'] += 1
            activations["attn0_rope_k_input"] = x.detach().clone()
            activations["attn0_rope_k_output"] = output.detach().clone()
    
    if attn_0.pos_embeddings is not None:
        hooks.append(attn_0.pos_embeddings.register_forward_hook(rope_hook))
    
    # Hook attention function inputs/outputs
    original_attn_fn = attn_0._attention_function
    attn_fn_inputs = {}
    attn_fn_outputs = {}
    
    def wrapped_attn_fn(*args, **kwargs):
        # Capture inputs
        if len(args) >= 3:
            attn_fn_inputs['q'] = args[0].detach().clone()
            attn_fn_inputs['k'] = args[1].detach().clone()
            attn_fn_inputs['v'] = args[2].detach().clone()
        if 'q' in kwargs:
            attn_fn_inputs['q'] = kwargs['q'].detach().clone()
            attn_fn_inputs['k'] = kwargs['k'].detach().clone()
            attn_fn_inputs['v'] = kwargs['v'].detach().clone()
        
        # Capture kwargs
        attn_fn_inputs['kwargs'] = {k: v for k, v in kwargs.items() if k not in ['q', 'k', 'v']}
        
        # Call original function
        result = original_attn_fn(*args, **kwargs)
        attn_fn_outputs['output'] = result.detach().clone()
        return result
    
    attn_0._attention_function = wrapped_attn_fn
    
    # Hook output_proj
    hooks.append(attn_0.output_proj.register_forward_pre_hook(
        make_pre_hook("attn0_output_proj", activations)
    ))
    hooks.append(attn_0.output_proj.register_forward_hook(
        make_hook("attn0_output_proj", activations)
    ))
    
    # Hook attention module output
    hooks.append(attn_0.register_forward_hook(
        make_hook("attn0", activations)
    ))
    
    # Hook mlp_norm
    hooks.append(layer_0.mlp_norm.register_forward_hook(
        make_hook("layer0_mlp_norm", activations)
    ))
    
    # Hook MLP components
    # Muse uses FeedForward with w1, w2, w3 instead of gate_proj, down_proj, up_proj
    mlp = layer_0.mlp
    hooks.append(mlp.w1.register_forward_hook(
        make_hook("layer0_mlp_gate_proj", activations)
    ))
    if mlp.w3 is not None:
        hooks.append(mlp.w3.register_forward_hook(
            make_hook("layer0_mlp_up_proj", activations)
        ))
    hooks.append(mlp.w2.register_forward_hook(
        make_hook("layer0_mlp_down_proj", activations)
    ))
    
    # Hook MLP output
    hooks.append(mlp.register_forward_hook(
        make_hook("layer0_mlp", activations)
    ))
    
    # Hook Layer 0 output
    hooks.append(layer_0.register_forward_hook(
        make_hook("layer0", activations)
    ))
    
    # Forward pass
    print("Running forward pass to capture activations...")
    with torch.no_grad():
        muse_inputs = {"tokens": model_inputs["input_ids"]}
        muse_logits = muse_model(**muse_inputs)
    
    # Add attention function inputs/outputs to activations
    activations.update(attn_fn_inputs)
    activations.update(attn_fn_outputs)
    
    # Restore original attention function
    attn_0._attention_function = original_attn_fn
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Save Muse activations
    muse_output_dir = os.path.join(output_dir, "muse")
    os.makedirs(muse_output_dir, exist_ok=True)
    
    print(f"\nSaving Muse activations to {muse_output_dir}...")
    
    # Save metadata (shared between HF and Muse)
    metadata = {
        "config": {
            "embed_dim": muse_config.embed_dim,
            "num_heads": muse_config.num_heads,
            "num_kv_heads": muse_config.num_kv_heads,
            "head_dim": muse_config.head_dim,
            "intermediate_dim": muse_config.intermediate_dim,
            "rope_base": muse_config.rope_base,
            "norm_eps": muse_config.norm_eps,
            "attn_dropout": muse_config.attn_dropout,
        },
        "input_info": {
            "prompt": prompt,
            "text": text,
            "input_ids_shape": model_inputs["input_ids"].shape,
            "seq_len": model_inputs["input_ids"].shape[1],
        },
        "device": str(device),
        "dtype": str(dtype),
    }
    # Save metadata to both directories
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))
    torch.save(metadata, os.path.join(muse_output_dir, "metadata.pt"))
    if dump_hf:
        torch.save(metadata, os.path.join(os.path.join(output_dir, "hf"), "metadata.pt"))
    
    # Save each Muse activation
    for name, value in activations.items():
        if isinstance(value, torch.Tensor):
            torch.save(value.cpu(), os.path.join(muse_output_dir, f"{name}.pt"))
        elif isinstance(value, (list, tuple)):
            # Save list/tuple of tensors
            saved_items = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    saved_items.append(item.cpu())
                else:
                    saved_items.append(item)
            torch.save(saved_items, os.path.join(muse_output_dir, f"{name}.pt"))
        elif isinstance(value, dict):
            # Save dict (e.g., kwargs)
            saved_dict = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    saved_dict[k] = v.cpu()
                else:
                    saved_dict[k] = v
            torch.save(saved_dict, os.path.join(muse_output_dir, f"{name}.pt"))
        else:
            # Save other types as-is
            torch.save(value, os.path.join(muse_output_dir, f"{name}.pt"))
    
    print(f"✓ Saved {len(activations)} Muse activation tensors to {muse_output_dir}")
    print(f"\nMuse activation names:")
    for name in sorted(activations.keys()):
        value = activations[name]
        if isinstance(value, torch.Tensor):
            print(f"  {name}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"  {name}: list/tuple with {len(value)} items")
        elif isinstance(value, dict):
            print(f"  {name}: dict with keys {list(value.keys())}")
        else:
            print(f"  {name}: {type(value)}")
    
    return activations, output_dir


if __name__ == "__main__":
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "layer0_dumps"
    prompt = sys.argv[3] if len(sys.argv) > 3 else None
    dump_hf_flag = sys.argv[4].lower() != "false" if len(sys.argv) > 4 else True
    
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dump HF: {dump_hf_flag}")
    if prompt:
        print(f"Prompt: {prompt}")
    
    activations, output_dir = dump_layer0_activations(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        prompt=prompt,
        dump_hf=dump_hf_flag
    )
    
    print(f"\n" + "=" * 80)
    print(f"✓ Dump complete!")
    print(f"=" * 80)
    print(f"Files saved to:")
    print(f"  - HF activations: {os.path.join(output_dir, 'hf')}")
    print(f"  - Muse activations: {os.path.join(output_dir, 'muse')}")
    print(f"  - Metadata: {os.path.join(output_dir, 'metadata.pt')}")

