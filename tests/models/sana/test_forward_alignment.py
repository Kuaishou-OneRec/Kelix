# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Forward Alignment Test for Sana DiT Model.

This test compares the muse implementation against the diffusers SanaTransformer2DModel
layer by layer to identify any numerical differences.

Usage:
    PYTHONPATH=/path/to/muse:$PYTHONPATH python tests/models/sana/test_forward_alignment.py
    
    # Or run with pytest
    pytest tests/models/sana/test_forward_alignment.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths - update these to your locations
# Diffusers format checkpoint directory (containing transformer/ subfolder)
DIFFUSERS_CHECKPOINT_PATH = "/llm_reco_ssd/zhouyang12/models/Sana_1600M_1024px_diffusers"

# Model configuration for Sana 1600M
MODEL_CONFIG = {
    "input_size": 32,
    "patch_size": 1,
    "in_channels": 32,
    "hidden_size": 2240,
    "depth": 20,
    "num_heads": 20,
    "mlp_ratio": 2.5,
    "caption_channels": 2304,
    "model_max_length": 300,
    "attn_type": "linear",
    "ffn_type": "glumbconv",
    "mlp_acts": ("silu", "silu", None),
    "use_pe": False,
    "qk_norm": True,       # Enable for diffusers alignment (diffusers always has RMSNorm)
    "cross_norm": True,    # Enable for diffusers alignment (diffusers always has RMSNorm)
    "y_norm": True,
    "y_norm_scale_factor": 0.01,
    "pred_sigma": False,
    "learn_sigma": False,
    "class_dropout_prob": 0.0,
    "drop_path": 0.0,
}

# Tolerance for numerical comparison
ATOL = 1e-5
RTOL = 1e-4


def load_diffusers_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load the diffusers SanaTransformer2DModel."""
    from diffusers.models import SanaTransformer2DModel
    
    # Try loading with subfolder="transformer" first
    transformer_path = os.path.join(checkpoint_path, "transformer")
    if os.path.exists(transformer_path):
        model = SanaTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=dtype
        )
    else:
        model = SanaTransformer2DModel.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype
        )
    
    model = model.to(device=device)
    model.eval()
    
    print(f"  Loaded diffusers model config: {model.config}")
    return model


def load_muse_model(diffusers_state_dict: dict, device: torch.device, dtype: torch.dtype):
    """Load the muse Sana model with diffusers state dict."""
    from muse.models.sana.modeling import SanaModel
    from muse.config.model_config import SanaConfig
    
    config = SanaConfig(
        model_class="SanaModel",
        input_size=MODEL_CONFIG["input_size"],
        patch_size=MODEL_CONFIG["patch_size"],
        in_channels=MODEL_CONFIG["in_channels"],
        hidden_size=MODEL_CONFIG["hidden_size"],
        depth=MODEL_CONFIG["depth"],
        num_heads=MODEL_CONFIG["num_heads"],
        mlp_ratio=MODEL_CONFIG["mlp_ratio"],
        caption_channels=MODEL_CONFIG["caption_channels"],
        model_max_length=MODEL_CONFIG["model_max_length"],
        attn_type=MODEL_CONFIG["attn_type"],
        ffn_type=MODEL_CONFIG["ffn_type"],
        mlp_acts=MODEL_CONFIG["mlp_acts"],
        use_pe=MODEL_CONFIG["use_pe"],
        qk_norm=MODEL_CONFIG["qk_norm"],
        cross_norm=MODEL_CONFIG["cross_norm"],
        y_norm=MODEL_CONFIG["y_norm"],
        y_norm_scale_factor=MODEL_CONFIG["y_norm_scale_factor"],
        pred_sigma=MODEL_CONFIG["pred_sigma"],
        learn_sigma=MODEL_CONFIG["learn_sigma"],
        class_dropout_prob=MODEL_CONFIG["class_dropout_prob"],
        drop_path=MODEL_CONFIG["drop_path"],
    )
    
    model = SanaModel(config)
    
    # Convert diffusers state dict to muse format using model's converter
    converted_state_dict = model.convert_diffusers_state_dict(diffusers_state_dict)
    
    # Expected missing keys (muse-only parameters not in diffusers):
    # - pos_embed: positional embedding buffer (not used when use_pe=False)
    # - y_embedder.y_embedding: learnable caption embedding for unconditional generation
    expected_missing = {"pos_embed", "y_embedder.y_embedding"}
    
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    
    unexpected_missing = set(missing) - expected_missing
    if unexpected_missing:
        print(f"  [ERROR] Unexpected missing keys: {list(unexpected_missing)}")
    elif missing:
        print(f"  [Info] Expected missing keys (muse-only): {missing}")
    
    if unexpected:
        print(f"  [ERROR] Unexpected keys in state dict: {unexpected[:10]}..." if len(unexpected) > 10 else f"  [ERROR] Unexpected keys: {unexpected}")
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    return model


def create_test_inputs(
    batch_size: int = 1,
    latent_size: int = 32,
    latent_channels: int = 32,
    text_length: int = 300,
    text_channels: int = 2304,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
):
    """Create test inputs for both models."""
    torch.manual_seed(seed)
    
    # Latent input: [B, C, H, W]
    x = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device, dtype=dtype)
    
    # Timestep: [B]
    timestep = torch.tensor([500] * batch_size, device=device, dtype=torch.long)
    
    # Text embeddings: [B, L, D] for diffusers, [B, 1, L, D] for muse
    y_diffusers = torch.randn(batch_size, text_length, text_channels, device=device, dtype=dtype)
    y_muse = y_diffusers.unsqueeze(1)  # [B, 1, L, D]
    
    # Attention mask: [B, L] for diffusers, [B, 1, 1, L] for muse
    mask_diffusers = torch.ones(batch_size, text_length, device=device, dtype=dtype)
    mask_muse = mask_diffusers.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
    
    return {
        "x": x,
        "timestep": timestep,
        "y_diffusers": y_diffusers,
        "y_muse": y_muse,
        "mask_diffusers": mask_diffusers,
        "mask_muse": mask_muse,
    }


def compare_tensors(name: str, tensor1: torch.Tensor, tensor2: torch.Tensor, atol=ATOL, rtol=RTOL):
    """Compare two tensors and print results."""
    if tensor1.shape != tensor2.shape:
        print(f"  [{name}] SHAPE MISMATCH: {tensor1.shape} vs {tensor2.shape}")
        return False
    
    diff = (tensor1 - tensor2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Check relative difference for non-zero values
    mask = tensor1.abs() > 1e-8
    if mask.any():
        rel_diff = (diff[mask] / tensor1.abs()[mask]).max().item()
    else:
        rel_diff = 0.0
    
    passed = max_diff < atol or rel_diff < rtol
    status = "PASS" if passed else "FAIL"
    
    print(f"  [{name}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}  {status}")
    
    return passed


def debug_first_block(diffusers_model, muse_model, inputs, dtype):
    """Debug the first transformer block step by step to find divergence."""
    print("\n" + "=" * 70)
    print("[DEBUG] First Transformer Block - Step by Step Comparison")
    print("=" * 70)
    
    with torch.no_grad():
        x = inputs["x"].to(dtype)
        timestep_float = inputs["timestep"].float()
        B = x.shape[0]
        H = W = x.shape[-1]  # 32x32 latent
        
        # ====== STEP 1: Embeddings (should be identical) ======
        print("\n[Step 1] Embeddings...")
        
        # Patch embedding
        diff_x = diffusers_model.patch_embed(x)
        muse_x = muse_model.x_embedder(x)
        compare_tensors("patch_embed", diff_x, muse_x)
        
        # Timestep embedding
        diff_time, diff_t_emb = diffusers_model.time_embed(
            timestep_float, batch_size=B, hidden_dtype=dtype
        )
        muse_t = muse_model.t_embedder(timestep_float.long().float())
        muse_t0 = muse_model.t_block(muse_t)
        compare_tensors("t_embedder", diff_t_emb, muse_t)
        compare_tensors("t_block", diff_time, muse_t0)
        
        # Caption embedding
        y_diff = inputs["y_diffusers"]
        y_muse = inputs["y_muse"]
        
        diff_caption = diffusers_model.caption_projection(y_diff)
        diff_caption = diff_caption.view(B, -1, diff_x.shape[-1])
        diff_caption = diffusers_model.caption_norm(diff_caption)
        
        muse_caption = muse_model.y_embedder(y_muse, False)
        if muse_model.y_norm:
            muse_caption = muse_model.attention_y_norm(muse_caption)
        muse_caption_squeezed = muse_caption.squeeze(1)
        compare_tensors("caption_embed", diff_caption, muse_caption_squeezed)
        
        # ====== STEP 2: First Block - Modulation ======
        print("\n[Step 2] First Block - Modulation Parameters...")
        
        diff_block = diffusers_model.transformer_blocks[0]
        muse_block = muse_model.blocks[0]
        
        # Diffusers modulation
        diff_shift_msa, diff_scale_msa, diff_gate_msa, diff_shift_mlp, diff_scale_mlp, diff_gate_mlp = (
            diff_block.scale_shift_table[None] + diff_time.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        
        # Muse modulation
        muse_shift_msa, muse_scale_msa, muse_gate_msa, muse_shift_mlp, muse_scale_mlp, muse_gate_mlp = (
            muse_block.scale_shift_table[None] + muse_t0.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        
        compare_tensors("shift_msa", diff_shift_msa, muse_shift_msa)
        compare_tensors("scale_msa", diff_scale_msa, muse_scale_msa)
        compare_tensors("gate_msa", diff_gate_msa, muse_gate_msa)
        compare_tensors("shift_mlp", diff_shift_mlp, muse_shift_mlp)
        compare_tensors("scale_mlp", diff_scale_mlp, muse_scale_mlp)
        compare_tensors("gate_mlp", diff_gate_mlp, muse_gate_mlp)
        
        # ====== STEP 3: First Block - Self Attention ======
        print("\n[Step 3] First Block - Self Attention...")
        
        # Diffusers: norm + modulate + attn
        diff_norm1 = diff_block.norm1(diff_x)
        diff_norm1_mod = diff_norm1 * (1 + diff_scale_msa) + diff_shift_msa
        diff_norm1_mod = diff_norm1_mod.to(dtype)
        
        # Muse: norm + modulate (using t2i_modulate)
        muse_norm1 = muse_block.norm1(muse_x)
        muse_norm1_mod = muse_norm1 * (1 + muse_scale_msa) + muse_shift_msa
        
        compare_tensors("norm1", diff_norm1, muse_norm1)
        compare_tensors("norm1_modulated", diff_norm1_mod, muse_norm1_mod)
        
        # ====== STEP 3b: Detailed Self-Attention comparison ======
        print("\n  [3b] Self-Attention step-by-step...")
        
        diff_attn = diff_block.attn1
        muse_attn = muse_block.attn
        N = diff_norm1_mod.shape[1]  # sequence length (H*W)
        C = diff_norm1_mod.shape[-1]  # hidden dimension
        
        # Step 0: QKV weight comparison
        print("\n    Checking QKV weights...")
        
        # Compare Q, K, V weights directly (now both use separate layers)
        compare_tensors("to_q_weight", diff_attn.to_q.weight, muse_attn.to_q.weight)
        compare_tensors("to_k_weight", diff_attn.to_k.weight, muse_attn.to_k.weight)
        compare_tensors("to_v_weight", diff_attn.to_v.weight, muse_attn.to_v.weight)
        
        # Check if bias exists and compare
        if hasattr(diff_attn.to_q, 'bias') and diff_attn.to_q.bias is not None:
            if muse_attn.to_q.bias is not None:
                compare_tensors("to_q_bias", diff_attn.to_q.bias, muse_attn.to_q.bias)
                compare_tensors("to_k_bias", diff_attn.to_k.bias, muse_attn.to_k.bias)
                compare_tensors("to_v_bias", diff_attn.to_v.bias, muse_attn.to_v.bias)
            else:
                print("    [qkv_bias] diffusers has bias but muse doesn't!")
        else:
            print("    [qkv_bias] Neither has bias (OK)")
        
        # Step 1: QKV linear projection
        print("\n    Checking QKV projection...")
        
        # Diffusers: separate q, k, v projections
        diff_q = diff_attn.to_q(diff_norm1_mod)
        diff_k = diff_attn.to_k(diff_norm1_mod)
        diff_v = diff_attn.to_v(diff_norm1_mod)
        
        # Muse: also separate q, k, v projections (aligned with diffusers)
        muse_q = muse_attn.to_q(muse_norm1_mod)
        muse_k = muse_attn.to_k(muse_norm1_mod)
        muse_v = muse_attn.to_v(muse_norm1_mod)
        
        compare_tensors("q_after_linear", diff_q, muse_q)
        compare_tensors("k_after_linear", diff_k, muse_k)
        compare_tensors("v_after_linear", diff_v, muse_v)
        
        # Step 2: QK normalization
        print("\n    Checking QK normalization...")
        if diff_attn.norm_q is not None:
            diff_q_normed = diff_attn.norm_q(diff_q)
            diff_k_normed = diff_attn.norm_k(diff_k)
        else:
            diff_q_normed = diff_q
            diff_k_normed = diff_k
        
        muse_q_normed = muse_attn.q_norm(muse_q)
        muse_k_normed = muse_attn.k_norm(muse_k)
        
        compare_tensors("q_after_norm", diff_q_normed, muse_q_normed)
        compare_tensors("k_after_norm", diff_k_normed, muse_k_normed)
        
        # Step 3: Reshape for attention
        print("\n    Checking reshape...")
        heads = diff_attn.heads
        head_dim = C // heads
        
        # Diffusers reshape
        diff_q_reshape = diff_q_normed.transpose(1, 2).unflatten(1, (heads, -1))  # [B, heads, head_dim, N]
        diff_k_reshape = diff_k_normed.transpose(1, 2).unflatten(1, (heads, -1)).transpose(2, 3)  # [B, heads, N, head_dim]
        diff_v_reshape = diff_v.transpose(1, 2).unflatten(1, (heads, -1))  # [B, heads, head_dim, N]
        
        # Muse reshape
        muse_q_t = muse_q_normed.transpose(-1, -2)  # [B, C, N]
        muse_k_t = muse_k_normed.transpose(-1, -2)
        muse_v_t = muse_v.transpose(-1, -2)
        
        muse_q_reshape = muse_q_t.reshape(B, heads, head_dim, N)  # [B, heads, head_dim, N]
        muse_k_reshape = muse_k_t.reshape(B, heads, head_dim, N).transpose(-1, -2)  # [B, heads, N, head_dim]
        muse_v_reshape = muse_v_t.reshape(B, heads, head_dim, N)  # [B, heads, head_dim, N]
        
        compare_tensors("q_reshaped", diff_q_reshape, muse_q_reshape)
        compare_tensors("k_reshaped", diff_k_reshape, muse_k_reshape)
        compare_tensors("v_reshaped", diff_v_reshape, muse_v_reshape)
        
        # Step 4: ReLU activation
        print("\n    Checking ReLU activation...")
        diff_q_relu = F.relu(diff_q_reshape)
        diff_k_relu = F.relu(diff_k_reshape)
        
        muse_q_relu = F.relu(muse_q_reshape)
        muse_k_relu = F.relu(muse_k_reshape)
        
        compare_tensors("q_after_relu", diff_q_relu, muse_q_relu)
        compare_tensors("k_after_relu", diff_k_relu, muse_k_relu)
        
        # Step 5: Float conversion and matmul
        print("\n    Checking linear attention computation...")
        diff_q_f = diff_q_relu.float()
        diff_k_f = diff_k_relu.float()
        diff_v_f = diff_v_reshape.float()
        
        muse_q_f = muse_q_relu.float()
        muse_k_f = muse_k_relu.float()
        muse_v_f = muse_v_reshape.float()
        
        # Pad value
        diff_v_pad = F.pad(diff_v_f, (0, 0, 0, 1), mode="constant", value=1.0)
        muse_v_pad = F.pad(muse_v_f, (0, 0, 0, 1), mode="constant", value=1.0)
        compare_tensors("v_padded", diff_v_pad, muse_v_pad)
        
        # First matmul: vk = v @ k
        diff_vk = torch.matmul(diff_v_pad, diff_k_f)
        muse_vk = torch.matmul(muse_v_pad, muse_k_f)
        compare_tensors("vk_matmul", diff_vk, muse_vk)
        
        # Second matmul: out = vk @ q
        diff_out = torch.matmul(diff_vk, diff_q_f)
        muse_out = torch.matmul(muse_vk, muse_q_f)
        compare_tensors("vkq_matmul", diff_out, muse_out)
        
        # Normalization
        diff_out_norm = diff_out[:, :, :-1] / (diff_out[:, :, -1:] + 1e-15)
        muse_out_norm = muse_out[:, :, :-1] / (muse_out[:, :, -1:] + 1e-15)
        compare_tensors("attn_normalized", diff_out_norm, muse_out_norm)
        
        # Reshape back
        diff_out_reshape = diff_out_norm.flatten(1, 2).transpose(1, 2).to(dtype)
        muse_out_reshape = muse_out_norm.view(B, C, N).permute(0, 2, 1).to(dtype)
        compare_tensors("attn_reshaped", diff_out_reshape, muse_out_reshape)
        
        # Output projection
        diff_out_proj = diff_attn.to_out[0](diff_out_reshape)
        muse_out_proj = muse_attn.proj(muse_out_reshape)
        compare_tensors("attn_projected", diff_out_proj, muse_out_proj)
        
        # Full self attention output
        print("\n    Full self-attention output...")
        diff_attn_out = diff_block.attn1(diff_norm1_mod)
        muse_attn_out = muse_block.attn(muse_norm1_mod, HW=(H, W))
        compare_tensors("self_attn_output", diff_attn_out, muse_attn_out)
        
        # After self-attention residual
        diff_x_after_attn = diff_x + diff_gate_msa * diff_attn_out
        muse_x_after_attn = muse_x + muse_gate_msa * muse_attn_out
        compare_tensors("after_self_attn", diff_x_after_attn, muse_x_after_attn)
        
        # ====== STEP 4: First Block - Cross Attention ======
        print("\n[Step 4] First Block - Cross Attention...")
        
        # Diffusers cross attention
        # encoder_attention_mask processing (same as in full forward)
        encoder_attention_mask = inputs["mask_diffusers"]
        if encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        
        diff_cross_out = diff_block.attn2(
            diff_x_after_attn,
            encoder_hidden_states=diff_caption,
            attention_mask=encoder_attention_mask,
        )
        diff_x_after_cross = diff_cross_out + diff_x_after_attn
        
        # Muse cross attention - need to prepare y and mask like in forward
        # Check for xformers
        _xformers_available = False
        try:
            import xformers.ops
            _xformers_available = True
        except ImportError:
            pass
        
        mask = inputs["mask_muse"].clone()
        y_for_cross = muse_caption.clone()
        
        if mask is not None:
            mask = mask.to(torch.int16)
            if mask.shape[0] != y_for_cross.shape[0]:
                mask = mask.repeat(y_for_cross.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1) if mask.ndim > 2 else mask
            if _xformers_available:
                y_for_cross = y_for_cross.squeeze(1) if y_for_cross.ndim == 4 else y_for_cross
                y_for_cross = y_for_cross.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, muse_x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = mask
                y_for_cross = y_for_cross.squeeze(1) if y_for_cross.ndim == 4 else y_for_cross
        elif _xformers_available:
            y_lens = [y_for_cross.shape[2]] * y_for_cross.shape[0] if y_for_cross.ndim == 4 else [y_for_cross.shape[1]] * y_for_cross.shape[0]
            y_for_cross = y_for_cross.squeeze(1) if y_for_cross.ndim == 4 else y_for_cross
            y_for_cross = y_for_cross.view(1, -1, muse_x.shape[-1])
        else:
            y_lens = None
            y_for_cross = y_for_cross.squeeze(1) if y_for_cross.ndim == 4 else y_for_cross
        
        muse_cross_out = muse_block.cross_attn(muse_x_after_attn, y_for_cross, y_lens)
        muse_x_after_cross = muse_x_after_attn + muse_cross_out
        
        # ====== STEP 4a: Cross Attention Step-by-Step ======
        print("\n  [4a] Cross-Attention step-by-step...")
        
        diff_cross_attn = diff_block.attn2
        muse_cross_attn = muse_block.cross_attn
        
        # 输入检查
        print("\n    Checking inputs...")
        compare_tensors("cross_attn_x_input", diff_x_after_attn, muse_x_after_attn)
        muse_caption_for_debug = muse_caption.squeeze(1) if muse_caption.ndim == 4 else muse_caption
        compare_tensors("cross_attn_cond_input", diff_caption, muse_caption_for_debug)
        
        # Step 0: 权重检查
        print("\n    Checking weights...")
        compare_tensors("cross_to_q_weight", diff_cross_attn.to_q.weight, muse_cross_attn.q_linear.weight)
        compare_tensors("cross_to_k_weight", diff_cross_attn.to_k.weight, muse_cross_attn.to_k.weight)
        compare_tensors("cross_to_v_weight", diff_cross_attn.to_v.weight, muse_cross_attn.to_v.weight)
        compare_tensors("cross_proj_weight", diff_cross_attn.to_out[0].weight, muse_cross_attn.proj.weight)
        
        # Step 1: Q 投影
        print("\n    Checking Q projection...")
        diff_cross_q = diff_cross_attn.to_q(diff_x_after_attn)
        muse_cross_q = muse_cross_attn.q_linear(muse_x_after_attn)
        compare_tensors("cross_q", diff_cross_q, muse_cross_q)
        
        # Step 2: K, V 投影 - 注意输入不同！
        # diffusers 用 diff_caption [B, L, C]
        # muse 用 y_for_cross (可能是 packed [1, L*B, C] 或 [B, L, C])
        print("\n    Checking K, V projection...")
        diff_cross_k = diff_cross_attn.to_k(diff_caption)
        diff_cross_v = diff_cross_attn.to_v(diff_caption)
        
        # 对于 muse，使用 squeeze 后的 muse_caption 来对比（绕过 xformers packing）
        muse_cross_k = muse_cross_attn.to_k(muse_caption_for_debug)
        muse_cross_v = muse_cross_attn.to_v(muse_caption_for_debug)
        compare_tensors("cross_k", diff_cross_k, muse_cross_k)
        compare_tensors("cross_v", diff_cross_v, muse_cross_v)
        
        # Step 3: QK norm (如果有)
        print("\n    Checking QK norm...")
        if hasattr(diff_cross_attn, 'norm_q') and diff_cross_attn.norm_q is not None:
            diff_cross_q_normed = diff_cross_attn.norm_q(diff_cross_q)
            diff_cross_k_normed = diff_cross_attn.norm_k(diff_cross_k)
        else:
            diff_cross_q_normed = diff_cross_q
            diff_cross_k_normed = diff_cross_k
            print("    [cross_qk_norm] diffusers has no qk_norm")
        
        muse_cross_q_normed = muse_cross_attn.q_norm(muse_cross_q)
        muse_cross_k_normed = muse_cross_attn.k_norm(muse_cross_k)
        compare_tensors("cross_q_normed", diff_cross_q_normed, muse_cross_q_normed)
        compare_tensors("cross_k_normed", diff_cross_k_normed, muse_cross_k_normed)
        
        # Step 4: Reshape for attention
        print("\n    Checking reshape...")
        heads = diff_cross_attn.heads
        head_dim = C // heads
        
        # diffusers reshape: [B, N, C] -> [B, heads, N, head_dim]
        diff_cross_q_reshape = diff_cross_q_normed.view(B, -1, heads, head_dim).transpose(1, 2)
        diff_cross_k_reshape = diff_cross_k_normed.view(B, -1, heads, head_dim).transpose(1, 2)
        diff_cross_v_reshape = diff_cross_v.view(B, -1, heads, head_dim).transpose(1, 2)
        
        # muse reshape: [B, N, C] -> [B, N, heads, head_dim] -> [B, heads, N, head_dim]
        muse_cross_q_reshape = muse_cross_q_normed.view(B, -1, heads, head_dim).transpose(1, 2)
        muse_cross_k_reshape = muse_cross_k_normed.view(B, -1, heads, head_dim).transpose(1, 2)
        muse_cross_v_reshape = muse_cross_v.view(B, -1, heads, head_dim).transpose(1, 2)
        
        compare_tensors("cross_q_reshape", diff_cross_q_reshape, muse_cross_q_reshape)
        compare_tensors("cross_k_reshape", diff_cross_k_reshape, muse_cross_k_reshape)
        compare_tensors("cross_v_reshape", diff_cross_v_reshape, muse_cross_v_reshape)
        
        # Step 5: Attention computation
        print("\n    Checking attention computation...")
        # 使用相同的 attention 计算
        cross_attn_out_diff = F.scaled_dot_product_attention(
            diff_cross_q_reshape, diff_cross_k_reshape, diff_cross_v_reshape, 
            attn_mask=None, dropout_p=0.0, is_causal=False
        )
        cross_attn_out_muse = F.scaled_dot_product_attention(
            muse_cross_q_reshape, muse_cross_k_reshape, muse_cross_v_reshape, 
            attn_mask=None, dropout_p=0.0, is_causal=False
        )
        compare_tensors("cross_attn_sdpa", cross_attn_out_diff, cross_attn_out_muse)
        
        # Step 6: Output projection
        print("\n    Checking output projection...")
        cross_attn_out_diff_flat = cross_attn_out_diff.transpose(1, 2).reshape(B, -1, C)
        cross_attn_out_muse_flat = cross_attn_out_muse.transpose(1, 2).reshape(B, -1, C)
        
        diff_cross_proj = diff_cross_attn.to_out[0](cross_attn_out_diff_flat)
        muse_cross_proj = muse_cross_attn.proj(cross_attn_out_muse_flat)
        compare_tensors("cross_proj_output", diff_cross_proj, muse_cross_proj)
        
        compare_tensors("cross_attn_output", diff_cross_out, muse_cross_out)
        compare_tensors("after_cross_attn", diff_x_after_cross, muse_x_after_cross)
        
        # ====== STEP 5: First Block - FFN ======
        print("\n[Step 5] First Block - FFN (GLUMBConv)...")
        
        # Diffusers: norm2 + modulate + ff
        diff_norm2 = diff_block.norm2(diff_x_after_cross)
        diff_norm2_mod = diff_norm2 * (1 + diff_scale_mlp) + diff_shift_mlp
        
        # Muse: norm2 + modulate
        muse_norm2 = muse_block.norm2(muse_x_after_cross)
        muse_norm2_mod = muse_norm2 * (1 + muse_scale_mlp) + muse_shift_mlp
        
        compare_tensors("norm2", diff_norm2, muse_norm2)
        compare_tensors("norm2_modulated", diff_norm2_mod, muse_norm2_mod)
        
        # FFN input reshaping
        # Diffusers: unflatten to (H, W) and permute to NCHW
        diff_ff_input = diff_norm2_mod.unflatten(1, (H, W)).permute(0, 3, 1, 2)
        
        # Muse: reshape to (H, W) and permute to NCHW  
        muse_ff_input = muse_norm2_mod.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        compare_tensors("ff_input_nchw", diff_ff_input, muse_ff_input)
        
        # ====== STEP 5b: Detailed GLUMBConv comparison ======
        print("\n  [5b] GLUMBConv step-by-step...")
        
        diff_ff = diff_block.ff
        muse_ff = muse_block.mlp
        
        # Compare weights first
        print("\n    Checking weights...")
        compare_tensors("conv_inverted.weight", diff_ff.conv_inverted.weight, muse_ff.inverted_conv.conv.weight)
        compare_tensors("conv_inverted.bias", diff_ff.conv_inverted.bias, muse_ff.inverted_conv.conv.bias)
        compare_tensors("conv_depth.weight", diff_ff.conv_depth.weight, muse_ff.depth_conv.conv.weight)
        compare_tensors("conv_depth.bias", diff_ff.conv_depth.bias, muse_ff.depth_conv.conv.bias)
        compare_tensors("conv_point.weight", diff_ff.conv_point.weight, muse_ff.point_conv.conv.weight)
        
        print("\n    Checking intermediate activations...")
        
        # Step 1: conv_inverted (before SiLU)
        diff_after_inv_conv = diff_ff.conv_inverted(diff_ff_input.clone())
        muse_after_inv_conv = muse_ff.inverted_conv.conv(muse_ff_input.clone())
        compare_tensors("after_conv_inverted (before SiLU)", diff_after_inv_conv, muse_after_inv_conv)
        
        # Step 2: after SiLU (clone to avoid inplace issues)
        diff_after_silu = diff_ff.nonlinearity(diff_after_inv_conv.clone())
        # Muse uses SiLU(inplace=True), so use F.silu to avoid inplace modification
        muse_after_silu = F.silu(muse_after_inv_conv.clone())
        compare_tensors("after_inverted_silu", diff_after_silu, muse_after_silu)
        
        # Step 3: after depth_conv
        diff_after_depth = diff_ff.conv_depth(diff_after_silu.clone())
        muse_after_depth = muse_ff.depth_conv.conv(muse_after_silu.clone())
        compare_tensors("after_depth_conv", diff_after_depth, muse_after_depth)
        
        # Step 4: after chunk
        diff_x_chunk, diff_gate = torch.chunk(diff_after_depth.clone(), 2, dim=1)
        muse_x_chunk, muse_gate = torch.chunk(muse_after_depth.clone(), 2, dim=1)
        compare_tensors("chunk_x", diff_x_chunk, muse_x_chunk)
        compare_tensors("chunk_gate", diff_gate, muse_gate)
        
        # Step 5: after gate activation
        diff_gate_act = diff_ff.nonlinearity(diff_gate.clone())
        muse_gate_act = F.silu(muse_gate.clone())  # glu_act is SiLU(inplace=False), but use F.silu for consistency
        compare_tensors("gate_after_silu", diff_gate_act, muse_gate_act)
        
        # Step 6: after multiplication
        diff_mult = diff_x_chunk * diff_gate_act
        muse_mult = muse_x_chunk * muse_gate_act
        compare_tensors("after_glu_mult", diff_mult, muse_mult)
        
        # Step 7: after point_conv
        diff_after_point = diff_ff.conv_point(diff_mult.clone())
        muse_after_point = muse_ff.point_conv.conv(muse_mult.clone())
        compare_tensors("after_point_conv", diff_after_point, muse_after_point)
        
        # Full FFN output comparison
        print("\n    Full FFN output...")
        diff_ff_out = diff_block.ff(diff_ff_input)
        diff_ff_out = diff_ff_out.flatten(2, 3).permute(0, 2, 1)
        
        muse_ff_out = muse_block.mlp(muse_norm2_mod, HW=(H, W))
        
        compare_tensors("ff_output", diff_ff_out, muse_ff_out)
        
        # After FFN residual
        diff_x_after_ff = diff_x_after_cross + diff_gate_mlp * diff_ff_out
        muse_x_after_ff = muse_x_after_cross + muse_gate_mlp * muse_ff_out
        
        compare_tensors("after_ff (block 0 output)", diff_x_after_ff, muse_x_after_ff)
        
        # ====== STEP 6: Check against full block forward ======
        print("\n[Step 6] Verify against full block forward...")
        
        # Run full block
        diff_block_out = diff_block(
            diff_x,
            encoder_hidden_states=diff_caption,
            encoder_attention_mask=encoder_attention_mask,
            timestep=diff_time,
            height=H,
            width=W,
        )
        
        muse_block_out = muse_block(muse_x, y_for_cross, muse_t0, y_lens, (H, W))
        
        compare_tensors("full_block_output", diff_block_out, muse_block_out)
        
        print("\n" + "=" * 70)


def compare_state_dicts(diffusers_model, muse_model):
    """Compare state dict keys and values between models."""
    print("\n[State Dict Comparison]")
    
    diff_sd = diffusers_model.state_dict()
    muse_sd = muse_model.state_dict()
    
    print(f"  Diffusers keys: {len(diff_sd)}")
    print(f"  Muse keys: {len(muse_sd)}")
    
    # Print some raw diffusers keys for debugging
    print("\n  Sample raw diffusers keys:")
    for i, key in enumerate(sorted(diff_sd.keys())):
        if i < 20:
            print(f"    {key}")
    
    # Print some muse keys for debugging
    print("\n  Sample muse keys:")
    for i, key in enumerate(sorted(muse_sd.keys())):
        if i < 20:
            print(f"    {key}")
    
    # Try to match keys using model's converter
    converted = muse_model.convert_diffusers_state_dict(diff_sd)
    
    matched = 0
    unmatched_diff = []
    unmatched_muse = []
    
    for key in converted:
        if key in muse_sd:
            matched += 1
        else:
            unmatched_diff.append(key)
    
    for key in muse_sd:
        if key not in converted:
            unmatched_muse.append(key)
    
    print(f"\n  Matched keys: {matched}")
    print(f"  Unmatched from diffusers (after conversion): {len(unmatched_diff)}")
    if unmatched_diff[:10]:
        print(f"    Examples: {unmatched_diff[:10]}")
    print(f"  Unmatched in muse: {len(unmatched_muse)}")
    if unmatched_muse[:10]:
        print(f"    Examples: {unmatched_muse[:10]}")


def run_full_alignment_test():
    """Run complete layer-by-layer alignment test."""
    print("=" * 70)
    print("Sana Forward Alignment Test - Diffusers vs Muse")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"Checkpoint: {DIFFUSERS_CHECKPOINT_PATH}")
    
    # Check path
    if not os.path.exists(DIFFUSERS_CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {DIFFUSERS_CHECKPOINT_PATH}")
        return False
    
    # Load diffusers model
    print("\n[Loading] Loading diffusers SanaTransformer2DModel...")
    try:
        diffusers_model = load_diffusers_model(DIFFUSERS_CHECKPOINT_PATH, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in diffusers_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load muse model
    print("\n[Loading] Loading muse SanaModel...")
    try:
        diffusers_state_dict = diffusers_model.state_dict()
        muse_model = load_muse_model(diffusers_state_dict, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in muse_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare state dicts
    compare_state_dicts(diffusers_model, muse_model)
    
    # Create inputs
    print("\n[Setup] Creating test inputs...")
    inputs = create_test_inputs(device=device, dtype=dtype)
    print(f"  x: {inputs['x'].shape}")
    print(f"  timestep: {inputs['timestep']}")
    print(f"  y_diffusers: {inputs['y_diffusers'].shape}")
    print(f"  y_muse: {inputs['y_muse'].shape}")
    
    all_passed = True
    
    # Run forward pass on both models
    print("\n[Forward] Running forward pass...")
    
    with torch.no_grad():
        # Diffusers forward
        diffusers_out = diffusers_model(
            hidden_states=inputs["x"],
            encoder_hidden_states=inputs["y_diffusers"],
            timestep=inputs["timestep"].float(),
            encoder_attention_mask=inputs["mask_diffusers"],
            return_dict=False,
        )[0]
        
        # Muse forward
        muse_out = muse_model(
            x=inputs["x"],
            timestep=inputs["timestep"].float(),
            y=inputs["y_muse"],
            mask=inputs["mask_muse"],
        )
    
    print(f"\n  Diffusers output: {diffusers_out.shape}")
    print(f"  Muse output: {muse_out.shape}")
    
    passed = compare_tensors("Full Model Output", diffusers_out, muse_out)
    all_passed = all_passed and passed
    
    # Layer-by-layer comparison
    print("\n[Layer Comparison]")
    
    with torch.no_grad():
        x = inputs["x"].to(dtype)
        
        # 1. Patch embedding
        print("\n  [1] Patch Embedding...")
        diff_patch = diffusers_model.patch_embed(x)
        muse_patch = muse_model.x_embedder(x)
        passed = compare_tensors("patch_embed", diff_patch, muse_patch)
        all_passed = all_passed and passed
        
        # 2. Timestep embedding
        print("\n  [2] Timestep Embedding...")
        timestep_float = inputs["timestep"].float()
        
        # Diffusers time_embed returns (shift_scale, embedded_timestep)
        diff_time, diff_t_emb = diffusers_model.time_embed(
            timestep_float, 
            batch_size=inputs["x"].shape[0],
            hidden_dtype=dtype
        )
        
        # Muse separate t_embedder and t_block
        muse_t = muse_model.t_embedder(timestep_float.long().float())
        muse_t0 = muse_model.t_block(muse_t)
        
        compare_tensors("t_embedder", diff_t_emb, muse_t)
        compare_tensors("t_block (6*dim)", diff_time, muse_t0)
        
        # 3. Caption embedding
        print("\n  [3] Caption Embedding...")
        y_diff = inputs["y_diffusers"]
        y_muse = inputs["y_muse"]
        
        diff_caption = diffusers_model.caption_projection(y_diff)
        diff_caption = diff_caption.view(y_diff.shape[0], -1, diff_patch.shape[-1])
        diff_caption = diffusers_model.caption_norm(diff_caption)
        
        muse_caption = muse_model.y_embedder(y_muse, False)
        if muse_model.y_norm:
            muse_caption = muse_model.attention_y_norm(muse_caption)
        muse_caption = muse_caption.squeeze(1)  # Remove extra dim for comparison
        
        compare_tensors("caption_embed", diff_caption, muse_caption)
    
    # Debug first block
    debug_first_block(diffusers_model, muse_model, inputs, dtype)
    
    # Debug all transformer blocks and final layer
    print("\n[All Blocks & Final Layer Debug]")
    
    with torch.no_grad():
        # Prepare inputs like in forward
        x = inputs["x"].to(dtype)
        timestep = inputs["timestep"].float()
        
        # Patch embedding
        diff_x = diffusers_model.patch_embed(x)
        muse_x = muse_model.x_embedder(x)
        
        patch_size = muse_model.patch_size
        h, w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        
        # Note: Not adding position embedding here (same as debug_first_block)
        # because we want to test block-by-block equivalence
        
        # Timestep embedding
        diff_time, diff_t_emb = diffusers_model.time_embed(
            timestep, batch_size=x.shape[0], hidden_dtype=dtype
        )
        muse_t = muse_model.t_embedder(timestep.long().float())
        muse_t0 = muse_model.t_block(muse_t)
        
        # Caption embedding
        y_diff = inputs["y_diffusers"]
        y_muse = inputs["y_muse"]
        
        diff_caption = diffusers_model.caption_projection(y_diff)
        diff_caption = diff_caption.view(y_diff.shape[0], -1, diff_x.shape[-1])
        diff_caption = diffusers_model.caption_norm(diff_caption)
        
        muse_caption = muse_model.y_embedder(y_muse, False)
        if muse_model.y_norm:
            muse_caption = muse_model.attention_y_norm(muse_caption)
        
        # Prepare mask/y for muse cross-attention
        _xformers_available = False
        try:
            import xformers.ops
            _xformers_available = True
        except ImportError:
            pass
        
        mask = inputs["mask_muse"].clone()
        y_for_cross = muse_caption.clone()
        if mask is not None:
            mask = mask.to(torch.int16)
            mask = mask.squeeze(1).squeeze(1) if mask.ndim > 2 else mask
            if _xformers_available:
                y_for_cross = y_for_cross.squeeze(1) if y_for_cross.ndim == 4 else y_for_cross
                y_for_cross = y_for_cross.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, muse_x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = mask
                y_for_cross = y_for_cross.squeeze(1) if y_for_cross.ndim == 4 else y_for_cross
        
        # encoder_attention_mask for diffusers
        encoder_attention_mask = inputs["mask_diffusers"]
        if encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        
        # Run all transformer blocks
        print("\n  Running all transformer blocks...")
        for i, (diff_block, muse_block) in enumerate(zip(diffusers_model.transformer_blocks, muse_model.blocks)):
            # Debug block 1 before running
            if i == 1:
                print(f"\n  [Debug Block 1]")
                # 1. Compare input to block 1 (should be block 0 output)
                compare_tensors(f"block1_input", diff_x, muse_x)
                
                # 2. Compare block 1 self-attention weights
                print(f"\n  Self-Attention weights:")
                compare_tensors(f"block1_to_q_weight", diff_block.attn1.to_q.weight, muse_block.attn.to_q.weight)
                compare_tensors(f"block1_to_k_weight", diff_block.attn1.to_k.weight, muse_block.attn.to_k.weight)
                compare_tensors(f"block1_to_v_weight", diff_block.attn1.to_v.weight, muse_block.attn.to_v.weight)
                compare_tensors(f"block1_proj_weight", diff_block.attn1.to_out[0].weight, muse_block.attn.proj.weight)
                # QK norm weights
                if hasattr(diff_block.attn1, 'norm_q') and diff_block.attn1.norm_q is not None:
                    compare_tensors(f"block1_norm_q_weight", diff_block.attn1.norm_q.weight, muse_block.attn.q_norm.weight)
                    compare_tensors(f"block1_norm_k_weight", diff_block.attn1.norm_k.weight, muse_block.attn.k_norm.weight)
                
                # 3. Compare cross-attention weights
                print(f"\n  Cross-Attention weights:")
                compare_tensors(f"block1_cross_q_weight", diff_block.attn2.to_q.weight, muse_block.cross_attn.q_linear.weight)
                compare_tensors(f"block1_cross_k_weight", diff_block.attn2.to_k.weight, muse_block.cross_attn.to_k.weight)
                compare_tensors(f"block1_cross_v_weight", diff_block.attn2.to_v.weight, muse_block.cross_attn.to_v.weight)
                compare_tensors(f"block1_cross_proj_weight", diff_block.attn2.to_out[0].weight, muse_block.cross_attn.proj.weight)
                
                # 4. Compare FFN/MLP weights
                print(f"\n  FFN/MLP weights:")
                # diffusers: ff.conv_inverted, ff.conv_depth, ff.conv_point
                # muse: mlp.inverted_conv.conv, mlp.depth_conv.conv, mlp.point_conv.conv
                compare_tensors(f"block1_ff_inverted_weight", diff_block.ff.conv_inverted.weight, muse_block.mlp.inverted_conv.conv.weight)
                compare_tensors(f"block1_ff_depth_weight", diff_block.ff.conv_depth.weight, muse_block.mlp.depth_conv.conv.weight)
                compare_tensors(f"block1_ff_point_weight", diff_block.ff.conv_point.weight, muse_block.mlp.point_conv.conv.weight)
                
                # 5. Compare scale_shift_table
                print(f"\n  Other weights:")
                compare_tensors(f"block1_scale_shift_table", diff_block.scale_shift_table, muse_block.scale_shift_table)
                
                # 6. Step-by-step forward for block 1
                print(f"\n  Block 1 Step-by-Step Forward:")
                B = diff_x.shape[0]
                C = diff_x.shape[-1]
                
                # Modulation
                diff_shift_msa, diff_scale_msa, diff_gate_msa, diff_shift_mlp, diff_scale_mlp, diff_gate_mlp = (
                    diff_block.scale_shift_table[None] + diff_time.reshape(B, 6, -1)
                ).chunk(6, dim=1)
                muse_shift_msa, muse_scale_msa, muse_gate_msa, muse_shift_mlp, muse_scale_mlp, muse_gate_mlp = (
                    muse_block.scale_shift_table[None] + muse_t0.reshape(B, 6, -1)
                ).chunk(6, dim=1)
                compare_tensors("block1_modulation", diff_shift_msa, muse_shift_msa)
                
                # Norm + modulate for self-attention
                diff_norm1 = diff_block.norm1(diff_x)
                muse_norm1 = muse_block.norm1(muse_x)
                compare_tensors("block1_norm1", diff_norm1, muse_norm1)
                
                diff_norm1_mod = diff_norm1 * (1 + diff_scale_msa) + diff_shift_msa
                muse_norm1_mod = muse_norm1 * (1 + muse_scale_msa) + muse_shift_msa
                compare_tensors("block1_norm1_mod", diff_norm1_mod, muse_norm1_mod)
                
                # Self-attention Q, K, V projection
                diff_q = diff_block.attn1.to_q(diff_norm1_mod)
                diff_k = diff_block.attn1.to_k(diff_norm1_mod)
                diff_v = diff_block.attn1.to_v(diff_norm1_mod)
                muse_q = muse_block.attn.to_q(muse_norm1_mod)
                muse_k = muse_block.attn.to_k(muse_norm1_mod)
                muse_v = muse_block.attn.to_v(muse_norm1_mod)
                compare_tensors("block1_self_attn_q", diff_q, muse_q)
                compare_tensors("block1_self_attn_k", diff_k, muse_k)
                compare_tensors("block1_self_attn_v", diff_v, muse_v)
                
                # QK norm
                diff_q_normed = diff_block.attn1.norm_q(diff_q)
                diff_k_normed = diff_block.attn1.norm_k(diff_k)
                muse_q_normed = muse_block.attn.q_norm(muse_q)
                muse_k_normed = muse_block.attn.k_norm(muse_k)
                compare_tensors("block1_q_normed", diff_q_normed, muse_q_normed)
                compare_tensors("block1_k_normed", diff_k_normed, muse_k_normed)
                
                # Reshape for attention
                num_heads = diff_block.attn1.heads
                head_dim = diff_q.shape[-1] // num_heads
                diff_q_reshape = diff_q_normed.view(B, -1, num_heads, head_dim).transpose(1, 2)
                diff_k_reshape = diff_k_normed.view(B, -1, num_heads, head_dim).transpose(1, 2)
                diff_v_reshape = diff_v.view(B, -1, num_heads, head_dim).transpose(1, 2)
                muse_q_reshape = muse_q_normed.view(B, -1, num_heads, head_dim).transpose(1, 2)
                muse_k_reshape = muse_k_normed.view(B, -1, num_heads, head_dim).transpose(1, 2)
                muse_v_reshape = muse_v.view(B, -1, num_heads, head_dim).transpose(1, 2)
                compare_tensors("block1_q_reshape", diff_q_reshape, muse_q_reshape)
                
                # Self-attention computation
                diff_attn_out = F.scaled_dot_product_attention(diff_q_reshape, diff_k_reshape, diff_v_reshape)
                muse_attn_out = F.scaled_dot_product_attention(muse_q_reshape, muse_k_reshape, muse_v_reshape)
                compare_tensors("block1_self_attn_sdpa", diff_attn_out, muse_attn_out)
                
                # Self-attention output projection
                diff_attn_out_t = diff_attn_out.transpose(1, 2).reshape(B, -1, C)
                muse_attn_out_t = muse_attn_out.transpose(1, 2).reshape(B, -1, C)
                diff_attn_proj = diff_block.attn1.to_out[0](diff_attn_out_t)
                muse_attn_proj = muse_block.attn.proj(muse_attn_out_t)
                compare_tensors("block1_self_attn_proj", diff_attn_proj, muse_attn_proj)
                
                # After self-attention (with gate and residual)
                diff_after_attn = diff_x + diff_gate_msa * diff_attn_proj
                muse_after_attn = muse_x + muse_gate_msa * muse_attn_proj
                compare_tensors("block1_after_self_attn", diff_after_attn, muse_after_attn)
                
                # Cross-attention - step by step
                print("\n  Block 1 Cross-Attention Step-by-Step:")
                
                # First verify inputs are identical
                compare_tensors("block1_cross_input_verify", diff_after_attn, muse_after_attn)
                
                # Check Q linear weights AND bias
                compare_tensors("block1_cross_q_weight_verify", diff_block.attn2.to_q.weight, muse_block.cross_attn.q_linear.weight)
                if diff_block.attn2.to_q.bias is not None and muse_block.cross_attn.q_linear.bias is not None:
                    compare_tensors("block1_cross_q_bias", diff_block.attn2.to_q.bias, muse_block.cross_attn.q_linear.bias)
                else:
                    print(f"  Q bias: diff={diff_block.attn2.to_q.bias is not None}, muse={muse_block.cross_attn.q_linear.bias is not None}")
                
                # Q from hidden states
                diff_cross_q = diff_block.attn2.to_q(diff_after_attn)
                muse_cross_q = muse_block.cross_attn.q_linear(muse_after_attn)
                compare_tensors("block1_cross_q", diff_cross_q, muse_cross_q)
                
                # Manual computation to verify
                diff_cross_q_manual = F.linear(diff_after_attn, diff_block.attn2.to_q.weight, diff_block.attn2.to_q.bias)
                muse_cross_q_manual = F.linear(muse_after_attn, muse_block.cross_attn.q_linear.weight, muse_block.cross_attn.q_linear.bias)
                compare_tensors("block1_cross_q_manual", diff_cross_q_manual, muse_cross_q_manual)
                
                # Check if there's any difference between the module call and manual F.linear
                compare_tensors("block1_diff_q_module_vs_manual", diff_cross_q, diff_cross_q_manual)
                compare_tensors("block1_muse_q_module_vs_manual", muse_cross_q, muse_cross_q_manual)
                
                # CRITICAL: Use SAME input with both weights to isolate the issue
                print("\n  Isolating the difference:")
                print(f"  diff_after_attn shape: {diff_after_attn.shape}, contiguous: {diff_after_attn.is_contiguous()}, stride: {diff_after_attn.stride()}")
                print(f"  muse_after_attn shape: {muse_after_attn.shape}, contiguous: {muse_after_attn.is_contiguous()}, stride: {muse_after_attn.stride()}")
                
                # Use diff input with both weights
                q_with_diff_weight = F.linear(diff_after_attn, diff_block.attn2.to_q.weight, diff_block.attn2.to_q.bias)
                q_with_muse_weight = F.linear(diff_after_attn, muse_block.cross_attn.q_linear.weight, muse_block.cross_attn.q_linear.bias)
                compare_tensors("block1_same_input_diff_weight", q_with_diff_weight, q_with_muse_weight)
                
                # Use muse input with both weights
                q_with_diff_weight2 = F.linear(muse_after_attn, diff_block.attn2.to_q.weight, diff_block.attn2.to_q.bias)
                q_with_muse_weight2 = F.linear(muse_after_attn, muse_block.cross_attn.q_linear.weight, muse_block.cross_attn.q_linear.bias)
                compare_tensors("block1_same_input_muse_weight", q_with_diff_weight2, q_with_muse_weight2)
                
                # Check if the actual tensor values are truly identical (element-wise)
                input_diff = (diff_after_attn - muse_after_attn).abs()
                print(f"  Input actual max diff: {input_diff.max().item()}, mean diff: {input_diff.mean().item()}")
                
                weight_diff = (diff_block.attn2.to_q.weight - muse_block.cross_attn.q_linear.weight).abs()
                print(f"  Weight actual max diff: {weight_diff.max().item()}, mean diff: {weight_diff.mean().item()}")
                
                bias_diff = (diff_block.attn2.to_q.bias - muse_block.cross_attn.q_linear.bias).abs()
                print(f"  Bias actual max diff: {bias_diff.max().item()}, mean diff: {bias_diff.mean().item()}")
                
                # K, V from encoder hidden states (caption)
                # Handle muse_caption shape if needed
                muse_caption_for_cross = muse_caption.squeeze(1) if muse_caption.ndim == 4 else muse_caption
                
                diff_cross_k = diff_block.attn2.to_k(diff_caption)
                diff_cross_v = diff_block.attn2.to_v(diff_caption)
                muse_cross_k = muse_block.cross_attn.to_k(muse_caption_for_cross)
                muse_cross_v = muse_block.cross_attn.to_v(muse_caption_for_cross)
                compare_tensors("block1_cross_k", diff_cross_k, muse_cross_k)
                compare_tensors("block1_cross_v", diff_cross_v, muse_cross_v)
                
                # QK norm
                diff_cross_q_normed = diff_block.attn2.norm_q(diff_cross_q)
                diff_cross_k_normed = diff_block.attn2.norm_k(diff_cross_k)
                muse_cross_q_normed = muse_block.cross_attn.q_norm(muse_cross_q)
                muse_cross_k_normed = muse_block.cross_attn.k_norm(muse_cross_k)
                compare_tensors("block1_cross_q_normed", diff_cross_q_normed, muse_cross_q_normed)
                compare_tensors("block1_cross_k_normed", diff_cross_k_normed, muse_cross_k_normed)
                
                # Full cross-attention output
                diff_cross_out = diff_block.attn2(diff_after_attn, encoder_hidden_states=diff_caption)
                muse_cross_out = muse_block.cross_attn(muse_after_attn, muse_caption)
                compare_tensors("block1_cross_attn_out", diff_cross_out, muse_cross_out)
                
                # After cross-attention (with residual)
                diff_after_cross = diff_after_attn + diff_cross_out
                muse_after_cross = muse_after_attn + muse_cross_out
                compare_tensors("block1_after_cross_attn", diff_after_cross, muse_after_cross)
                
                # FFN/MLP
                diff_norm2 = diff_block.norm2(diff_after_cross)
                muse_norm2 = muse_block.norm2(muse_after_cross)
                compare_tensors("block1_norm2", diff_norm2, muse_norm2)
                
                diff_norm2_mod = diff_norm2 * (1 + diff_scale_mlp) + diff_shift_mlp
                muse_norm2_mod = muse_norm2 * (1 + muse_scale_mlp) + muse_shift_mlp
                compare_tensors("block1_norm2_mod", diff_norm2_mod, muse_norm2_mod)
                
                # FFN forward
                # diffusers ff expects (B, C, H, W) format
                diff_ff_in = diff_norm2_mod.unflatten(1, (h, w)).permute(0, 3, 1, 2)  # B, C, H, W
                diff_ff_out = diff_block.ff(diff_ff_in)
                diff_ff_out_flat = diff_ff_out.permute(0, 2, 3, 1).flatten(1, 2)  # B, N, C
                
                # muse mlp expects (B, N, C) format
                muse_ff_out = muse_block.mlp(muse_norm2_mod)
                
                compare_tensors("block1_ff_output", diff_ff_out_flat, muse_ff_out)
                
                # After FFN (with gate and residual)
                diff_final = diff_after_cross + diff_gate_mlp * diff_ff_out_flat
                muse_final = muse_after_cross + muse_gate_mlp * muse_ff_out
                compare_tensors("block1_final_output", diff_final, muse_final)
            
            # Run blocks
            diff_out = diff_block(
                diff_x,
                encoder_hidden_states=diff_caption,
                encoder_attention_mask=encoder_attention_mask,
                timestep=diff_time,
                height=h,
                width=w,
            )
            muse_x = muse_block(muse_x, y_for_cross, muse_t0, y_lens, (h, w))
            
            # Check if diffusers returns tuple
            if isinstance(diff_out, tuple):
                print(f"  Block {i} diffusers output is tuple of length {len(diff_out)}")
                diff_x = diff_out[0]
            else:
                diff_x = diff_out
            
            # Compare every block to find where divergence starts
            if i < 3 or i == len(muse_model.blocks) - 1:  # First 3 and last block
                compare_tensors(f"block_{i}_output", diff_x, muse_x)
        
        # Compare output after all blocks
        compare_tensors("all_blocks_output", diff_x, muse_x)
        
        # Final layer comparison
        print("\n  Final Layer...")
        
        # Diffusers: norm_out + scale_shift_table + proj_out
        # norm_out needs: hidden_states, temb, scale_shift_table
        diff_norm_out = diffusers_model.norm_out(diff_x, diff_t_emb, diffusers_model.scale_shift_table)
        diff_final_out = diffusers_model.proj_out(diff_norm_out)
        
        # Muse: final_layer
        muse_final_out = muse_model.final_layer(muse_x, muse_t)
        
        compare_tensors("final_layer_output", diff_final_out, muse_final_out)
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("SUCCESS: All comparisons passed within tolerance!")
    else:
        print("FAILED: Some comparisons failed.")
    print("=" * 70)
    
    return all_passed


def run_muse_only_test():
    """Run a quick test of the muse model only."""
    print("=" * 60)
    print("Muse Sana Model Test (standalone)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    if not os.path.exists(DIFFUSERS_CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {DIFFUSERS_CHECKPOINT_PATH}")
        return
    
    print(f"\nCheckpoint: {DIFFUSERS_CHECKPOINT_PATH}")
    
    # Load diffusers model to get state dict
    print("\n[1/3] Loading diffusers model for state dict...")
    try:
        from diffusers.models import SanaTransformer2DModel
        transformer_path = os.path.join(DIFFUSERS_CHECKPOINT_PATH, "transformer")
        if os.path.exists(transformer_path):
            diffusers_model = SanaTransformer2DModel.from_pretrained(transformer_path, torch_dtype=dtype)
        else:
            diffusers_model = SanaTransformer2DModel.from_pretrained(DIFFUSERS_CHECKPOINT_PATH, torch_dtype=dtype)
        diffusers_state_dict = diffusers_model.state_dict()
        del diffusers_model
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[2/3] Loading muse Sana model...")
    try:
        muse_model = load_muse_model(diffusers_state_dict, device, dtype)
        print(f"  Parameters: {sum(p.numel() for p in muse_model.parameters()):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[3/3] Running forward pass...")
    inputs = create_test_inputs(device=device, dtype=dtype)
    
    try:
        with torch.no_grad():
            output = muse_model(
                x=inputs["x"],
                timestep=inputs["timestep"].float(),
                y=inputs["y_muse"],
                mask=inputs["mask_muse"],
            )
        print(f"  Output shape: {output.shape}")
        print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Muse model test PASSED!")
    print("=" * 60)


# Pytest test classes
@pytest.mark.skipif(
    not os.path.exists(DIFFUSERS_CHECKPOINT_PATH),
    reason=f"Checkpoint not found at {DIFFUSERS_CHECKPOINT_PATH}"
)
class TestSanaAlignment:
    """Test suite for Sana alignment."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
    
    def test_full_alignment(self):
        """Test full model alignment."""
        result = run_full_alignment_test()
        assert result, "Alignment test failed"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--muse-only", action="store_true", help="Only test muse model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    args = parser.parse_args()
    
    if args.checkpoint:
        DIFFUSERS_CHECKPOINT_PATH = args.checkpoint
    
    if args.muse_only:
        run_muse_only_test()
    else:
        run_full_alignment_test()
