"""
Integration test to ensure Muse KeyeVisionTransformer matches Hugging Face (Origin) implementation.
"""

import os
import sys
import types
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Ensure repository root (containing the `muse` package) is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Muse imports
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionModel as MuseKeyeVisionModel
from tests.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.training.common import set_default_dtype

# === Mock/Import HF Configs to support Origin Model loading ===
from transformers import PretrainedConfig

class HFKeyeVisionConfig(PretrainedConfig):
    model_type = "siglip_vision"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items(): setattr(self, k, v)

class HFKeyeConfig(PretrainedConfig):
    model_type = "keye"
    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vision_config = vision_config

def _ensure_origin_ready():
    # Helper to inject config classes so Origin model can import them
    mod = "tests.models.keye_vit.configuration_keye"
    if mod not in sys.modules:
        c = types.ModuleType(mod)
        c.KeyeConfig = HFKeyeConfig
        c.KeyeVisionConfig = HFKeyeVisionConfig
        sys.modules[mod] = c

_ensure_origin_ready()
# Import the Reference Implementation (Origin)
# Assuming this file exists in your path as per previous debug sessions
from tests.models.keye_vit import modeling_keye_origin as keye_origin
from tests.models.keye_vit.modeling_keye_origin import _DEBUG_ROPE_OUTPUTS as ORIGIN_ROPE_DEBUG
OriginKeyeVisionModel = keye_origin.SiglipVisionModel 

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"


@contextmanager
def _mock_context_parallel():
    """Mock context parallel helpers so tests can run without torch.distributed init."""
    patches = [
        patch("muse.training.parallel.get_context_parallel_world_size", new=lambda: 1),
        patch("muse.training.parallel.get_context_parallel_group", new=lambda backend="nccl": None),
        patch("muse.training.parallel.get_context_parallel_rank", new=lambda: 0),
        patch("muse.layers.attention.get_context_parallel_world_size", new=lambda: 1),
        patch("muse.layers.attention.get_context_parallel_group", new=lambda backend="nccl": None),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def _build_muse_config(hf_cfg: Dict[str, Any]) -> KeyeVisionConfig:
    """Map Hugging Face/Origin config dict to Muse KeyeVisionConfig."""
    return KeyeVisionConfig(
        hidden_size=hf_cfg.get("hidden_size", 1152),
        intermediate_size=hf_cfg.get("intermediate_size", 4304),
        num_hidden_layers=hf_cfg.get("num_hidden_layers", 27),
        num_attention_heads=hf_cfg.get("num_attention_heads", 16),
        num_channels=hf_cfg.get("num_channels", 3),
        image_size=hf_cfg.get("image_size", 384),
        patch_size=hf_cfg.get("patch_size", 14),
        layer_norm_eps=hf_cfg.get("layer_norm_eps", 1e-6),
        attention_dropout=hf_cfg.get("attention_dropout", 0.0),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        # Ensure we use eager for comparison to avoid kernel nondeterminism
        attention_function="flash_attention_2", 
        # Additional params
        use_qk_norm=hf_cfg.get("use_qk_norm", False),
        qk_norm_eps=hf_cfg.get("qk_norm_eps", 1e-6),
        has_learnable_position_embedding=hf_cfg.get("has_learnable_position_embedding", False)
    )

def create_dummy_image(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)


def log_header(title: str):
    logger.info(f"\n{'='*100}\n {title.center(98)} \n{'='*100}")


def compare_tensors(name: str, ref: torch.Tensor, cand: torch.Tensor, atol: float = 5e-2):
    """Compare tensors with automatic shape alignment to aid debugging."""
    if isinstance(ref, (tuple, list)):
        ref = ref[0]
    if isinstance(cand, (tuple, list)):
        cand = cand[0]

    ref = ref.detach().float().cpu()
    cand = cand.detach().float().cpu()

    if ref.shape != cand.shape:
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2:
            ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2:
            cand = cand.squeeze(0)

    if ref.shape != cand.shape and ref.numel() == cand.numel():
        if ref.dim() == 3 and ref.transpose(1, 2).shape == cand.shape:
            ref = ref.transpose(1, 2)

    if ref.shape != cand.shape:
        logger.error(f"{name:<35} | ❌ SHAPE MISMATCH: Origin {ref.shape} vs Muse {cand.shape}")
        return

    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    is_match = max_diff < atol
    tag = "✅ MATCH" if is_match else "❌ DIFF"

    logger.info(f"{name:<35} | {tag:<10} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    if not is_match:
        idx = torch.argmax(diff)
        logger.info(
            f"{' ':<35} | -> Max Diff Index: {idx.item()} "
            f"(Val: {ref.flatten()[idx]:.4f} vs {cand.flatten()[idx]:.4f})"
        )


def _load_checkpoint_state_dict(
    checkpoint_path: str, dtype: torch.dtype
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load checkpoint and return origin-formatted plus HF-prefixed weights."""
    raw_state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "module" in raw_state_dict:
        raw_state_dict = raw_state_dict["module"]

    origin_load_dict: Dict[str, torch.Tensor] = {}
    for k, v in raw_state_dict.items():
        clean_k = k
        for prefix in ["module.", "vision_tower.", "siglip."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix) :]
        if "vision_model" not in clean_k:
            clean_k = "vision_model." + clean_k
        origin_load_dict[clean_k] = v.to(dtype)

    hf_full_state_dict = {"siglip." + k: v for k, v in origin_load_dict.items()}
    return origin_load_dict, hf_full_state_dict

def check_weights(hf_state_dict: Dict[str, torch.Tensor], 
                  muse_model: nn.Module, 
                  device: torch.device, 
                  dtype: torch.dtype):
    """Compare weights layer by layer between HF dict and Muse model."""
    print(f"\n{'='*60}")
    print("Weight Value Comparison")
    print(f"{'='*60}")

    muse_state_dict = muse_model.state_dict()
    config = muse_model.config
    
    total_checked = 0
    total_matched = 0
    issues = []

    # 1. Check Embeddings
    # Origin: embeddings.patch_embedding.weight
    # Muse: embeddings.patch_embedding.weight
    embed_map = {
        "vision_model.embeddings.patch_embedding.weight": "embeddings.patch_embedding.weight",
        "vision_model.embeddings.patch_embedding.bias": "embeddings.patch_embedding.bias",
        "vision_model.embeddings.position_embedding.weight": "embeddings.position_embedding.weight",
    }
    
    for hf_key, muse_key in embed_map.items():
        # Remove 'siglip.' prefix if present in hf_state_dict keys for lookup
        lookup_key = "siglip." + hf_key
        if lookup_key in hf_state_dict and muse_key in muse_state_dict:
            total_checked += 1
            hf_w = hf_state_dict[lookup_key].to(device=device, dtype=dtype)
            muse_w = muse_state_dict[muse_key]
            diff = (hf_w - muse_w).abs().max().item()
            if diff > 1e-5:
                issues.append(f"{muse_key}: diff={diff:.6e}")
            else:
                total_matched += 1

    # 2. Check Encoder Layers
    for i in range(config.num_hidden_layers):
        # Mappings for one layer
        # Origin: vision_model.encoder.layers.0.self_attn.q_proj.weight
        # Muse: encoder.layers.0.attn.q_proj.weight
        layer_map = [
            # Norms
            (f"vision_model.encoder.layers.{i}.layer_norm1.weight", f"encoder.layers.{i}.sa_norm.weight"),
            (f"vision_model.encoder.layers.{i}.layer_norm1.bias",   f"encoder.layers.{i}.sa_norm.bias"),
            (f"vision_model.encoder.layers.{i}.layer_norm2.weight", f"encoder.layers.{i}.mlp_norm.weight"),
            (f"vision_model.encoder.layers.{i}.layer_norm2.bias",   f"encoder.layers.{i}.mlp_norm.bias"),
            # Attn
            (f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight", f"encoder.layers.{i}.attn.q_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias",   f"encoder.layers.{i}.attn.q_proj.bias"),
            (f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight", f"encoder.layers.{i}.attn.k_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias",   f"encoder.layers.{i}.attn.k_proj.bias"),
            (f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight", f"encoder.layers.{i}.attn.v_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias",   f"encoder.layers.{i}.attn.v_proj.bias"),
            (f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.attn.output_proj.weight"),
            (f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias",   f"encoder.layers.{i}.attn.output_proj.bias"),
            # MLP (KeyeMLP: fc1->gate_proj(w1), fc2->down_proj(w2))
            # Note: Muse FeedForward maps w1=gate, w2=down.
            (f"vision_model.encoder.layers.{i}.mlp.fc1.weight", f"encoder.layers.{i}.mlp.w1.weight"),
            (f"vision_model.encoder.layers.{i}.mlp.fc1.bias",   f"encoder.layers.{i}.mlp.w1.bias"),
            (f"vision_model.encoder.layers.{i}.mlp.fc2.weight", f"encoder.layers.{i}.mlp.w2.weight"),
            (f"vision_model.encoder.layers.{i}.mlp.fc2.bias",   f"encoder.layers.{i}.mlp.w2.bias"),
        ]

        for hf_k, muse_k in layer_map:
            lookup_key = "siglip." + hf_k
            if lookup_key in hf_state_dict and muse_k in muse_state_dict:
                total_checked += 1
                hf_w = hf_state_dict[lookup_key].to(device=device, dtype=dtype)
                muse_w = muse_state_dict[muse_k]
                
                # Reshape handling if needed (e.g. Linear vs Conv)
                # But here everything should align if convert_hf_state_dict is correct
                if hf_w.shape != muse_w.shape:
                     if hf_w.transpose(0,1).shape == muse_w.shape: hf_w = hf_w.transpose(0,1)

                diff = (hf_w - muse_w).abs().max().item()
                if diff > 1e-5:
                    issues.append(f"Layer {i} {muse_k}: diff={diff:.6e}")
                else:
                    total_matched += 1

    # 3. Final Norm
    final_map = {
        "vision_model.post_layernorm.weight": "ln_post.weight",
        "vision_model.post_layernorm.bias": "ln_post.bias"
    }
    for hf_k, muse_k in final_map.items():
        lookup_key = "siglip." + hf_k
        if lookup_key in hf_state_dict and muse_k in muse_state_dict:
            total_checked += 1
            hf_w = hf_state_dict[lookup_key].to(device=device, dtype=dtype)
            muse_w = muse_state_dict[muse_k]
            diff = (hf_w - muse_w).abs().max().item()
            if diff > 1e-5:
                issues.append(f"{muse_k}: diff={diff:.6e}")
            else:
                total_matched += 1

    print(f"Total weights checked: {total_checked}")
    print(f"Weights matched: {total_matched}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} weight mismatches:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10: print(f"  ... {len(issues)-10} more")
    else:
        print("✓ All checked weights match!")


def test_keye_vision_align_with_hf_checkpoint():
    """Ensure Muse KeyeVisionTransformer aligns with the Origin implementation."""
    with _mock_context_parallel():
        _run_keye_vision_align_with_hf_checkpoint()


def _run_keye_vision_align_with_hf_checkpoint():
    # === 1. Configuration ===
    checkpoint_path = os.environ.get("KEYE_VIT_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # Using BF16 as per your debugging requirement
    
    torch.manual_seed(0)
    
    print(f"Running alignment test on device={device}, dtype={dtype}")
    print(f"Checkpoint: {checkpoint_path}")

    # === 2. Load Origin (HF-style) Model ===
    print("\nLoading Origin Model...")
    # Use default config structure, populated with Muse defaults but HF class
    muse_dummy_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_dummy_config.dict())
    
    # Initialize empty Origin model
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config)
    
    # Load Weights manually from .pt
    origin_load_dict, hf_full_state_dict = _load_checkpoint_state_dict(checkpoint_path, dtype)

    # Load into Origin
    missing, unexpected = origin_model.load_state_dict(origin_load_dict, strict=False)
    # Ignore missing text model keys
    unexpected = [k for k in unexpected if "text_model" not in k]
    if len(unexpected) > 0:
        print(f"Origin Model Unexpected: {unexpected[:5]}...")

    origin_model.to(device, dtype)
    origin_model.eval()

    # === 3. Initialize Muse Model ===
    print("\nInitializing Muse Model...")
    muse_config = _build_muse_config(origin_config.to_dict())
    
    with set_default_dtype(dtype):
        muse_model = MuseKeyeVisionModel(muse_config)

    # === 4. Weight Conversion & Loading ===
    # We construct a "full" HF state dict to pass to converter
    print("Converting weights...")
    converted_state_dict = muse_model.convert_hf_state_dict(hf_full_state_dict)
    
    # Load into Muse
    m_missing, m_unexpected = muse_model.load_state_dict(converted_state_dict, strict=False)
    if m_missing: print(f"Muse Missing: {m_missing}")
    if m_unexpected: print(f"Muse Unexpected: {m_unexpected}")

    muse_model.to(device, dtype)
    muse_model.eval()

    # === 5. Weight Verification ===
    check_weights(hf_full_state_dict, muse_model, device, dtype)

    # === 6. Input Preparation ===
    print("\nPreparing Inputs...")
    processor = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    image = create_dummy_image(muse_config.image_size)
    
    processed = processor.preprocess(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    
    # Grid info
    grid_thw = processed["image_grid_thw"]
    if isinstance(grid_thw, torch.Tensor): grid_thw = grid_thw.tolist()
    image_grid_thw = [tuple(int(x) for x in g) for g in grid_thw]
    
    # Pack Inputs for Muse (5D tensor: [1, Seq, C, H, W])
    # Note: Origin likely needs this too based on previous debugs
    num_patches = [int(np.prod(g)) for g in image_grid_thw]
    seq_len = num_patches[0]
    
    # pixel_values from processor is [Seq, 3, 14, 14]
    # Muse expects [1, Seq, 3, 14, 14] for batch=1
    pixel_inputs = pixel_values.unsqueeze(0).to(device, dtype)
    
    # Position IDs & Cu Seqlens
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # === 7. Forward Pass & Comparison ===
    print(f"\n{'='*60}")
    print("Forward Pass & Hidden States Comparison")
    print(f"{'='*60}")

    with torch.no_grad():
        print("Running Origin Model Forward...")
        # Origin signature based on debug history
        origin_out = origin_model(
            pixel_inputs, 
            position_ids=position_ids, 
            image_grid_thw=image_grid_thw, 
            cu_seqlens=cu_seqlens, 
            interpolate_pos_encoding=True, 
            window_size=-1, 
            use_rope=True
        )
        # Extract tensor
        if hasattr(origin_out, "last_hidden_state"):
            hf_output = origin_out.last_hidden_state
        else:
            hf_output = origin_out
        
        # Handle list output
        if isinstance(hf_output, list):
            hf_output = torch.stack(hf_output, dim=0)

        print("Running Muse Model Forward...")
        muse_out_dict = muse_model(
            pixel_inputs, 
            position_ids=position_ids, 
            image_grid_thw=image_grid_thw, 
            cu_seqlens=cu_seqlens, 
            interpolate_pos_encoding=True, 
            has_learnable_position_embedding=True
        )
        muse_output = muse_out_dict["last_hidden_state"]

        # Ensure comparison on same device/dtype
        hf_output = hf_output.to(device, dtype)
        muse_output = muse_output.to(device, dtype)

        # === 8. Statistics ===
        print(f"\nOutput Shapes:")
        print(f"  HF:   {hf_output.shape}")
        print(f"  Muse: {muse_output.shape}")

        if hf_output.shape != muse_output.shape:
             print("⚠️ Shape Mismatch! Attempting squeeze...")
             if hf_output.dim() == 3 and hf_output.shape[0] == 1: hf_output = hf_output.squeeze(0)
             if muse_output.dim() == 3 and muse_output.shape[0] == 1: muse_output = muse_output.squeeze(0)

        diff = (hf_output - muse_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Relative diff
        rel_diff = diff / (hf_output.abs() + 1e-8)
        max_rel = rel_diff.max().item()

        print(f"\nDifference Statistics:")
        print(f"  Max Absolute Diff: {max_diff:.6e}")
        print(f"  Mean Absolute Diff: {mean_diff:.6e}")
        print(f"  Max Relative Diff: {max_rel:.6e}")

        # === 9. Pass/Fail Decision ===
        # BF16 tolerance: usually around 1e-2 is acceptable for complex attention models
        # FP32 tolerance: should be < 1e-5
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        
        print(f"\nTolerance Check (thresh={tol}):")
        if max_diff < tol:
            print("✓✓✓ SUCCESS: Outputs match within tolerance!")
        else:
            print("✗ FAILURE: Outputs differ beyond tolerance.")
            # Debug info
            idx = torch.argmax(diff)
            flat_hf = hf_output.flatten()
            flat_muse = muse_output.flatten()
            print(f"  Max diff index: {idx.item()}")
            print(f"  HF Value:   {flat_hf[idx].item()}")
            print(f"  Muse Value: {flat_muse[idx].item()}")


def test_keye_vision_layer0_step_by_step():
    """Layer-by-layer debugger mirroring test_keye_vit_layer_step_by_step."""
    with _mock_context_parallel():
        _run_keye_vision_layer0_step_by_step()


def _run_keye_vision_layer0_step_by_step():
    checkpoint_path = os.environ.get("KEYE_VIT_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)
    logger.info(f"Layer debugger checkpoint: {checkpoint_path}")

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    muse_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_config.dict())

    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config).eval()
        muse_model = MuseKeyeVisionModel(muse_config).eval()

    origin_load_dict, hf_full_state_dict = _load_checkpoint_state_dict(checkpoint_path, dtype)
    logger.info("Loading weights into Origin and Muse models...")

    muse_state = muse_model.convert_hf_state_dict(hf_full_state_dict)
    muse_model.load_state_dict(muse_state, strict=False)
    origin_model.load_state_dict(origin_load_dict, strict=False)

    origin_model.to(device, dtype)
    muse_model.to(device, dtype)

    activations = {"origin": {}, "muse": {}}

    def make_hook(model_name, layer_name, capture_input=False):
        def hook(module, inp, out):
            target = inp if capture_input else out
            if isinstance(target, (tuple, list)):
                target = target[0]
            activations[model_name][layer_name] = target.detach()
        return hook

    origin_l0 = origin_model.vision_model.encoder.layers[0]
    muse_l0 = muse_model.encoder.layers[0]

    origin_l0.layer_norm1.register_forward_hook(make_hook("origin", "1. LN1 Output"))
    muse_l0.sa_norm.register_forward_hook(make_hook("muse", "1. LN1 Output"))

    origin_l0.self_attn.q_proj.register_forward_hook(make_hook("origin", "2. Q_Proj Out"))
    origin_l0.self_attn.k_proj.register_forward_hook(make_hook("origin", "2. K_Proj Out"))
    origin_l0.self_attn.v_proj.register_forward_hook(make_hook("origin", "2. V_Proj Out"))
    muse_l0.attn.q_proj.register_forward_hook(make_hook("muse", "2. Q_Proj Out"))
    muse_l0.attn.k_proj.register_forward_hook(make_hook("muse", "2. K_Proj Out"))
    muse_l0.attn.v_proj.register_forward_hook(make_hook("muse", "2. V_Proj Out"))

    origin_l0.self_attn.out_proj.register_forward_hook(
        make_hook("origin", "3. Attn Raw (Pre-Proj)", capture_input=True)
    )
    muse_l0.attn.output_proj.register_forward_hook(
        make_hook("muse", "3. Attn Raw (Pre-Proj)", capture_input=True)
    )

    origin_l0.self_attn.out_proj.register_forward_hook(make_hook("origin", "4. Attn Out (Post-Proj)"))
    muse_l0.attn.output_proj.register_forward_hook(make_hook("muse", "4. Attn Out (Post-Proj)"))

    origin_l0.layer_norm2.register_forward_hook(
        make_hook("origin", "5. Residual1 (LN2 In)", capture_input=True)
    )
    muse_l0.mlp_norm.register_forward_hook(
        make_hook("muse", "5. Residual1 (LN2 In)", capture_input=True)
    )

    origin_l0.mlp.fc1.register_forward_hook(make_hook("origin", "6. MLP Hidden (fc1)"))
    muse_l0.mlp.w1.register_forward_hook(make_hook("muse", "6. MLP Hidden (fc1)"))

    origin_l0.mlp.fc2.register_forward_hook(make_hook("origin", "7. MLP Out (fc2)"))
    muse_l0.mlp.w2.register_forward_hook(make_hook("muse", "7. MLP Out (fc2)"))

    processor = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    image = create_dummy_image(muse_config.image_size)

    processed = processor.preprocess(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    image_grid_thw = processed["image_grid_thw"]
    if isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.tolist()
    image_grid_thw = [tuple(int(x) for x in grid) for grid in image_grid_thw]

    seq_len = int(np.prod(image_grid_thw[0]))
    pixel_batch = pixel_values.unsqueeze(0).to(device, dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    # 清空 Muse 模型 RoPE 调试输出
    rope_module = muse_model.encoder.rope
    if hasattr(rope_module, '_debug_rope_outputs'):
        rope_module._debug_rope_outputs = []
    if hasattr(rope_module, '_debug_rope_intermediates'):
        rope_module._debug_rope_intermediates = {}

    log_header("Running Inference")
    with torch.no_grad():
        origin_model(
            pixel_batch,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
            window_size=-1,
            use_rope=True,
        )
        muse_model(
            pixel_batch,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            interpolate_pos_encoding=True,
            has_learnable_position_embedding=True,
        )

    # --- 收集 RoPE 中间变量 ---
    # Origin 模型: 从全局变量读取
    if ORIGIN_ROPE_DEBUG["rope_emb"] is not None:
        activations["origin"]["0.20 rope_emb"] = ORIGIN_ROPE_DEBUG["rope_emb"]
    if ORIGIN_ROPE_DEBUG["cos_before_chunk"] is not None:
        activations["origin"]["0.21 cos_before_chunk"] = ORIGIN_ROPE_DEBUG["cos_before_chunk"]
    if ORIGIN_ROPE_DEBUG["sin_before_chunk"] is not None:
        activations["origin"]["0.21 sin_before_chunk"] = ORIGIN_ROPE_DEBUG["sin_before_chunk"]
    if ORIGIN_ROPE_DEBUG["cos_after_chunk"] is not None:
        activations["origin"]["0.22 cos_after_chunk"] = ORIGIN_ROPE_DEBUG["cos_after_chunk"]
    if ORIGIN_ROPE_DEBUG["sin_after_chunk"] is not None:
        activations["origin"]["0.22 sin_after_chunk"] = ORIGIN_ROPE_DEBUG["sin_after_chunk"]
    if ORIGIN_ROPE_DEBUG["q_after_rope"] is not None:
        activations["origin"]["0.25 Q After RoPE"] = ORIGIN_ROPE_DEBUG["q_after_rope"]
    if ORIGIN_ROPE_DEBUG["k_after_rope"] is not None:
        activations["origin"]["0.25 K After RoPE"] = ORIGIN_ROPE_DEBUG["k_after_rope"]
    
    # Muse 模型: 从 rope 模块读取中间变量
    if hasattr(rope_module, '_debug_rope_intermediates'):
        intermediates = rope_module._debug_rope_intermediates
        if intermediates.get("rope_emb") is not None:
            activations["muse"]["0.20 rope_emb"] = intermediates["rope_emb"]
        if intermediates.get("cos_before_chunk") is not None:
            activations["muse"]["0.21 cos_before_chunk"] = intermediates["cos_before_chunk"]
        if intermediates.get("sin_before_chunk") is not None:
            activations["muse"]["0.21 sin_before_chunk"] = intermediates["sin_before_chunk"]
        if intermediates.get("cos_after_chunk") is not None:
            activations["muse"]["0.22 cos_after_chunk"] = intermediates["cos_after_chunk"]
        if intermediates.get("sin_after_chunk") is not None:
            activations["muse"]["0.22 sin_after_chunk"] = intermediates["sin_after_chunk"]
    # 读取 RoPE 后的 q、k
    if hasattr(rope_module, '_debug_rope_outputs') and len(rope_module._debug_rope_outputs) >= 2:
        activations["muse"]["0.25 Q After RoPE"] = rope_module._debug_rope_outputs[0]
        activations["muse"]["0.25 K After RoPE"] = rope_module._debug_rope_outputs[1]

    log_header("Layer 0 Internal Tensor Diff Analysis")
    keys = [
        "0.20 rope_emb",
        "0.21 cos_before_chunk",
        "0.21 sin_before_chunk",
        "0.22 cos_after_chunk",
        "0.22 sin_after_chunk",
        "0.25 Q After RoPE",
        "0.25 K After RoPE",
        "1. LN1 Output",
        "2. Q_Proj Out",
        "2. K_Proj Out",
        "2. V_Proj Out",
        "3. Attn Raw (Pre-Proj)",
        "4. Attn Out (Post-Proj)",
        "5. Residual1 (LN2 In)",
        "6. MLP Hidden (fc1)",
        "7. MLP Out (fc2)",
    ]

    for key in keys:
        if key in activations["origin"] and key in activations["muse"]:
            compare_tensors(key, activations["origin"][key], activations["muse"][key])
        else:
            logger.warning(f"Missing capture for {key}")


if __name__ == "__main__":
    test_keye_vision_layer0_step_by_step()
    test_keye_vision_align_with_hf_checkpoint()