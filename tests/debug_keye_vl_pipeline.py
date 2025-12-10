"""
Keye-VL Pipeline Deep Debugger (Fixed Loader)
=============================================
Trace the entire data flow from Pixel -> ViT -> Projector -> VQ -> LLM.
Fixes: Supports loading checkpoints from directories (Safetensors/Bin).
"""

import os
import sys
import logging
import glob
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import torch
import numpy as np
import torch.nn as nn

# === 导入 Muse 模型 ===
from muse.models.keye_tokenizer_video import modeling as muse_mod
from muse.models.keye_tokenizer_video import modeling_keye_origin as origin_mod
from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig
from muse.training.common import set_default_dtype

# 配置日志
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_CKPT = "/mmu_mllm_hdd_2/maosiyang/output/Keye/vq_end2end_video/discrete/run_exp0.0.1_stage1_baseline/step16000/global_step16000/converted"

# =========================================================================
# Helper Functions
# =========================================================================

def format_tensor_val(t: Any, n: int = 5) -> str:
    if not isinstance(t, torch.Tensor): return str(type(t))
    vals = t.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join([f"{x:.6f}" for x in vals]) + "]"

def log_separator(title: str):
    logger.info(f"\n{'='*120}")
    logger.info(f" {title.center(118)} ")
    logger.info(f"{'='*120}")

def _load_config_json(ckpt_path: str) -> Dict[str, Any]:
    p = Path(ckpt_path)
    # 如果是文件，找同级目录下的 config.json
    base_dir = p if p.is_dir() else p.parent
    cfg_path = base_dir / "config.json"
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {base_dir}")
        
    with open(cfg_path, "r") as f:
        return json.load(f)

def _load_checkpoint_robust(path_str: str, device="cpu") -> Dict[str, torch.Tensor]:
    """Robustly load weights from a file or a directory (safetensors/bin)."""
    path = Path(path_str)
    
    # Case 1: Direct file
    if path.is_file():
        logger.info(f"Loading single file: {path}")
        state_dict = torch.load(path, map_location=device)
        return state_dict.get("module", state_dict)

    # Case 2: Directory
    if not path.is_dir():
        raise ValueError(f"Checkpoint path is neither file nor directory: {path}")

    logger.info(f"Scanning directory: {path}")
    state_dict = {}
    
    # 2.1 Try SafeTensors (High Priority)
    st_files = sorted(glob.glob(str(path / "*.safetensors")))
    if st_files:
        logger.info(f"Found {len(st_files)} safetensors files. Loading...")
        from safetensors.torch import safe_open
        for f in st_files:
            with safe_open(f, framework="pt", device=device) as open_f:
                for k in open_f.keys():
                    state_dict[k] = open_f.get_tensor(k)
        return state_dict

    # 2.2 Try PyTorch Bin (Standard HF)
    bin_files = sorted(glob.glob(str(path / "*.bin")))
    if bin_files:
        logger.info(f"Found {len(bin_files)} .bin files. Loading...")
        for f in bin_files:
            # Skip non-model files
            if any(x in f for x in ["training_args", "optimizer", "scheduler"]):
                continue
            try:
                part = torch.load(f, map_location=device)
                if "module" in part: part = part["module"]
                state_dict.update(part)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        return state_dict

    # 2.3 Try PyTorch PT (DeepSpeed/Megatron dumps)
    pt_files = sorted(glob.glob(str(path / "*.pt")))
    if pt_files:
        logger.info(f"Found {len(pt_files)} .pt files. Loading...")
        for f in pt_files:
            # Usually strict mapping needed, but loading all for now
            part = torch.load(f, map_location=device)
            if "module" in part: part = part["module"]
            state_dict.update(part)
        return state_dict

    raise ValueError(f"No valid checkpoint files (*.safetensors, *.bin, *.pt) found in {path}")

def compare_tensors_verbose(name: str, tensor_origin: Any, tensor_muse: Any, atol=1e-3):
    def unwrap(x):
        if hasattr(x, 'last_hidden_state'): return x.last_hidden_state
        if isinstance(x, (tuple, list)): return x[0]
        if isinstance(x, dict): 
            for k in ['logits', 'z_q', 'last_hidden_state']:
                if k in x: return x[k]
            if len(x) > 0: return list(x.values())[0]
        return x

    t1 = unwrap(tensor_origin)
    t2 = unwrap(tensor_muse)

    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        logger.warning(f"⚠️  [{name}] Skipped: Not tensors (Got {type(t1)} vs {type(t2)})")
        return

    t1 = t1.detach().float().cpu()
    t2 = t2.detach().float().cpu()
    
    if t1.shape != t2.shape:
        # Auto squeeze
        if t1.numel() == t2.numel(): 
            t2 = t2.view(t1.shape)
        elif t1.dim() == 3 and t2.dim() == 2 and t1.shape[1:] == t2.shape:
            t1 = t1.squeeze(0)
    
    if t1.shape != t2.shape:
        logger.error(f"{name:<40} | ❌ SHAPE ERR  | Origin={t1.shape} vs Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    match_status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH"
    
    logger.info(f"{name:<40} | {match_status:<12} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    
    if max_diff >= atol:
        logger.info(f"   -> Origin (first 3): {format_tensor_val(t1, 3)}")
        logger.info(f"   -> Muse   (first 3): {format_tensor_val(t2, 3)}")
        max_idx = torch.argmax(diff)
        logger.info(f"   -> Max Diff Val    : Origin={t1.flatten()[max_idx]:.6f}, Muse={t2.flatten()[max_idx]:.6f}")

# =========================================================================
# Weight Consistency Checker (borrowed from debug_keye_vl_weight)
# =========================================================================
def check_weight_consistency(model_name: str, model: nn.Module, ckpt_state_dict: Dict[str, torch.Tensor], tie_word_embeddings: bool = True):
    logger.info(f"\n{'='*80}")
    logger.info(f"Checking Weights for: {model_name}")
    logger.info(f"{'='*80}")
    
    model_state = model.state_dict()
    matched_value = 0
    mismatched_value = 0
    not_found_in_ckpt = 0
    mismatches = []
    missing = []
    
    # Prefer official convert mapping when available (Muse)
    mapped_ckpt: Dict[str, torch.Tensor] = ckpt_state_dict
    if hasattr(model, "convert_hf_state_dict"):
        try:
            mapped_ckpt = model.convert_hf_state_dict(ckpt_state_dict, tie_word_embeddings=tie_word_embeddings)
            logger.info("Using model.convert_hf_state_dict for mapping...")
        except Exception as e:
            logger.warning(f"convert_hf_state_dict failed ({e}), fallback to suffix match.")
            mapped_ckpt = ckpt_state_dict
    
    for param_name, param_val in model_state.items():
        # Skip buffers / non-fp params
        if not param_val.is_floating_point():
            continue
        
        ckpt_val = None
        
        # Direct match
        if param_name in mapped_ckpt:
            ckpt_val = mapped_ckpt[param_name]
        else:
            # Heuristic suffix match for Origin-style keys
            for k in ckpt_state_dict.keys():
                if k.endswith(param_name) and len(param_name) >= 10:
                    ckpt_val = ckpt_state_dict[k]
                    break
        
        if ckpt_val is None:
            not_found_in_ckpt += 1
            missing.append(param_name)
            continue
        
        # Align dtype/device for comparison
        ckpt_val = ckpt_val.to(param_val.device)
        if param_val.shape != ckpt_val.shape and param_val.shape == ckpt_val.t().shape:
            ckpt_val = ckpt_val.t()
        if param_val.shape != ckpt_val.shape:
            logger.error(f"❌ Shape Mismatch: {param_name} | Model {param_val.shape} != Ckpt {ckpt_val.shape}")
            mismatched_value += 1
            mismatches.append((param_name, float('inf'), param_val.mean().item(), ckpt_val.mean().item()))
            continue
        
        diff = (param_val - ckpt_val.to(param_val.dtype)).abs().max().item()
        if diff > 1e-3:
            mismatched_value += 1
            mismatches.append((param_name, diff, param_val.mean().item(), ckpt_val.mean().item()))
        else:
            matched_value += 1
    
    logger.info(f"✅ Matched Weights: {matched_value}")
    logger.info(f"❌ Mismatched Weights: {mismatched_value}")
    logger.info(f"❓ Not found in Checkpoint: {not_found_in_ckpt}")
    
    if mismatches:
        logger.info("\nTop 10 Mismatches (Key | Diff | Model Mean | Ckpt Mean):")
        for m in mismatches[:10]:
            logger.info(f"  {m[0]:<60} | {m[1]:.2e} | {m[2]:.4f} | {m[3]:.4f}")
            
    if missing:
        logger.info("\nTop 10 Missing Keys (Present in Model, Absent in Ckpt):")
        for m in missing[:10]:
            logger.info(f"  {m}")

# =========================================================================
# Hook System
# =========================================================================
activations = {"origin": {}, "muse": {}}

def make_hook(model_name, layer_name, capture_input=False, key=None):
    def hook(module, inp, out):
        target = inp if capture_input else out
        if isinstance(target, (tuple, list)): target = target[0]
        if isinstance(target, dict):
            if key and key in target: target = target[key]
            else:
                if 'z_q' in target: target = target['z_q']
                elif 'logits' in target: target = target['logits']
                elif 'last_hidden_state' in target: target = target['last_hidden_state']
        activations[model_name][layer_name] = target.detach() if isinstance(target, torch.Tensor) else target
    return hook

def register_hooks(model, name_prefix):
    logger.info(f"Registering hooks for {name_prefix}...")
    
    # 1. Visual Branch
    if hasattr(model, "visual_tokenizer"):
        vt = model.visual_tokenizer
        if hasattr(vt, 'visual'):
            vt.visual.register_forward_hook(make_hook(name_prefix, "1. ViT Output"))
        if hasattr(vt, 'mlp_AR'):
            vt.mlp_AR.register_forward_hook(make_hook(name_prefix, "2. Projector Output"))
        if hasattr(vt, 'encoder'):
            vt.encoder.register_forward_hook(make_hook(name_prefix, "3. VQ Encoder Output"))
        if hasattr(vt, 'quantizer') and len(vt.quantizer) > 0:
            vt.quantizer[0].register_forward_hook(make_hook(name_prefix, "4. VQ[0] Output", key="z_q"))

    # 2. Connector
    if hasattr(model, 'quant_projector') and len(model.quant_projector) > 0:
        model.quant_projector[0].register_forward_hook(make_hook(name_prefix, "5. Quant Projector[0] Output"))

    # 3. LLM Input & Output
    llm_backbone = None
    llm_head = None
    
    # Adaptive Structural Finding
    # Case A: Muse (model.text_model.model...) or (model.model...) if renamed
    # Case B: HF (model.model...)
    
    # Try to find the inner LLM backbone (Transformer)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # This covers HF and Muse (if renamed to self.model)
        llm_backbone = model.model
    elif hasattr(model, "text_model") and hasattr(model.text_model, "model"):
        # This covers Muse (old self.text_model structure)
        llm_backbone = model.text_model.model
    
    # Try to find Head
    if hasattr(model, "lm_head"):
        llm_head = model.lm_head
    elif hasattr(model, "text_model") and hasattr(model.text_model, "output"):
        llm_head = model.text_model.output

    if llm_backbone and hasattr(llm_backbone, 'layers'):
        llm_backbone.layers[0].register_forward_hook(make_hook(name_prefix, "6. LLM Layer 0 Input", capture_input=True))
    else:
        logger.error(f"❌ Could not find LLM backbone for {name_prefix}")
    
    if llm_head:
        llm_head.register_forward_hook(make_hook(name_prefix, "7. LM Head Logits"))
    else:
        # Fallback to model output
        model.register_forward_hook(make_hook(name_prefix, "7. LM Head Logits", key="logits"))

# =========================================================================
# Main Test
# =========================================================================
def test_pipeline_alignment():
    ckpt_path = DEFAULT_CKPT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 # FP16
    
    logger.info(f"Loading from: {ckpt_path}")
    raw_cfg = _load_config_json(ckpt_path)
    
    # Muse Configs
    qwen_cfg = Qwen3Config(
        model_class="Qwen3Model",
        vocab_size=raw_cfg["vocab_size"],
        embed_dim=raw_cfg["hidden_size"],
        num_layers=raw_cfg["num_hidden_layers"],
        num_heads=raw_cfg["num_attention_heads"],
        num_kv_heads=raw_cfg["num_key_value_heads"],
        head_dim=raw_cfg["head_dim"],
        intermediate_dim=raw_cfg["intermediate_size"],
        max_seq_len=raw_cfg["max_position_embeddings"],
        rope_base=float(raw_cfg.get("rope_theta", 1_000_000)),
        attention_function="flash_attention_2",
        tie_word_embeddings=raw_cfg.get("tie_word_embeddings", True),
    )
    
    outer_vcfg = raw_cfg["vision_config"]
    inner_vcfg = outer_vcfg["vision_config"]
    
    vision_cfg = KeyeVisionConfig(
        hidden_size=inner_vcfg["hidden_size"],
        num_hidden_layers=inner_vcfg["num_hidden_layers"],
        num_attention_heads=inner_vcfg["num_attention_heads"],
        image_size=inner_vcfg["image_size"],
        patch_size=inner_vcfg["patch_size"],
        intermediate_size=inner_vcfg["intermediate_size"],
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_function="flash_attention_2",
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
    )
    
    # Origin Config
    origin_cfg = origin_mod.KeyeConfig.from_pretrained(ckpt_path)

    # --- Initialize Models ---
    with set_default_dtype(dtype):
        logger.info("Initializing Muse Model...")
        muse_model = muse_mod.KeyeForConditionalGeneration(
            qwen_config=qwen_cfg,
            vision_config=vision_cfg,
            tokenizer_config=tokenizer_cfg,
            image_token_id=raw_cfg.get("image_token_id", 151655),
            pool="sum"
        ).to(device)
        
        logger.info("Initializing Origin Model...")
        origin_model = origin_mod.KeyeForConditionalGeneration(origin_cfg).to(device, dtype)

    # --- Load Weights (Robust Loader) ---
    logger.info("Loading Weights from checkpoint directory...")
    state_dict = _load_checkpoint_robust(ckpt_path, device="cpu") # Load to CPU first to save GPU mem

    # Origin Load
    origin_model.load_state_dict(state_dict, strict=False)
    
    # Muse Load (Convert)
    logger.info("Converting weights for Muse...")
    muse_state = muse_model.convert_hf_state_dict(state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)

    # --- Weight Consistency Check ---
    log_separator("Weight Consistency Check")
    check_weight_consistency("Origin Model", origin_model, state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    check_weight_consistency("Muse Model", muse_model, state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)

    # Move to GPU
    origin_model.to(device)
    muse_model.to(device)

    # --- Prepare Inputs ---
    logger.info("Constructing Dummy Inputs...")
    
    # Use packed 5D pixel_values: (batch, seq_len, C, H, W)
    patch = inner_vcfg.get("patch_size", 14)
    t_frames, h_patches, w_patches = 1, 2, 2  # grid -> seq_len = 4
    seq_len = t_frames * h_patches * w_patches
    pixel_values = torch.randn(seq_len, 3, patch, patch, device=device, dtype=dtype)
    image_grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], device=device, dtype=torch.long)
    
    image_token_id = raw_cfg.get("image_token_id", 151655)
    # Projector out size after merge (2x2): (h/2)*(w/2)*t = 1
    input_ids = torch.tensor([[1, image_token_id, 2]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    vision_token_mask = (input_ids == image_token_id)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "vision_token_mask": vision_token_mask
    }

    # --- Register Hooks ---
    register_hooks(origin_model, "origin")
    register_hooks(muse_model, "muse")

    # --- Forward Pass ---
    log_separator("Running Forward")
    origin_model.eval()
    muse_model.eval()
    
    origin_inputs = {k: v for k, v in inputs.items() if k != "vision_token_mask"}
    
    with torch.no_grad():
        logger.info("Running Origin Forward...")
        origin_out = origin_model(**origin_inputs)
        logger.info("Running Muse Forward...")
        muse_out = muse_model(**inputs)

    # --- Compare ---
    log_separator("Deep Dive Analysis")
    checkpoints = [
        "1. ViT Output",
        "2. Projector Output",
        "3. VQ Encoder Output",
        "4. VQ[0] Output",
        "5. Quant Projector[0] Output",
        "6. LLM Layer 0 Input",
        "7. LM Head Logits"
    ]
    
    for k in checkpoints:
        if k in activations["origin"] and k in activations["muse"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=1e-2)
        else:
            logger.warning(f"⚠️ Missing hook data for {k} (Origin={k in activations['origin']}, Muse={k in activations['muse']})")

if __name__ == "__main__":
    test_pipeline_alignment()