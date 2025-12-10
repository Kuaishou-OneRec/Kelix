"""
Keye-VL Pipeline Deep Debugger (Fast Version)
=============================================
Trace the entire data flow: ViT -> Projector -> VQ -> LLM Input -> LLM Output.
Optimized to skip slow AutoProcessor loading.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import numpy as np

# === 导入 Muse 模型 ===
from muse.models.keye_tokenizer_video import modeling as muse_mod
from muse.models.keye_tokenizer_video import modeling_keye_origin as origin_mod
from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig
from muse.training.common import set_default_dtype

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
    
    # 自动对齐 Batch/Seq
    if t1.shape != t2.shape:
        if t1.numel() == t2.numel(): t2 = t2.view(t1.shape)
        elif t1.dim() == 3 and t2.dim() == 2 and t1.shape[1:] == t2.shape: t1 = t1.squeeze(0)
    
    if t1.shape != t2.shape:
        logger.error(f"{name:<40} | ❌ SHAPE ERR  | Origin={t1.shape} vs Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    
    match_status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH"
    logger.info(f"{name:<40} | {match_status:<12} | Max: {max_diff:.2e} | Mean: {diff.mean().item():.2e}")
    
    if max_diff >= atol:
        logger.info(f"   -> Origin (first 3): {format_tensor_val(t1, 3)}")
        logger.info(f"   -> Muse   (first 3): {format_tensor_val(t2, 3)}")
        max_idx = torch.argmax(diff)
        logger.info(f"   -> Max Diff Val    : Origin={t1.flatten()[max_idx]:.6f}, Muse={t2.flatten()[max_idx]:.6f}")

def _load_config_json(ckpt_path: str) -> Dict[str, Any]:
    p = Path(ckpt_path)
    cfg_path = p / "config.json" if p.is_dir() else p.with_name("config.json")
    with open(cfg_path, "r") as f: return json.load(f)

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
    
    # Adapt to structure
    if hasattr(model, "text_model"): # Muse
        if hasattr(model.text_model, "model"):
            llm_backbone = model.text_model.model
            if hasattr(model.text_model.model, "output"): llm_head = model.text_model.model.output # Tied linear case
    elif hasattr(model, "model"): # HF
        llm_backbone = model.model
        if hasattr(model, "lm_head"): llm_head = model.lm_head

    if llm_backbone and hasattr(llm_backbone, 'layers'):
        llm_backbone.layers[0].register_forward_hook(make_hook(name_prefix, "6. LLM Layer 0 Input", capture_input=True))
    
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
    dtype = torch.float16
    
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

    # --- Load Weights ---
    logger.info("Loading Weights from checkpoint...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "module" in state_dict: state_dict = state_dict["module"]

    # Origin Load
    origin_model.load_state_dict(state_dict, strict=False)
    
    # Muse Load (Convert)
    logger.info("Converting weights for Muse...")
    muse_state = muse_model.convert_hf_state_dict(state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)

    # --- Prepare Inputs (Manual Construction to avoid hang) ---
    logger.info("Constructing Dummy Inputs...")
    
    # 构造一个极简输入：1张图，Patch尺寸匹配 kernel size
    # 假设 projector kernel (2,2)，输入 spatial 必须能整除
    # 28x28 (patch 14) -> 2x2 patches -> merge后 1x1 token
    H, W = 28, 28 
    t_frames = 1
    
    pixel_values = torch.randn(1, 3, H, W, device=device, dtype=dtype)
    # Muse 需要 tensor grid: [[t, h, w]]
    # h = H // 14 = 2, w = W // 14 = 2
    image_grid_thw = torch.tensor([[t_frames, 2, 2]], device=device, dtype=torch.long)
    
    image_token_id = raw_cfg.get("image_token_id", 151655)
    # Input: [Text, Image, Text]
    # 注意：Image token 数量需匹配。
    # Projector out size: (h/2) * (w/2) * t = 1*1*1 = 1
    input_ids = torch.tensor([[1, image_token_id, 2]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    # Muse 额外需要的 mask
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
    
    # Origin 兼容性调整：Origin 模型可能不接受 vision_token_mask
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