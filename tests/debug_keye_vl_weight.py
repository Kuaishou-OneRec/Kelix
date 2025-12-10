"""
Keye-VL Pipeline Deep Debugger (Enhanced ViT Internal Trace)
============================================================
Trace: Pixel -> Patch Embed -> ViT Layer 0 (Detailed) -> ViT Final -> Projector -> LLM.
Focus: Pinpoint where the ViT diverges inside the full VLM pipeline.
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
    base_dir = p if p.is_dir() else p.parent
    cfg_path = base_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {base_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)

def _load_checkpoint_robust(path_str: str, device="cpu") -> Dict[str, torch.Tensor]:
    path = Path(path_str)
    if path.is_file():
        state_dict = torch.load(path, map_location=device)
        return state_dict.get("module", state_dict)

    if not path.is_dir():
        raise ValueError(f"Checkpoint path is neither file nor directory: {path}")

    logger.info(f"Scanning directory: {path}")
    state_dict = {}
    
    st_files = sorted(glob.glob(str(path / "*.safetensors")))
    if st_files:
        logger.info(f"Found {len(st_files)} safetensors files. Loading...")
        from safetensors.torch import safe_open
        for f in st_files:
            with safe_open(f, framework="pt", device=device) as open_f:
                for k in open_f.keys():
                    state_dict[k] = open_f.get_tensor(k)
        return state_dict

    bin_files = sorted(glob.glob(str(path / "*.bin")))
    if bin_files:
        logger.info(f"Found {len(bin_files)} .bin files. Loading...")
        for f in bin_files:
            if any(x in f for x in ["training_args", "optimizer", "scheduler"]): continue
            try:
                part = torch.load(f, map_location=device)
                if "module" in part: part = part["module"]
                state_dict.update(part)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        return state_dict
    
    pt_files = sorted(glob.glob(str(path / "*.pt")))
    if pt_files:
         for f in pt_files:
            part = torch.load(f, map_location=device)
            if "module" in part: part = part["module"]
            state_dict.update(part)
         return state_dict

    raise ValueError(f"No valid checkpoint files found in {path}")

def compare_tensors_verbose(name: str, tensor_origin: Any, tensor_muse: Any, atol=1e-3):
    def unwrap(x):
        if hasattr(x, 'last_hidden_state'): return x.last_hidden_state
        if isinstance(x, (tuple, list)): return x[0]
        if isinstance(x, dict): 
            for k in ['logits', 'z_q', 'last_hidden_state']:
                if k in x: return x[k]
            return list(x.values())[0]
        return x

    t1 = unwrap(tensor_origin)
    t2 = unwrap(tensor_muse)

    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        logger.warning(f"⚠️  [{name}] Skipped: Not tensors")
        return

    t1 = t1.detach().float().cpu()
    t2 = t2.detach().float().cpu()
    
    # Auto squeeze for broadcasting issues (e.g. [1, S, D] vs [S, D])
    if t1.shape != t2.shape:
        if t1.numel() == t2.numel(): t2 = t2.view(t1.shape)
        elif t1.dim() == 3 and t2.dim() == 2 and t1.shape[1:] == t2.shape: t1 = t1.squeeze(0)
        elif t2.dim() == 3 and t1.dim() == 2 and t2.shape[1:] == t1.shape: t2 = t2.squeeze(0)
    
    if t1.shape != t2.shape:
        logger.error(f"{name:<45} | ❌ SHAPE ERR  | Origin={t1.shape} vs Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    match_status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH"
    
    logger.info(f"{name:<45} | {match_status:<12} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    
    if max_diff >= atol:
        max_idx = torch.argmax(diff)
        logger.info(f"   -> Max Diff Index: {max_idx.item()}")
        logger.info(f"   -> Origin Val    : {t1.flatten()[max_idx]:.6f}")
        logger.info(f"   -> Muse Val      : {t2.flatten()[max_idx]:.6f}")

# =========================================================================
# Enhanced Hook System (ViT Deep Dive)
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

def register_detailed_hooks(model, name_prefix):
    logger.info(f"Registering DETAILED hooks for {name_prefix}...")
    
    # -------------------------------------------------------------------------
    # 1. 寻找 Visual Transformer Backbone (ViT)
    # -------------------------------------------------------------------------
    # Origin 结构通常: model.visual_tokenizer.visual.vision_model
    # Muse 结构通常:   model.visual_tokenizer.visual
    
    vit_backbone = None
    if hasattr(model, "visual_tokenizer") and hasattr(model.visual_tokenizer, "visual"):
        visual_module = model.visual_tokenizer.visual
        
        # Origin (HF Siglip) usually has .vision_model wrapper
        if hasattr(visual_module, "vision_model"):
            vit_backbone = visual_module.vision_model
            logger.info(f"[{name_prefix}] Found Origin-style SiglipVisionModel")
        else:
            vit_backbone = visual_module
            logger.info(f"[{name_prefix}] Found Muse-style KeyeVisionTransformer")
    
    if vit_backbone is None:
        logger.error(f"❌ Could not find ViT backbone for {name_prefix}")
        return

    # -------------------------------------------------------------------------
    # 2. Hook: Patch Embeddings (Layer 0 Input)
    # -------------------------------------------------------------------------
    # 检查 Embeddings 输出 (通常是 Layer 0 的输入)
    if hasattr(vit_backbone, "embeddings"):
        vit_backbone.embeddings.register_forward_hook(make_hook(name_prefix, "0.0 ViT Embeddings Out"))
    
    # -------------------------------------------------------------------------
    # 3. Hook: Layer 0 Internals (Detailed Debugging)
    # -------------------------------------------------------------------------
    # 这里的逻辑完全照搬了你的 ViT 调试脚本
    
    layer0 = None
    if hasattr(vit_backbone, "encoder") and hasattr(vit_backbone.encoder, "layers"):
        layer0 = vit_backbone.encoder.layers[0]
    
    if layer0:
        logger.info(f"[{name_prefix}] Hooking ViT Layer 0...")
        
        # === Origin (HF) Style Hooks ===
        if name_prefix == "origin":
            # LN1
            if hasattr(layer0, "layer_norm1"):
                layer0.layer_norm1.register_forward_hook(make_hook(name_prefix, "0.1 LN1 Output"))
            # Self Attention Projections
            if hasattr(layer0, "self_attn"):
                layer0.self_attn.q_proj.register_forward_hook(make_hook(name_prefix, "0.2 Q_Proj Out"))
                layer0.self_attn.k_proj.register_forward_hook(make_hook(name_prefix, "0.2 K_Proj Out"))
                layer0.self_attn.v_proj.register_forward_hook(make_hook(name_prefix, "0.2 V_Proj Out"))
                # Pre-Projection Attention Output
                layer0.self_attn.out_proj.register_forward_hook(make_hook(name_prefix, "0.3 Attn Raw (Pre-Proj)", capture_input=True))
                # Post-Projection
                layer0.self_attn.out_proj.register_forward_hook(make_hook(name_prefix, "0.4 Attn Out (Post-Proj)"))
            
            # MLP
            if hasattr(layer0, "mlp"):
                if hasattr(layer0.mlp, "fc1"):
                    layer0.mlp.fc1.register_forward_hook(make_hook(name_prefix, "0.6 MLP Hidden (fc1)"))
                if hasattr(layer0.mlp, "fc2"):
                    layer0.mlp.fc2.register_forward_hook(make_hook(name_prefix, "0.7 MLP Out (fc2)"))

        # === Muse Style Hooks ===
        elif name_prefix == "muse":
            # LN1
            if hasattr(layer0, "sa_norm"):
                layer0.sa_norm.register_forward_hook(make_hook(name_prefix, "0.1 LN1 Output"))
            
            # Self Attention Projections
            if hasattr(layer0, "attn"):
                if hasattr(layer0.attn, "q_proj"):
                    layer0.attn.q_proj.register_forward_hook(make_hook(name_prefix, "0.2 Q_Proj Out"))
                if hasattr(layer0.attn, "k_proj"):
                    layer0.attn.k_proj.register_forward_hook(make_hook(name_prefix, "0.2 K_Proj Out"))
                if hasattr(layer0.attn, "v_proj"):
                    layer0.attn.v_proj.register_forward_hook(make_hook(name_prefix, "0.2 V_Proj Out"))
                
                # Output Proj
                if hasattr(layer0.attn, "output_proj"):
                    layer0.attn.output_proj.register_forward_hook(make_hook(name_prefix, "0.3 Attn Raw (Pre-Proj)", capture_input=True))
                    layer0.attn.output_proj.register_forward_hook(make_hook(name_prefix, "0.4 Attn Out (Post-Proj)"))

            # MLP
            if hasattr(layer0, "mlp"):
                if hasattr(layer0.mlp, "w1"):
                    layer0.mlp.w1.register_forward_hook(make_hook(name_prefix, "0.6 MLP Hidden (fc1)"))
                if hasattr(layer0.mlp, "w2"):
                    layer0.mlp.w2.register_forward_hook(make_hook(name_prefix, "0.7 MLP Out (fc2)"))
    
    # -------------------------------------------------------------------------
    # 4. Global Outputs
    # -------------------------------------------------------------------------
    # ViT Final Output
    vit_backbone.register_forward_hook(make_hook(name_prefix, "1.0 ViT Final Output"))
    
    # Projector
    if hasattr(model, "visual_tokenizer") and hasattr(model.visual_tokenizer, "mlp_AR"):
         model.visual_tokenizer.mlp_AR.register_forward_hook(make_hook(name_prefix, "2.0 Projector Output"))
         
    # VQ Quantizer
    if hasattr(model, "visual_tokenizer") and hasattr(model.visual_tokenizer, "quantizer"):
         model.visual_tokenizer.quantizer[0].register_forward_hook(make_hook(name_prefix, "3.0 VQ[0] Output", key="z_q"))

    # LLM Input
    llm_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"): # Qwen/HF
        llm_layers = model.model.layers
    elif hasattr(model, "text_model") and hasattr(model.text_model, "model"): # Muse legacy
        llm_layers = model.text_model.model.layers
    
    if llm_layers:
        llm_layers[0].register_forward_hook(make_hook(name_prefix, "4.0 LLM Layer 0 Input", capture_input=True))


# =========================================================================
# Main Test
# =========================================================================
def test_pipeline_alignment():
    ckpt_path = DEFAULT_CKPT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用与 ViT 测试一致的 dtype (bfloat16) 以避免精度转换带来的误差
    dtype = torch.bfloat16 
    
    logger.info(f"Loading from: {ckpt_path}")
    raw_cfg = _load_config_json(ckpt_path)
    
    # ... (Config 代码保持不变) ...
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
    logger.info("Loading Weights...")
    state_dict = _load_checkpoint_robust(ckpt_path, device="cpu")
    origin_model.load_state_dict(state_dict, strict=False)
    muse_state = muse_model.convert_hf_state_dict(state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)

    origin_model.to(device)
    muse_model.to(device)

    # --- Hooks ---
    # 使用新的 Detailed Hooks
    register_detailed_hooks(origin_model, "origin")
    register_detailed_hooks(muse_model, "muse")

    # --- Inputs ---
    patch = inner_vcfg.get("patch_size", 14)
    # 构造与 ViT 测试一致的简单 Grid (1, 1, 1) -> seq_len = 1 frame * 1*1 patches
    # 也可以保持你之前的 2x2，这里为了调试简单设为 14x14 像素
    t_frames, h_patches, w_patches = 1, 1, 1 
    seq_len = t_frames * h_patches * w_patches # = 1
    
    # 输入像素： [Batch * Seq, C, H, W]
    # Muse ViT 需要 [1, Seq, C, H, W] 或 [Seq, C, H, W] 取决于 forward 内部
    # 这里我们构造 5D, VLM wrapper 应该会自动处理
    pixel_values = torch.randn(1 * seq_len, 3, patch, patch, device=device, dtype=dtype)
    
    # ⚠️ 关键点：Image Grid 
    image_grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], device=device, dtype=torch.long)
    
    image_token_id = raw_cfg.get("image_token_id", 151655)
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

    # --- Forward ---
    log_separator("Running Forward")
    origin_model.eval()
    muse_model.eval()
    
    origin_inputs = {k: v for k, v in inputs.items() if k != "vision_token_mask"}
    
    with torch.no_grad():
        logger.info("Running Origin Forward...")
        origin_out = origin_model(**origin_inputs)
        logger.info("Running Muse Forward...")
        muse_out = muse_model(**inputs)

    # --- Analysis ---
    log_separator("Deep Dive Analysis (Layer 0 Specifics)")
    
    # 按照数据流向进行检查
    checkpoints = [
        "0.0 ViT Embeddings Out",  # 最关键：如果这里不对，就是输入处理或者 PatchEmbed 权重问题
        "0.1 LN1 Output",
        "0.2 Q_Proj Out",
        "0.2 K_Proj Out",
        "0.2 V_Proj Out",
        "0.3 Attn Raw (Pre-Proj)", # 如果这里对，说明 Attention 计算逻辑对
        "0.4 Attn Out (Post-Proj)",
        "0.6 MLP Hidden (fc1)",
        "0.7 MLP Out (fc2)",
        "1.0 ViT Final Output",    # 整个 ViT 的输出
        "2.0 Projector Output",
        "3.0 VQ[0] Output",
        "4.0 LLM Layer 0 Input"
    ]
    
    # 增加容差，FP16/BF16 在 Attention 累积后可能会有 1e-2 级别的误差
    for k in checkpoints:
        if k in activations["origin"] and k in activations["muse"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=2e-2)
        else:
            status_o = "Found" if k in activations["origin"] else "MISSING"
            status_m = "Found" if k in activations["muse"] else "MISSING"
            logger.warning(f"⚠️  Missing hook: {k} (Origin={status_o}, Muse={status_m})")

if __name__ == "__main__":
    test_pipeline_alignment()