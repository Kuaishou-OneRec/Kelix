"""
Keye-VL Pipeline Deep Debugger (Real KeyeProcessor)
===================================================
Trace: Random Image -> KeyeProcessor -> ViT -> ... -> LLM.
Fixes: Uses the official KeyeProcessor to ensure perfect alignment between
       token counts and image features (solves RoPE mismatches).
"""

import os
import sys
import logging
import glob
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import torch.nn as nn
from PIL import Image

# === 导入 Muse 模型 ===
from muse.models.keye_tokenizer_video import modeling as muse_mod
from muse.models.keye_tokenizer_video import modeling_keye_origin as origin_mod
from muse.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig
from muse.training.common import set_default_dtype

# === 导入 Processor 相关 ===
from transformers import AutoTokenizer
# 假设 KeyeProcessor 在 muse.models.keye.modular_Keye，如果不是请修改路径
# 或者将 KeyeProcessor 类定义直接粘贴在脚本上方
try:
    from muse.models.keye_tokenizer_video.processing_keye import KeyeProcessor
except ImportError:
    # 如果找不到路径，请将你刚才发的 KeyeProcessor 代码保存为 modular_Keye.py 并放在同级目录
    sys.path.append(os.getcwd())
    from muse.models.keye_tokenizer_video.processing_keye import KeyeProcessor

# 配置日志
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_CKPT = "/mmu_mllm_hdd_2/maosiyang/output/Keye/vq_end2end_video/discrete/run_exp0.0.1_stage1_baseline/step16000/global_step16000/converted"

# =========================================================================
# Helper Functions (保持不变)
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
        raise ValueError(f"Checkpoint path error: {path}")
    
    # Simple SafeTensor/Bin loader
    state_dict = {}
    st_files = sorted(glob.glob(str(path / "*.safetensors")))
    if st_files:
        from safetensors.torch import safe_open
        for f in st_files:
            with safe_open(f, framework="pt", device=device) as open_f:
                for k in open_f.keys(): state_dict[k] = open_f.get_tensor(k)
        return state_dict
    
    bin_files = sorted(glob.glob(str(path / "*.bin")))
    if bin_files:
        for f in bin_files:
            if any(x in f for x in ["training_args", "optimizer", "scheduler"]): continue
            part = torch.load(f, map_location=device)
            if "module" in part: part = part["module"]
            state_dict.update(part)
        return state_dict
    
    pt_files = sorted(glob.glob(str(path / "*.pt")))
    if pt_files:
         for f in pt_files:
            part = torch.load(f, map_location=device)
            if "module" in part: part = part["module"]
            state_dict.update(part)
         return state_dict
    raise ValueError("No checkpoint found.")

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
                for k in ['z_q', 'logits', 'last_hidden_state']:
                    if k in target: target = target[k]; break
        activations[model_name][layer_name] = target.detach() if isinstance(target, torch.Tensor) else target
    return hook

def register_detailed_hooks(model, name_prefix):
    logger.info(f"Registering DETAILED hooks for {name_prefix}...")
    
    vit_backbone = None
    if hasattr(model, "visual_tokenizer") and hasattr(model.visual_tokenizer, "visual"):
        visual_module = model.visual_tokenizer.visual
        if hasattr(visual_module, "vision_model"): vit_backbone = visual_module.vision_model # Origin
        else: vit_backbone = visual_module # Muse
    
    if vit_backbone:
        if hasattr(vit_backbone, "embeddings"):
            vit_backbone.embeddings.register_forward_hook(make_hook(name_prefix, "0.0 ViT Embeddings Out"))
        
        layer0 = None
        if hasattr(vit_backbone, "encoder") and hasattr(vit_backbone.encoder, "layers"):
            layer0 = vit_backbone.encoder.layers[0]
        
        if layer0:
            if name_prefix == "origin":
                layer0.layer_norm1.register_forward_hook(make_hook(name_prefix, "0.1 LN1 Output"))
                layer0.self_attn.q_proj.register_forward_hook(make_hook(name_prefix, "0.2 Q_Proj Out"))
                layer0.self_attn.v_proj.register_forward_hook(make_hook(name_prefix, "0.2 V_Proj Out"))
                layer0.self_attn.out_proj.register_forward_hook(make_hook(name_prefix, "0.3 Attn Raw (Pre-Proj)", capture_input=True))
                layer0.self_attn.out_proj.register_forward_hook(make_hook(name_prefix, "0.4 Attn Out (Post-Proj)"))
                layer0.mlp.fc1.register_forward_hook(make_hook(name_prefix, "0.6 MLP Hidden (fc1)"))
                layer0.mlp.fc2.register_forward_hook(make_hook(name_prefix, "0.7 MLP Out (fc2)"))
            elif name_prefix == "muse":
                layer0.sa_norm.register_forward_hook(make_hook(name_prefix, "0.1 LN1 Output"))
                layer0.attn.q_proj.register_forward_hook(make_hook(name_prefix, "0.2 Q_Proj Out"))
                layer0.attn.v_proj.register_forward_hook(make_hook(name_prefix, "0.2 V_Proj Out"))
                layer0.attn.output_proj.register_forward_hook(make_hook(name_prefix, "0.3 Attn Raw (Pre-Proj)", capture_input=True))
                layer0.attn.output_proj.register_forward_hook(make_hook(name_prefix, "0.4 Attn Out (Post-Proj)"))
                layer0.mlp.w1.register_forward_hook(make_hook(name_prefix, "0.6 MLP Hidden (fc1)"))
                layer0.mlp.w2.register_forward_hook(make_hook(name_prefix, "0.7 MLP Out (fc2)"))

        vit_backbone.register_forward_hook(make_hook(name_prefix, "1.0 ViT Final Output"))

    if hasattr(model, "visual_tokenizer"):
        vt = model.visual_tokenizer
        if hasattr(vt, 'mlp_AR'):
            vt.mlp_AR.register_forward_hook(make_hook(name_prefix, "2.0 Projector Output"))
        if hasattr(vt, 'quantizer') and len(vt.quantizer) > 0:
            vt.quantizer[0].register_forward_hook(make_hook(name_prefix, "3.0 VQ[0] Output", key="z_q"))

    llm_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"): llm_layers = model.model.layers
    elif hasattr(model, "text_model") and hasattr(model.text_model, "model"): llm_layers = model.text_model.model.layers
    if llm_layers:
        llm_layers[0].register_forward_hook(make_hook(name_prefix, "4.0 LLM Layer 0 Input", capture_input=True))

# =========================================================================
# Input Preparation (KeyeProcessor Logic)
# =========================================================================

def prepare_inputs_via_processor(ckpt_path: str, device: str, dtype: torch.dtype):
    """
    Creates inputs using a random image and KeyeProcessor.
    This mimics the real inference pipeline: ChatML -> Processor -> Model Input.
    """
    logger.info("🎨 Generating Random Image (384x384)...")
    # 生成随机图片
    image = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    
    # 1. 加载 Tokenizer 和 ImageProcessor
    logger.info("⚙️ Loading Tokenizer & ImageProcessor...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    image_processor = KeyeVisionImageProcessor.from_pretrained(ckpt_path)
    
    # 2. 初始化 KeyeProcessor
    logger.info("🧠 Initializing KeyeProcessor...")
    processor = KeyeProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    # 确认 image token
    image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
    logger.info(f"   -> Using Image Token: {image_token}")

    # 3. 构造 ChatML 格式输入
    # KeyeProcessor 会自动扫描文本中的 image_token，并将其展开为对应 Patch 数量的 token
    prompt = f"<|im_start|>user\n{image_token}\nDescribe this noise.<|im_end|>\n<|im_start|>assistant\n"
    logger.info(f"   -> Raw Prompt: {repr(prompt)}")

    # 4. 调用 Processor 处理
    # return_tensors='pt' 会返回 BatchFeature，包含 input_ids, pixel_values, image_grid_thw 等
    logger.info("🔄 Running Processor...")
    inputs = processor(
        text=[prompt], 
        images=image, 
        return_tensors="pt"
    )

    # 5. 转移到 Device 并转换格式
    logger.info("📦 Preparing Model Inputs...")
    
    # 获取数据
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    
    # 计算 vision_token_mask (用于 Muse 模型内部)
    # Processor 已经将 <|image_pad|> 替换成了多个 image_token_id
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    vision_token_mask = (input_ids == image_token_id)
    
    # 打印一些统计信息用于确认
    num_img_tokens = vision_token_mask.sum().item()
    grid_size = image_grid_thw[0].prod().item()
    # 考虑 merge_size (默认2)
    merge_size = image_processor.merge_size
    expected_tokens = grid_size // (merge_size * merge_size)
    
    logger.info(f"   -> Input IDs Shape: {input_ids.shape}")
    logger.info(f"   -> Pixel Values Shape: {pixel_values.shape}")
    logger.info(f"   -> Image Grid: {image_grid_thw.tolist()}")
    logger.info(f"   -> Actual Image Tokens in Sequence: {num_img_tokens}")
    logger.info(f"   -> Expected Tokens (Grid/Merge^2): {expected_tokens}")
    
    if num_img_tokens != expected_tokens:
        logger.warning(f"⚠️ Token mismatch! Processor produced {num_img_tokens}, expected {expected_tokens} based on grid.")

    # 模型期望输入为 [num_patches, C, H, W]，若 Processor 返回 [1, num_patches, C, H, W] 则去掉批维
    if pixel_values.dim() == 5 and pixel_values.shape[0] == 1:
        pixel_values = pixel_values.squeeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "vision_token_mask": vision_token_mask
    }

# =========================================================================
# Main Test
# =========================================================================
def test_pipeline_alignment():
    ckpt_path = DEFAULT_CKPT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 
    
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
        attention_dropout=inner_vcfg.get("attention_dropout", 0.0),
        rope_theta=inner_vcfg.get("rope_theta", 10000.0),
        use_qk_norm=inner_vcfg.get("use_qk_norm", False),
        qk_norm_eps=inner_vcfg.get("qk_norm_eps", 1e-6),
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
    register_detailed_hooks(origin_model, "origin")
    register_detailed_hooks(muse_model, "muse")

    # --- Inputs (Via Real Processor) ---
    log_separator("Running Processor Pipeline")
    inputs = prepare_inputs_via_processor(ckpt_path, device, dtype)
    
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
    log_separator("Deep Dive Analysis")
    
    checkpoints = [
        "0.0 ViT Embeddings Out",
        "0.1 LN1 Output",
        "0.2 Q_Proj Out",
        "0.3 Attn Raw (Pre-Proj)",
        "0.4 Attn Out (Post-Proj)",
        "0.6 MLP Hidden (fc1)",
        "0.7 MLP Out (fc2)",
        "1.0 ViT Final Output",
        "2.0 Projector Output",
        "3.0 VQ[0] Output",
        "4.0 LLM Layer 0 Input"
    ]
    
    for k in checkpoints:
        if k in activations["origin"] and k in activations["muse"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=2e-2)
        else:
            status_o = "Found" if k in activations["origin"] else "MISSING"
            status_m = "Found" if k in activations["muse"] else "MISSING"
            logger.warning(f"⚠️  Missing hook: {k} (Origin={status_o}, Muse={status_m})")

if __name__ == "__main__":
    test_pipeline_alignment()