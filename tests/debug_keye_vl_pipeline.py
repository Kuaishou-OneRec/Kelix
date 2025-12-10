"""
Keye-VL Pipeline Deep Debugger
==============================
Trace the entire data flow from Pixel -> ViT -> Projector -> VQ -> LLM.
"""

import os
import sys
import logging
import glob
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import numpy as np
from transformers import AutoProcessor

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
# Helper Functions (Printing & Input Gen)
# =========================================================================

def format_tensor_val(t: torch.Tensor, n: int = 5) -> str:
    """Format first n values of a tensor for display."""
    if not isinstance(t, torch.Tensor): return str(type(t))
    vals = t.detach().float().cpu().flatten()[:n].numpy()
    return "[" + ", ".join([f"{x:.6f}" for x in vals]) + "]"

def log_separator(title: str):
    logger.info(f"\n{'='*120}")
    logger.info(f" {title.center(118)} ")
    logger.info(f"{'='*120}")

def compare_tensors_verbose(name: str, tensor_origin: Any, tensor_muse: Any, atol=1e-3):
    """Deep comparison of two tensors/objects."""
    # 解包常见的 Output 对象或 Tuple
    def unwrap(x):
        if hasattr(x, 'last_hidden_state'): return x.last_hidden_state
        if isinstance(x, (tuple, list)): return x[0]
        if isinstance(x, dict): 
            # 优先找 logits, z_q 等关键key
            for k in ['logits', 'z_q', 'last_hidden_state']:
                if k in x: return x[k]
            return list(x.values())[0] # Fallback
        return x

    t1 = unwrap(tensor_origin)
    t2 = unwrap(tensor_muse)

    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        logger.warning(f"⚠️  [{name}] Skipped: Not tensors (Got {type(t1)} vs {type(t2)})")
        return

    t1 = t1.detach().float().cpu()
    t2 = t2.detach().float().cpu()
    
    # 尝试自动对齐 Batch/Seq 维度
    if t1.shape != t2.shape:
        if t1.numel() == t2.numel():
            t2 = t2.view(t1.shape)
    
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
        # 打印最大差异位置的值
        max_idx = torch.argmax(diff)
        logger.info(f"   -> Max Diff Val    : Origin={t1.flatten()[max_idx]:.6f}, Muse={t2.flatten()[max_idx]:.6f}")

def _load_config_json(ckpt_path: str) -> Dict[str, Any]:
    p = Path(ckpt_path)
    cfg_path = p / "config.json" if p.is_dir() else p.with_name("config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)

# =========================================================================
# Hook Logic
# =========================================================================
activations = {"origin": {}, "muse": {}}

def make_hook(model_name, layer_name, capture_input=False, key=None):
    """
    Args:
        key: If the output is a dict (like VQ output), specify which key to capture.
    """
    def hook(module, inp, out):
        target = inp if capture_input else out
        
        # Handle Tuple/List outputs
        if isinstance(target, (tuple, list)):
            target = target[0]
        
        # Handle Dict outputs (like VQ)
        if isinstance(target, dict) and key:
            target = target[key]
        elif isinstance(target, dict):
            # Default to z_q if unspecified dict
            target = target.get('z_q', list(target.values())[0])
            
        activations[model_name][layer_name] = target.detach()
    return hook

def register_hooks(model, name_prefix):
    """
    Register hooks on critical components of KeyeForConditionalGeneration.
    """
    # 1. ViT Output (Input to Projector)
    # Origin: visual_tokenizer.visual
    # Muse: visual_tokenizer.visual
    if hasattr(model.visual_tokenizer, 'visual'):
        model.visual_tokenizer.visual.register_forward_hook(
            make_hook(name_prefix, "1. ViT Output")
        )

    # 2. Projector Output (Input to VQ Encoder)
    # Origin/Muse: visual_tokenizer.mlp_AR
    if hasattr(model.visual_tokenizer, 'mlp_AR'):
        model.visual_tokenizer.mlp_AR.register_forward_hook(
            make_hook(name_prefix, "2. Projector (mlp_AR) Output")
        )

    # 3. VQ Encoder Output (Before Quantization)
    if hasattr(model.visual_tokenizer, 'encoder'):
        model.visual_tokenizer.encoder.register_forward_hook(
            make_hook(name_prefix, "3. VQ Encoder Output")
        )

    # 4. Quantizer Output (z_q)
    # Assuming quantizer is a ModuleList, hook the first one
    if hasattr(model.visual_tokenizer, 'quantizer') and len(model.visual_tokenizer.quantizer) > 0:
        model.visual_tokenizer.quantizer[0].register_forward_hook(
            make_hook(name_prefix, "4. VQ[0] Output (z_q)", key="z_q")
        )

    # 5. Quant Projector Output (Projection to LLM Dim)
    # KeyeForConditionalGeneration.quant_projector (ModuleList)
    if hasattr(model, 'quant_projector') and len(model.quant_projector) > 0:
        model.quant_projector[0].register_forward_hook(
            make_hook(name_prefix, "5. Quant Projector[0] Output")
        )

    # 6. LLM Input (Embedding Merging Check)
    # Hook Layer 0 of the LLM to see what actually entered the Transformer
    if hasattr(model.model, 'layers'):
        model.model.layers[0].register_forward_hook(
            make_hook(name_prefix, "6. LLM Layer 0 Input", capture_input=True)
        )
    
    # 7. LLM Output Logits (Before Loss)
    if hasattr(model, 'lm_head'):
        model.lm_head.register_forward_hook(
            make_hook(name_prefix, "7. LM Head Logits")
        )

# =========================================================================
# Main Test
# =========================================================================
def test_pipeline_alignment():
    ckpt_path = DEFAULT_CKPT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 # Use float16 as per your previous run
    
    logger.info(f"Loading from: {ckpt_path}")
    
    # --- 1. Load Configs ---
    raw_cfg = _load_config_json(ckpt_path)
    
    # Build Qwen Config
    qwen_cfg = Qwen3Config(
        model_class="Qwen3Model",  # required field in Muse Qwen3Config
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
    
    # Build Vision & Tokenizer Config
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
        attention_function="flash_attention_2", # Consistent with previous successful tests
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
        vq_sampling_mode="argmin", # Force argmin for deterministic testing
    )
    
    # Origin Config
    origin_cfg = origin_mod.KeyeConfig.from_pretrained(ckpt_path)

    # --- 2. Initialize Models ---
    with set_default_dtype(dtype):
        logger.info("Initializing Muse Model...")
        muse_model = muse_mod.KeyeForConditionalGeneration(
            qwen_config=qwen_cfg,
            vision_config=vision_cfg,
            tokenizer_config=tokenizer_cfg,
            image_token_id=raw_cfg.get("image_token_id", 151655),
            pool="sum" # Explicitly match your debug output
        ).to(device)
        
        logger.info("Initializing Origin Model...")
        origin_model = origin_mod.KeyeForConditionalGeneration(origin_cfg).to(device, dtype)

    # --- 3. Load Weights ---
    logger.info("Loading Weights...")
    # Load raw state dict
    if os.path.isdir(ckpt_path):
        # Simplistic loading for directory (assuming you have _load_checkpoint from previous script)
        from safetensors.torch import safe_open
        state_dict = {}
        for f in sorted(glob.glob(f"{ckpt_path}/*.safetensors")):
            with safe_open(f, framework="pt", device="cpu") as open_f:
                for k in open_f.keys():
                    state_dict[k] = open_f.get_tensor(k)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "module" in state_dict: state_dict = state_dict["module"]

    # Origin Load
    origin_model.load_state_dict(state_dict, strict=False)
    
    # Muse Load (Convert)
    muse_state = muse_model.convert_hf_state_dict(state_dict, tie_word_embeddings=qwen_cfg.tie_word_embeddings)
    muse_model.load_state_dict(muse_state, strict=False)




class SlowFastVisionPadder:
    """
    极简版 SlowFast padding 构造器，按照用户提供的片段实现，只保留需要的字段。
    - 仅生成 image_pad（不使用 fast_video 以避免额外依赖）。
    - position_ids 交由模型内部生成，确保两端一致。
    """

    def __init__(self, model_dir: str):
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = processor
        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        assert (
            self.merge_size == 2
        ), f"SlowFastVisionPadder only supports merge_size==2, got {self.merge_size}"

        self.image_pad = processor.tokenizer.encode("<|image_pad|>")[0]
        self.video_pad = processor.tokenizer.encode("<|video_pad|>")[0]
        fast_video_pad = processor.tokenizer.encode("<|fast_video_pad|>")
        assert len(fast_video_pad) == 1, f"Decode fast_video_pad failed: {fast_video_pad}"
        self.fast_video_pad = fast_video_pad[0]
        self.vision_start = processor.tokenizer.encode("<|vision_start|>")[0]
        self.vision_end = processor.tokenizer.encode("<|vision_end|>")[0]
        self.frame = processor.tokenizer.encode("<|frame|>")[0]

    def gen_img_pad(self, n_merged_slow_tokens: int = 1) -> Dict[str, Any]:
        input_ids = [self.vision_start] + [self.image_pad] * n_merged_slow_tokens + [self.vision_end]
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64),
            "attention_mask": torch.tensor([[1] * (n_merged_slow_tokens + 2)], dtype=torch.int64),
            # merge_size=2 -> 每个 merged token 对应 2x2 patch = 4 patch tokens
            "pixel_values": torch.rand(
                n_merged_slow_tokens * 4, 3, self.patch_size, self.patch_size
            ).float(),
            "image_grid_thw": torch.tensor([[1, 2, n_merged_slow_tokens * 2]], dtype=torch.int64),
            "loss_mask": torch.zeros(len(input_ids), dtype=torch.int64),
        }
        # 让模型内部生成 position_ids，保持两端一致
        return inputs

def _build_inputs(
    image_token_id: int, device: torch.device, dtype: torch.dtype, slowfast_dir: Union[str, None]
) -> Tuple[dict, torch.Tensor]:
    """
    优先使用 SlowFastVisionPadder 生成占位输入；否则退回最小随机样例。
    Both models expect image_grid_thw as tensor [num_images, 3].
    """
    if slowfast_dir:
        padder = SlowFastVisionPadder(slowfast_dir)
        img_pad = padder.gen_img_pad(n_merged_slow_tokens=1)
        # image_grid_thw as tensor [num_images, 3]
        grid_thw = img_pad["image_grid_thw"].to(device)  # [1, 3]
        inputs = {
            "input_ids": img_pad["input_ids"].to(device),
            "attention_mask": img_pad["attention_mask"].to(device),
            "pixel_values": img_pad["pixel_values"].to(device, dtype),  # [num_patches, 3, H, W]
            "image_grid_thw": grid_thw,
            # 让模型内部根据 image_token_id 生成 mask
        }
        vision_token_mask = None
        return inputs, vision_token_mask

    # fallback: One 14x14 patch -> one vision token
    # pixel_values: [num_patches, C, H, W] where num_patches = t*h*w = 1*1*1 = 1
    pixel_values = torch.randn(1, 3, 14, 14, device=device, dtype=dtype)
    # image_grid_thw: [num_images, 3] where num_images = 1
    image_grid_thw = torch.tensor([[1, 1, 1]], device=device, dtype=torch.long)
    input_ids = torch.tensor([[image_token_id, 1, 2, 3]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    vision_token_mask = input_ids == image_token_id
    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "vision_token_mask": vision_token_mask,
        },
        vision_token_mask,
    )

    # # --- 4. Prepare Inputs ---
    # # The model expects pixel_values in packed format: (num_patches, C, patch_H, patch_W)
    # # where num_patches = t * h * w from image_grid_thw.
    # # 
    # # For image_grid_thw = [[1, 2, 2]] (t=1, h=2, w=2), we have 4 patches.
    # # Each patch is 14x14 pixels (the patch_size from config).
    # # So pixel_values should be [4, 3, 14, 14].
    
    # patch_size = inner_vcfg.get("patch_size", 14)
    # t, h, w = 1, 2, 2  # Grid dimensions
    # num_patches = t * h * w  # = 4
    
    # pixel_values = torch.randn(num_patches, 3, patch_size, patch_size, device=device, dtype=dtype)
    # image_grid_thw = torch.tensor([[t, h, w]], device=device, dtype=torch.long)
    
    # # Token IDs
    # image_token_id = raw_cfg.get("image_token_id", 151655)
    
    # # Calculate how many image tokens the Projector outputs
    # # Projector uses merge_kernel_size = (2, 2)
    # # After merge: spatial_h = h/2, spatial_w = w/2
    # # With temporal merge, output = (h/2) * (w/2) = 1 token for h=2, w=2
    # # So we need exactly 1 image_token_id in input_ids.
    # n_image_tokens = (h // 2) * (w // 2)  # = 1
    
    # # Input: [Text, ImageToken(s), Text]
    # input_ids_list = [1] + [image_token_id] * n_image_tokens + [2]
    # input_ids = torch.tensor([input_ids_list], device=device, dtype=torch.long)
    # attention_mask = torch.ones_like(input_ids)
    
    # inputs = {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "pixel_values": pixel_values,
    #     "image_grid_thw": image_grid_thw,
    # }
    
    # logger.info(f"Input shapes: pixel_values={pixel_values.shape}, image_grid_thw={image_grid_thw}, input_ids={input_ids.shape}")
    image_token_id = padder_token_id if padder_token_id is not None else raw_cfg.get("image_token_id", 151655)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    slowfast_dir = 
    inputs, vision_mask = _build_inputs(image_token_id, device, dtype, DEFAULT_CKPT)
    # --- 5. Register Hooks ---
    register_hooks(origin_model, "origin")
    register_hooks(muse_model, "muse")

    # --- 6. Forward Pass ---
    log_separator("Running Forward")
    origin_model.eval()
    muse_model.eval()
    
    with torch.no_grad():
        origin_model(**inputs)
        muse_model(**inputs)

    # --- 7. Compare ---
    log_separator("Deep Dive Analysis")
    
    checkpoints = [
        "1. ViT Output",
        "2. Projector (mlp_AR) Output",
        "3. VQ Encoder Output",
        "4. VQ[0] Output (z_q)",
        "5. Quant Projector[0] Output",
        "6. LLM Layer 0 Input",
        "7. LM Head Logits"
    ]
    
    for k in checkpoints:
        if k in activations["origin"] and k in activations["muse"]:
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=1e-2)
        else:
            logger.warning(f"⚠️ Missing hook data for {k}")

if __name__ == "__main__":
    test_pipeline_alignment()