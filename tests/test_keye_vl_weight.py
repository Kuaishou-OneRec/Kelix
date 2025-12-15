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
from muse.models.keye_tokenizer_end2end_image import modeling as muse_mod
from tests.models.keye_vl_tokenizer_image import modeling_keye_origin as origin_mod
from tests.models.keye_vl_tokenizer_image.image_processing_keye import SiglipImageProcessor
from muse.config import Qwen3Config, KeyeVisionConfig, KeyeTokenizerConfig
from muse.training.common import set_default_dtype

# 导入 Origin 模型的 RoPE debug 变量
from tests.models.keye_vl_tokenizer_image.modeling_keye_origin import _DEBUG_ROPE_OUTPUTS as ORIGIN_ROPE_DEBUG

# === 导入 Processor 相关 ===
from transformers import AutoTokenizer
# 假设 KeyeProcessor 在 muse.models.keye.modular_Keye，如果不是请修改路径
# 或者将 KeyeProcessor 类定义直接粘贴在脚本上方
try:
    from tests.models.keye_vl_tokenizer_image.processing_keye import KeyeProcessor
except ImportError:
    # 如果找不到路径，请将你刚才发的 KeyeProcessor 代码保存为 modular_Keye.py 并放在同级目录
    sys.path.append(os.getcwd())
    from tests.models.keye_vl_tokenizer_image.processing_keye import KeyeProcessor

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

def compare_tensors_verbose(name: str, tensor_origin: Any, tensor_muse: Any, atol=1e-3, print_values=False):
    def unwrap(x):
        if hasattr(x, 'last_hidden_state'): return x.last_hidden_state
        if isinstance(x, (tuple, list)): return x[0]
        if isinstance(x, dict): 
            for k in ['logits', 'z_q', 'last_hidden_state']:
                if k in x: return x[k]
            return list(x.values())[0]
        return x

    t1_raw = unwrap(tensor_origin)
    t2_raw = unwrap(tensor_muse)

    if not isinstance(t1_raw, torch.Tensor) or not isinstance(t2_raw, torch.Tensor):
        logger.warning(f"⚠️  [{name}] Skipped: Not tensors")
        return

    # 保存原始 dtype 用于打印
    t1_dtype = t1_raw.dtype
    t2_dtype = t2_raw.dtype
    
    t1 = t1_raw.detach().float().cpu()
    t2 = t2_raw.detach().float().cpu()
    
    if t1.shape != t2.shape:
        # 处理 [b, h, s, d] vs [b, s, h, d] 的情况（RoPE 输出格式差异）
        # 如果两个 tensor 的 dim 相同且只是中间两个维度交换，使用 transpose
        if t1.dim() == 4 and t2.dim() == 4 and t1.numel() == t2.numel():
            # 检查是否是 [b, h, s, d] vs [b, s, h, d] 的情况
            if t1.shape[0] == t2.shape[0] and t1.shape[3] == t2.shape[3]:
                if t1.shape[1] == t2.shape[2] and t1.shape[2] == t2.shape[1]:
                    # t2 是 [b, s, h, d]，需要 transpose(1, 2) 变成 [b, h, s, d]
                    t2 = t2.transpose(1, 2)
        elif t1.numel() == t2.numel():
            t2 = t2.view(t1.shape)
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
    
    if print_values:
        # 打印 dtype 和 shape
        logger.info(f"   -> Origin dtype: {t1_dtype}, shape: {t1_raw.shape}")
        logger.info(f"   -> Muse   dtype: {t2_dtype}, shape: {t2_raw.shape}")
        # 打印前 10 个值
        t1_first10 = t1.flatten()[:10].numpy()
        t2_first10 = t2.flatten()[:10].numpy()
        logger.info(f"   -> Origin first 10: [{', '.join([f'{v:.6f}' for v in t1_first10])}]")
        logger.info(f"   -> Muse   first 10: [{', '.join([f'{v:.6f}' for v in t2_first10])}]")
    
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
    llm_layer_count = 0
    
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
    # Qwen3: KeyeForConditionalGeneration.model -> Qwen3Model -> .model (TransformerDecoder) -> .layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        llm_layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        # Fallback: some wrappers nest another `.model`
        llm_layers = model.model.model.layers
    elif hasattr(model, "text_model") and hasattr(model.text_model, "model"):
        llm_layers = model.text_model.model.layers
    if llm_layers:
        llm_layer_count = len(llm_layers)
        for idx, layer in enumerate(llm_layers):
            layer.register_forward_hook(
                make_hook(name_prefix, f"4.{idx:02d} LLM Layer {idx} Input", capture_input=True)
            )
            layer.register_forward_hook(make_hook(name_prefix, f"4.{idx:02d} LLM Layer {idx} Output"))
            
            # 为 Layer 0 添加细粒度 hook
            if idx == 0:
                if name_prefix == "origin":
                    # Origin: KeyeDecoderLayer 结构
                    layer.input_layernorm.register_forward_hook(
                        make_hook(name_prefix, "4.00a LLM L0 InputLN"))
                    layer.self_attn.q_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00b LLM L0 Q_Proj"))
                    layer.self_attn.k_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00c LLM L0 K_Proj"))
                    layer.self_attn.v_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00d LLM L0 V_Proj"))
                    layer.self_attn.o_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00e LLM L0 Attn Out"))
                    layer.post_attention_layernorm.register_forward_hook(
                        make_hook(name_prefix, "4.00f LLM L0 PostAttnLN"))
                    layer.mlp.gate_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00g LLM L0 MLP Gate"))
                    layer.mlp.up_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00h LLM L0 MLP Up"))
                    layer.mlp.down_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00i LLM L0 MLP Down"))
                elif name_prefix == "muse":
                    # Muse: TransformerSelfAttentionLayer 结构
                    layer.sa_norm.register_forward_hook(
                        make_hook(name_prefix, "4.00a LLM L0 InputLN"))
                    layer.attn.q_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00b LLM L0 Q_Proj"))
                    layer.attn.k_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00c LLM L0 K_Proj"))
                    layer.attn.v_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00d LLM L0 V_Proj"))
                    layer.attn.output_proj.register_forward_hook(
                        make_hook(name_prefix, "4.00e LLM L0 Attn Out"))
                    layer.mlp_norm.register_forward_hook(
                        make_hook(name_prefix, "4.00f LLM L0 PostAttnLN"))
                    layer.mlp.w1.register_forward_hook(
                        make_hook(name_prefix, "4.00g LLM L0 MLP Gate"))
                    layer.mlp.w3.register_forward_hook(
                        make_hook(name_prefix, "4.00h LLM L0 MLP Up"))
                    layer.mlp.w2.register_forward_hook(
                        make_hook(name_prefix, "4.00i LLM L0 MLP Down"))

    # Final LLM states (pre-head and logits)
    decoder = None
    if hasattr(model, "model"):
        decoder = model.model
        if hasattr(decoder, "model"):
            decoder = decoder.model
    elif hasattr(model, "text_model") and hasattr(model.text_model, "model"):
        decoder = model.text_model.model

    decoder_norm = decoder.norm if decoder and hasattr(decoder, "norm") else None
    decoder_output_proj = None
    if hasattr(model, "lm_head"):
        decoder_output_proj = model.lm_head
    elif decoder and hasattr(decoder, "output"):
        decoder_output_proj = decoder.output

    if decoder_norm:
        decoder_norm.register_forward_hook(make_hook(name_prefix, "5.0 LLM Final Hidden"))
    if decoder_output_proj and hasattr(decoder_output_proj, "register_forward_hook"):
        decoder_output_proj.register_forward_hook(make_hook(name_prefix, "5.1 LLM Logits"))
    else:
        logger.warning(f"⚠️  {name_prefix}: Skip logits hook (module not hookable)")

    return llm_layer_count

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
    image_processor = SiglipImageProcessor.from_pretrained(ckpt_path)
    
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
    # 从 rope_scaling 中提取 mrope_section
    rope_scaling = raw_cfg.get("rope_scaling")
    mrope_section = rope_scaling.get("mrope_section") if rope_scaling else None
    
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
        hidden_act=raw_cfg.get("hidden_act", "silu"),
        attention_bias=raw_cfg.get("attention_bias", False),
        rope_base=float(raw_cfg.get("rope_theta", 1_000_000)),
        rope_theta=float(raw_cfg.get("rope_theta", 1_000_000)),
        rope_scaling=rope_scaling,
        attention_function=raw_cfg.get("_attn_implementation", "flash_attention_2"),
        use_sliding_window=raw_cfg.get("use_sliding_window", False),
        sliding_window=raw_cfg.get("sliding_window"),
        norm_eps=raw_cfg.get("norm_eps", 1e-6),
        rms_norm_eps=raw_cfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=raw_cfg.get("tie_word_embeddings", True),
        # 3D Multimodal RoPE 配置
        use_multimodal_rope=True,
        mrope_section=mrope_section,
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
        hidden_act=inner_vcfg.get("hidden_act", "gelu_pytorch_tanh"),
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_dropout=inner_vcfg.get("attention_dropout", 0.0),
        rope_theta=inner_vcfg.get("rope_theta", 10000.0),
        use_qk_norm=inner_vcfg.get("use_qk_norm", False),
        qk_norm_eps=inner_vcfg.get("qk_norm_eps", 1e-6),
        attention_function=raw_cfg.get("_attn_implementation", "flash_attention_2"),
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
        muse_model = muse_mod.KeyeTokenizerEnd2EndImage(
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
    muse_model.to(device, dtype)  # 确保 Muse 模型也转换为 bfloat16，与 Origin 保持一致

    # --- Hooks ---
    origin_llm_layers = register_detailed_hooks(origin_model, "origin")
    muse_llm_layers = register_detailed_hooks(muse_model, "muse")
    max_llm_layers = max(origin_llm_layers, muse_llm_layers)

    # --- Inputs (Via Real Processor) ---
    log_separator("Running Processor Pipeline")
    inputs = prepare_inputs_via_processor(ckpt_path, device, dtype)
    
    # --- Forward ---
    log_separator("Running Forward")
    origin_model.eval()
    muse_model.eval()
    
    origin_inputs = {k: v for k, v in inputs.items() if k != "vision_token_mask"}
    
    # 清空 Muse 模型 RoPE 调试输出
    vit_backbone_muse = None
    if hasattr(muse_model, "visual_tokenizer") and hasattr(muse_model.visual_tokenizer, "visual"):
        vit_backbone_muse = muse_model.visual_tokenizer.visual
    if vit_backbone_muse and hasattr(vit_backbone_muse, "encoder"):
        rope_module = vit_backbone_muse.encoder.rope
        if hasattr(rope_module, '_debug_rope_outputs'):
            rope_module._debug_rope_outputs = []

    # 清空 LLM RoPE 调试输出
    if hasattr(origin_mod, "_DEBUG_ROPE_OUTPUTS"):
        for key in origin_mod._DEBUG_ROPE_OUTPUTS.keys():
            origin_mod._DEBUG_ROPE_OUTPUTS[key] = None
    muse_llm_rope = None
    if hasattr(muse_model, "model") and hasattr(muse_model.model, "rope"):
        muse_llm_rope = muse_model.model.rope
        if hasattr(muse_llm_rope, "_debug_rope_outputs"):
            muse_llm_rope._debug_rope_outputs = []
        if hasattr(muse_llm_rope, "_debug_rope_intermediates"):
            # reset intermediates to avoid stale values
            for k in muse_llm_rope._debug_rope_intermediates.keys():
                muse_llm_rope._debug_rope_intermediates[k] = None

    with torch.no_grad():
        logger.info("Running Origin Forward...")
        origin_out = origin_model(**origin_inputs)
        logger.info("Running Muse Forward...")
        muse_out = muse_model(**inputs)

    # 手动捕获 logits（TiedLinear 无法挂 hook；Muse 返回 dict）
    def _extract_logits(out_obj):
        if hasattr(out_obj, "logits"):
            return out_obj.logits
        if isinstance(out_obj, dict):
            return out_obj.get("logits")
        if isinstance(out_obj, (tuple, list)) and out_obj:
            first = out_obj[0]
            if isinstance(first, dict):
                return first.get("logits")
            if isinstance(first, torch.Tensor):
                return first
        return None

    origin_logits = _extract_logits(origin_out)
    muse_logits = _extract_logits(muse_out)

    if origin_logits is not None:
        activations["origin"]["5.1 LLM Logits"] = origin_logits
    if muse_logits is not None:
        activations["muse"]["5.1 LLM Logits"] = muse_logits

    # --- 收集 RoPE 中间变量和输出 ---
    # Origin 模型: 从全局变量读取
    if ORIGIN_ROPE_DEBUG["inv_freq"] is not None:
        activations["origin"]["0.18 inv_freq"] = ORIGIN_ROPE_DEBUG["inv_freq"]
    if ORIGIN_ROPE_DEBUG["rope_emb_max_grid"] is not None:
        activations["origin"]["0.19 rope_emb_max_grid"] = ORIGIN_ROPE_DEBUG["rope_emb_max_grid"]
    if ORIGIN_ROPE_DEBUG["pids"] is not None:
        activations["origin"]["0.19 pids"] = ORIGIN_ROPE_DEBUG["pids"]
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
    
    # LLM RoPE: Origin 模型（Keye） - 全局 _DEBUG_ROPE_OUTPUTS
    if origin_mod._DEBUG_ROPE_OUTPUTS.get("inv_freq") is not None:
        activations["origin"]["4.R inv_freq"] = origin_mod._DEBUG_ROPE_OUTPUTS["inv_freq"]
    if origin_mod._DEBUG_ROPE_OUTPUTS.get("cos_before_chunk") is not None:
        activations["origin"]["4.R cos_before_chunk"] = origin_mod._DEBUG_ROPE_OUTPUTS["cos_before_chunk"]
    if origin_mod._DEBUG_ROPE_OUTPUTS.get("sin_before_chunk") is not None:
        activations["origin"]["4.R sin_before_chunk"] = origin_mod._DEBUG_ROPE_OUTPUTS["sin_before_chunk"]
    if origin_mod._DEBUG_ROPE_OUTPUTS.get("cos_after_chunk") is not None:
        activations["origin"]["4.R cos_after_chunk"] = origin_mod._DEBUG_ROPE_OUTPUTS["cos_after_chunk"]
    if origin_mod._DEBUG_ROPE_OUTPUTS.get("sin_after_chunk") is not None:
        activations["origin"]["4.R sin_after_chunk"] = origin_mod._DEBUG_ROPE_OUTPUTS["sin_after_chunk"]
    if origin_mod._DEBUG_ROPE_OUTPUTS.get("q_after_rope") is not None:
        activations["origin"]["4.R Q After RoPE"] = origin_mod._DEBUG_ROPE_OUTPUTS["q_after_rope"]
    if origin_mod._DEBUG_ROPE_OUTPUTS.get("k_after_rope") is not None:
        activations["origin"]["4.R K After RoPE"] = origin_mod._DEBUG_ROPE_OUTPUTS["k_after_rope"]

    # Muse 模型: 从 rope 模块读取中间变量
    if vit_backbone_muse and hasattr(vit_backbone_muse, "encoder"):
        rope_module = vit_backbone_muse.encoder.rope
        # 读取 inv_freq, rope_emb_max_grid, pids, rope_emb, cos, sin 中间变量
        if hasattr(rope_module, '_debug_rope_intermediates'):
            intermediates = rope_module._debug_rope_intermediates
            if intermediates.get("inv_freq") is not None:
                activations["muse"]["0.18 inv_freq"] = intermediates["inv_freq"]
            if intermediates.get("rope_emb_max_grid") is not None:
                activations["muse"]["0.19 rope_emb_max_grid"] = intermediates["rope_emb_max_grid"]
            if intermediates.get("pids") is not None:
                activations["muse"]["0.19 pids"] = intermediates["pids"]
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
            # 第一个是 q，第二个是 k (基于 attention.py 中的调用顺序)
            activations["muse"]["0.25 Q After RoPE"] = rope_module._debug_rope_outputs[0]
            activations["muse"]["0.25 K After RoPE"] = rope_module._debug_rope_outputs[1]

    # Muse 模型: LLM RoPE 中间变量
    if muse_llm_rope is not None:
        if hasattr(muse_llm_rope, "_debug_rope_intermediates"):
            intermediates = muse_llm_rope._debug_rope_intermediates
            if intermediates.get("inv_freq") is not None:
                activations["muse"]["4.R inv_freq"] = intermediates["inv_freq"]
            if intermediates.get("position_ids") is not None:
                activations["muse"]["4.R position_ids"] = intermediates["position_ids"]
            if intermediates.get("cos_before_chunk") is not None:
                activations["muse"]["4.R cos_before_chunk"] = intermediates["cos_before_chunk"]
            if intermediates.get("sin_before_chunk") is not None:
                activations["muse"]["4.R sin_before_chunk"] = intermediates["sin_before_chunk"]
            if intermediates.get("cos_after_chunk") is not None:
                activations["muse"]["4.R cos_after_chunk"] = intermediates["cos_after_chunk"]
            if intermediates.get("sin_after_chunk") is not None:
                activations["muse"]["4.R sin_after_chunk"] = intermediates["sin_after_chunk"]
            if intermediates.get("mrope_section") is not None:
                activations["muse"]["4.R mrope_section"] = intermediates["mrope_section"]
        if hasattr(muse_llm_rope, "_debug_rope_outputs") and len(muse_llm_rope._debug_rope_outputs) >= 2:
            activations["muse"]["4.R Q After RoPE"] = muse_llm_rope._debug_rope_outputs[0]
            activations["muse"]["4.R K After RoPE"] = muse_llm_rope._debug_rope_outputs[1]

    # --- Analysis ---
    log_separator("Deep Dive Analysis")
    
    checkpoints = [
        "0.0 ViT Embeddings Out",
        "0.1 LN1 Output",
        "0.2 Q_Proj Out",
        # ViT RoPE checkpoints (Muse only, Origin doesn't have debug info)
        # "0.18 inv_freq",
        # "0.19 rope_emb_max_grid",
        # "0.19 pids",
        # "0.20 rope_emb",
        # "0.21 cos_before_chunk",
        # "0.21 sin_before_chunk",
        # "0.22 cos_after_chunk",
        # "0.22 sin_after_chunk",
        # "0.25 Q After RoPE",
        # "0.25 K After RoPE",
        # LLM RoPE checkpoints
        "4.R inv_freq",
        "4.R position_ids",
        "4.R cos_before_chunk",
        "4.R sin_before_chunk",
        "4.R cos_after_chunk",
        "4.R sin_after_chunk",
        "4.R mrope_section",
        "4.R Q After RoPE",
        "4.R K After RoPE",
        "0.3 Attn Raw (Pre-Proj)",
        "0.4 Attn Out (Post-Proj)",
        "0.6 MLP Hidden (fc1)",
        "0.7 MLP Out (fc2)",
        "1.0 ViT Final Output",
        "2.0 Projector Output",
        "3.0 VQ[0] Output",
    ]

    for i in range(max_llm_layers):
        checkpoints.append(f"4.{i:02d} LLM Layer {i} Input")
        # Layer 0 细粒度检查点
        if i == 0:
            checkpoints.extend([
                "4.00a LLM L0 InputLN",
                "4.00b LLM L0 Q_Proj",
                "4.00c LLM L0 K_Proj",
                "4.00d LLM L0 V_Proj",
                "4.00e LLM L0 Attn Out",
                "4.00f LLM L0 PostAttnLN",
                "4.00g LLM L0 MLP Gate",
                "4.00h LLM L0 MLP Up",
                "4.00i LLM L0 MLP Down",
            ])
        checkpoints.append(f"4.{i:02d} LLM Layer {i} Output")

    checkpoints.extend([
        "5.0 LLM Final Hidden",
        "5.1 LLM Logits",
    ])
    
    # 需要详细打印值的检查点 (打印 dtype 和前 10 个值)
    rope_detail_checkpoints = {
        # LLM RoPE detailed
        "4.R inv_freq",
        "4.R position_ids",
        "4.R cos_before_chunk",
        "4.R sin_before_chunk",
        "4.R cos_after_chunk",
        "4.R sin_after_chunk",
        "4.R Q After RoPE",
        "4.R K After RoPE",
    }
    
    for k in checkpoints:
        if k in activations["origin"] and k in activations["muse"]:
            print_values = k in rope_detail_checkpoints
            compare_tensors_verbose(k, activations["origin"][k], activations["muse"][k], atol=2e-2, print_values=print_values)
        else:
            status_o = "Found" if k in activations["origin"] else "MISSING"
            status_m = "Found" if k in activations["muse"] else "MISSING"
            logger.warning(f"⚠️  Missing hook: {k} (Origin={status_o}, Muse={status_m})")

if __name__ == "__main__":
    test_pipeline_alignment()