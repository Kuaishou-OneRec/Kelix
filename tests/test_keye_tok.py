"""
Integration test to ensure Muse KeyeImageTokenizer matches Origin KeyeImageTokenizer implementation.
Focuses on final outputs only, without intermediate RoPE debugging.

This script:
1. Loads a Keye-VL checkpoint
2. Extracts the visual_tokenizer (KeyeImageTokenizer) weights and saves to local path
3. Loads the saved weights into Muse KeyeImageTokenizer (keye_tok)
4. Compares forward pass final outputs on a fake image with origin model
"""

import os
import sys
import logging
import glob
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from contextlib import contextmanager
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
from tests.models.keye_vl_tokenizer_image import modeling_keye_origin as origin_mod
from muse.models.keye_tokenizer.modeling import KeyeImageTokenizer as MuseKeyeImageTokenizer
from tests.models.keye_vl_tokenizer_image.image_processing_keye import SiglipImageProcessor
from muse.config import KeyeVisionConfig, KeyeTokenizerConfig
from muse.training.common import set_default_dtype

from PIL import Image, ImageDraw

# No RoPE debug imports needed for final output test

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_CKPT = os.environ.get(
    "KEYE_VL_CHECKPOINT",
    "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/"
)

# Path to save extracted tokenizer weights
SAVE_TOKENIZER_PATH = "/llm_reco/maosiyang/keye_tok"


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
    
    # SafeTensor/Bin loader
    state_dict = {}
    st_files = sorted(glob.glob(str(path / "*.safetensors")))
    if st_files:
        try:
            from safetensors.torch import safe_open
            for f in st_files:
                with safe_open(f, framework="pt", device=device) as open_f:
                    for k in open_f.keys():
                        state_dict[k] = open_f.get_tensor(k)
        except ImportError:
            logger.warning("safetensors not found, skipping .safetensors files.")
        return state_dict
    
    bin_files = sorted(glob.glob(str(path / "*.bin")))
    if bin_files:
        for f in bin_files:
            if any(x in f for x in ["training_args", "optimizer", "scheduler"]):
                continue
            part = torch.load(f, map_location=device)
            if "module" in part:
                part = part["module"]
            state_dict.update(part)
        return state_dict
    
    pt_files = sorted(glob.glob(str(path / "*.pt")))
    if pt_files:
        for f in pt_files:
            part = torch.load(f, map_location=device)
            if "module" in part:
                part = part["module"]
            state_dict.update(part)
        return state_dict
    raise ValueError("No checkpoint found.")



def create_dummy_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象，用于测试。
    
    :param size: 图像的大小，默认为 (64, 64)
    :param fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
    :param outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
    :param outline_width: 圆的轮廓宽度，默认为 5
    :return: 生成的 PIL Image 对象
    """
    # 创建一个新的图像对象
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # 计算圆的坐标（图像中心为圆心）
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    # 绘制圆
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color,
                 outline=outline_color,
                 width=outline_width)
    return image




def compare_tensors_verbose(name: str, tensor_origin: Any, tensor_muse: Any, atol=1e-3, print_values=False):
    """Compare two tensors with verbose output."""
    def unwrap(x):
        if hasattr(x, 'last_hidden_state'):
            return x.last_hidden_state
        if isinstance(x, (tuple, list)):
            return x[0] if x else None
        if isinstance(x, dict):
            for k in ['logits', 'z_q', 'last_hidden_state', 'x', 'z_e', 'indices']:
                if k in x:
                    return x[k]
            return list(x.values())[0] if x else None
        return x

    t1_raw = unwrap(tensor_origin)
    t2_raw = unwrap(tensor_muse)

    if not isinstance(t1_raw, torch.Tensor) or not isinstance(t2_raw, torch.Tensor):
        logger.warning(f"⚠️  [{name}] Skipped: Not tensors (Origin: {type(t1_raw)}, Muse: {type(t2_raw)})")
        return False, None

    # Handle lists of tensors
    if isinstance(t1_raw, list) and all(isinstance(x, torch.Tensor) for x in t1_raw):
        if len(t1_raw) != len(t2_raw):
            logger.error(f"{name:<45} | ❌ LIST LENGTH MISMATCH: Origin={len(t1_raw)} vs Muse={len(t2_raw)}")
            return False, None
        
        all_match = True
        max_diff_val = 0.0
        for i, (item1, item2) in enumerate(zip(t1_raw, t2_raw)):
            match, diff = compare_tensors_verbose(f"{name}[{i}]", item1, item2, atol, print_values)
            if not match:
                all_match = False
            if diff is not None:
                max_diff_val = max(max_diff_val, diff)
        return all_match, max_diff_val

    t1_dtype = t1_raw.dtype
    t2_dtype = t2_raw.dtype
    
    t1 = t1_raw.detach().float().cpu()
    t2 = t2_raw.detach().float().cpu()
    
    # Try common shape adjustments
    if t1.shape != t2.shape:
        if t1.numel() == t2.numel():
            t2 = t2.view(t1.shape)
        elif t1.dim() == 3 and t2.dim() == 2 and t1.shape[1:] == t2.shape:
            t1 = t1.squeeze(0)
        elif t2.dim() == 3 and t1.dim() == 2 and t2.shape[1:] == t1.shape:
            t2 = t2.squeeze(0)
        elif t1.dim() == 4 and t2.dim() == 5 and t2.shape[0] == 1 and t1.shape == t2.squeeze(0).shape:
            t2 = t2.squeeze(0)

    if t1.shape != t2.shape:
        logger.error(f"{name:<45} | ❌ SHAPE ERR  | Origin={t1_raw.shape} vs Muse={t2_raw.shape}")
        return False, None

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    match_status = "✅ MATCH" if max_diff <= atol else "❌ MISMATCH"
    logger.info(f"{name:<45} | {match_status:<12} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    
    if print_values:
        logger.info(f"   -> Origin dtype: {t1_dtype}, shape: {t1_raw.shape}")
        logger.info(f"   -> Muse   dtype: {t2_dtype}, shape: {t2_raw.shape}")
        t1_first10 = t1.flatten()[:10].numpy()
        t2_first10 = t2.flatten()[:10].numpy()
        logger.info(f"   -> Origin first 10: [{', '.join([f'{v:.6f}' for v in t1_first10])}]")
        logger.info(f"   -> Muse   first 10: [{', '.join([f'{v:.6f}' for v in t2_first10])}]")
    
    if max_diff >= atol:
        max_idx = torch.argmax(diff)
        logger.info(f"   -> Max Diff Index: {max_idx.item()}")
        logger.info(f"   -> Origin Val    : {t1.flatten()[max_idx]:.6f}")
        logger.info(f"   -> Muse Val      : {t2.flatten()[max_idx]:.6f}")
        return False, max_diff
    return True, max_diff


# =========================================================================
# Hook System
# =========================================================================
activations = {"origin": {}, "muse": {}}


def make_hook(model_name, layer_name, capture_input=False, key=None):
    def hook(module, inp, out):
        target = inp if capture_input else out
        if isinstance(target, (tuple, list)):
            target = target[0]
        if isinstance(target, dict):
            if key and key in target:
                target = target[key]
            else:
                for k in ['z_q', 'logits', 'last_hidden_state', 'x', 'z_e', 'indices']:
                    if k in target:
                        target = target[k]
                        break
        activations[model_name][layer_name] = target.detach() if isinstance(target, torch.Tensor) else target
    return hook


def register_tokenizer_hooks(model, name_prefix):
    """Register detailed hooks for KeyeImageTokenizer components."""
    logger.info(f"Registering hooks for {name_prefix} KeyeImageTokenizer...")
    
    tokenizer = model

    # Hooks for visual (ViT backbone)
    if hasattr(tokenizer, "visual"):
        visual_model = tokenizer.visual
        
        # Handle origin vs muse structure difference
        if hasattr(visual_model, "vision_model"):
            # Origin: visual -> SiglipVisionModel -> vision_model
            vit_backbone = visual_model.vision_model
        else:
            # Muse: visual -> KeyeVisionTransformer (direct)
            vit_backbone = visual_model
        
        if hasattr(vit_backbone, "embeddings"):
            vit_backbone.embeddings.register_forward_hook(make_hook(name_prefix, "0.0 ViT Embeddings Out"))
        
        if hasattr(vit_backbone, "encoder") and hasattr(vit_backbone.encoder, "layers"):
            layer0 = vit_backbone.encoder.layers[0]
            if name_prefix == "origin":
                layer0.layer_norm1.register_forward_hook(make_hook(name_prefix, "0.1 LN1 Output"))
                layer0.self_attn.q_proj.register_forward_hook(make_hook(name_prefix, "0.2 Q_Proj Out"))
                layer0.self_attn.k_proj.register_forward_hook(make_hook(name_prefix, "0.2 K_Proj Out"))
                layer0.self_attn.v_proj.register_forward_hook(make_hook(name_prefix, "0.2 V_Proj Out"))
                layer0.self_attn.out_proj.register_forward_hook(make_hook(name_prefix, "0.3 Attn Raw (Pre-Proj)", capture_input=True))
                layer0.self_attn.out_proj.register_forward_hook(make_hook(name_prefix, "0.4 Attn Out (Post-Proj)"))
                layer0.mlp.fc1.register_forward_hook(make_hook(name_prefix, "0.6 MLP Hidden (fc1)"))
                layer0.mlp.fc2.register_forward_hook(make_hook(name_prefix, "0.7 MLP Out (fc2)"))
            elif name_prefix == "muse":
                layer0.sa_norm.register_forward_hook(make_hook(name_prefix, "0.1 LN1 Output"))
                layer0.attn.q_proj.register_forward_hook(make_hook(name_prefix, "0.2 Q_Proj Out"))
                layer0.attn.k_proj.register_forward_hook(make_hook(name_prefix, "0.2 K_Proj Out"))
                layer0.attn.v_proj.register_forward_hook(make_hook(name_prefix, "0.2 V_Proj Out"))
                layer0.attn.output_proj.register_forward_hook(make_hook(name_prefix, "0.3 Attn Raw (Pre-Proj)", capture_input=True))
                layer0.attn.output_proj.register_forward_hook(make_hook(name_prefix, "0.4 Attn Out (Post-Proj)"))
                layer0.mlp.w1.register_forward_hook(make_hook(name_prefix, "0.6 MLP Hidden (fc1)"))
                layer0.mlp.w2.register_forward_hook(make_hook(name_prefix, "0.7 MLP Out (fc2)"))
        
        # Final ViT output
        vit_backbone.register_forward_hook(make_hook(name_prefix, "1.0 ViT Final Output"))

    # Hooks for Projector (mlp_AR)
    if hasattr(tokenizer, 'mlp_AR'):
        tokenizer.mlp_AR.register_forward_hook(make_hook(name_prefix, "2.0 Projector Output"))
    
    # Hooks for pre_llm_aligner
    if hasattr(tokenizer, 'pre_llm_aligner') and not isinstance(tokenizer.pre_llm_aligner, nn.Identity):
        tokenizer.pre_llm_aligner.register_forward_hook(make_hook(name_prefix, "2.5 Pre-LLM Aligner Output"))

    # Hooks for Encoder
    if hasattr(tokenizer, 'encoder'):
        tokenizer.encoder.register_forward_hook(make_hook(name_prefix, "3.0 Encoder Output"))

    # Hooks for Quantizer
    if hasattr(tokenizer, 'quantizer') and len(tokenizer.quantizer) > 0:
        for i, vq in enumerate(tokenizer.quantizer):
            vq.register_forward_hook(make_hook(name_prefix, f"4.{i} VQ[{i}] Z_Q Output", key="z_q"))


def _build_muse_tokenizer_config(raw_cfg: Dict[str, Any]) -> KeyeTokenizerConfig:
    """Build Muse KeyeTokenizerConfig from raw config dictionary."""
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
        vq_temperature=1.0,
        vq_temperature_decay=0.999,
        vq_min_temperature=0.1,
    )
    return tokenizer_cfg


def extract_and_save_tokenizer_weights(
    full_state_dict: Dict[str, torch.Tensor],
    save_path: str,
    raw_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Extract visual_tokenizer weights from full Keye-VL checkpoint,
    convert to Muse format, and save to local path.
    
    Returns the converted Muse-style state dict.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract visual_tokenizer.* keys from full state dict
    origin_tokenizer_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith("visual_tokenizer."):
            new_k = k[len("visual_tokenizer."):]
            origin_tokenizer_state_dict[new_k] = v
    
    logger.info(f"Extracted {len(origin_tokenizer_state_dict)} keys for visual_tokenizer")
    

    muse_state_dict = {}
    for k, v in origin_tokenizer_state_dict.items():
        new_k = k
        
        # Convert visual.vision_model.* to visual.*
        if k.startswith("visual.vision_model."):
            new_k = "visual." + k[len("visual.vision_model."):]
        
        # Convert encoder.layers.X.layer_norm1 -> encoder.layers.X.sa_norm
        new_k = new_k.replace(".layer_norm1.", ".sa_norm.")
        # Convert encoder.layers.X.layer_norm2 -> encoder.layers.X.mlp_norm
        new_k = new_k.replace(".layer_norm2.", ".mlp_norm.")
        # Convert self_attn -> attn
        new_k = new_k.replace(".self_attn.", ".attn.")
        # Convert out_proj -> output_proj
        new_k = new_k.replace(".out_proj.", ".output_proj.")
        # Convert mlp.fc1 -> mlp.w1
        new_k = new_k.replace(".mlp.fc1.", ".mlp.w1.")
        # Convert mlp.fc2 -> mlp.w2
        new_k = new_k.replace(".mlp.fc2.", ".mlp.w2.")
        # Convert post_layernorm -> ln_post
        new_k = new_k.replace(".post_layernorm.", ".ln_post.")
        
        muse_state_dict[new_k] = v
    
    logger.info(f"Converted to {len(muse_state_dict)} Muse-format keys")
    
    # 3. Save the converted weights
    weights_path = save_dir / "pytorch_model.bin"
    torch.save(muse_state_dict, weights_path)
    logger.info(f"Saved Muse tokenizer weights to: {weights_path}")
    
    # 4. Save the config as JSON
    config_path = save_dir / "config.json"
    tokenizer_cfg = _build_muse_tokenizer_config(raw_cfg)
    with open(config_path, "w") as f:
        json.dump(tokenizer_cfg.dict(), f, indent=2)
    logger.info(f"Saved tokenizer config to: {config_path}")
    
    # 5. Copy preprocessor_config.json if exists in original checkpoint
    original_ckpt_dir = Path(DEFAULT_CKPT)
    preprocessor_config_src = original_ckpt_dir / "preprocessor_config.json"
    if preprocessor_config_src.exists():
        import shutil
        shutil.copy(preprocessor_config_src, save_dir / "preprocessor_config.json")
        logger.info(f"Copied preprocessor_config.json to: {save_dir}")
    
    return muse_state_dict


def load_muse_tokenizer_from_saved(
    save_path: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[MuseKeyeImageTokenizer, KeyeTokenizerConfig]:
    """
    Load Muse KeyeImageTokenizer from saved weights.
    """
    save_dir = Path(save_path)
    
    # Load config
    config_path = save_dir / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Build KeyeTokenizerConfig from saved config
    vision_cfg_dict = config_dict.get("vision_config", {})
    vision_cfg = KeyeVisionConfig(**vision_cfg_dict)
    
    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=config_dict.get("llm_hidden_size", 4096),
        embedding_dim=config_dict.get("embedding_dim", 128),
        init_embedding_dim=config_dict.get("init_embedding_dim", 4096),
        codebook_size=config_dict.get("codebook_size", 65536),
        n_q_tokens=config_dict.get("n_q_tokens", 8),
        split_voc=config_dict.get("split_voc", 1),
        add_voc_reducer=config_dict.get("add_voc_reducer", False),
        split_dim=config_dict.get("split_dim", False),
        vq_sampling_mode=config_dict.get("vq_sampling_mode", "argmin"),
        vq_temperature=config_dict.get("vq_temperature", 1.0),
        vq_temperature_decay=config_dict.get("vq_temperature_decay", 0.999),
        vq_min_temperature=config_dict.get("vq_min_temperature", 0.1),
    )
    
    # Initialize model
    with set_default_dtype(dtype):
        muse_tokenizer = MuseKeyeImageTokenizer(tokenizer_cfg).to(device)
    
    # Load weights
    weights_path = save_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location="cpu")
    
    missing, unexpected = muse_tokenizer.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Muse tokenizer missing keys: {len(missing)}")
        for k in missing[:10]:
            logger.warning(f"  - {k}")
    if unexpected:
        logger.warning(f"Muse tokenizer unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            logger.warning(f"  - {k}")
    
    muse_tokenizer.eval()
    logger.info(f"Loaded Muse tokenizer from: {save_path}")
    
    return muse_tokenizer, tokenizer_cfg


def prepare_tokenizer_inputs(ckpt_path: str, device: str, dtype: torch.dtype, image_size: int = 384):
    """
    Prepare inputs for tokenizer testing using ImageProcessor.
    Returns pixel_values and image_grid_thw for tokenizer forward pass.
    """
    logger.info("🎨 Generating Random Image...")
    image = create_dummy_image()
    
    logger.info("⚙️ Loading ImageProcessor...")
    image_processor = SiglipImageProcessor.from_pretrained(ckpt_path)
    
    logger.info("🔄 Processing Image...")
    processed = image_processor.preprocess(images=image, return_tensors="pt")
    
    pixel_values = processed["pixel_values"].to(device, dtype=dtype)
    image_grid_thw = processed["image_grid_thw"].to(device)
    
    logger.info(f"   -> Pixel Values Shape: {pixel_values.shape}")
    logger.info(f"   -> Image Grid: {image_grid_thw.tolist()}")
    
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def test_keye_tokenizer_alignment():
    """Ensure Muse KeyeImageTokenizer aligns with Origin KeyeImageTokenizer."""
    with _mock_context_parallel():
        _run_keye_tokenizer_alignment()


def _run_keye_tokenizer_alignment():
    # === 1. Configuration ===
    checkpoint_path = DEFAULT_CKPT
    save_path = SAVE_TOKENIZER_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    
    logger.info(f"Running tokenizer alignment test on device={device}, dtype={dtype}")
    logger.info(f"Source Checkpoint: {checkpoint_path}")
    logger.info(f"Save Path: {save_path}")

    # Load raw config and full checkpoint
    raw_cfg = _load_config_json(checkpoint_path)
    tokenizer_cfg = _build_muse_tokenizer_config(raw_cfg)
    
    # === 2. Load full checkpoint and extract tokenizer weights ===
    log_separator("Extracting and Saving Tokenizer Weights")
    full_state_dict = _load_checkpoint_robust(checkpoint_path, device="cpu")
    
    # Extract, convert, and save weights to local path
    muse_state_dict = extract_and_save_tokenizer_weights(
        full_state_dict, save_path, raw_cfg
    )
    
    # === 3. Initialize Origin KeyeImageTokenizer ===
    log_separator("Initializing Origin KeyeImageTokenizer")
    origin_tokenizer_config = origin_mod.KeyeImageTokenizerConfig.from_pretrained(checkpoint_path)
    with set_default_dtype(dtype):
        origin_tokenizer = origin_mod.KeyeImageTokenizer(
            origin_tokenizer_config,
            vq_sampling_mode="argmin",
        ).to(device, dtype)
    origin_tokenizer.eval()
    logger.info(f"Origin tokenizer n_q_tokens: {origin_tokenizer.n_q_tokens}")

    # Load weights into Origin tokenizer
    origin_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith("visual_tokenizer."):
            new_k = k[len("visual_tokenizer."):]
            origin_state_dict[new_k] = v
    
    missing_o, unexpected_o = origin_tokenizer.load_state_dict(origin_state_dict, strict=False)
    if missing_o:
        logger.warning(f"Origin tokenizer missing keys: {len(missing_o)} keys")
        for k in missing_o[:5]:
            logger.warning(f"  - {k}")
    if unexpected_o:
        logger.warning(f"Origin tokenizer unexpected keys: {len(unexpected_o)} keys")

    # === 4. Load Muse KeyeImageTokenizer from saved weights ===
    log_separator("Loading Muse KeyeImageTokenizer from Saved Weights")
    muse_tokenizer, loaded_cfg = load_muse_tokenizer_from_saved(save_path, device, dtype)
    logger.info(f"Muse tokenizer n_q_tokens: {muse_tokenizer.n_q_tokens}")

    # === 5. Register Hooks ===
    log_separator("Registering Hooks")
    register_tokenizer_hooks(origin_tokenizer, "origin")
    register_tokenizer_hooks(muse_tokenizer, "muse")

    # === 6. Input Preparation ===
    log_separator("Preparing Inputs")
    inputs = prepare_tokenizer_inputs(
        checkpoint_path, device, dtype, 
        image_size=tokenizer_cfg.vision_config.image_size
    )
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    # No RoPE debug outputs to clear for final output test

    # === 7. Forward Pass ===
    log_separator("Running Forward Pass")
    with torch.no_grad():
        logger.info("Running Origin KeyeImageTokenizer Forward...")
        origin_output = origin_tokenizer(pixel_values, image_grid_thw)
        
        logger.info("Running Muse KeyeImageTokenizer Forward...")
        muse_output = muse_tokenizer(pixel_values, image_grid_thw)

    # Store final outputs in activations for comparison
    activations["origin"]["Final z_q"] = origin_output["z_q"]
    activations["origin"]["Final z_e"] = origin_output["z_e"]
    activations["origin"]["Final codebook_loss"] = origin_output["codebook_loss"]
    activations["origin"]["Final commitment_loss"] = origin_output["commitment_loss"]
    activations["origin"]["Final indices"] = origin_output["indices"]
    activations["origin"]["Final x (image_embeds)"] = origin_output["x"]

    activations["muse"]["Final z_q"] = muse_output["z_q"]
    activations["muse"]["Final z_e"] = muse_output["z_e"]
    activations["muse"]["Final codebook_loss"] = muse_output["codebook_loss"]
    activations["muse"]["Final commitment_loss"] = muse_output["commitment_loss"]
    activations["muse"]["Final indices"] = muse_output["indices"]
    activations["muse"]["Final x (image_embeds)"] = muse_output["x"]

    # No RoPE intermediate debugging needed for final output test

    # === 8. Analysis ===
    log_separator("Deep Dive Analysis - Tokenizer Components")
    
    comparison_keys = [
        "0.0 ViT Embeddings Out",
        "0.1 LN1 Output",
        "0.2 Q_Proj Out",
        "0.2 K_Proj Out",
        "0.2 V_Proj Out",
        "0.3 Attn Raw (Pre-Proj)",
        "0.4 Attn Out (Post-Proj)",
        "0.6 MLP Hidden (fc1)",
        "0.7 MLP Out (fc2)",
        "1.0 ViT Final Output",
        "2.0 Projector Output",
        "2.5 Pre-LLM Aligner Output",
        "3.0 Encoder Output",
    ]
    
    # Add VQ output keys dynamically
    n_q = tokenizer_cfg.n_q_tokens
    for i in range(n_q):
        comparison_keys.append(f"4.{i} VQ[{i}] Z_Q Output")
    
    # Final outputs
    comparison_keys.extend([
        "Final z_q",
        "Final z_e",
        "Final codebook_loss",
        "Final commitment_loss",
        "Final indices",
        "Final x (image_embeds)",
    ])

    all_matches = True
    for k in comparison_keys:
        if k in activations["origin"] and k in activations["muse"]:
            print_values = "Final" in k
            is_match, max_diff = compare_tensors_verbose(
                k, activations["origin"][k], activations["muse"][k],
                atol=2e-2, print_values=print_values
            )
            if not is_match:
                all_matches = False
        else:
            status_o = "Found" if k in activations["origin"] else "MISSING"
            status_m = "Found" if k in activations["muse"] else "MISSING"
            if status_o != "MISSING" or status_m != "MISSING":
                logger.warning(f"⚠️  Missing hook data for {k} (Origin={status_o}, Muse={status_m})")

    # === 9. Test forward_image_tokens ===
    log_separator("Testing forward_image_tokens")
    
    # 模拟 VLM 模型的 vocab_size（与 origin 模型对齐）
    # 从 config 中获取或使用默认值
    vocab_size = raw_cfg.get("vocab_size", 151936)  # Qwen3 默认 vocab_size
    logger.info(f"Using vocab_size: {vocab_size}")
    
    with torch.no_grad():
        # Origin 模型的 forward_image_tokens 逻辑（在 KeyeForConditionalGeneration 中）
        # 这里手动实现同样的逻辑
        logger.info("Computing Origin forward_image_tokens...")
        origin_vq_out = origin_tokenizer(pixel_values, image_grid_thw)
        origin_indices = torch.stack([x_i for x_i in origin_vq_out['indices']], dim=0).T
        n_q_tokens_origin = tokenizer_cfg.n_q_tokens
        codebook_size_origin = tokenizer_cfg.codebook_size
        device = next(iter(origin_tokenizer.parameters())).device
        codebook_offsets_origin = torch.arange(n_q_tokens_origin, device=device)[None] * codebook_size_origin // n_q_tokens_origin
        origin_aligned_indices = vocab_size + origin_indices + codebook_offsets_origin
        
        # Muse 模型的 forward_image_tokens
        logger.info("Computing Muse forward_image_tokens...")
        muse_aligned_indices = muse_tokenizer.forward_image_tokens(pixel_values, image_grid_thw, vocab_size)
    
    logger.info(f"Origin aligned_indices shape: {origin_aligned_indices.shape}")
    logger.info(f"Muse aligned_indices shape: {muse_aligned_indices.shape}")
    
    # 对比 forward_image_tokens 输出
    forward_image_tokens_match, max_diff = compare_tensors_verbose(
        "forward_image_tokens", origin_aligned_indices, muse_aligned_indices,
        atol=0, print_values=True  # indices 应该完全匹配
    )
    
    if forward_image_tokens_match:
        logger.info("✅ forward_image_tokens outputs match exactly!")
    else:
        all_matches = False
        logger.info("❌ forward_image_tokens outputs differ!")
        # 打印更多调试信息
        diff_mask = origin_aligned_indices != muse_aligned_indices
        num_diff = diff_mask.sum().item()
        logger.info(f"   -> Number of different indices: {num_diff} / {origin_aligned_indices.numel()}")
        if num_diff > 0 and num_diff <= 20:
            diff_positions = torch.where(diff_mask)
            for i in range(min(num_diff, 10)):
                pos = tuple(d[i].item() for d in diff_positions)
                logger.info(f"   -> Position {pos}: Origin={origin_aligned_indices[pos].item()}, Muse={muse_aligned_indices[pos].item()}")

    # === 10. Final Result ===
    log_separator("Test Result")
    if all_matches:
        logger.info("✅✅✅ SUCCESS: All KeyeImageTokenizer outputs match within tolerance!")
    else:
        logger.info("❌ FAILURE: Some KeyeImageTokenizer outputs differ beyond tolerance.")
    
    return all_matches


if __name__ == "__main__":
    test_keye_tokenizer_alignment()
