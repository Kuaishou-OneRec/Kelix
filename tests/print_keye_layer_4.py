"""
Keye Vision Debugger: Layer 0 Internal Trace
============================================
功能：逐层、逐算子对比 Origin 和 Muse 模型的中间 Tensor。
特别是 'Attention Pre-Proj'，用于判断 RoPE 是否正确。
"""

import logging
import sys
import types
import numpy as np
import torch
from PIL import Image
from transformers import PretrainedConfig

# Muse imports
from muse.config.model_config import KeyeVisionConfig
from muse.models.keye_vit import KeyeVisionTransformer as MuseKeyeVisionModel
from muse.models.keye_vit.image_processing_keye import KeyeVisionImageProcessor
from muse.training.common import set_default_dtype

# === 路径设置 ===
CHECKPOINT_PATH = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/SigLIP/3.0.0.3/global_step18200/mp_rank_00_model_states.pt"

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# === 1. Config Hack (兼容 Origin 代码) ===
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
    mod = "muse.muse.models.keye_vit.configuration_keye"
    if mod in sys.modules: return
    c = types.ModuleType(mod)
    c.KeyeConfig = HFKeyeConfig
    c.KeyeVisionConfig = HFKeyeVisionConfig
    sys.modules[mod] = c
_ensure_origin_ready()
from muse.models.keye_vit import modeling_keye_origin as keye_origin
OriginKeyeVisionModel = keye_origin.SiglipVisionModel 

# === 2. 辅助函数 ===
def create_dummy_image(size: int = 384) -> Image.Image:
    rng = np.random.default_rng(seed=42)
    data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)

def log_header(title: str):
    logger.info(f"\n{'='*100}\n {title.center(98)} \n{'='*100}")

def compare_tensors(name: str, ref: torch.Tensor, cand: torch.Tensor, atol: float = 5e-2):
    """详细对比两个 Tensor，自动处理维度对齐"""
    # 1. 解包 tuple
    if isinstance(ref, (tuple, list)): ref = ref[0]
    if isinstance(cand, (tuple, list)): cand = cand[0]
    
    # 2. 转 float32 CPU
    ref = ref.detach().float().cpu()
    cand = cand.detach().float().cpu()

    # 3. 维度对齐尝试
    # 处理 Batch 维度不一致 (Origin: [1, S, D] vs Muse: [S, D] 或类似)
    if ref.shape != cand.shape:
        if ref.dim() == 3 and ref.shape[0] == 1 and cand.dim() == 2: ref = ref.squeeze(0)
        elif cand.dim() == 3 and cand.shape[0] == 1 and ref.dim() == 2: cand = cand.squeeze(0)
    
    # 处理转置 (Linear Output: Origin可能是 [B, S, D], Muse有时在某些中间层是 [B, D, S]?)
    if ref.shape != cand.shape and ref.numel() == cand.numel():
         if ref.dim() == 3 and ref.transpose(1, 2).shape == cand.shape: ref = ref.transpose(1, 2)

    # 4. 最终检查
    if ref.shape != cand.shape:
        logger.error(f"{name:<35} | ❌ SHAPE MISMATCH: Origin {ref.shape} vs Muse {cand.shape}")
        return

    # 5. 计算 Diff
    diff = (ref - cand).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 判定
    is_match = max_diff < atol
    tag = "✅ MATCH" if is_match else f"❌ DIFF"
    
    # 打印详细信息
    logger.info(f"{name:<35} | {tag:<10} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e}")
    if not is_match:
        # 打印最大误差出现的位置
        idx = torch.argmax(diff)
        logger.info(f"{' ':<35} |    -> Max Diff Index: {idx.item()} (Val: {ref.flatten()[idx]:.4f} vs {cand.flatten()[idx]:.4f})")


# === 3. 主测试逻辑 ===
def debug_layer0():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # 使用 BF16 复现问题
    
    muse_config = KeyeVisionConfig()
    origin_config = HFKeyeVisionConfig(**muse_config.dict())
    
    # --- Init Models ---
    with set_default_dtype(dtype):
        origin_model = OriginKeyeVisionModel(origin_config).eval()
        muse_model = MuseKeyeVisionModel(muse_config).eval()
    
    # --- Load Weights ---
    logger.info(f"Loading weights from {CHECKPOINT_PATH} ...")
    raw = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "module" in raw: raw = raw["module"]
    
    origin_state = {}
    for k, v in raw.items():
        clean = k
        for p in ["module.", "vision_tower.", "siglip."]:
            if clean.startswith(p): clean = clean[len(p):]
        if "vision_model" not in clean: clean = "vision_model." + clean
        origin_state["siglip." + clean] = v.to(dtype)
    
    muse_state = muse_model.convert_hf_state_dict(origin_state)
    muse_model.load_state_dict(muse_state, strict=False)
    
    origin_load = {k.replace("siglip.", ""): v for k, v in origin_state.items()}
    origin_model.load_state_dict(origin_load, strict=False)
    
    origin_model.to(device)
    muse_model.to(device)

    # --- Register Hooks ---
    activations = {"origin": {}, "muse": {}}

    def make_hook(model_name, layer_name, capture_input=False):
        def hook(module, inp, out):
            target = inp if capture_input else out
            if isinstance(target, (tuple, list)): target = target[0]
            activations[model_name][layer_name] = target.detach()
        return hook

    # Pointer to Layer 0
    origin_l0 = origin_model.vision_model.encoder.layers[0]
    muse_l0 = muse_model.encoder.layers[0]

    # [1] Input Norm
    origin_l0.layer_norm1.register_forward_hook(make_hook("origin", "1. LN1 Output"))
    muse_l0.sa_norm.register_forward_hook(make_hook("muse", "1. LN1 Output"))

    # [2] Q/K/V Projections (Linear Outputs)
    origin_l0.self_attn.q_proj.register_forward_hook(make_hook("origin", "2. Q_Proj Out"))
    origin_l0.self_attn.k_proj.register_forward_hook(make_hook("origin", "2. K_Proj Out"))
    origin_l0.self_attn.v_proj.register_forward_hook(make_hook("origin", "2. V_Proj Out"))
    
    muse_l0.attn.q_proj.register_forward_hook(make_hook("muse", "2. Q_Proj Out"))
    muse_l0.attn.k_proj.register_forward_hook(make_hook("muse", "2. K_Proj Out"))
    muse_l0.attn.v_proj.register_forward_hook(make_hook("muse", "2. V_Proj Out"))

    # [3] Attention Output (Pre-Projection) -> CRITICAL for RoPE check
    # Hook the INPUT of the output_proj linear layer to get the raw attention result
    origin_l0.self_attn.out_proj.register_forward_hook(make_hook("origin", "3. Attn Raw (Pre-Proj)", capture_input=True))
    muse_l0.attn.output_proj.register_forward_hook(make_hook("muse", "3. Attn Raw (Pre-Proj)", capture_input=True))

    # [4] Attention Output (Post-Projection)
    origin_l0.self_attn.out_proj.register_forward_hook(make_hook("origin", "4. Attn Out (Post-Proj)"))
    muse_l0.attn.output_proj.register_forward_hook(make_hook("muse", "4. Attn Out (Post-Proj)"))

    # [5] Residual 1 (Input to LN2) -> Checks (Embed + Attn_Out)
    origin_l0.layer_norm2.register_forward_hook(make_hook("origin", "5. Residual1 (LN2 In)", capture_input=True))
    muse_l0.mlp_norm.register_forward_hook(make_hook("muse", "5. Residual1 (LN2 In)", capture_input=True))

    # [6] MLP Hidden (fc1 / gate_proj -> w1)
    origin_l0.mlp.fc1.register_forward_hook(make_hook("origin", "6. MLP Hidden (fc1)"))
    muse_l0.mlp.w1.register_forward_hook(make_hook("muse", "6. MLP Hidden (fc1)"))  # 改为 w1

    # [7] MLP Output (fc2 / down_proj -> w2)
    origin_l0.mlp.fc2.register_forward_hook(make_hook("origin", "7. MLP Out (fc2)"))
    muse_l0.mlp.w2.register_forward_hook(make_hook("muse", "7. MLP Out (fc2)"))    # 改为 w2
    
    # --- Prepare Input ---
    proc = KeyeVisionImageProcessor(patch_size=muse_config.patch_size)
    img = create_dummy_image(muse_config.image_size)
    
    processed = proc.preprocess(images=img, return_tensors="pt")
    pix_val = processed["pixel_values"] # [784, 3, 14, 14]
    
    # Origin expects [B, S, C, H, W] for packing? Or just [B, ...]? 
    # SigLIP Vision Model logic in script: (b l) c h w -> b l d. 
    # If we pass [1, 784, 3, 14, 14], b=1, l=784. Correct.
    pixel_batch = pix_val.unsqueeze(0).to(device, dtype)
    
    seq_len = 784
    pids = torch.arange(seq_len, device=device).unsqueeze(0)
    grid = [(1, 28, 28)]
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    log_header("Running Inference")
    with torch.no_grad():
        origin_model(pixel_batch, position_ids=pids, image_grid_thw=grid, cu_seqlens=cu, interpolate_pos_encoding=True, window_size=-1, use_rope=True)
        muse_model(pixel_batch, position_ids=pids, image_grid_thw=grid, cu_seqlens=cu, interpolate_pos_encoding=True, has_learnable_position_embedding=True)

    # --- Analysis ---
    log_header("Layer 0 Internal Tensor Diff Analysis")
    keys = [
        "1. LN1 Output",
        "2. Q_Proj Out", "2. K_Proj Out", "2. V_Proj Out",
        "3. Attn Raw (Pre-Proj)",   # <--- 重点看这个
        "4. Attn Out (Post-Proj)",
        "5. Residual1 (LN2 In)",    # <--- 如果Attn Out对，这里错，就是残差加法顺序或精度问题
        "6. MLP Hidden (fc1)",
        "7. MLP Out (fc2)"
    ]

    for k in keys:
        if k in activations["origin"]:
            compare_tensors(k, activations["origin"][k], activations["muse"][k])
        else:
            logger.warning(f"Missing capture for {k}")

if __name__ == "__main__":
    debug_layer0()