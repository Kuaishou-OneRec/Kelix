"""
SigLIP Layer 0 Internal Debugging (Deep Dive into RoPE)
Focus: Verifying Axial RoPE Logic against a Reference Implementation
"""

import os
import sys
import logging
import torch
import numpy as np
from transformers import AutoImageProcessor, SiglipVisionModel as HFSiglipVisionModel


import transformers.models.siglip.modeling_siglip as hf_siglip_code

# 1. 保存原始函数（以防万一）
original_apply_rope = hf_siglip_code.apply_rotary_pos_emb

# 2. 定义带打印的新函数
def debug_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 打印形状
    print(f"\n[HOOK DEBUG] apply_rotary_pos_emb inputs:")
    print(f"  q shape: {q.shape}")     # 预期: [Batch, Seq, Heads, HeadDim]
    print(f"  cos shape: {cos.shape}") # 预期: [Seq, HeadDim] (注意这里是否包含 batch 维)
    
    # 打印数值样本
    # 检查 HeadDim 维度上的 cos 值
    # 如果是 Axial RoPE，前一半应该对应 Height 频率，后一半对应 Width 频率
    # 我们可以打印前几个数和中间几个数
    head_dim = q.shape[-1]
    mid = head_dim // 2
    
    cos_flat = cos.flatten()
    print(f"  cos sample (Head Start): {cos_flat[:5].detach().cpu().numpy()}")
    print(f"  cos sample (Head Mid - Boundary): {cos_flat[mid-2:mid+3].detach().cpu().numpy()}")
    
    # 执行原始逻辑
    cos_unsqueezed = cos.unsqueeze(unsqueeze_dim)
    sin_unsqueezed = sin.unsqueeze(unsqueeze_dim)
    
    # 手动计算一遍 q_embed 的第一个值，看看公式到底长啥样
    # q[0] * cos[0] + rotate_half(q)[0] * sin[0]
    
    q_embed = (q * cos_unsqueezed) + (hf_siglip_code.rotate_half(q) * sin_unsqueezed)
    k_embed = (k * cos_unsqueezed) + (hf_siglip_code.rotate_half(k) * sin_unsqueezed)
    
    print(f"  q output sample: {q_embed[0, 0, 0, :5].detach().cpu().numpy()}")
    print("-" * 50)
    
    return q_embed, k_embed

# 3. 替换掉 HF 的函数
hf_siglip_code.apply_rotary_pos_emb = debug_apply_rotary_pos_emb

logger.info("✅ Successfully monkey-patched HF apply_rotary_pos_emb for debugging!")


from muse.config import SiglipVisionConfig
from muse.models.Siglip import SiglipVisionTransformer as SiglipVisionModel
from muse.training.common import set_default_dtype

# Logging Setup
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. Reference Implementation (Ground Truth)
#    Copied standard logic to verify against Muse's Module
# =============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Standard RoPE Application"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
class ReferenceAxialRoPE:
    """
    A minimal, functional implementation of Axial RoPE to serve as Ground Truth.
    """
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base
        # Precompute inv_freq (Defaults to CPU)
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    def get_freqs(self, seq_len, device):
        # 1. Ensure inv_freq is on the target device
        inv_freq = self.inv_freq.to(device)
        
        # 2. Create t on the target device with matching dtype
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        
        # 3. Outer Product (Both are now on device)
        freqs = torch.outer(t, inv_freq)
        
        # 4. Concat to get [cos, sin] structure equivalent
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, q, height, width):
        # q shape: [batch, seq, heads, head_dim]
        head_dim = q.shape[-1]
        
        # 1. Split Q into Height part and Width part (Axial Split)
        q_h, q_w = q.chunk(2, dim=-1)
        
        axis_dim = head_dim // 2 # 32 for dim 64
        
        # 2. Get Cos/Sin for H and W
        sub_rope = ReferenceAxialRoPE(axis_dim, self.base)
        
        cos_h, sin_h = sub_rope.get_freqs(height, q.device) # [H, 32]
        cos_w, sin_w = sub_rope.get_freqs(width, q.device)  # [W, 32]
        
        # 3. Broadcast to Sequence (Meshgrid equivalent)
        # We need to map linear sequence index to (y, x) grid coordinates
        # Row-major: (0,0), (0,1), ..., (1,0), ...
        
        # h_idx: [0, 0, ..., 1, 1, ...] -> Repeats each row index 'width' times
        h_idx = torch.arange(height, device=q.device).unsqueeze(1).repeat(1, width).flatten()
        
        # w_idx: [0, 1, ..., 0, 1, ...] -> Repeats the column sequence 'height' times
        w_idx = torch.arange(width, device=q.device).repeat(height)
        
        # Handle cases where actual seq_len might be shorter/longer (e.g. padding or cls token issues, though SigLIP usually strictly grid)
        seq_len = q.shape[1]
        valid_len = min(seq_len, len(h_idx))
        
        h_idx = h_idx[:valid_len]
        w_idx = w_idx[:valid_len]
        
        cos_h_flat = cos_h[h_idx] # [seq, 32]
        sin_h_flat = sin_h[h_idx]
        cos_w_flat = cos_w[w_idx] # [seq, 32]
        sin_w_flat = sin_w[w_idx]
        
        # 4. Apply RoPE Independently
        # q_h: [B, Seq, Heads, 32]
        # cos: [Seq, 32] -> unsqueeze to [1, Seq, 1, 32]
        
        # Apply to Height Part
        q_h_embed, _ = apply_rotary_pos_emb(q_h, q_h, cos_h_flat.unsqueeze(0), sin_h_flat.unsqueeze(0), unsqueeze_dim=2)
        
        # Apply to Width Part
        q_w_embed, _ = apply_rotary_pos_emb(q_w, q_w, cos_w_flat.unsqueeze(0), sin_w_flat.unsqueeze(0), unsqueeze_dim=2)
        
        # 5. Concat
        q_out = torch.cat([q_h_embed, q_w_embed], dim=-1)
        return q_out

# =============================================================================
# Helper Functions
# =============================================================================

def compare_tensors(name, t1, t2, atol=1e-3):
    t1 = t1.detach().float().cpu()
    t2 = t2.detach().float().cpu()
    
    if t1.shape != t2.shape:
        logger.error(f"❌ {name} SHAPE MISMATCH: Ref={t1.shape}, Muse={t2.shape}")
        return

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    status = "✅ MATCH" if max_diff < atol else f"❌ MISMATCH (Max: {max_diff:.2e})"
    
    logger.info(f"{name:<35} | {status} | MeanDiff: {mean_diff:.2e}")
    if max_diff >= atol:
        logger.info(f"   Ref Sample  : {t1.flatten()[:5].numpy()}")
        logger.info(f"   Muse Sample : {t2.flatten()[:5].numpy()}")

def _build_siglip_config(hf_cfg):
    """Build Muse config from HF config."""
    image_size = hf_cfg.get("image_size", 384)
    patch_size = hf_cfg.get("patch_size", 14)
    default_max_seq_len = (image_size // patch_size) ** 2
    return SiglipVisionConfig(
        model_class="SiglipVisionTransformer",
        image_size=image_size,
        patch_size=patch_size,
        num_channels=hf_cfg.get("num_channels", 3),
        hidden_size=hf_cfg.get("hidden_size", 1152),
        num_hidden_layers=hf_cfg.get("num_hidden_layers", 27),
        num_attention_heads=hf_cfg.get("num_attention_heads", 16),
        intermediate_size=hf_cfg.get("intermediate_size", 4304),
        max_seq_len=hf_cfg.get("max_seq_len", default_max_seq_len),
        layer_norm_eps=hf_cfg.get("layer_norm_eps", 1e-6),
        attention_dropout=hf_cfg.get("attention_dropout", 0.0),
        has_learnable_position_embedding=hf_cfg.get("has_learnable_position_embedding", False),
        use_qk_norm=hf_cfg.get("use_qk_norm", False),
        qk_norm_eps=hf_cfg.get("qk_norm_eps", 1e-6),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        attention_function="eager",
        output_attentions=False,
        output_hidden_states=False,
    )

# =============================================================================
# Main Test
# =============================================================================

def test_rope_implementation():
    torch.manual_seed(0)
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch14-384"
    
    logger.info("1. Loading Models...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    hf_model = HFSiglipVisionModel.from_pretrained(checkpoint_dir, device_map="auto", torch_dtype="auto")
    hf_model.eval()
    
    device = hf_model.device
    dtype = hf_model.dtype
    
    # Muse Setup
    config_dict = hf_model.config.to_dict()
    muse_config = _build_siglip_config(config_dict)
    with set_default_dtype(dtype):
        muse_model = SiglipVisionModel(muse_config)

    # Weight Loading
    logger.info("2. Loading Weights...")
    hf_state_dict = hf_model.state_dict()
    prefixed_dict = {}
    for k, v in hf_state_dict.items():
        if k.startswith("vision_model."): new_k = f"siglip.{k}"
        else: new_k = f"siglip.vision_model.{k}"
        prefixed_dict[new_k] = v
        
    converted_dict = muse_model.convert_hf_state_dict(prefixed_dict)
    for k in converted_dict: converted_dict[k] = converted_dict[k].to(dtype)
    muse_model.load_state_dict(converted_dict, strict=False)
    muse_model = muse_model.to(device)
    muse_model.eval()

    # =========================================================================
    # 3. Hook Setup
    # =========================================================================
    activations = {"hf": {}, "muse": {}}

    def get_hook(model_name, layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple): output = output[0]
            activations[model_name][layer_name] = output.detach()
        return hook

    # Hook Q Projection (Pre-RoPE) - We use HF's as the source of truth for input
    hf_model.vision_model.encoder.layers[0].self_attn.q_proj.register_forward_hook(get_hook("hf", "q_proj"))
    
    # Hook Muse RoPE Output (Post-RoPE)
    # The Muse Encoder passes RoPE as a module to Attn. 
    # Usually Attn calls: self.pos_embeddings(q, input_pos)
    # So we want to hook `muse_model.encoder.rope`
    muse_model.encoder.rope.register_forward_hook(get_hook("muse", "rope_out"))

    # =========================================================================
    # 4. Forward
    # =========================================================================
    logger.info("3. Running Forward...")
    # Use standard 384x384 image -> 27x27 grid
    image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        hf_model(**inputs)
        
        # Manual grid for Muse
        h_grid, w_grid = 384//14, 384//14
        image_grid = [(1, h_grid, w_grid)] * inputs["pixel_values"].shape[0]
        muse_model(pixel_values=inputs["pixel_values"], image_grid_thw=image_grid)

    logger.info("\n" + "="*60)
    logger.info("RoPE DIAGNOSIS")
    logger.info("="*60)

    # 1. Get the Raw Q from HF (which we know matches Muse's Q Linear output)
    # Shape: [Batch, Seq, Hidden] -> Needs Reshape to [Batch, Seq, Heads, HeadDim]
    q_raw = activations["hf"]["q_proj"]
    batch_size, seq_len, _ = q_raw.shape
    num_heads = muse_config.num_attention_heads
    head_dim = muse_config.hidden_size // num_heads
    
    # Reshape Q to match what RoPE expects: [B, S, H, D]
    q_reshaped = q_raw.view(batch_size, seq_len, num_heads, head_dim)
    
    # 2. Run Reference Axial RoPE Logic locally
    logger.info("Calculating Reference RoPE...")
    ref_rope = ReferenceAxialRoPE(head_dim // 2, base=10000) # dim passed is axis_dim
    # q_reshaped should be passed. 
    # Note: Our ref_rope.forward expects q, height, width
    q_ref_rotated = ref_rope.forward(q_reshaped, h_grid, w_grid)

    # 3. Get Muse RoPE Output
    q_muse_rotated = activations["muse"]["rope_out"]
    
    # Muse RoPE forward takes x, then splits x.chunk(2, dim=-1) -> x_h, x_w -> concat
    # Muse forward also returns concatenated result [B, S, H, D] (same shape as input)
    
    # 4. Compare
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    compare_tensors("RoPE Calculation Check", q_ref_rotated, q_muse_rotated, atol=tol)

if __name__ == "__main__":
    test_rope_implementation()