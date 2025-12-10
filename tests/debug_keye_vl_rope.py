"""
Keye Vision RoPE Strict Debugger
================================
Comparing Muse vs Origin RoPE Logic with bit-level precision check.
"""

import torch
import torch.nn as nn
import logging
import sys
import math

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
logger = logging.getLogger()

# ==============================================================================
# 1. MOCK Flash Attn (To run without installing flash-attn)
# ==============================================================================
def mock_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def mock_apply_rotary_emb(x, cos, sin):
    # This mimics flash_attn.layers.rotary.apply_rotary_emb
    # x: [Batch, Seq, Head, Dim]
    # cos, sin: [Batch, Seq, Dim] (we need to unsqueeze for Head dim broadcasting)
    
    # FlashAttn assumes cos/sin shape is broadcastable. 
    # Usually cos is [Seq, Dim] or [Batch, Seq, Dim].
    # We need to unsqueeze head dim (-2) for elementwise mul.
    if cos.dim() == x.dim() - 1:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        
    return (x * cos) + (mock_rotate_half(x) * sin)

# ==============================================================================
# 2. MUSE Implementation (Copied from your snippet)
# ==============================================================================
class Muse_TwoD_RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, *, max_grid_size: int = 4096, base: int = 10000) -> None:
        super().__init__()
        self.dim = head_dim // 2
        self.base = base
        
        # Note: You used torch.arange(..., dtype=torch.float) in your snippet
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("freqs_cache", torch.empty(0), persistent=False)

        self.debug_cos = None # DEBUG HOOK

    def build_freq_cache(self, seqlen: int):
        dtype = self.inv_freq.dtype
        device = self.inv_freq.device
        seq = torch.arange(seqlen, device=device, dtype=dtype)
        freqs = torch.outer(seq, self.inv_freq)
        self.register_buffer("freqs_cache", freqs, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
        if isinstance(input_pos, dict):
            height_ids = input_pos["height"]
            width_ids = input_pos["width"]
        else:
            height_ids, width_ids = input_pos
            
        max_pos = max(height_ids.max().item(), width_ids.max().item()) + 1
        if self.freqs_cache.numel() == 0 or max_pos > self.freqs_cache.shape[0]:
            self.build_freq_cache(max_pos + 128)

        freqs_h = self.freqs_cache[height_ids]
        freqs_w = self.freqs_cache[width_ids]
        
        # Muse Logic: Concat H and W
        rope_emb_half = torch.cat([freqs_h, freqs_w], dim=-1)
        
        cos_half = rope_emb_half.cos() 
        sin_half = rope_emb_half.sin()
        
        self.debug_cos = cos_half # Capture for debug comparison

        # Muse calls flash_apply_rotary_emb(x.float(), cos.float(), sin.float()).to(x.dtype)
        out = mock_apply_rotary_emb(
            x.float(), cos_half.float(), sin_half.float()
        ).to(dtype=x.dtype)
        return out

# ==============================================================================
# 3. ORIGIN Implementation (Reconstructed from your snippet)
# ==============================================================================
class Origin_SigLIPRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

def origin_apply_rotary_pos_emb_flashatt(q, cos, sin):
    # Origin Logic: Chunk -> Contiguous -> Float -> Apply -> TypeAs
    # Note: Origin snippet chunks `dim=-1`, takes [0]
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    
    # Origin snippet calls apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    return mock_apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q), cos

# ==============================================================================
# 4. TEST RUNNER
# ==============================================================================
def run_rope_comparison():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 # Use BF16 to match your training/inference environment
    
    # Parameters matching Keye/SigLIP
    HEAD_DIM = 72 # Typical for SigLIP (1152 embed / 16 heads = 72)
    DIM_HALF = HEAD_DIM // 2 # 36
    
    # 14x14 Grid Input
    GRID_H, GRID_W = 14, 14
    SEQ_LEN = GRID_H * GRID_W
    BATCH = 1
    NUM_HEADS = 1 # Test one head is enough
    
    logger.info(f"Running Comparison. Device={device}, Dtype={dtype}, HeadDim={HEAD_DIM}")

    # 1. Create Inputs
    q = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    
    # IDs: 0,0, 0,1, ..., 13,13
    h_ids = torch.arange(GRID_H, device=device).repeat_interleave(GRID_W) 
    w_ids = torch.arange(GRID_W, device=device).repeat(GRID_H)
    
    # ==========================
    # Run ORIGIN Logic
    # ==========================
    origin_rope_module = Origin_SigLIPRotaryEmbedding(DIM_HALF).to(device).to(dtype)
    
    # Simulate SiglipEncoder logic
    pids = torch.stack([h_ids, w_ids], dim=-1) # [Seq, 2]
    max_grid_size = pids.max() + 1
    
    rope_emb_max_grid = origin_rope_module(max_grid_size) # [Max, DimHalf]
    rope_emb = rope_emb_max_grid[pids].flatten(1) # [Seq, 2*DimHalf] -> [Seq, HeadDim]
    
    # CRITICAL ORIGIN STEP: Repeat (1, 2)
    rope_emb_repeated = rope_emb.repeat(1, 2) # [Seq, 2*HeadDim]
    
    cos_origin_full = rope_emb_repeated.cos()
    sin_origin_full = rope_emb_repeated.sin()
    
    # Apply (Function mimics Origin's apply_rotary_pos_emb_flashatt)
    q_origin_out, cos_origin_chunked = origin_apply_rotary_pos_emb_flashatt(q, cos_origin_full, sin_origin_full)

    # ==========================
    # Run MUSE Logic
    # ==========================
    muse_rope_module = Muse_TwoD_RotaryEmbedding(HEAD_DIM).to(device).to(dtype)
    
    input_pos = {"height": h_ids, "width": w_ids}
    q_muse_out = muse_rope_module(q, input_pos=input_pos)
    cos_muse = muse_rope_module.debug_cos

    # ==========================
    # Compare
    # ==========================
    logger.info("\n--- Results ---")
    
    # 1. Compare the Cosine Tensors used for calculation
    # Origin's effective cos is the chunked one
    cos_diff = (cos_origin_chunked - cos_muse).abs().max().item()
    logger.info(f"Cosine Tensor Diff: {cos_diff:.6e}")
    if cos_diff > 1e-6:
        logger.error("❌ Cosine tensors differ! Logic for generating freqs is inconsistent.")
    else:
        logger.info("✅ Cosine tensors match.")

    # 2. Compare Final Output
    out_diff = (q_origin_out - q_muse_out).abs().max().item()
    logger.info(f"Output Q Diff     : {out_diff:.6e}")
    
    if out_diff < 1e-5: # BF16 tolerance
        logger.info("\n✅ SUCCESS: Implementations are equivalent in BF16.")
    else:
        logger.info("\n❌ FAILURE: Significant difference detected.")
        idx = torch.argmax((q_origin_out - q_muse_out).abs())
        logger.info(f"Max diff at index {idx.item()}: Origin={q_origin_out.flatten()[idx]}, Muse={q_muse_out.flatten()[idx]}")

if __name__ == "__main__":
    run_rope_comparison()