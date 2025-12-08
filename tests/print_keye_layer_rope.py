"""
Keye Vision RoPE Unit Test (Strict Comparison)
==============================================
Goal: Isolate RoPE calculation to pinpoint diff source.
"""

import torch
import torch.nn as nn
import math
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
logger = logging.getLogger()

# ==========================================
# 1. HuggingFace Implementation (Copied & Minimal)
# ==========================================
class HF_SigLIPRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        # HF Logic: dim is head_dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

def hf_rotate_half(x):
    # Standard GPT-NeoX style rotate half
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def hf_apply_rotary_pos_emb(q, cos, sin):
    # Simulated HF application (Python version of flash_attn kernel)
    # HF Logic:
    # cos is [Batch, Seq, HeadDim] (after repeat(1,2))
    # But inside apply_rotary_pos_emb_flashatt, it does cos.chunk(2)[0]
    # So effective cos is [Batch, Seq, HeadDim//2]
    
    # However, to simulate what the Kernel does on the full q:
    # The kernel effectively applies the rotation using the half-size cos/sin
    # to the pairs of q.
    
    # We can simulate this using the formula:
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # BUT, we need to handle the dimensions of cos/sin.
    
    # In HF code provided:
    # rope_emb = rope_emb.repeat(1, 2) -> Shape [..., 72]
    # cos = rope_emb.cos() -> Shape [..., 72]
    # cos_chunk = cos.chunk(2, dim=-1)[0] -> Shape [..., 36]
    
    # If we use Python formula with q(72) and cos_chunk(36), we need to broadcast/repeat cos_chunk.
    # q * cos_full + rotate_half(q) * sin_full
    # where cos_full is [cos_chunk, cos_chunk].
    
    # Let's strictly follow the HF provided snippet logic flow:
    # 1. rope_emb (72) -> cos (72)
    # 2. chunk(2)[0] -> cos_half (36)
    # 3. apply_rotary_emb(q, cos_half, ...)
    # 4. FlashAttn apply_rotary_emb assumes cos matches half-dim of q.
    
    # So for Python simulation:
    # Effective COS for elementwise mul is cat([cos_half, cos_half])
    # Which is exactly what 'rope_emb.cos()' was BEFORE chunking!
    # So we can just use rope_emb.cos() directly for simulation.
    
    return (q * cos) + (hf_rotate_half(q) * sin)

# ==========================================
# 2. Muse Implementation (Your Reverted Code)
# ==========================================
class Muse_KeyeAxialRotaryEmbedding(nn.Module):
    """
    Axial RoPE that strictly mimics HuggingFace SigLIP's BF16 precision behavior.
    
    CRITICAL DETAILS to match HF:
    1. Full RoPE: Rotates all 72 dimensions (not partial).
    2. BF16 Indices: 'seq' uses inv_freq.dtype. If BF16, indices > 256 are quantized!
    3. BF16 Trig: cos/sin computed in BF16.
    4. FP32 App: Final rotate application is in FP32.
    """

    def __init__(self, head_dim: int, *, max_grid_size: int = 4096, base: int = 10000) -> None:
        super().__init__()
        # SigLIP config: dim = head_dim // 2 (e.g. 36 for head_dim 72)
        self.dim = head_dim // 2
        self.base = base
        
        # Init inv_freq in FP32 (Standard), registered as buffer.
        # It will become BF16 when model.to(bfloat16) is called.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build initial cache
        self.build_freq_cache(max_grid_size)

    def build_freq_cache(self, seqlen: int):
        # [CRITICAL] Use the current dtype of inv_freq (BF16).
        # HF: seq = torch.arange(..., dtype=self.inv_freq.dtype)
        # This truncates indices > 256 if dtype is BF16. We MUST replicate this.
        dtype = self.inv_freq.dtype
        
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=dtype)
        freqs = torch.outer(seq, self.inv_freq) # BF16 * BF16 -> BF16
        self.register_buffer("freqs_cache", freqs, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # GPT-NeoX style rotation
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
        if input_pos is None:
            return x
        
        if isinstance(input_pos, dict):
            height_ids = input_pos["height"]
            width_ids = input_pos["width"]
        else:
            height_ids, width_ids = input_pos

        # Dynamic resize
        max_pos = max(height_ids.max().item(), width_ids.max().item()) + 1
        if max_pos > self.freqs_cache.shape[0]:
            self.build_freq_cache(max_pos + 128)

        # 1. Fetch Frequencies (BF16)
        freqs_h = self.freqs_cache[height_ids] # [Seq, 18]
        freqs_w = self.freqs_cache[width_ids]
        
        # 2. Concat [H, W] (BF16) -> [Seq, 36]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        
        # 3. Compute Cos/Sin (BF16)
        cos = freqs.cos() 
        sin = freqs.sin()
        
        # 4. Expand for GPT-NeoX RoPE (BF16)
        # We need to apply cos(36) to both halves of x(72).
        # x1(36) * cos(36) ... x2(36) * cos(36)
        # So we repeat cos to 72.
        cos = torch.cat([cos, cos], dim=-1) # [Seq, 72]
        sin = torch.cat([sin, sin], dim=-1)

        # 5. Apply Rotation in FP32 (Matches HF apply_rotary_emb logic)
        cos = cos.unsqueeze(-2) # Broadcast heads: [Seq, 1, 72]
        sin = sin.unsqueeze(-2)

        x_float = x.float()
        cos_float = cos.float()
        sin_float = sin.float()

        # x * cos + rotate_half(x) * sin
        out = (x_float * cos_float) + (self._rotate_half(x_float) * sin_float)

        return out.to(dtype=x.dtype)
# ==========================================
# 3. Test Runner
# ==========================================
def run_rope_test():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # Config
    HEAD_DIM = 72 # SigLIP default
    DIM = HEAD_DIM // 2 # 36
    SEQ_LEN = 196 # 14x14
    GRID_H, GRID_W = 14, 14
    BATCH = 1
    
    logger.info(f"Test Config: HeadDim={HEAD_DIM}, Dtype={dtype}")

    # 1. Inputs
    # Random Query [B, S, H, D]
    q = torch.randn(BATCH, SEQ_LEN, 4, HEAD_DIM, device=device, dtype=dtype)
    
    # Position IDs (Grid)
    # h_ids: 0,0,0... 1,1,1...
    # w_ids: 0,1,2... 0,1,2...
    h_ids = torch.arange(GRID_H, device=device).repeat_interleave(GRID_W)
    w_ids = torch.arange(GRID_W, device=device).repeat(GRID_H)
    
    # ==========================================
    # Run HF Logic
    # ==========================================
    hf_rope = HF_SigLIPRotaryEmbedding(DIM).to(device).to(dtype)
    
    # HF Encoder Logic Simulation
    # 1. Get max grid freqs
    # NOTE: HF 'forward' returns freqs [Max, Dim//2]
    # Assume max_grid is enough
    rope_emb_max = hf_rope(4096) # [4096, 18]
    
    # 2. Indexing
    pids = torch.stack([h_ids, w_ids], dim=-1) # [Seq, 2]
    # rope_emb_max is 2D. HF logic uses pids to select.
    # In HF code: rope_emb = rope_emb_max_grid[pids]
    # rope_emb_max_grid has shape [MaxGrid, 18].
    # pids has shape [Seq, 2].
    # Result rope_emb has shape [Seq, 2, 18].
    rope_emb_hf = rope_emb_max[pids] 
    
    # 3. Flatten & Repeat
    rope_emb_hf = rope_emb_hf.flatten(1) # [Seq, 36]
    rope_emb_hf = rope_emb_hf.repeat(1, 2) # [Seq, 72]
    
    # 4. Cos/Sin (Simulate BF16 behavior if HF is BF16)
    # The crucial question: Does HF compute cos/sin in BF16?
    # rope_emb_hf is BF16 because model is BF16.
    cos_hf = rope_emb_hf.cos()
    sin_hf = rope_emb_hf.sin()
    
    # 5. Apply
    # HF casts to float BEFORE apply in 'apply_rotary_pos_emb_flashatt'
    # But cos_hf was computed in BF16!
    # q is BF16.
    cos_hf_apply = cos_hf.unsqueeze(1) # Broadcast head [Seq, 1, 72]
    sin_hf_apply = sin_hf.unsqueeze(1)
    
    q_hf_out = hf_apply_rotary_pos_emb(q.float(), cos_hf_apply.float(), sin_hf_apply.float()).to(dtype)
    
    # ==========================================
    # Run Muse Logic
    # ==========================================
    muse_rope = Muse_KeyeAxialRotaryEmbedding(HEAD_DIM).to(device) # Keep defaults (FP32 cache)
    
    # Forward
    input_pos = {"height": h_ids, "width": w_ids}
    q_muse_out = muse_rope(q, input_pos=input_pos)
    
    # ==========================================
    # Comparison
    # ==========================================
    logger.info("\n--- Tensor Diffs ---")
    
    # 1. Compare Output
    diff = (q_hf_out - q_muse_out).abs()
    logger.info(f"Output Diff | Max: {diff.max().item():.2e} | Mean: {diff.mean().item():.2e}")
    
    # 2. Debug: Compare Cosines
    # Muse internal cos (re-compute for check)
    with torch.no_grad():
        muse_freqs_h = muse_rope.freqs_cache[h_ids]
        muse_freqs_w = muse_rope.freqs_cache[w_ids]
        muse_freqs = torch.cat([muse_freqs_h, muse_freqs_w], dim=-1)
        muse_cos = torch.cat([muse_freqs.cos(), muse_freqs.cos()], dim=-1) # FP32
        
    # HF Cos (BF16 -> Float)
    hf_cos_float = cos_hf.float()
    
    cos_diff = (hf_cos_float - muse_cos).abs()
    logger.info(f"Cos Diff    | Max: {cos_diff.max().item():.2e} | Mean: {cos_diff.mean().item():.2e}")
    
    if cos_diff.max().item() > 1e-3:
        logger.info(">>> Cause Identified: HF computes cos/sin in BF16, Muse uses FP32!")
        logger.info(">>> Trying to emulate HF BF16 behavior in Muse...")
        
        # Emulation Test
        muse_freqs_bf16 = muse_freqs.to(dtype)
        muse_cos_bf16 = muse_freqs_bf16.cos()
        muse_cos_bf16_full = torch.cat([muse_cos_bf16, muse_cos_bf16], dim=-1)
        
        cos_diff_emulated = (hf_cos_float - muse_cos_bf16_full.float()).abs()
        logger.info(f"Emulated Cos Diff | Max: {cos_diff_emulated.max().item():.2e}")
        
        if cos_diff_emulated.max().item() < 1e-3:
            logger.info(">>> FIX FOUND: Cast freqs to BF16 before cos/sin!")

if __name__ == "__main__":
    run_rope_test()