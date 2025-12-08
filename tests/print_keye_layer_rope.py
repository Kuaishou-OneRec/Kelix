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
    
    This implementation matches the Logic that PASSED the bitwise unit test.
    """

    def __init__(self, head_dim: int, *, max_grid_size: int = 4096, base: int = 10000) -> None:
        super().__init__()
        self.dim = head_dim // 2
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
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

        max_pos = max(height_ids.max().item(), width_ids.max().item()) + 1
        
        seq = torch.arange(max_pos, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        # 1. Fetch Frequencies (BF16)
        freqs_h = freqs[height_ids] 
        freqs_w = freqs[width_ids]
        
        # 2. Concat [H, W] -> [Seq, 36] (BF16)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        
        # 3. Compute Cos/Sin (BF16)
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 4. Expand to Full Head Dim
        cos = torch.cat([cos, cos], dim=-1) # [Seq, 72]
        sin = torch.cat([sin, sin], dim=-1)

        # 5. Apply Rotation in FP32 (Matches FlashAttn Kernel Logic)
        cos = cos.unsqueeze(-2) # Broadcast heads
        sin = sin.unsqueeze(-2)

        x_float = x.float()
        cos_float = cos.float()
        sin_float = sin.float()

        out = (x_float * cos_float) + (self._rotate_half(x_float) * sin_float)

        return out.to(dtype=x.dtype)
    
# ==========================================
# 3. Test Runner
# ==========================================
def run_rope_test():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    HEAD_DIM = 72
    DIM = 36
    SEQ_LEN = 196
    GRID_H, GRID_W = 14, 14
    BATCH = 1
    
    logger.info(f"Test Config: HeadDim={HEAD_DIM}, Dtype={dtype}")

    q = torch.randn(BATCH, SEQ_LEN, 4, HEAD_DIM, device=device, dtype=dtype)
    
    h_ids = torch.arange(GRID_H, device=device).repeat_interleave(GRID_W)
    w_ids = torch.arange(GRID_W, device=device).repeat(GRID_H)
    
    # --- HF Setup ---
    hf_rope = HF_SigLIPRotaryEmbedding(DIM).to(device).to(dtype)
    
    # --- HF Forward ---
    rope_emb_max = hf_rope(4096)
    pids = torch.stack([h_ids, w_ids], dim=-1)
    rope_emb_hf = rope_emb_max[pids].flatten(1).repeat(1, 2)
    
    cos_hf = rope_emb_hf.cos()
    sin_hf = rope_emb_hf.sin()
    
    cos_hf_apply = cos_hf.unsqueeze(1)
    sin_hf_apply = sin_hf.unsqueeze(1)
    
    q_hf_out = hf_apply_rotary_pos_emb(q.float(), cos_hf_apply.float(), sin_hf_apply.float()).to(dtype)
    
    # --- Muse Setup ---
    muse_rope = Muse_KeyeAxialRotaryEmbedding(HEAD_DIM).to(device).to(dtype) 
    
    # --- Muse Forward ---
    input_pos = {"height": h_ids, "width": w_ids}
    q_muse_out = muse_rope(q, input_pos=input_pos)
    
    # --- Comparison ---
    logger.info("\n--- Tensor Diffs ---")
    
    diff = (q_hf_out - q_muse_out).abs()
    logger.info(f"Output Diff | Max: {diff.max().item():.2e} | Mean: {diff.mean().item():.2e}")
    
    # Verify internal frequencies
    with torch.no_grad():
        # Force rebuild to ensure we are checking what happened inside forward
        # muse_rope.build_freq_cache(4096) 
        muse_freqs_h = muse_rope.freqs_cache[h_ids]
        muse_freqs_w = muse_rope.freqs_cache[w_ids]
        muse_freqs = torch.cat([muse_freqs_h, muse_freqs_w], dim=-1)
        muse_cos = torch.cat([muse_freqs.cos(), muse_freqs.cos()], dim=-1)
        
    hf_cos_native = cos_hf # This is BF16
    
    cos_diff = (hf_cos_native - muse_cos).abs()
    logger.info(f"Cos Diff    | Max: {cos_diff.max().item():.2e}")

    if diff.max().item() < 1e-6:
        logger.info("\n✅ SUCCESS: RoPE outputs match perfectly!")
    else:
        logger.info("\n❌ FAILURE: Diffs persist.")

if __name__ == "__main__":
    run_rope_test()