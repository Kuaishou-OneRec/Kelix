"""
Keye Vision RoPE Strict Debugger (Fixed Mock)
=============================================
Comparing Muse vs Origin RoPE Logic with bit-level precision check.
Fix: Mock function now handles half-size cos/sin broadcasting (mimicking flash_attn).
"""

import torch
import torch.nn as nn
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
logger = logging.getLogger()

# ==============================================================================
# 1. FIXED MOCK Flash Attn 
# ==============================================================================
def mock_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def mock_apply_rotary_emb(x, cos, sin):
    # Fix: Handle FlashAttn convention where cos/sin can be half size
    # x: [..., HeadDim] (72)
    # cos: [..., HalfHeadDim] (36) or [..., HeadDim] (72)
    
    # 1. Unsqueeze head dim for broadcasting if needed
    if cos.dim() == x.dim() - 1:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    
    # 2. Handle Half-Size Cos/Sin (Implicit repeat for broadcasting)
    # Origin's cos is 36, x is 72. We need to repeat cos to 72 to multiply.
    if cos.shape[-1] == x.shape[-1] // 2:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
    return (x * cos) + (mock_rotate_half(x) * sin)

# ==============================================================================
# 2. MUSE Implementation (Based on your provided code)
# ==============================================================================
class Muse_TwoD_RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, *, max_grid_size: int = 4096, base: int = 10000) -> None:
        super().__init__()
        self.dim = head_dim // 2 # 36
        self.base = base
        
        # arange(0, 36, 2) -> 18 elements. inv_freq size is 18.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("freqs_cache", torch.empty(0), persistent=False)

        self.debug_cos = None 

    def build_freq_cache(self, seqlen: int):
        dtype = self.inv_freq.dtype
        device = self.inv_freq.device
        seq = torch.arange(seqlen, device=device, dtype=dtype)
        freqs = torch.outer(seq, self.inv_freq) # [Seq, 18]
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

        freqs_h = self.freqs_cache[height_ids] # [Seq, 18]
        freqs_w = self.freqs_cache[width_ids] # [Seq, 18]
        
        # Muse: Cat H and W -> [Seq, 36]
        rope_emb_half = torch.cat([freqs_h, freqs_w], dim=-1)
        
        cos_half = rope_emb_half.cos() # [Seq, 36]
        sin_half = rope_emb_half.sin()
        
        self.debug_cos = cos_half 

        # x is [Seq, 72], cos is [Seq, 36]. Mock handles repeat.
        out = mock_apply_rotary_emb(
            x.float(), cos_half.float(), sin_half.float()
        ).to(dtype=x.dtype)
        return out

# ==============================================================================
# 3. ORIGIN Implementation (Based on your provided code)
# ==============================================================================
class Origin_SigLIPRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim # 36
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        # arange(0, 36, 2) -> 18 elements. inv_freq size is 18.
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq) # [Seq, 18]
        return freqs

def origin_apply_rotary_pos_emb_flashatt(q, cos, sin):
    # Origin Logic: Chunk 72 -> 36
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    
    # cos is now 36. Mock handles repeat.
    return mock_apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q), cos

# ==============================================================================
# 4. TEST RUNNER
# ==============================================================================
def run_rope_comparison():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    HEAD_DIM = 72
    DIM_HALF = HEAD_DIM // 2 # 36
    
    GRID_H, GRID_W = 14, 14
    SEQ_LEN = GRID_H * GRID_W
    BATCH = 1
    NUM_HEADS = 1 
    
    logger.info(f"Running Comparison. Device={device}, Dtype={dtype}, HeadDim={HEAD_DIM}")

    q = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    
    h_ids = torch.arange(GRID_H, device=device).repeat_interleave(GRID_W) 
    w_ids = torch.arange(GRID_W, device=device).repeat(GRID_H)
    
    # --- Origin Run ---
    origin_rope = Origin_SigLIPRotaryEmbedding(DIM_HALF).to(device).to(dtype)
    
    pids = torch.stack([h_ids, w_ids], dim=-1)
    max_grid_size = pids.max() + 1
    
    rope_emb_max = origin_rope(max_grid_size) # [Max, 18]
    rope_emb = rope_emb_max[pids].flatten(1)  # [Seq, 2*18=36]
    
    rope_emb_repeated = rope_emb.repeat(1, 2) # [Seq, 72]
    
    cos_origin_full = rope_emb_repeated.cos() # [Seq, 72]
    sin_origin_full = rope_emb_repeated.sin()
    
    q_origin_out, cos_origin_chunked = origin_apply_rotary_pos_emb_flashatt(q, cos_origin_full, sin_origin_full)

    # --- Muse Run ---
    muse_rope = Muse_TwoD_RotaryEmbedding(HEAD_DIM).to(device).to(dtype)
    input_pos = {"height": h_ids, "width": w_ids}
    q_muse_out = muse_rope(q, input_pos=input_pos)
    cos_muse = muse_rope.debug_cos # [Seq, 36]

    # --- Compare ---
    logger.info("\n--- Results ---")
    
    # Compare Cosine (36 vs 36)
    cos_diff = (cos_origin_chunked - cos_muse).abs().max().item()
    logger.info(f"Cosine Tensor Diff: {cos_diff:.6e}")
    if cos_diff > 1e-6:
        logger.error("❌ Cosine tensors differ!")
    else:
        logger.info("✅ Cosine tensors match.")

    # Compare Output (72 vs 72)
    out_diff = (q_origin_out - q_muse_out).abs().max().item()
    logger.info(f"Output Q Diff     : {out_diff:.6e}")
    
    if out_diff < 1e-5:
        logger.info("\n✅ SUCCESS: Implementations are equivalent in BF16.")
    else:
        logger.info("\n❌ FAILURE: Significant difference detected.")
        idx = torch.argmax((q_origin_out - q_muse_out).abs())
        logger.info(f"Max diff at index {idx.item()}")

if __name__ == "__main__":
    run_rope_comparison()