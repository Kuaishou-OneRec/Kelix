class KeyeAxialRotaryEmbedding(nn.Module):
    """把二维 (h, w) RoPE 封装成 attention 可直接调用的模块。"""

    def __init__(self, head_dim: int, *, max_grid_size: int, base: int = 10_000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim 必须能被 2 整除，才能按 h/w 分半。")
        axis_dim = head_dim // 2
        self.height_rope = RotaryPositionalEmbeddings(
            dim=axis_dim, max_seq_len=max_grid_size, base=base
        )
        self.width_rope = RotaryPositionalEmbeddings(
            dim=axis_dim, max_seq_len=max_grid_size, base=base
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _lookup(self, rope: RotaryPositionalEmbeddings, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_ids.dtype != torch.long:
            pos_ids = pos_ids.long()
        # cache shape: [max_seq_len, dim, 2]
        # gathered shape: [..., dim, 2]
        cache = rope.cache
        gathered = cache[pos_ids]
        
        # Split into cos/sin
        cos = gathered[..., 0] # [..., dim]
        sin = gathered[..., 1] # [..., dim]
        
        # Handle broadcasting if needed (e.g. if dim=3 results in [seq, dim])
        # Based on your provided code, you want to ensure the last dim is preserved
        # and potentially unsqueeze for broadcasting if necessary.
        # But standard Attention expects [B, S, H, D] or [B, S, D].
        # Here we just return the values, unsqueezing logic usually happens inside lookup if needed 
        # or we let broadcasting handle it. Your original code unsqueezed -2.
        
        if cos.dim() == 2: # [Batch*Seq, Dim] or [Seq, Dim]
             pass # Dimensions are likely fine for broadcasting against [B, S, H, D] 
                  # as long as we reshape later or rely on broadcasting rules.
                  # But looking at your original code:
                  # cos = gathered[..., 0].unsqueeze(-2) -> [..., 1, dim]
                  # This suggests preparing for [B, S, NumHeads, HeadDim]
        
        # Replicating your original _lookup logic for safety:
        cos = gathered[..., 0].unsqueeze(-2)
        sin = gathered[..., 1].unsqueeze(-2)
        
        if cos.dim() == 3:  # [seq, 1, dim] -> [1, seq, 1, dim] for batch broadcasting?
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            
        return cos, sin

    def forward(self, x: torch.Tensor, *, input_pos=None, **_) -> torch.Tensor:
        if input_pos is None:
            return x
        if isinstance(input_pos, dict):
            height_ids = input_pos["height"]
            width_ids = input_pos["width"]
        else:
            height_ids, width_ids = input_pos

        # Lookup returns: [..., axis_dim] where axis_dim = head_dim // 2 (e.g., 36)
        # RotaryPositionalEmbeddings creates [freqs, freqs] -> [H, H]
        cos_h, sin_h = self._lookup(self.height_rope, height_ids)
        cos_w, sin_w = self._lookup(self.width_rope, width_ids)

        # [FIX 1] Alignment Logic: Interleave H and W
        # Origin SigLIP Logic: [H_freqs, W_freqs] -> repeat(1, 2) -> [H, W, H, W]
        # Muse Current Logic: cos_h is [H, H], cos_w is [W, W]
        
        # Step 1: Split the duplicated halves
        h1, h2 = cos_h.chunk(2, dim=-1)
        w1, w2 = cos_w.chunk(2, dim=-1)
        
        sh1, sh2 = sin_h.chunk(2, dim=-1)
        sw1, sw2 = sin_w.chunk(2, dim=-1)
        
        # Step 2: Re-assemble to match Origin [H, W, H, W]
        # Note: We do NOT cast to x.dtype here yet, we want to keep FP32 if possible
        cos = torch.cat([h1, w1, h2, w2], dim=-1)
        sin = torch.cat([sh1, sw1, sh2, sw2], dim=-1)
        
        # [FIX 2] Precision Alignment: Force FP32 calculation
        # Origin calculates: apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
        # This eliminates the 5e-2 error caused by BF16 accumulation
        
        x_float = x.float()
        cos_float = cos.to(device=x.device, dtype=torch.float32)
        sin_float = sin.to(device=x.device, dtype=torch.float32)
        
        # Apply rotation in FP32
        x_out = (x_float * cos_float) + (self._rotate_half(x_float) * sin_float)
        
        # Cast back to original dtype (e.g. bfloat16)
        return x_out.to(dtype=x.dtype)