
class AttentionInterface(Protocol):
  def forward(self,
              q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              is_causal: bool = False,
              attn_dropout: float = 0.0,
              **kwargs) -> torch.Tensor:
    ...