
from typing import Protocol, Optional, Any
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func


class AttentionFunction(Protocol):
    """Unified attention function interface protocol
    
    All functions implementing attention mechanisms should follow this interface signature.
    """
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor, 
                v: torch.Tensor,
                is_causal: bool = False,
                attn_dropout: float = 0.0,
                **kwargs: Any) -> torch.Tensor:
        """Forward propagation function
        
        Args:
            q: Query tensor, shape: (batch_size, seq_len, d_model) or (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor, same shape as q
            v: Value tensor, same shape as q  
            is_causal: Whether to use causal mask (only see information before current position)
            attn_dropout: Dropout probability for attention weights
            **kwargs: Other optional parameters, such as attention mask, positional encoding, etc.
            
        Returns:
            torch.Tensor: Attention output, same shape as input
        """
        ...


# Example: Concrete implementation conforming to AttentionFunction protocol
class EagerAttention:
    """Standard eager attention implementation, conforming to AttentionFunction protocol"""
    
    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 is_causal: bool = False,
                 attn_dropout: float = 0.0,
                 **kwargs: Any) -> torch.Tensor:
        """Make the class callable, delegates to forward method"""
        return self.forward(q, k, v, is_causal, attn_dropout, **kwargs)
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False,
                attn_dropout: float = 0.0,
                **kwargs: Any) -> torch.Tensor:
        """Implements standard eager attention"""
        # Calculate attention scores
        dim = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)
        
        # Apply causal mask
        if is_causal:
            seq_len = q.size(-2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -float('inf'))
        
        # Apply custom mask (if provided)
        if 'attn_mask' in kwargs and kwargs['attn_mask'] is not None:
            scores = scores.masked_fill(kwargs['attn_mask'], -float('inf'))
            
        # Calculate attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if attn_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=attn_dropout, training=kwargs.get('training', False))
        
        # Calculate output
        output = torch.matmul(attn_weights, v)
        return output


class FlashAttention2:
    """FlashAttention2 implementation, also conforming to AttentionFunction protocol"""
    
    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 is_causal: bool = False,
                 attn_dropout: float = 0.0,
                 **kwargs: Any) -> torch.Tensor:
        """Make the class callable, delegates to forward method"""
        return self.forward(q, k, v, is_causal, attn_dropout, **kwargs)
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False,
                attn_dropout: float = 0.0,
                **kwargs: Any) -> torch.Tensor:
        """Flash Attention implementation (pseudo-code example)"""
        # Here you can call the actual Flash Attention implementation
        # Example: from flash_attn import flash_attn_func
        # return flash_attn_func(q, k, v, dropout_p=attn_dropout, causal=is_causal)

        cu_seqlens = kwargs.get("cu_seqlens")
        cu_seqlens_q, cu_seqlens_k = kwargs.get("cu_seqlens_q"), kwargs.get("cu_seqlens_k")

        if cu_seqlens is not None and cu_seqlens_q is None:
            cu_seqlens_q = cu_seqlens
        if cu_seqlens is not None and cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens
        
        max_seqlen_q = None
        max_seqlen_k = None
        if cu_seqlens_q is not None:
            max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        if cu_seqlens_k is not None:
            max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
            cu_seqlens_k = cu_seqlens_k.to(torch.int32)

        window_size = kwargs.get("window_size", -1)
        if cu_seqlens_q is None or cu_seqlens_k is None:
            attn_output = flash_attn_func(
                q=q.squeeze(0),
                k=k.squeeze(0),
                v=v.squeeze(0),
                dropout_p=attn_dropout,
                window_size=(window_size, window_size),
                causal=is_causal
            )
        else:
            attn_output = flash_attn_varlen_func(
                q=q.squeeze(0),
                k=k.squeeze(0),
                v=v.squeeze(0),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=attn_dropout,
                window_size=(window_size, window_size),
                causal=is_causal
            )
        # TODO: support return_attn_weights
        if attn_output is not None:
            return attn_output.unsqueeze(0)
        return None



# Usage example: Factory function
def get_attention_function(attention_type: str) -> AttentionFunction:
    """Factory function that returns the corresponding attention function based on type"""
    if attention_type == "eager":
        return EagerAttention()
    elif attention_type == "flash_attention_2":
        return FlashAttention2()
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
