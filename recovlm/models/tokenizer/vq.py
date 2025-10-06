import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer
    """
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int,
                commitment_cost: float = 0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize the codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        
    def forward(self, e: torch.Tensor):
        """
        Args:
            e(torch.Tensor): (batch_size, embedding_dim), the encoded features
        Returns:
            z_q_st(torch.Tensor): (batch_size, embedding_dim), the quantized features
            loss(torch.Tensor): (float), the codebook loss
            perplexity(torch.Tensor): (float), the perplexity, for measuring the diversity of the codebook
        """
        distances = torch.cdist(e, self.embedding.weight, p=2).pow(2)
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        print(indices)
        z_q = self.embedding(indices).squeeze(1)
        print("zq", z_q)
        # Compute the loss
        e_latent_loss = F.mse_loss(z_q.detach(), e)
        q_latent_loss = F.mse_loss(z_q, e.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through gradient, pass the gradient of the quantized features to the encoder
        z_q_st = e + (z_q - e).detach()
        print("e", e)
        print("zq", z_q) 
        print("z_q_st", z_q_st)
        
        # Compute perplexity
        # a more efficient method: use bincount to directly count frequencies
        counts = torch.bincount(indices.flatten(), minlength=self.num_embeddings)
        avg_probs = counts.float() / indices.shape[0]
        non_zero_probs = avg_probs[avg_probs > 0]
        perplexity = torch.exp(-torch.sum(non_zero_probs * torch.log(non_zero_probs)))
        
        return z_q_st, loss, indices, perplexity
