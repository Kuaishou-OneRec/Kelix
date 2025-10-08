import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer with support for both argmin and softmax sampling
    """
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int,
                sampling_mode: str = "argmin",
                temperature: float = 1.0,
                temperature_decay: float = 0.999,
                min_temperature: float = 0.1):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.sampling_mode = sampling_mode  # "argmin" or "softmax"
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        
        # Initialize the codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        
        # Register temperature as a buffer so it's saved/loaded with the model
        self.register_buffer('current_temperature', torch.tensor(temperature))
        
    def _get_indices_argmin(self, distances):
        """Traditional argmin selection"""
        return torch.argmin(distances, dim=1)
    
    def _get_indices_softmax(self, distances):
        """Softmax sampling with temperature"""
        # Convert distances to similarities (negative distances)
        # Shape: (batch_size, num_embeddings)
        similarities = -distances
        
        # Apply temperature scaling
        logits = similarities / self.current_temperature
        
        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        
        # Sample from the distribution
        if self.training:
            # During training, use sampling for diversity
            indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # During inference, can choose to be deterministic (argmax) or sample
            # Here we use argmax for consistency
            indices = torch.argmax(probs, dim=1)
            
        return indices, probs
    
    def update_temperature(self):
        """Update temperature with decay (call this once per training step)"""
        if self.training:
            new_temp = max(
                self.current_temperature * self.temperature_decay, 
                self.min_temperature
            )
            self.current_temperature.fill_(new_temp)
    
    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e(torch.Tensor): (batch_size, embedding_dim), the encoded features
        Returns:
            z_q(torch.Tensor): (batch_size, embedding_dim), the quantized features
            codebook_loss(torch.Tensor): (float), the codebook loss, push codebook embedding e close to z_e
            commitment_loss(torch.Tensor): (float), the commitment loss, push z_e close to e
            indices(torch.Tensor): (batch_size,), the indices of the quantized features
            sampling_probs(torch.Tensor, optional): (batch_size, num_embeddings), sampling probabilities (only for softmax mode)
        """
        # Compute distances between z_e and all codebook vectors
        # Shape: (batch_size, num_embeddings)
        distances = torch.cdist(z_e, self.embedding.weight, p=2).pow(2)
        
        # Select indices based on sampling mode
        sampling_probs = None
        if self.sampling_mode == "argmin":
            indices = self._get_indices_argmin(distances)
        elif self.sampling_mode == "softmax":
            indices, sampling_probs = self._get_indices_softmax(distances)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")
        
        # Get the selected embeddings
        e = self.embedding(indices)

        # Compute losses
        # codebook loss: push codebook embedding e close to z_e
        codebook_loss = F.mse_loss(z_e.detach(), e)
        # commitment loss: push z_e close to e  
        commitment_loss = F.mse_loss(z_e, e.detach())
        
        # Straight-through gradient estimation
        # Forward: z_q = e (quantized embeddings)
        # Backward: gradients flow through z_e
        z_q = z_e + (e - z_e).detach()
        
        result = {
            "z_q": z_q,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "indices": indices,
        }
        
        # Add sampling probabilities for softmax mode
        if sampling_probs is not None:
            result["sampling_probs"] = sampling_probs
            
        return result
