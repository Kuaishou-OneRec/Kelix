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
        
        print(f"[DEBUG] VectorQuantizer.__init__: sampling_mode={sampling_mode}, temperature={temperature}")
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.sampling_mode = sampling_mode  # "argmin" or "softmax"
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        
        # Initialize the codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        
        # Initialize current temperature (not as buffer to avoid loading issues)
        self._current_temperature = temperature
    
    @property
    def current_temperature(self):
        """Get current temperature as a tensor on the correct device"""
        device = self.embedding.weight.device
        return torch.tensor(self._current_temperature, device=device, dtype=torch.float32)
        
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
        # if self.training:
        print("Trainingxxxxxxxxx Softmax sampling")
        # During training, use sampling for diversity
        # Note: self.training is automatically set by parent model's .train()/.eval()
        indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        # else:
        #     # During inference, use deterministic selection (argmax)
        #     # Note: self.training is automatically set by parent model's .train()/.eval()
        #     indices = torch.argmax(probs, dim=1)
            
        return indices, probs
    
    def update_temperature(self):
        """Update temperature with decay (call this once per training step)"""
        # Always update temperature when called (we only call this during training)
        # Removed self.training check as it may not be reliable in FSDP environment
        new_temp = max(
            self._current_temperature * self.temperature_decay, 
            self.min_temperature
        )
        self._current_temperature = new_temp
    
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
        print(f"[DEBUG] VectorQuantizer sampling_mode: {self.sampling_mode}")
        if self.sampling_mode == "argmin":
            print("[DEBUG] Using argmin selection")
            indices = self._get_indices_argmin(distances)
        elif self.sampling_mode == "softmax":
            print("[DEBUG] Using softmax sampling")
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
