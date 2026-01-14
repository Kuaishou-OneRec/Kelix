import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer with support for multiple tokens quantizing the same z_e
    """
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int,
                init_embedding_dim: int,
                n_tokens: int = 1,
                sampling_mode: str = "argmin",
                norm_type: str = 'LayerNorm',
                temperature: float = 1.0,
                temperature_decay: float = 0.999,
                min_temperature: float = 0.1,
                ):
        super(VectorQuantizer, self).__init__()
        
        print(f"[DEBUG] VectorQuantizer.__init__: sampling_mode={sampling_mode}, temperature={temperature}, n_tokens={n_tokens}")
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.sampling_mode = sampling_mode  # "argmin" or "softmax"
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.n_tokens = n_tokens
        
        # Initialize multiple codebooks and projection layers for each token
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings, init_embedding_dim) for _ in range(n_tokens)])
        self.embedding_projs = nn.ModuleList([nn.Linear(init_embedding_dim, embedding_dim) for _ in range(n_tokens)])
        
        # Freeze embeddings initially
        for embedding in self.embeddings:
            for p in embedding.parameters():
                p.requires_grad = False
        
        # Initialize current temperature (not as buffer to avoid loading issues)
        self._current_temperature = temperature
        
        def make_norm():
            # return lambda x: torch.norm(x, p=2, dim=-1)
            if norm_type == 'LayerNorm':
                return nn.LayerNorm(embedding_dim) # lzx norm
            elif norm_type == 'l2':
                return lambda x: torch.norm(x, p=2, dim=-1)
            elif norm_type is None:
                return nn.Identity()
            else:
                raise f"{norm_type} not support."
        
        # Create norms: only one z_norm for the single input, multiple q_norms for each token
        self.z_norm = make_norm()
        self.q_norms = nn.ModuleList([make_norm() for _ in range(n_tokens)])

    def train_code_book(self):
        print(f"train code book embeddings for {self.n_tokens} tokens.")
        for embedding in self.embeddings:
            for p in embedding.parameters():
                p.requires_grad = True
                
    @property
    def current_temperature(self):
        """Get current temperature as a tensor on the correct device"""
        device = self.embeddings[0].weight.device
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
        print("Trainingxxxxxxxxx Softmax sampling")
        indices = torch.multinomial(probs, num_samples=1).squeeze(1)
            
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
            result(dict): containing z_q (sum of all quantized tokens), losses, and indices
        """
        # Compute distances between z_e and all codebook vectors
        # Shape: (batch_size, num_embeddings)
        
        # Normalize z_e once (only one z_norm)
        z_e_norm = self.z_norm(z_e)
        
        # Initialize results
        indices_list = []
        codebook_loss_list = []
        commitment_loss_list = []
        sampling_probs_list = []
        total_e = 0
        
        # Process each token separately, using the same z_e for all
        for i in range(self.n_tokens):
            # Project codebook for current token
            quant_codebook = self.embedding_projs[i](self.embeddings[i].weight)
            quant_codebook = self.q_norms[i](quant_codebook)
            
            # Calculate distances using the same normalized z_e for all tokens
            distances = torch.cdist(z_e_norm, quant_codebook, p=2).pow(2)
            
            # Select indices based on sampling mode
            sampling_probs = None
            if self.sampling_mode == "argmin":
                print(f"[DEBUG] Token {i}: Using argmin selection")
                indices = self._get_indices_argmin(distances)
            elif self.sampling_mode == "softmax":
                print(f"[DEBUG] Token {i}: Using softmax sampling")
                indices, sampling_probs = self._get_indices_softmax(distances)
            else:
                raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")
            
            # Get the selected embeddings for current token
            e_token = quant_codebook[indices]
            
            # Accumulate to total_e (sum of all tokens' embeddings)
            total_e += e_token
            
            # Store indices and probabilities
            indices_list.append(indices)
            if sampling_probs is not None:
                sampling_probs_list.append(sampling_probs)
            
            # Compute mask for loss calculation
            encode_length = z_e_norm.shape[0]
            loss_mask = torch.cat([torch.ones(encode_length-1), torch.zeros(1)]).to(z_e_norm)[:,None]
            
            # Compute losses for each token
            codebook_loss = F.mse_loss(z_e_norm.detach() * loss_mask, e_token * loss_mask)
            commitment_loss = F.mse_loss(z_e_norm * loss_mask, e_token.detach() * loss_mask)
            
            # Add losses to list
            codebook_loss_list.append(codebook_loss)
            commitment_loss_list.append(commitment_loss)
        
        # Combine all indices
        indices_combined = torch.stack(indices_list, dim=1)  # Shape: (batch_size, n_tokens)
        
        # Compute the quantized result as the sum of all tokens
        z_q = z_e_norm + (total_e - z_e_norm).detach()
        
        # Sum all losses
        total_codebook_loss = sum(codebook_loss_list) / self.n_tokens
        total_commitment_loss = sum(commitment_loss_list) / self.n_tokens
        
        result = {
            "z_q": z_q,  # Final quantized result is the sum of all tokens
            "codebook_loss": total_codebook_loss,
            "commitment_loss": total_commitment_loss,
            "indices": indices_combined,
            "indices_list": indices_list,
            "codebook_loss_list": codebook_loss_list,
            "commitment_loss_list": commitment_loss_list
        }
        
        # Add sampling probabilities for softmax mode
        if sampling_probs_list:
            result["sampling_probs_list"] = sampling_probs_list
            
        return result