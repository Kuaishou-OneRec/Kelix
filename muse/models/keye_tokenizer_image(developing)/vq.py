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
                init_embedding_dim: int,
                sampling_mode: str = "argmin",
                norm_type: str = 'LayerNorm',
                temperature: float = 1.0,
                temperature_decay: float = 0.999,
                min_temperature: float = 0.1,
                split_voc=1, # 分割成几个词表
                split_voc_index=0,
                add_voc_reducer=False, # 是否添加一个voc reducer
                
                ):
        super(VectorQuantizer, self).__init__()
        print(f"[DEBUG] VectorQuantizer.__init__: sampling_mode={sampling_mode}, temperature={temperature}, split_voc={split_voc}, add_voc_reducer={add_voc_reducer}, split_voc_index={split_voc_index}")
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.sampling_mode = sampling_mode  # "argmin" or "softmax"
        self.split_voc_index = split_voc_index
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.split_voc = split_voc
        self.add_voc_reducer = add_voc_reducer
        self.split_voc_size = self.num_embeddings // self.split_voc
        
        # Initialize the codebook
        self.embedding = nn.Embedding(num_embeddings, init_embedding_dim)

        for p in self.embedding.parameters():
            p.requires_grad = False

        if add_voc_reducer:
            print(f"add_voc_reducer shape={self.num_embeddings, self.split_voc_size}")
            self.voc_reducer = nn.Parameter(torch.randn(self.num_embeddings, self.split_voc_size) * 0.01)
            self.slice_indices = slice(0, num_embeddings)
        else:
            if split_voc > 1:
                self.slice_indices = slice(num_embeddings // split_voc * self.split_voc_index, num_embeddings // split_voc * (self.split_voc_index + 1))
                print(f"self.slice_indices={self.slice_indices}")
            else:
                self.slice_indices = slice(0, num_embeddings)

        self.embedding_proj = nn.Linear(init_embedding_dim, embedding_dim)

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
        
        self.q_norm = make_norm()
        self.z_norm = make_norm()

    def train_code_book(self):
        print(f"train code book embeddings.")
        for p in self.embedding.parameters():
            p.requires_grad = True
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
        z_e = self.z_norm(z_e)

        if self.add_voc_reducer:
            embedding = self.voc_reducer.T @ self.embedding.weight[self.slice_indices]
        else:
            embedding = self.embedding.weight[self.slice_indices]

        quant_codebook = self.embedding_proj(embedding)
        quant_codebook = self.q_norm(quant_codebook)

        distances = torch.cdist(z_e, quant_codebook, p=2).pow(2)
        
        # Select indices based on sampling mode
        sampling_probs = None
        if self.sampling_mode == "argmin":
            # print("[DEBUG] Using argmin selection")
            indices = self._get_indices_argmin(distances)
        elif self.sampling_mode == "softmax":
            # print("[DEBUG] Using softmax sampling")
            indices, sampling_probs = self._get_indices_softmax(distances)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")
        
        # Get the selected embeddings
        e = quant_codebook[indices]

        encode_length = z_e.shape[0] # torch.Size([6857, 128])
        loss_mask = torch.cat([torch.ones(encode_length-1), torch.zeros(1)]).to(z_e)[:,None]
        # Compute losses
        # codebook loss: push codebook embedding e close to z_e
        # codebook_loss = F.mse_loss(z_e.detach(), e)
        codebook_loss = F.mse_loss(z_e.detach() * loss_mask, e * loss_mask)
        # commitment loss: push z_e close to e  
        commitment_loss = F.mse_loss(z_e * loss_mask, e.detach() * loss_mask)
        
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
