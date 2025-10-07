import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer
    """
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        # Initialize the codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        
    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e(torch.Tensor): (batch_size, embedding_dim), the encoded features
        Returns:
            z_q(torch.Tensor): (batch_size, embedding_dim), the quantized features
            codebook_loss(torch.Tensor): (float), the codebook loss, push codebook embedding e close to z_e
            commitment_loss(torch.Tensor): (float), the commitment loss, push z_e close to e
            indices(torch.Tensor): (batch_size, 1), the indices of the quantized features
        """
        # TODO: add normalization to stabilize the training
        distances = torch.cdist(z_e, self.embedding.weight, p=2)
        indices = torch.argmin(distances, dim=1)
        e = self.embedding(indices)

        # codebook loss, push codebook embedding e close to z_e
        codebook_loss = F.mse_loss(z_e.detach(), e)
        # commitment loss, push z_e close to e
        commitment_loss = F.mse_loss(z_e, e.detach())
        
        # Straight-through gradient, pass the gradient of the quantized features to the encoder
        # WARNING: 如果e和z_e相差比较大，在bf16下会有较大精度损失
        # TODO: 1）使用fp32计算，2）对z_e和e做normalization，确保均值方差一致
        z_q = z_e + (e - z_e).detach()
        # 理解下z_q:
        # z_q其实就是量化后的embedding(e)，但直接使用e输入到decoder，梯度无法回传到encoder
        #  z_e + stop_gradient(e - z_e)，前向时就等于e，
        # 反向时，传导到z_q的梯度会直接接到z_e上，但这里能work的前提是e和z_e没有差太远，所以需要加码本loss
        
        # Compute perplexity
        # a more efficient method: use bincount to directly count frequencies
        # counts = torch.bincount(indices.flatten(), minlength=self.num_embeddings)
        # avg_probs = counts.float() / indices.shape[0]
        # non_zero_probs = avg_probs[avg_probs > 0]
        # perplexity = torch.exp(-torch.sum(non_zero_probs * torch.log(non_zero_probs)))
        
        return {
            "z_q": z_q,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "indices": indices,
        }
