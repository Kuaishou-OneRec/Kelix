import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .. import models
from ..models import register
from utils import select_representative_embeddings_for_codebook

@register("bottleneck")
class Bottleneck(nn.Module):
    def __init__(
        self,
        bottleneck_dim: int,
        input_dim: int,
        output_dim: int,
        token_nums: int,
        regularizer=None,
        llm_model=None,
        **kwargs
    ):  
        super().__init__()
        self.token_nums = token_nums
        self.input_dim = input_dim
        self.output_dim = output_dim
        if bottleneck_dim > 0:
            self.bottleneck_dim = bottleneck_dim
        else:
            assert self.input_dim == self.output_dim, "input_dim and output_dim must be the same when bottleneck_dim is not specified"
            self.bottleneck_dim = self.input_dim
        
        self.project_dim = self.bottleneck_dim
        
        
        # FIXME: Codebook * W , dimension align !!!
        if self.bottleneck_dim > 0:
            self.in_linear = nn.Linear(self.input_dim, self.project_dim)
            self.out_linear = nn.Linear(self.bottleneck_dim, self.output_dim)
        else:
            self.in_linear = self.out_linear = lambda x: x
    
        
        regularizer['args']['dim'] = self.bottleneck_dim
        regularizer['args']['token_nums'] = self.token_nums
        regularizer['args']['llm_model'] = llm_model
        self.regularizer = models.make(regularizer) # simvq
        self.train(True)

    def project_in(self, x):
        assert len(x.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        z = self.in_linear(x)
        return z

    def project_out(self, z_cat):
        z = self.out_linear(z_cat)
        return z

    def decode(self, bottleneck_rep):
        regularized_z = self.regularizer.decode(bottleneck_rep)
        return self.project_out(regularized_z)

    def forward(self, x):  
        z = self.project_in(x)
        # z = x
        projected_z = z
        regularized_output = self.regularizer(z)
        x_hat = self.project_out(regularized_output['regularized_z']) # quantized
        # x_hat = regularized_output['regularized_z']
        bottleneck_rep = regularized_output.pop('bottleneck_rep') # q_indices
        return {
            'output': x_hat, # quantized
            'bottleneck_rep': bottleneck_rep, # q_indices
            'projected_z': projected_z,
            **regularized_output,
        }


@register("simvq")
class SimVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        llm_model,
        codebook_size,
        l2_normalized=False,
        same_index_shape=True,
        stochastic=False,
        stochastic_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        assert isinstance(l2_normalized, bool)
        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.eval_deterministic = False
        self.default_stochastic_temperature = stochastic_temperature
        
        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        # for clear inference code, we remove the codebook init from LLM's embedding
        # FIXME: CODEBOOK INIT!!!
        with torch.no_grad():
            llm_emb = llm_model.get_input_embeddings().weight  # shape [vocab_size, dim]

            init_emb, selected_indices = select_representative_embeddings_for_codebook(
                llm_embeddings=llm_emb,
                codebook_size=self.codebook_size,
                normalize=True  # 推荐归一化以确保语义覆盖
            )

        # 用选中的向量初始化 nn.Embedding
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        self.embedding.weight.data.copy_(init_emb)
        # 冻结 codebook
        # TODO:
        self.embedding.weight.requires_grad = False
        self.embedding_proj = nn.Linear(self.dim, self.dim)






        self.same_index_shape = same_index_shape
        self.train(True)

    def set_eval_deterministic(self, deterministic=True):
        self.eval_deterministic = deterministic

    def set_stochastic_temperature(self, temperature):
        self.stochastic_temperature_inv = 1 / temperature

    @torch.autocast(device_type='cuda', enabled=False)
    def get_emb(self):
        emb = self.embedding_proj(self.embedding.weight) 
        # emb = self.embedding.weight
        if self.l2_normalized:
            emb = F.normalize(emb, p=2, dim=-1)
        # assert emb.dtype == torch.float32, f"Embedding weight dtype is {emb.dtype}, expected float32"
        return emb

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, z):
        print("self.training in bottleneck: ", self.training) # False???
        self.train(True)
        emb = self.get_emb()


        print("z.requires_grad: before to", z.requires_grad) # 
        z = z.to(emb)
        print("z.requires_grad: afater to ", z.requires_grad) # 
        
        # z = z.to(dtype=emb.dtype, device=emb.device)
        # z = z.float()
        assert len(z.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        if self.l2_normalized:
            z = F.normalize(z, p=2, dim=-1)
        print("z.requires_grad: after l2_normalized", z.requires_grad) # true
        z_flattened = rearrange(z, 'b n d -> (b n) d')
        print("z.requires_grad: after flatten", z_flattened.requires_grad) # true

        if self.stochastic:
            # sample the softmaxed cosine similarity
            assert self.l2_normalized, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, emb) # TODO: sample the softmaxed cosine similarity
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                q_indices = torch.argmax(probs, dim=-1)
            else:
                q_indices = torch.multinomial(probs, 1).squeeze(-1)
        else:
            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(emb**2, dim=1)
                - 2
                * torch.einsum(
                    "bd,dn->bn", z_flattened, rearrange(emb, "n d -> d n")
                )
            )
            q_indices = torch.argmin(d, dim=1)

        quantized = F.embedding(q_indices, emb, self.embedding.padding_idx, self.embedding.max_norm,
            self.embedding.norm_type, self.embedding.scale_grad_by_freq, self.embedding.sparse)
        
        print("quantized.requires_grad:", quantized.requires_grad) # True
        print("z_flattened.requires_grad:", z_flattened.requires_grad) # True
        # FIXME:
        beta = 100
        # print("quantized value: ", quantized)
        # print("z_flattened value: ", z_flattened)
        print("quantized shape: ", quantized.shape)
        print("z_flattened shape: ", z_flattened.shape)
        print("q_indices value: ", q_indices)
        codebook_loss = beta * (torch.mean((quantized.detach() - z_flattened)**2) + torch.mean((quantized - z_flattened.detach())**2)) # (b n) d
        print("codebook_loss.requires_grad:", codebook_loss.requires_grad)

        quantized = quantized.view(z.shape)  # (b, n, d)
        
        # preserve gradients
        # TODO:
        quantized = z + (quantized - z).detach() # True
        print("quantized.requires_grad after preserve gradients:", quantized.requires_grad) # True

        if self.same_index_shape:
            q_indices = q_indices.reshape(quantized.shape[0], quantized.shape[1])

        
        print("quantized mean/std:", quantized.mean().item(), quantized.std().item())
        print("z_flattened mean/std:", z_flattened.mean().item(), z_flattened.std().item())
        print("commitment loss:", torch.mean((quantized.detach() - z_flattened)**2).item())
        print("embedding loss:", torch.mean((quantized - z_flattened.detach())**2).item())
        print("total codebook loss:", codebook_loss.item())
        # 看看 embedding 是否有梯度
        # emb = self.embedding.weight
        print("embedding requires_grad:", self.embedding.weight.requires_grad)
        print("embedding grad:", self.embedding.weight.grad.norm().item() if self.embedding.weight.grad is not None else None)


        return_dict = {
            'unregularized_z': z, # but l2 normalized if l2_normalized=True
            'emb': emb, # but l2 normalized if l2_normalized=True
            'regularized_z': quantized,
            'bottleneck_rep': q_indices,
            'codebook_loss': codebook_loss
        }
        return return_dict
    
    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        indices_shape = indices.shape
        indices_flatten = rearrange(indices, '... -> (...)')

        # get quantized latent vectors
        emb = self.get_emb()
        z_q = F.embedding(indices_flatten, emb)
        # z_q = self.embedding(indices_flatten)
        if self.l2_normalized:
            z_q = F.normalize(z_q, p=2, dim=-1)

        if shape is not None:
            z_q = z_q.reshape(shape)
        else:
            z_q = z_q.reshape([*indices_shape, self.dim])
        return z_q

    def decode(self, indices):
        return self.get_codebook_entry(indices)