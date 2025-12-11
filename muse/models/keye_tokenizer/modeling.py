from typing import Dict, Callable, List, Optional, Tuple
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging
from einops import rearrange

from muse.models.base import Model
from muse.config import  KeyeVisionConfig
from muse.config.model_config import ModelConfig, KeyeTokenizerConfig
from muse.models.keye_vit.modeling import KeyeVisionTransformer

# Import will be done when muse.models is imported, avoiding circular import
# The actual registration happens in __init__.py after import

logger = logging.getLogger(__name__)


def lecun_normal_(tensor: torch.Tensor) -> None:
    """LeCun normal initialization.
    
    LeCun normal initialization: std = sqrt(1 / fan_in)
    This is similar to Kaiming normal but uses fan_in instead of fan_out.
    
    For Linear layers: fan_in = in_features
    For Conv2d layers: fan_in = in_channels * kernel_size[0] * kernel_size[1]
    """
    if tensor.dim() < 2:
        # For 1D tensors (bias, etc.), use a small std
        std = 0.01
    elif tensor.dim() == 2:
        # Linear layer: (out_features, in_features)
        fan_in = tensor.size(1)
        std = math.sqrt(1.0 / fan_in)
    else:
        # Convolutional layer: (out_channels, in_channels, kernel_h, kernel_w, ...)
        # fan_in = in_channels * product of kernel sizes
        fan_in = tensor.size(1)
        for s in tensor.size()[2:]:
            fan_in *= s
        std = math.sqrt(1.0 / fan_in)
    init.normal_(tensor, mean=0.0, std=std)


def default_flax_embed_init(tensor: torch.Tensor) -> None:
    """Default Flax embedding initialization.
    
    Uses normal distribution with std = 1.0
    """
    init.normal_(tensor, mean=0.0, std=1.0)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer with support for both argmin and softmax sampling.
    Note: 不依赖 Model 基类（无 config），避免初始化报错。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_embedding_dim: int,
        sampling_mode: str = "argmin",
        norm_type: str = "LayerNorm",
        temperature: float = 1.0,
        temperature_decay: float = 0.999,
        min_temperature: float = 0.1,
        split_voc: int = 1,  # 分割成几个词表
        split_voc_index: int = 0,
        add_voc_reducer: bool = False,  # 是否添加一个voc reducer
    ):
        super().__init__()
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

class Projector(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 merge_kernel_size: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size

        self.hidden_size = (
            in_channels * self.merge_kernel_size[0] * self.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            self.hidden_size, out_channels, bias=True
        )

    def forward(self,
                image_features: torch.Tensor,
                image_grid_thw: List[Tuple[int, int, int]]) -> torch.Tensor:
        m1, m2 = self.merge_kernel_size

        if isinstance(image_features, (list, tuple)):
            processed_features = list()
            for image_feature, image_grid in zip(image_features, image_grid_thw):
                t, h, w = image_grid
                from einops import rearrange
                image_feature = rearrange(image_feature, "(t h p1 w p2) d -> (t h w) (p1 p2 d)", t=t, h=h // m1, p1=m1, w=w // m2, p2=m2)
                image_feature = self.pre_norm(image_feature)
                hidden_states = self.linear_1(image_feature)
                hidden_states = self.act(hidden_states)
                hidden_states = self.linear_2(hidden_states)
                processed_features.append(hidden_states)
            processed_features = torch.concat(processed_features, dim=0)

            return processed_features
        print("maosiyangdebug:::",image_features.shape)
        #assert image_features.dim() == 2
        dim = image_features.shape[-1]
        hidden_states = self.pre_norm(image_features.view(-1, self.hidden_size))
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states




def _build_position_ids(
    image_grid_thw: List[Tuple[int, int, int]], device: torch.device
) -> torch.Tensor:
    """根据(t,h,w)构造扁平的position ids (长度=sum(t*h*w))."""
    pos_list = []
    for t, h, w in image_grid_thw:
        numel = int(t * h * w)
        pos_list.append(torch.arange(numel, device=device) % (h * w))
    return torch.cat(pos_list, dim=0)


def split_thw(tensor: torch.Tensor) -> torch.Tensor:
    """将 (n,3) 的 thw 展开时间维，得到 [sum(t),3]."""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    repeats = tensor[:, 0]
    new_thw = torch.cat(
        [
            torch.ones(tensor.shape[0], 1, dtype=tensor.dtype, device=tensor.device),
            tensor[:, 1:],
        ],
        dim=1,
    )
    return torch.repeat_interleave(new_thw, repeats, dim=0)


class KeyeImageTokenizer(Model):
    """使用Keye ViT + VQ 的视觉Tokenizer（无Transformers依赖）。"""

    def __init__(self, config: KeyeTokenizerConfig):
        super().__init__(config)
        self.config: KeyeTokenizerConfig = config
        self.n_q_tokens = config.n_q_tokens
        self.visual = KeyeVisionTransformer(config.vision_config)
        self.mlp_AR = Projector(config.vision_config.hidden_size, config.llm_hidden_size)

        self.pre_llm_align = getattr(config, "pre_llm_align", False)
        llm_align_size = getattr(config, "llm_align_size", config.llm_hidden_size)
        align_in_dim = config.llm_hidden_size if self.pre_llm_align else config.llm_hidden_size
        # pre_llm_aligner: only used when pre_llm_align=True; otherwise Identity
        self.pre_llm_aligner = nn.Linear(config.llm_hidden_size, llm_align_size) if self.pre_llm_align else nn.Identity()

        # encoder 输入维度：若 pre_llm_align=True，用对齐后的 llm_align_size；否则直接用 llm_hidden_size
        encoder_in_dim = llm_align_size if self.pre_llm_align else config.llm_hidden_size
        proj_out_dim = config.embedding_dim if config.split_dim else config.n_q_tokens * config.embedding_dim
        self.encoder = nn.Linear(encoder_in_dim, proj_out_dim)

        per_token_dim = (
            config.embedding_dim // config.n_q_tokens if config.split_dim else config.embedding_dim
        )
        self.quantizer: nn.ModuleList = nn.ModuleList(
            [
                VectorQuantizer(
                    num_embeddings=config.codebook_size,
                    embedding_dim=per_token_dim,
                    init_embedding_dim=config.init_embedding_dim,
                    sampling_mode=config.vq_sampling_mode,
                    temperature=config.vq_temperature,
                    temperature_decay=config.vq_temperature_decay,
                    min_temperature=config.vq_min_temperature,
                    split_voc=config.split_voc,
                    split_voc_index=i,
                    add_voc_reducer=config.add_voc_reducer,
                )
                for i in range(config.n_q_tokens)
            ]
        )

    def get_image_embeds(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Get dtype from model parameters
        target_dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.type(target_dtype)
        pixel_values = pixel_values.unsqueeze(0)
        image_grid_thw_split = split_thw(image_grid_thw.squeeze(0))
        siglip_position_ids = []
        image_grid_hws = []
        sample_indices = []
        cu_seqlens = [0]

        for idx, thw in enumerate(image_grid_thw_split):
            thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
            numel = np.prod(thw_tuple)
            image_grid_hws.append(thw_tuple)
            image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
            siglip_position_ids.append(image_position_ids)
            sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
            cu_seqlens.append(cu_seqlens[-1] + numel)

        siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(pixel_values.device)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)
        sample_indices = torch.concat(sample_indices, dim=0).to(pixel_values.device)

        vision_outputs = self.visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_hws,
            position_ids=siglip_position_ids,
            interpolate_pos_encoding=True,
            cu_seqlens=cu_seqlens,
        )
        image_embeds = vision_outputs['last_hidden_state']
        image_embeds = self.mlp_AR(image_embeds, image_grid_hws)
        return image_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: 视觉patches，形状 [num_patches, C, H, W]（与原版一致）。
            image_grid_thw: 形状 [num_images, 3]，每张图的(t,h,w)。
        """
        # 与原版一致：输入是 4D (num_patches, C, H, W)
        # get_image_embeds 内部会 unsqueeze(0) 变成 5D
        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values 维度应为4 (num_patches, C, H, W)，实际 {pixel_values.shape}")

        image_embeds = self.get_image_embeds(pixel_values, image_grid_thw)
        image_embeds = self.pre_llm_aligner(image_embeds)

        z_e = self.encoder(image_embeds).chunk(self.n_q_tokens, dim=-1)

        vq_outputs = [self.quantizer[i](z_e_i) for i, z_e_i in enumerate(z_e)]
        z_q = [v["z_q"] for v in vq_outputs]
        codebook_loss = [v["codebook_loss"] for v in vq_outputs]
        commitment_loss = [v["commitment_loss"] for v in vq_outputs]
        indices = [v["indices"] for v in vq_outputs]

        return {
            "z_q": z_q,
            "z_e": z_e,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "indices": indices,
            "x": image_embeds,
        }

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        # 直接复用LeCun初始化
        return lecun_normal_

    def get_layers_to_shard(self):
        return self.visual.encoder.layers

    def get_checkpointable_module_classes(self):
        return {TransformerSelfAttentionLayer}