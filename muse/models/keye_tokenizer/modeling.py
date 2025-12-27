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
from muse.models.keye_vit.modeling import KeyeVisionModel
from muse.layers.vq import VectorQuantizer
from muse.models.keye_tokenizer._layers import Projector
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




class KeyeImageTokenizer(Model):
    """使用Keye ViT + VQ 的视觉Tokenizer（无Transformers依赖）。"""

    def __init__(self, config: KeyeTokenizerConfig):
        super().__init__(config)
        self.config: KeyeTokenizerConfig = config
        self.n_q_tokens = config.n_q_tokens
        self.visual = KeyeVisionModel(config.vision_config)
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
        self.up_projectors = nn.ModuleList(
            [nn.Linear(config.embedding_dim // config.n_q_tokens \
                if config.split_dim else config.embedding_dim,
                config.output_dim, bias=False) for _ in range(self.n_q_tokens)])


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
        siglip_position_ids = []
        image_grid_hws = []
        sample_indices = []
        cu_seqlens = [0]

        for idx, thw in enumerate(image_grid_thw):
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
        
        # Convert tensor to list of tensors (one per image) to match origin model behavior
        # image_embeds: [1, total_seq, hidden] or [total_seq, hidden] -> list of [seq_i, hidden]
        if image_embeds.dim() == 3:
            image_embeds = image_embeds.squeeze(0)  # [total_seq, hidden]
        
        # Split by cu_seqlens to get list of embeddings
        image_embeds_list = []
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            image_embeds_list.append(image_embeds[start_idx:end_idx])
        
        image_embeds = self.mlp_AR(image_embeds_list, image_grid_hws)
        return image_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: List[Tuple[int, int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: 视觉patches，形状 [num_patches, C, H, W]（与原版一致）。
            image_grid_thw: 形状 [num_images, 3]，每张图的(t,h,w)。
        """
        # 与原版一致：输入是 4D (num_patches, C, H, W)
        # get_image_embeds 内部会 unsqueeze(0) 变成 5D
        image_embeds = self.get_image_embeds(pixel_values, image_grid_thw)
        image_embeds = self.pre_llm_aligner(image_embeds)

        z_e = self.encoder(image_embeds).chunk(self.n_q_tokens, dim=-1)

        vq_outputs = [self.quantizer[i](z_e_i) for i, z_e_i in enumerate(z_e)]
        z_q = [v["z_q"] for v in vq_outputs]
        codebook_loss = [v["codebook_loss"] for v in vq_outputs]
        commitment_loss = [v["commitment_loss"] for v in vq_outputs]
        indices = [v["indices"] for v in vq_outputs]
        token_embeds = torch.stack([self.up_projectors[i](x_i) \
            for i, x_i in enumerate(z_q)], dim=1)

        if self.config.fusion_type == "mean":
            token_embeds = torch.mean(token_embeds, dim=1)
        elif self.config.fusion_type == "sum":
            token_embeds = torch.sum(token_embeds, dim=1)
        else:
            raise ValueError(f"Invalid embedding reduction: {embedding_reduction}")

        return {
            "z_q": z_q,
            "z_e": z_e,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "indices": indices,
            "x": image_embeds,
            "token_embeds": token_embeds,
        }
    
    def forward_indices(self, indices):
        z_q = [self.quantizer[i].lookup(part_indices) for i, part_indices in enumerate(indices)]

        token_embeds = torch.stack(
            [
                self.up_projectors[i](x_i)
                for i, x_i in enumerate(z_q)
            ],
            dim=1
        )

        if self.config.fusion_type == "mean":
            token_embeds = torch.mean(token_embeds, dim=1)
        elif self.config.fusion_type == "sum":
            token_embeds = torch.sum(token_embeds, dim=1)
        else:
            raise ValueError(f"Invalid embedding reduction: {embedding_reduction}")

        return token_embeds

    @torch.inference_mode()
    def tokenize(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_pad_to: int = None,
        return_attention_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Tokenize images with dynamic padding and attention mask.
        
        Args:
            pixel_values: Pixel values tensor [num_total_patches, ...]
            image_grid_thw: Grid info tensor [B, 3] where each row is (t, h, w)
            max_pad_to: Pad to this sequence length
        
        Returns:
            fused_embeddings: [B, max_seq_len, embed_dim]
            attention_mask: [B, 1, 1, max_seq_len]
            max_seq_len: The actual padded sequence length (min of max_in_batch and max_condition_length)
        """
        # Get token embeddings from forward pass
        embeddings: torch.Tensor = self.forward(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )["token_embeds"]

        if not max_pad_to:
            return embeddings
        
        _, embed_dim = embeddings.shape
        
        # Compute merge_length from Projector's merge_kernel_size
        merge_length = self.mlp_AR.merge_kernel_size[0] * self.mlp_AR.merge_kernel_size[1]
        
        # Split by image_grid_thw and pad to max_seq_len
        # image_grid_thw: [B, 3] where each row is (t, h, w)
        lengths = (image_grid_thw[:, 1] * image_grid_thw[:, 2] // merge_length).tolist()
        batch_size = len(lengths)
        
        # Compute dynamic padding length: min of max_in_batch and max_condition_length
        max_seq_len = min(max(lengths), max_pad_to)
        
        # Split embeddings according to lengths
        split_embeddings = torch.split(embeddings, lengths, dim=0)
        
        # Pad each to max_seq_len and stack
        padded = []
        for emb in split_embeddings:
            seq_len = emb.shape[0]
            if seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, embed_dim, 
                                        device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, padding], dim=0)
            else:
                emb = emb[:max_seq_len]
            padded.append(emb)
        
        embeddings = torch.stack(padded, dim=0)  # [B, max_seq_len, embed_dim]
        if not return_attention_mask:
            return embeddings, max_seq_len

        # Create attention mask based on actual lengths
        attention_mask = torch.zeros(batch_size, max_seq_len, device=embeddings.device)
        for i, length in enumerate(lengths):
            attention_mask[i, :min(length, max_seq_len)] = 1
        attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, max_seq_len]

        return embeddings, attention_mask, max_seq_len
    
    @torch.inference_mode()
    def tokenize_indices(
        self,
        indices: torch.LongTensor,
        max_pad_to: int = None,
        return_attention_mask: bool = True,
        lengths: Optional[list[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Tokenize images with dynamic padding and attention mask.
        
        Args:
            pixel_values: Pixel values tensor [num_total_patches, ...]
            image_grid_thw: Grid info tensor [B, 3] where each row is (t, h, w)
            max_pad_to: Pad to this sequence length
        
        Returns:
            fused_embeddings: [B, max_seq_len, embed_dim]
            attention_mask: [B, 1, 1, max_seq_len]
            max_seq_len: The actual padded sequence length (min of max_in_batch and max_condition_length)
        """
        # Get token embeddings from forward pass
        embeddings: torch.Tensor = self.forward_indices(indices.unbind(-1))

        if not max_pad_to:
            return embeddings
        
        _, embed_dim = embeddings.shape
        
        # Compute merge_length from Projector's merge_kernel_size
        merge_length = self.mlp_AR.merge_kernel_size[0] * self.mlp_AR.merge_kernel_size[1]
        
        if lengths is None:
            # Split by image_grid_thw and pad to max_seq_len
            # image_grid_thw: [B, 3] where each row is (t, h, w)
            lengths = (image_grid_thw[:, 1] * image_grid_thw[:, 2] // merge_length).tolist()
        else:
            pass

        batch_size = len(lengths)
        
        # Compute dynamic padding length: min of max_in_batch and max_condition_length
        max_seq_len = min(max(lengths), max_pad_to)
        
        # Split embeddings according to lengths
        split_embeddings = torch.split(embeddings, lengths, dim=0)
        
        # Pad each to max_seq_len and stack
        padded = []
        for emb in split_embeddings:
            seq_len = emb.shape[0]
            if seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, embed_dim, 
                                        device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, padding], dim=0)
            else:
                emb = emb[:max_seq_len]
            padded.append(emb)
        
        embeddings = torch.stack(padded, dim=0)  # [B, max_seq_len, embed_dim]
        if not return_attention_mask:
            return embeddings, max_seq_len

        # Create attention mask based on actual lengths
        attention_mask = torch.zeros(batch_size, max_seq_len, device=embeddings.device)
        for i, length in enumerate(lengths):
            attention_mask[i, :min(length, max_seq_len)] = 1
        attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, max_seq_len]

        return embeddings, attention_mask, max_seq_len

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        # 直接复用LeCun初始化
        return lecun_normal_

    def get_layers_to_shard(self):
        return self.visual.encoder.layers

    def get_checkpointable_module_classes(self):
        return self.visual.get_checkpointable_module_classes()