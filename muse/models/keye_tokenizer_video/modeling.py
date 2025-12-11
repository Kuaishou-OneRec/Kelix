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
from muse.config import Qwen3Config, KeyeVisionConfig
from muse.config.model_config import ModelConfig, KeyeTokenizerConfig
from muse.models.qwen3.modeling import Qwen3Model
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
    """
    视觉特征降采样/投影模块，移植自 origin，实现时序/空间合并。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_merge_position: str = "before",
        temporal_merge_mode: str = "avg",
        merge_kernel_size: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.temporal_merge_position = temporal_merge_position
        self.temporal_merge_mode = temporal_merge_mode

        if self.temporal_merge_position not in ["before", "after"]:
            raise ValueError(f"Unsupported temporal_merge_position={self.temporal_merge_position}")
        if self.temporal_merge_position == "before" and self.temporal_merge_mode not in ["avg", "delta"]:
            raise ValueError("temporal_merge_mode must be 'avg' or 'delta' when temporal_merge_position='before'")
        if self.temporal_merge_position == "after" and self.temporal_merge_mode not in ["avg", "delta"]:
            raise ValueError("temporal_merge_mode must be 'avg' or 'delta' when temporal_merge_position='after'")

        self.hidden_size = in_channels * self.merge_kernel_size[0] * self.merge_kernel_size[1]
        self.temporal_delta_norm_before = None
        self.temporal_delta_rnn_before = None
        self.temporal_delta_norm_after = None
        self.temporal_delta_rnn_after = None
        if self.temporal_merge_position == "before" and self.temporal_merge_mode == "delta":
            self.temporal_delta_norm_before = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
            self.temporal_delta_rnn_before = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        if self.temporal_merge_position == "after" and self.temporal_merge_mode == "delta":
            self.temporal_delta_norm_after = torch.nn.LayerNorm(out_channels, eps=1e-05)
            self.temporal_delta_rnn_after = nn.GRU(
                input_size=out_channels,
                hidden_size=out_channels,
                num_layers=1,
                batch_first=True,
            )

        self.pre_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(self.hidden_size, out_channels, bias=True)

    def split(self, last_hidden_state: torch.Tensor, grid_thw: torch.Tensor):
        sample_hidden_state = list()
        lengths = np.prod(grid_thw.cpu().numpy(), axis=1).tolist()
        assert sum(lengths) == last_hidden_state.shape[1]
        start = 0
        for length in lengths:
            end = start + length
            tensor = last_hidden_state[:, start:end, :].squeeze(0)
            sample_hidden_state.append(tensor)
            start = end
        return sample_hidden_state

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.pre_norm(tokens)
        tokens = self.linear_1(tokens)
        tokens = self.act(tokens)
        tokens = self.linear_2(tokens)
        return tokens

    def _temporal_reduce_before(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_merge_mode == "avg":
            return x.mean(dim=0)
        if self.temporal_merge_mode == "delta":
            x = x.permute(1, 0, 2)  # (n, t, hidden)
            if x.size(1) == 1:
                x = x.repeat(1, 2, 1)
            x = self.temporal_delta_norm_before(x)
            _, h_n = self.temporal_delta_rnn_before(x)
            return h_n[-1]
        raise ValueError(f"Unsupported temporal_merge_mode {self.temporal_merge_mode} for before merge.")

    def _temporal_reduce_after(self, x: torch.Tensor, spatial_h: int, spatial_w: int) -> torch.Tensor:
        if self.temporal_merge_mode == "avg":
            return x.mean(dim=0)
        if self.temporal_merge_mode == "delta":
            x = x.permute(1, 0, 2)  # (n, t, hidden)
            if x.size(1) == 1:
                x = x.repeat(1, 2, 1)
            x = self.temporal_delta_norm_after(x)
            _, h_n = self.temporal_delta_rnn_after(x)
            return h_n[-1]

    def _process_sequence(self, image_feature: torch.Tensor, image_grid: Tuple[int, int, int]) -> torch.Tensor:
        t, h, w = [int(x) for x in image_grid]
        if t == 0:
            return image_feature.new_zeros((0, self.linear_2.out_features))
        m1, m2 = self.merge_kernel_size
        spatial_h = max(1, h // m1)
        spatial_w = max(1, w // m2)
        x = rearrange(
            image_feature,
            "(t h p1 w p2) d -> t (h w) (p1 p2 d)",
            t=t,
            h=spatial_h,
            p1=m1,
            w=spatial_w,
            p2=m2,
        )
        if self.temporal_merge_position == "before":
            reduced = self._temporal_reduce_before(x)
            reduced = reduced.reshape(-1, self.hidden_size)
            return self._project_tokens(reduced)
        projected = self._project_tokens(x.reshape(-1, self.hidden_size))
        projected = projected.view(t, spatial_h * spatial_w, -1)
        reduced = self._temporal_reduce_after(projected, spatial_h, spatial_w)
        return reduced.reshape(-1, projected.shape[-1])

    def forward(self, image_features: torch.Tensor, image_grid_thw: List[Tuple[int, int, int]]) -> torch.Tensor:
        image_grid_thw_tensor = torch.tensor(image_grid_thw, device=image_features.device)
        image_features = self.split(image_features, image_grid_thw_tensor)
        if isinstance(image_features, (list, tuple)):
            return torch.cat(
                [
                    self._process_sequence(image_feature, image_grid)
                    for image_feature, image_grid in zip(image_features, image_grid_thw)
                ],
                dim=0,
            )
        outputs = []
        start = 0
        m1, m2 = self.merge_kernel_size
        for grid in image_grid_thw:
            t, h, w = [int(x) for x in grid]
            spatial_h = max(1, h // m1)
            spatial_w = max(1, w // m2)
            num_tokens = t * spatial_h * spatial_w
            sample = image_features[start : start + num_tokens]
            start += num_tokens
            outputs.append(self._process_sequence(sample, (t, h, w)))
        return torch.cat(outputs, dim=0)





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
        image_embeds = self.mlp_AR(image_embeds, image_grid_thw)
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
        return []

    def get_checkpointable_module_classes(self):
        return set()


class KeyeForConditionalGeneration(Model):
    """简单的多模态生成模型：Keye ViT Tokenizer + Qwen3 LLM。"""

    def __init__(
        self,
        qwen_config: Qwen3Config,
        vision_config: KeyeVisionConfig,
        tokenizer_config: Optional[KeyeTokenizerConfig] = None,
        image_token_id: int = -1,
        pool: str = "avg",
        amplifier: float = 1.0,
    ):
        super().__init__(qwen_config)
        self.model = Qwen3Model(qwen_config)
        tokenizer_config = tokenizer_config or KeyeTokenizerConfig(
            vision_config=vision_config, llm_hidden_size=qwen_config.embed_dim
        )
        self.visual_tokenizer = KeyeImageTokenizer(tokenizer_config)
        self.quant_projector = nn.ModuleList(
            [
                nn.Linear(tokenizer_config.embedding_dim, qwen_config.embed_dim, bias=False)
                for _ in range(tokenizer_config.n_q_tokens)
            ]
        )
        self.image_token_id = image_token_id
        self.pool = pool
        self.amplifier = amplifier
        self.vocab_size = qwen_config.vocab_size
        self.lm_head = nn.Linear(qwen_config.embed_dim, qwen_config.vocab_size, bias=False)
        # cache rope deltas (aligned with origin implementation)
        self.rope_deltas: Optional[torch.Tensor] = None

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        # 文本子模块的初始化交给其自身逻辑，其余使用LeCun。
        if name.startswith("model."):
            sub_name = name[len("model.") :]
            return self.model.get_initializer(sub_name)
        return lecun_normal_

    def get_layers_to_shard(self):
        return self.model.get_layers_to_shard()

    def get_checkpointable_module_classes(self):
        return self.model.get_checkpointable_module_classes()

    @classmethod
    def convert_hf_state_dict(cls,
                              hf_state_dict: Dict[str, torch.Tensor],
                              tie_word_embeddings: bool = True,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to Muse model state dictionary.
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            tie_word_embeddings: Whether the model ties embeddings (skip lm_head if True).
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
        converted_state_dict = {}
        skipped_keys = []
        
        for hf_key, tensor in hf_state_dict.items():
            # ============ Visual Tokenizer ============
            if hf_key.startswith("visual_tokenizer."):
                rest_key = hf_key[len("visual_tokenizer."):]
                
                # visual.vision_model.* -> visual_tokenizer.visual.*
                if rest_key.startswith("visual.vision_model."):
                    vision_rest = rest_key[len("visual.vision_model."):]
                    # Convert vision model keys using KeyeVisionTransformer patterns
                    # Skip pooling head (vision_model.head.*) – not used in Muse KeyeVisionTransformer
                    if vision_rest.startswith("head."):
                        skipped_keys.append(hf_key)
                        continue
                    # Handle embeddings
                    if vision_rest.startswith("embeddings."):
                        converted_key = f"visual_tokenizer.visual.{vision_rest}"
                        converted_state_dict[converted_key] = tensor
                        continue
                    
                    # Handle post_layernorm -> ln_post
                    if vision_rest.startswith("post_layernorm."):
                        suffix = vision_rest.replace("post_layernorm.", "")
                        converted_key = f"visual_tokenizer.visual.ln_post.{suffix}"
                        converted_state_dict[converted_key] = tensor
                        continue
                    
                    # Handle encoder layers
                    if vision_rest.startswith("encoder.layers."):
                        parts = vision_rest.split(".", 3)  # ["encoder", "layers", "{i}", "rest"]
                        if len(parts) >= 4:
                            layer_idx = parts[2]
                            layer_rest = parts[3]
                            
                            if layer_rest.startswith("layer_norm1."):
                                suffix = layer_rest.replace("layer_norm1.", "")
                                converted_key = f"visual_tokenizer.visual.encoder.layers.{layer_idx}.sa_norm.{suffix}"
                                converted_state_dict[converted_key] = tensor
                                continue
                            
                            if layer_rest.startswith("layer_norm2."):
                                suffix = layer_rest.replace("layer_norm2.", "")
                                converted_key = f"visual_tokenizer.visual.encoder.layers.{layer_idx}.mlp_norm.{suffix}"
                                converted_state_dict[converted_key] = tensor
                                continue
                            
                            if layer_rest.startswith("self_attn."):
                                attn_key = layer_rest.replace("self_attn.", "attn.")
                                attn_key = attn_key.replace("out_proj.", "output_proj.")
                                converted_key = f"visual_tokenizer.visual.encoder.layers.{layer_idx}.{attn_key}"
                                converted_state_dict[converted_key] = tensor
                                continue
                            
                            if layer_rest.startswith("mlp."):
                                new_rest_key = layer_rest.replace("fc1", "w1").replace("fc2", "w2")
                                converted_key = f"visual_tokenizer.visual.encoder.layers.{layer_idx}.{new_rest_key}"
                                converted_state_dict[converted_key] = tensor
                                continue
                    
                    skipped_keys.append(hf_key)
                    continue
                
                # mlp_AR, pre_llm_aligner, encoder, quantizer - direct mapping
                converted_key = f"visual_tokenizer.{rest_key}"
                converted_state_dict[converted_key] = tensor
                continue
            
            # ============ Quant Projector ============
            if hf_key.startswith("quant_projector."):
                # Direct mapping
                converted_state_dict[hf_key] = tensor
                continue
            
            # ============ LLM Model ============
            # Skip lm_head if tie_word_embeddings is True
            if tie_word_embeddings and hf_key == "lm_head.weight":
                skipped_keys.append(hf_key)
                continue
            
            # Handle embedding layer: model.embed_tokens.* -> model.model.tok_embeddings.*
            if hf_key == "model.embed_tokens.weight":
                converted_key = "model.model.tok_embeddings.weight"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle final norm (RMSNorm uses 'scale' not 'weight')
            if hf_key == "model.norm.weight":
                converted_key = "model.model.norm.scale"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle output layer (lm_head)
            if hf_key == "lm_head.weight":
                converted_key = "model.model.output.weight"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle transformer layers: model.layers.* -> model.model.layers.*
            if hf_key.startswith("model.layers."):
                parts = hf_key.split(".", 3)  # ["model", "layers", "{i}", "rest"]
                if len(parts) < 4:
                    skipped_keys.append(hf_key)
                    continue
                
                layer_idx = parts[2]
                rest_key = parts[3]
                
                # Handle attention weights
                if rest_key.startswith("self_attn."):
                    attn_key = rest_key.replace("self_attn.", "attn.")
                    attn_key = attn_key.replace("o_proj", "output_proj")
                    attn_key = attn_key.replace("q_norm.weight", "q_norm.scale")
                    attn_key = attn_key.replace("k_norm.weight", "k_norm.scale")
                    converted_key = f"model.model.layers.{layer_idx}.{attn_key}"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                # Handle MLP weights: gate_proj->w1, up_proj->w3, down_proj->w2
                if rest_key.startswith("mlp."):
                    mlp_key = rest_key.replace("mlp.", "")
                    if mlp_key == "gate_proj.weight":
                        converted_key = f"model.model.layers.{layer_idx}.mlp.w1.weight"
                    elif mlp_key == "up_proj.weight":
                        converted_key = f"model.model.layers.{layer_idx}.mlp.w3.weight"
                    elif mlp_key == "down_proj.weight":
                        converted_key = f"model.model.layers.{layer_idx}.mlp.w2.weight"
                    else:
                        skipped_keys.append(hf_key)
                        continue
                    converted_state_dict[converted_key] = tensor
                    continue
                
                # Handle layer norms (RMSNorm uses 'scale' not 'weight')
                if rest_key == "input_layernorm.weight":
                    converted_key = f"model.model.layers.{layer_idx}.sa_norm.scale"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                if rest_key == "post_attention_layernorm.weight":
                    converted_key = f"model.model.layers.{layer_idx}.mlp_norm.scale"
                    converted_state_dict[converted_key] = tensor
                    continue
            
            # If key doesn't match any pattern, skip it
            skipped_keys.append(hf_key)
        
        if skipped_keys:
            logger.warning(
                f"Skipped {len(skipped_keys)} keys during conversion. "
                f"First few: {skipped_keys[:10]}"
            )
        
        logger.info(
            f"Converted {len(converted_state_dict)} keys from "
            f"{len(hf_state_dict)} Hugging Face keys"
        )
        
        return converted_state_dict

    def _project_visual_tokens(self, z_q_list: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(z) for proj, z in zip(self.quant_projector, z_q_list)]
        merged = sum(projected) * self.amplifier
        if self.pool == "avg":
            merged = merged / len(projected)
        return merged
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def get_rope_index_slowfast(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        fast_video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        fast_video_token_id = self.config.fast_video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas



    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        fast_pixel_values_videos: Optional[torch.FloatTensor] = None,
        fast_video_grid_thw: Optional[torch.LongTensor] = None,
        vision_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: 文本token id，形状 [b, s]
            attention_mask: 可选，传递给LLM的mask（布尔或同dtype）。
            pixel_values: 视觉输入，形状 [num_patches, C, H, W] 或 [1, num_patches, C, H, W]。
            image_grid_thw: 形状 [num_images, 3]，每张图的(t,h,w)。
            labels: 语言模型标签，计算自回归loss。
            vision_token_mask: 指定哪些位置替换为视觉嵌入，形状同input_ids的bool。
        """
        # 文本嵌入
        inputs_embeds = self.model.model.tok_embeddings(input_ids)

        aux_losses: Dict[str, torch.Tensor] = {}
        if pixel_values is not None:
            vq_out = self.visual_tokenizer(pixel_values, image_grid_thw)
            image_embeds = self._project_visual_tokens(vq_out["z_q"])

            if vision_token_mask is None:
                if self.image_token_id < 0:
                    raise ValueError("需提供 vision_token_mask 或设置有效的 image_token_id。")
                vision_token_mask = input_ids == self.image_token_id

            if vision_token_mask.sum() != image_embeds.size(0):
                raise ValueError(
                    f"视觉token数量({image_embeds.size(0)})与mask中位置({vision_token_mask.sum().item()})不一致"
                )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[vision_token_mask] = image_embeds.to(inputs_embeds)

            # 记录loss
            codebook_loss, commitment_loss, vq_indices = vq_out['codebook_loss'], vq_out['commitment_loss'], vq_out['indices']
            aux_losses["codebook_loss"] = codebook_loss
            aux_losses["commitment_loss"] = commitment_loss
            aux_losses["indices"] = vq_indices



            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]



            if n_image_tokens != n_image_features:
                fast_image_embeds = torch.cat(fast_image_embeds,dim=0)
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, slow features {n_image_features - fast_image_embeds.shape[0]}, fast features {fast_image_embeds.shape[0]}"
                )
            mask = (input_ids == self.config.image_token_id)
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)




        #maosiyang::debug here
        if pixel_values_videos is not None:
            video_beta = 1.0
            vq_out_video = self.visual_tokenizer(pixel_values_videos, video_grid_thw)
            video_embeds = self._project_visual_tokens(vq_out_video["z_q"])

            if vision_token_mask is None:
                if self.image_token_id < 0:
                    raise ValueError("需提供 vision_token_mask 或设置有效的 image_token_id。")
                vision_token_mask = input_ids == self.image_token_id

            if vision_token_mask.sum() != image_embeds.size(0):
                raise ValueError(
                    f"视觉token数量({image_embeds.size(0)})与mask中位置({vision_token_mask.sum().item()})不一致"
                )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[vision_token_mask] = image_embeds.to(inputs_embeds)

            # 记录loss
            codebook_loss, commitment_loss, vq_indices = vq_out['codebook_loss'], vq_out['commitment_loss'], vq_out['indices']
            aux_losses["codebook_loss"] = codebook_loss
            aux_losses["commitment_loss"] = commitment_loss
            aux_losses["indices"] = vq_indices



            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]



             if n_image_tokens != n_image_features:
                fast_image_embeds = torch.cat(fast_image_embeds,dim=0)
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, slow features {n_image_features - fast_image_embeds.shape[0]}, fast features {fast_image_embeds.shape[0]}"
                )
            mask = (input_ids == self.config.image_token_id)
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 计算 3D position_ids（对齐 origin 的 slowfast 逻辑）
        # 当 position_ids 缺省时，使用 get_rope_index_slowfast 生成 3D 位置，并缓存 rope_deltas
        position_ids_3d, rope_deltas = self.get_rope_index_slowfast(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            fast_video_grid_thw=fast_video_grid_thw,
            attention_mask=attention_mask,
        )
        self.rope_deltas = rope_deltas

        # Call through Qwen3Model.forward which delegates to TransformerDecoder
        logits = self.model(
            tokens=None, mask=attention_mask, input_embeds=inputs_embeds, input_pos=position_ids_3d
        )

        # TransformerDecoder可能返回list/张量，这里取最后一个为logits
        if isinstance(logits, list):
            logits = logits[-1]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            **aux_losses,
        }