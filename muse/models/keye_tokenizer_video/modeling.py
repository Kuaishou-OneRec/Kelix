from typing import Dict, Callable, List, Optional, Tuple
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging

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


class VectorQuantizer(Model):
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



def _build_position_ids(
    image_grid_thw: List[Tuple[int, int, int]], device: torch.device
) -> torch.Tensor:
    """根据(t,h,w)构造扁平的position ids (长度=sum(t*h*w))."""
    pos_list = []
    for t, h, w in image_grid_thw:
        numel = int(t * h * w)
        pos_list.append(torch.arange(numel, device=device) % (h * w))
    return torch.cat(pos_list, dim=0)


class KeyeImageTokenizer(Model):
    """使用Keye ViT + VQ 的视觉Tokenizer（无Transformers依赖）。"""

    def __init__(self, config: KeyeTokenizerConfig):
        super().__init__(config)
        self.config: KeyeTokenizerConfig = config
        self.visual = KeyeVisionTransformer(config.vision_config)

        align_in_dim = config.vision_config.hidden_size
        self.pre_llm_align = (
            nn.Linear(align_in_dim, config.llm_hidden_size)
            if config.pre_llm_align or align_in_dim != config.llm_hidden_size
            else nn.Identity()
        )
        proj_out_dim = (
            config.embedding_dim if config.split_dim else config.n_q_tokens * config.embedding_dim
        )
        self.encoder = nn.Linear(config.llm_hidden_size, proj_out_dim)

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

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: List[Tuple[int, int, int]],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: 形状 [b, seq, c, h, w]，当前只支持 b=1。
            image_grid_thw: 每张图的(t,h,w)列表，长度等于seq。
        """
        if pixel_values.dim() != 5:
            raise ValueError(f"pixel_values 维度应为5，实际 {pixel_values.shape}")
        if pixel_values.size(0) != 1:
            raise ValueError("当前实现假定 batch=1，方便与位置编码对齐。")

        device = pixel_values.device
        if position_ids is None:
            position_ids = _build_position_ids(image_grid_thw, device)

        vision_out = self.visual(
            pixel_values=pixel_values,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            interpolate_pos_encoding=True,
            has_learnable_position_embedding=True,
        )
        vision_hidden = vision_out["last_hidden_state"].squeeze(0)  # (seq, dim)
        vision_hidden = self.pre_llm_align(vision_hidden)

        z_e = self.encoder(vision_hidden)
        z_chunks = torch.chunk(z_e, self.config.n_q_tokens, dim=-1)

        vq_outputs = [quant(z) for quant, z in zip(self.quantizer, z_chunks)]
        z_q_list = [o["z_q"] for o in vq_outputs]
        codebook_loss = [o["codebook_loss"] for o in vq_outputs]
        commitment_loss = [o["commitment_loss"] for o in vq_outputs]
        indices = [o["indices"] for o in vq_outputs]

        return {
            "z_q_list": z_q_list,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "indices": indices,
            "vision_hidden": vision_hidden,
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
        self.text_model = Qwen3Model(qwen_config)
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

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        # 文本子模块的初始化交给其自身逻辑，其余使用LeCun。
        if name.startswith("text_model."):
            sub_name = name[len("text_model.") :]
            return self.text_model.get_initializer(sub_name)
        return lecun_normal_

    def get_layers_to_shard(self):
        return self.text_model.get_layers_to_shard()

    def get_checkpointable_module_classes(self):
        return self.text_model.get_checkpointable_module_classes()

    def _project_visual_tokens(self, z_q_list: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(z) for proj, z in zip(self.quant_projector, z_q_list)]
        merged = sum(projected) * self.amplifier
        if self.pool == "avg":
            merged = merged / len(projected)
        return merged

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
        labels: Optional[torch.LongTensor] = None,
        vision_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: 文本token id，形状 [b, s]
            attention_mask: 可选，传递给LLM的mask（布尔或同dtype）。
            pixel_values: 视觉输入，形状 [b, seq, c, h, w]，当前仅支持 b=1。
            image_grid_thw: 对应每个视觉切片的(t,h,w)列表。
            labels: 语言模型标签，计算自回归loss。
            vision_token_mask: 指定哪些位置替换为视觉嵌入，形状同input_ids的bool。
        """
        # 文本嵌入
        inputs_embeds = self.text_model.model.tok_embeddings(input_ids)

        aux_losses: Dict[str, torch.Tensor] = {}
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("使用视觉输入时必须提供 image_grid_thw。")
            vq_out = self.visual_tokenizer(pixel_values, image_grid_thw=image_grid_thw)
            visual_tokens = self._project_visual_tokens(vq_out["z_q_list"])

            if vision_token_mask is None:
                if self.image_token_id < 0:
                    raise ValueError("需提供 vision_token_mask 或设置有效的 image_token_id。")
                vision_token_mask = input_ids == self.image_token_id

            if vision_token_mask.sum() != visual_tokens.size(0):
                raise ValueError(
                    f"视觉token数量({visual_tokens.size(0)})与mask中位置({vision_token_mask.sum().item()})不一致"
                )
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[vision_token_mask] = visual_tokens.to(inputs_embeds)

            # 记录loss
            aux_losses["codebook_loss"] = sum(vq_out["codebook_loss"]) / len(vq_out["codebook_loss"])
            aux_losses["commitment_loss"] = sum(vq_out["commitment_loss"]) / len(
                vq_out["commitment_loss"]
            )
            aux_losses["indices"] = vq_out["indices"]

        logits = self.text_model.model(
            tokens=None, mask=attention_mask, input_embeds=inputs_embeds
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