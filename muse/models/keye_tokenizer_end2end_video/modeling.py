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
from muse.models.keye_tokenizer_end2end_video._layers import Projector
from muse.models.keye_vit.modeling import KeyeVisionModel
from muse.models.keye_tokenizer.modeling import KeyeImageTokenizer
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


class KeyeVideoTokenizer(KeyeImageTokenizer):

    def __init__(self, config: KeyeTokenizerConfig):
        super().__init__(config)
        self.mlp_AR = Projector(config.vision_config.hidden_size, config.llm_hidden_size)
    def get_image_embeds(self, pixel_values: Optional[torch.Tensor] = None, 
                            image_grid_thw: Optional[torch.LongTensor] = None, 
                            **kwargs):


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


class KeyeTokenizerEnd2EndVideo(Model):
    """简单的多模态生成模型：Keye ViT Tokenizer + Qwen3 LLM。"""

    def __init__(
        self,
        qwen_config: Qwen3Config,
        vision_config: KeyeVisionConfig,
        tokenizer_config: Optional[KeyeTokenizerConfig] = None,
        image_token_id: int = -1,
        video_token_id: int = -1,
        pool: str = "avg",
        amplifier: float = 1.0,
    ):
        super().__init__(qwen_config)
        self.model = Qwen3Model(qwen_config)
        self.use_multimodal_rope = qwen_config.use_multimodal_rope
        tokenizer_config = tokenizer_config or KeyeTokenizerConfig(
            vision_config=vision_config, llm_hidden_size=qwen_config.embed_dim
        )
        self.visual_tokenizer = KeyeVideoTokenizer(tokenizer_config)
        self.quant_projector = nn.ModuleList(
            [
                nn.Linear(tokenizer_config.embedding_dim, qwen_config.embed_dim, bias=False)
                for _ in range(tokenizer_config.n_q_tokens)
            ]
        )
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.pool = pool
        self.amplifier = amplifier
        self.vocab_size = qwen_config.vocab_size
        # self.lm_head = nn.Linear(qwen_config.embed_dim, qwen_config.vocab_size, bias=False)
        # cache rope deltas (aligned with origin implementation)
        self.rope_deltas: Optional[torch.Tensor] = None

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        # 文本子模块的初始化交给其自身逻辑，其余使用LeCun。
        if name.startswith("model."):
            sub_name = name[len("model.") :]
            return self.model.get_initializer(sub_name)
        return lecun_normal_

    def get_layers_to_shard(self):
        # Combine layers from both model and visual_tokenizer components
        model_layers = self.model.get_layers_to_shard()
        visual_layers = self.visual_tokenizer.get_layers_to_shard()
        return list(model_layers) + list(visual_layers)

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
        # Output embeddings are inside TransformerDecoder.output
        return self.model.model.output

    def set_output_embeddings(self, new_embeddings):
        # Output embeddings are inside TransformerDecoder.output
        self.model.model.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model



    def generate_positional_id(self,thw):
        """
        将3x1xL的张量转换为1D positional_id矩阵
        
        参数:
            thw: 形状为(3, 1, L)的PyTorch张量
        
        返回:
            positional_id: 形状为(L,)的1D张量，包含连续的序列编号
        """
        # 检查输入形状是否正确
        assert thw.shape[0] == 3 and thw.shape[1] == 1, "输入必须是3x1xL的张量"
        
        # 取出第一位置并flatten
        seq = thw[0, 0, :].flatten()  # 形状变为(L,)
        
        # 识别子序列边界（假设以0为新子序列的开始）
        subsequence_starts = torch.where(seq == 0)[0].tolist()
        L = seq.numel()
        positional_id = torch.zeros_like(seq, dtype=torch.long)
        
        # 处理每个子序列
        for i, start in enumerate(subsequence_starts):
            # 确定当前子序列的结束位置
            if i < len(subsequence_starts) - 1:
                end = subsequence_starts[i + 1]
            else:
                end = L
            
            # 为当前子序列生成连续编号
            subsequence_length = end - start
            positional_id[start:end] = torch.arange(subsequence_length, dtype=torch.long)
        
        return positional_id


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
            # 记录loss
            codebook_loss, commitment_loss, vq_indices = vq_out['codebook_loss'], vq_out['commitment_loss'], vq_out['indices']
            aux_losses["codebook_loss"] = codebook_loss
            aux_losses["commitment_loss"] = commitment_loss
            aux_losses["indices"] = vq_indices



            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                fast_image_embeds = torch.cat(fast_image_embeds,dim=0)
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, slow features {n_image_features - fast_image_embeds.shape[0]}, fast features {fast_image_embeds.shape[0]}"
                )
            mask = (input_ids == self.image_token_id)
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        if pixel_values_videos is not None:      
            print('maosiyangdebugvideo', pixel_values_videos.shape, video_grid_thw)
            video_beta = 1.0  
            vq_out_video = self.visual_tokenizer(pixel_values_videos, video_grid_thw)
            print('maosiyangdebugvq_out_video', vq_out_video)
            video_embeds = self._project_visual_tokens(vq_out_video["z_q"])
            # 记录loss
            video_codebook_loss, video_commitment_loss, video_vq_indices = vq_out_video['codebook_loss'], vq_out_video['commitment_loss'], vq_out_video['indices']
            if "codebook_loss" in aux_losses:
                aux_losses["codebook_loss"] = [
                        img_loss + video_beta * vid_loss
                        for img_loss, vid_loss in zip(aux_losses["codebook_loss"], video_codebook_loss)
                    ]
            else:
                aux_losses["codebook_loss"] = [video_beta * vid_loss for vid_loss in video_codebook_loss]
            if "commitment_loss" in aux_losses:
                aux_losses["commitment_loss"] = [
                        img_loss + video_beta * vid_loss
                        for img_loss, vid_loss in zip(aux_losses["commitment_loss"], video_commitment_loss)
                    ]
            else:
                aux_losses["commitment_loss"] = [video_beta * vid_loss for vid_loss in video_commitment_loss]

            aux_losses["video_indices"] = video_vq_indices
            n_video_tokens = (input_ids == self.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = (input_ids == self.video_token_id)
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)


        #maosiyang: for debug infer
        if position_ids is None:
            position_ids_3d, _ = self.get_rope_index_slowfast(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                fast_video_grid_thw=fast_video_grid_thw,
                attention_mask=attention_mask,
            )
            position_ids = self.generate_positional_id(position_ids_3d).to(position_ids_3d)[None, :] # 1 x l, 这个是用来计算rope的东西
        else:
            raise ValueError("position id wrong!")


        # Call through Qwen3Model.forward which delegates to TransformerDecoder
        logits = self.model(
            tokens=None,
            mask=attention_mask,
            input_embeds=inputs_embeds,
            input_pos=position_ids
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