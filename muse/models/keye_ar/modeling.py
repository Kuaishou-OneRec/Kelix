from typing import Dict, Callable, Union, List, Optional, Tuple, Any
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging
from einops import rearrange

from muse.layers.transformer import TransformerDecoder, TransformerSelfAttentionLayer
from muse.models.base import Model
from muse.config import Qwen3Config, KeyeVisionConfig, UnifiedQwen3Config
from muse.config.model_config import ModelConfig, KeyeTokenizerConfig, UnifiedTokenDecoderConfig, KeyeARConfig
from muse.models.qwen3.modeling import Qwen3Model
from muse.models.keye_tokenizer.modeling import KeyeImageTokenizer
from .unified_token_decoder import UnifiedTokenDecoder

# Import will be done when muse.models is imported, avoiding circular import
# The actual registration happens in __init__.py after import

logger = logging.getLogger(__name__)


class UnifiedTokenEmbedding(nn.Module):
    def __init__(self, vocab_size, codebook_size, pad_token_id, hidden_size, n_q_tokens, q_eos_token, image_token_id, pre_embedding_size=None, pre_embedding_tokens=None):
        super().__init__()  # 添加这行确保正确初始化
        self.pre_embedding_size = pre_embedding_size
        self.pre_embedding_tokens = pre_embedding_tokens
        self.padding_idx = pad_token_id
        self.n_q_tokens = n_q_tokens
        self.q_eos_token = q_eos_token
        self.image_token_id = image_token_id
        self.vocab_size = vocab_size  # 添加这行保存vocab_size属性
        
        # 如果启用了new_table且没有pre_embedding_size，则扩展词汇表
        if self.pre_embedding_size is None:
            # 扩展嵌入层以支持视觉token
            old_vocab_size = vocab_size
            new_vocab_size = old_vocab_size + codebook_size
            # 创建新的嵌入层
            new_embed_tokens = nn.Embedding(new_vocab_size, hidden_size, self.padding_idx)
            # 替换嵌入层
            self.embed_tokens = new_embed_tokens
        
        # 如果有pre_embedding_size，创建相应的层
        if self.pre_embedding_size is not None:
            self.pre_embedding = nn.Embedding(self.pre_embedding_tokens, self.pre_embedding_size)
            self.pre_embedding_linear = nn.Linear(self.pre_embedding_size, hidden_size)

    @classmethod
    def convert_hf_state_dict(cls, hf_state_dict: Dict[str, torch.Tensor], 
                             **kwargs) -> Dict[str, torch.Tensor]:
        converted_state_dict = hf_state_dict
        return converted_state_dict

    def _embedding_aggregation(self, extended_tokens, embeddings):
        # 获取q_eos token id
        q_eos_token_id = self.q_eos_token

        # 生成eos位置掩码和前缀有效掩码
        eos_mask = (extended_tokens == q_eos_token_id)
        eos_cumsum = eos_mask.cumsum(dim=2)
        valid_mask = (eos_cumsum == 0)
        
        # 应用掩码并聚合
        valid_mask_expanded = valid_mask.unsqueeze(-1).expand(embeddings.shape)
        masked_embeddings = embeddings * valid_mask_expanded.to(embeddings.dtype)
        aggregated_embeddings = masked_embeddings.sum(dim=2)
        return aggregated_embeddings

    def forward(self, extended_tokens, aggregation=True):
        """
        对query tokens的embedding进行聚合：仅对q_eos_token之前的token求和，支持image_ids处理
        
        Args:
            extended_tokens: 输入张量，shape=[batch_size, length, n_q_tokens+1]
            aggregation: 是否进行聚合，默认为True，但是如果是构造小transformer的输入，则需要设置为False
        
        Returns:
            aggregated_embeddings: 聚合后的embedding，shape=[batch_size, length, hidden_size]
        """

        # 1. 获取混合token（普通+image）的embedding
        embeddings = self._get_token_embeddings(extended_tokens)
        if aggregation:
            aggregated_embeddings = self._embedding_aggregation(extended_tokens, embeddings)
            return aggregated_embeddings
        else:
            return embeddings

    def _get_token_embeddings(self, extended_tokens, group_size=None):
        """
        input extended_tokens: batchsize x seqlen x (n_q_tokens + 1)
        output token_inputs_embeds: batchsize x seqlen x (n_q_tokens + 1) x dim
        """
        if group_size is None:
            group_size = self.n_q_tokens + 1

        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        input_ids_reshaped = extended_tokens

        # 2. 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.vocab_size)

        first_token[(first_token>=self.vocab_size) | (first_token<0)] = 0 # 把vision tokens 置零
        text_embeds = self.embed_tokens(first_token)
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 这里的 0 是为了安全计算，这些计算结果最后会被 mask 掉
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))

        if self.pre_embedding_size is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input[(vis_emb_input >= self.pre_embedding_tokens) | (vis_emb_input<0)] = 0 #  把text tokens 置零
            stage1_embeds = self.pre_embedding(vis_emb_input).detach()
            stage1_embeds = self.pre_embedding_linear(stage1_embeds)
            visual_embeds_final = stage1_embeds
        else:
            stage2_embeds = self.embed_tokens(safe_visual_indices)
            visual_embeds_final = stage2_embeds

        mask_final = is_visual_group.unsqueeze(-1).expand_as(text_embeds)

        text_embeds = text_embeds[:,:,None]
        if group_size > 1:
            text_embeds = text_embeds.repeat_interleave(group_size - 1, dim=2)

        token_inputs_embeds = torch.where(mask_final[:, :, None, :], visual_embeds_final, text_embeds)
        return token_inputs_embeds


class UnifiedTransformerDecoder(TransformerDecoder):
    def __init__(self, *args, token_head, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_head = token_head
    
    def forward(
        self,
        tokens: Optional[torch.Tensor],
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            tokens (Optional[torch.Tensor]): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.
            input_embeds (Optional[torch.Tensor]): Pass these instead of tokens to short-circuit token embeddings
                and skip straight to the transformer layers. Shape ``[b x s x d]``. Default: None
            **kwargs: Additional arguments to pass to transformer layers and attention. Common kwargs include:
                - cu_seqlens (torch.Tensor): cumulative sequence lengths for packed sequences
                - window_size (int): sliding window size for local attention

        Returns:
            Union[torch.Tensor, list[torch.Tensor]]: output tensor with shape ``[b x s x v]`` if `self.skip_output_layer=False`
            and ``[b x s x d]`` otherwise, or a list of layer output tensors defined by ``output_hidden_states`` with the
            final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """

        self._validate_inputs(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
            input_embeds=input_embeds,
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens) if input_embeds is None else input_embeds

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
                **kwargs,
            )

        if len(self.layers) in self.output_hidden_states:
            hidden.append(h)

        token_inputs_embeds = self.tok_embeddings(tokens, aggregation=False)
        next_token_inputs_embeds = torch.roll(token_inputs_embeds, shifts=-1, dims=1)
        h = torch.cat([h[:,:,None], next_token_inputs_embeds], dim=2).to(h)
        h = h.reshape(-1, self.n_q_tokens + 1, h.size(-1))

        # shape: [b, seq_len, out_dim]
        output = self.unembed(h)

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]
        return output

class UnifiedQwen3Model(Qwen3Model):
    """
    UnifiedQwen3Model类，继承自Qwen3Model，支持input_image_ids处理
    """
    
    def __init__(self, qwen_config: UnifiedQwen3Config, token_decoder_config: UnifiedTokenDecoderConfig):
        """
        初始化UnifiedQwen3Model
        
        Args:
            qwen_config: Qwen3配置对象
            token_decoder_config: Token解码器配置对象
        """
        # 调用父类初始化
        super().__init__(qwen_config)
        
        # 正确设置padding_idx和pre_embedding相关属性
        self.padding_idx = qwen_config.pad_token_id
        tok_embeddings = UnifiedTokenEmbedding(
            vocab_size=qwen_config.vocab_size,
            hidden_size=qwen_config.embed_dim,
            pre_embedding_size=qwen_config.pre_embedding_size,
            pre_embedding_tokens=qwen_config.pre_embedding_tokens,
            codebook_size=qwen_config.codebook_size,
            n_q_tokens=qwen_config.n_q_tokens,
            q_eos_token=qwen_config.q_eos_token,
        )
        token_head = UnifiedTokenDecoder(
            config=token_decoder_config,
            token_embedding=None,             # 不使用外部token_embedding
            lm_head=None,  # 训练的时候，不使用外部lm_head
            infer_id_embs_fn=None             # 训练的时候，不使用外部id_embs_fn
        )
        self.model = UnifiedTransformerDecoder(
            tok_embeddings=tok_embeddings,
            layers=self.model.layers,
            max_seq_len=qwen_config.max_seq_len,
            num_heads=self.model.num_heads,
            head_dim=self.model.head_dim,
            norm=self.model.norm,
            output=self.model.output,
            token_head=token_head
        )


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_image_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        前向传播函数，支持input_image_ids处理
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            position_ids: 位置IDs
            past_key_values: 过去的key/value缓存
            inputs_embeds: 输入嵌入
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式
            input_image_ids: 输入图像token IDs
            cache_position: 缓存位置
        """
        if input_ids.size(-1) == 1:
            input_ids = self.model.tok_embeddings.expand_input_ids(
                input_image_ids=input_image_ids,
                input_ids=input_ids,
            )

        # 调用父类的forward方法获取基本功能
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )
        return outputs

    @classmethod
    def convert_hf_state_dict(cls,
                              hf_state_dict: Dict[str, torch.Tensor],
                              tie_word_embeddings: bool = True,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to Muse model state dictionary.
        
        This implementation reuses the Qwen3Model's convert_hf_state_dict logic for the main model
        and UnifiedTokenDecoder's convert_hf_state_dict logic for the token head.
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            tie_word_embeddings: Whether the model ties embeddings (skip lm_head if True).
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
        # First, use Qwen3Model's convert_hf_state_dict for the main model components
        converted_state_dict = super().convert_hf_state_dict(
            hf_state_dict, 
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        # Then, handle the token_head components using UnifiedTokenDecoder's convert_hf_state_dict
        # Extract token_head related keys from hf_state_dict
        token_head_state_dict = {}
        token_head_prefix = "model.token_head."
        
        for hf_key, tensor in hf_state_dict.items():
            # Check if this key belongs to the token_head
            if hf_key.startswith(token_head_prefix):
                # Remove the prefix to get the relative key for UnifiedTokenDecoder
                relative_key = hf_key[len(token_head_prefix):]
                token_head_state_dict[relative_key] = tensor
        
        # If we have token_head keys, convert them using UnifiedTokenDecoder's logic
        if token_head_state_dict:
            # Use UnifiedTokenDecoder's convert_hf_state_dict method
            converted_token_head_state_dict = UnifiedTokenDecoder.convert_hf_state_dict(
                token_head_state_dict, 
                reduce_mode=True  # Assuming reduce=True for the token_head
            )
            
            # Add the converted token_head keys back with the proper prefix
            for key, tensor in converted_token_head_state_dict.items():
                converted_state_dict[f"model.token_head.{key}"] = tensor
        
        return converted_state_dict


class KeyeARModel(Model):
    """
    KeyeAR模型实现，基于Qwen3架构和视觉tokenizer
    """
    
    def __init__(self, config: KeyeARConfig):
        # vision_config, qwen3_config, 
        super().__init__(config)
        self.config = config

        qwen_config = config.qwen_config
        tokenizer_config = config.tokenizer_config
        token_decoder_config = config.token_decoder_config
        
        # 视觉相关组件
        self.visual_tokenizer = KeyeImageTokenizer(tokenizer_config)
        
        # 量化投影器
        original_hidden_size = qwen_config.embed_dim
        
        # 主语言模型
        self.model = UnifiedQwen3Model(qwen_config=qwen_config, token_decoder_config=token_decoder_config)
        
        # 配置参数
        self.vocab_size = qwen_config.vocab_size
        
        # 位置相关
        self.rope_deltas = None
        
        # LM头
        lm_head_size = qwen_config.vocab_size + tokenizer_config.codebook_size 
        self.lm_head = nn.Linear(qwen_config.embed_dim, lm_head_size, bias=False)
    @classmethod
    def convert_hf_state_dict(cls,
                              hf_state_dict: Dict[str, torch.Tensor],
                              tie_word_embeddings: bool = True,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to KeyeARModel state dictionary.
        
        This implementation reuses the UnifiedQwen3Model's convert_hf_state_dict logic for the main model
        and adds handling for the lm_head parameter.
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            tie_word_embeddings: Whether the model ties embeddings (skip lm_head if True).
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
        # First, use UnifiedQwen3Model's convert_hf_state_dict for the main model components
        converted_state_dict = super().convert_hf_state_dict(
            hf_state_dict, 
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        # Handle the lm_head parameter
        for hf_key, tensor in hf_state_dict.items():
            # Handle lm_head weight
            if hf_key == "lm_head.weight":
                converted_key = "lm_head.weight"
                converted_state_dict[converted_key] = tensor
                continue
        
        return converted_state_dict

    def expand_with_image_tokens(
        self,
        input_image_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        拓展input_ids矩阵，将image_token_id对应的行替换为input_image_ids和eos_token
        
        参数说明：
            input_image_ids: 图像索引矩阵，维度为 (im_len, n_q_tokens)
            input_ids: 原始输入ID矩阵，维度为 (len, 1)
            padded_token: 填充标记的整数ID
            image_token_id: 用于标识需要替换为图像tokens的特殊标记ID
        
        返回值：
            expanded_ids: 拓展后的矩阵，维度为 (len, 1 + n_q_tokens)
        """
        # 校验输入维度
        assert input_ids.dim() == 2 and input_ids.size(1) == 1, \
            f"input_ids必须是 (len, 1) 维度，当前为 {input_ids.shape}"
        assert input_image_ids.dim() == 2, \
            f"input_image_ids必须是 2D 矩阵，当前为 {input_image_ids.shape}"
        
        assert input_ids.size(-1) == 1, \
            f"拓展之前的input_ids的列数必须为1，当前为 {input_ids.size(1)}"

        len_seq = input_ids.size(0)  # 序列长度
        output_dim = 1 + self.config.n_q_tokens  # 输出矩阵的列数
        
        # 1. 初始化输出矩阵，所有位置先填充padded_token
        expanded_ids = torch.full(
            size=(len_seq, output_dim),
            fill_value=self.config.q_eos_token,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        
        # 2. 找到input_ids中等于image_token_id的行索引
        image_token_mask = (input_ids.squeeze(1) == self.config.image_token_id)  # (len,)
        image_token_indices = torch.nonzero(image_token_mask, as_tuple=True)[0]  # 满足条件的行索引
        
        # 校验：input_image_ids的行数必须等于image_token的数量
        assert input_image_ids.size(0) == len(image_token_indices), \
            f"input_image_ids的行数 ({input_image_ids.size(0)}) 必须等于input_ids中image_token_id的数量 ({len(image_token_indices)})"
        
        # 3. 处理非image_token的行
        # 第一列填充input_ids的值，第二列填充self.q_eos_token，其余保持padded_token
        non_image_mask = ~image_token_mask  # (len,)
        expanded_ids[non_image_mask, 0] = input_ids[non_image_mask, 0]  # 第一列：原始input_id
        if output_dim > 1:  # 确保至少有第二列
            expanded_ids[non_image_mask, 1] = self.config.q_eos_token  # self.q_eos_token
        
        # 4. 处理image_token的行
        # 前n_q_tokens列填充input_image_ids，最后一列填充eself.q_eos_token
        if len(image_token_indices) > 0:  # 只有存在image_token时才处理
            # 前n_q_tokens列：填充对应位置的input_image_ids
            expanded_ids[image_token_indices, :self.config.n_q_tokens] = input_image_ids
            # 最后一列：填充self.q_eos_token
            expanded_ids[image_token_indices, -1] = self.config.q_eos_token
        
        return expanded_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ):                
        if pixel_values is not None:
            with torch.no_grad():
                vq_out = self.visual_tokenizer(pixel_values, image_grid_thw)
                aligned_indices = torch.stack([x_i for x_i in vq_out['indices']], 0).T
                aligned_indices = self.vocab_size + aligned_indices + torch.arange(self.config.vision_config.n_q_tokens).\
                    to(input_ids)[None] * self.config.vision_config.codebook_size // self.config.vision_config.n_q_tokens
        else:
            aligned_indices = torch.zeros(0, self.config.n_q_tokens).to(input_ids)
        input_ids = self.expand_with_image_tokens(aligned_indices, input_ids)
        assert position_ids.ndim == 2, "position_ids must be 2D"
        # 调用Qwen3Model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        return outputs