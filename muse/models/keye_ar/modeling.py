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
        if embeddings.size(2) == extended_tokens.size(2) - 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, 1), value=0)

        # 获取q_eos token id
        q_eos_token_id = self.q_eos_token

        # 生成eos位置掩码和前缀有效掩码
        eos_mask = (extended_tokens == q_eos_token_id)
        eos_cumsum = eos_mask.cumsum(dim=2)
        valid_mask = (eos_cumsum == 0)

        # 应用掩码并聚合
        valid_mask_expanded = valid_mask.unsqueeze(-1).expand(embeddings.shape)

        # 
        # masked_embeddings = embeddings.float() * valid_mask_expanded.to(embeddings.dtype)
        # aggregated_embeddings = masked_embeddings.sum(dim=2)
        # aggregated_embeddings = aggregated_embeddings.float().bfloat16()# 跟baseline对齐

        masked_embeddings = embeddings * valid_mask_expanded.to(embeddings)
        aggregated_embeddings = masked_embeddings.sum(dim=2).to(embeddings)
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

        embeddings = self._get_token_embeddings(extended_tokens)
        if aggregation:
            aggregated_embeddings = self._embedding_aggregation(extended_tokens, embeddings)
            torch.save(aggregated_embeddings, "aggregated_embeddings.pt")
            return aggregated_embeddings
        else:
            return embeddings

    def _get_token_embeddings(self, extended_tokens, group_size=None):
        """
        input extended_tokens: batchsize x seqlen x (n_q_tokens + 1)
        output token_inputs_embeds: batchsize x seqlen x (n_q_tokens + 1) x dim
        """
        # 修复点1：对齐reshape逻辑（和SecondClass完全一致）
        if group_size is None:
            group_size = self.n_q_tokens + 1
        
        print(f"extended_tokens={extended_tokens.shape}\n{extended_tokens}")
        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        input_ids_reshaped = extended_tokens

        # 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.vocab_size)

        first_token[(first_token>=self.vocab_size) | (first_token<0)] = 0
        text_embeds = self.embed_tokens(first_token)
        
        # 修复点2：对齐visual indices切片逻辑（和SecondClass完全一致）
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 安全索引处理
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))
        if self.pre_embedding_size is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input[(vis_emb_input >= self.pre_embedding_tokens) | (vis_emb_input<0)] = 0
            stage1_embeds = self.pre_embedding(vis_emb_input).detach()
            stage1_embeds = self.pre_embedding_linear(stage1_embeds)
            visual_embeds_final = stage1_embeds
        else:
            stage2_embeds = self.embed_tokens(safe_visual_indices)
            visual_embeds_final = stage2_embeds

        mask_final = is_visual_group.unsqueeze(-1).expand_as(text_embeds)
        
        # 修复点3：对齐repeat_interleave逻辑（和SecondClass完全一致）
        text_embeds = text_embeds[:,:,None]
        if group_size > 1:
            text_embeds = text_embeds.repeat_interleave(group_size - 1, dim=2)
        
        token_inputs_embeds = torch.where(mask_final[:, :, None, :], visual_embeds_final, text_embeds)

        return token_inputs_embeds
    

class UnifiedTransformerDecoder(TransformerDecoder):
    def __init__(self, *args, token_head: UnifiedTokenDecoder, output_last_hidden_states_only: bool = False, token_decoder_with_teacher_forcing: bool = True, token_head_max_new_tokens: int = 9, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_head = token_head
        self.output_last_hidden_states_only = output_last_hidden_states_only
        self.token_decoder_with_teacher_forcing = token_decoder_with_teacher_forcing
        self.token_head_max_new_tokens = token_head_max_new_tokens
    
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
        
        if self.output_last_hidden_states_only:
            return h

        if len(self.layers) in self.output_hidden_states:
            hidden.append(h)


        if self.token_decoder_with_teacher_forcing:
            token_inputs_embeds = self.tok_embeddings(tokens, aggregation=False)
            next_token_inputs_embeds = torch.roll(token_inputs_embeds, shifts=-1, dims=1)

            # batchsize x length x (n_q_tokens + 1) x embed_dim
            h = torch.cat([h[:,:,None], next_token_inputs_embeds], dim=2).to(h)

            h = self.token_head(h.flatten(0,1)).reshape(h.shape)
        else:
            self.token_head.set_infer_id_embs_fn(self.tok_embeddings._get_token_embeddings)
            _, h = self.token_head.generate(
                h.flatten(0,1),
                return_logits=True,
                max_new_tokens=self.token_head_max_new_tokens
            )
            self.token_head.reset_infer_id_embs_fn()

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
    
    def __init__(self, qwen_config: UnifiedQwen3Config, token_decoder_config: UnifiedTokenDecoderConfig, tokenizer_config: KeyeTokenizerConfig):
        """
        初始化UnifiedQwen3Model
        
        Args:
            qwen_config: Qwen3配置对象
            token_decoder_config: Token解码器配置对象
        """
        # 调用父类初始化
        super().__init__(qwen_config)
        assert qwen_config.tie_word_embeddings == False, "tie_word_embeddings must be False in UnifiedQwen3Model"
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
            pad_token_id=qwen_config.pad_token_id,
            image_token_id=qwen_config.image_token_id,
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
            output_last_hidden_states_only=qwen_config.output_last_hidden_states_only,
            output=nn.Linear(qwen_config.embed_dim, qwen_config.vocab_size + tokenizer_config.codebook_size, bias=False),
            token_head=token_head,
            token_decoder_with_teacher_forcing=qwen_config.token_decoder_with_teacher_forcing,
            token_head_max_new_tokens=qwen_config.n_q_tokens+1
        )


    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_image_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        前向传播函数，支持input_image_ids处理
        
        Args:
            tokens: 输入token IDs
            attention_mask: 注意力掩码
            input_pos: 位置IDs
            past_key_values: 过去的key/value缓存
            inputs_embeds: 输入嵌入
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            return_dict: 是否返回字典格式
            input_image_ids: 输入图像token IDs
            cache_position: 缓存位置
        """
        if tokens.size(-1) == 1:
            tokens = self.model.tok_embeddings.expand_input_ids(
                input_image_ids=input_image_ids,
                tokens=tokens,
            )
        # 调用父类的forward方法获取基本功能
        outputs = super().forward(
            tokens=tokens,
            attention_mask=attention_mask,
            input_pos=input_pos,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )
        return outputs

    def convert_hf_state_dict(self, 
                              hf_state_dict: Dict[str, torch.Tensor],
                              tie_word_embeddings: bool = True,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to UnifiedQwen3Model state dictionary.
        
        This implementation reuses the Qwen3Model's convert_hf_state_dict logic for the main model
        and adds handling for the token_head parameter.
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
        # First, use Qwen3Model's convert_hf_state_dict for the main model components
        # Extract the keys that belong to the main model (excluding token_head)
        main_model_state_dict = {}
        token_head_state_dict = {}
        
        for hf_key, tensor in hf_state_dict.items():
            # Extract token_head.* keys for separate processing
            if hf_key.startswith("model.token_head."):
                new_k = hf_key[len("model.token_head."):]
                token_head_state_dict[new_k] = tensor
            else:
                main_model_state_dict[hf_key] = tensor
        
        # Convert the main model state dict using Qwen3Model's convert_hf_state_dict
        converted_state_dict = super().convert_hf_state_dict(
            hf_state_dict=main_model_state_dict,
            **kwargs
        )
        
        # Handle token_head weights using UnifiedTokenDecoder's convert_hf_state_dict
        if token_head_state_dict:
            converted_token_head_state_dict = UnifiedTokenDecoder.convert_hf_state_dict(
                state_dict=token_head_state_dict,
                reduce_mode=True  # We want to reduce the output dimensions
            )
            
            # Add back the "model.token_head." prefix
            for k, v in converted_token_head_state_dict.items():
                converted_key = f"model.token_head.{k}"
                converted_state_dict[converted_key] = v
        
        # model.tok_embeddings.weight
        converted_state_dict["model.tok_embeddings.embed_tokens.weight"] = converted_state_dict["model.tok_embeddings.weight"]
        del converted_state_dict["model.tok_embeddings.weight"]

        if 'model.token_head.token_embedding.weight' in converted_state_dict:
            print("delete model.token_head.token_embedding.weight")
            del converted_state_dict['model.token_head.token_embedding.weight']

        if not tie_word_embeddings:
            converted_state_dict["model.output.weight"] = hf_state_dict["lm_head.weight"]

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
        
        # 主语言模型
        self.model = UnifiedQwen3Model(qwen_config=qwen_config, token_decoder_config=token_decoder_config, tokenizer_config=tokenizer_config)
        
        # 配置参数
        self.vocab_size = qwen_config.vocab_size
        
        # 位置相关
        self.rope_deltas = None
        
        # LM头
        lm_head_size = qwen_config.vocab_size + tokenizer_config.codebook_size 
        self.lm_head = nn.Linear(qwen_config.embed_dim, lm_head_size, bias=False)

    def convert_hf_state_dict(self, 
                              hf_state_dict: Dict[str, torch.Tensor],
                              tie_word_embeddings: bool = True,
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to KeyeARModel state dictionary.
        
        This implementation reuses the UnifiedQwen3Model's convert_hf_state_dict logic for the main model
        and adds handling for the lm_head and visual_tokenizer parameters.
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            tie_word_embeddings: Whether the model ties embeddings (skip lm_head if True).
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
    
        # First, use UnifiedQwen3Model's convert_hf_state_dict for the main model components
        # Extract the keys that belong to the main model (excluding visual_tokenizer and lm_head)
        main_model_state_dict = {}
        lm_head_weight = None
        visual_tokenizer_state_dict = {}
        
        for hf_key, tensor in hf_state_dict.items():
            if hf_key == "lm_head.weight":
                lm_head_weight = tensor
            elif hf_key.startswith("visual_tokenizer."):
                # Extract visual_tokenizer weights
                new_k = hf_key[len("visual_tokenizer."):]
                visual_tokenizer_state_dict[new_k] = tensor
            elif hf_key.startswith("visual."):
                # Extract visual weights and add visual_tokenizer prefix
                new_k = "visual_tokenizer." + hf_key
                main_model_state_dict[new_k] = tensor
            elif hf_key.startswith("quant_projector."):
                # Convert quant_projector to up_projectors
                new_k = hf_key.replace("quant_projector.", "up_projectors.")
                visual_tokenizer_state_dict[new_k] = tensor
            elif hf_key.startswith("model.model.layers."):
                # Handle nested model structure: model.model.layers.* -> model.layers.*
                # Remove the extra "model." prefix to match Qwen3Model's expected format
                new_k = hf_key.replace("model.model.layers.", "model.layers.")
                main_model_state_dict[new_k] = tensor
            elif hf_key.startswith("model.visual_tokenizer.model.model.layers."):
                # Handle nested visual_tokenizer structure: model.visual_tokenizer.model.model.layers.* -> visual_tokenizer.model.layers.*
                new_k = hf_key.replace("model.visual_tokenizer.model.model.layers.", "visual_tokenizer.model.layers.")
                main_model_state_dict[new_k] = tensor
            elif hf_key.startswith("model.visual_tokenizer.model."):
                # Handle other visual_tokenizer nested structure: model.visual_tokenizer.model.* -> visual_tokenizer.model.*
                new_k = hf_key.replace("model.visual_tokenizer.model.", "visual_tokenizer.model.")
                main_model_state_dict[new_k] = tensor
            # 修复：不要预先转换model.embed_tokens.weight，让Qwen3Model来处理
            # elif hf_key == "model.embed_tokens.weight":
            #     # Convert model.embed_tokens.weight to model.tok_embeddings.embed_tokens.weight
            #     new_k = "model.tok_embeddings.embed_tokens.weight"
            #     main_model_state_dict[new_k] = tensor
            else:
                # 修复：对于其他键，如果以"model."开头，需要保留这个前缀
                # 因为UnifiedQwen3Model期望接收带有"model."前缀的键
                if hf_key.startswith("model."):
                    main_model_state_dict[hf_key] = tensor
                else:
                    # 对于不以"model."开头的键，需要添加"model."前缀
                    main_model_state_dict[f"model.{hf_key}"] = tensor

            if hf_key == "lm_head.weight":
                main_model_state_dict["lm_head.weight"] = tensor

        # Convert the main model state dict using UnifiedQwen3Model's convert_hf_state_dict
        # 修复：正确传递参数，将tie_word_embeddings作为关键字参数而不是位置参数
        converted_state_dict = self.model.convert_hf_state_dict(
            hf_state_dict=main_model_state_dict,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        # 修复：将"model."前缀加回到转换后的键上
        final_converted_state_dict = {}
        for k, v in converted_state_dict.items():
            # 如果键不是以"model."开头，则添加"model."前缀
            final_converted_state_dict[f"model.{k}"] = v
            
        # 更新converted_state_dict引用
        converted_state_dict = final_converted_state_dict

        # Handle the lm_head parameter
        if lm_head_weight is not None and not tie_word_embeddings:
            converted_state_dict["lm_head.weight"] = lm_head_weight
            
        # Handle visual_tokenizer weights using KeyeImageTokenizer's convert_hf_state_dict method
        if visual_tokenizer_state_dict:
            # Convert using KeyeImageTokenizer's convert_hf_state_dict method
            converted_visual_tokenizer_state_dict = self.visual_tokenizer.convert_hf_state_dict(visual_tokenizer_state_dict)
            
            for k, v in converted_visual_tokenizer_state_dict.items():
                converted_key = f"visual_tokenizer.{k}"
                converted_state_dict[converted_key] = v

        return converted_state_dict

    def expand_with_image_tokens(
        self,
        input_image_ids: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        拓展input_ids矩阵，将image_token_id对应的行替换为input_image_ids和eos_token
        
        参数说明：
            input_image_ids: 图像索引矩阵，维度为 (im_len, n_q_tokens)
            tokens: 原始输入ID矩阵，维度为 (batch_size, len) 或 (batch_size, len, 1)
            padded_token: 填充标记的整数ID
            image_token_id: 用于标识需要替换为图像tokens的特殊标记ID
        
        返回值：
            expanded_ids: 拓展后的矩阵，维度为 (batch_size, len, 1 + n_q_tokens)
        """
        if tokens.ndim == 3 and tokens.size(2) != 1:
            return tokens # 已经拓展过了
        
        # 记录原始维度
        original_shape = tokens.shape
        batch_size = original_shape[0]
        
        # 如果是3D且最后一维为1，则squeeze最后一维
        if tokens.dim() == 3 and tokens.size(-1) == 1:
            tokens = tokens.squeeze(-1)
        elif tokens.dim() != 2:
            raise ValueError(f"input_ids必须是2D或3D张量，当前为 {tokens.shape}")
            
        # 确保input_ids是2D (batch_size, len)
        assert tokens.dim() == 2, f"input_ids必须是2D张量，当前为 {tokens.shape}"
        
        # 获取序列长度
        len_seq = tokens.size(1)
        output_dim = 1 + self.config.tokenizer_config.n_q_tokens  # 输出矩阵的列数
        
        # 将input_ids flatten成 (batch_size * len, 1) 的形式以便处理
        flattened_input_ids = tokens.view(-1, 1)  # (batch_size * len, 1)
        
        # 初始化输出矩阵，所有位置先填充q_eos_token
        flattened_expanded_ids = torch.full(
            size=(flattened_input_ids.size(0), output_dim),
            fill_value=self.config.qwen_config.q_eos_token,
            dtype=flattened_input_ids.dtype,
            device=flattened_input_ids.device
        )
        
        # 找到flattened_input_ids中等于image_token_id的行索引
        image_token_mask = (flattened_input_ids.squeeze(1) == self.config.qwen_config.image_token_id)  # (batch_size * len,)
        image_token_indices = torch.nonzero(image_token_mask, as_tuple=True)[0]  # 满足条件的行索引
        
        # 校验：input_image_ids的行数必须等于image_token的数量
        assert input_image_ids.size(0) == len(image_token_indices), \
            f"input_image_ids的行数 ({input_image_ids.size(0)}) 必须等于input_ids中image_token_id的数量 ({len(image_token_indices)})"
        
        # 处理非image_token的行
        # 第一列填充原始input_ids的值
        non_image_mask = ~image_token_mask  # (batch_size * len,)
        flattened_expanded_ids[non_image_mask, 0] = flattened_input_ids[non_image_mask, 0]
        
        if output_dim > 1:  # 确保至少有第二列
            flattened_expanded_ids[non_image_mask, 1] = self.config.qwen_config.q_eos_token  # self.q_eos_token

        # 处理image_token的行
        if len(image_token_indices) > 0:  # 只有存在image_token时才处理
            # 前n_q_tokens列：填充对应位置的input_image_ids
            flattened_expanded_ids[image_token_indices, :self.config.qwen_config.n_q_tokens] = input_image_ids
            # 最后一列：填充q_eos_token
            flattened_expanded_ids[image_token_indices, -1] = self.config.qwen_config.q_eos_token
        
        # 将结果reshape回原来的batch格式 (batch_size, len, output_dim)
        expanded_ids = flattened_expanded_ids.view(batch_size, len_seq, output_dim)
        
        return expanded_ids

    def forward_image_tokens(
            self,
            pixel_values,
            image_grid_thw,
            **kwargs
            ):
        vq_out = self.visual_tokenizer(pixel_values, image_grid_thw)
        indices = torch.stack([x_i for x_i in vq_out['indices']], 0).T 
        aligned_indices = self.vocab_size + indices + torch.arange(self.config.tokenizer_config.n_q_tokens).\
            to(next(iter(self.parameters())).device)[None] * self.config.tokenizer_config.codebook_size // self.config.tokenizer_config.n_q_tokens
        return aligned_indices

    def forward(
        self,
        tokens: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs
    ):                
        if pixel_values is not None:
            with torch.no_grad():
                vq_out = self.visual_tokenizer(pixel_values, image_grid_thw)
                aligned_indices = torch.stack([x_i for x_i in vq_out['indices']], 0).T
                aligned_indices = self.vocab_size + aligned_indices + torch.arange(self.config.tokenizer_config.n_q_tokens).\
                    to(tokens)[None] * self.config.tokenizer_config.codebook_size // self.config.tokenizer_config.n_q_tokens

        else:
            aligned_indices = torch.zeros(0, self.config.tokenizer_config.n_q_tokens).to(tokens)
        
        tokens = self.expand_with_image_tokens(aligned_indices, tokens)
        assert input_pos.ndim == 2, "input_pos must be 2D"
        assert tokens.ndim == 3, "tokens must be 3D after expansion, get {}".format(tokens.shape)
        assert tokens.size(2) == self.config.qwen_config.n_q_tokens + 1, \
            "tokens must have {} columns after expansion, get {}. aligned_indices: {}".format(self.config.qwen_config.n_q_tokens + 1, tokens.size(2), aligned_indices)
        # print(f"tokens={tokens.shape}, input_pos={input_pos.shape}")
        # 调用Qwen3Model
        outputs = self.model(
            tokens=tokens,
            attention_mask=attention_mask,
            input_pos=input_pos,
            **kwargs
        )
        return outputs
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=18,  # 指的是原始token数量 (未repeat前的)
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        return_1d_ids=False,
        **model_kwargs
    ):
        """
        多模态生成函数，处理image token扩展和循环生成逻辑

            输入的input_ids可以是batchsize x length，也可以是batchsize x length x (n_q_tokens + 1)
        如果是batchsize x length，你需要自动repeat到batchsize x length x (n_q_tokens + 1)
            规则是，如果id位置是self.config.qwen_config.image_token_id，则将拓展之后的前n_q_tokens个id替换为self.config.qwen_config.image_token_id，最后一个替换为self.config.qwen_config.q_eos_token
            否则，第二个位置就替换为self.config.qwen_config.q_eos_token，其余位置是0。
        
        模型每次输出的logits是batchsize x length x 9 x voc_size, 其中有一个token是self.config.qwen_config.q_eos_token, 实际有效的token是self.config.qwen_config.q_eos_token之前token。
        你需要根据这个输出的logits，恢复出batchsize x length x 9 input_ids，然后输回模型。

        input_ids: batchsize x length, 输入的token id，其中第一个位置是self.config.q_bos_token，其余位置是0。
        
        自动从model_kwargs中移除attention_mask以适配flash attention。

        note:
            这个方法及其消耗显存

        Args:
            input_ids: 输入token，支持shape=(batch, len)或(batch, len, 9)
            max_new_tokens: 最大生成步数
            temperature: 采样温度
            top_k: Top-K采样参数
            top_p: Top-P采样参数
            **model_kwargs: 其他模型参数
        
        Returns:
            generated_ids: 生成的token，shape=(batch, gen_len, 9)
        
        """
        self.eval()
        assert self.config.qwen_config.token_decoder_with_teacher_forcing == self.model.model.token_decoder_with_teacher_forcing == False, \
            "token_decoder_with_teacher_forcing must be True, but get configured as {} and param set as {}".format(
                self.config.qwen_config.token_decoder_with_teacher_forcing,
                self.model.model.token_decoder_with_teacher_forcing
            )

        # 核心参数定义
        n_q_tokens = self.config.tokenizer_config.n_q_tokens
        n_tokens = n_q_tokens + 1  # 每组token数（9）
        batch_size = input_ids.size(0)
        image_token_id = self.config.qwen_config.image_token_id
        q_eos_token = self.config.qwen_config.q_eos_token
        pad_token_id = q_eos_token  # self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 0

        input_seq_len = input_ids.size(1)

        # 处理max_new_tokens参数
        if max_new_tokens is not None:
            # 如果指定了max_new_tokens，计算生成的总长度
            max_length = input_seq_len + max_new_tokens

        self.model.model.setup_caches(
            batch_size=batch_size,
            dtype=next(self.model.model.parameters()).dtype,
            decoder_max_seq_len=max_length
        )

        # 删除attention_mask以适配flash attention
        model_kwargs.pop('attention_mask', None)
        
        # ==============================================
        # 1. 输入处理：将2D input_ids扩展为3D (batch, len, 9)
        # ==============================================
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
            # 初始化扩展后的tensor
            expanded_ids = torch.full(
                (batch_size, seq_len, n_tokens),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            
            # 填充第一个token（保持原输入）
            expanded_ids[:, :, 0] = input_ids
            
            # 识别image token位置
            image_mask = (input_ids == image_token_id)

            # 处理image组：前8个设为image_token_id，最后一个设为q_eos_token
            expanded_ids[..., :n_q_tokens][image_mask] = image_token_id
            expanded_ids[..., -1][image_mask] = q_eos_token
            
            # 处理文本组：第二个位置设为q_eos_token，其余保持pad
            non_image_mask = ~image_mask
            expanded_ids[..., 1][non_image_mask] = q_eos_token
            
            current_ids = expanded_ids  # (batch, prompt_groups, 9)
        elif input_ids.dim() == 3:
            # 已为3D输入，直接使用
            current_ids = input_ids.clone()
        else:
            raise ValueError(f"input_ids维度必须是2或3，当前为{input_ids.dim()}")
        
        prompt_groups = current_ids.shape[1]  # prompt的组数
        cache = None  # 初始化KV Cache
        
        # ==============================================
        # 辅助函数：采样单个token组
        # ==============================================
        def _sample_group(logits, temperature, top_k, top_p):
            """采样一组token（9个）"""
            # temperature缩放
            if temperature > 0:
                logits = logits / (temperature + 1e-5)
            
            # Top-K过滤
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf')).scatter_(-1, indices, values)
            
            # Top-P过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 0] = False  # 至少保留第一个token
                sorted_logits[mask] = float('-inf')
                logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_indices, dim=-1))
            
            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1)[..., 0]
            #print(f"next_tokens={next_tokens.shape}, probs={probs.shape}") # next_tokens=torch.Size([350, 1]), probs=torch.Size([350, 217472])
            #print(next_tokens)
            #print(probs[-1])
            next_tokens = next_tokens.reshape(batch_size, -1, next_tokens.shape[-1])
            #print(f"next_tokensnext_tokens", next_tokens.shape)
            #print(f"next_tokens={next_tokens}")
            next_tokens[...,-1] = q_eos_token
            #print(f"new_next_tokens", next_tokens)

            # 处理组内EOS：EOS之后的token保持pad值
            eos_mask = (next_tokens == q_eos_token)
            eos_indices = eos_mask.int().argmax(dim=-1)  # 每组第一个EOS的位置
            pos_indices = torch.arange(n_tokens, device=next_tokens.device).expand(batch_size, -1)
            keep_pad_mask = pos_indices > eos_indices.unsqueeze(-1)
            #print(f"keep_pad_mask: {keep_pad_mask.shape}\n{keep_pad_mask}")
            #print(f"next_tokens: {next_tokens.shape}\n{next_tokens}")
            next_tokens = torch.where(keep_pad_mask, pad_token_id, next_tokens)
            #print(f"n333333", next_tokens.shape)
            #print(next_tokens)
            return next_tokens
        
        # ==============================================
        # 2. 生成逻辑：Prefill + Decode
        # ==============================================
        # Prefill阶段：首次输入完整prompt，获取初始cache
        if prompt_groups > 0:
            prefill_pos = torch.arange(input_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            outputs = self(
                current_ids,
                input_pos=prefill_pos,
                **model_kwargs
            )
            logits = outputs.logits  # (batch, 9, vocab_size)
            logits = torch.nn.functional.pad(logits, (0,0,0, n_tokens - logits.shape[1]), value=0)

            #print(f"logits0000={logits.shape}")
            logits = logits.reshape(batch_size, -1, logits.shape[-2], logits.shape[-1])
            
            #print(f"logits={logits.shape}")
            # 采样最后一个prompt group的下一个group
            last_group_logits = logits[:, -1, :]  # (batch, 9, vocab_size)
            next_group = _sample_group(last_group_logits, temperature, top_k, top_p)
            #print(f"next_group={next_group.shape}, current_ids={current_ids.shape}") #  current_ids=
            #print(f"current_ids={current_ids}")
            # next_group=torch.Size([1, 350, 9]), current_ids=torch.Size([1, 350, 9])
            current_ids = torch.cat([current_ids, next_group], dim=1)
        
        # Decode阶段：增量生成，仅输入新增group
        for step in range(1, max_new_tokens):
            #print(f"\n\n\n")
            # 仅取最后一个group作为输入（增量生成）
            last_group = current_ids[:, -1:, :]  # (batch, 1, 9)
            current_pos = torch.tensor([[step]], device=input_ids.device).expand(batch_size, -1)

            # 移除不需要的视觉参数
            for key in ["pixel_values", "image_grid_thw", "video_grid_thw", 
                    "fast_video_grid_thw", "pixel_values_videos", "input_image_ids"]:
                model_kwargs.pop(key, None)
            
            #print(f"input last_group={last_group}")
            # 模型前向（使用cache）
            outputs = self(
                last_group,
                input_pos=current_pos,
                **model_kwargs
            )
            logits = outputs.logits  # (batch, 1, 9, vocab_size)
            logits = torch.nn.functional.pad(logits, (0,0,0, n_tokens - logits.shape[1]), value=q_eos_token)

            # 采样新group
            current_logits = logits[:, :, :]  # (batch, 9, vocab_size)
            next_group = _sample_group(current_logits, temperature, top_k, top_p)

            # Append新生成的group
            current_ids = torch.cat([current_ids, next_group], dim=1)
            
            # 提前终止：新增group的第一个token是EOS
            # next_group, batchsize x length x n_tokens
            if (next_group[..., 0] == self.config.eos_token_id).all():
                break
        
        self.model.model.reset_caches()

        # ==============================================
        # 3. 返回结果
        # ==============================================
        generated_ids = current_ids
        
        if return_1d_ids:
            generated_ids = generated_ids[...,0]
        return generated_ids