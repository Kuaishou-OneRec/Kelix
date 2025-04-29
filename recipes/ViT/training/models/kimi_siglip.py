import os
import os.path as osp
import torch
import numpy as np
import deepspeed
from collections import Counter
from PIL import Image
from einops import rearrange
import torch.nn as nn
import torch.distributed as dist
import transformers
from transformers.activations import GELUActivation, ACT2FN, PytorchGELUTanh
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from .siglip.modeling_siglip import SiglipPreTrainedModel, SiglipModel
from .siglip.processing_siglip import SiglipProcessor
import torch.nn.functional as F
import logging
from torch.nn.utils.rnn import pad_sequence
from recipes.ViT.helpers.context import Context

from .MoonVision.modeling_kimi_vl import MoonVitPretrainedModel
from .MoonVision.image_processing_kimi_vl import KimiVLImageProcessor
from .MoonVision.configuration_kimi_vl import MoonViTConfig, KimiVLConfig
from .MoonVision.modeling_kimi_vl import KimiVLMultiModalProjector_Contrastive

logger = logging.getLogger(__name__)


class DisCoGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, context):
        if not dist.is_initialized():
            raise "torch.distributed is not initialized"

        world_size = context.world_size
        ctx.world_size = context.world_size
        ctx.rank = context.rank
        ctx.batch_size = tensor.size(0)

        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(world_size)
        ]

        dist.all_gather(gathered_tensors, tensor.contiguous())

        gathered_tensors = torch.cat(gathered_tensors, dim=0)
        gathered_tensors.requires_grad_(True)

        return gathered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        batch_size = ctx.batch_size

        dist.all_reduce(grad_output, op=dist.ReduceOp.AVG)
        return grad_output[rank * batch_size: batch_size * (rank + 1)], None


def disco_gather(tensor, context):
    return DisCoGather.apply(tensor, context)


class KimiVLMultiModalProjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.vision_config.hidden_size

        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(
            self.hidden_size, config.text_config.hidden_size, bias=True
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        config = self.config
        if isinstance(image_features, (list, tuple)):
            processed_features = list()
            for image_feature in image_features:
                hidden_states = self.pre_norm(image_feature).view(-1, self.hidden_size)
                hidden_states = self.linear_1(hidden_states)
                hidden_states = self.act(hidden_states)
                hidden_states = self.linear_2(hidden_states)
                processed_features.append(hidden_states)

            return processed_features

        dims = image_features.shape[:-1]
        dim = image_features.shape[-1]
        image_features = image_features.view(np.prod(dims), dim)
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states.view(*dims, -1)


class KimiViT(nn.Module):

    def __init__(self, config, ctx):
        super().__init__()
        self.config = config
        self.ctx = ctx
        self.is_dist = self.ctx.is_dist
        self.use_packing = config.packing.enabled
        self.packing_max_length = config.packing.max_length
        self.packing_drop_ratio = config.packing.drop_ratio
        self.use_decoder = config.text_decoder.enabled
        self.vision_return_embed_list = config.vision_return_embed_list

        # T5 text model for Contrastive loss
        self.siglip = SiglipModel.from_pretrained(config.siglip_dir, ignore_mismatched_sizes=True)
        self.processor = SiglipProcessor.from_pretrained(config.siglip_dir)

        self.hidden_size = self.siglip.hidden_size
        self.vocab_size = self.siglip.vocab_size
        self.patch_size = self.siglip.patch_size

        MoonViT_config = MoonViTConfig()
        KimiVL_Config_AR = KimiVLConfig()
        KimiVL_Config = KimiVLConfig()
        MoonViT_config._attn_implementation = 'flash_attention_2'

        self.MoonVIT_processor = KimiVLImageProcessor()
        self.MoonVIT = MoonVitPretrainedModel(MoonViT_config)
        
        self.loss_lambda = config.text_decoder.loss_lambda

        # LLM for capation AR loss
        if config.text_decoder.enabled:
            self.text_decoder = AutoModelForCausalLM.from_pretrained(
                config.text_decoder.model_dir,
                use_cache=False
            )
            self.text_decoder_tokenizer = AutoTokenizer.from_pretrained(config.text_decoder.model_dir)
            KimiVL_Config_AR.text_config.hidden_size = AutoConfig.from_pretrained(config.text_decoder.model_dir).hidden_size
            KimiVL_Config.text_config.hidden_size = self.siglip.hidden_size

            self.mlp = KimiVLMultiModalProjector_Contrastive(KimiVL_Config)
            self.mlp_AR = KimiVLMultiModalProjector(KimiVL_Config_AR)

        else:
            self.text_decoder = None
            self.image_proj = None
            self.text_proj = None
            self.regression_loss_fn = None

        self.siglip.gradient_checkpointing_enable()
        self.text_decoder.gradient_checkpointing_enable()

    def calcul_regression_loss(self, input_ids, image_feature, loss_mask):
        """
        logits: B * L * D
        labels: B * L
        loss_mask: B * L
        """
        pass

    def calcul_loss(self, outputs):
        if self.is_dist:
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            device = text_embeds.device

            gathered_image_embeds = disco_gather(image_embeds, self.ctx)
            gathered_text_embeds = disco_gather(text_embeds, self.ctx)

            logits_per_text = torch.matmul(gathered_text_embeds, gathered_image_embeds.t().to(device))

            logit_scale = self.siglip.logit_scale.to(device)
            logit_bias = self.siglip.logit_bias.to(device)
            logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

            eye = torch.eye(logits_per_text.size(0), device=device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
            return loss
        
    def _extract_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_hws: torch.LongTensor
    ):
        """
        Args:
            pixel_values (:obj:`torch.FloatTensor` of shape :obj:`(image_token_nums, 3, patch_size, patch_size)`):
                The pixel values of the images processed by image processor.

        Returns:
            image_features (:obj:`torch.FloatTensor` of shape :obj:`(image_token_nums, image_feature_dim)`):
                The selected image features to use as input to the projector head.
        """
        # [(image_token_nums_0, image_feature_dim), (image_token_nums_1, image_feature_dim), ...]
        image_features: list[torch.Tensor] = self.MoonVIT(
            pixel_values, image_grid_hws
        )
        # [(image_token_nums_0, qwen_hidden_dim), (image_token_nums_1, qwen_hidden_dim), ...]
        image_features_for_AR: torch.Tensor = self.mlp_AR(image_features)

        # (B, T5 dim)
        image_features: torch.Tensor = self.mlp(image_features)

        return image_features, image_features_for_AR

    def _merge_with_image_features(
            self,
            inputs_embeds: torch.Tensor,
            image_features: list[torch.Tensor],
        ):
        """
        Args:
            inputs_embeds (torch.Tensor of shape (batch_size, sequence_length, input_embed_dim)):
                The input embeddings.
            image_features (list[torch.Tensor]): A list where each element is a tensor of shape 
                (image_token_nums_i, image_feature_dim) corresponding to the i-th sample in the batch.
                The image_feature_dim should match input_embed_dim.
        """
        batch_size, sequence_length, input_embed_dim = inputs_embeds.shape
        
        # Validate input dimensions
        assert len(image_features) == batch_size, "Number of image features must match batch size"
        for img_feat in image_features:
            assert img_feat.shape[-1] == input_embed_dim, "Image feature dimension must match input embedding dimension"
        
        image_token_nums = [img_feat.shape[0] for img_feat in image_features]
        max_image_tokens = max(image_token_nums) if image_token_nums else 0
        new_sequence_length = sequence_length + max_image_tokens
        
        new_inputs_embeds = torch.zeros(
            (batch_size, new_sequence_length, input_embed_dim),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype
        )
        
        for i in range(batch_size):
            img_feat = image_features[i]
            img_token_num = img_feat.shape[0]
            new_inputs_embeds[i, :img_token_num, :] = img_feat
            new_inputs_embeds[i, img_token_num:img_token_num+sequence_length, :] = inputs_embeds[i]
        
        return new_inputs_embeds

    def compute_loss_with_image_features(
            self,
            inputs_embeds: torch.Tensor,
            image_features: list[torch.Tensor],
            labels: torch.LongTensor,
        ):
        """
        Args:
            inputs_embeds (torch.Tensor of shape (batch_size, sequence_length, input_embed_dim)):
                The input text embeddings.
            image_features (list[torch.Tensor]): A list where each element is a tensor of shape
                (image_token_nums_i, image_feature_dim) corresponding to the i-th sample in the batch.
            labels (torch.LongTensor of shape (batch_size, sequence_length)):
                Labels for language modeling loss. Will be extended with -100 for image positions.
        """
        batch_size = inputs_embeds.size(0)
        
        # Merge text and image embeddings
        combined_embeds = self._merge_with_image_features(inputs_embeds, image_features)
        
        if labels is not None:
            new_labels = torch.full_like(
                combined_embeds[:, :, 0],  
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            for i in range(batch_size):
                img_token_num = image_features[i].size(0)
                new_labels[i, img_token_num:img_token_num+labels.size(1)] = labels[i]
            
            labels = new_labels
        
        outputs = self.text_decoder(inputs_embeds=combined_embeds, labels=labels)
        
        return outputs.loss

    def to_cuda(self, inputs, device):
        if isinstance(inputs, (list, tuple)):
            inputs = list(inputs)
            for idx, item in enumerate(inputs):
                inputs[idx] = self.to_cuda(item, device)
            return inputs
        elif isinstance(inputs, (dict, transformers.tokenization_utils_base.BatchEncoding)):
            for key in inputs:
                inputs[key] = self.to_cuda(inputs[key], device)
            return inputs
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        return inputs

    def forward(self, images, texts, package=None, use_attn_mask=True):
        device = torch.cuda.current_device()

        assert package is not None
        inputs = package
        images = inputs.get("images")
        texts = inputs.get("texts")
        source = inputs.get("source", list())
        if "input_ids" not in inputs:
            text_inputs = self.processor(text=texts, padding="longest", return_tensors="pt")
            # text_inputs = self.text_decoder_tokenizer(texts, return_tensors="pt", padding="longest")
        inputs.update(text_inputs)
            # sample_indices
            # position_ids
            # height_position_ids
            # width_position_ids
            # pixel_values
            # image_indices
            # image_attention_mask
            # cu_seqlens
        for name in ["images", "texts", "source", "task", "image_indices", "height_position_ids", "width_position_ids"]:
            inputs.pop(name, None)

        inputs = self.to_cuda(inputs, device)
        if use_attn_mask:
            attention_mask = (inputs["input_ids"] != self.processor.tokenizer.pad_token_id).long()
            inputs["attention_mask"] = self.to_cuda(attention_mask, device)
        outputs = self.siglip(**inputs, vision_return_embed_list=self.vision_return_embed_list)
        Contrastive_loss = self.calcul_loss(outputs)

        vision_output = outputs.vision_model_output
        vision_embeds = vision_output.last_hidden_state
        text_output = outputs.text_model_output
        siglip_text_embeds = text_output.last_hidden_state

        batch_size = siglip_text_embeds.shape[0]

        ViTOutputs = type("ViTOutputs", (Context, ), {})

        if self.use_decoder:
            text_inputs = self.text_decoder_tokenizer(texts, return_tensors="pt", padding="longest")
            input_ids = self.to_cuda(text_inputs["input_ids"], device)

            text_embeds = self.text_decoder.get_input_embeddings()(input_ids)

            text_embeds = self.to_cuda(text_embeds, device)

            pad_token_id = self.text_decoder_tokenizer.pad_token_id

            if isinstance(vision_embeds, (list, tuple)):
                vision_embeds = list(vision_embeds)
                vision_seqlens = [0] + [emb.shape[0] for emb in vision_embeds]
                vision_cu_seqlens = list(np.cumsum(vision_seqlens))
                vision_embeds = self.mlp_AR(vision_embeds)

                batch_size = len(vision_embeds)
                vision_lengths = [emb.shape[0] for emb in vision_embeds]
                max_vision_length = max(vision_lengths)

                text_labels = input_ids.clone()
                text_labels[input_ids == pad_token_id] = -100

                max_length = max_vision_length + input_ids.shape[1]
                hidden_dim = text_embeds.shape[-1]

                labels = input_ids.new_full((batch_size, max_length), -100, dtype=torch.int64)
                decoder_inputs_embeds = text_embeds.new_zeros((batch_size, max_length, hidden_dim))

                assert text_embeds.shape[0] == batch_size

                for i in range(batch_size):
                    vision_len = vision_lengths[i]
                    text_len = text_labels.shape[1]
                    labels[i, vision_len: vision_len + text_len] = text_labels[i]

                    decoder_inputs_embeds[i, :vision_len, :] = vision_embeds[i]
                    decoder_inputs_embeds[i, vision_len: vision_len + text_len, :] = text_embeds[i]
                columns_all_padding = (labels == -100).all(dim=0).flip(0).long()
                reversed_columns_num = labels.shape[1] - (columns_all_padding.argmin(dim=0).item())
                labels = labels[:, :reversed_columns_num]
                decoder_inputs_embeds = decoder_inputs_embeds[:, :reversed_columns_num]

                assert reversed_columns_num <= max_vision_length + input_ids.shape[1]

                decoder_outputs = self.text_decoder(inputs_embeds=decoder_inputs_embeds, labels=labels)

                AR_loss = decoder_outputs.loss
                loss = Contrastive_loss + self.loss_lambda * AR_loss

                return ViTOutputs(
                    Contrastive_loss=Contrastive_loss,
                    AR_loss=AR_loss,
                    loss=loss,
                    total_image_num_tokens=sum(vision_lengths),
                    total_text_num_tokens=np.prod(siglip_text_embeds.shape[:2]).item(),
                    total_num_samples=batch_size,
                    total_text_num_valid_tokens=(input_ids != pad_token_id).long().detach().cpu().sum().item(),
                    source=source,
                    outputs=outputs
                )
            else:
                vision_embeds = self.mlp_AR(vision_embeds)
                decoder_inputs_embeds = torch.concat([vision_embeds, text_embeds], dim=1)
                vision_labels = decoder_inputs_embeds.new_full(vision_embeds.shape[:2], -100, dtype=torch.int64)
                # text_labels = decoder_inputs_embeds.new_full(text_embeds.shape[:2], -100, dtype=torch.int64)
                text_labels = input_ids.clone()
                text_labels[input_ids == pad_token_id] = -100
                labels = torch.concat([vision_labels, text_labels], dim=1)
                decoder_outputs = self.text_decoder(inputs_embeds=decoder_inputs_embeds, labels=labels)

                AR_loss = decoder_outputs.loss
                loss = Contrastive_loss + self.loss_lambda * AR_loss
            
                return ViTOutputs(
                    Contrastive_loss=Contrastive_loss,
                    AR_loss=AR_loss,
                    loss=loss,
                    total_image_num_tokens=np.prod(vision_embeds.shape[:2]).item(),
                    total_text_num_tokens=np.prod(siglip_text_embeds.shape[:2]).item(),
                    total_num_samples=batch_size,
                    total_text_num_valid_tokens=(input_ids != pad_token_id).long().detach().cpu().sum().item(),
                    source=source,
                    outputs=outputs
                )

        return ViTOutputs(
            Contrastive_loss=loss,
            AR_loss=torch.FloatTensor(0),
            loss=loss,
            total_image_num_tokens=np.prod(vision_embeds.shape[:2]).item(),
            total_text_num_tokens=np.prod(siglip_text_embeds.shape[:2]).item(),
            total_num_samples=batch_size,
            total_text_num_valid_tokens=(input_ids != pad_token_id).long().sum().item(),
            source=source,
            outputs=outputs
        )
