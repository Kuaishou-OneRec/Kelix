import os
import os.path as osp
import torch
import numpy as np
import deepspeed
from PIL import Image
from einops import rearrange
import torch.nn as nn
import torch.distributed as dist
import transformers
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
from .MoonVision.modeling_kimi_vl import KimiVLMultiModalProjector, KimiVLMultiModalProjector_Contrastive

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
        
        self.loss_lambda = 2.0

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
        # (image_token_nums_0 + image_token_nums_1 + ..., image_feature_dim)
        image_features_for_AR: torch.Tensor = self.mlp_AR(image_features)
        image_features: torch.Tensor = self.mlp(image_features)

        return image_features, image_features_for_AR

    def _merge_with_image_features(
            self,
            inputs_embeds: torch.Tensor,
            image_features: torch.Tensor,
        ):
            """
            Args:
                inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, input_embed_dim)`):
                    The input embeddings.
                image_features (:obj:`torch.Tensor` of shape :obj:`(image_token_nums, image_feature_dim)`):
                    The image features to prepend to the input embeddings.
            """
            batch_size, sequence_length, input_embed_dim = inputs_embeds.shape
            image_feature_nums, image_feature_dim = image_features.shape
            
            assert image_feature_dim == input_embed_dim
            
            # Create new embeddings tensor with expanded sequence length
            new_sequence_length = sequence_length + image_feature_nums
            new_inputs_embeds = torch.zeros(
                (batch_size, new_sequence_length, input_embed_dim),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )
            
            # Prepend image features to each sequence in the batch
            new_inputs_embeds[:, :image_feature_nums, :] = image_features.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Copy original text embeddings after the image features
            new_inputs_embeds[:, image_feature_nums:, :] = inputs_embeds
            
            return new_inputs_embeds

    def compute_loss_with_image_features(
            self,
            inputs_embeds: torch.Tensor,
            image_features: torch.Tensor,
            labels: torch.LongTensor,
        ):
        """
        Args:
            inputs_embeds (torch.Tensor of shape (batch_size, sequence_length, input_embed_dim)):
                The input text embeddings.
            image_features (torch.Tensor of shape (image_token_nums, image_feature_dim)):
                The image features to prepend.
            labels (torch.LongTensor of shape (batch_size, sequence_length)):
                Labels for language modeling loss. Will be extended with -100 for image positions.
        """
        batch_size = inputs_embeds.size(0)
        image_feature_nums = image_features.size(0)
        
        combined_embeds = self._merge_with_image_features(inputs_embeds, image_features)
        
        if labels is not None:
            new_labels = torch.full(
                (batch_size, combined_embeds.size(1)),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            new_labels[:, image_feature_nums:] = labels
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
        outputs = self.siglip(**inputs)
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
                raise NotImplementedError
            else:
                vision_embeds = self.mlp_AR(vision_embeds)
                decoder_inputs_embeds = torch.concat([vision_embeds, text_embeds], dim=1)
                vision_labels = decoder_inputs_embeds.new_full(vision_embeds.shape[:2], -100, dtype=torch.int64)
                # text_labels = decoder_inputs_embeds.new_full(text_embeds.shape[:2], -100, dtype=torch.int64)
                text_labels = input_ids.clone()
                text_labels[input_ids == pad_token_id] == -100
                labels = torch.concat([vision_labels, text_labels], dim=1)
                decoder_outputs = self.text_decoder(inputs_embeds=decoder_inputs_embeds, labels=labels)

                AR_loss = decoder_outputs.loss
                loss = Contrastive_loss + self.loss_lambda * AR_loss
            
            return outputs, ViTOutputs(
                Contrastive_loss=Contrastive_loss,
                AR_loss=AR_loss,
                loss=loss,
                total_image_num_tokens=np.prod(vision_embeds.shape[:2]).item(),
                total_text_num_tokens=np.prod(siglip_text_embeds.shape[:2]).item(),
                total_num_samples=batch_size,
                total_text_num_valid_tokens=(input_ids != pad_token_id).long().detach().cpu().sum().item()
            )

        return outputs, ViTOutputs(
            Contrastive_loss=loss,
            AR_loss=torch.FloatTensor(0),
            loss=loss,
            total_image_num_tokens=np.prod(vision_embeds.shape[:2]).item(),
            total_text_num_tokens=np.prod(siglip_text_embeds.shape[:2]).item(),
            total_num_samples=batch_size,
            total_text_num_valid_tokens=(input_ids != pad_token_id).long().sum().item()
        )
