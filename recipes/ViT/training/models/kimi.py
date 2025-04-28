import os
import os.path as osp
import torch
import numpy as np
import deepspeed
from PIL import Image
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .siglip.modeling_siglip import SiglipPreTrainedModel, SiglipModel
from .siglip.processing_siglip import SiglipProcessor
import torch.nn.functional as F
from recipes.ViT.helpers.context import Context

from .MoonVision.modeling_kimi_vl import MoonVitPretrainedModel
from .MoonVision.image_processing_kimi_vl import KimiVLImageProcessor
from .MoonVision.configuration_kimi_vl import MoonViTConfig, KimiVLConfig
from .MoonVision.modeling_kimi_vl import KimiVLMultiModalProjector, KimiVLMultiModalProjector_Contrastive

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

        # T5 text model for Contrastive loss
        self.siglip = SiglipModel.from_pretrained(config.siglip_dir, ignore_mismatched_sizes=True).cuda()
        self.siglip_processor = SiglipProcessor.from_pretrained(config.siglip_dir)

        MoonViT_config = MoonViTConfig()
        KimiVL_Config_AR = KimiVLConfig()
        KimiVL_Config = KimiVLConfig()
        MoonViT_config._attn_implementation = 'flash_attention_2'

        self.MoonVIT_processor = KimiVLImageProcessor()
        self.MoonVIT = MoonVitPretrainedModel(MoonViT_config).cuda()
        
        self.loss_lambda = 2

        # LLM for capation AR loss
        if config.text_decoder.enabled:
            self.text_decoder = AutoModelForCausalLM.from_pretrained(
                config.text_decoder.model_dir,
                use_cache=False
            ).cuda()
            self.text_decoder_tokenizer = AutoTokenizer.from_pretrained(config.text_decoder.model_dir)
            KimiVL_Config_AR.text_config.hidden_size = AutoConfig.from_pretrained(config.text_decoder.model_dir).hidden_size
            KimiVL_Config.text_config.hidden_size = self.siglip.hidden_size

            self.mlp = KimiVLMultiModalProjector_Contrastive(KimiVL_Config).cuda()
            self.mlp_AR = KimiVLMultiModalProjector(KimiVL_Config_AR).cuda()

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

    def calcul_loss(self, text_embeds, image_embeds):
        if self.is_dist:
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

    def forward(self, images, texts):
        vit_inputs = self.MoonVIT_processor(images=images, return_tensors="pt")
        inputs = self.siglip_processor(images=images, text=texts, padding="longest", return_tensors="pt")
        input_ids = self.text_decoder_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        for key in vit_inputs:
            vit_inputs[key] = vit_inputs[key].cuda()

        vit_inputs['pixel_values'] = vit_inputs['pixel_values'].to(torch.bfloat16)

        for key in inputs:
            inputs[key] = inputs[key].cuda()

        for key in input_ids:
            input_ids[key] = input_ids[key].cuda()

        outputs = self.siglip(**inputs)

        image_features, image_features_for_AR = self._extract_image_features(**vit_inputs)
        inputs_embeds = self.text_decoder.get_input_embeddings()(input_ids.input_ids)

        # Autoregression loss
        AR_loss = self.compute_loss_with_image_features(inputs_embeds=inputs_embeds, image_features=image_features_for_AR, labels=input_ids.input_ids)

        # Contrastive loss
        Contrastive_loss = self.calcul_loss(outputs.text_embeds, image_features)

        loss = Contrastive_loss + self.loss_lambda * AR_loss

        text_output = outputs.text_model_output
        image_output = outputs.vision_model_output

        text_hidden_embeds = text_output.last_hidden_state
        image_hidden_embeds = image_output.last_hidden_state

        batch_size = image_hidden_embeds.shape[0]
        assert text_hidden_embeds.shape[0] == image_hidden_embeds.shape[0]

        return outputs, Context(
            Contrastive_loss=Contrastive_loss,
            AR_loss=AR_loss,
            loss=loss,
            total_image_num_tokens=np.prod(image_hidden_embeds.shape[:2]).item(),
            total_text_num_tokens=np.prod(text_hidden_embeds.shape[:2]).item(),
            total_num_samples=batch_size,
            total_text_num_valid_tokens=(inputs["input_ids"] != self.text_decoder_tokenizer.pad_token_id).long().sum().item()
        )
