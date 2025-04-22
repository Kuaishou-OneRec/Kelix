import os
import os.path as osp
import torch
import deepspeed
from PIL import Image, ImageOps
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import base64
from transformers import AutoProcessor, AutoModel
from recipes.ViT.training.models.siglip.modeling_siglip import SiglipPreTrainedModel, SiglipModel
from recipes.ViT.training.models.siglip.processing_siglip import SiglipProcessor
from recipes.ViT.training.models.moonvit.modeling_moonvit import MoonVitPretrainedModel
from recipes.ViT.training.models.moonvit.image_processing_moonvit import MoonViTImageProcessor
import torch.nn.functional as F
from PIL.Image import Resampling as PILImageResampling
from transformers.image_transforms import resize
import transformers
from recipes.ViT.helpers.context import Context


class DisCoGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, context):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")

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


class MoonViT(nn.Module):

    def __init__(self, config, ctx):
        super().__init__()
        self.config = config
        self.ctx = ctx
        self.is_dist = self.ctx.is_dist

        self.text_model = SiglipModel.from_pretrained(
            config.dir, ignore_mismatched_sizes=True
        )
        self.image_processor = MoonViTImageProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/MoonViT-SO-400M")
        self.image_model = MoonVitPretrainedModel.from_pretrained(
            "/llm_reco_ssd/zhouyang12/models/MoonViT-SO-400M",
            ignore_mismatched_sizes=True
        )
        self.text_processor = SiglipProcessor.from_pretrained(config.dir)
        self.tokenizer = self.text_processor.tokenizer

        hidden_size = self.text_model.hidden_size
        vocab_size = self.text_model.vocab_size
        self.logit_scale = self.text_model.logit_scale
        self.logit_bias = self.text_model.logit_bias
        self.text_model = self.text_model.text_model
        
        # Add projection layer for text embeddings to match vision embedding size

        if config.text_decoder.enabled:
            self.text_decoder = AutoModel.from_pretrained(
                config.text_decoder.model_dir,
                use_cache=False
            )
            self.image_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.image_end_embed = nn.Parameter(
                torch.zeros(size=(1, 1, hidden_size), dtype=torch.float32),
                requires_grad=True
            )
            self.text_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.vocab_proj = nn.Linear(hidden_size, vocab_size, bias=False)
            self.regression_loss_fn = nn.CrossEntropyLoss()
            raise NotImplementedError("Not finish yet.")
        else:
            self.text_decoder = None
            self.image_proj = None
            self.text_proj = None
            self.regression_loss_fn = None

    def calcul_regression_loss(self, logits, labels, loss_mask):
        """
        logits: B * L * D
        labels: B * L
        loss_mask: B * L
        """
        pass

    def calcul_loss(self, text_embeds, image_pooler):
        # Project text embeddings to match image embedding dimension
        
        if self.is_dist:
            device = text_embeds.device

            gathered_image_pooler = disco_gather(image_pooler, self.ctx)
            gathered_text_embeds = disco_gather(text_embeds, self.ctx)

            logits_per_text = torch.matmul(gathered_text_embeds, gathered_image_pooler.t().to(device))

            logit_scale = self.logit_scale.to(device)
            logit_bias = self.logit_bias.to(device)
            logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

            logits_per_image = logits_per_text.t()

            eye = torch.eye(logits_per_text.size(0), device=device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
            return loss
        else:
            # Calculate loss without distributed training
            device = text_embeds.device
            logits_per_text = torch.matmul(text_embeds, image_pooler.t().to(device))
            
            logit_scale = self.logit_scale.to(device)
            logit_bias = self.logit_bias.to(device)
            logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
            
            eye = torch.eye(logits_per_text.size(0), device=device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
            return loss

    def to_cuda(self, inputs, device):
        if isinstance(inputs, (list, tuple)):
            inputs = list(inputs)
            for idx, item in enumerate(inputs):
                inputs[idx] = self.to_cuda(item, device)
            return inputs
        elif isinstance(inputs, (dict, transformers.tokenization_utils_base.BatchEncoding, transformers.image_processing_base.BatchFeature)):
            for key in inputs:
                inputs[key] = self.to_cuda(inputs[key], device)
            return inputs
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        return inputs

    def forward(self, images, texts):
        text_inputs = self.text_processor(images=images, text=texts, padding="longest", return_tensors="pt")
        device = torch.cuda.current_device()
        text_inputs = self.to_cuda(text_inputs, device)
        # print('--------------------------------')
        # for key in text_inputs:
        #     print(key, type(text_inputs[key]))
        #     print(text_inputs[key].shape)
        #     print(text_inputs[key].dtype)
        # print('--------------------------------')
        text_outputs = self.text_model(
            input_ids=text_inputs.input_ids
        )
        text_embeds = text_outputs.pooler_output
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        processed_images = []
        for image in images:
            #将base64编码的图片转换为PIL.Image.Image
            image = base64.b64decode(image)
            image = Image.open(image)
            processed_images.append(image)
        images_processed = self.image_processor(processed_images, return_tensors="pt").to(dtype=self.image_model.dtype, device=self.image_model.device)
        # image_outputs = self.image_model(images_processed.pixel_values, images_processed.image_grid_hws)
        # image_embeds = image_outputs
        pooler = self.image_model.get_image_embeddings(images_processed.pixel_values, images_processed.image_grid_hws)
        loss = self.calcul_loss(text_embeds, pooler)
        text_output = text_outputs
        image_output = pooler

        # text_hidden_embeds = text_output.last_hidden_state
        # image_hidden_embeds = image_output.last_hidden_state

        batch_size = image_output.shape[0]
        assert text_embeds.shape[0] == image_output.shape[0]


        return pooler, text_embeds, Context(
            loss=loss,
            total_image_num_tokens=np.prod(image_output.shape[:2]).item(),
            total_text_num_tokens=np.prod(text_embeds.shape[:2]).item(),
            total_num_samples=batch_size,
            total_text_num_valid_tokens=(text_inputs.input_ids != self.tokenizer.pad_token_id).long().sum().item()
        )
