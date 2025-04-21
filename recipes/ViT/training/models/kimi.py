import os
import os.path as osp
import torch
import numpy as np
import deepspeed
from PIL import Image
import torch.nn as nn
import torch.distributed as dist
import transformers
from transformers import AutoProcessor, AutoModel
from .siglip.modeling_siglip import SiglipPreTrainedModel, SiglipModel
from .siglip.processing_siglip import SiglipProcessor
import torch.nn.functional as F
from recipes.ViT.helpers.context import Context


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
        self.packing = ctx.packing.enabled
        self.packing_max_length = ctx.packing.max_length

        self.model = SiglipModel.from_pretrained(config.dir, ignore_mismatched_sizes=True)
        self.processor = SiglipProcessor.from_pretrained(config.dir)
        self.tokenizer = self.processor.tokenizer

        hidden_size = self.model.hidden_size
        vocab_size = self.model.vocab_size

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

    def calcul_loss(self, outputs):
        if self.is_dist:
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            device = text_embeds.device

            gathered_image_embeds = disco_gather(image_embeds, self.ctx)
            gathered_text_embeds = disco_gather(text_embeds, self.ctx)

            logits_per_text = torch.matmul(gathered_text_embeds, gathered_image_embeds.t().to(device))

            logit_scale = self.model.logit_scale.to(device)
            logit_bias = self.model.logit_bias.to(device)
            logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

            logits_per_image = logits_per_text.t()

            eye = torch.eye(logits_per_text.size(0), device=device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
            return loss
        else:
            return outputs.loss

    def _packing(self):
        pass

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

    def forward(self, images, texts, use_attn_mask=True):
        device = torch.cuda.current_device()
        inputs = self.processor(images=images, text=texts, padding="longest", return_tensors="pt")
        inputs = self.to_cuda(inputs, device)

        if use_attn_mask:
            attention_mask = (inputs["input_ids"] != self.processor.tokenizer.pad_token_id).long()
            inputs["attention_mask"] = attention_mask.to(device)
        outputs = self.model(**inputs)
        loss = self.calcul_loss(outputs)

        text_output = outputs.text_model_output
        image_output = outputs.vision_model_output

        text_hidden_embeds = text_output.last_hidden_state
        image_hidden_embeds = image_output.last_hidden_state

        batch_size = image_hidden_embeds.shape[0]
        assert text_hidden_embeds.shape[0] == image_hidden_embeds.shape[0]

        if self.text_decoder is not None:
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = self.image_proj(image_embeds)
            text_embeds = self.text_proj(text_embeds)
            image_end_embeds = self.image_end_embed.repeat(batch_size, 1, 1)
            embeds = torch.concat((image_embeds, image_end_embeds.to(text_embeds.dtype), text_embeds), dim=1)

            num_images = image_embeds.shape[1]
            num_texts = text_embeds.shape[1]
            num_tokens = num_images + 1 + num_texts
            image_mask = torch.ones(size=(num_images + 1, num_tokens), dtype=torch.bool)
            text_mask = torch.zeros(size=(num_texts, num_tokens), dtype=torch.bool)
            mask = torch.concat([image_mask, text_mask], dim=0)
            tril_mask = torch.tril(torch.ones(size=(num_tokens, num_tokens), dtype=torch.bool), diagonal=-1)
            attention_mask = (tril_mask | mask).long()

            token_is_pad = (inputs["input_ids"] == self.tokenizer.pad_token_id)
            pad_start_indices = torch.where(
                token_is_pad.any(dim=1),
                token_is_pad.long().argmax(dim=1),
                -torch.ones(size=(batch_size, ), dtype=torch.int64)
            ).unbind()
            loss_mask_list = list()
            for idx, start_index in enumerate(pad_start_indices):
                loss_mask = torch.ones(size=(num_tokens, ), dtype=torch.int64)
                loss_mask[:start_index] = 0
                loss_mask_list.append(loss_mask)
            loss_mask = torch.stack(loss_mask_list, dim=0)
            embeds = self.text_decoder(embeds, attention_mask=attention_mask)
            embeds = self.vocab_proj(embeds)
            self.calcul_regression_loss(embeds, inputs["input_ids"], loss_mask)
            return outputs, loss

        return outputs, Context(
            loss=loss,
            total_image_num_tokens=np.prod(image_hidden_embeds.shape[:2]).item(),
            total_text_num_tokens=np.prod(text_hidden_embeds.shape[:2]).item(),
            total_num_samples=batch_size,
            total_text_num_valid_tokens=(inputs["input_ids"] != self.tokenizer.pad_token_id).long().sum().item()
        )
