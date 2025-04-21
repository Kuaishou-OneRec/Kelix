import os
import os.path as osp
import torch
import deepspeed
from PIL import Image, ImageOps
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from transformers import AutoProcessor, AutoModel
from recipes.ViT.training.models.MoonVIT.modeling_moonvit import MoonVitPretrainedModel
from recipes.ViT.training.models.MoonVIT.image_processing_moonvit import MoonViTImageProcessor
import torch.nn.functional as F
from recipes.ViT.training.models.siglip.modeling_siglip import SiglipPreTrainedModel, SiglipModel
from recipes.ViT.training.models.siglip.processing_siglip import SiglipProcessor
from recipes.ViT.training.models.vivit.vivit_utils import read_video_pyav, read_image_pil, sample_frame_indices
from PIL.Image import Resampling as PILImageResampling
from transformers.image_transforms import resize
import transformers
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


class MoonViT(nn.Module):

    def __init__(self, config, ctx):
        super().__init__()
        self.config = config
        self.ctx = ctx
        self.is_dist = self.ctx.is_dist

        self.text_model = SiglipModel.from_pretrained(
            config.dir, ignore_mismatched_sizes=True
        )
        self.image_processor = VivitImageProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/vivit-b-16x2-kinetics400")
        self.image_model = VivitModel.from_pretrained(
            "/llm_reco_ssd/zhouyang12/models/vivit-b-16x2-kinetics400",
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
        self.text_embed_proj = nn.Linear(1152, 768, bias=False)

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
            return outputs.loss

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
        text_embeds = self.text_embed_proj(text_embeds)
        processed_images = []
        for image in images:
            frame = np.array(image)
            frame = resize(frame, size=(224, 224), resample=PILImageResampling.BILINEAR)
            processed_images.append([frame])

        videos = [np.concatenate([frame[np.newaxis, ...] for frame in video],axis=0) for video in processed_images]
        extended_videos = []
        for video in videos:
            indices = sample_frame_indices(clip_len=4, frame_sample_rate=4, seg_len=video.shape[0])
            video = read_image_pil(video,indices)
            video = list(video)
            extended_videos.append(video)
        image_inputs = self.image_processor(extended_videos, return_tensors="pt").to(text_inputs.input_ids.device)
        image_inputs = self.to_cuda(image_inputs, device)
        image_inputs = self.to_cuda(image_inputs, torch.bfloat16)
        image_outputs = self.image_model(**image_inputs)
        image_embeds = image_outputs.last_hidden_state
        pooler = image_outputs.pooler_output
        loss = self.calcul_loss(text_embeds, pooler)

        text_output = text_outputs
        image_output = image_outputs

        text_hidden_embeds = text_output.last_hidden_state
        image_hidden_embeds = image_output.last_hidden_state

        batch_size = image_hidden_embeds.shape[0]
        assert text_hidden_embeds.shape[0] == image_hidden_embeds.shape[0]


        return pooler, text_embeds, Context(
            loss=loss,
            total_image_num_tokens=np.prod(image_hidden_embeds.shape[:2]).item(),
            total_text_num_tokens=np.prod(text_hidden_embeds.shape[:2]).item(),
            total_num_samples=batch_size,
            total_text_num_valid_tokens=(text_inputs.input_ids != self.tokenizer.pad_token_id).long().sum().item()
        )
