raise NotImplementedError
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
from transformers import AutoProcessor, AutoModel
from .siglip.modeling_siglip import SiglipPreTrainedModel, SiglipModel
from .siglip.processing_siglip import SiglipProcessor
import torch.nn.functional as F
import logging
from torch.nn.utils.rnn import pad_sequence
from recipes.ViT.helpers.context import Context

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


class PackingBuffer(object):

    def __init__(self, max_len, buffer_idx=-1, drop_ratio=0., patch_size=14):
        self.buffer_idx = buffer_idx
        self.drop_ratio = drop_ratio
        self.patch_size = patch_size

        self.max_len = max_len
        self.cur_len = 0
        self.images = list()
        self.sample_cnt = 0

        self.sample_indices = list()
        self.image_indices = list()

        self.height_positions = list()
        self.width_positions = list()
        self.positions = list()

        self.cu_seqlens = [0]
        self.texts = list()
        self.indices = list()

    def reset(self):
        self.images = list()
        self.cur_len = 0
        self.sample_indices = list()
        self.image_indices = list()
        self.height_positions = list()
        self.width_positions = list()
        self.positions = list()
        self.cu_seqlens = [0]
        self.texts = list()
        self.indices = list()
        self.sample_cnt = 0

    def can_append(self, n_token_after_drop):
        return self.cur_len + n_token_after_drop <= self.max_len

    def append(self, img_list, texts, n_token_after_drop):
        sample_idx = self.buffer_idx
        self.images.extend(img_list)
        self.cur_len += n_token_after_drop
        self.texts.append(texts)

        image_indices = list()

        total_token = 0
        positions_list = list()
        height_positions_list = list()
        width_positions_list = list()

        for idx, img in enumerate(img_list):
            width, height = img.size
            img_n_token = (width // self.patch_size) * (height // self.patch_size)
            img_n_token_after_drop = int(img_n_token * (1. - self.drop_ratio))
            total_token += img_n_token_after_drop

            image_indices = torch.LongTensor([idx for _ in range(img_n_token_after_drop)])

            positions = torch.arange(img_n_token)
            indices = torch.randperm(img_n_token)[:img_n_token_after_drop].sort().values
            positions = positions[indices]
            
            height_positions = positions // width
            width_positions = positions % width

            self.image_indices.append(image_indices)
            self.positions.append(positions)
            self.height_positions.append(height_positions)
            self.width_positions.append(width_positions)
            self.sample_indices.append(torch.LongTensor([self.sample_cnt for _ in range(img_n_token_after_drop)]))
            self.indices.append(sample_idx)

        assert total_token == n_token_after_drop

        self.cu_seqlens.append(self.cu_seqlens[-1] + total_token)
        self.sample_cnt += 1


class KimiViT(nn.Module):

    def __init__(self, config, ctx):
        super().__init__()
        self.config = config
        self.ctx = ctx
        self.is_dist = self.ctx.is_dist
        self.use_packing = config.packing.enabled
        self.packing_max_length = config.packing.max_length
        self.packing_drop_ratio = config.packing.drop_ratio

        self.model = SiglipModel.from_pretrained(config.dir, ignore_mismatched_sizes=True)
        self.processor = SiglipProcessor.from_pretrained(config.dir)
        self.tokenizer = self.processor.tokenizer

        hidden_size = self.model.hidden_size
        vocab_size = self.model.vocab_size
        self.patch_size = self.model.patch_size

        # if self.use_packing:
        #     self.padding_image_embedding = nn.Parameter(
        #         torch.zeros(size=(1, 3, self.patch_size, self.patch_size), dtype=torch.float32),
        #         requires_grad=True
        #     )

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

    def calcul_image_tokens(self, images, texts):
        """
        计算图片的token数
        """

        num_token_list = list()
        for image_idx, image_obj in enumerate(images):
            if not isinstance(image_obj, (list, tuple)):
                image_obj = [image_obj]
            num_token = 0
            for image in image_obj:
                width, height = image.size
                assert width % self.patch_size == 0 and height % self.patch_size == 0, image.size
                num_token += (width // self.patch_size) * (height // self.patch_size)

            num_token_list.append(num_token)
            images[image_idx] = image_obj

        data = list(zip(images, num_token_list, texts))
        data.sort(key=lambda x: -x[1])
        images, num_token_list, texts = zip(*data)

        return images, num_token_list, texts if texts[0] is not None else None

    def _packing(self, images, texts):
        """
        贪心调度，根据token数从大到小排序，每次从buffer队列里寻找剩余空间最大的buffer分配，如果没有，重新分配一个buffer
        """
        assert images is not None and texts is not None
        images, num_token_list, texts = self.calcul_image_tokens(images, texts)

        buffers = list()
        for idx, (image_list, num_token, text) in enumerate(zip(images, num_token_list, texts)):
            num_token_after_drop = int(num_token * (1. - self.packing_drop_ratio))
            assert num_token_after_drop <= self.packing_max_length
            can_append_buffer_list = [
                buffer for buffer in buffers if buffer.can_append(num_token_after_drop)
            ]
            if can_append_buffer_list:
                buffer = min(can_append_buffer_list, key=lambda x: x.cur_len)
            else:
                buffer = PackingBuffer(
                    max_len=self.packing_max_length,
                    buffer_idx=len(buffers),
                    drop_ratio=self.packing_drop_ratio,
                    patch_size=self.patch_size
                )
                buffers.append(buffer)
            buffer.append(image_list, text, num_token_after_drop)

        texts = list()
        images = list()

        sample_indices = list()
        image_indices = list()
        indices = list()

        position_ids = list()
        height_position_ids = list()
        width_position_ids = list()

        cu_seqlens = list()

        for buffer in buffers:
            images.extend(buffer.images)
            texts.extend(buffer.texts)
            sample_indices.extend(buffer.sample_indices)
            image_indices.extend(buffer.image_indices)
            position_ids.extend(buffer.positions)
            height_position_ids.extend(buffer.height_positions)
            width_position_ids.extend(buffer.width_positions)
            cu_seqlens.append(buffer.cu_seqlens)
            indices.extend(buffer.indices)

        length_list = [len(images), len(texts), len(sample_indices), len(image_indices), len(position_ids), len(height_position_ids), len(width_position_ids), len(indices)]
        assert len(set(length_list)) == 1, length_list

        return {
            "images": images,
            "texts": texts,
            "sample_indices": sample_indices,
            "image_indices": image_indices,
            "position_ids": position_ids,
            "height_position_ids": height_position_ids,
            "width_position_ids": width_position_ids,
            "cu_seqlens": cu_seqlens,
            "indices": indices
        }

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

    def build_image_mask(self, info_dict):
        sample_indices = info_dict["sample_indices"]
        padding_mask = (sample_indices < 0).long()
        attn_mask = (sample_indices[:, None, :] == sample_indices[:, :, None])
        attn_mask = attn_mask[:, None, :, :].long()
        return attn_mask, padding_mask

    def merge_images_inputs(self, images_inputs_list, packing_info_dict):
        patch_size = self.patch_size
        images = packing_info_dict["images"]
        texts = packing_info_dict["texts"]
        sample_indices = packing_info_dict["sample_indices"]
        image_indices = packing_info_dict["image_indices"]
        position_ids = packing_info_dict["position_ids"]
        width_position_ids = packing_info_dict["width_position_ids"]
        height_position_ids = packing_info_dict["height_position_ids"]
        cu_seqlens = packing_info_dict["cu_seqlens"]
        indices = packing_info_dict["indices"]

        merged_images_dict = dict()

        for processed_image, sample_idx, image_idx, positions, width_positions, height_positions, idx in zip(
                images_inputs_list,
                sample_indices,
                image_indices,
                position_ids,
                width_position_ids,
                height_position_ids,
                indices
        ):
            if idx not in merged_images_dict:
                merged_images_dict[idx] = {
                    "pixel_values": list(),
                    "sample_indices": list(),
                    "image_indices": list(),
                    "position_ids": list(),
                    "width_position_ids": list(),
                    "height_position_ids": list(),
                    "cu_seqlens": cu_seqlens[idx]
                }
            image = processed_image["pixel_values"]
            assert image.dim() in [3, 4]
            if image.dim() == 3:
                image = rearrange(image, "c (h p1) (w p2) -> (h w) c p1 p2", p1=patch_size, p2=patch_size)
            else:
                assert image.shape[0] == 1
                image = image.squeeze(0)
                image = rearrange(image, "c (h p1) (w p2) -> (h w) c p1 p2", p1=patch_size, p2=patch_size)
            # patches = torch.gather(image, 0, positions[:, None, None, None].repeat(1, 3, patch_size, patch_size))
            patches = image[positions]
            merged_images_dict[idx]["pixel_values"].append(patches)
            merged_images_dict[idx]["sample_indices"].append(sample_idx)
            merged_images_dict[idx]["image_indices"].append(image_idx)
            merged_images_dict[idx]["position_ids"].append(positions)
            merged_images_dict[idx]["width_position_ids"].append(width_positions)
            merged_images_dict[idx]["height_position_ids"].append(height_positions)

        merged_patches = list()
        merged_sample_indices = list()
        merged_image_indices = list()
        merged_position_ids = list()
        merged_height_position_ids = list()
        merged_width_position_ids = list()
        merged_cu_seqlens = list()

        for idx in sorted(merged_images_dict.keys()):
            patches = torch.concat(merged_images_dict[idx]["pixel_values"], dim=0)
            sample_indices = torch.concat(merged_images_dict[idx]["sample_indices"], dim=0)
            image_indices = torch.concat(merged_images_dict[idx]["image_indices"], dim=0)
            position_ids = torch.concat(merged_images_dict[idx]["position_ids"], dim=0)
            width_position_ids = torch.concat(merged_images_dict[idx]["width_position_ids"], dim=0)
            height_position_ids = torch.concat(merged_images_dict[idx]["height_position_ids"], dim=0)

            merged_patches.append(patches)
            merged_sample_indices.append(sample_indices)
            merged_image_indices.append(image_indices)
            merged_position_ids.append(position_ids)
            merged_height_position_ids.append(height_position_ids)
            merged_width_position_ids.append(width_position_ids)
            merged_cu_seqlens.append(merged_images_dict[idx]["cu_seqlens"])

        max_length = max([_patches.shape[0] for _patches in merged_patches])
        tmp_merged_patches = list()
        dtype = merged_patches[0].dtype
        dim = merged_patches[0].shape[-1]
        for _patches in merged_patches:
            padding_length = max_length - _patches.shape[0]
            padding = _patches.new_zeros(size=(padding_length, 3, patch_size, patch_size))
            padding_pathes = torch.concat([_patches, padding], dim=0)
            tmp_merged_patches.append(padding_pathes)
        merged_patches = torch.stack(tmp_merged_patches, dim=0)
        merged_sample_indices = pad_sequence(merged_sample_indices, batch_first=True, padding_value=-1)
        merged_image_indices = pad_sequence(merged_image_indices, batch_first=True, padding_value=-1)
        merged_positions = pad_sequence(merged_position_ids, batch_first=True, padding_value=0)
        merged_height_positions = pad_sequence(merged_height_position_ids, batch_first=True, padding_value=0)
        merged_width_positions = pad_sequence(merged_width_position_ids, batch_first=True, padding_value=0)

        merged_info_dict = {
            "pixel_values": merged_patches,
            "sample_indices": merged_sample_indices,
            "image_indices": merged_image_indices,
            "image_position_ids": merged_positions,
            "height_position_ids": merged_height_positions,
            "width_position_ids": merged_width_positions,
            "cu_seqlens": merged_cu_seqlens
        }
        attn_mask, padding_mask = self.build_image_mask(merged_info_dict)
        merged_info_dict["image_attention_mask"] = attn_mask.float()
        merged_info_dict["padding_mask"] = padding_mask.float()

        return merged_info_dict

    def forward(self, images, texts, use_attn_mask=True):
        device = torch.cuda.current_device()
        if self.use_packing:
            package = self._packing(images, texts)

            images_inputs_list = list()
            for images_list in package["images"]:
                processed_images = self.processor(images=images_list, return_tensors="pt", do_resize=(not self.use_packing))
                images_inputs_list.append(processed_images)
            merged_info_dict = self.merge_images_inputs(images_inputs_list, package)

            inputs = self.processor(text=texts, padding="longest", return_tensors="pt")
            inputs.update(merged_info_dict)
        else:
            inputs = self.processor(images=images, text=texts, padding="longest", return_tensors="pt",
                                    do_resize=(not self.use_packing))
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
        assert text_hidden_embeds.shape[0] == image_hidden_embeds.shape[0], (text_hidden_embeds.shape, image_hidden_embeds.shape)

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
                -torch.ones(size=(batch_size,), dtype=torch.int64)
            ).unbind()
            loss_mask_list = list()
            for idx, start_index in enumerate(pad_start_indices):
                loss_mask = torch.ones(size=(num_tokens,), dtype=torch.int64)
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
