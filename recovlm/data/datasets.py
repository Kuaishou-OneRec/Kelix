from typing import Union, Iterable, Optional, List, Dict, Tuple, Any
import logging

import os
import sys
import re
import wids
import json
import time
import traceback
import pickle
import random
import base64
import pyarrow.parquet as pq
from datetime import datetime

import webdataset as wds

from io import BytesIO
from PIL import Image

from collections import defaultdict

import multiprocessing
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset, DataLoader

from transformers import AutoTokenizer, AutoProcessor, \
    PreTrainedTokenizer, PreTrainedTokenizerFast

from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from recovlm.utils.qwen_vl_utils import process_vision_info
from recovlm.utils.common import shell_hdfs_ls, pytorch_worker_info

from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_world_size
from recovlm.utils.common import print_rank_0, Timer

import glob

from .templates import get_template
from .prompts import PromptLoader

logger = logging.getLogger(__name__)

RESPONSE_TEMPLATE = "{% for message in messages %}{{message['content'] + '<|im_end|>'}}{% endfor %}"

def zero_pad_sequences(sequences, side: str = "left", value=0):
  assert side in ("left", "right")
  max_len = max(seq.size(-1) for seq in sequences)
  padded_sequences = []
  for seq in sequences:
    pad_len = max_len - seq.size(-1)
    padding = (pad_len, 0) if side == "left" else (0, pad_len)
    padded_sequences.append(F.pad(seq, padding, value=value))
  return torch.stack(padded_sequences, dim=0)

# TODO: use IterableDataset
class ChatCompletionDataset(Dataset):
  """Text Completion Dataset with ChatML format"""
  def __init__(self,
               source: Union[str, Iterable],
               tokenizer: \
                Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str],
               input_key: str = "messages",
               role_key: str = "role",
               content_key: str = "content",
               system_name: str = "system",
               user_name: str = "user",
               assistant_name: str = "assistant", 
               system_prompt: str = "You are a helpful assistant.",
               chat_template: str = "chat_template",
               file_format: str = "jsonl",
               max_length: Optional[int] = None):
    super(ChatCompletionDataset).__init__()
    self.source = source
    if isinstance(tokenizer, str):
      tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    self.tokenizer = tokenizer
    self.input_key = input_key
    self.role_key = role_key
    self.content_key = content_key
    self.role_name_mappings = {
      system_name: "system",
      user_name: "user",
      assistant_name: "assistant", 
    }
    self.format = file_format
    self.system_prompt = PromptLoader().load(system_prompt)
    self.chat_template = get_template(chat_template)
    self.records = []
    if isinstance(source, str):
      # TODO: support parquet, support hdfs
      if self.format == "jsonl":
        with open(self.source, encoding="utf-8") as f:
          for line in f:
            self.records.append(json.loads(line))
      elif self.format == "json":
        with open(self.source, encoding="utf-8") as f:
          self.records = json.loads(f.read())
      else:
        raise NotImplementedError()
    else:
      self.records = source
    self.max_length = max_length

  def __len__(self):
    return len(self.records)

  def rename(self, messages):
    new_messages = []
    for message in messages:
      new_message = {}
      new_message["role"] = \
        self.role_name_mappings[message[self.role_key]]
      new_message["content"] = message[self.content_key]
      new_messages.append(new_message)
    return new_messages

  def __getitem__(self, index):
    messages = self.rename(self.records[index][self.input_key])
    if self.system_prompt:
      if messages[0]["role"] == "system":
        messages[0]["content"] = self.system_prompt
      else:
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + messages
    tokenized = self.tokenizer.apply_chat_template(
        [messages],
        chat_template=self.chat_template,
        return_assistant_tokens_mask=True,
        return_dict=True
    )
    tokenized["loss_mask"] = tokenized.pop("assistant_masks")
    #TODO: improve truncation strategy
    if self.max_length:
      for key in list(tokenized.keys()):
        tokenized[key] = tokenized[key][0][:self.max_length]
    return tokenized

  def collate_fn(self, items):
    all_input_ids = [torch.tensor(item["input_ids"]) for item in items]
    all_attention_mask = [
        torch.tensor(
            item["attention_mask"]) for item in items]
    all_loss_mask = [torch.tensor(item["loss_mask"]) for item in items]
    batch = {}
    batch["input_ids"] = zero_pad_sequences(
        all_input_ids, "right", self.tokenizer.pad_token_id)
    batch["attention_mask"] = zero_pad_sequences(
        all_attention_mask, "right")
    batch["loss_mask"] = zero_pad_sequences(
        all_loss_mask, "right")
    return batch

def get_rope_index(
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        spatial_merge_size: Optional[int] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        vision_start_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

  Explanation:
      Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

      For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
      Examples:
          input_ids: [T T T T T], here T is for text.
          temporal position_ids: [0, 1, 2, 3, 4]
          height position_ids: [0, 1, 2, 3, 4]
          width position_ids: [0, 1, 2, 3, 4]

      For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
      and 1D rotary position embeddin for text part.
      Examples:
          Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
          input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
          vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
          vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
          vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
          text temporal position_ids: [3, 4, 5, 6, 7]
          text height position_ids: [3, 4, 5, 6, 7]
          text width position_ids: [3, 4, 5, 6, 7]
          Here we calculate the text start position_ids as the max vision position_ids plus 1.

  Args:
      input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
          Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
          it.
      image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
          The temporal, height and width of feature shape of each image in LLM.
      video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
          The temporal, height and width of feature shape of each video in LLM.
      attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
          Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

          - 1 for tokens that are **not masked**,
          - 0 for tokens that are **masked**.

  Returns:
      position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
      mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
  """
  # spatial_merge_size = self.config.vision_config.spatial_merge_size
  # image_token_id = self.config.image_token_id
  # video_token_id = self.config.video_token_id
  # vision_start_token_id = self.config.vision_start_token_id
  mrope_position_deltas = []
  if input_ids is not None and (
          image_grid_thw is not None or video_grid_thw is not None):
    total_input_ids = input_ids
    if attention_mask is None:
      attention_mask = torch.ones_like(total_input_ids)
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device)
    image_index, video_index = 0, 0
    for i, input_ids in enumerate(total_input_ids):
      input_ids = input_ids[attention_mask[i] == 1]
      image_nums, video_nums = 0, 0
      vision_start_indices = torch.argwhere(
          input_ids == vision_start_token_id).squeeze(1)
      vision_tokens = input_ids[vision_start_indices + 1]
      image_nums = (vision_tokens == image_token_id).sum()
      video_nums = (vision_tokens == video_token_id).sum()
      input_tokens = input_ids.tolist()
      llm_pos_ids_list: list = []
      st = 0
      remain_images, remain_videos = image_nums, video_nums
      for _ in range(image_nums + video_nums):
        if image_token_id in input_tokens and remain_images > 0:
          ed_image = input_tokens.index(image_token_id, st)
        else:
          ed_image = len(input_tokens) + 1
        if video_token_id in input_tokens and remain_videos > 0:
          ed_video = input_tokens.index(video_token_id, st)
        else:
          ed_video = len(input_tokens) + 1
        if ed_image < ed_video:
          t, h, w = (
              image_grid_thw[image_index][0],
              image_grid_thw[image_index][1],
              image_grid_thw[image_index][2],
          )
          image_index += 1
          remain_images -= 1
          ed = ed_image
        else:
          t, h, w = (
              video_grid_thw[video_index][0],
              video_grid_thw[video_index][1],
              video_grid_thw[video_index][2],
          )
          video_index += 1
          remain_videos -= 1
          ed = ed_video
        llm_grid_t, llm_grid_h, llm_grid_w = (
            t.item(),
            h.item() // spatial_merge_size,
            w.item() // spatial_merge_size,
        )
        text_len = ed - st

        st_idx = llm_pos_ids_list[-1].max() + \
            1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(torch.arange(
            text_len).view(1, -1).expand(3, -1) + st_idx)

        t_index = torch.arange(llm_grid_t).view(-1,
                                                1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(
            1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(
            1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack(
            [t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

      if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + \
            1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(torch.arange(
            text_len).view(1, -1).expand(3, -1) + st_idx)

      llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
      position_ids[..., i, attention_mask[i] ==
                   1] = llm_positions.to(position_ids.device)
      mrope_position_deltas.append(
          llm_positions.max() + 1 - len(total_input_ids[i]))
    mrope_position_deltas = torch.tensor(
        mrope_position_deltas,
        device=input_ids.device).unsqueeze(1)
    return position_ids
  else:
    if attention_mask is not None:
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      position_ids = position_ids.unsqueeze(
          0).expand(3, -1, -1).to(input_ids.device)
      max_position_ids = position_ids.max(0, keepdim=False)[
          0].max(-1, keepdim=True)[0]
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

    return position_ids

class ImageTextPairDatasetWithPacking(IterableDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               max_length: int = 1024,
               min_visual_tokens: int = 4,
               max_visual_tokens: int = 512,
               shrink_ratio: float = 0.9,
               max_retry: int = 5,
               multiple_of: int = 8,
               data_format: str = "chatml",
               shuffle_size: int = 100000,
               base_model_dir: Optional[str] = None,
               processor: Optional[Qwen2VLProcessor] = None,
               spatial_merge_size: int = 2,
               patch_size: int = 14,
               image_token_id: int = 151655,
               video_token_id: int = 151656,
               vision_start_token_id: int = 151652):
    super(ImageTextPairDatasetWithPacking).__init__()
    if base_model_dir:
      processor = Qwen2VLProcessor.from_pretrained(base_model_dir)
      model_config = Qwen2VLConfig.from_pretrained(base_model_dir)
      spatial_merge_size = model_config.vision_config.spatial_merge_size
      patch_size = model_config.vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id

    self.processor = processor
    self.max_length = max_length
    self.min_visual_tokens = min_visual_tokens
    self.max_visual_tokens = max_visual_tokens
    self.patch_size = patch_size
    self.shrink_ratio = shrink_ratio
    self.max_retry = max_retry
    self.spatial_merge_size = spatial_merge_size
    self.image_token_id = image_token_id
    self.video_token_id = video_token_id
    self.vision_start_token_id = vision_start_token_id
    self.patch_size = patch_size
    # Pad sequence to multiple of `multiple_of`
    self.multiple_of = multiple_of
    self.data_format = data_format
    self.total_samples = 0
    urls = []
    if isinstance(sources, str):
      sources = sources.split(",")
    for source in sources:
      with open(source, encoding="utf-8") as f:
        index = json.loads(f.read())["shardlist"]
        for item in index:
          urls.append(os.path.join(os.path.dirname(source), item["url"]))
          self.total_samples += item["nsamples"]

    dataset = wds.WebDataset(
        urls,
        handler=wds.warn_and_continue,
        resampled=True,
        shardshuffle=True,
        cache_dir="/tmp/_wids_cache",
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker
    )

    dataset = dataset.shuffle(shuffle_size).decode(
      "pil", handler=wds.warn_and_continue)
    
    self.dataset = dataset

  def _may_filter(self, sample):
    image = sample["jpg"]
    # caption = sample[".txt"]
    width, height = image.size
    if max(height, width) / min(height, width) > 10:
      raise ValueError("Too larged aspect ratio, skip samples")
    if (sample["json"].get("clip_similarity_vitl14", 0.0) > 0.3):
      raise ValueError("Too low clip score")

  def _process_chat(self,
                    sample: Dict[str, Union[str, Image.Image]],
                    max_visual_tokens: int = 1280):
    max_visual_tokens = max(max_visual_tokens, self.max_visual_tokens)
    image = sample["jpg"]
    caption = sample["txt"]
    if image.mode != "RGB":
      image = image.convert("RGB")
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.min_visual_tokens * (self.patch_size ** 2) * (self.spatial_merge_size ** 2),
                    "max_pixels": max_visual_tokens * (self.patch_size ** 2) * (self.spatial_merge_size ** 2)
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = self.processor.apply_chat_template(
        [prompt], tokenize=False, add_generation_prompt=True
    )
    # TODO: datacamp的aspect_ratio过大会触发异常，提前处理或丢掉？
    image_inputs, video_inputs = process_vision_info(prompt)

    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    resposne = [{"content": caption}]
    response_ids = self.processor.tokenizer.apply_chat_template(
      [resposne],
      chat_template=get_template("chat_template_response_only"),
      add_generation_prompt=False,
      return_tensors="pt"
    )

    response_mask = (
      response_ids != self.processor.tokenizer.pad_token_id).type(torch.int64)
    loss_mask = torch.cat(
      [torch.zeros_like(inputs["input_ids"]), response_mask], dim=-1
    )
    inputs["input_ids"] = torch.cat(
        [inputs["input_ids"], response_ids], dim=-1)
    inputs["loss_mask"] = loss_mask
    inputs["position_ids"] = get_rope_index(
        inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        spatial_merge_size=self.spatial_merge_size,
        image_token_id=self.image_token_id,
        video_token_id=self.video_token_id,
        vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs

  def _process_completion(self,
                          sample: Dict[str, Union[str, Image.Image]],
                          max_visual_tokens: int = 128) -> Dict[str, torch.Tensor]:
    max_visual_tokens = max(max_visual_tokens, self.min_visual_tokens)
    image = sample["jpg"]
    caption = sample["txt"]
    if image.mode != "RGB":
      image = image.convert("RGB")

    # TODO: fix hard code
    text = "<|vision_start|><|image_pad|><|vision_end|>"

    image_inputs, video_inputs = process_vision_info([
        {
          "role": "user",
          "content": [
            {
              "type": "image",
              "image": image,
              "min_pixels": self.min_visual_tokens * (self.patch_size ** 2) * (self.spatial_merge_size ** 2),
              "max_pixels": max_visual_tokens * (self.patch_size ** 2) * (self.spatial_merge_size ** 2)
            },
            {"type": "text", "text": "Describe this image."},
          ],
        }
      ]
    )

    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    resposne = caption + self.processor.tokenizer.eos_token
    response_ids = self.processor.tokenizer.encode(
      resposne,
      return_tensors="pt"
    )

    response_mask = (
      response_ids != self.processor.tokenizer.pad_token_id).type(torch.int64)
    loss_mask = torch.cat(
      [torch.zeros_like(inputs["input_ids"]), response_mask], dim=-1
    )
    inputs["input_ids"] = torch.cat(
        [inputs["input_ids"], response_ids], dim=-1)
    inputs["loss_mask"] = loss_mask
    inputs["position_ids"] = get_rope_index(
        inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        spatial_merge_size=self.spatial_merge_size,
        image_token_id=self.image_token_id,
        video_token_id=self.video_token_id,
        vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs

  def _process(self, sample):
    # self._may_filter(sample)
    max_visual_tokens = self.max_visual_tokens
    for retry in range(self.max_retry):
      if self.data_format == "chatml":
        inputs = self._process_chat(sample, max_visual_tokens)
      elif self.data_format == "completion":
        inputs = self._process_completion(sample, max_visual_tokens)
      else:
        raise NotImplementedError(f"Unsupported dataset format `{self.format}`")
      if not inputs:
        raise ValueError("Empty inputs, skip")
      if inputs["input_ids"].shape[-1] > self.max_length:
        max_visual_tokens = (max_visual_tokens * self.shrink_ratio)
        continue
      else:
        assert inputs["input_ids"].shape[-1] <= self.max_length, "inputs too long"
        return inputs
    else:
      raise ValueError(
        f"Unable to generate sample within max_length={self.max_length} after {retry} retrys"
      )
  
  def _packing(self, buffer: List[Dict[str, torch.Tensor]]):
    packed_input_ids = []
    packed_position_ids = []
    packed_loss_mask = []
    packed_pixel_values = []
    packed_image_gird_thw = []
    cu_seqlens = [0]

    for inputs in buffer:
      packed_input_ids.append(inputs["input_ids"].flatten())
      packed_loss_mask.append(inputs["loss_mask"].flatten())
      packed_position_ids.append(inputs["position_ids"])
      packed_pixel_values.append(inputs["pixel_values"])
      packed_image_gird_thw.append(inputs["image_grid_thw"])
      cu_seqlens.append(cu_seqlens[-1] + len(inputs["input_ids"][0]))

    packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
    packed_loss_mask = torch.cat(packed_loss_mask, dim=0).unsqueeze(0)
    packed_position_ids = torch.cat(packed_position_ids, dim=-1)
    packed_pixel_values = torch.cat(packed_pixel_values, dim=0)
    packed_image_gird_thw = torch.cat(packed_image_gird_thw, dim=0)
    
    # pad to multiple of, necessary for sequence parallel
    if (
      self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
    ):  # not divisible by multiple_of; here we align for grouping
      padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
      packed_input_ids = F.pad(
        packed_input_ids, (0, padding_len), value=self.processor.tokenizer.pad_token_id)
      packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
      packed_loss_mask = F.pad(packed_loss_mask, (0, padding_len), value=0)
      cu_seqlens.append(cu_seqlens[-1] + padding_len)
 
    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "pixel_values": packed_pixel_values,
      "image_grid_thw": packed_image_gird_thw,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32)
    }
    return inputs

  def __iter__(self):
    buffer = []
    cur_length = 0
    for sample in self.dataset:
      try:
        inputs = self._process(sample)
      except:
        print(traceback.format_exc())
        continue
      sample_length = inputs["input_ids"].shape[-1]
      if cur_length + sample_length > self.max_length:
        packed_inputs = self._packing(buffer)
        yield packed_inputs
        buffer = [inputs]
        cur_length = sample_length
      else:
        buffer.append(inputs)
        cur_length += sample_length

def get_assistant_mask(batch_input_ids: torch.Tensor,
                       start_pattern: Optional[List[int]],
                       end_pattern: Optional[List[int]]):
  if not start_pattern:
    start_pattern = [151644, 77091, 198]
  if not end_pattern:
    end_pattern = [151645, 198]

  masks = []
  for input_ids in batch_input_ids:
    mask = []
    assistant_start = []
    assistant_end = []
    to_mask = False
    for _id in input_ids:
      mask.append(int(to_mask))
      if not to_mask:
        if _id in start_pattern:
          assistant_start.append(_id.item())
        else:
          assistant_start = []
        if assistant_start[-3:] == start_pattern:
          to_mask = True
          assistant_start = []
      else:
        if _id in end_pattern:
          assistant_end.append(_id.item())
        else:
          assistant_end = []
        if assistant_end[-2:] == end_pattern:
          to_mask = False
          assistant_end = []
    masks.append(mask)
  return torch.tensor(masks)


class ChatCompletionVisionDataset(IterableDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               max_length: int = 1024,
               min_visual_tokens_per_image: int = 4,
               max_visual_tokens_per_image: int = 512,
               video_nframe: int = -1,
               video_fps: float = 2.0,
               video_min_frames: int = -1,
               video_max_frames: int = 120,
               shrink_ratio: float = 0.9,
               max_retry: int = 5,
               multiple_of: int = 8,
               shuffle_size: int = 100000,
               shuffle_initial_size: int = 20000,
               base_model_dir: Optional[str] = None,
               processor: Optional[Qwen2VLProcessor] = None,
               spatial_merge_size: int = 2,
               patch_size: int = 14,
               image_token_id: int = 151655,
               video_token_id: int = 151656,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {}):
    """
    datasource_config: 默认覆盖全局配置
                      key: datasource_name
                      Dict: datasource config, support params:
                        min_visual_tokens_per_image
                        max_visual_tokens_per_image
                        video_nframe
                        video_fps
                        video_min_frames
                        video_max_frames
    """
    if base_model_dir:
      processor = Qwen2VLProcessor.from_pretrained(base_model_dir)
      model_config = Qwen2VLConfig.from_pretrained(base_model_dir)
      spatial_merge_size = model_config.vision_config.spatial_merge_size
      patch_size = model_config.vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    self.processor = processor
    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.video_nframe = video_nframe
    self.video_fps = video_fps
    self.video_min_frames = video_min_frames
    self.video_max_frames = video_max_frames
    if video_nframe > 0 and (video_fps > 0 or video_min_frames > 0 or video_max_frames > 0):
      logger.warning(
        f"ChatCompletionVisionDataset(video_fps=...): video_fps, video_min_frames, "\
          f"video_max_frames will be ignored when video_nframe>0 ({video_nframe=})"
      )
    self.patch_size = patch_size
    self.shrink_ratio = shrink_ratio
    self.max_retry = max_retry
    self.spatial_merge_size = spatial_merge_size
    self.image_token_id = image_token_id
    self.video_token_id = video_token_id
    self.vision_start_token_id = vision_start_token_id
    self.vision_end_token_id = vision_end_token_id
    self.pad_token_id = pad_token_id
    self.patch_size = patch_size
    # Pad sequence to multiple of `multiple_of`
    self.multiple_of = multiple_of
    self.shuffle_size = shuffle_size
    self.shuffle_initial_size = shuffle_initial_size
    
    self.dataset, self.total_samples = self._build_source_dataset(sources)

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
  
  def _build_source_dataset(self, sources):
    total_samples = 0
    if isinstance(sources, str):
      sources = sources.split(",")
    with Timer("Read urls"):
      urls = []
      for source in sources:
        with open(source, encoding="utf-8") as f:
          index = json.loads(f.read())["shardlist"]
          for item in index:
            urls.append(os.path.join(os.path.dirname(source), item["url"]))
            total_samples += item["nsamples"]

    with Timer("Sort -> Shuffle -> Broadcast"):
      # broadcast all urls
      urls.sort()
      random.shuffle(urls)
      t = [urls]
      dist.broadcast_object_list(t, src=0)
      urls = t[0]
      logger.info(f"[RANK{dist.get_rank()}] {urls=}")

    with Timer("Build dataset"):
      dataset = wds.WebDataset(
          urls,
          handler=wds.warn_and_continue,
          resampled=True,
          shardshuffle=True,
          cache_dir="/tmp/_wids_cache",
          nodesplitter=wds.split_by_node,
          workersplitter=wds.split_by_worker
      )

      dataset = dataset.shuffle(
          self.shuffle_size, initial=self.shuffle_initial_size).decode(
        "pil", handler=wds.warn_and_continue)
      
    return dataset, total_samples

  def _fill_image_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):

    min_visual_tokens_per_image = conf["min_visual_tokens_per_image"]
    max_visual_tokens_per_image = conf["max_visual_tokens_per_image"]

    if isinstance(block["image"], str):
      image = sample_dict[block["image"]]
    else:
      image = block["image"]
    if image.mode != "RGB":
      image = image.convert("RGB")
    block["image"] = image
    block["min_pixels"] = min_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
    block["max_pixels"] = max_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
  
  def _fill_video_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):

    min_visual_tokens_per_image = conf["min_visual_tokens_per_image"]
    max_visual_tokens_per_image = conf["max_visual_tokens_per_image"]

    if isinstance(block["video"], list):
        if all([isinstance(image_block, str) for image_block in block["video"]]):
          block["video"] = [
            {
              "type": "image",
              "image": image_str
            }
            for image_str in block["video"]
          ]
        for image_block in block["video"]:
          assert image_block["type"] == "image" and "image" in image_block
          self._fill_image_block(image_block, sample_dict, conf)

    elif isinstance(block["video"], str) or isinstance(block["video"], bytes):
      # video in local tar, replace by video bytes
      if isinstance(block["video"], str) and block["video"] in sample_dict:
        block["video"] = sample_dict[block["video"]]
      # fill other params
      block["min_pixels"] = min_visual_tokens_per_image * (self.patch_size ** 2) * \
          (self.spatial_merge_size ** 2)
      block["max_pixels"] = max_visual_tokens_per_image * (self.patch_size ** 2) * \
          (self.spatial_merge_size ** 2)
      # video split params
      if conf["video_nframe"] > 0:
        block["nframes"] = conf["video_nframe"]
      else:
        if conf["video_fps"] > 0:
          block["fps"] = conf["video_fps"]
        if conf["video_min_frames"] > 0:
          block["min_frames"] = conf["video_min_frames"]
        if conf["video_max_frames"] > 0:
          block["max_frames"] = conf["video_max_frames"]
    else:
      raise ValueError(f"Unsupport video type. {type(block['video'])=}")
  
  def _process_completion(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "segments" in sample["json"]
    data_conf["max_visual_tokens_per_image"] = max(
        data_conf["max_visual_tokens_per_image"], data_conf["min_visual_tokens_per_image"])

    text = ""
    vision_infos = []
    segments = sample["json"]["segments"]
    for segment in segments:
      if segment["type"] == "text":
        text += segment["text"]
      elif segment["type"] == "image":
        text += "<|vision_start|><|image_pad|><|vision_end|>"
        self._fill_image_block(segment, sample,
                                conf=data_conf)
        vision_infos.append(segment)
      elif segment["type"] == "video":
        text += "<|vision_start|><|video_pad|><|vision_end|>"
        self._fill_video_block(segment, sample,
                                conf=data_conf)
        vision_infos.append(segment)
      else:
        logger.warning(f"!!! Unsupport {segment['type']=}, skip this segment.")
    
    # append EOS token
    text += "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(vision_infos = vision_infos)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > 32768:
      print(f"Sample is too long. token_len={inputs['input_ids'].shape[-1]}")
    
    # mask all vision token
    # <|vision_start|>: 151652 , <|vision_end|>: 151653, <|image_pad|>: 151655, <|video_pad|>: 151656
    input_ids = inputs["input_ids"]
    inputs["loss_mask"] = torch.ones_like(input_ids)
    inputs["loss_mask"][
        (input_ids == self.vision_start_token_id) | 
        (input_ids == self.vision_end_token_id) |
        (input_ids == self.image_token_id) |
        (input_ids == self.video_token_id)
      ] = 0
    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
    if inputs["loss_mask"].sum() == 0:
      raise ValueError(
        f"Unable to generate sample with 0 loss_mask."
      )

    inputs["position_ids"] = get_rope_index(
      inputs["input_ids"],
      image_grid_thw=inputs.get("image_grid_thw"),
      video_grid_thw=inputs.get("video_grid_thw"),
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs

  def _process_chat(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "message" in sample["json"] or "messages" in sample["json"]
    data_conf["max_visual_tokens_per_image"] = max(
        data_conf["max_visual_tokens_per_image"], data_conf["min_visual_tokens_per_image"])
    
    msg_key = "message" if "message" in sample["json"] else "messages"
    messages = sample["json"][msg_key]
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue
      for block in content:
        if block["type"] == "image":
          self._fill_image_block(block, sample, 
                                  conf=data_conf)
        elif block["type"] == "video":
          self._fill_video_block(block, sample,
                                  conf=data_conf)
        elif block["type"] == "text":
          continue
        else:
          raise ValueError(f"sample process error, unsupport value type: {block['type']}")

    text = self.processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=False
    )

    # append EOS token
    text += "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > 32768:
      raise ValueError(f"Sample is too long. text_len={len(text)=}, token_len={inputs['input_ids'].shape[-1]}")
    
    inputs["loss_mask"] = get_assistant_mask(
      inputs["input_ids"],
      start_pattern=[151644, 77091, 198],
      end_pattern=[151645, 198]
    )
    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
    if inputs["loss_mask"].sum() == 0:
      # try to process no text block, like content=""
      inputs["loss_mask"] = get_assistant_mask(
        inputs["input_ids"],
        start_pattern=[151644, 77091],
        end_pattern=[151645, 198]
      )
      if inputs["loss_mask"].sum() == 0:
        raise ValueError(
          f"Unable to generate sample with 0 loss_mask."
        )

    inputs["position_ids"] = get_rope_index(
      inputs["input_ids"],
      image_grid_thw=inputs.get("image_grid_thw"),
      video_grid_thw=inputs.get("video_grid_thw"),
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs
  
  def _gen_pad_input(self, pad_len):
    text = "<|endoftext|>" * pad_len
    inputs = self.processor.tokenizer(text)
    inputs["input_ids"] = torch.tensor([inputs["input_ids"]], dtype=torch.int64) # shape=[1, N], for get_rope_index
    inputs["loss_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["position_ids"] = get_rope_index(
      inputs["input_ids"],
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs
  
  def _gen_img_pad(self):
    """
    append an image, to trigger vit for pure text sample
    return 6 token: vstart, 4 * image_token, vend
    """
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    pad_image = {
        "type": "image",
        "image": Image.new("RGB", (1, 1), (255, 255, 255))
    }

    self._fill_image_block(pad_image, sample_dict={}, conf={
        "min_visual_tokens_per_image": self.min_visual_tokens_per_image,
        "max_visual_tokens_per_image": self.max_visual_tokens_per_image,
        "video_nframe": self.video_nframe,
        "video_fps": self.video_fps,
        "video_min_frames": self.video_min_frames,
        "video_max_frames": self.video_max_frames
    })
    image_inputs, _ = process_vision_info(vision_infos=[pad_image])
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=None,
        return_tensors="pt"
    )

    inputs["loss_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["position_ids"] = get_rope_index(
        inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        spatial_merge_size=self.spatial_merge_size,
        image_token_id=self.image_token_id,
        video_token_id=self.video_token_id,
        vision_start_token_id=self.vision_start_token_id
    )

    inputs.pop("attention_mask")
    return inputs

  def _process(self, sample, source_name=None):
    # self._may_filter(sample)

    # get data format
    if "messages" in sample["json"] or "message" in sample["json"]:
      data_format = "chatml"
    elif "segments" in sample["json"]:
      data_format = "completion"
    else:
      raise NotImplementedError(f"Unsupported dataset format.")
    
    source_conf = {
      "min_visual_tokens_per_image": self.min_visual_tokens_per_image,
      "max_visual_tokens_per_image": self.max_visual_tokens_per_image,
      "video_nframe": self.video_nframe,
      "video_fps": self.video_fps,
      "video_min_frames": self.video_min_frames,
      "video_max_frames": self.video_max_frames
    }

    if source_name != None and source_name in self.datasource_config:
      for key in source_conf:
        if key in self.datasource_config[source_name]:
          source_conf[key] = self.datasource_config[source_name][key]
    
    for retry in range(self.max_retry):
      if data_format == "chatml":
        inputs = self._process_chat(sample, source_conf)
      elif data_format == "completion":
        inputs = self._process_completion(sample, source_conf)
      else:
        raise NotImplementedError(
            f"Unsupported dataset format `{data_format}`")

      if not inputs:
        raise ValueError("Empty inputs, skip")

      if inputs["input_ids"].shape[-1] > self.max_length:
        source_conf["max_visual_tokens_per_image"] = (
            source_conf["max_visual_tokens_per_image"] * self.shrink_ratio)
        continue
      else:
        assert inputs["input_ids"].shape[-1] <= self.max_length, "inputs too long"
        return inputs
    else:
      raise ValueError(
          f"Unable to generate sample within max_length={self.max_length} after {retry} retrys"
      )
  
  def _append_sample_packing(self,
                             inputs: Dict[str, torch.Tensor],
                             packed_input_ids: List[torch.Tensor],
                             packed_position_ids: List[torch.Tensor],
                             packed_loss_mask: List[torch.Tensor],
                             packed_pixel_values: List[torch.Tensor],
                             packed_pixel_values_videos: List[torch.Tensor],
                             packed_image_gird_thw: List[torch.Tensor],
                             packed_video_grid_thw: List[torch.Tensor],
                             packed_sample_idx: List[torch.Tensor],
                             cu_seqlens: List[int],
                             sample_idx: Optional[int] = None):

    packed_input_ids.append(inputs["input_ids"].flatten())
    packed_loss_mask.append(inputs["loss_mask"].flatten())
    packed_position_ids.append(inputs["position_ids"])
    if sample_idx is None:
      sample_idx = len(cu_seqlens) - 1
    packed_sample_idx.append(
      torch.full_like(packed_input_ids[-1], sample_idx))

    if "pixel_values" in inputs:
      packed_pixel_values.append(inputs["pixel_values"])
      packed_image_gird_thw.append(inputs["image_grid_thw"])
    if "pixel_values_videos" in inputs:
      packed_pixel_values_videos.append(inputs["pixel_values_videos"])
      packed_video_grid_thw.append(inputs["video_grid_thw"])
    cu_seqlens.append(cu_seqlens[-1] + len(inputs["input_ids"][0]))
    return len(inputs["input_ids"][0])

  def _packing(self, buffer: List[Dict[str, torch.Tensor]]):
    packed_input_ids: List[torch.Tensor] = []
    packed_position_ids: List[torch.Tensor] = []
    packed_loss_mask: List[torch.Tensor] = []
    packed_pixel_values: List[torch.Tensor] = []
    packed_pixel_values_videos: List[torch.Tensor] = []
    packed_image_gird_thw: List[torch.Tensor] = []
    packed_video_grid_thw: List[torch.Tensor] = []
    packed_sample_idx: List[torch.Tensor] = []
    cu_seqlens: List[int] = [0]

    valid_seq_len = 0
    for _, inputs in enumerate(buffer):
      valid_seq_len += self._append_sample_packing(inputs,
                                      packed_input_ids,
                                      packed_position_ids,
                                      packed_loss_mask,
                                      packed_pixel_values,
                                      packed_pixel_values_videos,
                                      packed_image_gird_thw,
                                      packed_video_grid_thw,
                                      packed_sample_idx,
                                      cu_seqlens)

    # append a pad image sequence to trigger ViT
    image_pad = self._gen_img_pad()
    self._append_sample_packing(image_pad,
                                packed_input_ids,
                                packed_position_ids,
                                packed_loss_mask,
                                packed_pixel_values,
                                packed_pixel_values_videos,
                                packed_image_gird_thw,
                                packed_video_grid_thw,
                                packed_sample_idx,
                                cu_seqlens,
                                sample_idx=-1)

    packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
    packed_loss_mask = torch.cat(packed_loss_mask, dim=0).unsqueeze(0)
    packed_position_ids = torch.cat(packed_position_ids, dim=-1)
    packed_sample_idx = torch.cat(packed_sample_idx, dim=0).unsqueeze(0)
    packed_pixel_values = None if len(packed_pixel_values) == 0 else \
      torch.cat(packed_pixel_values, dim=0)
    packed_image_gird_thw = None if len(packed_image_gird_thw) == 0 else \
      torch.cat(packed_image_gird_thw, dim=0)
    packed_pixel_values_videos = \
      None if len(packed_pixel_values_videos) == 0 else \
        torch.cat(packed_pixel_values_videos, dim=0)
    packed_video_grid_thw = None if len(packed_video_grid_thw) == 0 else \
      torch.cat(packed_video_grid_thw, dim=0)

    # pad seq len to multiple_of
    if (
      self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
    ):
      padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
      packed_input_ids = F.pad(
        packed_input_ids, (0, padding_len),
        value=self.processor.tokenizer.pad_token_id)
      packed_sample_idx = F.pad(
        packed_sample_idx, (0, padding_len), value=-1)
      packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
      packed_loss_mask = F.pad(packed_loss_mask, (0, padding_len), value=0)
      cu_seqlens.append(cu_seqlens[-1] + padding_len)

    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "pixel_values": packed_pixel_values,
      "image_grid_thw": packed_image_gird_thw,
      "pixel_values_videos": packed_pixel_values_videos,
      "video_grid_thw": packed_video_grid_thw,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
      "sample_idx": packed_sample_idx.to(torch.int32)
    }
    return inputs

  def __iter__(self):
    buffer = []
    source_list = []
    cur_length = 0

    for sample in self.dataset:
      sample_key = sample["__key__"] if "__key__" in sample else ""
      sample_url = sample["__url__"] if "__url__" in sample else ""

      try:
        source_name = sample["json"]["source"]
        # WARN: ugly code, for dirty dataset.
        if source_name.startswith("PDFA"):
          source_name = "PDFA"
        elif source_name.startswith("/llm_reco_ssd/luoxinchen/dataset/"):
          source_name = source_name.split("/")[4]
      except:
        source_name = "None"

      self.source_sample_cnt.setdefault(source_name, 0)
      self.source_sample_cnt[source_name] += 1

      try:
        inputs = self._process(sample, source_name)
      except:
        self.source_error_cnt.setdefault(source_name, 0)
        self.source_error_cnt[source_name] += 1
        error_ratio = self.source_error_cnt[source_name] * 1.0 / \
          self.source_sample_cnt[source_name]
        logger.error(
          f"ChatCompletionVisionDataset process sample error. "
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, "
          f"errmsg={traceback.format_exc()}")
        continue

      sample_length = inputs["input_ids"].shape[-1]
      if cur_length + sample_length > self.max_length:
        packed_inputs = self._packing(buffer)
        packed_inputs["data_source"] = source_list
        buffer = [inputs]
        source_list = [source_name]
        cur_length = sample_length

        # skip pure text sample
        # 有pad image，原则上不会出现纯文本输入
        if packed_inputs["pixel_values"] is None and \
            packed_inputs["pixel_values_videos"] is None:
          logger.warning("Skip pure text sample.")
          continue

        # skip 0 label pack
        if packed_inputs["loss_mask"].sum() == 0:
          logger.warning("Skip 0 lable sample.")
          continue

        yield packed_inputs

      else:
        buffer.append(inputs)
        source_list.append(source_name)
        cur_length += sample_length

class ChatCompletionVisionDpoDataset(IterableDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               max_length: int = 1024,
               min_visual_tokens_per_image: int = 4,
               max_visual_tokens_per_image: int = 512,
               video_nframe: int = -1,
               video_fps: float = 2.0,
               video_min_frames: int = -1,
               video_max_frames: int = 120,
               shrink_ratio: float = 0.9,
               max_retry: int = 5,
               multiple_of: int = 8,
               shuffle_size: int = 100000,
               shuffle_initial_size: int = 20000,
               base_model_dir: Optional[str] = None,
               processor: Optional[Qwen2VLProcessor] = None,
               spatial_merge_size: int = 2,
               patch_size: int = 14,
               image_token_id: int = 151655,
               video_token_id: int = 151656,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {}):
    """
    datasource_config: 默认覆盖全局配置
                      key: datasource_name
                      Dict: datasource config, support params:
                        min_visual_tokens_per_image
                        max_visual_tokens_per_image
                        video_nframe
                        video_fps
                        video_min_frames
                        video_max_frames
    """
    if base_model_dir:
      processor = Qwen2VLProcessor.from_pretrained(base_model_dir)
      model_config = Qwen2VLConfig.from_pretrained(base_model_dir)
      spatial_merge_size = model_config.vision_config.spatial_merge_size
      patch_size = model_config.vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    self.processor = processor
    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.video_nframe = video_nframe
    self.video_fps = video_fps
    self.video_min_frames = video_min_frames
    self.video_max_frames = video_max_frames
    if video_nframe > 0 and (video_fps > 0 or video_min_frames > 0 or video_max_frames > 0):
      logger.warning(
        f"ChatCompletionVisionDataset(video_fps=...): video_fps, video_min_frames, "\
          f"video_max_frames will be ignored when video_nframe>0 ({video_nframe=})"
      )
    self.patch_size = patch_size
    self.shrink_ratio = shrink_ratio
    self.max_retry = max_retry
    self.spatial_merge_size = spatial_merge_size
    self.image_token_id = image_token_id
    self.video_token_id = video_token_id
    self.vision_start_token_id = vision_start_token_id
    self.vision_end_token_id = vision_end_token_id
    self.pad_token_id = pad_token_id
    self.patch_size = patch_size
    # Pad sequence to multiple of `multiple_of`
    self.multiple_of = multiple_of
    self.shuffle_size = shuffle_size
    self.shuffle_initial_size = shuffle_initial_size
    
    self.dataset, self.total_samples = self._build_source_dataset(sources)

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    print(f"datasource_config: {self.datasource_config}")
  
  def _build_source_dataset(self, sources):
    total_samples = 0
    if isinstance(sources, str):
      sources = sources.split(",")
    with Timer("Read urls"):
      urls = []
      for source in sources:
        with open(source, encoding="utf-8") as f:
          index = json.loads(f.read())["shardlist"]
          for item in index:
            urls.append(os.path.join(os.path.dirname(source), item["url"]))
            total_samples += item["nsamples"]

    with Timer("Sort -> Shuffle -> Broadcast"):
      # broadcast all urls
      urls.sort()
      random.shuffle(urls)
      t = [urls]
      dist.broadcast_object_list(t, src=0)
      urls = t[0]
      logger.info(f"[RANK{dist.get_rank()}] {urls=}")

    with Timer("Build dataset"):
      dataset = wds.WebDataset(
          urls,
          handler=wds.warn_and_continue,
          resampled=True,
          shardshuffle=True,
          cache_dir="/tmp/_wids_cache",
          nodesplitter=wds.split_by_node,
          workersplitter=wds.split_by_worker
      )

      dataset = dataset.shuffle(
          self.shuffle_size, initial=self.shuffle_initial_size).decode(
        "pil", handler=wds.warn_and_continue)
      
    return dataset, total_samples

  def _fill_image_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):

    min_visual_tokens_per_image = conf["min_visual_tokens_per_image"]
    max_visual_tokens_per_image = conf["max_visual_tokens_per_image"]

    if isinstance(block["image"], str):
      image = sample_dict[block["image"]]
    else:
      image = block["image"]
    if image.mode != "RGB":
      image = image.convert("RGB")
    block["image"] = image
    block["min_pixels"] = min_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
    block["max_pixels"] = max_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
  
  def _fill_video_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):

    min_visual_tokens_per_image = conf["min_visual_tokens_per_image"]
    max_visual_tokens_per_image = conf["max_visual_tokens_per_image"]

    if isinstance(block["video"], list):
      #TODO:把数据格式统一成，video 的list中的image都是dict格式。
      if all([isinstance(image_block, str) for image_block in block["video"]]):
        block["video"] = [
          {
            "type": "image",
            "image": image_str
          }
          for image_str in block["video"]
        ]
      for image_block in block["video"]:
        assert image_block["type"] == "image" and "image" in image_block
        self._fill_image_block(image_block, sample_dict, conf)

    elif isinstance(block["video"], str) or isinstance(block["video"], bytes):
      # video in local tar, replace by video bytes
      if isinstance(block["video"], str) and block["video"] in sample_dict:
        block["video"] = sample_dict[block["video"]]
      # fill other params
      block["min_pixels"] = min_visual_tokens_per_image * (self.patch_size ** 2) * \
          (self.spatial_merge_size ** 2)
      block["max_pixels"] = max_visual_tokens_per_image * (self.patch_size ** 2) * \
          (self.spatial_merge_size ** 2)
      # video split params
      if conf["video_nframe"] > 0:
        block["nframes"] = conf["video_nframe"]
      else:
        if conf["video_fps"] > 0:
          block["fps"] = conf["video_fps"]
        if conf["video_min_frames"] > 0:
          block["min_frames"] = conf["video_min_frames"]
        if conf["video_max_frames"] > 0:
          block["max_frames"] = conf["video_max_frames"]
    else:
      raise ValueError(f"Unsupport video type. {type(block['video'])=}")
  
  def _process_completion(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "segments" in sample["json"]
    data_conf["max_visual_tokens_per_image"] = max(
        data_conf["max_visual_tokens_per_image"], data_conf["min_visual_tokens_per_image"])

    text = ""
    vision_infos = []
    segments = sample["json"]["segments"]
    for segment in segments:
      if segment["type"] == "text":
        text += segment["text"]
      elif segment["type"] == "image":
        text += "<|vision_start|><|image_pad|><|vision_end|>"
        self._fill_image_block(segment, sample,
                                conf=data_conf)
        vision_infos.append(segment)
      elif segment["type"] == "video":
        text += "<|vision_start|><|video_pad|><|vision_end|>"
        self._fill_video_block(segment, sample,
                                conf=data_conf)
        vision_infos.append(segment)
      else:
        logger.warning(f"!!! Unsupport {segment['type']=}, skip this segment.")
    
    # append EOS token
    text += "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(vision_infos = vision_infos)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > 32768:
      raise ValueError(f"Sample is too long. text_len={len(text)=}, token_len={inputs['input_ids'].shape[-1]}")
    
    # mask all vision token
    # <|vision_start|>: 151652 , <|vision_end|>: 151653, <|image_pad|>: 151655, <|video_pad|>: 151656
    input_ids = inputs["input_ids"]
    inputs["loss_mask"] = torch.ones_like(input_ids)
    inputs["loss_mask"][
        (input_ids == self.vision_start_token_id) | 
        (input_ids == self.vision_end_token_id) |
        (input_ids == self.image_token_id) |
        (input_ids == self.video_token_id)
      ] = 0
    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
    if inputs["loss_mask"].sum() == 0:
      raise ValueError(
        f"Unable to generate sample with 0 loss_mask."
      )

    inputs["position_ids"] = get_rope_index(
      inputs["input_ids"],
      image_grid_thw=inputs.get("image_grid_thw"),
      video_grid_thw=inputs.get("video_grid_thw"),
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs

  def _process_chat(self,
                    sample: Dict[str, Any],
                    sample_type: str,
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "message" in sample["json"] or "messages" in sample["json"]
    assert sample_type in ["chosen", "rejected"]
    assert sample_type in sample["json"]
    data_conf["max_visual_tokens_per_image"] = max(
        data_conf["max_visual_tokens_per_image"], data_conf["min_visual_tokens_per_image"])
    
    msg_key = "message" if "message" in sample["json"] else "messages"
    messages = sample["json"][msg_key]
    if messages is None:
        raise ValueError(f"Messages is None for sample : {sample}, "
                        f"source: {sample_type}")
    
    if not isinstance(messages, (list, tuple)):
        raise ValueError(f"Messages must be list or tuple, got {type(messages)} for sample: {sample}")
    
    if len(messages) == 0:
        raise ValueError(f"Messages is empty for sample key: {sample}")
    if sample_type == "chosen":
      messages = messages + [sample["json"]["chosen"]]
    elif sample_type == "rejected":
      messages = messages + [sample["json"]["rejected"]]
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue
      for block in content:
        if block["type"] == "image":
          self._fill_image_block(block, sample, 
                                  conf=data_conf)
        elif block["type"] == "video":
          self._fill_video_block(block, sample,
                                  conf=data_conf)
        elif block["type"] == "text":
          continue
        else:
          raise ValueError(f"sample process error, unsupport value type: {block['type']}")

    text = self.processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=False
    )

    # append EOS token
    text += "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > 32768:
      raise ValueError(f"Sample is too long. text_len={len(text)=}, token_len={inputs['input_ids'].shape[-1]}")
    
    inputs["loss_mask"] = get_assistant_mask(
      inputs["input_ids"],
      start_pattern=[151644, 77091, 198],
      end_pattern=[151645, 198]
    )
    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
    if inputs["loss_mask"].sum() == 0:
      # try to process no text block, like content=""
      inputs["loss_mask"] = get_assistant_mask(
        inputs["input_ids"],
        start_pattern=[151644, 77091],
        end_pattern=[151645, 198]
      )
      if inputs["loss_mask"].sum() == 0:
        raise ValueError(
          f"Unable to generate sample with 0 loss_mask."
        )

    inputs["position_ids"] = get_rope_index(
      inputs["input_ids"],
      image_grid_thw=inputs.get("image_grid_thw"),
      video_grid_thw=inputs.get("video_grid_thw"),
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs
  
  def _gen_pad_input(self, pad_len):
    text = "<|endoftext|>" * pad_len
    inputs = self.processor.tokenizer(text)
    inputs["input_ids"] = torch.tensor([inputs["input_ids"]], dtype=torch.int64) # shape=[1, N], for get_rope_index
    inputs["loss_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["position_ids"] = get_rope_index(
      inputs["input_ids"],
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )
    inputs.pop("attention_mask")
    return inputs
  
  def _gen_img_pad(self):
    """
    append an image, to trigger vit for pure text sample
    return 6 token: vstart, 4 * image_token, vend
    """
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    pad_image = {
        "type": "image",
        "image": Image.new("RGB", (1, 1), (255, 255, 255))
    }

    self._fill_image_block(pad_image, sample_dict={}, conf={
        "min_visual_tokens_per_image": self.min_visual_tokens_per_image,
        "max_visual_tokens_per_image": self.max_visual_tokens_per_image,
        "video_nframe": self.video_nframe,
        "video_fps": self.video_fps,
        "video_min_frames": self.video_min_frames,
        "video_max_frames": self.video_max_frames
    })
    image_inputs, _ = process_vision_info(vision_infos=[pad_image])
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=None,
        return_tensors="pt"
    )

    inputs["loss_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["position_ids"] = get_rope_index(
        inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        spatial_merge_size=self.spatial_merge_size,
        image_token_id=self.image_token_id,
        video_token_id=self.video_token_id,
        vision_start_token_id=self.vision_start_token_id
    )

    inputs.pop("attention_mask")
    return inputs

  def _process(self, sample, source_name=None):
    # self._may_filter(sample)

    # get data format
    if "messages" in sample["json"] or "message" in sample["json"]:
      data_format = "chatml"
    elif "segments" in sample["json"]:
      data_format = "completion"
    else:
      raise NotImplementedError(f"Unsupported dataset format.")
    
    source_conf = {
      "min_visual_tokens_per_image": self.min_visual_tokens_per_image,
      "max_visual_tokens_per_image": self.max_visual_tokens_per_image,
      "video_nframe": self.video_nframe,
      "video_fps": self.video_fps,
      "video_min_frames": self.video_min_frames,
      "video_max_frames": self.video_max_frames
    }

    if source_name != None and source_name in self.datasource_config:
      for key in source_conf:
        if key in self.datasource_config[source_name]:
          source_conf[key] = self.datasource_config[source_name][key]
    
    for retry in range(self.max_retry):
      if data_format == "chatml":
        chosen_inputs = self._process_chat(sample, "chosen", source_conf)
        rejected_inputs = self._process_chat(sample, "rejected", source_conf)
        inputs = {
          "chosen_input": chosen_inputs,
          "rejected_input": rejected_inputs,
        }
      else:
        raise NotImplementedError(
            f"Unsupported dataset format `{data_format}`")

      if not inputs:
        raise ValueError("Empty inputs, skip")

      if inputs["chosen_input"]["input_ids"].shape[-1] > self.max_length or inputs["rejected_input"]["input_ids"].shape[-1] > self.max_length:
        source_conf["max_visual_tokens_per_image"] = (
            source_conf["max_visual_tokens_per_image"] * self.shrink_ratio)
        continue
      else:
        assert inputs["chosen_input"]["input_ids"].shape[-1] <= self.max_length and inputs["rejected_input"]["input_ids"].shape[-1] <= self.max_length, "inputs too long"
        return inputs
    else:
      raise ValueError(
          f"Unable to generate sample within max_length={self.max_length} after {retry} retrys"
      )
  
  def _append_sample_packing(self,
                             inputs: Dict[str, torch.Tensor],
                             packed_input_ids: List[torch.Tensor],
                             packed_position_ids: List[torch.Tensor],
                             packed_loss_mask: List[torch.Tensor],
                             packed_pixel_values: List[torch.Tensor],
                             packed_pixel_values_videos: List[torch.Tensor],
                             packed_image_gird_thw: List[torch.Tensor],
                             packed_video_grid_thw: List[torch.Tensor],
                             packed_sample_idx: List[torch.Tensor],
                             cu_seqlens: List[int],
                             sample_idx: Optional[int] = None):

    packed_input_ids.append(inputs["input_ids"].flatten())
    packed_loss_mask.append(inputs["loss_mask"].flatten())
    packed_position_ids.append(inputs["position_ids"])
    if sample_idx is None:
      sample_idx = len(cu_seqlens) - 1
    packed_sample_idx.append(
      torch.full_like(packed_input_ids[-1], sample_idx))

    if "pixel_values" in inputs:
      packed_pixel_values.append(inputs["pixel_values"])
      packed_image_gird_thw.append(inputs["image_grid_thw"])
    if "pixel_values_videos" in inputs:
      packed_pixel_values_videos.append(inputs["pixel_values_videos"])
      packed_video_grid_thw.append(inputs["video_grid_thw"])
    cu_seqlens.append(cu_seqlens[-1] + len(inputs["input_ids"][0]))
    return len(inputs["input_ids"][0])

  def _packing(self, buffer: List[Dict[str, torch.Tensor]]):
    packed_input_ids: List[torch.Tensor] = []
    packed_position_ids: List[torch.Tensor] = []
    packed_loss_mask: List[torch.Tensor] = []
    packed_pixel_values: List[torch.Tensor] = []
    packed_pixel_values_videos: List[torch.Tensor] = []
    packed_image_gird_thw: List[torch.Tensor] = []
    packed_video_grid_thw: List[torch.Tensor] = []
    packed_sample_idx: List[torch.Tensor] = []
    cu_seqlens: List[int] = [0]

    valid_seq_len = 0
    for _, inputs in enumerate(buffer):
      valid_seq_len += self._append_sample_packing(inputs,
                                      packed_input_ids,
                                      packed_position_ids,
                                      packed_loss_mask,
                                      packed_pixel_values,
                                      packed_pixel_values_videos,
                                      packed_image_gird_thw,
                                      packed_video_grid_thw,
                                      packed_sample_idx,
                                      cu_seqlens)

    # append a pad image sequence to trigger ViT
    image_pad = self._gen_img_pad()
    self._append_sample_packing(image_pad,
                                packed_input_ids,
                                packed_position_ids,
                                packed_loss_mask,
                                packed_pixel_values,
                                packed_pixel_values_videos,
                                packed_image_gird_thw,
                                packed_video_grid_thw,
                                packed_sample_idx,
                                cu_seqlens,
                                sample_idx=-1)

    packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
    packed_loss_mask = torch.cat(packed_loss_mask, dim=0).unsqueeze(0)
    packed_position_ids = torch.cat(packed_position_ids, dim=-1)
    packed_sample_idx = torch.cat(packed_sample_idx, dim=0).unsqueeze(0)
    packed_pixel_values = None if len(packed_pixel_values) == 0 else \
      torch.cat(packed_pixel_values, dim=0)
    packed_image_gird_thw = None if len(packed_image_gird_thw) == 0 else \
      torch.cat(packed_image_gird_thw, dim=0)
    packed_pixel_values_videos = \
      None if len(packed_pixel_values_videos) == 0 else \
        torch.cat(packed_pixel_values_videos, dim=0)
    packed_video_grid_thw = None if len(packed_video_grid_thw) == 0 else \
      torch.cat(packed_video_grid_thw, dim=0)

    # pad seq len to multiple_of
    if (
      self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
    ):
      padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
      packed_input_ids = F.pad(
        packed_input_ids, (0, padding_len),
        value=self.processor.tokenizer.pad_token_id)
      packed_sample_idx = F.pad(
        packed_sample_idx, (0, padding_len), value=-1)
      packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
      packed_loss_mask = F.pad(packed_loss_mask, (0, padding_len), value=0)
      cu_seqlens.append(cu_seqlens[-1] + padding_len)

    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "pixel_values": packed_pixel_values,
      "image_grid_thw": packed_image_gird_thw,
      "pixel_values_videos": packed_pixel_values_videos,
      "video_grid_thw": packed_video_grid_thw,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
      "sample_idx": packed_sample_idx.to(torch.int32)
    }
    return inputs

  def __iter__(self):
    buffer_chosen = []
    buffer_rejected = []
    source_list = []
    cur_length_chosen = 0
    cur_length_rejected = 0

    for sample in self.dataset:
      sample_key = sample["__key__"] if "__key__" in sample else ""
      sample_url = sample["__url__"] if "__url__" in sample else ""

      try:
        source_name = sample["json"]["source"]
        # WARN: ugly code, for dirty dataset.
        if source_name.startswith("PDFA"):
          source_name = "PDFA"
        elif source_name.startswith("/llm_reco_ssd/luoxinchen/dataset/"):
          source_name = source_name.split("/")[4]
      except:
        source_name = "None"

      self.source_sample_cnt.setdefault(source_name, 0)
      self.source_sample_cnt[source_name] += 1

      try:
        inputs = self._process(sample, source_name)
      except:
        self.source_error_cnt.setdefault(source_name, 0)
        self.source_error_cnt[source_name] += 1
        error_ratio = self.source_error_cnt[source_name] * 1.0 / \
          self.source_sample_cnt[source_name]
        logger.error(
          f"ChatCompletionVisionDataset process sample error. "
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, "
          f"errmsg={traceback.format_exc()}")
        continue

      sample_length_chosen = inputs["chosen_input"]["input_ids"].shape[-1]
      sample_length_rejected = inputs["rejected_input"]["input_ids"].shape[-1]
      if cur_length_chosen + sample_length_chosen > self.max_length or cur_length_rejected + sample_length_rejected > self.max_length:
        packed_inputs_chosen = self._packing(buffer_chosen)
        packed_inputs_chosen["data_source"] = source_list
        packed_inputs_rejected = self._packing(buffer_rejected)
        packed_inputs_rejected["data_source"] = source_list
        buffer_chosen = [inputs["chosen_input"]]
        buffer_rejected = [inputs["rejected_input"]]
        source_list = [source_name]
        cur_length_chosen = sample_length_chosen
        cur_length_rejected = sample_length_rejected
        # skip pure text sample
        # 有pad image，原则上不会出现纯文本输入
        if (packed_inputs_chosen["pixel_values"] is None and \
            packed_inputs_chosen["pixel_values_videos"] is None) or \
            (packed_inputs_rejected["pixel_values"] is None and \
            packed_inputs_rejected["pixel_values_videos"] is None):
          logger.warning("Skip pure text sample.")
          continue

        # skip 0 label pack
        if packed_inputs_chosen["loss_mask"].sum() == 0 or packed_inputs_rejected["loss_mask"].sum() == 0:
          logger.warning("Skip 0 lable sample.")
          continue

        yield packed_inputs_chosen, packed_inputs_rejected

      else:
        buffer_chosen.append(inputs["chosen_input"])
        buffer_rejected.append(inputs["rejected_input"])
        source_list.append(source_name)
        cur_length_chosen += sample_length_chosen
        cur_length_rejected += sample_length_rejected


class ParquetDataset(IterableDataset):
  def __init__(self, data_files, num_workers):
    self.data_files = data_files
    self.num_workers = num_workers

    manager = multiprocessing.Manager()

    self.finish_dict_all = manager.dict()
    self.offset_dict_all = manager.dict()
    for i in range(self.num_workers):
      self.finish_dict_all[i] = manager.dict()
      self.offset_dict_all[i] = manager.dict()

  def state_dict(self,):
    rank, world_size, worker, num_workers = pytorch_worker_info()

    state_dict = {
      "finish_dict": dict(self.finish_dict_all[worker]),
      "offset_dict": dict(self.offset_dict_all[worker])
    }
    return state_dict
  
  def load_state_dict(self, state_dict):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    finish_dict = state_dict["finish_dict"]
    offset_dict = state_dict["offset_dict"]

    # support old ckpt format
    tmp_finish_dict = dict()
    tmp_offset_dict = dict()

    for k, v in finish_dict.items():
      if isinstance(k, str):
        tmp_finish_dict[(k, 0)] = v
      elif isinstance(k, tuple) and len(k) == 2:
        tmp_finish_dict[k] = v
      else:
        raise NotImplementedError(f"Unsupported dataloader checkpoint format.") 
    
    for k, v in offset_dict.items():
      if isinstance(k, str):
        fn, group_idx = k.split("|")
        group_idx = int(group_idx)
        tmp_offset_dict[(fn, 0, group_idx)] = v
      elif isinstance(k, tuple) and len(k) == 3:
        tmp_offset_dict[k] = v
      else:
        raise NotImplementedError(f"Unsupported dataloader checkpoint format.") 

    # clear cur state
    self.finish_dict_all[worker].clear()
    self.offset_dict_all[worker].clear()

    # update
    self.finish_dict_all[worker].update(tmp_finish_dict)
    self.offset_dict_all[worker].update(tmp_offset_dict)
    logger.warning(f"[rank{rank}-woker{worker}] load checkpoint success.")

  def _parser(self, raw_row_data, file_url):
    try:
      messages = None
      segments = None
      chosen = None
      rejected = None

      if "messages" in raw_row_data:
        messages = raw_row_data["messages"]
        if isinstance(messages, str):
          messages = json.loads(messages)
          
      if "segments" in raw_row_data:
        segments = raw_row_data["segments"]
        if isinstance(segments, str):
          segments = json.loads(segments)
          
      images = raw_row_data["images"]
      data_source = raw_row_data["source"]
      key = raw_row_data["uuid"]

      samples = {
        "__key__": key,
        "__url__": file_url,
      }

      # process message or segments -> webdataset_key = json
      sample_data = {
        "source": data_source,
      }

      if "chosen" in raw_row_data:
        chosen = raw_row_data["chosen"]
        if isinstance(chosen, str):
          chosen = json.loads(chosen)
        sample_data["chosen"] = chosen
          
      if "rejected" in raw_row_data:
        rejected = raw_row_data["rejected"]
        if isinstance(rejected, str):
          rejected = json.loads(rejected)
        sample_data["rejected"] = rejected

      if messages is not None and isinstance(messages, list):
        sample_data["messages"] = messages
      elif segments is not None and isinstance(segments, list):
        sample_data["segments"] = segments
      elif messages is not None and isinstance(messages, np.ndarray):
        sample_data["messages"] = messages.tolist()
      else:
        raise NotImplementedError(f"Unsupported sample, message type is {type(messages)}, message={messages}, segments type is {type(segments)}, segments={segments}")
      samples["json"] = sample_data

      # process images
      if isinstance(images, str):
        images = json.loads(images)
      elif isinstance(images, dict):
        pass
      else:
        raise NotImplementedError(f"Unsupported image field type, {type(raw_row_data['images'])=}")

      for image_name in images:
        image_b64 = images[image_name]
        image_bytes = base64.b64decode(image_b64)
        image_bytes_stream = BytesIO(image_bytes)
        image = Image.open(image_bytes_stream)
        samples[image_name] = image
      return samples
    except:
      logger.error(f"ParquetDataset parse sample error!!! err_msg={traceback.format_exc()}")
      return None

  def __iter__(self,):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    assert num_workers == self.num_workers

    finish_dict = self.finish_dict_all[worker]
    offset_dict = self.offset_dict_all[worker]

    total_num_workers = num_workers * world_size
    local_worker_idx = rank * num_workers + worker
    fn_list = [fn for idx, fn in enumerate(self.data_files) if idx % total_num_workers == local_worker_idx]
    logger.warning(
      f"ParquetDataset Info: {rank=}, {world_size=}, {worker=}, {num_workers=}, {len(fn_list)=}"
    )
    
    try:
      for epoch_fn in fn_list:
        fn, epoch_idx = epoch_fn
        if (fn, epoch_idx) in finish_dict:
          logger.warning(f"[Rank{rank}-{worker}] skip {fn}")
          continue
        
        # open parquet file
        try:
          parquet_file = pq.ParquetFile(fn)
        except Exception as e:
          logger.error(f"ParquetDataset error, open parquet fail!!! {fn=}, error_msg={traceback.format_exc()}")
          parquet_file = None
        
        # process file content
        if parquet_file is not None:
          logger.warning(f"[Rank{rank}-{worker}] {fn} total row_groups: {parquet_file.num_row_groups}")
          for group_idx in range(parquet_file.num_row_groups):
            try:
              offset = 0
              fn_group_key = (fn, epoch_idx, group_idx)
              if fn_group_key in offset_dict:
                if offset_dict[fn_group_key] == -1:
                  logger.warning(f"[Rank{rank}-{worker}] skip {fn}-epoch{epoch_idx}-group{group_idx}")
                  continue
                else:
                  offset = offset_dict[fn_group_key] + 1
              
              row_group = parquet_file.read_row_group(group_idx)
              if offset >= row_group.num_rows:
                continue
              logger.warning(f"[Rank{rank}-{worker}] start {fn}-epoch{epoch_idx}-group{group_idx}-offset{offset}")
              row_pandas = row_group.to_pandas().reset_index().iloc[offset:]

              for row_idx, row in row_pandas.iterrows():
                if row_idx < offset:
                  continue

                try:
                  sample = self._parser(row, fn)
                  if sample is not None:
                    yield sample
                  offset_dict[fn_group_key] = row_idx
                except GeneratorExit:
                  # 正确处理生成器退出
                  logger.warning(f"Generator exited at {fn}-epoch{epoch_idx}-group{group_idx}-row{row_idx}")
                  return
                except Exception as e:
                  logger.error(f"Error processing row {row_idx}: {str(e)}")
                  continue

                if row_idx % 1000 == 0:
                  logger.warning(f"Processing row {row_idx} in {fn}-epoch{epoch_idx}-group{group_idx}")

              # group finish
              logger.warning(f"[Rank{rank}-{worker}] {fn}-epoch{epoch_idx}-group{group_idx} finish.")
              offset_dict[fn_group_key] = -1
              
            except GeneratorExit:
              # 正确处理生成器退出
              logger.warning(f"Generator exited during group processing")
              return
            except Exception as e:
              logger.error(f"Error processing group {group_idx}: {str(e)}")
              continue
          
          # file finish
          logger.warning(f"[Rank{rank}-{worker}] {fn} finish.")
          finish_dict[(fn, epoch_idx)] = True

    except GeneratorExit:
      # 正确处理生成器退出
      logger.warning("Generator exited during file processing")
      return
    except Exception as e:
      logger.error(f"Error in dataset iterator: {str(e)}\n{traceback.format_exc()}")
      raise
class ChatCompletionVisionParquetDataset(ChatCompletionVisionDataset):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    super().__init__(sources, **kargs)

  def _build_source_dataset(self, sources):
    data_file_list = []
    if dist.get_rank() == 0:
      data_files = []
      if isinstance(sources, str) and sources.endswith(".json"):
        with open(sources, "r") as fp:
          data_files = json.loads(fp.read())
          data_files = [fn for fn in data_files if fn.endswith(".parquet")]
      elif isinstance(sources, list):
        for source in sources:
          hdfs_files = shell_hdfs_ls(source)
          data_files += [fn for fn in hdfs_files if fn.endswith(".parquet")]
      # repeat
      for i in range(self.num_epochs):
        data_files.sort()
        self.rng.shuffle(data_files)
        data_file_list += [(fn, i) for fn in data_files]
      logger.error(f"ChatCompletionVisionParquetDataset rank{dist.get_rank()}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

    t = [data_file_list]
    dist.broadcast_object_list(t, src=0)
    data_file_list = t[0]

    logger.error(f"ChatCompletionVisionParquetDataset rank{dist.get_rank()}: file_num={len(data_file_list)}")
    if len(data_file_list) == 0:
      raise ValueError(f"no datafile found!")

    dataset = ParquetDataset(data_file_list, self.num_workers)
    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)

class ChatCompletionVisionDpoParquetDataset(ChatCompletionVisionDpoDataset):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    super().__init__(sources, **kargs)

  def _build_source_dataset(self, sources):
    data_file_list = []
    if dist.get_rank() == 0:
      data_files = []
      if isinstance(sources, str) and sources.endswith(".json"):
        with open(sources, "r") as fp:
          data_files = json.loads(fp.read())
          data_files = [fn for fn in data_files if fn.endswith(".parquet")]
      elif isinstance(sources, list):
        for source in sources:
          hdfs_files = shell_hdfs_ls(source)
          data_files += [fn for fn in hdfs_files if fn.endswith(".parquet")]
      # repeat
      for i in range(self.num_epochs):
        data_files.sort()
        self.rng.shuffle(data_files)
        data_file_list += [(fn, i) for fn in data_files]
      logger.error(f"ChatCompletionVisionDpoParquetDataset rank{dist.get_rank()}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

    t = [data_file_list]
    dist.broadcast_object_list(t, src=0)
    data_file_list = t[0]

    logger.error(f"ChatCompletionVisionDpoParquetDataset rank{dist.get_rank()}: file_num={len(data_file_list)}")
    if len(data_file_list) == 0:
      raise ValueError(f"no datafile found!")

    dataset = ParquetDataset(data_file_list, self.num_workers)
    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)
