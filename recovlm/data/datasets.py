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
import math
import pyarrow.parquet as pq
from datetime import datetime
import os.path as osp
import webdataset as wds
from recovlm.utils.ds_utils import print_input_info

from io import BytesIO
from PIL import Image

from collections import defaultdict

import multiprocessing
import numpy as np
import queue
import threading
import bisect

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
from recovlm.utils.intern_vl_utils import process_vision_info_internvl,dynamic_preprocess,load_video,build_transform

from recovlm.models.internvl import InternVLChatConfig

from recovlm.training.parallel import get_sequence_parallel_group, \
  get_sequence_parallel_world_size
from recovlm.utils.common import print_rank_0, Timer

import glob
from recovlm.utils.common import shell_hdfs_ls, load_parquet_file
from .templates import get_template
from .prompts import PromptLoader
from .service import balance_sequence, DatasetServer
from recovlm.data import balance
from recovlm.services.clients import PidInfoClient


_DATASET_SKIP_MM = os.environ.get("_DATASET_SKIP_MM", "")
assert _DATASET_SKIP_MM in ["", "SKIP_MM"]
print(f"_DATASET_SKIP_MM={_DATASET_SKIP_MM}")



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
      raise SampleTooLongError(
          sample=sample,
          max_length=self.max_length,
          retry=retry
      )


class SampleTooLongError(Exception):
    """Exception raised when a sample exceeds maximum allowed length."""
    
    def __init__(self, sample, max_length, retry):
        self.sample = sample
        self.max_length = max_length
        self.retry = retry
        message = (f"Unable to generate sample within max_length={max_length} "
                  f"after {retry} retries. Sample length: {len(sample)}")
        super().__init__(message)
        
    def get_sample(self):
        """Get the problematic sample."""
        return self.sample
        
    def get_max_length(self):
        """Get the maximum allowed length."""
        return self.max_length
        
    def get_retry_count(self):
        """Get the number of retries attempted."""
        return self.retry
  
    def _packing(self, buffer: List[Dict[str, torch.Tensor]] ):
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
               datasource_config:Dict[str, Dict[str, Any]] = {},
               **kargs
               ):
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

    if isinstance(block["image"], str) and os.path.exists(block["image"]):
      image = Image.open(block["image"])
    elif isinstance(block["image"], str):
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
      if _DATASET_SKIP_MM == "SKIP_MM" and segment["type"] != "text": continue

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
    if inputs["input_ids"].shape[-1] > self.max_length:
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
      try:
        content = turn["content"]
        if isinstance(content, str):
          continue
        for block in content:
          if _DATASET_SKIP_MM == "SKIP_MM" and block["type"] != "text": continue

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
      except Exception as e:
        print(f"sample process error, messages={str(messages)[:500]}\n, sample=\n{str(sample)[:500]}")
        raise e

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
    if inputs["input_ids"].shape[-1] > self.max_length:
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
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, sample=\n{str(sample)[:500]}"
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
    print_input_info(
      inputs,
      "_process_completion",
    )
    if inputs["input_ids"].shape[1] <= inputs["pixel_values"].shape[0] * 256:
      print("baddddddd")
      print_input_info(
        inputs,
        "_process_completion",
      )
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
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=},  sample=\n{str(sample)[:500]}"
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
    self.num_readers = 8
    self.sample_queue = queue.Queue(1024)
    # self.lock = threading.Lock()

    # manager = multiprocessing.Manager()

    self.finish_dict_all = {}
    self.offset_dict_all = {}
    for i in range(self.num_workers):
      self.finish_dict_all[i] = {}
      self.offset_dict_all[i] = {}

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
        raise NotImplementedError(f"Unsupported dataloader checkpoint format. {tmp_finish_dict}") 
    
    for k, v in offset_dict.items():
      if isinstance(k, str):
        fn, group_idx = k.split("|")
        group_idx = int(group_idx)
        tmp_offset_dict[(fn, 0, group_idx)] = v
      elif isinstance(k, tuple) and len(k) == 3:
        tmp_offset_dict[k] = v
      else:
        raise NotImplementedError(f"Unsupported dataloader checkpoint format. {tmp_offset_dict}") 

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

      if messages is not None and isinstance(messages, list) and len(messages) > 0:
        sample_data["messages"] = messages
      elif segments is not None and isinstance(segments, list) and len(segments) > 0:
        sample_data["segments"] = segments
      elif messages is not None and isinstance(messages, np.ndarray):
        sample_data["messages"] = messages.tolist()
      else:
        raise NotImplementedError(f"Unsupported sample, message type is {type(messages)}, message={messages}, segments type is {type(segments)}, segments={segments}")

      samples["json"] = sample_data

      self._load_images_to_samples(images, samples, raw_row_data)

      return samples
    except:
      logger.error(f"ParquetDataset parse sample error!!! err_msg={traceback.format_exc()}, images={str(images)[:500]}\nsamples={str(samples)[:500]}")
      return None

  def _load_images_to_samples(self, images, samples, raw_row_data):
    # process images
    if isinstance(images, str):
      images = json.loads(images)
    elif isinstance(images, dict):
      pass
    else:
      raise NotImplementedError(f"Unsupported image field type, {type(raw_row_data['images'])=}")

    for image_name in images:
      image_b64 = images[image_name]
      # 先检查是否是有效文件路径
      if isinstance(image_b64, str) and os.path.exists(image_b64):
          try:
              image = Image.open(image_b64)
              samples[image_name] = image
          except Exception as e:
              raise ValueError(f"Failed to load image from path {image_b64}: {str(e)}")
      # 否则按base64处理
      else:
          try:
              image_bytes = base64.b64decode(image_b64)
              image_bytes_stream = BytesIO(image_bytes)
              image = Image.open(image_bytes_stream)
              samples[image_name] = image
          except Exception as e:
              raise ValueError(f"Failed to decode base64 image {image_name}: {str(e)}")
            
  def read_parquet_runner(self, fn_list, tid):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    finish_dict = self.finish_dict_all[worker]
    offset_dict = self.offset_dict_all[worker]
    try:
      for i, epoch_fn in enumerate(fn_list):
        if i % self.num_readers != tid:
          continue
        fn, epoch_idx = epoch_fn
        if (fn, epoch_idx) in finish_dict:
          logger.warning(f"[Rank{rank}-{worker}] skip {fn}")
          continue
        
        # open parquet file
        try:
          #parquet_file = pq.ParquetFile(fn)
          parquet_file = load_parquet_file(fn)

        except Exception as e:
          logger.error(f"ParquetDataset error, open parquet fail!!! {fn=}, error_msg={traceback.format_exc()}")
          parquet_file = None
        
        # # process file content
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
                    # yield sample
                    self.sample_queue.put(sample)
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
    
  def shuffle_runner(self, window):
    buffer = []
    while True:
      buffer.append(self.sample_queue.get())
      if len(buffer) == window:
        random.shuffle(buffer)
        for sample in buffer:
          self.shuffled_queue.put(sample)
        buffer = []

  def __iter__(self,):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    # assert num_workers == self.num_workers, f"{num_workers} : {self.num_workers}"

    finish_dict = self.finish_dict_all[worker]
    offset_dict = self.offset_dict_all[worker]

    total_num_workers = num_workers * world_size
    local_worker_idx = rank * num_workers + worker
    fn_list = [fn for idx, fn in enumerate(self.data_files) if idx % total_num_workers == local_worker_idx]
    logger.warning(
      f"ParquetDataset Info: {rank=}, {world_size=}, {worker=}, {num_workers=}, {len(fn_list)=}"
    )
    
    self.readers = []
    for i in range(self.num_readers):
      reader = threading.Thread(target=self.read_parquet_runner, args=(fn_list, i), daemon=True)
      reader.start()
      self.readers.append(reader)
      
    shuffle_window = 50000
    self.shuffled_queue = queue.Queue(shuffle_window * 2)
    self.shuffle_task = threading.Thread(target=self.shuffle_runner, args=(shuffle_window, ), daemon=True)
    self.shuffle_task.start()
    
    while True:
      sample = self.shuffled_queue.get()
      yield sample

  
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


    


class InternVLChatCompletionVisionDataset(IterableDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               max_length: int = 1024,
               shrink_ratio: float = 0.9,
               max_retry: int = 5,
               multiple_of: int = 8,
               shuffle_size: int = 100000,
               shuffle_initial_size: int = 20000,
               base_model_dir: Optional[str] = None,
               image_size:int=448,
               patch_size: int = 14,
               down_sample_ratio:float=0.5,
               min_dynamic_patch=1,
               max_dynamic_patch=12,
               normalize_type:str ='siglip',
               use_thumbnail:bool = True,
               num_segments:int= 10,
               datasource_config:Dict[str, Dict[str, Any]] = {},
               cut_to_pad:bool = False,
               **kargs):
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
      tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
      model_config = InternVLChatConfig.from_pretrained(base_model_dir)
      patch_size = model_config.vision_config.patch_size
      image_size = model_config.force_image_size
    
    self.tokenizer = tokenizer
    self.visual_tokens_per_image = int((image_size//patch_size)** 2 * (down_sample_ratio ** 2))
    self.cut_to_pad = cut_to_pad

    self.down_sample_ratio=down_sample_ratio
    self.pid_info_client = PidInfoClient('10.84.241.154')

    self.image_size = image_size
    self.patch_size = patch_size
    self.shrink_ratio = shrink_ratio
    self.max_retry = max_retry
    self.multiple_of = multiple_of
    self.shuffle_size = shuffle_size
    self.shuffle_initial_size = shuffle_initial_size
    self.min_dynamic_patch = min_dynamic_patch
    self.max_dynamic_patch = max_dynamic_patch
    self.normalize_type = normalize_type
    self.use_thumbnail = use_thumbnail

    self.img_context_token = '<IMG_CONTEXT>'
    self.img_start_token = '<img>'
    self.img_end_token = '</img>'

    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    self.end_of_text_id = self.tokenizer.encode('<|endoftext|>')[0]

    self.dataset, self.total_samples = self._build_source_dataset(sources)

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    # append image_pad for each packing
    self.image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    self.max_length = max_length
    self.image_pad = self._gen_img_pad()
    assert self.max_length - self.image_pad_len > 0

    self.datasource_config = datasource_config
    self.kargs = kargs
  
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

  def _gen_img_pad(self):
    """
    append an image, to trigger vit for pure text sample
    return 6 token: vstart, 4 * image_token, vend
    """
    def generate_base64_image():
        img = Image.fromarray(np.zeros((50,50, 3), dtype=np.uint8))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
      
    fake_sample = {
          "images": {"0.jpg":Image.fromarray(np.zeros((50,50, 3), dtype=np.uint8))
           #generate_base64_image()
           },
          "videos": None,
          "source": "__image_pad__",
          "messages": None,
          "segments": [
              # {"type": "text", "text": "0"},
              {"type": "image", "image": "0.jpg"},
          ],
          "metadata": None,
          "uuid": "23333333333112432536"
      }
    fake_sample["json"] = fake_sample
    fake_sample.update(fake_sample["images"])
    inputs = self._process_completion(fake_sample, data_conf={"min_dynamic_patch": 1, "max_dynamic_patch":1})
    inputs["loss_mask"] *= 0
    return inputs
    
  def _fill_image_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):

    if isinstance(block["image"], str) and os.path.exists(block["image"]):
      image = Image.open(block["image"])
    elif isinstance(block["image"], str):
      image = sample_dict[block["image"]]
    else:
      image = block["image"]

    if image.mode != "RGB":
      image = image.convert("RGB")
    block["image"] = image

  def _fill_video_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]
                        ):

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
      
      if isinstance(block["video"], str) and not os.path.exists(block["video"]):
        # media_path
        pid_info = self.pid_info_client.get_pid_info(block["video"].split(".")[0].split('/')[-1])
        if pid_info['media_type'] != 'video': raise ValueError(f"media_type={pid_info['media_type']} is not video")
        block["video"] = pid_info["media_path"]

    else:
      raise ValueError(f"Unsupport video type. {type(block['video'])=}")
      
  def _process_completion(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "segments" in sample["json"]
    data_conf["max_dynamic_patch"] = max(
        data_conf["max_dynamic_patch"], data_conf["min_dynamic_patch"])
    
    images = []
    new_conversations = []
    text = ""
    segments = sample["json"]["segments"]

    if _DATASET_SKIP_MM == "SKIP_MM": 
      segments = sample["json"]["segments"] = [x for x in segments if x["type"] == "text"]

    for segment in segments: 

      if segment["type"] == "image":
        self._fill_image_block(segment, sample,
                                conf=data_conf)
        turn_images = dynamic_preprocess(segment["image"], min_num=self.min_dynamic_patch, max_num=data_conf["max_dynamic_patch"],
                                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        images += [image for image in turn_images]
        num_image_tokens = self.visual_tokens_per_image * len(turn_images)
        text += f'{self.img_start_token}{self.img_context_token * num_image_tokens}{self.img_end_token}\n'

      elif segment["type"] == "video":
        self._fill_video_block(segment, sample,
                                conf=data_conf)
        
        nframes = []
        num_patches_list = []
        if isinstance(segment["video"], str) and "480p_60s_4fps" in segment["video"]:
            path = segment["video"]
            pid_str = osp.basename(osp.splitext(path)[0])
            if not osp.exists(path):
                post = str(int(pid_str[-4:]))
                path = path.replace("480p_60s_4fps_v2", "480p_60s_4fps_0215_0316/{}".format(post))
            nframes,num_patches_list = load_video(path,num_segments = self.num_segments)

        elif isinstance(segment["video"],list):
            for img in segment["video"]:
                imgs = dynamic_preprocess(img["image"], min_num=self.min_dynamic_patch, max_num=data_conf["max_dynamic_patch"],
                                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                num_patches_list.append(len(imgs))
                nframes += imgs

        else:
            raise ValueError(f"process_vision_info_internvl failed,failed type {segment}")
        
        for i,num_image in enumerate(num_patches_list):
            #当前帧的token数
            num_image_tokens = self.visual_tokens_per_image * num_image
            text += f"Frame{i+1}: {self.img_start_token}{self.img_context_token * num_image_tokens}{self.mg_end_token}\n"
            
        images += nframes
      elif segment["type"] == "text":
        text += segment["text"]
      else:
        logger.warning(f"!!! Unsupport {segment['type']=}, skip this segment.")

    # append EOS token
    text += "<|endoftext|>"
    inputs = self.tokenizer(
        text=text,
        return_tensors="pt"
    )

    image_flag = 1 if len(images) > 0 else 0
    # 如果是纯文本增加一张图片做引导
    # if image_flag==0:
    #   image = Image.new('RGB', (224, 224), (255, 255, 255))
    #   images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
    #                                     image_size=self.image_size, use_thumbnail=self.use_thumbnail)

    if image_flag:
      transform = build_transform(is_train=True, input_size=self.image_size,normalize_type=self.normalize_type)
      pixel_values = [transform(image) for image in images]
      pixel_values = torch.stack(pixel_values)
      inputs["pixel_values"] = pixel_values
      inputs["image_flags"] = torch.tensor([image_flag] * len(images), dtype=torch.long)
    else:
      inputs["pixel_values"] = self.image_pad["pixel_values"][:0]
      inputs["image_flags"] = self.image_pad["image_flags"][:0]
    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > 32768:
      print(f"Sample is too long. token_len={inputs['input_ids'].shape[-1]}")
    
    input_ids = inputs["input_ids"]
    inputs["loss_mask"] = torch.ones_like(input_ids)
    inputs["loss_mask"][
        (input_ids == self.img_start_token_id) | 
        (input_ids == self.img_end_token_id) |
        (input_ids == self.img_context_token_id)
      ] = 0
    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
    if inputs["loss_mask"].sum() == 0:
      raise ValueError(
        f"Unable to generate sample with 0 loss_mask."
      )

    position_ids = inputs['attention_mask'].long().cumsum(-1) - 1
    position_ids.masked_fill_(inputs['attention_mask'] == 0, 1)
    inputs["position_ids"] = position_ids
    inputs.pop("attention_mask")

    if inputs["input_ids"].shape[1] <= inputs["pixel_values"].shape[0] * 256:
      print("baddddddd")
      print_input_info(
        inputs,
        "_process_completion",
      )
    return inputs
    

  def _process_chat(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "message" in sample["json"] or "messages" in sample["json"]
    data_conf["max_dynamic_patch"] = max(
        data_conf["max_dynamic_patch"], data_conf["min_dynamic_patch"])
    
    msg_key = "message" if "message" in sample["json"] else "messages"
    messages = sample["json"][msg_key]
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue

      if _DATASET_SKIP_MM == "SKIP_MM":
        content = turn["content"] = [x for x in content if x["type"] == "text"]

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

    #inputs 输出 input_ids,label,attention_mask,pixel_values,image_flags
    inputs = process_vision_info_internvl(messages,self.tokenizer,self.visual_tokens_per_image,
                                          self.min_dynamic_patch,data_conf["max_dynamic_patch"],
                                          self.use_thumbnail,self.image_size,self.img_start_token,
                                          self.img_context_token,self.img_end_token,self.normalize_type)

    if "pixel_values" not in inputs:
      inputs["pixel_values"] = self.image_pad["pixel_values"][:0]
      inputs["image_flags"] = self.image_pad["image_flags"][:0]


    if inputs["input_ids"].shape[1] <= inputs["pixel_values"].shape[0] * 256:
      print("baddddddd")
      print_input_info(
        inputs,
        "_process_chat",
      )

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    # if inputs["input_ids"].shape[-1] > 32768:
    #   raise ValueError(f"Sample is too long, token_len={inputs['input_ids'].shape[-1]}")
    
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
    position_ids = inputs['attention_mask'].long().cumsum(-1) - 1
    position_ids.masked_fill_(inputs['attention_mask'] == 0, 1)
    inputs["position_ids"] = position_ids
    inputs.pop("attention_mask")
    return inputs


  def _process(self, sample, source_name=None):
    # self._may_filter(sample)
    # get data format
    if ("messages" in sample["json"] and sample["json"]['messages'] is not None and len(sample["json"]['messages']) ) or \
          ("message" in sample["json"] and sample["json"]['message'] is not None and len(sample["json"]['message']) ):      
      data_format = "chatml"
    elif "segments" in sample["json"]:
      data_format = "completion"
    else:
      raise NotImplementedError(f"Unsupported dataset format.")
    
    source_conf = {
      "max_dynamic_patch":self.max_dynamic_patch,
      "min_dynamic_patch":self.min_dynamic_patch
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

      if inputs["input_ids"].shape[-1] > self.max_length - self.image_pad_len:
        source_conf["max_dynamic_patch"] = int(source_conf["max_dynamic_patch"]*self.shrink_ratio)
        continue
      else:
        assert inputs["input_ids"].shape[-1] <= self.max_length - self.image_pad_len, "inputs too long"
        return inputs
    else:
      raise ValueError(
          f"Unable to generate sample within max_length={self.max_length - self.image_pad_len} after {retry} retrys"
      )
  
  def _append_sample_packing(self,
                             inputs: Dict[str, torch.Tensor],
                             packed_input_ids: List[torch.Tensor],
                             packed_position_ids: List[torch.Tensor],
                             packed_loss_mask: List[torch.Tensor],
                             packed_pixel_values: List[torch.Tensor],
                             packed_pixel_values_videos: List[torch.Tensor],
                             packed_image_gird_thw: List[torch.Tensor], # dont care
                             packed_video_grid_thw: List[torch.Tensor], # dont care
                             packed_sample_idx: List[torch.Tensor],
                             packed_image_flags:List[torch.Tensor],
                             cu_seqlens: List[int],
                             sample_idx: Optional[int] = None,
                             ):
    if self.cut_to_pad:
      '''
      input_ids格式如下：
      inputs: Dict: keys=5
      inputs: 'input_ids':
      inputs:   Tensor: shape=(1, 3369), dtype=torch.int64, device=cpu, data=tensor([151665, 151667, 151667, 151667])...tensor([ 45436,   3589,     13, 151643])
      inputs: 'pixel_values':
      inputs:   Tensor: shape=(13, 3, 448, 448), dtype=torch.float32, device=cpu, data=tensor([0.1451, 0.1608, 0.1843, 0.2078])...tensor([0.2549, 0.2627, 0.2627, 0.2627])
      inputs: 'image_flags':
      inputs:   Tensor: shape=(13,), dtype=torch.int64, device=cpu, data=tensor([1, 1, 1, 1])...tensor([1, 1, 1, 1])
      inputs: 'loss_mask':
      inputs:   Tensor: shape=(1, 3369), dtype=torch.int64, device=cpu, data=tensor([0, 0, 0, 0])...tensor([1, 1, 1, 0])
      inputs: 'position_ids':
      inputs:   Tensor: shape=(1, 3369), dtype=torch.int64, device=cpu, data=tensor([0, 1, 2, 3])...tensor([3365, 3366, 3367, 3368])
      '''
      packable_length = self.max_length - self.image_pad_len - cu_seqlens[-1]

      if sample_idx is None and packable_length < inputs["input_ids"].size(1): # 1 x len, 不是image padding才有这个逻辑
        # if dist.get_rank() == 0:
        #   print_input_info(inputs, prefix="inputs_cut_before: ")
        inputs["input_ids"] = inputs["input_ids"][:, :packable_length]
        inputs["loss_mask"] = inputs["loss_mask"][:, :packable_length]
        inputs["position_ids"] = inputs["position_ids"][:, :packable_length]

        # if inputs["input_ids"][0, -1] in [self.img_start_token_id, self.img_context_token_id]:
        last_start_index = torch.nonzero(inputs["input_ids"][0] == self.img_start_token_id)
        if len(last_start_index) == 0: last_start_index = packable_length # 这里没有图片
        else: last_start_index = last_start_index[-1].item()

        inputs["input_ids"][:, last_start_index:] = 0 # 随便一个id, 反正不要图片id
        inputs["loss_mask"][:, last_start_index:] = 0 # 不要计算loss

        num_tiles_ids = torch.nonzero(inputs["input_ids"][0] == self.img_context_token_id).size(0) # 计算留下多少tile
        assert num_tiles_ids % 256 == 0, f"num_tiles_ids should be multiple of 256, get {num_tiles_ids}"
        num_tiles = num_tiles_ids // 256
        # cu_seqlens
        inputs["pixel_values"] = inputs["pixel_values"][:num_tiles]
        inputs["image_flags"] = inputs["image_flags"][:num_tiles]

        # if dist.get_rank() == 0:
        #   print_input_info(inputs, prefix="inputs_cut_im: ")


        assert inputs["input_ids"].shape ==  inputs["loss_mask"].shape == inputs["position_ids"].shape and inputs["input_ids"].ndim == 2, f'inputs: {inputs["input_ids"].shape} ==  {inputs["loss_mask"].shape} == {inputs["position_ids"].shape}'
        assert inputs["image_flags"].size(0) == inputs["pixel_values"].size(0), f'inputs: {inputs["image_flags"].shape}, {inputs["pixel_values"].shape}'

    packed_input_ids.append(inputs["input_ids"].flatten())
    packed_loss_mask.append(inputs["loss_mask"].flatten())
    packed_position_ids.append(inputs["position_ids"])
    if sample_idx is None:
      sample_idx = len(cu_seqlens) - 1
    packed_sample_idx.append(
      torch.full_like(packed_input_ids[-1], sample_idx))

    if "pixel_values" in inputs:
      packed_pixel_values.append(inputs["pixel_values"])
    if "pixel_values_videos" in inputs:
      packed_pixel_values_videos.append(inputs["pixel_values_videos"])

    cu_seqlens.append(cu_seqlens[-1] + len(inputs["input_ids"][0]))
    packed_image_flags.append(inputs["image_flags"])
    


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
    packed_image_flags:List[torch.Tensor] = []
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
                                      packed_image_flags,
                                      cu_seqlens)

    valid_seq_len += self._append_sample_packing(self._gen_img_pad(),
                                      packed_input_ids,
                                      packed_position_ids,
                                      packed_loss_mask,
                                      packed_pixel_values,
                                      packed_pixel_values_videos,
                                      packed_image_gird_thw,
                                      packed_video_grid_thw,
                                      packed_sample_idx,
                                      packed_image_flags,
                                      cu_seqlens,
                                      sample_idx=-1)

    if self.cut_to_pad: assert valid_seq_len == self.max_length, f"set cut_to_pad={self.cut_to_pad}, then require valid_seq_len/{valid_seq_len} == self.max_length/{self.max_length}"
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

    packed_image_flags = None if len(packed_image_flags) == 0 else \
      torch.cat(packed_image_flags, dim=0)


    # pad seq len to multiple_of
    if (
      self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
    ):
      padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
      # assert self.max_length % self.multiple_of == 0

      if self.cut_to_pad: assert padding_len == 0, f"padding_len={padding_len}, not equal to 0"

      packed_input_ids = F.pad(
        packed_input_ids, (0, padding_len),
        value=self.tokenizer.pad_token_id)
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
      "image_flags":packed_image_flags,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
      "sample_idx": packed_sample_idx.to(torch.int32)
    }
    if packed_input_ids.flatten().shape[0] < packed_pixel_values.shape[0] * 256:
      print_input_info(inputs, "inputs111: ")
      raise Exception("!!!!!! Error occurs in image padding. input ids are shorted than image tokens")
    return inputs

  def _find_in_range(self, seq_lens, max_len, delta, target_count, max_trials=5000):
    t1 = time.perf_counter()
    results = set()
    trials = 0 

    while len(results) < target_count and trials < max_trials:
        trials += 1

        # Randomly shuffle indices
        indices = list(range(len(seq_lens)))
        random.shuffle(indices)

        current_sum = 0 
        current_indices = []

        for idx in indices:
            if current_sum + seq_lens[idx] <= max_len:
                current_sum += seq_lens[idx]
                current_indices.append(idx)

            if max_len - delta <= current_sum <= max_len:
                results.add(tuple(sorted(current_indices)))  # use tuple to be hashable
                break  # once one valid subset found, restart next trial

    t2 = time.perf_counter()
    print(f"Found {len(results)} valid subsets in {trials} trials, dur={t2-t1}")
    return [list(subset) for subset in results]
  
  def _process_task(self):
    while True:
      sample = self.sample_queue.get()
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
        self.processed_buffer.put((inputs, source_name))
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

  def _balance_score(self, image_lens):
      max_len = max(image_lens)
      min_len = min(image_lens)
      target_len = min(max_len, 55)
      var = sum((v - target_len) ** 2 for v in image_lens) / len(image_lens)
      var = float(np.sqrt(var)) if var > 0 else 0
      return (max_len - min_len) ** 2 + var


  def _select_nearest_equal(self, local_image_lens, all_image_lens):
      def find_nearest(arr, target):
          pos = bisect.bisect_left(arr, target)
          if pos == 0:
              return arr[0]
          if pos == len(arr):
              return arr[-1]
          if target == arr[pos]:
              return target
          before, after = arr[pos-1], arr[pos]
          if after - target < target - before:
              return after
          else:
              return before
      found = None
      min_score = sys.maxsize
      debug_info = []
      for n in local_image_lens:
          current = []
          for i, arr in enumerate(all_image_lens):
              if i == dist.get_rank():
                  current.append(n)
              else:
                  current.append(find_nearest(arr, n))
          score = self._balance_score(current)
          debug_info.append((score, current))
          if score < min_score or (score == min_score and sum(current) > sum(found)):
              found = current
              min_score = score

      # print(f'[rank={dist.get_rank()}] debug_info: {debug_info}')
      return found + [min_score]

  def _select_global(self, candidates):
      scores = [candidate[-1] for candidate in candidates]
      idx = int(np.argmin(scores))
      return candidates[idx][:-1]
      # min_var = sys.maxsize
      # found = None
      # max_sum = -1
      # for candidate in candidates:
      #     cur_var = max(candidate) - min(candidate)
      #     if cur_var < min_var:
      #         found = candidate
      #         min_var = cur_var
      #         max_sum = sum(candidate)
      #     elif cur_var == min_var:
      #         cur_sum = sum(candidate)
      #         if cur_sum > max_sum:
      #             max_sum = cur_sum
      #             found = candidate

      # return found 
  
  def _prefetched_task(self, delta_ratio: float = 0.02, buffer_size: int = 1000, target_count: int = 100):
    delta = int(self.max_length * delta_ratio)
    buffer = []
    source_list = []
    while True:
      inputs, source_name = self.processed_buffer.get()
      buffer.append(inputs)
      source_list.append(source_name)
      if len(buffer) == buffer_size:
        raw_input_ids = [data["input_ids"].shape[-1] for data in buffer]
        raw_image_len = [data["pixel_values"].size(0) for data in buffer]
        if dist.get_rank() == 0:
          print(f"[rank={dist.get_rank()}] raw_input_ids_len={sorted(raw_input_ids)}, raw_image_len={sorted(raw_image_len)}")

        t1 = time.perf_counter()
        # small_input_ids = balance.sampling(raw_input_ids, 200)
        t2 = time.perf_counter()
        # sampling_index = [raw_input_ids.index(v) for v in small_input_ids]
        # small_image_len = [raw_image_len[i] for i in sampling_index]
        # if dist.get_rank() == 0:
        #   print(f"small_ids: {small_input_ids}, small_img: {small_image_len}, idx: {sampling_index}")
        # candidates = balance.greedy_subsets_nearst_sum(small_input_ids, self.max_length)
        candidates = balance.greedy_subsets_nearst_sum(raw_input_ids, self.max_length)
        if dist.get_rank() == 0:
          print(f"candidates: {candidates}")
        candidates = candidates[:50]
        flops = []
        len_info = []
        for c in candidates:
          llm_len = [raw_input_ids[i] for i in c]
          vit_len = [raw_image_len[i] for i in c]
          len_info.append((llm_len, vit_len))
          flops.append(balance.llm_flops(llm_len))
          flops.append(balance.vit_flops(vit_len))
        t3 = time.perf_counter()
        all_flops = [None] * dist.get_world_size()
        dist.all_gather_object(all_flops, flops)
        t4 = time.perf_counter()
        print(f"rank={dist.get_rank()} len_info={len_info}, flops={flops}")
        if dist.get_rank() == 0:
          print(f"[rank=0] all_flops: {all_flops}")
        local_best = balance.select_by_flops(all_flops, dist.get_rank())
        t5 = time.perf_counter()
        local_best_flat = [v for sub in local_best for v in sub]
        all_local = [None] * dist.get_world_size()
        dist.all_gather_object(all_local, local_best_flat)
        t6 = time.perf_counter()
        selected = balance.find_global(all_local)
        print(f"rank={dist.get_rank()} local_best={local_best}, all_local={all_local}, global_best={selected}")
        local_selected = selected[dist.get_rank()]
        found = -1
        for i in range(0, len(flops) // 2, 2):
            if math.isclose(flops[2*i], local_selected[0], rel_tol=1e-6) and math.isclose(flops[2*i+1], local_selected[1], rel_tol=1e-6):
                found = i
                break
        if found == -1:
            print(f"not_found rank={dist.get_rank()}, flops={flops}, sel={local_selected}")
        assert found >= 0
        # selected_index = [sampling_index[i] for i in candidates[found]]
        selected_index = candidates[found]
        selected_llm = [raw_input_ids[i] for i in selected_index]
        selected_vit = [raw_image_len[i] for i in selected_index]
        t7 = time.perf_counter()
        print(f"[rank={dist.get_rank()}] llm={selected_llm} vit={selected_vit}")
        print(f"sampling={t2-t1}, find_subsets={t3-t2}, gather1={t4-t3}, balance={t5-t4}, gather2={t6-t5}, other={t7-t6}")

        # t1 = time.perf_counter()
        # # candidates = self._find_in_range(raw_input_ids, self.max_length, delta, target_count)
        # input_ids_len = [sum(buffer[idx]["input_ids"].shape[-1] for idx in candidate) for candidate in candidates]
        # image_len = [sum(buffer[idx]["pixel_values"].size(0) for idx in candidate) for candidate in candidates]
        # print(f"[rank={dist.get_rank()}]  candidate_images: {sorted(image_len)}, candidate_llm: {input_ids_len}")
        # sorted_image_len = sorted(image_len)
        # t2 = time.perf_counter()
        # all_image_lens = [None] * dist.get_world_size()
        # dist.all_gather_object(all_image_lens, sorted_image_len)
        # local_found = self._select_nearest_equal(sorted_image_len, all_image_lens)
        # all_local_found = [None] * dist.get_world_size()
        # dist.all_gather_object(all_local_found, local_found)
        # selected_len = self._select_global(all_local_found)
        # if dist.get_rank() == 0:
        #   print(f"[rank={dist.get_rank()}] selected_global: {selected_len}, diff={max(selected_len) - min(selected_len)}")
        # selected_index = candidates[image_len.index(selected_len[dist.get_rank()])]
        t3 = time.perf_counter()
        packed_inputs = self._packing([buffer[idx] for idx in selected_index])
        t4 = time.perf_counter()
        print(f"[rank={dist.get_rank()}]find_input_ids={t2-t1}, balance_image={t3-t2}, packing={t4-t3}")
        packed_inputs["data_source"] = [source_list[idx] for idx in selected_index]
        self.cache.put(packed_inputs)
        
        buffer = [x for i, x in enumerate(buffer) if i not in selected_index]
        source_list = [x for i, x in enumerate(source_list) if i not in selected_index]

  def __iter__(self):
    # rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    # world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    # if not torch.distributed.is_initialized():
    #     print(f'init process_group in dataset, {rank}, {world_size}, {worker_id}')
    #     dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    self.cache = queue.Queue(maxsize=1)
    delta_ratio = self.kargs.get("input_ids_len_delta_ratio", 0.02)
    buffer_size = self.kargs.get("balance_buffer_size", 1000)
    target_count = self.kargs.get("balance_candidate_count", 100)

    self.sample_queue = queue.Queue(maxsize=32)
    def reader_task():
        dataset_iter = iter(self.dataset)
        while True:
            sample = next(dataset_iter)
            self.sample_queue.put(sample)
    self.reader_thread = threading.Thread(target=reader_task, daemon=True)
    self.reader_thread.start()

    self.processed_buffer = queue.Queue(buffer_size)
    self.process_threads = [threading.Thread(target=self._process_task, daemon=True) for _ in range(16)]
    for t in self.process_threads:
      t.start()
    self.prefetch_thread = threading.Thread(target=self._prefetched_task, args=(delta_ratio, buffer_size, target_count), daemon=True)
    self.prefetch_thread.start()

    while True:
        t1 = time.perf_counter()
        result = self.cache.get()
        t2 = time.perf_counter()
        print(f'next_batch[{dist.get_rank()}]={t2-t1}')
        if False:
            yield result
        else:
            continue

  def __iter_v2__(self):
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
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, sample=\n{str(sample)[:500]}"
          f"errmsg={traceback.format_exc()}")
        continue

      sample_length = inputs["input_ids"].shape[-1]
      if cur_length + sample_length > self.max_length - self.image_pad_len:

        if self.cut_to_pad:
          buffer.append(inputs)
          source_list.append(source_name)
          packed_inputs = self._packing(buffer)

          packed_inputs["data_source"] = source_list
          buffer = []
          source_list = []
          cur_length = 0
          if packed_inputs["loss_mask"].sum().item() == 0:
            continue # packing失败，这种情况通常是只有一个样本，而且这个样本以图片开头，而且图片占满了所有有效token
        else:
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


class InternVLChatCompletionVisionParquetDataset(InternVLChatCompletionVisionDataset):
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



