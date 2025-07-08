from typing import Union, Iterable, Optional, List, Dict, Tuple, Any
import logging
import copy
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
import os.path as osp
import webdataset as wds
from recovlm.utils.ds_utils import print_input_info
from recovlm.data.image_augs import AutoAugmentWrapper
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


from recovlm.models.qwen_2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor_moonvit,Qwen2_5_VLProcessor_siglip
from recovlm.models.qwen_2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from recipes.ViT.training.models.MoonVision.configuration_kimi_vl import MoonViTConfig
from recipes.ViT.training.models.siglip.configuration_siglip import SiglipConfig

from recovlm.models.qwen3siglip.processing_qwen3siglip import Qwen3SiglipProcessor_navit
from recovlm.models.keye.processing_keye import KeyeProcessor
from recovlm.models.keye.modeling_keye import KeyeConfig
from recovlm.models.keye.keye_vl_utils import process_vision_info as process_vision_info_keye
from recovlm.models.keye_vitrope.keye_vl_utils import process_vision_info as process_vision_info_keye_vitrope
from recovlm.models.keye_vitrope_slowfast.keye_vl_utils import process_vision_info as process_vision_info_keye_vitrope_slowfast

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
from recovlm.data import balance, transfer
from recovlm.services.clients import PidInfoClient


_DATASET_SKIP_MM = os.environ.get("_DATASET_SKIP_MM", "")
assert _DATASET_SKIP_MM in ["", "SKIP_MM", "SKIP_VI"]
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
      tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
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
               cut_to_pad=True,
               min_visual_tokens_per_frame: int = 4,
               max_visual_tokens_per_frame: int = 512,
               use_flops_balance=False,
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
      processor = Qwen2VLProcessor.from_pretrained(base_model_dir)
      model_config = Qwen2VLConfig.from_pretrained(base_model_dir)
      spatial_merge_size = model_config.vision_config.spatial_merge_size
      patch_size = model_config.vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    self.use_flops_balance = use_flops_balance
    self.process_vision_info = process_vision_info
    self.auto_aug = AutoAugmentWrapper(policy=kargs.get("autoaug_policy", None))
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")

    self.processor = processor
    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.min_visual_tokens_per_frame = min_visual_tokens_per_frame
    self.max_visual_tokens_per_frame = max_visual_tokens_per_frame
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

    if self.use_flops_balance: self.dataset, self.total_samples = None, None
    else:  self.dataset, self.total_samples = self._build_source_dataset(sources)
    self.sources = sources

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = self._gen_img_pad()["input_ids"].shape[-1] # 6
    
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    kargs["use_flops_balance"] = self.use_flops_balance
    self.kargs = self.kwargs = kargs
  
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

    if isinstance(block["image"], 
    str) and os.path.exists(block["image"]):
      image = Image.open(block["image"])
    elif isinstance(block["image"], str):
      image = sample_dict[block["image"]]
    else:
      image = block["image"]
    if image.mode != "RGB":
      image = image.convert("RGB")
    image = self.auto_aug(image)
    block["image"] = image
    block["min_pixels"] = min_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
    block["max_pixels"] = max_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
  
  def _fill_video_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):

    min_visual_tokens_per_frame = conf["min_visual_tokens_per_frame"]
    max_visual_tokens_per_frame = conf["max_visual_tokens_per_frame"]

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
      block["min_pixels"] = min_visual_tokens_per_frame * (self.patch_size ** 2) * \
          (self.spatial_merge_size ** 2)
      block["max_pixels"] = max_visual_tokens_per_frame * (self.patch_size ** 2) * \
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
    data_conf["max_visual_tokens_per_frame"] = max(
        data_conf["max_visual_tokens_per_frame"], data_conf["min_visual_tokens_per_frame"]) 

    text = ""
    vision_infos = []
    
    if _DATASET_SKIP_MM == "SKIP_MM": sample["json"]["segments"] = [x for x in sample["json"]["segments"] if x['type'] == 'text']
    if _DATASET_SKIP_MM == "SKIP_VI": sample["json"]["segments"] = [x for x in sample["json"]["segments"] if x['type'] != 'video']
    segments = sample["json"]["segments"]
    for segment in segments:
      # if _DATASET_SKIP_MM == "SKIP_MM" and segment["type"] != "text": continue
      # if _DATASET_SKIP_MM == "SKIP_VI" and segment["type"] == "video": continue

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
    text += self.kargs.get("endoftext", "<|endoftext|>")

    time0 = time.time()
    # 这里做一个调整，process_vision_info_args默认为空字典（不会生效）
    # 但是允许用户传入process_vision_info_args相关参数，主要是navit的时候，可以传入image_factor=None,从而不对图片进行resize，而是让self.processor负责resize
    image_inputs, video_inputs = self.process_vision_info(vision_infos = vision_infos, **self.process_vision_info_args)
    inputs = self.processor( 
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )
    if time.time() - time0 > 4:
      print(f"long process time source={sample['json']['source']}, it consumes {time.time() - time0} secs", )

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
    
    data_conf["max_visual_tokens_per_frame"] = max(
        data_conf["max_visual_tokens_per_frame"], data_conf["min_visual_tokens_per_frame"])
    
    msg_key = "message" if "message" in sample["json"] else "messages"
    messages = sample["json"][msg_key]
    for turn in messages:
      try:
        content = turn["content"]
        if isinstance(content, str):
          continue

        if _DATASET_SKIP_MM == "SKIP_MM": turn["content"] = [x for x in turn["content"] if x['type'] == 'text']
        content = turn["content"]
        for block in content:
          # if _DATASET_SKIP_MM == "SKIP_MM" and block["type"] != "text": continue
          if _DATASET_SKIP_MM == "SKIP_VI" and block["type"] == "video": continue

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
        print(f"sample process error, messages={str(messages)[:50]}\n, sample=\n{str(sample)[:50]}")
        raise e

    text = self.processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=False
    )


    # append EOS token
    text += self.kargs.get("endoftext", "<|endoftext|>")
    # 这里做一个调整，process_vision_info_args默认为空字典（不会生效）
    # 但是允许用户传入process_vision_info_args相关参数，主要是navit的时候，可以传入image_factor=None,从而不对图片进行resize，而是让self.processor负责resize

    time0 = time.time()
    image_inputs, video_inputs = self.process_vision_info(messages, **self.process_vision_info_args)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    if time.time() - time0 > 4: 
      print(f"long process time source={sample['json']['source']}, it consumes {time.time() - time0} secs", )

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > self.max_length:
      raise ValueError(f"Sample is too long. text_len={len(text)=}, token_len={inputs['input_ids'].shape[-1]}")
    
    inputs["loss_mask"] = get_assistant_mask(
      inputs["input_ids"],
      start_pattern=self.kargs.get("start_pattern", [151644, 77091, 198]), # [151644, 77091, 198],
      end_pattern=self.kargs.get("end_pattern", [151645, 198]), # self.end_pattern, #[151645, 198]
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
  
  def _gen_img_pad(self, with_vid=True, sz=(16,16)):
    """
    append an image, to trigger vit for pure text sample
    return 6 token: vstart, 4 * image_token, vend
    """
    # Image.fromarray(np.zeros((50,50, 3), dtype=np.uint8))
    text = "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>" if with_vid else "<|vision_start|><|image_pad|><|vision_end|>"
    pad_image = {
        "type": "image",
        "image": Image.fromarray(np.zeros((*sz, 3), dtype=np.uint8)) # Image.new("RGB", (3, 1, 1), (255, 255, 255))
    }
    pad_video = {
        "type": "video",
        "video": [{"type": "image", "image": Image.fromarray(np.zeros((16,16, 3), dtype=np.uint8))}],
    }
    source_conf = {
      "min_visual_tokens_per_image": self.min_visual_tokens_per_image,
      "max_visual_tokens_per_image": self.max_visual_tokens_per_image,
      "min_visual_tokens_per_frame": self.min_visual_tokens_per_frame,
      "max_visual_tokens_per_frame": self.max_visual_tokens_per_frame, 
      "video_nframe": self.video_nframe,
      "video_fps": self.video_fps,
      "video_min_frames": self.video_min_frames,
      "video_max_frames": self.video_max_frames
    }
    if 'only_slow' in self.kargs:
      source_conf["only_slow"] = self.kargs["only_slow"]
    if 'max_slow_frames' in self.kargs:
      source_conf["max_slow_frames"] = self.kargs["max_slow_frames"]
    self._fill_image_block(pad_image, sample_dict={}, conf=source_conf)
    self._fill_video_block(pad_video, sample_dict={}, conf=source_conf)
    image_inputs, video_inputs = self.process_vision_info(vision_infos=[pad_image, pad_video] if with_vid else [pad_image])
    print("self.processorself.processor", type(self.processor))
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs if with_vid else None,
        return_tensors="pt",
        image_video_pad=True,
    )
    print("self.processorself.processor__donnnn")
    # tensor([[151652, 151655, 151655, 151655, 151655, 151653, 151652, 151656, 151656, 151656, 151656, 151656, 151656, 151656, 151656, 151653]])
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

  # def _gen_vid_pad(self):
  #   """
  #   append an image, to trigger vit for pure text sample
  #   return 6 token: vstart, 4 * image_token, vend
  #   """
  #   # Image.fromarray(np.zeros((50,50, 3), dtype=np.uint8))
  #   text = "<|vision_start|><|image_pad|><|vision_end|>"
  #   pad_image = {
  #       "type": "image",
  #       "image": Image.fromarray(np.zeros((16,16, 3), dtype=np.uint8)) # Image.new("RGB", (3, 1, 1), (255, 255, 255))
  #   }

  #   self._fill_image_block(pad_image, sample_dict={}, conf={
  #       "min_visual_tokens_per_image": self.min_visual_tokens_per_image,
  #       "max_visual_tokens_per_image": self.max_visual_tokens_per_image,
  #       "video_nframe": self.video_nframe,
  #       "video_fps": self.video_fps,
  #       "video_min_frames": self.video_min_frames,
  #       "video_max_frames": self.video_max_frames
  #   })
  #   image_inputs, _ = self.process_vision_info(vision_infos=[pad_image])
  #   inputs = self.processor(
  #       text=text,
  #       images=image_inputs,
  #       videos=None,
  #       return_tensors="pt"
  #   )

  #   inputs["loss_mask"] = torch.zeros_like(inputs["input_ids"])
  #   inputs["position_ids"] = get_rope_index(
  #       inputs["input_ids"],
  #       image_grid_thw=inputs.get("image_grid_thw"),
  #       video_grid_thw=inputs.get("video_grid_thw"),
  #       spatial_merge_size=self.spatial_merge_size,
  #       image_token_id=self.image_token_id,
  #       video_token_id=self.video_token_id,
  #       vision_start_token_id=self.vision_start_token_id
  #   )
  #   inputs.pop("attention_mask")
  #   return inputs

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
      "min_visual_tokens_per_frame": self.min_visual_tokens_per_frame,
      "max_visual_tokens_per_frame": self.max_visual_tokens_per_frame, 
      "video_nframe": self.video_nframe,
      "video_fps": self.video_fps,
      "video_min_frames": self.video_min_frames,
      "video_max_frames": self.video_max_frames
    }

    if 'only_slow' in self.kargs:
      source_conf["only_slow"] = self.kargs["only_slow"]
    if 'max_slow_frames' in self.kargs:
      source_conf["max_slow_frames"] = self.kargs["max_slow_frames"]

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
      inputs['epoch_idx'] = sample['epoch_idx']
      if not inputs:
        raise ValueError("Empty inputs, skip")

      process_max_length = min(int(self.max_length // 1.5), 8000) if self.use_flops_balance else self.max_length
      if inputs["input_ids"].shape[-1] > process_max_length:
        source_conf["max_visual_tokens_per_image"] = (
            source_conf["max_visual_tokens_per_image"] * self.shrink_ratio)
        source_conf["max_visual_tokens_per_frame"] = (
            source_conf["max_visual_tokens_per_frame"] * self.shrink_ratio)
        continue
      else:
        assert inputs["input_ids"].shape[-1] <= process_max_length, "inputs too long"
        lenf = inputs["input_ids"].shape[-1]

        # print(f"rank{dist.get_rank()}_process{lenf}=============== ")
        # print_input_info(
        #   inputs,
        #   f"rank{dist.get_rank()}_process{lenf}: "
        # )
        return inputs
    else:
      raise ValueError(
          f"Unable to generate sample within max_length={process_max_length} after {retry} retrys"
      )
      
  def _cut_sample(self, inputs, packable_length):
    # if 'pixel_values_videos' in inputs and dist.get_rank() == 0:

    inputs["input_ids"] = inputs["input_ids"][:, :packable_length]
    inputs["loss_mask"] = inputs["loss_mask"][:, :packable_length]

    inputs["position_ids"] = inputs["position_ids"][..., :packable_length]

    vision_starts = torch.nonzero(inputs["input_ids"][0] == self.vision_start_token_id)
    vision_ends = torch.nonzero(inputs["input_ids"][0] == self.vision_end_token_id)

    if len(vision_starts) and len(vision_starts) > len(vision_ends): # 说明图片不完整
      # inputs["input_ids"][:, vision_starts[-1]:] = 0 # 随便什么id都可以
      # inputs["loss_mask"][:, vision_starts[-1]:] = 0
      # inputs["position_ids"][:, vision_starts[-1]:] = 0
      inputs["input_ids"] = inputs["input_ids"][:, :vision_starts[-1]] # 继续截断,截断到vision_starts token,因为vision_start之后的内容都不会有loss
      inputs["loss_mask"] = inputs["loss_mask"][:, :vision_starts[-1]]
      inputs["position_ids"] = inputs["position_ids"][..., :vision_starts[-1]]
      
    if 'image_grid_thw' in inputs and len(inputs["pixel_values"]) and 'video_grid_thw' in inputs and len(inputs["pixel_values_videos"]):
      raise Exception("Unexpected inputs: there are both pixel_values and pixel_values_videos: {}/{}".format(inputs["pixel_values"].shape, inputs["pixel_values_videos"].shape))

    if 'image_grid_thw' in inputs: # 如果有图片
      n_tokens = 0
      for i in range(len(vision_ends), len(inputs["image_grid_thw"])):
        n_tokens_hw = inputs["image_grid_thw"][i]
        n_tokens += n_tokens_hw[1] * n_tokens_hw[2]

      if n_tokens: inputs["pixel_values"] = inputs["pixel_values"][:-n_tokens]
      inputs["image_grid_thw"] = inputs["image_grid_thw"][:len(vision_ends)]

    elif 'video_grid_thw' in inputs: # 如果有视频
      # if dist.get_rank() == 0 or True: print_input_info(inputs, f"inputs000000_{dist.get_rank()}")
      # print(f"inputs000000_{dist.get_rank()}", inputs["input_ids"].shape, inputs["input_ids"].flatten().tolist())
      n_tokens = 0
      for i in range(len(vision_ends), len(inputs["video_grid_thw"])):
        n_tokens_hw = inputs["video_grid_thw"][i]
        n_tokens += n_tokens_hw[0] * n_tokens_hw[1] * n_tokens_hw[2]

      if n_tokens: inputs["pixel_values_videos"] = inputs["pixel_values_videos"][:-n_tokens]
      inputs["video_grid_thw"] = inputs["video_grid_thw"][:len(vision_ends)]
      inputs["second_per_grid_ts"] = inputs["second_per_grid_ts"][:len(vision_ends)]

      # if dist.get_rank() == 0 or True: print_input_info(inputs, f"inputs111111_{dist.get_rank()}")
      # print(f"inputs111111_{dist.get_rank()}", inputs["input_ids"].shape, inputs["input_ids"].flatten().tolist())

      if len(inputs["pixel_values_videos"]) == 0:
        del inputs["pixel_values_videos"]
        del inputs["video_grid_thw"]
        del inputs["second_per_grid_ts"]
    # num_thw = 0
    # if "image_grid_thw" in inputs:
    #   thw = inputs["image_grid_thw"]
    #   num_thw = sum([(thw[i][1] * thw[i][2]).item() for i in range(thw.size(0))])
    # num_image_id = (inputs["input_ids"] == self.image_token_id).sum()
    # if num_thw != num_image_id * 4:
    #   print(f"{num_thw=}, {num_image_id=}, {inputs=}")
    # pvs = [0]
    # if "pixel_values" in inputs:
    #   pvs = inputs["pixel_values"].shape
    # if pvs[0] != num_thw:
    #   print(f"{num_thw=}, pixel_values={pvs}")
    return inputs
  
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
                             packed_second_per_grid_ts: List[torch.Tensor],
                             sample_idx: Optional[int] = None,
                             image_pad: bool = False):

    if not image_pad:
      packable_length = self.max_length - cu_seqlens[-1]
      if packable_length == 0: return

    if not image_pad and self.cut_to_pad and inputs['input_ids'].shape[1] > packable_length:
      inputs = self._cut_sample(inputs, packable_length)

    packed_input_ids.append(inputs["input_ids"].flatten())
    packed_loss_mask.append(inputs["loss_mask"].flatten())
    packed_position_ids.append(inputs["position_ids"])
    if sample_idx is None:
      sample_idx = len(cu_seqlens) - 1
    packed_sample_idx.append(
      torch.full_like(packed_input_ids[-1], sample_idx))

    if "pixel_values" in inputs and len(inputs["pixel_values"]):
      packed_pixel_values.append(inputs["pixel_values"])
      packed_image_gird_thw.append(inputs["image_grid_thw"])
    if "pixel_values_videos" in inputs:
      # print('''pixel_values_videos66666666''', inputs["pixel_values_videos"])
      # if torch.isnan(inputs["pixel_values_videos"][0]).any(): print('pixel_values_videos66666666nannnnn', inputs["pixel_values_videos"])
      packed_pixel_values_videos.append(inputs["pixel_values_videos"])
      packed_video_grid_thw.append(inputs["video_grid_thw"])
      packed_second_per_grid_ts.append(inputs["second_per_grid_ts"])
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
    packed_second_per_grid_ts: List[torch.Tensor] = []
    cu_seqlens: List[int] = [0]
    epochs = []
    valid_seq_len = 0
    n_pixels = 0
    for _, inputs in enumerate(buffer):
      if "pixel_values" in inputs: n_pixels += inputs["pixel_values"].shape[0]
      if "pixel_values_videos" in inputs: n_pixels += inputs["pixel_values_videos"].shape[0]
      epochs.append(inputs.get("epoch_idx", None)) # inputs["image_grid_thw"][i]
      image_pad = True if self.use_flops_balance else False
      valid_seq_len += self._append_sample_packing(inputs,
                                      packed_input_ids,
                                      packed_position_ids,
                                      packed_loss_mask,
                                      packed_pixel_values,
                                      packed_pixel_values_videos,
                                      packed_image_gird_thw,
                                      packed_video_grid_thw,
                                      packed_sample_idx,
                                      cu_seqlens,
                                      packed_second_per_grid_ts,
                                      image_pad=image_pad
                                      )

    # 
    # append a pad image sequence to trigger ViT
    image_pad = self._gen_img_pad() if n_pixels % 8 == 0 else self._gen_img_pad(sz=(4, round(self.patch_size * 1.4))) # 1.4 处于 (1.25 ~ 1.5)之间
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
                                packed_second_per_grid_ts,
                                sample_idx=-1,
                                image_pad=True
                                )
    

    packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
    packed_loss_mask = torch.cat(packed_loss_mask, dim=0).unsqueeze(0)
    packed_position_ids = torch.cat(packed_position_ids, dim=-1)
    packed_sample_idx = torch.cat(packed_sample_idx, dim=0).unsqueeze(0)
    packed_second_per_grid_ts = None if len(packed_second_per_grid_ts) == 0 else \
      torch.cat([torch.tensor(x) for x in packed_second_per_grid_ts], dim=0)
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
    ) or True:
      # padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
      max_length = max(self.max_length, packed_input_ids.numel())
      padding_len = (max_length + 7) // 8 * 8 + 64 - packed_input_ids.numel()
      assert padding_len > 0, f"padding_len should be greater than 0, got {padding_len}"
      packed_input_ids = F.pad(
        packed_input_ids, (0, padding_len),
        value=self.processor.tokenizer.pad_token_id)
      packed_sample_idx = F.pad(
        packed_sample_idx, (0, padding_len), value=-1)
      packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
      packed_loss_mask = F.pad(packed_loss_mask, (0, padding_len), value=0)
      cu_seqlens.append(cu_seqlens[-1] + padding_len)

    # print("packed_pixel_values_videospacked_pixel_values_videos", packed_pixel_values_videos)
    # if packed_pixel_values_videos is not None and torch.isnan(packed_pixel_values_videos).any(): print('packed_pixel_values_videospacked_pixel_values_videos_annnnnnn', packed_pixel_values_videos)
    epochs = [x for x in epochs if x is not None]
    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "pixel_values": packed_pixel_values,
      "image_grid_thw": packed_image_gird_thw, # 
      "pixel_values_videos": packed_pixel_values_videos,
      "video_grid_thw": packed_video_grid_thw,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
      "sample_idx": packed_sample_idx.to(torch.int32),
      "epoch_idx": torch.tensor([sum(epochs) / len(epochs)], dtype=torch.float32),
      "second_per_grid_ts": packed_second_per_grid_ts,
    }

    return inputs

  def init(self):
    if self.dataset is None:
      self.dataset, self.total_samples = self._build_source_dataset(self.sources)

  def __iter__(self):
    self.init()
    buffer = []
    source_list = []
    cur_length = 0
    ds_iter = iter(self.dataset)
    while True:
      #for sample in self.dataset:
      try:
        sample = next(ds_iter)
        sample_key = sample["__key__"] if "__key__" in sample else ""
        sample_url = sample["__url__"] if "__url__" in sample else ""

        try:
          source_name = sample["json"]["source"]
          # # WARN: ugly code, for dirty dataset.
          # if source_name.startswith("PDFA"):
          #   source_name = "PDFA"
          # elif source_name.startswith("/llm_reco_ssd/luoxinchen/dataset/"):
          #   source_name = source_name.split("/")[4]
        except:
          source_name = "None"

        self.source_sample_cnt.setdefault(source_name, 0)
        self.source_sample_cnt[source_name] += 1
      
        inputs = self._process(sample, source_name)
      except StopIteration as e:
        logger.info(f"StopIteration: {e}")
        break
      except:
        self.source_error_cnt.setdefault(source_name, 0)
        self.source_error_cnt[source_name] += 1
        error_ratio = self.source_error_cnt[source_name] * 1.0 / \
          self.source_sample_cnt[source_name]
        logger.error(
          f"ChatCompletionVisionDataset process sample error. "
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, sample=\n{str(sample)[:50]}"
          f"errmsg={traceback.format_exc()}")
        continue

      sample_length = inputs["input_ids"].shape[-1]
      if cur_length + sample_length >= self.max_length:
        # packed_inputs = self._packing(buffer)
        # packed_inputs["data_source"] = source_list
        # buffer = [inputs]
        # source_list = [source_name]
        # cur_length = sample_length

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




class ChatCompletionVisionDataset_moonvit(ChatCompletionVisionDataset):
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
               cut_to_pad=True,
               use_flops_balance=False,
               **kwargs
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
      processor = Qwen2_5_VLProcessor_moonvit.from_pretrained(base_model_dir)
      model_config = Qwen2_5_VLConfig.from_pretrained(base_model_dir)
      vision_config = MoonViTConfig()
      spatial_merge_size = vision_config.merge_kernel_size[0]
      patch_size = vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    self.use_flops_balance = use_flops_balance
    self.auto_aug = AutoAugmentWrapper(policy=kwargs.get("autoaug_policy", None))
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")
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
    
    if self.use_flops_balance: self.dataset, self.total_samples = None, None
    else:  self.dataset, self.total_samples = self._build_source_dataset(sources)
    self.sources = sources

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    self.kargs = self.kwargs = kwargs


class ChatCompletionVisionDataset_siglip(ChatCompletionVisionDataset):
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
               cut_to_pad=True,
               process_vision_info_args={"image_factor": 32},
               min_visual_tokens_per_frame: int = 4,
               max_visual_tokens_per_frame: int = 512,
               use_flops_balance=False,
               **kwargs
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
      processor = Qwen2_5_VLProcessor_siglip.from_pretrained(base_model_dir)
      model_config = Qwen2_5_VLConfig.from_pretrained(base_model_dir)
      vision_config = SiglipConfig.from_pretrained('/llm_reco/liuyang76/Models/siglip2-so400m-patch14-384').vision_config
      spatial_merge_size = 2#vision_config.merge_kernel_size[0]
      patch_size = vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    self.use_flops_balance = use_flops_balance
    self.process_vision_info = process_vision_info
    self.auto_aug = AutoAugmentWrapper(policy=kwargs.get("autoaug_policy", None))
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")
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
    
    if self.use_flops_balance: self.dataset, self.total_samples = None, None
    else:  self.dataset, self.total_samples = self._build_source_dataset(sources)
    self.sources = sources

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    self.kargs = self.kwargs = kwargs

class ChatCompletionVisionDataset_keye(ChatCompletionVisionDataset):
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
               patch_size: int = 16,
               image_token_id: int = 151655,
               video_token_id: int = 151656,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {},
               cut_to_pad=True,
               process_vision_info_args={"image_factor":32},
               min_visual_tokens_per_frame: int = 4,
               max_visual_tokens_per_frame: int = 512,
               **kwargs
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
      try:
        processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
      except:
        processor = KeyeProcessor.from_pretrained(base_model_dir)
      model_config = KeyeConfig.from_pretrained(base_model_dir)
      spatial_merge_size = model_config.vision_config.spatial_merge_size
      patch_size = model_config.vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    self.use_flops_balance = kwargs.get("use_flops_balance", False)
    self.auto_aug = AutoAugmentWrapper(policy=kwargs.get("autoaug_policy", None))
    self.process_vision_info_args = process_vision_info_args
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")
    self.processor = processor
    self.process_vision_info = process_vision_info_keye

    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.min_visual_tokens_per_frame = min_visual_tokens_per_frame
    self.max_visual_tokens_per_frame = max_visual_tokens_per_frame

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
    if self.use_flops_balance: self.dataset, self.total_samples = None, None
    else:  self.dataset, self.total_samples = self._build_source_dataset(sources)
    self.sources = sources

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = self._gen_img_pad()["input_ids"].shape[-1] # 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    self.kargs = self.kwargs = kwargs


class ChatCompletionVisionDataset_keye_vitrope(ChatCompletionVisionDataset):
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
               patch_size: int = 16,
               image_token_id: int = 151655,
               video_token_id: int = 151656,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {},
               cut_to_pad=True,
               process_vision_info_args={"image_factor":28},
               min_visual_tokens_per_frame: int = 4,
               max_visual_tokens_per_frame: int = 512,
               use_flops_balance=False,
               **kwargs
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
      processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
      model_config = KeyeConfig.from_pretrained(base_model_dir)
      spatial_merge_size = model_config.vision_config.spatial_merge_size
      patch_size = model_config.vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    self.auto_aug = AutoAugmentWrapper(policy=kwargs.get("autoaug_policy", None))
    self.process_vision_info = process_vision_info_keye_vitrope
    self.process_vision_info_args = process_vision_info_args
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")
    self.processor = processor
    self.use_flops_balance = use_flops_balance

    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.min_visual_tokens_per_frame = min_visual_tokens_per_frame
    self.max_visual_tokens_per_frame = max_visual_tokens_per_frame

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
    self.use_flops_balance = use_flops_balance
    if self.use_flops_balance: self.dataset, self.total_samples = None, None
    else:  self.dataset, self.total_samples = self._build_source_dataset(sources)
    self.sources = sources
    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = self._gen_img_pad()["input_ids"].shape[-1] # 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0
    print(f"patch_size: {patch_size}")
    print(f"processor: {self.processor}")
    print(f"self.process_vision_info: {self.process_vision_info}")

    self.datasource_config = datasource_config
    self.kargs = self.kwargs = kwargs

class ChatCompletionVisionDataset_navit(ChatCompletionVisionDataset):
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
               patch_size: int = 16,
               image_token_id: int = 151655,
               video_token_id: int = 151656,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {},
               cut_to_pad=True,
               process_vision_info_args={"image_factor": 32},
               min_visual_tokens_per_frame: int = 4,
               max_visual_tokens_per_frame: int = 512,
               **kwargs
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
      from recovlm.models.qwen3siglip.configuration_qwen3siglip import Qwen3SiglipConfig
      processor = Qwen3SiglipProcessor_navit.from_pretrained(base_model_dir)
      model_config = Qwen3SiglipConfig.from_pretrained(base_model_dir)
      vision_config = SiglipConfig.from_pretrained('/llm_reco_ssd/zhouyang12/models/siglip2-so400m-patch16-naflex').vision_config
      spatial_merge_size = 2 # vision_config.merge_kernel_size[0]
      patch_size = vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id


    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.min_visual_tokens_per_frame = min_visual_tokens_per_frame
    self.max_visual_tokens_per_frame = max_visual_tokens_per_frame

    self.process_vision_info = process_vision_info_keye
    self.auto_aug = AutoAugmentWrapper(policy=kwargs.get("autoaug_policy", None))
    self.process_vision_info_args = process_vision_info_args
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")
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
    self.dataset, self.total_samples = None, None
    self.sources = sources

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = self._gen_img_pad()["input_ids"].shape[-1] # 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    self.kargs = self.kwargs = kwargs

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
    
    self.dataset, self.total_samples = None, None
    self.sources = sources

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
    text += self.kargs.get("endoftext", "<|endoftext|>")
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
    print(34335344444, self.kargs.get("endoftext", "<|endoftext|>"))
    # append EOS token
    text += self.kargs.get("endoftext", "<|endoftext|>")
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

    print("ggggggg", self.kargs.get("start_pattern", [151644, 77091, 198]), self.kargs.get("end_pattern", [151645, 198]))
    inputs["loss_mask"] = get_assistant_mask(
      inputs["input_ids"],
      start_pattern=self.kargs.get("start_pattern", [151644, 77091, 198]), # [151644, 77091, 198],
      end_pattern=self.kargs.get("end_pattern", [151645, 198]), # self.end_pattern, #[151645, 198]
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
        return_tensors="pt",
        image_video_pad=True,
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
        inputs['epoch_idx'] = sample['epoch_idx']
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
                                      cu_seqlens,
                                      )

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

  def init(self):
    if self.dataset is None:
      self.dataset, self.total_samples = self._build_source_dataset(self.sources)

  def __iter__(self):
    self.init()
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
        # # WARN: ugly code, for dirty dataset.
        # if source_name.startswith("PDFA"):
        #   source_name = "PDFA"
        # elif source_name.startswith("/llm_reco_ssd/luoxinchen/dataset/"):
        #   source_name = source_name.split("/")[4]
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
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=},  sample=\n{str(sample)[:50]}"
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




class NaiveParquetDataset(IterableDataset):
  def __init__(self, data_files, num_workers, n_local_shuffle_files_window=5, use_flops_balance=False, **kargs):
    self.data_files = data_files
    self.num_workers = num_workers
    self.n_local_shuffle_files_window = n_local_shuffle_files_window
    self.use_flops_balance = use_flops_balance
    print(f"NaiveParquetDataset: use_flops_balance={use_flops_balance}")

    # print(f"ParquetDataset set n_local_shuffle_files_window={n_local_shuffle_files_window}, vit_token_balance={vit_token_balance}")

    if use_flops_balance:
      def make_dict(): return {}
    else:
      manager = multiprocessing.Manager()
      def make_dict(): return manager.dict()

    self.finish_dict_all = make_dict()
    self.offset_dict_all = make_dict()
    for i in range(self.num_workers):
      self.finish_dict_all[i] = make_dict()
      self.offset_dict_all[i] = make_dict()


  def state_dict(self,):
    rank, world_size, worker, num_workers = pytorch_worker_info()

    state_dict = {
      "finish_dict": dict(self.finish_dict_all[worker]),
      "offset_dict": dict(self.offset_dict_all[worker])
    }
    print("NaiveParquetDataset__state_dict")
    return state_dict

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
      videos = raw_row_data["videos"]
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
      self._load_videos_to_samples(videos, samples, raw_row_data)
      return samples
    except:
      logger.error(f"ParquetDataset parse sample error!!! err_msg={traceback.format_exc()}, images={str(images)[:50]}\nsamples={str(samples)[:50]}")
      return None

  def _load_videos_to_samples(self, videos, samples, raw_row_data):
    # process images
    if isinstance(videos, str):
      videos = json.loads(videos)
      if not videos: return
    
    if isinstance(videos, dict):
      pass
    else:
      raise NotImplementedError(f"Unsupported video field type, {type(raw_row_data['videos'])=}, {raw_row_data['videos']}\n{type(videos)}, videos={videos}, samples={samples}")
    
    for video_name in videos:
      video_path = videos[video_name]
      # 先检查是否是有效文件路径
      if isinstance(video_path, str) and os.path.exists(video_path):
          try:
              messages = samples['json']['messages']
              for message in messages:
                contents = message['content']
                for content in contents:
                  if content['type'] == 'video' and content['video'] == video_name:
                    content['video'] = video_path
          except Exception as e:
              raise ValueError(f"Failed to substitute video path for {video_path}: {str(e)}")
      # 否则按base64处理
      else:
          raise NotImplementedError(f"base64 video is not supported")


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
              raise ValueError(f"Failed to load image from path {image_b64}: {str(e)}, image_b64={image_b64[:100]}")
      # 否则按base64处理
      else:
          try:
              image_bytes = base64.b64decode(image_b64)
              image_bytes_stream = BytesIO(image_bytes)
              image = Image.open(image_bytes_stream)
              samples[image_name] = image
          except Exception as e:
              raise ValueError(f"Failed to decode base64 image {image_name}: {str(e)}, image_b64={image_b64[:100]}")

  def read_fn(self, epoch_fn):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    finish_dict = self.finish_dict_all[worker]
    offset_dict = self.offset_dict_all[worker]
    fn, epoch_idx = epoch_fn
    if (fn, epoch_idx) in finish_dict:
      logger.warning(f"[Rank{rank}-{worker}] skip {fn}")
      return
    
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
              sample['epoch_idx'] = torch.tensor(epoch_idx)
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

  def __iter__local_shuffle(self):
    import pandas as pd
    rank, world_size, worker, num_workers = pytorch_worker_info()
    rank, world_size, worker, num_workers = pytorch_worker_info()
    finish_dict = self.finish_dict_all[worker]
    offset_dict = self.offset_dict_all[worker]
    assert num_workers == self.num_workers
    from multiprocessing.pool import ThreadPool as Pool

    total_num_workers = num_workers * world_size
    local_worker_idx = rank * num_workers + worker
    fn_list = [fn for idx, fn in enumerate(self.data_files) if idx % total_num_workers == local_worker_idx]
    logger.warning(
      f"ParquetDataset Info: {rank=}, {world_size=}, {worker=}, {num_workers=}, {len(fn_list)=}"
    )   
    import tqdm

    # np.random.shuffle(fn_list)
    def shuffle_parquet_rows(parquet_files_list, n_buffer_files):
        # file_index = 0
        row_counts = []
        all_rows = []
        # 一开始读取 n_buffer_files 个文件
        # while file_index < n_buffer_files and file_index < len(parquet_files_list):
        assert min(n_buffer_files,len(parquet_files_list)) != 0, f"n_buffer_files={n_buffer_files}, len(parquet_files_list)={len(parquet_files_list)}"
        # for file_index in  tqdm.tqdm(range(min(n_buffer_files, len(parquet_files_list)))):
        file_index = 0
        while len(all_rows) < min(n_buffer_files, len(parquet_files_list)):
            fn, epoch_idx = parquet_files_list[file_index]
            if (fn, epoch_idx) in finish_dict:
              logger.warning(f"[Rank{rank}-{worker}] skip {fn}-epoch{epoch_idx}")
              file_index += 1
              continue

            finish_dict[(fn, epoch_idx)] = True
            logger.warning(f"[Rank{rank}-{worker}] {fn}-epoch{epoch_idx} start.")
            file_index += 1
            try:
                df = load_parquet_file(fn).read_row_group(0).to_pandas()
            except Exception as e:
                logger.error(str(e))
                logger.error(f"load parquet file {fn} failed")
                continue
            df['epoch_idx'] = epoch_idx
            row_counts.append(len(df))
            all_rows.append(df)

        all_rows = pd.concat(all_rows, ignore_index=True)
        all_rows = all_rows.sample(frac=1).reset_index(drop=True)

        rows_processed = 0

        while True:
            for i, (_, row) in enumerate(all_rows.iterrows()):
                try:
                  sample = self._parser(row, "tmp")
                  sample['epoch_idx'] = torch.tensor(row['epoch_idx'])
                  if sample is not None:
                    yield sample

                except GeneratorExit:
                  # 正确处理生成器退出
                  logger.warning(f"Generator exited")
                  return

                except Exception as e:
                  logger.error(f"Error processing row : {str(e)}")
                  continue

                rows_processed += 1

                # 当处理的行数达到当前文件的行数且还有文件未处理
                if rows_processed == row_counts[0] and file_index < len(parquet_files_list):
                  break
            
            try:
              while True:
                fn, epoch_idx = parquet_files_list[file_index]
                if (fn, epoch_idx) in finish_dict:
                  logger.warning(f"[Rank{rank}-{worker}] skip {fn}-epoch{epoch_idx}")
                  file_index += 1
                  continue

                finish_dict[(fn, epoch_idx)] = True
                file_index += 1
                try:
                  new_df = load_parquet_file(fn).read_row_group(0).to_pandas()
                  break
                except Exception as e:
                  logger.error(f"Error processing fn {fn}\n{str(e)}")
                
              logger.warning(f"[Rank{rank}-{worker}] {fn}-epoch{epoch_idx} start.")
              all_rows = pd.concat([all_rows[i + 1:], new_df], ignore_index=True)
              all_rows = all_rows.sample(frac=1).reset_index(drop=True)
              row_counts.pop(0)
              row_counts.append(len(new_df))
              rows_processed = 0
            except Exception as e:
              
              print(e)
              print("error in ParquetDataset!!!")
              print(traceback.format_exc())
            
            # 如果已经处理完所有文件且当前数据都已处理完，则退出循环
            if file_index >= len(parquet_files_list) and rows_processed == row_counts[0]:
                break
    
    for sample in shuffle_parquet_rows(fn_list, self.n_local_shuffle_files_window):
      yield sample

  def __iter__(self,):
      for sample in self.__iter__local_shuffle():
        if sample is None: continue
        yield sample

  def state_dict(self,):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    state_dict = {
      "finish_dict": dict(self.finish_dict_all[worker]),
      "offset_dict": dict(self.offset_dict_all[worker])
    }
    return state_dict
  
  def load_state_dict(self, state_dict):
    """
    {
      "_snapshot": {
        "_snapshot_step": 13,
        "_last_yielded_worker_id": 0,
        "_main_snapshot": {
          "_num_workers": 1,
          "_sampler_iter_state": None,
          "_index_sampler_state": {
            "samples_yielded": 13
          },
          "_sampler_iter_yielded": 13,
          "_IterableDataset_len_called": None,
          "_shared_seed": None,
          "_base_seed": 9110132061529926470
        },
        "_worker_snapshots": {
          "worker_0": {
            "worker_id": 0,
            "dataset_state": {
              "offset_dict": {

              },
              "finish_dict": {
                "('viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/SyntheticOCR_EN_HW/0.2.0/rank210-10.parquet', 0)": True,
                "('viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/InfinityStage2_7M_0712/0.1.0/rank466-10.parquet', 0)": True,
                "('viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/Laion_zh/0.1.0/rank1007_19.parquet', 0)": True,
                "('viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/KwaiiTextQA_zh/0.0.0/rank481-30.parquet', 0)": True,
                "('viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/Vript_long/0.1.0/rank109-0.parquet', 0)": True
              }
            },
            "fetcher_state": {
              "fetcher_ended": False,
              "dataset_iter_state": None
            }
          }
        }
      },
      "_steps_since_snapshot": 0,
      "_iterator_finished": False
    }
    """
    rank, world_size, worker, num_workers = pytorch_worker_info()

    # state_dict['_snapshot']['_worker_snapshots']['worker_0'].keys()
    if self.use_flops_balance:
      state_dict = state_dict['_snapshot']['_worker_snapshots']['worker_0']['dataset_state']

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
    logger.warning(f"[rank{rank}-woker{worker}] load checkpoint success. self.finish_dict_all={self.finish_dict_all}")


class ParquetDataset(IterableDataset):
  def __init__(self, data_files, num_workers, num_readers=1, shuffle_window=0, **kargs):
    self.data_files = data_files
    self.num_workers = num_workers
    self.num_readers = num_readers
    self.shuffle_window = shuffle_window
    self.kargs = kargs
    self.finish_dict_all = {}
    self.offset_dict_all = {}
    for i in range(self.num_workers * num_readers):
      self.finish_dict_all[i] = {}
      self.offset_dict_all[i] = {}

  def state_dict(self):
    rank, world_size, worker, num_workers = pytorch_worker_info()

    state_dict = {
      "finish_dict": self.finish_dict_all,
      "offset_dict": self.offset_dict_all,
    }
    return state_dict
  
  def load_state_dict(self, state_dict):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    finish_dict = state_dict["finish_dict"]
    offset_dict = state_dict["offset_dict"]

    tmp_finish = {}
    for i, finish in finish_dict.items():
      # support old ckpt format
      tmp_finish_dict = dict()

      for k, v in finish.items():
        if isinstance(k, str):
          tmp_finish_dict[(k, 0)] = v
        elif isinstance(k, tuple) and len(k) == 2:
          tmp_finish_dict[k] = v
        else:
          raise NotImplementedError(f"Unsupported dataloader checkpoint format. {tmp_finish_dict}") 
      tmp_finish[i] = tmp_finish_dict

    tmp_offset = {}
    for i, offset in offset_dict.items():
      tmp_offset_dict = dict()
      for k, v in offset_dict.items():
        if isinstance(k, str):
          fn, group_idx = k.split("|")
          group_idx = int(group_idx)
          tmp_offset_dict[(fn, 0, group_idx)] = v
        elif isinstance(k, tuple) and len(k) == 3:
          tmp_offset_dict[k] = v
        else:
          raise NotImplementedError(f"Unsupported dataloader checkpoint format. {tmp_offset_dict}") 
      tmp_offset[i] = tmp_offset_dict

    # clear cur state
    self.finish_dict_all.clear()
    self.offset_dict_all.clear()

    # update
    self.finish_dict_all.update(tmp_finish)
    self.offset_dict_all.update(tmp_offset)
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
      videos = raw_row_data["videos"]

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
      self._load_videos_to_samples(videos, samples, raw_row_data)

      return samples
    except:
      logger.error(f"ParquetDataset parse sample error!!! err_msg={traceback.format_exc()}, images={str(images)[:50]}\nsamples={str(samples)[:50]}")
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
              raise ValueError(f"Failed to load image from path {image_b64}: {str(e)}, image_b64={image_b64[:100]}")
      # 否则按base64处理
      else:
          try:
              image_bytes = base64.b64decode(image_b64)
              image_bytes_stream = BytesIO(image_bytes)
              image = Image.open(image_bytes_stream)
              samples[image_name] = image
          except Exception as e:
              raise ValueError(f"Failed to decode base64 image {image_name}: {str(e)}, image_b64={image_b64[:100]}")

  def _load_videos_to_samples(self, videos, samples, raw_row_data):
    # process images
    if isinstance(videos, str):
      videos = json.loads(videos)
      if not videos: return
    
    if isinstance(videos, dict):
      pass
    else:
      raise NotImplementedError(f"Unsupported video field type, {type(raw_row_data['videos'])=}, {raw_row_data['videos']}\n{type(videos)}, videos={videos}, samples={samples}")
    
    for video_name in videos:
      video_path = videos[video_name]
      # 先检查是否是有效文件路径
      if isinstance(video_path, str) and os.path.exists(video_path):
          try:
              messages = samples['json']['messages']
              for message in messages:
                contents = message['content']
                for content in contents:
                  if content['type'] == 'video' and content['video'] == video_name:
                    content['video'] = video_path
          except Exception as e:
              raise ValueError(f"Failed to substitute video path for {video_path}: {str(e)}")
      # 否则按base64处理
      else:
          raise NotImplementedError(f"base64 video is not supported")

  def read_parquet_runner(self, fn_list, tid):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    worker = worker + tid
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
                  sample['epoch_idx'] = torch.tensor(epoch_idx)
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
      if len(buffer) == window: # 每3k shuffle。
        random.shuffle(buffer)
        for sample in buffer:
          self.shuffled_queue.put(sample)
        buffer = []

  def __iter__(self):
    rank, world_size, worker, num_workers = pytorch_worker_info()
    assert num_workers == self.num_workers, f"{num_workers} : {self.num_workers}"

    total_num_workers = num_workers * world_size
    local_worker_idx = rank * num_workers + worker
    fn_list = [fn for idx, fn in enumerate(self.data_files) if idx % total_num_workers == local_worker_idx]
    logger.warning(
      f"ParquetDataset Info: {rank=}, {world_size=}, {worker=}, {num_workers=}, {len(fn_list)=}"
    )
    
    self.sample_queue = queue.Queue(maxsize=16)
    self.readers = []
    for i in range(self.num_readers):
      reader = threading.Thread(target=self.read_parquet_runner, args=(fn_list, i), daemon=True)
      reader.start()
      self.readers.append(reader)
    input_q = self.sample_queue
      
    if self.shuffle_window > 0:
      self.shuffled_queue = queue.Queue(self.shuffle_window * 2)
      self.shuffle_task = threading.Thread(target=self.shuffle_runner, args=(self.shuffle_window, ), daemon=True)
      self.shuffle_task.start()
      input_q = self.shuffled_queue
    
    while True:
      sample = input_q.get()
      yield sample


ParquetDataset = NaiveParquetDataset


class ChatCompletionVisionParquetDataset(ChatCompletionVisionDataset):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.cut_to_pad = kargs.get("cut_to_pad", True)
    self.kargs = kargs
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

    dataset = ParquetDataset(data_file_list, self.num_workers, **self.kargs)
    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)
    

class ChatCompletionVisionParquetDataset_keye(ChatCompletionVisionDataset_keye):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.cut_to_pad = kargs.get("cut_to_pad", True)
    self.kargs = kargs
    self.num_readers = kargs.get("num_readers", 1)
    self.shuffle_window = kargs.get("shuffle_window", 0)
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
      logger.error(f"ChatCompletionVisionParquetDataset_keye rank{dist.get_rank()}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

    t = [data_file_list]
    dist.broadcast_object_list(t, src=0)
    data_file_list = t[0]

    logger.error(f"ChatCompletionVisionParquetDataset_keye rank{dist.get_rank()}: file_num={len(data_file_list)}")
    if len(data_file_list) == 0:
      raise ValueError(f"no datafile found!")

    dataset = NaiveParquetDataset(data_file_list, self.num_workers, **self.kargs)
    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)




class ChatCompletionVisionParquetDataset_keye_vitrope(ChatCompletionVisionDataset_keye_vitrope):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.cut_to_pad = kargs.get("cut_to_pad", True)
    self.kargs = kargs
    self.num_readers = kargs.get("num_readers", 1)
    self.shuffle_window = kargs.get("shuffle_window", 0)
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
      logger.error(f"ChatCompletionVisionParquetDataset_keye_vitrope rank{dist.get_rank()}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

    t = [data_file_list]
    dist.broadcast_object_list(t, src=0)
    data_file_list = t[0]

    logger.error(f"ChatCompletionVisionParquetDataset_keye_vitrope rank{dist.get_rank()}: file_num={len(data_file_list)}")
    if len(data_file_list) == 0:
      raise ValueError(f"no datafile found!")

    dataset = NaiveParquetDataset(data_file_list, self.num_workers, **self.kargs)
    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)

  def state_dict(self):
    if self.dataset is None:
      return {}
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    if self.dataset is None:
      return
    self.dataset.load_state_dict(state_dict)



class ChatCompletionVisionParquetDataset_moonvit(ChatCompletionVisionDataset_moonvit):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.cut_to_pad = kargs.get("cut_to_pad", True)
    self.kargs = kargs
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

    dataset = ParquetDataset(data_file_list, self.num_workers, **self.kargs)
    return dataset, -1

  def state_dict(self, ):
    
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)


class ChatCompletionVisionParquetDataset_siglip(ChatCompletionVisionDataset_siglip):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.cut_to_pad = kargs.get("cut_to_pad", True)
    self.kargs = kargs
    self.num_readers = kargs.get("num_readers", 1)
    self.shuffle_window = kargs.get("shuffle_window", 0)
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

    dataset = ParquetDataset(data_file_list, self.num_workers, **self.kargs)
    return dataset, -1

  def state_dict(self, ):
    
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)


class ChatCompletionVisionParquetDataset_navit(ChatCompletionVisionDataset_navit):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.cut_to_pad = kargs.get("cut_to_pad", True)
    self.kargs = kargs
    self.num_readers = kargs.get("num_readers", 1)
    self.shuffle_window = kargs.get("shuffle_window", 0)
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

    dataset = ParquetDataset(data_file_list, self.num_workers, shuffle_window=self.shuffle_window, use_flops_balance=self.use_flops_balance, **kargs)
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
               vit_token_balance=False,
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
      tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
      model_config = InternVLChatConfig.from_pretrained(base_model_dir)
      patch_size = model_config.vision_config.patch_size
      image_size = model_config.force_image_size
    
    self.tokenizer = tokenizer
    self.visual_tokens_per_image = int((image_size//patch_size)** 2 * (down_sample_ratio ** 2))
    self.cut_to_pad = cut_to_pad
    # if int(vit_token_balance) + int(cut_to_pad) <= 1:
    #   print(f"The parameters vit_token_balance({vit_token_balance}) and cut_to_pad({cut_to_pad}) cannot be set to True simultaneously (it damages throughput!!!). We set cut_to_pad to false")
    #   cut_to_pad = False

    self.down_sample_ratio=down_sample_ratio
    self.pid_info_client = PidInfoClient('10.84.241.154')
    print(f"set cut_to_pad={cut_to_pad}, vit_token_balance={vit_token_balance}")

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

    self.dataset, self.total_samples = None, None
    self.sources = sources

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
    text += self.kargs.get("endoftext", "<|endoftext|>")
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
      print("unexpected shape")
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
      inputs['epoch_idx'] = sample['epoch_idx']

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
    epochs = []
    for _, inputs in enumerate(buffer):
      epochs.append(inputs.get("epoch_idx", None))
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

    if self.cut_to_pad and not valid_seq_len == self.max_length and np.random.rand() < 0.01: 
      print(f"intern_data warning: set cut_to_pad={self.cut_to_pad}, then require valid_seq_len/{valid_seq_len} == self.max_length/{self.max_length}")
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
    ) or self.cut_to_pad:
      # padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
      padding_len = self.max_length - packed_input_ids.numel()


      if self.cut_to_pad and padding_len:
        print(f"padding_len={padding_len}, not equal to 0")
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

    epochs = [x for x in epochs if x is not None]
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
      "sample_idx": packed_sample_idx.to(torch.int32),
      "epoch_idx": torch.tensor([sum(epochs) / len(epochs)], dtype=torch.float32),
    }
    if packed_input_ids.flatten().shape[0] < packed_pixel_values.shape[0] * 256:
      print_input_info(inputs, "inputs111: ")
      raise Exception("!!!!!! Error occurs in image padding. input ids are shorted than image tokens")
    return inputs

  def init(self):
    if self.dataset is None:
      self.dataset, self.total_samples = self._build_source_dataset(self.sources)
  
  def __iter__(self):
    self.init()
    buffer = []
    source_list = []
    cur_length = 0

    for sample in self.dataset:
      sample_key = sample["__key__"] if "__key__" in sample else ""
      sample_url = sample["__url__"] if "__url__" in sample else ""

      try:
        source_name = sample["json"]["source"]
        # # WARN: ugly code, for dirty dataset.
        # if source_name.startswith("PDFA"):
        #   source_name = "PDFA"
        # elif source_name.startswith("/llm_reco_ssd/luoxinchen/dataset/"):
        #   source_name = source_name.split("/")[4]
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
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, sample=\n{str(sample)[:50]}"
          f"errmsg={traceback.format_exc()}")
        continue

      sample_length = inputs["input_ids"].shape[-1]
      if cur_length + sample_length >= self.max_length - self.image_pad_len:

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
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, n_local_shuffle_files_window=5, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.num_readers = kargs.get("num_readers", 1)
    self.shuffle_window = kargs.get("shuffle_window", 0)
    self.n_local_shuffle_files_window = n_local_shuffle_files_window
    super().__init__(sources, **kargs)
    
  def _get_file_list(self, sources):
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
    return data_file_list

  def _build_source_dataset(self, sources):
    data_file_list = self._get_file_list(sources)
    dataset = ParquetDataset(data_file_list, self.num_workers,
                             num_readers=self.num_readers,
                             shuffle_window=self.shuffle_window,)

    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)





class ChatCompletionVisionDataset_keye_vitrope_slowfast(ChatCompletionVisionDataset):
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
               patch_size: int = 16,
               image_token_id: int = 151655,
               video_token_id: int = 151656,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {},
               cut_to_pad=True,
               process_vision_info_args={"image_factor":32},
               min_visual_tokens_per_frame: int = 4,
               max_visual_tokens_per_frame: int = 512,
               **kwargs
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
      try:
        processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
      except:
        processor = KeyeProcessor.from_pretrained(base_model_dir)
      model_config = KeyeConfig.from_pretrained(base_model_dir)
      spatial_merge_size = model_config.vision_config.spatial_merge_size
      patch_size = model_config.vision_config.patch_size
      image_token_id = model_config.image_token_id
      video_token_id = model_config.video_token_id
      fast_video_token_id = model_config.fast_video_token_id
      vision_start_token_id = model_config.vision_start_token_id
      vision_end_token_id = model_config.vision_end_token_id
      pad_token_id = model_config.pad_token_id

    kwargs['use_flops_balance'] = kwargs.get("use_flops_balance", False)
    self.use_flops_balance = kwargs['use_flops_balance']
    print(111111111, self.use_flops_balance)
    self.auto_aug = AutoAugmentWrapper(policy=kwargs.get("autoaug_policy", None))
    self.process_vision_info_args = process_vision_info_args
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")
    self.processor = processor
    self.process_vision_info = process_vision_info_keye_vitrope_slowfast

    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.min_visual_tokens_per_frame = min_visual_tokens_per_frame
    self.max_visual_tokens_per_frame = max_visual_tokens_per_frame

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
    self.fast_video_token_id = fast_video_token_id
    self.vision_start_token_id = vision_start_token_id
    self.vision_end_token_id = vision_end_token_id
    self.pad_token_id = pad_token_id
    self.patch_size = patch_size
    # Pad sequence to multiple of `multiple_of`
    self.multiple_of = multiple_of
    self.shuffle_size = shuffle_size
    self.shuffle_initial_size = shuffle_initial_size
    if self.use_flops_balance: self.dataset, self.total_samples = None, None
    else:  self.dataset, self.total_samples = self._build_source_dataset(sources)
    self.sources = sources

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    image_pad_len = self._gen_img_pad()["input_ids"].shape[-1] # 6
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    self.kargs = self.kwargs = kwargs
    
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
      "min_visual_tokens_per_frame": self.min_visual_tokens_per_frame,
      "max_visual_tokens_per_frame": self.max_visual_tokens_per_frame, 
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
      inputs['epoch_idx'] = sample['epoch_idx']
      if not inputs:
        raise ValueError("Empty inputs, skip")

      process_max_length = min(int(self.max_length // 1.5), 8000) if self.use_flops_balance else self.max_length
      if inputs["input_ids"].shape[-1] > process_max_length:
        source_conf["max_visual_tokens_per_image"] = (
            source_conf["max_visual_tokens_per_image"] * self.shrink_ratio)
        source_conf["max_visual_tokens_per_frame"] = (
            source_conf["max_visual_tokens_per_frame"] * self.shrink_ratio)
        continue
      else:
        assert inputs["input_ids"].shape[-1] <= process_max_length, "inputs too long"
        lenf = inputs["input_ids"].shape[-1]

        # print(f"rank{dist.get_rank()}_process{lenf}=============== ")
        # print_input_info(
        #   inputs,
        #   f"rank{dist.get_rank()}_process{lenf}: "
        # )
        return inputs
    else:
      raise ValueError(
          f"Unable to generate sample within max_length={process_max_length} after {retry} retrys"
      )
  def _cut_sample(self, inputs, packable_length):
    return self._cut_sample_cjx(inputs, packable_length)

  def _cut_sample_cjx(self, inputs, packable_length):
    inputs1 = copy.deepcopy(inputs)

    # if 'pixel_values_videos' in inputs and dist.get_rank() == 0:
    inputs["input_ids"] = inputs["input_ids"][:, :packable_length]
    inputs["loss_mask"] = inputs["loss_mask"][:, :packable_length]

    inputs["position_ids"] = inputs["position_ids"][..., :packable_length]

    vision_starts = torch.nonzero(inputs["input_ids"][0] == self.vision_start_token_id)
    vision_ends = torch.nonzero(inputs["input_ids"][0] == self.vision_end_token_id)


    if len(vision_starts) and len(vision_starts) > len(vision_ends): # 说明图片不完整
      # inputs["input_ids"][:, vision_starts[-1]:] = 0 # 随便什么id都可以
      # inputs["loss_mask"][:, vision_starts[-1]:] = 0
      # inputs["position_ids"][:, vision_starts[-1]:] = 0
      inputs["input_ids"] = inputs["input_ids"][:, :vision_starts[-1]] # 继续截断,截断到vision_starts token,因为vision_start之后的内容都不会有loss
      inputs["loss_mask"] = inputs["loss_mask"][:, :vision_starts[-1]]
      inputs["position_ids"] = inputs["position_ids"][..., :vision_starts[-1]]
    
    vision_start_indices = torch.nonzero(inputs["input_ids"][0] == self.vision_start_token_id)
    if vision_start_indices.numel() == 0:  # 检查是否为空
      image_nums = 0
      video_token_nums = 0
      fast_video_token_nums = 0
    else:
      vision_tokens = inputs["input_ids"][0][vision_start_indices + 1]
      image_nums = (vision_tokens == self.image_token_id).sum()
      video_token_nums = (inputs["input_ids"][0] == self.video_token_id).sum()
      fast_video_token_nums = (inputs["input_ids"][0] == self.fast_video_token_id).sum()

    if 'image_grid_thw' in inputs and len(inputs["pixel_values"]) and 'video_grid_thw' in inputs and len(inputs["pixel_values_videos"]):
      raise Exception("Unexpected inputs: there are both pixel_values and pixel_values_videos: {}/{}".format(inputs["pixel_values"].shape, inputs["pixel_values_videos"].shape))

    if 'image_grid_thw' in inputs: # 如果有图片
      n_tokens = 0
      for i in range(len(vision_ends), len(inputs["image_grid_thw"])):
        n_tokens_hw = inputs["image_grid_thw"][i]
        n_tokens += n_tokens_hw[1] * n_tokens_hw[2]

      if n_tokens: inputs["pixel_values"] = inputs["pixel_values"][:-n_tokens]
      inputs["image_grid_thw"] = inputs["image_grid_thw"][:len(vision_ends)]

    if 'video_grid_thw' in inputs: # 如果有视频
      # if dist.get_rank() == 0 or True: print_input_info(inputs, f"inputs000000_{dist.get_rank()}")
      # print(f"inputs000000_{dist.get_rank()}", inputs["input_ids"].shape, inputs["input_ids"].flatten().tolist())
      used_n_token = 0
      video_token_nums = video_token_nums * 4 # 注意这里的乘4是因为我们有2*2的merge patch操作
      video_used_idx = len(inputs["video_grid_thw"])
      for idx, n_tokens_hw in enumerate(inputs["video_grid_thw"]):
        if used_n_token == video_token_nums:
          video_used_idx = idx
          break
        used_n_token += (n_tokens_hw[0] * n_tokens_hw[1] * n_tokens_hw[2])

      inputs["pixel_values_videos"] = inputs["pixel_values_videos"][:video_token_nums]
      inputs["video_grid_thw"] = inputs["video_grid_thw"][:video_used_idx]
    
    if 'fast_video_grid_thw' in inputs: # 如果有视频
      # if dist.get_rank() == 0 or True: print_input_info(inputs, f"inputs000000_{dist.get_rank()}")
      # print(f"inputs000000_{dist.get_rank()}", inputs["input_ids"].shape, inputs["input_ids"].flatten().tolist())
      fast_used_n_token = 0
      fast_video_token_nums = fast_video_token_nums * 4 # 注意这里的乘4是因为我们有2*2的merge patch操作
      fast_video_used_idx = len(inputs["fast_video_grid_thw"])
      for idx, n_tokens_hw in enumerate(inputs["fast_video_grid_thw"]):
        if fast_used_n_token == fast_video_token_nums:
          fast_video_used_idx = idx
          break
        fast_used_n_token += (n_tokens_hw[0] * n_tokens_hw[1] * n_tokens_hw[2])

      inputs["fast_pixel_values_videos"] = inputs["fast_pixel_values_videos"][:fast_video_token_nums]
      inputs["fast_video_grid_thw"] = inputs["fast_video_grid_thw"][:fast_video_used_idx]

      # if dist.get_rank() == 0 or True: print_input_info(inputs, f"inputs111111_{dist.get_rank()}")
      # print(f"inputs111111_{dist.get_rank()}", inputs["input_ids"].shape, inputs["input_ids"].flatten().tolist())

    if "pixel_values" in inputs and len(inputs["pixel_values"]) == 0:
        del inputs["pixel_values"]
        del inputs["image_grid_thw"]

    if "pixel_values_videos" in inputs and len(inputs["pixel_values_videos"]) == 0:
        del inputs["pixel_values_videos"]
        del inputs["video_grid_thw"]
        
    if "fast_pixel_values_videos" in inputs and len(inputs["fast_pixel_values_videos"]) == 0:
        del inputs["fast_pixel_values_videos"]
        del inputs["fast_video_grid_thw"]
          
    # num_thw = 0
    # if "image_grid_thw" in inputs:
    #   thw = inputs["image_grid_thw"]
    #   num_thw = sum([(thw[i][1] * thw[i][2]).item() for i in range(thw.size(0))])
    # num_image_id = (inputs["input_ids"] == self.image_token_id).sum()
    # if num_thw != num_image_id * 4:
    #   print(f"{num_thw=}, {num_image_id=}, {inputs=}")
    # pvs = [0]
    # if "pixel_values" in inputs:
    #   pvs = inputs["pixel_values"].shape
    # if pvs[0] != num_thw:
    #   print(f"{num_thw=}, pixel_values={pvs}")

    ############# debug cjx #############
    # rank = f"rank{dist.get_rank()}"
    # s = print_input_info(
    #   inputs1,
    #   f"{rank}_before____cut_sample_pl_{packable_length}_cjx",
    #   return_str=True
    # )
    # s += '\n' + print_input_info(
    #   inputs,
    #   f"{rank}_after____cut_sample_pl{packable_length}_cjx",
    #   return_str=True
    # )
    # s += f"{rank}_before____cut_sample_pl_{packable_length}_cjx_input_list_"
    # print(s)
    ############# debug cjx #############
    return inputs

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
                             packed_fast_pixel_values_videos: List[torch.Tensor],
                             packed_fast_video_grid_thw: List[torch.Tensor],
                             sample_idx: Optional[int] = None,
                             image_pad: bool = False):
    
    if not image_pad:
      packable_length = self.max_length - cu_seqlens[-1]
      if packable_length == 0: return

    if not image_pad and self.cut_to_pad and inputs['input_ids'].shape[1] > packable_length:
        print(packable_length, "packable_lengthpackable_lengthpackable_length")
        inputs = self._cut_sample_cjx(inputs, packable_length)

    packed_input_ids.append(inputs["input_ids"].flatten())
    packed_loss_mask.append(inputs["loss_mask"].flatten())
    packed_position_ids.append(inputs["position_ids"])
    if sample_idx is None:
      sample_idx = len(cu_seqlens) - 1
    packed_sample_idx.append(
      torch.full_like(packed_input_ids[-1], sample_idx))

    if "pixel_values" in inputs and len(inputs["pixel_values"]):
      packed_pixel_values.append(inputs["pixel_values"])
      packed_image_gird_thw.append(inputs["image_grid_thw"])
    if "pixel_values_videos" in inputs:
      packed_pixel_values_videos.append(inputs["pixel_values_videos"])
      packed_video_grid_thw.append(inputs["video_grid_thw"])
    ##### fast #####
    if "fast_pixel_values_videos" in inputs:
      packed_fast_pixel_values_videos.append(inputs["fast_pixel_values_videos"])
      packed_fast_video_grid_thw.append(inputs["fast_video_grid_thw"])
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
    packed_fast_pixel_values_videos: List[torch.Tensor] = []
    packed_fast_video_grid_thw: List[torch.Tensor] = []
    cu_seqlens: List[int] = [0]
    epochs = []
    valid_seq_len = 0
    n_pixels = 0
    for _, inputs in enumerate(buffer):
      if "pixel_values" in inputs: n_pixels += inputs["pixel_values"].shape[0]
      if "pixel_values_videos" in inputs: n_pixels += inputs["pixel_values_videos"].shape[0]
      image_pad = True if self.use_flops_balance else False
      # print(22222, self.use_flops_balance, image_pad)
      epochs.append(inputs.get("epoch_idx", None)) # inputs["image_grid_thw"][i]
      valid_seq_len += self._append_sample_packing(inputs,
                                      packed_input_ids,
                                      packed_position_ids,
                                      packed_loss_mask,
                                      packed_pixel_values,
                                      packed_pixel_values_videos,
                                      packed_image_gird_thw,
                                      packed_video_grid_thw,
                                      packed_sample_idx,
                                      cu_seqlens,
                                      packed_fast_pixel_values_videos,
                                      packed_fast_video_grid_thw,
                                      image_pad=image_pad
                                      )


    # 
    # append a pad image sequence to trigger ViT
    image_pad = self._gen_img_pad() if n_pixels % 8 == 0 else self._gen_img_pad(sz=(4, round(self.patch_size * 1.4))) # 1.4 处于 (1.25 ~ 1.5)之间
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
                                packed_fast_pixel_values_videos,
                                packed_fast_video_grid_thw,
                                sample_idx=-1,
                                image_pad=True
                                )
    

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
    ####fast####
    packed_fast_pixel_values_videos = None if len(packed_fast_pixel_values_videos) == 0 else \
      torch.cat(packed_fast_pixel_values_videos, dim=0)
    packed_fast_video_grid_thw = None if len(packed_fast_video_grid_thw) == 0 else \
      torch.cat(packed_fast_video_grid_thw, dim=0)
    ############

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

    epochs = [x for x in epochs if x is not None]
    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "pixel_values": packed_pixel_values,
      "image_grid_thw": packed_image_gird_thw, # 
      "pixel_values_videos": packed_pixel_values_videos,
      "video_grid_thw": packed_video_grid_thw,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
      "sample_idx": packed_sample_idx.to(torch.int32),
      "epoch_idx": torch.tensor([sum(epochs) / len(epochs)], dtype=torch.float32),
      "fast_pixel_values_videos": packed_fast_pixel_values_videos,
      "fast_video_grid_thw": packed_fast_video_grid_thw,
    }

    return inputs





class ChatCompletionVisionParquetDataset_keye_vitrope_slowfast(ChatCompletionVisionDataset_keye_vitrope_slowfast):
  def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.cut_to_pad = kargs.get("cut_to_pad", True)
    self.kargs = kargs
    self.num_readers = kargs.get("num_readers", 1)
    self.shuffle_window = kargs.get("shuffle_window", 0)
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
      logger.error(f"ChatCompletionVisionParquetDataset_keye_vitrope_slowfast rank{dist.get_rank()}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

    t = [data_file_list]
    dist.broadcast_object_list(t, src=0)
    data_file_list = t[0]

    logger.error(f"ChatCompletionVisionParquetDataset_keye_vitrope_slowfast rank{dist.get_rank()}: file_num={len(data_file_list)}")
    if len(data_file_list) == 0:
      raise ValueError(f"no datafile found!")

    dataset = NaiveParquetDataset(data_file_list, self.num_workers, **self.kargs)
    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)

  def state_dict(self):
    if self.dataset is None:
      return {}
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    if self.dataset is None:
      return
    self.dataset.load_state_dict(state_dict)




class BalanceParquetDataset(IterableDataset):
  def __init__(self, input, model_type, base_model_dir=None, **kwargs):
    self.input = input
    self.model_type = model_type
    self.buffer_size = kwargs.get("buffer_size", 1000)
    self.shuffle_group = kwargs.get("shuffle_group", False)
    self.base_model_dir = base_model_dir
    with open(os.path.join(self.base_model_dir, "config.json"), "r") as fp:
      config = json.load(fp)
      self.arch = config["architectures"][0]
    self._initialized = False
    self.kwargs = kwargs

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

      self.input.source_sample_cnt.setdefault(source_name, 0)
      self.input.source_sample_cnt[source_name] += 1
      
      try:
        inputs = self.input._process(sample, source_name)
        self.processed_buffer.put((inputs, source_name))
      except:
        self.input.source_error_cnt.setdefault(source_name, 0)
        self.input.source_error_cnt[source_name] += 1
        error_ratio = self.input.source_error_cnt[source_name] * 1.0 / \
          self.input.source_sample_cnt[source_name]
        logger.error(
          f"ChatCompletionVisionDataset process sample error. "
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, "
          f"errmsg={traceback.format_exc()}")
        continue
      
  def _balance_global(self, raw_input_ids, buffer, source_list):
    t2 = time.perf_counter()
    image_pad_len = getattr(self.input, "image_pad_len", 0)
    candidates = balance.greedy_subsets_without_replacement(
      raw_input_ids, self.input.max_length - image_pad_len, self.fm)
    ids_list = [[raw_input_ids[i] for i in idx] for idx in candidates]
    seq_lens = [sum(ids) for ids in ids_list]
    filtered_num = sum(s > self.input.max_length * 0.98 for s in seq_lens)
    if filtered_num > 0:
      candidates = candidates[:filtered_num]
    # if dist.get_rank() == 0:
    #   print(f"candidates: {candidates}")
    info_list = []
    num_group = len(self.fm.flops_range) + 1
    groups = [[] for _ in range(num_group)]

    flops = [[] for _ in range(num_group)]
    for c in candidates:
      llm_len = [raw_input_ids[i] for i in c]
      llm_flops = self.fm.llm_flops(llm_len)
      gid = bisect.bisect_right(self.fm.flops_range, llm_flops)
      groups[gid].append(c)
      flops[gid].append(llm_flops)
    # print(f"local_group: rank={dist.get_rank()}, flops={flops}")
      
    info_list = [len(g) for g in groups]
    t3 = time.perf_counter()
    ws = dist.get_world_size()
    all_infos = [None] * dist.get_world_size()
    dist.all_gather_object(all_infos, info_list)
    t4 = time.perf_counter()
    # if dist.get_rank() == 10:
    #   print(f"[rank=10] all_infos: {all_infos}")
    # groups_by_rank = [rank_info[-1] for rank_info in all_infos]
    # all_infos = [rank_info[:-1] for rank_info in all_infos]
    groups_by_rank = all_infos
    
    avail = []
    remains = []
    all_groups = []
    for i in range(num_group):
      group = [groups_by_rank[j][i] for j in range(ws)]
      all_groups.append(group)
      min_count = min(group)
      avail.append(min_count)
      remains.append([v - min_count for v in group])
      # remains = sum(group) - min_count * ws
      # if remains >= ws:
      #   print(group)
      #   # todo: shuffle remains
    if dist.get_rank() == 0:
      print(f"all_groups={all_groups}")
      
    found = []
    send_idx = []
    pivot = "__ds__"
    found_by_group = [[] for _ in range(len(all_groups))]
    for gid, size_list in enumerate(all_groups):
      v, scheme = balance.calculate_transfer_scheme(size_list)
      if self.rank == 0:
        print(f"rank={self.rank}, gid={gid}, v={v}, scheme={scheme}")
      self_r = size_list[self.rank]
      if v == 0:
        continue
      begin = 0
      send_data = {}
      for t in scheme:
        if t[0] != self.rank:
          continue
        assert begin < len(groups[gid]), f"{self.rank}, {begin}, {groups[gid]}, {t}"
        sends = groups[gid][begin : begin + t[2]]
        send_idx.extend(sends)
        send_data[t[1]] = []
        for idx in sends:
          samples = []
          for sid in idx:
            samples.append(buffer[sid])
            samples[-1][pivot] = source_list[sid]
          send_data[t[1]].append(samples)
        begin += t[2]
      
      # print_input_info(send_data, "send_data")
      recvs = transfer.exchange_batch_data(scheme, send_data)
      if self_r < v:
        assert self_r + len(recvs) == v, f"{self_r}, {recvs}, {v}"
        for off in range(self_r):
          # found.append((groups[gid][begin + off], True))
          found_by_group[gid].append((groups[gid][begin + off], True))

        for recv in recvs:
          found.append((recv, False))
          found_by_group[gid].append((recv, False))

      else:
        for off in range(v):
          # found.append((groups[gid][begin + off], True))
          found_by_group[gid].append((groups[gid][begin + off], True))
      
      if self.rank == 0:
        print(f'found_by_group={found_by_group}')
      # for group in found_by_group:
      #   if self.shuffle_group: 
      #     np.random.shuffle(group)

      found = sum(found_by_group, [])
    return found, send_idx

  def _balance_task(self):
    # processed_buffer -> _balance_buf -> _result_buf
    buffer, cumsum = [], 0
    maxlen = self.input.max_length - self.image_pad_len
    pivot = "__ds__"
    next_inputs = None
    while True:
      try:
        if next_inputs is not None:
          inputs, source_name = next_inputs
          next_inputs = None
        else:
          inputs, source_name = self.processed_buffer.get()

        inputs[pivot] = source_name
        buffer.append(inputs)
        cumsum += inputs["input_ids"].shape[-1]
        if cumsum < maxlen: # 可以保证这个样本有loss
          continue
        elif cumsum > maxlen:
          if 1:
            sample_original_len = inputs["input_ids"].shape[-1]
            reserved = inputs["input_ids"].shape[-1] - (cumsum - maxlen)
            # rank2__balance_task_balance_task_inputstorch.Size([1, 1380]),cumsum=17114,maxlen=13972,reserved=-1762,id=140644323789360

            print(f"rank{dist.get_rank()}__balance_task_balance_task_inputs{inputs['input_ids'].shape},cumsum={cumsum},maxlen={maxlen},reserved={reserved},sample_original_len={sample_original_len},id={id(inputs)}")
            
            cut = self.input._cut_sample_cjx(copy.deepcopy(inputs), reserved)
            
            all_loss_tokens = cut["loss_mask"].sum().item()
            if all_loss_tokens == 0 and len(buffer) == 1: # 一条样本就占满了seq len, 而且不能计算loss，删掉
              buffer.pop()
              cumsum -= sample_original_len
              if len(buffer) == 0: 
                print(f"rank{dist.get_rank()}__balance_task_balance_remove_{sample_original_len}_from_{cumsum}")
                continue # 继续填, 没有样本


            if all_loss_tokens == 0 and inputs["input_ids"].shape[-1] < maxlen:
              print(f"rank{dist.get_rank()}__balance_task_balance_task_put_bakkk_id={id(inputs)}")
              next_inputs = (inputs, source_name) # 放回去


            else:
              # 截断
              buffer[-1] = cut
        # t0 = time.perf_counter()
        send_info, send_data = balance.calculate_exchange_info(buffer, self.fm)
        # t1 = time.perf_counter()
        recvs = transfer.exchange_batch_data(send_info, send_data)
        # t2 = time.perf_counter()
        buffer = [recv[0] for recv in recvs]

        source_list = [inputs.pop(pivot) for inputs in buffer]
        # stats = balance.exchange_batch_info(buffer, source_list, self.fm)
        self._balance_buf.put((buffer, source_list, [0, 0, 0]))
        
        buffer, cumsum = [], 0
      except Exception as e:
        import traceback
        print(f"rank{dist.get_rank()} errors")
        print(traceback.format_exc())
        print(e)

  def _packing_task(self):
    while True:
      try:
        t1 = time.perf_counter()
        inputs, data_source, step_info = self._balance_buf.get()
        t2 = time.perf_counter()
        if dist.get_rank() == 0: print("packed0....")
        packed_inputs = self.input._packing(inputs)
        loss_sum = packed_inputs["loss_mask"].sum().item()
        assert loss_sum != 0, f"loss_sum=0\n{print_input_info(packed_inputs, 'packed_inputs')}"

        t3 = time.perf_counter()
        packed_inputs["data_source"] = data_source
        packed_inputs["num_samples"] = step_info[0]
        packed_inputs["num_tokens"] = step_info[1]
        packed_inputs["num_image_tokens"] = step_info[2]
        #if 'pixel_values_videos' in packed_inputs and torch.isnan(packed_inputs["pixel_values_videos"]).any(): print(f"packed_inputspacked_inputspacked_inputs_nanannnn", packed_inputs["pixel_values_videos"])
        self._post_process(packed_inputs)
        #if 'pixel_values_videos' in packed_inputs and torch.isnan(packed_inputs["pixel_values_videos"]).any(): print(f"afterrrr_packed_inputspacked_inputspacked_inputs_nanannnn", packed_inputs["pixel_values_videos"])
        t4 = time.perf_counter()
        print(f"rank={dist.get_rank()} packing_get={t2-t1}, packing={t3-t2}, post={t4-t3}")
        self._result_buf.put(packed_inputs)
      except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        print("error in _packing_task")

  def _post_process(self, inputs):
    num_thw = 0
    if "image_grid_thw" in inputs:
      thw = inputs["image_grid_thw"]
      num_thw = sum([(thw[i][1] * thw[i][2]).item() for i in range(thw.size(0))])
    num_image_id = (inputs["input_ids"] == self.input.image_token_id).sum()
    if num_thw != num_image_id * 4:
      print(f"post_debug: {num_thw=}, {num_image_id=}, {inputs=}")
    pvs = [0]
    if "pixel_values" in inputs:
      pvs = inputs["pixel_values"].shape
    if pvs[0] != num_thw:
        print(f"post_debug: {num_thw=}, pixel_values={pvs}")
    image_grid_thw = inputs.get("image_grid_thw", None)
    pixel_values = inputs.get("pixel_values", None)
    if all([v is not None for v in [pixel_values, image_grid_thw]]):
      siglip_position_ids = list()
      image_grid_hws = list()
      sample_indices = list()
      cu_seqlens = [0]

      for idx, thw in enumerate(image_grid_thw):
          thw_tuple = tuple(thw.numpy().tolist())
          numel = np.prod(thw_tuple)
          image_grid_hws.append(thw_tuple)
          image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
          siglip_position_ids.append(image_position_ids)
          sample_indices.append(torch.full((numel, ), idx, dtype=torch.int64))
          cu_seqlens.append(cu_seqlens[-1] + numel)
        
      siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(pixel_values.device)
      cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)
      sample_indices = torch.concat(sample_indices, dim=0).to(pixel_values.device)
      inputs["image_grid_hws"] = image_grid_hws
      inputs["image_position_ids"] = siglip_position_ids
      inputs["image_cu_seqlens"] = cu_seqlens
      inputs["image_sample_indices"] = sample_indices
      inputs["image_max_seqlen_q"] = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
      inputs["image_max_seqlen_k"] = inputs["image_max_seqlen_q"]
      cu_seqlens = inputs["cu_seqlens"]
      inputs["max_seqlen_q"] = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
      inputs["max_seqlen_k"] =  inputs["max_seqlen_q"]

  def _resume(self):
    resume_flag = self.kwargs.get("resume_dataloader", False)
    if not resume_flag:
      return
    resume_from = self.kwargs.get("resume_from", None)
    resume_path = os.path.join(resume_from, "dataloader_ckpt", f"rank{dist.get_rank()}.pt")
    if not os.path.exists(resume_path):
      print_rank_0(f"Warning: Dataloader checkpoint {resume_path} does not exist")
      print_rank_0("Will start training without resuming dataloader state")
    else:
      try:
        state_dict = torch.load(resume_path)["dataloader_state_dict"]
        
        self.load_state_dict(state_dict)
        print_rank_0(f"Successfully loaded dataloader state from {resume_path}, state_dict={state_dict}")
      except Exception as e:
        import traceback
        traceback.print_exc()
        print_rank_0(f"Error loading dataloader checkpoint: {str(e)}")
        print_rank_0("Will start training without resuming dataloader state")
        state_dict = None

  def __iter__(self):
    try:
      self.rank = dist.get_rank()
      self.world_size = dist.get_world_size()
      self._balance_buf = queue.Queue(maxsize=32)
      self._result_buf = queue.Queue(maxsize=16)
      
      self.input.init()
      self.dataset = self.input.dataset
      self.image_pad_len = getattr(self.input, "image_pad_len", 0)
      dist.barrier()
      self._initialized = True
      self._resume()
      from recovlm.data.balance import CustomModelFlops
      self.fm = CustomModelFlops(base_model_config=os.path.join(self.base_model_dir, "config.json"),
                                max_length=(self.input.max_length - self.image_pad_len))

      self.sample_queue = queue.Queue(maxsize=32)
      def reader_task():
          dataset_iter = iter(self.dataset)
          while True:
              sample = next(dataset_iter)
              self.sample_queue.put(sample)
      self.reader_thread = threading.Thread(target=reader_task, daemon=True)
      self.reader_thread.start()

      self.processed_buffer = queue.Queue(self.buffer_size)
      self.process_threads = [threading.Thread(target=self._process_task, daemon=True) for _ in range(16)]
      for t in self.process_threads:
        t.start()
      self.balance_thread = threading.Thread(target=self._balance_task, daemon=True)
      self.packing_thread = threading.Thread(target=self._packing_task, daemon=True)
      self.balance_thread.start()
      self.packing_thread.start()

      while True:
          t1 = time.perf_counter()
          result = self._result_buf.get()
          # print(f"11111111_rank{self.rank}_world_size{self.world_size}", result)
          # print(f"rrrrrr_{torch.isnan(result['pixel_values_videos']).any()  }", result['pixel_values_videos'])
          t2 = time.perf_counter()
          if np.random.rand() < 0.01: print(f"rank={dist.get_rank()} io_next: {t2-t1}")
          yield result
    except Exception as e:
      import traceback
      traceback.print_exc()
      print(e)
      print(f"error in __iter__")

  def state_dict(self):
    if not self._initialized:
        return {}
    print(f"save_dataset_state_dict: {self.dataset.state_dict()}")
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    print(f"load_dataset_state_dict: {self.dataset.state_dict()}, self._initialized={self._initialized}")
    if not self._initialized:
        return
    print(f"load_dataset_state_dict: {self.dataset.state_dict()}")
    self.dataset.load_state_dict(state_dict)
