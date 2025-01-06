from typing import Union, Iterable, Optional, List, Dict, Tuple
# from absl import logging
import logging

import os
import re
import wids
import json
import math
import torch
import tarfile
import itertools
import traceback
import pickle
import torch.distributed as dist

import webdataset as wds

from PIL import Image

from collections import defaultdict

import multiprocessing
import numpy as np

from torch.utils.data import IterableDataset, Dataset, DataLoader

import torch.nn.functional as F


import random
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, \
    PreTrainedTokenizer, PreTrainedTokenizerFast
# from processing_qwen2_vl import Qwen2VLProcessor
from recovlm.utils.qwen_vl_utils import process_vision_info
import glob

from .templates import get_template
from .prompts import PromptLoader

# from utils import get_world_size, is_rank_0


logger = logging.getLogger(__name__)

RESPONSE_TEMPLATE = "{% for message in messages %}{{message['content'] + '<|im_end|>'}}{% endfor %}"

class LLaVA_CC3M_Dataset(Dataset):
  def __init__(self, source, processor_path, max_length=None):
    super(LLaVA_CC3M_Dataset).__init__()
    self.source = source
    with open(os.path.join(self.source, "chat.json"), encoding="utf-8") as f:
      self.sessions = json.loads(f.read())
    self.processor = AutoProcessor.from_pretrained(processor_path)
    self.tokenizer = AutoTokenizer.from_pretrained(
        processor_path, use_fast=False)
    self.tokenizer.padding_side = "right"
    self.max_length = max_length
    if not self.max_length:
      self.max_length = self.tokenizer.model_max_length

  def __getitem__(self, index):
    session = self.sessions[index]
    return session

  def build_collate_fn(self):
    # TODO: SUPPORT TRUNCATE
    def collate_fn(sessions):
      prompt_messages = []
      for session in sessions:
        prompt = session["conversations"][0]["value"]
        img = os.path.join(self.source, session["image"])
        if prompt.endswith("<image>"):
          content = [
              {"type": "text", "text": re.sub(r"\n<image>", "", prompt)},
              {
                  "type": "image",
                  "image": img,
                  "resized_height": 224,
                  "resized_width": 224,
              }
          ]
        else:
          content = [
              {
                  "type": "image",
                  "image": img,
                  "resized_height": 224,
                  "resized_width": 224,
              },
              {"type": "text", "text": re.sub(r"<image>\n", "", prompt)}
          ]
        prompt_messages.append([{
            "role": "user",
            "content": content,
        }])
      text = self.processor.apply_chat_template(
          prompt_messages, tokenize=False, add_generation_prompt=True
      )
      image_inputs, video_inputs = process_vision_info(prompt_messages)
      inputs = self.processor(
          text=text,
          images=image_inputs,
          videos=video_inputs,
          padding=True,
          padding_side="left",
          return_tensors="pt",
      )

      response_messages = []
      for session in sessions:
        response = session["conversations"][1]["value"]
        response_messages.append([{"content": response}])
      # right pad response to concat with prompt
      response_inputs = self.tokenizer.apply_chat_template(
          response_messages,
          chat_template=RESPONSE_TEMPLATE,
          padding=True,
          padding_side="right",
          add_generation_prompt=False,
          return_tensors="pt",
      )
      response_mask = (
          response_inputs != self.tokenizer.pad_token_id).type(torch.int64)
      loss_mask = torch.cat(
          [torch.zeros_like(inputs["input_ids"]), response_mask], dim=-1
      )
      inputs["attention_mask"] = torch.cat(
          [inputs["attention_mask"], response_mask], dim=-1)
      inputs["input_ids"] = torch.cat(
          [inputs["input_ids"], response_inputs], dim=-1)
      inputs["loss_mask"] = loss_mask

      # TODO: improve truncate
      for key in ["input_ids", "attention_mask", "loss_mask"]:
        inputs[key] = inputs[key][:, :self.max_length]
      _type = {
          "input_ids": torch.int64,
          "attention_mask": torch.int64,
          "pixel_values": torch.float32,
          "image_grid_thw": torch.int64,
          "video_grid_thw": torch.int64,
          "loss_mask": torch.int64
      }
      assert inputs["input_ids"].shape == inputs["loss_mask"].shape
      assert inputs["input_ids"].shape == inputs["attention_mask"].shape
      return inputs

    return collate_fn

  def __len__(self):
    return len(self.sessions)

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
      image_nums = 0 if image_token_id == None else (vision_tokens == image_token_id).sum()
      video_nums = 0 if video_token_id == None else (vision_tokens == video_token_id).sum()
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

class BlendedWebDataset(IterableDataset):
  """Blend Multiple dataset"""

  def __init__(self, source_datasets: List[Dataset],
               weights: List[float],
               rank: int,
               world_size: int,
               num_workers: int,
               chunk_size=1000000,
               sample_buffer_size=1000,
               random_seed=1024):

    super(BlendedWebDataset).__init__()
    
    assert len(source_datasets) == len(weights)

    self.datasets = source_datasets
    # weights: >1=过采样，<1=降采样
    self.weighted_data_len = [weights[idx] * len(d) for idx, d in enumerate(self.datasets)]
    self.sample_ratio = [d_len / sum(self.weighted_data_len) for d_len in self.weighted_data_len]
    self.total_samples = sum(self.weighted_data_len)

    self.rank, self.world_size, self.num_workers = rank, world_size, num_workers
    self.chunk_size, self.sample_buffer_size = chunk_size, sample_buffer_size
    self.random_seed = random_seed

    # dataset_sample_idx[data_idx][worker_idx] = [(chunk1_s, chunk1_e), ..., (chunkn_s, chunkn_e),]
    # dataset_states[data_idx][worker_idx] = [chunk_idx, chunk_inner_idx]
    self.dataset_range_idx, self.dataset_states = self._chunked_sampler()

  def __len__(self):
    return self.total_samples

  def _chunked_sampler(self,):
    manager = multiprocessing.Manager()
    dataset_range_idx = manager.list()
    dataset_states = manager.list()

    num_replicas = self.world_size * self.num_workers
    for d_idx, dataset in enumerate(self.datasets):
      dataset_range_idx.append(manager.list())
      dataset_states.append(manager.list())

      num_samples = len(dataset)
      for worker_id in range(self.num_workers):
        worker_chunk = (num_samples + num_replicas - 1) // num_replicas
        worker_start = worker_id * worker_chunk
        worker_end = min(worker_start + worker_chunk, num_samples)
        ranges = [(i, min(i + self.chunk_size, worker_end)) for i in range(worker_start, worker_end, self.chunk_size)]
        random.Random(self.random_seed + worker_id).shuffle(ranges)
        dataset_range_idx[d_idx].append(manager.list(ranges))
        dataset_states[d_idx].append(manager.list([0, 0]))
      
    return dataset_range_idx, dataset_states
  
  def set_state(self, state_dict):
    for d_idx in state_dict:
      for w_idx in state_dict[d_idx]:
        chunk_idx, chunk_inner_idx, _ = state_dict[d_idx][w_idx]
        self.dataset_states[d_idx][w_idx][0] = chunk_idx
        self.dataset_states[d_idx][w_idx][1] = chunk_inner_idx

    print(f"load_success: {self.dataset_states}")

  def _sample_buffer(self, worker_id):
    np.random.seed(self.random_seed + worker_id) 
    dataset_indices = np.random.choice(list(range(len(self.datasets))), size=self.sample_buffer_size, p=self.sample_ratio).tolist()

    idx_buffer = []
    for d_idx in dataset_indices:
      chunk_idx, chunk_inner_idx = self.dataset_states[d_idx][worker_id]
      d_start, d_end = self.dataset_range_idx[d_idx][worker_id][chunk_idx]
      sample_indexes = list(range(d_start, d_end))
      random.Random(self.random_seed + worker_id + chunk_idx).shuffle(sample_indexes)
      
      data_indices = sample_indexes[chunk_inner_idx]
      idx_buffer.append((d_idx, data_indices, chunk_idx, chunk_inner_idx))

      # update sampler index
      chunk_inner_idx += 1 
      if chunk_inner_idx >= len(sample_indexes):
        chunk_inner_idx = 0
        chunk_idx += 1
        if chunk_idx >= len(self.dataset_range_idx[d_idx][worker_id]):
          chunk_idx = 0
      
      self.dataset_states[d_idx][worker_id] = [chunk_idx, chunk_inner_idx]
    
    return idx_buffer
  
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = 0 if worker_info is None else worker_info.id
    num_workers = 1 if worker_info is None else worker_info.num_workers

    cur_samples = 0

    assert num_workers == self.num_workers

    while True:
      if cur_samples > self.total_samples:
        break
      idx_buffer = self._sample_buffer(worker_id)
      while len(idx_buffer) > 0:
        d_idx, data_indices, chunk_idx, chunk_inner_idx = idx_buffer.pop()
        ann = self.datasets[d_idx][data_indices]
        ann["this_worker_sample_index"] = (d_idx, worker_id, chunk_idx, chunk_inner_idx, cur_samples)
        yield ann
        cur_samples += 1
  
class BlendDatasetCkptManager:
  def __init__(self, ckpt_path, rank, world_size, num_data_source, num_workers):
    self.ckpt_path = ckpt_path
    self.rank, self.world_size = rank, world_size
    self.num_data_source = num_data_source
    self.num_workers = num_workers

    # init state_dict: state_dict[d_idx][w_idx]
    self.state_dict = {}
    self.latest_step = 0
    
  @staticmethod
  def merge_all_state(state_list):
    merge_state_dict = {}
    for d_idx, worker_id, chunk_idx, chunk_inner_idx, sample_idx in state_list:
      merge_state_dict.setdefault(d_idx, {})
      if worker_id not in merge_state_dict[d_idx]:
        merge_state_dict[d_idx][worker_id] = (chunk_idx, chunk_inner_idx, sample_idx)
      else:
        _, _, m_sample_idx = merge_state_dict[d_idx][worker_id]
        if sample_idx > m_sample_idx:
          merge_state_dict[d_idx][worker_id] = (chunk_idx, chunk_inner_idx, sample_idx)
    return merge_state_dict
  
  def update_step(self, step, batch_state_dict):
    self.latest_step = step
    for d_idx in batch_state_dict:
      self.state_dict.setdefault(d_idx, {})
      for w_idx in batch_state_dict[d_idx]:
        self.state_dict[d_idx].setdefault(w_idx, (0, 0, 0))
        chunk_idx, chunk_inner_idx, sample_idx = batch_state_dict[d_idx][w_idx]
        if self.state_dict[d_idx][w_idx][2] < sample_idx:
          self.state_dict[d_idx][w_idx] = (chunk_idx, chunk_inner_idx, sample_idx)

  def save_ckpt(self):
    os.makedirs(self.ckpt_path, exist_ok=True)
    ckpt_file = os.path.join(self.ckpt_path, f"rank{self.rank}-dataset-step{self.latest_step}.pkl")
    with open(ckpt_file, "wb+") as fp:
      pickle.dump(self.state_dict, fp)
  
  def load_ckpt(self, ckpt_file):
    with open(ckpt_file, "rb") as fp:
      self.state_dict = pickle.load(fp)
    
    # clear all sample idx
    for d_idx in self.state_dict:
      for w_idx in self.state_dict[d_idx]:
        chunk_idx, chunk_inner_idx, _ = self.state_dict[d_idx][w_idx]
        self.state_dict[d_idx][w_idx] = (chunk_idx, chunk_inner_idx, 0)
    print(f"[rank{self.rank}] load dataset ckpt {ckpt_file} success.")
    return self.state_dict
  
  def extract_rank_and_step(self, filename):
    pattern = r'rank(\d+)-dataset-step(\d+)\.pkl'
    match = re.match(pattern, filename)
    if match:
      rank = int(match.group(1))
      step = int(match.group(2))
      return rank, step
    return None, None
  
  def load_latest_ckpt(self,):
    max_step = 0
    max_step_ckpt_fn = ""
    for fn in os.listdir(self.ckpt_path):
      rank, step = self.extract_rank_and_step(fn)
      if rank is not None and step is not None:
        if rank == self.rank and step > max_step:
          max_step = step
          max_step_ckpt_fn = os.path.join(self.ckpt_path, fn)
    if max_step_ckpt_fn == "":
      return self.state_dict
    else:
      return self.load_ckpt(max_step_ckpt_fn)

class ImageTextPairDatasetWithPacking(IterableDataset):
  def __init__(self,
               sources: str,
               processor,
               max_length: int,
               min_visual_tokens: int,
               max_visual_tokens: int,
               spatial_merge_size: int,
               image_token_id: int,
               video_token_id: int,
               vision_start_token_id: int,
               patch_size: int,
               shrink_ratio: float = 0.9,
               max_retry: int = 5,
               multiple_of: int = 8,
               data_format: str = "chatml",
               shuffle_size: int = 100000):
    super(ImageTextPairDatasetWithPacking).__init__()
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
    for source in sources.split(","):
      with open(source, encoding="utf-8") as f:
        index = json.loads(f.read())["shardlist"]
        for item in index:
          urls.append(os.path.join(os.path.dirname(source), item["url"]))
          self.total_samples += item["nsamples"]

    # def warn_and_continue(e):
    #   print("Warning: skipping a corrupt sample.", e)

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
    max_visual_tokens = max(max_visual_tokens, self.min_visual_tokens)
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
                    "min_pixels": self.min_visual_tokens * self.patch_size ** 2,
                    "max_pixels": max_visual_tokens * self.patch_size ** 2
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
              "min_pixels": self.min_visual_tokens * self.patch_size ** 2,
              "max_pixels": max_visual_tokens * self.patch_size ** 2
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


class VisionTextDatasetWithPacking(IterableDataset):
  def __init__(self,
               sources: str,
               processor,
               max_length: int,
               min_visual_tokens: int,
               max_visual_tokens: int,
               spatial_merge_size: int,
               image_token_id: int,
               video_token_id: int,
               vision_start_token_id: int,
               patch_size: int,
               n_frames: int = -1,
               fps: int = -1,
               min_frame_visual_tokens: int = -1,
               max_frame_visual_tokens: int = -1,
               shrink_ratio: float = 0.9,
               max_retry: int = 5,
               multiple_of: int = 8,
               shuffle_size: int = 100000):
    super(VisionTextDatasetWithPacking).__init__()
    self.processor = processor
    self.max_length = max_length
    self.min_visual_tokens = min_visual_tokens
    self.max_visual_tokens = max_visual_tokens

    self.n_frames = n_frames
    self.fps = fps
    if self.n_frames > 0 and self.fps > 0:
      logger.info(f"{fps=} not work when n_frames>0 {n_frames=}")

    self.min_frame_visual_tokens = min_frame_visual_tokens if min_frame_visual_tokens > 0 else min_visual_tokens
    self.max_frame_visual_tokens = max_frame_visual_tokens if max_frame_visual_tokens > 0 else max_visual_tokens
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
    self.total_samples = 0

    # init webdataset
    urls = []
    for source in sources.split(","):
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
    ).shuffle(shuffle_size)

    dataset = dataset.decode("pil", handler=wds.warn_and_continue) # for image decode
    # dataset = dataset.decode("pil", handler=wds.warn_and_continue) # for video decode
    
    self.dataset = dataset

  def _split_messages(self, messages):
    output_msg = None
    input_msgs = []
    for msg in messages:
      if msg["role"] == "system" or msg["role"] == "user":
        input_msgs.append(msg)
      if msg["role"] == "assistant":
        output_msg = msg
    return input_msgs, output_msg
  
  def _get_text_content(self, message):
    content_res = ""
    assert "content" in message
    content = message["content"]
    if isinstance(content, str):
      content_res = content
    elif isinstance(content, list):
      for c in content:
        if c["type"] == "text":
          content_res = c["text"]
    return content_res

  def _gen_image_video_extend(self, max_visual_tokens, max_frame_visual_tokens):
    image_extend = {
      "min_pixels": self.min_visual_tokens * ((2 * self.patch_size) ** 2),
      "max_pixels": max_visual_tokens * ((2 * self.patch_size) ** 2)
    }

    video_extend = {
      "min_pixels": self.min_frame_visual_tokens * ((2 * self.patch_size) ** 2),
      "max_pixels": max_frame_visual_tokens * ((2 * self.patch_size) ** 2)
    }
    if self.n_frames > 0:
      video_extend["nframes"] = self.n_frames
    
    if self.fps > 0:
      video_extend["fps"] = self.fps

    return image_extend, video_extend

  def _fill_prompt(self, messages, videos, images,
                   video_extend={}, image_extend={}):
    for msg in messages:
      if msg["role"] == "user":
        for content in msg["content"]:
          if content["type"] == "video":
            content.update(video_extend) # TODO: add a flag to check whether rewrite
            if content["video"] in videos:
              content['video'] = videos[content['video']]
          elif content["type"] == "image":
            content.update(image_extend) # TODO: add a flag to check whether rewrite
            if content['image'] in images:
              content["image"] = images[content['image']]

    return messages

  def _process_sample(self, samples: Dict[str, Union[str, bytes, Image.Image]],
                      max_visual_tokens: int = 1280, max_frame_visual_tokens: int = 1280 * 5):

    max_visual_tokens = max(max_visual_tokens, self.min_visual_tokens)
    max_frame_visual_tokens = max(max_frame_visual_tokens, self.min_frame_visual_tokens)

    videos = {}
    images = {}
    messages = []
    for key in samples:
      if key.endswith("jpg"):
        images[key] = samples[key]
        if images[key].mode != "RGB":
          images[key] = images[key].convert("RGB")
      elif key.endswith("mp4") or key.endswith("mov"):
        videos[key] = samples[key]
      elif key == "json":
        if "messages" in samples["json"]:
          messages = samples["json"]["messages"]

        # TODO: remove "message" key support
        if "message" in samples["json"]:
          messages = samples["json"]["message"]

    if len(messages) == 0:
      raise ValueError(
          f"Unable to generate sample without messages field."
      )

    # generate prompt & response
    input_msgs, output_msg = self._split_messages(messages)
    if len(input_msgs) == 0 or output_msg is None:
      raise ValueError(
          f"Unable to generate prompt with incomplete message."
      )
    image_extend, video_extend = self._gen_image_video_extend(
        max_visual_tokens, max_frame_visual_tokens)
    prompt = self._fill_prompt(
        input_msgs, videos, images, image_extend=image_extend, video_extend=video_extend)
    text = self.processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    # TODO: datacamp的aspect_ratio过大会触发异常，提前处理或丢掉？
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    response = [{"content": self._get_text_content(output_msg)}]
    response_ids = self.processor.tokenizer.apply_chat_template(
      [response],
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

  def _process(self, sample):
    # self._may_filter(sample)
    max_visual_tokens = self.max_visual_tokens
    max_frame_visual_tokens = self.max_frame_visual_tokens

    for retry in range(self.max_retry):
      inputs = self._process_sample(sample, max_visual_tokens, max_frame_visual_tokens)
      if not inputs:
        raise ValueError("Empty inputs, skip")
      if inputs["input_ids"].shape[-1] > self.max_length:
        max_visual_tokens = (max_visual_tokens * self.shrink_ratio)
        max_frame_visual_tokens = (max_frame_visual_tokens * self.shrink_ratio)
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
    packed_pixel_values_videos = []
    packed_video_grid_thw = []
    cu_seqlens = [0]

    for inputs in buffer:
      video_flag = False
      image_flag = False
      
      packed_input_ids.append(inputs["input_ids"].flatten())
      packed_loss_mask.append(inputs["loss_mask"].flatten())
      packed_position_ids.append(inputs["position_ids"])
      cu_seqlens.append(cu_seqlens[-1] + len(inputs["input_ids"][0]))

      if "pixel_values" in inputs:
        packed_pixel_values.append(inputs["pixel_values"])
        packed_image_gird_thw.append(inputs["image_grid_thw"])
        image_flag = True
        
      if "pixel_values_videos" in inputs:
        packed_pixel_values_videos.append(inputs["pixel_values_videos"])
        packed_video_grid_thw.append(inputs["video_grid_thw"])
        video_flag = True

    packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
    packed_loss_mask = torch.cat(packed_loss_mask, dim=0).unsqueeze(0)
    packed_position_ids = torch.cat(packed_position_ids, dim=-1)
    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32)
    }

    if image_flag:
      packed_pixel_values = torch.cat(packed_pixel_values, dim=0)
      packed_image_gird_thw = torch.cat(packed_image_gird_thw, dim=0)
      inputs["pixel_values"] = packed_pixel_values
      inputs["image_grid_thw"] = packed_image_gird_thw

    if video_flag:
      packed_pixel_values_videos = torch.cat(packed_pixel_values_videos, dim=0)
      packed_video_grid_thw = torch.cat(packed_video_grid_thw, dim=0)
      inputs["pixel_values_videos"] = packed_pixel_values_videos
      inputs["video_grid_thw"] = packed_video_grid_thw
    
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
