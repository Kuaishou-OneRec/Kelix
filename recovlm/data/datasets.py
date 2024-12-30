from typing import Union, Iterable, Optional, List

import os
import re
import json
import math
import torch
import tarfile
import itertools
import torch.distributed as dist
import multiprocessing
import numpy as np

from torch.utils.data import IterableDataset, Dataset, DataLoader

import torch.nn.functional as F


import random
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, \
    PreTrainedTokenizer, PreTrainedTokenizerFast
# from processing_qwen2_vl import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import glob

from .templates import get_template
from .prompts import PromptLoader

# from utils import get_world_size, is_rank_0

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

class BlendedWebDataset(IterableDataset):
  """Blend Multiple dataset"""
  def __init__(self, source: str, start_seed: int = 123):
    super(BlendedWebDataset).__init__()
    with open(source, encoding="utf-8") as f:
      self.datasets = json.loads(f.read())
    self.indexes = []
    self.path = []
    # 加权的样本量
    self.w_num_samples = []
    # 预设的权重，>1=过采样，<1=降采样
    self.weights = []
    # 实际样本量
    self.num_samples = []
    # 自定义预处理
    self.processors = []
    for dataset in self.datasets:
      index_file = dataset["index"]
      with open(index_file, encoding="utf-8") as f:
        index = json.loads(f.read())
      self.indexes.append(index)
      self.num_samples.append(self.get_num_samples(index))
      self.path.append(os.path.dirname(index_file))
      self.weights.append(dataset.get("weight", 1.0))
      self.processors.append(dataset.get("processor", "default"))
      self.w_num_samples.append(int(self.num_samples[-1] * self.weights[-1]))
    self.length = sum(self.num_samples)
    self.w_length = int(sum(self.w_num_samples))
    self.buffer_size = 1000
    self.normed_weights = self.normlize_weight(self.w_length)

  def get_num_samples(self, index):
    return sum([shard["nsamples"] for shard in index["shardlist"]])

  def normlize_weight(self, weights: List[float]) -> np.array:
    return weights / np.sum(weights)

  def sample_buffer(self):
    return np.random.choice(
      list(range(len(self.datasets))),
      size=self.buffer_size, p=self.normed_weights).tolist()

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = 0 if worker_info is None else worker_info.id
    num_workers = 0 if worker_info is None else worker_info.num_workers
    this_datasets_consumed_cnt = defaultdict(int)
    for i in itertools.count(start=self.start_seed):
        np.random.seed(i)
        if worker_id == num_workers - 1:
            self._update_buffer_start_seed(i)
        
        this_worker_dataset_indices, this_worker_sample_indices = self._get_dataset_sample_index(worker_id, this_datasets_consumed_cnt)
        for i in range(len(this_worker_dataset_indices)):

            ann = self.datasets[this_worker_dataset_indices[i]][this_worker_sample_indices[i]]
            ann["this_worker_sample_index"] = this_worker_sample_indices[i]
            yield ann


  def __len__(self):
    # 训练使用加权的总长度
    return self.w_length


class BlendedWebDataset(IterableDataset):
  """Blend Multiple dataset"""

  def __init__(self, source_datasets: List[Dataset],
               weights: List[float],
               rank: int,
               world_size: int,
               num_workers: int,
               chunk_size=2000,
               sample_buffer_size=1000,
               random_seed=1024,
               state_file=""):

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

    # print(f"zzxdebug: {self.dataset_range_idx}")
    # print(f"zzxdebug: {self.dataset_states}")

    # load dataset state
    if state_file != "" and os.path.isfile(state_file):
      self._load_state(state_file)
  
  def __len__(self):
    return self.total_samples

  def _chunked_sampler(self,):
    dataset_range_idx = []
    dataset_states = []

    num_replicas = self.world_size * self.num_workers
    for d_idx, dataset in enumerate(self.datasets):
      dataset_range_idx.append([])
      dataset_states.append([])

      num_samples = len(dataset)
      for worker_id in range(self.num_workers):
        worker_chunk = (num_samples + num_replicas - 1) // num_replicas
        worker_start = worker_id * worker_chunk
        worker_end = min(worker_start + worker_chunk, num_samples)
        ranges = [(i, min(i + self.chunk_size, worker_end)) for i in range(worker_start, worker_end, self.chunk_size)]
        random.Random(self.random_seed + worker_id).shuffle(ranges)
        dataset_range_idx[d_idx].append(ranges)
        dataset_states[d_idx].append([0, 0])
      
    return dataset_range_idx, dataset_states
  
  def _load_state(self, stat_file):
    with open(stat_file, "rb") as fp:
      state_str = fp.read()
      state_dict = json.loads(state_str)
      for d_idx, _ in enumerate(self.datasets):
        for w_idx in range(self.num_workers):
          self.dataset_states[d_idx][w_idx] =  state_dict[self.rank][d_idx][w_idx]
    
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
        ann["this_worker_sample_index"] = (d_idx, worker_id, chunk_idx, chunk_inner_idx)
        yield ann
        cur_samples += 1
  
if __name__ == "__main__":
  import wids
  sources = [
    "/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json",
    "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
  ]
  datasets = [wids.ShardListDataset(source) for source in sources]

  blend_dataset1 = BlendedWebDataset(datasets, [1.1, 0.5], 0, 1, 1)

  datasets = [wids.ShardListDataset(source) for source in sources]

  blend_dataset2 = BlendedWebDataset(datasets, [1.1, 0.5], 0, 1, 1)

  assert blend_dataset1.dataset_range_idx == blend_dataset2.dataset_range_idx
  assert blend_dataset1.dataset_states == blend_dataset2.dataset_states

  for d in blend_dataset1:
    print(d)
    break

  for d in blend_dataset2:
    print(d)
    break

  # for d in datasets[0]:
  #   print(d)
  #   break

  # for d in datasets[1]:
  #   print(d)
  #   break