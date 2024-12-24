from typing import Union, Iterable

import os
import re
import json
import math
import torch
import pandas as pd
import tarfile
import torch.distributed as dist

from torch.utils.data import IterableDataset, Dataset, DataLoader

import torch.nn.functional as F


import random
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, \
    PreTrainedTokenizer, PreTrainedTokenizerFast
# from processing_qwen2_vl import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import glob

# from utils import get_world_size, is_rank_0

RESPONSE_TEMPLATE = "{% for message in messages %}{{message['content'] + '<|im_end|>'}}{% endfor %}"


def extract_tar_file(tar_file_path, destination_path):
  if not os.path.exists(destination_path):
    os.makedirs(destination_path)

  with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall(path=destination_path)
    print(f"Files extracted to {destination_path}")


def shard(data_dir, rank=0, world_size=1, shuffle=True, reshard=False):
  shard_dir = os.path.join(data_dir, f"sharded_{world_size}")
  if is_rank_0():
    if reshard or (not os.path.exists(shard_dir)):
      files = glob.glob(os.path.join(data_dir, "*.parquet"))
      num_files = len(files)
      num_files_per_rank = num_files // world_size
      print(f"Shard {num_files} files to {world_size} parts.")
      random.shuffle(files)
      os.makedirs(shard_dir)
      for rank in range(world_size):
        start = rank * num_files_per_rank
        end = (rank + 1) * \
            num_files_per_rank if rank < (world_size - 1) else num_files
        with open(os.path.join(shard_dir, f"rank_{rank}.txt"), "w", encoding="utf-8") as f:
          for file in files[start:end]:
            f.write(file + "\n")
  else:
    dist.barrier()
  return os.path.join(shard_dir, f"rank_{rank}")


class MscocoDataset(IterableDataset):
  def __init__(self, source, world_size, rank):
    super(MscocoDataset).__init__()
    self.meta = shard(source, world_size=world_size, rank=rank)
    self.file_list = []
    with open(self.meta) as f:
      for line in f:
        self.file_list.append(line.strip())

  def tokenize(self, text, img):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    response_messages = {"role": "assistant", "content": text}
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    response = processor.tokenizer.apply_chat_template(
        response_messages, chat_template="", add_generation_prompt=False)
    label_mask = len(inputs["input_ids"]) * 0 + len(response) * 1
    inputs["label_mask"] = label_mask
    return inputs

  def build_iter(self, files):
    def iterable():
      for file in files:
        tar_file = re.sub(r".parquet$", ".tar", file)
        img_dir = re.sub(r".parquet$", "", file)
        extract_tar_file(tar_file, img_dir)
        df = pd.read_parquet(file)
        for text, img, status in zip(df["caption"], df["key"], df["status"]):
          if status != "success":
            continue
          tokenized = self.tokenize(text, image)
          yield tokenized
    return iterable()

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    start = 0
    end = len(self.file_list)
    if worker_info is None:  # single-process data loading, return the full iterator
      iter_start = 0
      iter_end = len(self.file_list)
    else:  # in a worker process
      # split workload
      per_worker = int(
          math.ceil(
              (end -
               start) /
              float(
                  worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.end)
    return self.build_iter(self.file_list[iter_start:iter_end])


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


chat_template = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{% if (message['role'] in ['system', 'user']) %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% generation %}"
    "{{ message['content'] + '<|im_end|>\n' }}"
    "{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}")


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
  def __init__(self,
               source: Union[str, Iterable],
               tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str],
               input_key: str = "messages",
               system_prompt: str = "You are a helpful assistant."):
    super(ChatCompletionDataset).__init__()
    self.source = source
    if isinstance(tokenizer, str):
      tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    self.tokenizer = tokenizer
    self.input_key = input_key
    self.system_prompt = system_prompt
    self.records = []
    if isinstance(source, str):
      # TODO: support parquet, support hdfs
      with open(self.source, encoding="utf-8") as f:
        for line in f:
          self.records.append(json.loads(line))
    else:
      self.records = source

  def __len__(self):
    return len(self.records)

  def __getitem__(self, index):
    messages = self.records[index][self.input_key]
    if self.system_prompt:
      if messages[0]["role"] == "system":
        messages[0]["content"] = self.system_prompt
      else:
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + messages
    tokenized = self.tokenizer.apply_chat_template(
        [messages],
        chat_template=chat_template,
        return_assistant_tokens_mask=True,
        return_dict=True,
        return_tensors="pt",
    )
    tokenized["loss_mask"] = tokenized.pop("assistant_masks")
    return tokenized

  def collate_fn(self, items):
    all_input_ids = [torch.tensor(item["input_ids"][0]) for item in items]
    all_attention_mask = [
        torch.tensor(
            item["attention_mask"][0]) for item in items]
    all_loss_mask = [torch.tensor(item["loss_mask"][0]) for item in items]
    batch = {}
    batch["input_ids"] = zero_pad_sequences(
        all_input_ids, "right", self.tokenizer.pad_token_id)
    batch["attention_mask"] = zero_pad_sequences(
        all_attention_mask, "right")
    batch["loss_mask"] = zero_pad_sequences(
        all_loss_mask, "right")
    return batch
