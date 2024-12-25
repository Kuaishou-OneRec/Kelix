from typing import Union, Iterable, Optional

import os
import re
import json
import math
import torch
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
               tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str],
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
          self.records = json.loads(f.reads())
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
