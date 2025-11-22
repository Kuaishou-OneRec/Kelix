from typing import Dict, Any, Union, List

import json
import torch

from transformers import AutoTokenizer
from muse.data.datasets.base import DistributedDataset
from muse.data.templates import PromptLoader, TemplateLoader

from jinja2 import Template

class TextDataset(DistributedDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               system_prompt: str = "default",
               add_system_prompt: bool = True,
               chat_template: str = "chat",
               add_prompt_loss: bool = False,
               tokenizer_path: str = None,
               max_length_per_sample: int = 2048,
               pad_to_multiple_of: int = 1,
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed, **kwargs)
    prompt_loader = PromptLoader()
    self.system_prompt = prompt_loader.load(system_prompt)
    self.add_system_prompt = add_system_prompt
    template_loader = TemplateLoader()
    self.chat_template = Template(template_loader.load(chat_template))
    self.tokenizer_path = tokenizer_path
    if self.tokenizer_path:
      self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
    else:
      self.tokenizer = None
    self.max_length_per_sample = max_length_per_sample
    self.pad_to_multiple_of = pad_to_multiple_of

  def process_messages(self,
                       messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

    """Process messages into input_ids and loss_mask"""
    if self.add_system_prompt and not messages[0]["role"] == "system":
      system = {"role": "system", "content": self.system_prompt}
      messages.insert(0, system)
    
    input_ids = []
    loss_mask = []
    prompt_loss_mask = int(self.add_prompt_loss)

    for turn in messages:
      if self.add_system_prompt and turn["role"] == "system":
        _text = self.chat_template.render(messages=[turn])
        _input_ids = self.tokenizer.encode(_text)
        _loss_mask = [prompt_loss_mask] * len(_input_ids)
        input_ids.extend(_input_ids)
        loss_mask.extend(_loss_mask)
      elif turn["role"] == "user":
        _text = self.chat_template.render(
          messages=[turn],
          add_system_prompt=False,
          add_generation_prompt=True
        )
        _input_ids = self.tokenizer.encode(_text)
        _loss_mask = [prompt_loss_mask] * len(_input_ids)
        input_ids.extend(_input_ids)
        loss_mask.extend(_loss_mask)
      elif turn["role"] == "assistant":
        _text = self.chat_template.render(
          messages=[turn],
          add_system_prompt=False,
          add_prefix=False
        )
        _input_ids = self.tokenizer.encode(_text)
        _loss_mask = [1] * len(_input_ids)
        input_ids.extend(_input_ids)
        loss_mask.extend(_loss_mask)
      else:
        raise ValueError(f"Invalid role: {turn['role']}")
    
    # pad to multiple of pad_to_multiple_of
    pad_length = self.pad_to_multiple_of - len(input_ids) % self.pad_to_multiple_of
    if pad_length > 0:
      input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
      loss_mask = loss_mask + [0] * pad_length

    if len(input_ids) > self.max_length_per_sample:
      input_ids = input_ids[:self.max_length_per_sample]
      loss_mask = loss_mask[:self.max_length_per_sample]
    
    # if not loss_mask is all 0, return None
    if sum(loss_mask) == 0:
      return None

    # batchify
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    loss_mask = torch.tensor(loss_mask).unsqueeze(0)
    position_ids = torch.arange(len(input_ids)).unsqueeze(0)

    return {
      "input_ids": input_ids,
      "loss_mask": loss_mask,
      "position_ids": position_ids,
    }

  def process_segments(self,
                       segments: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process segments into input_ids and loss_mask"""

    input_ids = []
    loss_mask = []

    for segment in segments:
      if segment["type"] == "text":
        _input_ids = self.tokenizer.encode(segment["text"])
        _loss_mask = [1] * len(_input_ids)
        input_ids.extend(_input_ids)
        loss_mask.extend(_loss_mask)
    
    # pad to multiple of pad_to_multiple_of
    pad_length = self.pad_to_multiple_of - len(input_ids) % self.pad_to_multiple_of
    if pad_length > 0:
      input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
      loss_mask = loss_mask + [0] * pad_length

    if len(input_ids) > self.max_length_per_sample:
      input_ids = input_ids[:self.max_length_per_sample]
      loss_mask = loss_mask[:self.max_length_per_sample]
    
    # if not loss_mask is all 0, return None
    if sum(loss_mask) == 0:
      return None

    # batchify
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    loss_mask = torch.tensor(loss_mask).unsqueeze(0)
    position_ids = torch.arange(len(input_ids)).unsqueeze(0)

    return {
      "input_ids": input_ids,
      "loss_mask": loss_mask,
      "position_ids": position_ids,
    }
  
  def get_content(self,
                  sample: Dict[str, Any],
                  key: str) -> List[Dict[str, Any]]:
    """Get content from sample"""
    content = sample.get(key, "[]")
    try:
      content = json.loads(content)
    except json.JSONDecodeError:
      return []
    return content

  def process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    messages = self.get_content(sample, "messages")
    segments = self.get_content(sample, "segments")
    if messages:
      return self.process_messages(messages)
    elif segments:
      return self.process_segments(segments)
    else:
      return None
    
  def pack_sample(self,
                  inputs: Dict[str, torch.Tensor],
                  new_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pack new_inputs into inputs"""
    for key in new_inputs:
      if not key in inputs:
        inputs[key] = new_inputs[key]
        continue
      inputs[key] = torch.cat([inputs[key], new_inputs[key]], dim=-1)
    return inputs
  
  def get_sample_length(self, sample: Dict[str, torch.Tensor]) -> int:
    """Get sample length"""
    return sample["input_ids"].shape[1]
