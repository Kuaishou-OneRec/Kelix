# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal dataset
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Iterator
import os
import random
import json
import logging
import base64
from io import BytesIO

from muse.data.datasets.tokenizer_dataset import get_rope_index_qwen3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T

from transformers import AutoTokenizer, AutoProcessor, Qwen2VLProcessor, Qwen2VLConfig
from muse.data.keye_vl_utils_video import process_vision_info

import signal
import time
import copy
import traceback
import torch.distributed as dist

from muse.data.image_augs import AutoAugmentWrapper
from muse.data.datasets.base import DistributedDataset, load_image

from torch.utils.data import IterableDataset

from muse.utils.common import print_rank_0
from .ar_utils.resolution_finder import ResolutionFinder
from .ar_utils.prompt_setter import SystemPromptByTask
from .ar_utils.pre_resize_ops import resize_with_aspect_ratio_check, resize_and_center_crop

logger = logging.getLogger(__name__)


from .tokenizer_dataset import \
  ChatCompletionVisionDataset_keye_vitrope_slowfast, \
  get_assistant_mask, \
  get_rope_index_qwen3


class ARChatCompletionVisionDataset(ChatCompletionVisionDataset_keye_vitrope_slowfast):
  """
  Merged dataset class for keye vitrope slowfast image.
  Directly inherits from DistributedDataset, combining functionality with slowfast-specific enhancements.
  """
  def __init__(self,
               aspect_ratio_threshold=None,
               force_assistant_image_size=None,
               use_resolution_finder=False,
               resolution_finder_kwargs=None,
               **kwargs
               ):
      super().__init__(**kwargs)
      if resolution_finder_kwargs is None: resolution_finder_kwargs = {}
      self.force_assistant_image_size = force_assistant_image_size
      self.aspect_ratio_threshold = aspect_ratio_threshold
      self.reso_finder = ResolutionFinder(**resolution_finder_kwargs) if use_resolution_finder else None
      task2prompt_coarse = {
        "image_edit": self.reso_finder.get_system_prompt("edit"),
        "image_generation": self.reso_finder.get_system_prompt("generation"),
        "__default__": self.reso_finder.get_system_prompt("understanding")
      }
      self.system_prompt_setter = SystemPromptByTask(task2prompt_coarse)

  def _fill_image_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):

    min_visual_tokens_per_image = conf["min_visual_tokens_per_image"]
    max_visual_tokens_per_image = conf["max_visual_tokens_per_image"]
    if isinstance(block["image"], str) and block["image"] in sample_dict:
      image = sample_dict[block["image"]]
    elif isinstance(block["image"], 
    str) and os.path.exists(block["image"]) and os.path.isabs(block["image"]):
      image = Image.open(block["image"])
    else:
      image = block["image"]

    if self.force_assistant_image_size is not None:
      if self.aspect_ratio_threshold:
        image = resize_with_aspect_ratio_check(image, self.force_assistant_image_size, self.force_assistant_image_size, aspect_ratio_threshold=self.aspect_ratio_threshold)
      else:
        image = resize_and_center_crop(image, self.force_assistant_image_size, self.force_assistant_image_size)

    if self.aspect_ratio_threshold is None and self.reso_finder is not None:
      image = self.reso_finder.crop_and_resize_image(image)

    if image.mode != "RGB":
      image = image.convert("RGB")

    image = self.auto_aug(image)
    block["image"] = image
    block["min_pixels"] = min_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
    block["max_pixels"] = max_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)

    if 'min_visual_tokens_per_fast_image' in conf:
      block["fast_min_pixels"] = conf["min_visual_tokens_per_fast_image"] * (self.kwargs["fast_patch_size"] ** 2) * \
          (self.spatial_merge_size ** 2)
    if 'max_visual_tokens_per_fast_image' in conf:
      block["fast_max_pixels"] = conf["max_visual_tokens_per_fast_image"] * (self.kwargs["fast_patch_size"] ** 2) * \
          (self.spatial_merge_size ** 2)
  
  def _process_chat(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "message" in sample["json"] or "messages" in sample["json"]
    data_conf["max_visual_tokens_per_image"] = max(
        data_conf["max_visual_tokens_per_image"], data_conf["min_visual_tokens_per_image"])
    
    data_conf["max_visual_tokens_per_frame"] = max(
        data_conf["max_visual_tokens_per_frame"], data_conf["min_visual_tokens_per_frame"])
    
    data_gen_conf = copy.deepcopy(data_conf)

    if getattr(self, "min_visual_tokens_per_gen_image", -1) != -1:
      data_gen_conf["min_visual_tokens_per_gen_image"] = self.min_visual_tokens_per_gen_image
    if getattr(self, "max_visual_tokens_per_gen_image", -1) != -1:
      data_gen_conf["max_visual_tokens_per_gen_image"] = self.max_visual_tokens_per_gen_image

    if self.kargs.get("force_assistant_image_size", None) is not None:
      data_gen_conf["force_assistant_image_size"] = self.kargs["force_assistant_image_size"]
    data_gen_conf["resolution_finder"] = self.reso_finder
    data_gen_conf["aspect_ratio_threshold"] = self.kargs.get("aspect_ratio_threshold")

    msg_key = "message" if "message" in sample["json"] else "messages"
    messages = sample["json"][msg_key]

    if messages is None or not isinstance(messages, list):
      raise ValueError(f"Invalid messages format: messages is None or not a list, got {type(messages)}")

    for turn in messages:
      if not isinstance(turn, dict):
        raise ValueError(f"Invalid turn format: expected dict, got {type(turn)}, value={str(turn)[:100]}")
      try:
        content = turn["content"]
        if isinstance(content, str):
          continue

        content = turn["content"]
        for block in content:
          if block["type"] == "image_gen":
            block["type"] = "image"

          if block["type"] == "image":
            self._fill_image_block(block, sample, 
                                    conf=data_gen_conf if (turn["role"] == "assistant" and 'chart' not in sample['json']['source'].lower()) else data_conf)

          elif block["type"] == "video":
            self._fill_video_block(block, sample,
                                    conf=data_gen_conf if (turn["role"] == "assistant" and 'chart' not in sample['json']['source'].lower()) else data_conf)

          elif block["type"] == "text":
            continue
          else:
            raise ValueError(f"sample process error, unsupport value type: {block['type']}")
      except Exception as e:
        if np.random.rand() < 0.01: print(f"sample process error, messages={str(messages)[:50]}\n, sample=\n{str(sample)[:50]}")
        raise e

    text = self.processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=False
    )

    text += self.kargs.get("endoftext", "<|endoftext|>")

    time0 = time.time()
    image_inputs, video_inputs = self.process_vision_info(messages, **self.process_vision_info_args)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )
  
    inputs = self._convert_pixels_types(inputs)

    if time.time() - time0 > 10:
      print(f"long process time source={sample['json']['source']}, it consumes {time.time() - time0} secs", )

    if inputs["input_ids"].shape[-1] > self.max_length:
      return inputs
      
    inputs["loss_mask"] = get_assistant_mask(
      inputs["input_ids"],
      start_pattern=self.kargs.get("start_pattern", [151644, 77091, 198]),
      end_pattern=self.kargs.get("end_pattern", [151645, 198]),
    )

    inputs["loss_mask"][-1][-1] = 0

    if self.kargs.get("no_loss_mask", False) == True:
        inputs["loss_mask"][...] = 1

    if self.kargs.get("enable_vision_loss", False) == False:
      inputs["loss_mask"][
          ((inputs["input_ids"] == self.vision_start_token_id) | 
          (inputs["input_ids"] == self.vision_end_token_id) |
          (inputs["input_ids"] == self.image_token_id) |
          (inputs["input_ids"] == self.video_token_id))
        ] = 0

    if inputs["loss_mask"].sum() == 0:
      inputs["loss_mask"] = get_assistant_mask(
        inputs["input_ids"],
        start_pattern=[151644, 77091],
        end_pattern=[151645, 198]
      )
      if inputs["loss_mask"].sum() == 0:
        raise ValueError(
          f"Unable to generate sample with 0 loss_mask."
        )

    inputs["position_ids"] = self.get_rope_index_fn(
        input_ids = inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw", None),
        video_grid_thw=inputs.get("video_grid_thw", None),
        fast_video_grid_thw=inputs.get("fast_video_grid_thw", None),
        image_token_id=self.image_token_id,
        video_token_id=self.video_token_id,
        fast_video_token_id=self.fast_video_token_id,
        spatial_merge_size=self.spatial_merge_size,
        vision_start_token_id=self.vision_start_token_id,
    )

    inputs.pop("attention_mask")
    return inputs