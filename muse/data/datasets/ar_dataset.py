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
               resolution_finder_kwargs={},
               **kwargs
               ):
      super().__init__(**kwargs)
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