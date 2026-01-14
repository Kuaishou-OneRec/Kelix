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


logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
  raise TimeoutError("Process excel 60 secs")

_DATASET_SKIP_MM = os.environ.get("_DATASET_SKIP_MM", "")
assert _DATASET_SKIP_MM in ["", "SKIP_MM", "SKIP_VI"]

from .tokenizer_dataset_video import \
  ChatCompletionVisionDataset_keye_vitrope_slowfast_video, \
  get_assistant_mask, \
  get_rope_index_qwen3

class ARChatCompletionVisionDataset(ChatCompletionVisionDataset_keye_vitrope_slowfast_video):
  """
  Merged dataset class for keye vitrope slowfast image.
  Directly inherits from DistributedDataset, combining functionality with slowfast-specific enhancements.
  """
  def __init__(self,
               **kwargs
               ):
      super().__init__(**kwargs)