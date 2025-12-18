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
        if assistant_start[-len(start_pattern):] == start_pattern:
          to_mask = True
          assistant_start = []
      else:
        if _id in end_pattern:
          assistant_end.append(_id.item())
        else:
          assistant_end = []
        if assistant_end[-len(end_pattern):] == end_pattern:
          to_mask = False
          assistant_end = []
    masks.append(mask)
  return torch.tensor(masks)




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



def get_rope_index_slowfast(
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        fast_video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        fast_video_token_id: Optional[int] = None,
        spatial_merge_size: Optional[int] = None,
        vision_start_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index, fast_video_index = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]

                if image_grid_thw is not None:
                    image_nums = image_grid_thw.size(0) # 这里实际上是图片的数量
                else:
                    image_nums = 0

                if video_grid_thw is not None:
                    video_nums = video_grid_thw.size(0) # 这里实际上是slow_frame的数量
                else:
                    video_nums = 0

                if fast_video_grid_thw is not None:
                    fast_video_nums = fast_video_grid_thw.size(0) # 这里实际上是fast_frame的数量
                else:
                    fast_video_nums = 0

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos_frames, remain_fast_videos_frames = image_nums, video_nums, fast_video_nums
                # remain_images, remain_videos = image_nums, video_grid_thw.size(0)//2
                for _ in range(image_nums + video_nums + fast_video_nums):

                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_videos_frames > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    
                    if fast_video_token_id in input_tokens and remain_fast_videos_frames > 0:
                        ed_fast_video = input_tokens.index(fast_video_token_id, st)
                    else:
                        ed_fast_video = len(input_tokens) + 1
                    
                    if ed_image < min(ed_video, ed_fast_video):
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    elif ed_video < min(ed_image, ed_fast_video):
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos_frames -= 1
                        ed = ed_video
                    
                    elif ed_fast_video < min(ed_image, ed_video):
                        t, h, w = (
                            fast_video_grid_thw[fast_video_index][0],
                            fast_video_grid_thw[fast_video_index][1],
                            fast_video_grid_thw[fast_video_index][2],
                        )
                        fast_video_index += 1
                        remain_fast_videos_frames -= 1
                        ed = ed_fast_video


                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    t_index = expanded_range.flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                assert remain_fast_videos_frames == 0
                assert remain_videos_frames == 0
                assert remain_images == 0
                
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
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



class ChatCompletionVisionDataset(DistributedDataset):
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
               fast_video_token_id: int = 151678,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {},
               cut_to_pad=True,
               min_visual_tokens_per_frame: int = 4,
               max_visual_tokens_per_frame: int = 512,
               use_flops_balance=False,
               train_video: bool = True,
               process_vision_info_args: Dict[str, Any] = {},
               use_slowfast: bool = False,
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
      try:
        processor = Qwen2VLProcessor.from_pretrained(base_model_dir)
        model_config = Qwen2VLConfig.from_pretrained(base_model_dir)
        spatial_merge_size = model_config.vision_config.spatial_merge_size
        patch_size = model_config.vision_config.patch_size
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        vision_end_token_id = model_config.vision_end_token_id
        pad_token_id = model_config.pad_token_id
      except Exception as e:
        logger.warning(f"Failed to load config/processor from {base_model_dir}: {e}")

    self.train_video = train_video
    self.process_vision_info_args = process_vision_info_args
    self.use_slowfast = use_slowfast
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
    self.fast_video_token_id = fast_video_token_id
    self.vision_start_token_id = vision_start_token_id
    self.vision_end_token_id = vision_end_token_id
    self.pad_token_id = pad_token_id
    # self.patch_size = patch_size # Duplicate
    # Pad sequence to multiple of `multiple_of`
    self.multiple_of = multiple_of
    self.shuffle_size = shuffle_size
    self.shuffle_initial_size = shuffle_initial_size

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    self.img_start_token = "<|vision_start|>"
    self.img_end_token = "<|vision_end|>"
    self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
    self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    image_pad_len = 8
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0
    self.datasource_config = datasource_config

    kargs["use_flops_balance"] = self.use_flops_balance
    self.kargs = self.kwargs = kargs
    
    # Call DistributedDataset __init__
    super().__init__(
        sources=sources,
        packing=True, # Always enable packing for this dataset as per requirement
        max_length=max_length, # Passing original max_length, adjust internally if needed
        **kargs
    )
    
    # Recalculate max_length after super init to account for image pad if needed
    # But wait, DistributedDataset stores max_length.
    # We need to subtract image_pad_len from self.max_length for processing logic
    
    try:
        if self.processor:
             image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    except Exception:
        image_pad_len = 6
        
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0
    
    print("Dataset init done")

  # Removed _build_source_dataset as it is handled by DistributedDataset

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

  def _fill_video_block(self, block: Dict[str, Any],
                        sample_dict: Dict[str, Any],
                        conf: Dict[str, Any]):
    if not self.train_video:
      raise Exception("skip video")
    min_visual_tokens_per_frame = conf["min_visual_tokens_per_frame"]
    max_visual_tokens_per_frame = conf["max_visual_tokens_per_frame"]

    if "video_total_pixels" in conf:
      block["video_total_pixels"] = int(conf["video_total_pixels"])


    if "max_slow_frames" in conf:
      block["max_slow_frames"] = conf["max_slow_frames"]
    if "only_slow" in conf:
      block["only_slow"] = conf["only_slow"]

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
      
      if 'min_visual_tokens_per_fast_frame' in conf:
        block["fast_min_pixels"] = conf["min_visual_tokens_per_fast_frame"] * (self.kwargs["fast_patch_size"] ** 2) * \
            (self.spatial_merge_size ** 2)
      if 'max_visual_tokens_per_fast_frame' in conf:
        block["fast_max_pixels"] = conf["max_visual_tokens_per_fast_frame"] * (self.kwargs["fast_patch_size"] ** 2) * \
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
  
  def _convert_pixels_types(self, inputs):
    if self.kargs.get("pixel_bf16", False):
      for k, v in inputs.items():
        if 'pixel_value' not in k: continue
        inputs[k] = v.bfloat16()
    return inputs

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

    if time.time() - time0 > 10:
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
    cjx_test = True
    if cjx_test:
      inputs["loss_mask"] = 1 - get_assistant_mask(
        inputs["input_ids"],
        start_pattern=self.kargs.get("start_pattern", [151652]), 
        end_pattern=self.kargs.get("end_pattern", [151653]), #
      )
    # inputs["loss_mask"] = torch.ones_like(input_ids)
    if self.kargs.get("enable_vision_loss", False) == False:
      inputs["loss_mask"][
          (input_ids == self.vision_start_token_id) | 
          (input_ids == self.vision_end_token_id) |
          (input_ids == self.image_token_id) |
          (input_ids == self.video_token_id)
        ] = 0
    else:
      inputs["loss_mask"][:,:] = 1
    
    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
    if inputs["loss_mask"].sum() == 0:
      raise ValueError(
        f"Unable to generate sample with 0 loss_mask."
      )

    if self.use_slowfast:
      inputs["position_ids"] = get_rope_index_slowfast(
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
    else:
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
    
    data_gen_conf = copy.deepcopy(data_conf)

    if getattr(self, "min_visual_tokens_per_gen_image", -1) != -1:
      data_gen_conf["min_visual_tokens_per_gen_image"] = self.min_visual_tokens_per_gen_image
    if getattr(self, "max_visual_tokens_per_gen_image", -1) != -1:
      data_gen_conf["max_visual_tokens_per_gen_image"] = self.max_visual_tokens_per_gen_image

    msg_key = "message" if "message" in sample["json"] else "messages"
    messages = sample["json"][msg_key]

    # Validate messages format
    if messages is None or not isinstance(messages, list):
      print('maosiyangdebug', messages)
      raise ValueError(f"Invalid messages format: messages is None or not a list, got {type(messages)}")

    for turn in messages:
      # Validate turn format - each turn should be a dict with 'role' and 'content'
      if not isinstance(turn, dict):
        print('maosiyangdebugturn', turn)
        raise ValueError(f"Invalid turn format: expected dict, got {type(turn)}, value={str(turn)[:100]}")
      try:
        content = turn["content"]
        if isinstance(content, str):
          continue

        if _DATASET_SKIP_MM == "SKIP_MM": turn["content"] = [x for x in turn["content"] if x['type'] == 'text']
        content = turn["content"]
        for block in content:
          if block["type"] == "image_gen":
            block["type"] = "image"

          # if _DATASET_SKIP_MM == "SKIP_MM" and block["type"] != "text": continue
          if _DATASET_SKIP_MM == "SKIP_VI" and block["type"] == "video": continue

          if block["type"] == "image":
            # if turn["role"] == "assistant" and np.random.rand() < 0.01:
            #   print(f"data_gen_conf={data_gen_conf}")
            
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
  
    inputs = self._convert_pixels_types(inputs)


    if time.time() - time0 > 10:
      print(f"long process time source={sample['json']['source']}, it consumes {time.time() - time0} secs", )

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > self.max_length:
      # raise ValueError(f"Sample is too long. text_len={len(text)=}, token_len={inputs['input_ids'].shape[-1]}")
      return inputs
      
    inputs["loss_mask"] = get_assistant_mask(
      inputs["input_ids"],
      start_pattern=self.kargs.get("start_pattern", [151644, 77091, 198]), # [151644, 77091, 198],
      end_pattern=self.kargs.get("end_pattern", [151645, 198]), # self.end_pattern, #[151645, 198]
    )

    # mask EOS token
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

    if self.use_slowfast:
      inputs["position_ids"] = get_rope_index_slowfast(
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
    else:
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
    # inputs["position_ids"] = get_rope_index_slowfast(
    #     inputs["input_ids"],
    #     image_grid_thw=inputs.get("image_grid_thw", None),
    #     video_grid_thw=inputs.get("video_grid_thw", None),
    #     fast_video_grid_thw=inputs.get("fast_video_grid_thw", None),
    #     image_token_id=self.image_token_id,
    #     video_token_id=self.video_token_id,
    #     fast_video_token_id=self.fast_video_token_id,
    #     spatial_merge_size=self.spatial_merge_size,
    #     vision_start_token_id=self.vision_start_token_id,
    # )
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
    

    if 'video_total_pixels' in self.kargs and 'video_total_pixels' not in source_conf:
      source_conf["video_total_pixels"] = self.kargs["video_total_pixels"]
    if 'max_slow_frames' in self.kargs:
      source_conf["max_slow_frames"] = self.kargs["max_slow_frames"]
    if 'only_slow' in self.kargs:
      source_conf["only_slow"] = self.kargs["only_slow"]
    
    self._fill_image_block(pad_image, sample_dict={}, conf=source_conf)
    self._fill_video_block(pad_video, sample_dict={}, conf=source_conf)
    image_inputs, video_inputs = self.process_vision_info(vision_infos=[pad_image, pad_video] if with_vid else [pad_image])

    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs if with_vid else None,
        return_tensors="pt",
        image_video_pad=True,
    )

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
    # inputs["position_ids"] = get_rope_index_slowfast(
    #     input_ids = inputs["input_ids"],
    #     image_grid_thw=inputs.get("image_grid_thw"),
    #     video_grid_thw=inputs.get("video_grid_thw"),
    #     fast_video_grid_thw=inputs.get("fast_video_grid_thw"),
    #     image_token_id=self.image_token_id,
    #     video_token_id=self.video_token_id,
    #     fast_video_token_id=self.fast_video_token_id,
    #     spatial_merge_size=self.spatial_merge_size,
    #     vision_start_token_id=self.vision_start_token_id,
    # )
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
      "video_max_frames": self.video_max_frames,
      **{k:v for k,v in self.kwargs.items() if k in ["fast_patch_size", "min_visual_tokens_per_fast_frame", "max_visual_tokens_per_fast_frame", "min_visual_tokens_per_fast_image", "max_visual_tokens_per_fast_image"]}
    }
    if 'video_total_pixels' in self.kargs and 'video_total_pixels' not in source_conf:
      source_conf["video_total_pixels"] = self.kargs["video_total_pixels"]
    if 'max_slow_frames' in self.kargs:
      source_conf["max_slow_frames"] = self.kargs["max_slow_frames"]
    if 'only_slow' in self.kargs:
      source_conf["only_slow"] = self.kargs["only_slow"]

    if 'video_total_pixels' in self.kargs and 'video_total_pixels' not in source_conf:
      source_conf["video_total_pixels"] = self.kargs["video_total_pixels"]
    if 'max_slow_frames' in self.kargs:
      source_conf["max_slow_frames"] = self.kargs["max_slow_frames"]
    if 'only_slow' in self.kargs:
      source_conf["only_slow"] = self.kargs["only_slow"]

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

      process_max_length = self.kwargs.get("max_sample_length", 8000)
      process_max_length = min(process_max_length, self.max_length)
      # min(int(self.max_length // 1.5), ) # if self.use_flops_balance else self.max_length
      # process_max_length = self.max_length
      if inputs["input_ids"].shape[-1] > process_max_length:
        source_conf["max_visual_tokens_per_image"] = (
            source_conf["max_visual_tokens_per_image"] * self.shrink_ratio)
        source_conf["max_visual_tokens_per_frame"] = (
            source_conf["max_visual_tokens_per_frame"] * self.shrink_ratio)
        if "video_total_pixels" in source_conf:
          source_conf["video_total_pixels"] = source_conf["video_total_pixels"] * self.shrink_ratio
          # print("rank test video_total_pixels shrink {}, id_source_conf={}".format(source_conf["video_total_pixels"]//28//28, id(source_conf)))
        continue
      else:
        assert inputs["input_ids"].shape[-1] <= process_max_length, "inputs too long"
        return inputs
    else:
      raise ValueError(
          f"Unable to generate sample within max_length={process_max_length} after {retry} retrys, \nlength={inputs['input_ids'].shape}\n" + "" if np.random.rand() < 0.99 else f"messages={str(sample)[:1000]}"
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

  def process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Process a single sample from Parquet.
    Wraps the sample to match the expected format of _process.
    """
    source_name = sample.get("source", "None")
    # Wrap sample to match existing logic which expects {"json": sample}
    # If the parquet schema matches the json schema, this should work.
    wrapper = {"json": sample}
    
    # Update epoch_idx if available from DistributedDataset
    # but DistributedDataset doesn't inject epoch_idx into sample automatically unless we do it in _parser?
    # Actually DistributedDataset logic handles epochs by iterating num_epochs. 
    # The sample itself comes from parquet.
    # We can inject a dummy epoch_idx if needed by _process
    wrapper["epoch_idx"] = 0 # Default, or fetch from self if we track it
    
    return self._process(wrapper, source_name)

  def pack_sample(self, buffer: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
      return self._packing(buffer)

  def get_sample_length(self, sample: Dict[str, torch.Tensor]) -> int:
      return sample["input_ids"].shape[-1]







class ChatCompletionVisionDataset_keye_vitrope_slowfast(ChatCompletionVisionDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               max_length: int = 1024,
               min_visual_tokens_per_image: int = 4,
               max_visual_tokens_per_image: int = 512,
               min_visual_tokens_per_gen_image: int = -1,
               max_visual_tokens_per_gen_image: int = -1,
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
               fast_video_token_id: int = 151678,
               vision_start_token_id: int = 151652,
               vision_end_token_id: int = 151653,
               pad_token_id: int = 151643,
               datasource_config:Dict[str, Dict[str, Any]] = {},
               cut_to_pad=True,
               process_vision_info_args={"image_factor":28},
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
        from muse.models.keye_tokenizer.configuration_keye import KeyeConfig
        
        processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
        model_config = KeyeConfig.from_pretrained(base_model_dir)
        
        spatial_merge_size = model_config.vision_config.spatial_merge_size
        patch_size = model_config.vision_config.patch_size
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        fast_video_token_id = model_config.fast_video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        vision_end_token_id = model_config.vision_end_token_id
        pad_token_id = model_config.pad_token_id
        
        logger.info(f"Loaded config from {base_model_dir}: spatial_merge_size={spatial_merge_size}, patch_size={patch_size}")
      except ImportError:
         logger.warning("KeyeConfig not found in muse.models.keye_tokenizer.configuration_keye, skipping config loading from base_model_dir.")
      except Exception as e:
         logger.warning(f"Failed to load KeyeConfig/Processor from {base_model_dir}: {e}. Using default args.")

    kwargs['use_flops_balance'] = kwargs.get("use_flops_balance", False)
    self.use_flops_balance = kwargs['use_flops_balance']
    self.slowfast_padder = SlowFastVisionPadder(base_model_dir)
    self.auto_aug = AutoAugmentWrapper(policy=kwargs.get("autoaug_policy", None))
    self.process_vision_info_args = process_vision_info_args
    self.cut_to_pad = cut_to_pad
    print(f"set cut_to_pad={cut_to_pad}")
    self.processor = processor
    # self.process_vision_info = process_vision_info_keye_vitrope_slowfast

    self.min_visual_tokens_per_image = min_visual_tokens_per_image
    self.max_visual_tokens_per_image = max_visual_tokens_per_image
    self.min_visual_tokens_per_frame = min_visual_tokens_per_frame
    self.max_visual_tokens_per_frame = max_visual_tokens_per_frame

    self.min_visual_tokens_per_gen_image = min_visual_tokens_per_gen_image
    self.max_visual_tokens_per_gen_image = max_visual_tokens_per_gen_image

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
    
    # Initialize base class
    super().__init__(
        sources=sources,
        max_length=max_length,
        processor=processor,
        use_flops_balance=self.use_flops_balance,
        use_slowfast=True, # Enable slowfast logic in base class if needed, or handle here
        **kwargs
    )
    
    # self.sources = sources # Handled by super

    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

    if base_model_dir:
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
        self.img_start_token = "<|vision_start|>"
        self.img_end_token = "<|vision_end|>"
        self.img_start_token_id = self.tokenizer.encode(self.img_start_token)[0]
        self.img_end_token_id = self.tokenizer.encode(self.img_end_token)[0]
    # self.img_context_token_id = self.tokenizer.encode(self.img_context_token)[0]

    # append image_pad for each packing
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1]
    # image_pad_len = self._gen_img_pad()["input_ids"].shape[-1] # 6
    image_pad_len = 8
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    self.kargs = self.kwargs = kwargs
    
  def _cut_sample(self, inputs, packable_length):
    return self._cut_sample_cjx(inputs, packable_length)

  def _cut_sample_cjx(self, inputs, packable_length):
    # Same logic as provided
    # ... (implementation from provided snippet)
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

    for _, inputs in enumerate(buffer):
      image_pad = True if self.use_flops_balance else False
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



    # 1. 保证image, slow_video, fast_video的 vit token数量都是8的倍数。
    # 2. 保证每张卡都会经过VIT
    if self.slowfast_padder:
        paddings = self.slowfast_padder( 
          packed_pixel_values,
          packed_pixel_values_videos,
          packed_fast_pixel_values_videos
        )
        for pad in paddings:
          self._append_sample_packing(pad,
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
      "epoch_idx": torch.tensor([sum(epochs) / len(epochs)], dtype=torch.float32) if epochs else torch.tensor([0.0]),
      "fast_pixel_values_videos": packed_fast_pixel_values_videos,
      "fast_video_grid_thw": packed_fast_video_grid_thw,
    }
    inputs = self._convert_pixels_types(inputs)
    return inputs




class SlowFastVisionPadder:
    """
    给slow fast的padding，最多使用4+6个token
    """
    def __init__(self, model_dir):
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = processor
        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        assert self.merge_size == 2, f"SlowFastVisionPadder does not support self.merge_size({self.merge_size}) != 2"
        self.image_pad = processor.tokenizer.encode("<|image_pad|>")[0]
        self.video_pad = processor.tokenizer.encode("<|video_pad|>")[0]
        fast_video_pad = processor.tokenizer.encode("<|fast_video_pad|>")
        assert len(fast_video_pad) == 1, "Decode fast_video_pad failed: {}".format(fast_video_pad)
        self.fast_video_pad = fast_video_pad[0]
        self.vision_start = processor.tokenizer.encode("<|vision_start|>")[0]
        self.vision_end = processor.tokenizer.encode("<|vision_end|>")[0]
        self.frame = processor.tokenizer.encode("<|frame|>")[0]

    def __call__(self, packed_pixel_values, packed_pixel_values_videos, packed_fast_pixel_values_videos):
          return [
            self.gen_img_pad(n_merged_slow_tokens=1),
            self.gen_video_pad(n_merged_slow_tokens=1),
          ]
          paddings = []
          n_pixel_values = sum([x.shape[0] for x in packed_pixel_values], 0)
          n_pixel_values_videos = sum([x.shape[0] for x in packed_pixel_values_videos], 0)
          n_fast_pixel_values_videos = sum([x.shape[0] for x in packed_fast_pixel_values_videos], 0)

          if n_pixel_values % 8 == 4: paddings.append(self.gen_img_pad(n_merged_slow_tokens=1))
          elif n_pixel_values == 0: paddings.append(self.gen_img_pad(n_merged_slow_tokens=2))

          paddings.append(
            self.gen_video_pad(
              n_merged_slow_tokens=1 if n_pixel_values_videos % 8 == 4 else 2, 
              n_merged_fast_tokens=1 if n_fast_pixel_values_videos % 8 == 4 else 2, 
              )
          )

          return paddings

    def gen_img_pad(self, n_merged_slow_tokens=1):
        input_ids = [self.vision_start] + [self.image_pad] * n_merged_slow_tokens + [self.vision_end]
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64),
            "attention_mask": torch.tensor([[1] * (n_merged_slow_tokens + 2)], dtype=torch.int64),
            "pixel_values": torch.rand(n_merged_slow_tokens * 4, 3, self.patch_size, self.patch_size).float(),
            "image_grid_thw": torch.tensor([[1, 2, n_merged_slow_tokens * 2]], dtype=torch.int64),
            "loss_mask": torch.zeros(len(input_ids), dtype=torch.int64),
        }
        inputs["position_ids"] = get_rope_index(
          inputs["input_ids"],
          image_grid_thw=inputs.get("image_grid_thw"),
          video_grid_thw=inputs.get("video_grid_thw"),
          spatial_merge_size=self.merge_size,
          image_token_id=self.image_pad,
          video_token_id=self.video_pad,
          vision_start_token_id=self.vision_start
        )

        return inputs

    def gen_video_pad(self, n_merged_slow_tokens=1, n_merged_fast_tokens=2):
        """
        demo: 
        'input_ids':
        Tensor: shape=(1, 42), dtype=torch.int64, device=cpu, data=tensor([151652,     27,     91,   6763])...tensor([  6213,     91,     29, 151653])
        'attention_mask':
        Tensor: shape=(1, 42), dtype=torch.int64, device=cpu, data=tensor([1, 1, 1, 1])...tensor([1, 1, 1, 1])
        'pixel_values_videos':
        Tensor: shape=(16, 3, 14, 14), dtype=torch.float32, device=cpu, data=tensor([-1., -1., -1., -1.])...tensor([-1., -1., -1., -1.])
        'video_grid_thw':
        Tensor: shape=(1, 3), dtype=torch.int64, device=cpu, data=tensor([1, 4, 4])...tensor([1, 4, 4])
        'fast_pixel_values_videos':
        Tensor: shape=(32, 3, 14, 14), dtype=torch.float32, device=cpu, data=tensor([-1., -1., -1., -1.])...tensor([-1., -1., -1., -1.])
        'fast_video_grid_thw':
        Tensor: shape=(1, 3), dtype=torch.int64, device=cpu, data=tensor([1, 8, 4])...tensor([1, 8, 4])
        """
        # 标准是这个
        # # <|frame|>ts<|placeholder|><|placeholder|> ... n_slow ... <|placeholder|><|fast_start|><|fast_placeholder|><|fast_placeholder|> ... n_fast ... <|fast_placeholder|><|fast_end|>
        # 但是我们不需要那么多token
        #   只需要<|placeholder|><|placeholder|> ... n_slow ... <|placeholder|><|fast_placeholder|><|fast_placeholder|> ... n_fast ... <|fast_placeholder|>
        # total_slow_tokens = pass # 
        # video_inputs = (
        input_ids = [self.vision_start] + [self.video_pad] * n_merged_slow_tokens + [self.fast_video_pad] * n_merged_fast_tokens + [self.vision_end]
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64),
            "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.int64),
            "video_grid_thw": torch.tensor([[1, 2, n_merged_slow_tokens * 2]], dtype=torch.int64),
            "fast_video_grid_thw": torch.tensor([[1, 2, n_merged_fast_tokens * 2]], dtype=torch.int64),
            "fast_pixel_values_videos": torch.rand(n_merged_fast_tokens * 4, 3, self.patch_size, self.patch_size).float(),
            "pixel_values_videos": torch.rand(n_merged_slow_tokens * 4, 3, self.patch_size, self.patch_size).float(),
            "loss_mask": torch.zeros(len(input_ids), dtype=torch.int64),
        }
        inputs["position_ids"] = get_rope_index(
          inputs["input_ids"],
          image_grid_thw=inputs.get("image_grid_thw"),
          video_grid_thw=inputs.get("video_grid_thw"),
          spatial_merge_size=self.merge_size,
          image_token_id=self.image_pad,
          video_token_id=self.video_pad,
          vision_start_token_id=self.vision_start
        )

        return inputs