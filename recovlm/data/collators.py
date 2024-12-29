# TODO: support multi images, videos
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

import torch
import torch.nn.functional as F

from .templates import get_template
from qwen_vl_utils import process_vision_info


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

class ImageTextPackingCollator:
  """Image Text Collator with Packing"""

  def __init__(self,
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
               multiple_of: int = 8):
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

  def _process_single(self,
                      sample: Dict[str, Union[str, Image.Image]],
                      max_visual_tokens: 1.0) -> Dict[str, torch.Tensor]:
    max_visual_tokens = max(max_visual_tokens, self.min_visual_tokens)
    image = sample[".jpg"]
    caption = sample[".txt"]
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

  def _packing(self, all_inputs):

    packed_input_ids = []
    packed_position_ids = []
    packed_loss_mask = []
    packed_pixel_values = []
    packed_image_gird_thw = []
    cu_seqlens = [0]

    for inputs in all_inputs:
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

    # if (
    #     self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
    # ):  # not divisible by multiple_of; here we align for grouping
    #     padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
    #     packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.processor.tokenizer.pad_token_id)

    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "pixel_values": packed_pixel_values,
      "image_grid_thw": packed_image_gird_thw,
      "cu_seqlens": torch.tensor(cu_seqlens)
    }
    return inputs

  def __call__(self, samples):
    all_inputs = []
    for sample in samples:
      max_visual_tokens = self.max_visual_tokens
      failed = True
      try:
        for _ in range(self.max_retry):
          _inputs = self._process_single(
            sample, max_visual_tokens=max_visual_tokens)
          if _inputs["input_ids"].shape[-1] <= self.max_length:
            failed = False
            break
          max_visual_tokens = (max_visual_tokens * 0.9)
      except:
        failed = True
      if failed:
        continue
      all_inputs.append(_inputs)
    # TODO: fix empty all_inputs
    inputs = self._packing(all_inputs)
    return inputs
