from typing import Union, Iterable, Optional, List, Dict, Tuple, Any
import random
import json
import torch
import traceback
import pyarrow.parquet as pq
import numpy as np
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image

import torch.nn.functional as F

import multiprocessing
import torch.distributed as dist

from torch.utils.data import IterableDataset

from recovlm.utils.common import get_worker_info
from recovlm.utils.logger import init_logger
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from recovlm.utils.qwen_vl_utils import process_vision_info

from recovlm.data.prompts import PromptLoader

from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from recovlm.utils.common import shell_hdfs_ls, load_parquet_file

from tqdm import tqdm

logger = init_logger(__name__)

DEFAULT_SYSTEM_PROMPT = \
"""You are a helpful assistant."""

COT_SYSTEM_PROMPT = \
"""You are a AI assistant expert in reasoning. 
Before answering any question, 
you should first think step-by-step, 
then provide your conclusion based on your reasoning process. 
Your output should follow this format:

First, include your thinking process within <think></think> tags.
Then, provide your final answer within <answer></answer> tags.

For example:
<think>
Let me analyze this question carefully...
</think>

<answer>
[Your final, concise answer here]
</answer>"""

class Qwen2VLInputBuilder:
  def __init__(self,
               pretrained_model_name_or_path: Optional[str] = None,
               **kwargs):
    self.processor = \
        AutoProcessor.from_pretrained(pretrained_model_name_or_path)
    self.model_config = \
        AutoConfig.from_pretrained(pretrained_model_name_or_path)
    self.spatial_merge_size = \
        self.model_config.vision_config.spatial_merge_size
    self.patch_size = self.model_config.vision_config.patch_size
    self.image_token_id = self.model_config.image_token_id
    self.video_token_id = self.model_config.video_token_id
    self.vision_start_token_id = self.model_config.vision_start_token_id
    self.vision_end_token_id = self.model_config.vision_end_token_id
    self.pad_token_id = self.model_config.pad_token_id
    self.video_nframe = kwargs.get("video_nframe", -1)
    self.video_fps = kwargs.get("video_fps", 2.0)
    self.video_min_frames = kwargs.get("video_min_frames", -1)
    self.video_max_frames = kwargs.get("video_max_frames", 120)
    self.min_visual_tokens_per_image = \
        kwargs.get("min_visual_tokens_per_image", 4)
    self.max_visual_tokens_per_image = \
        kwargs.get("max_visual_tokens_per_image", 512)
    self.max_images = kwargs.get("max_images", 10)

  def fill_image_block(self,
                       block: Dict[str, Any],
                       sample: Optional[Dict[str, Any]] = None,
                       **kwargs):
    min_visual_tokens_per_image = \
        kwargs.get(
          "min_visual_tokens_per_image", self.min_visual_tokens_per_image)
    max_visual_tokens_per_image = \
        kwargs.get(
          "max_visual_tokens_per_image", self.max_visual_tokens_per_image)

    if isinstance(block["image"], str):
      if sample is None:
        raise ValueError(
          "raw sample dict is required to decode image, "
          "but got None")
      image = sample[block["image"]]
    else:
      image = block["image"]
    if image.mode != "RGB":
      image = image.convert("RGB")

    block["image"] = image
    block["min_pixels"] = \
        min_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)
    block["max_pixels"] = \
        max_visual_tokens_per_image * (self.patch_size ** 2) * \
        (self.spatial_merge_size ** 2)

    return block

  def fill_video_block(self,
                       block: Dict[str, Any],
                       sample: Dict[str, Any],
                       **kwargs):
    min_visual_tokens_per_image = \
        kwargs.get(
          "min_visual_tokens_per_image", self.min_visual_tokens_per_image)
    max_visual_tokens_per_image = \
        kwargs.get(
          "max_visual_tokens_per_image", self.max_visual_tokens_per_image)

    if isinstance(block["video"], list):
      for image_block in block["video"]:
        assert image_block["type"] == "image" and "image" in image_block
        self.fill_image_block(image_block, sample, **kwargs)

    elif isinstance(block["video"], str) or isinstance(block["video"], bytes):
      # video in local tar, replace by video bytes
      if isinstance(block["video"], str) and block["video"] in sample:
        block["video"] = sample[block["video"]]
      # fill other params
      block["min_pixels"] = \
        min_visual_tokens_per_image * (self.patch_size ** 2) * \
          (self.spatial_merge_size ** 2)
      block["max_pixels"] = \
        max_visual_tokens_per_image * (self.patch_size ** 2) * \
          (self.spatial_merge_size ** 2)
      # video split params
      if kwargs.get("video_nframe", self.video_nframe) > 0:
        block["nframes"] = kwargs.get("video_nframe", self.video_nframe)
      if kwargs.get("video_nframe", self.video_fps) > 0:
        block["fps"] = kwargs.get("video_nframe", self.video_fps)
      if kwargs.get("video_min_frames", self.video_min_frames) > 0:
        block["min_frames"] = kwargs.get("video_min_frames", self.video_min_frames)
      if kwargs.get("video_max_frames", self.video_max_frames) > 0:
        block["max_frames"] = kwargs.get("video_max_frames", self.video_max_frames)
    else:
      raise ValueError(
        f"Unsupport video type. {type(block['video'])=}")

    return block

  def get_rope_index(
          self,
          input_ids: torch.LongTensor,
          image_grid_thw: Optional[torch.LongTensor] = None,
          video_grid_thw: Optional[torch.LongTensor] = None,
          attention_mask: Optional[torch.Tensor] = None,
          spatial_merge_size: Optional[int] = None,
          image_token_id: Optional[int] = None,
          video_token_id: Optional[int] = None,
          vision_start_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

  def tokenize_messages(self,
                        messages: List[Dict[str, Any]],
                        **kwargs):
    text = self.processor.apply_chat_template(
      messages, tokenize=False,
      add_generation_prompt=True
    )
    try:
      image_inputs, video_inputs = process_vision_info(messages)
    except Exception as e:
      import traceback
      traceback.print_exc()
      raise ValueError(f"Failed to parse vision info: {e}")
    
    if image_inputs:
      image_inputs = image_inputs[:self.max_images]
    if video_inputs:
      video_inputs = video_inputs[:self.max_images]
    inputs = self.processor(
      text=text,
      images=image_inputs,
      videos=video_inputs,
      return_tensors="pt"
    )
    inputs["position_ids"] = self.get_rope_index(
      inputs["input_ids"],
      image_grid_thw=inputs.get("image_grid_thw"),
      video_grid_thw=inputs.get("video_grid_thw"),
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )
    return inputs, text, image_inputs, video_inputs

  def gen_img_pad(self):
    """Append an image placeholder, to trigger vit for pure text sample
       return 6 tokens: vstart, 4 * image_token, vend
    """
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    pad_image = {
      "type": "image",
      "image": Image.new("RGB", (1, 1), (255, 255, 255))
    }

    self.fill_image_block(pad_image)
    image_inputs, _ = process_vision_info(vision_infos=[pad_image])
    inputs = self.processor(
      text=text,
      images=image_inputs,
      videos=None,
      return_tensors="pt"
    )

    inputs["output_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["position_ids"] = self.get_rope_index(
      inputs["input_ids"],
      image_grid_thw=inputs.get("image_grid_thw"),
      video_grid_thw=inputs.get("video_grid_thw"),
      spatial_merge_size=self.spatial_merge_size,
      image_token_id=self.image_token_id,
      video_token_id=self.video_token_id,
      vision_start_token_id=self.vision_start_token_id
    )

    return inputs

class ParquetDataset(IterableDataset):
  def __init__(self, files, num_workers):
    self.files = files
    self.num_workers = num_workers

    manager = multiprocessing.Manager()

    self.finish_dict_all = manager.dict()
    self.offset_dict_all = manager.dict()
    for i in range(self.num_workers):
      self.finish_dict_all[i] = manager.dict()
      self.offset_dict_all[i] = manager.dict()

  def state_dict(self,):
    worker, num_workers = get_worker_info()

    state_dict = {
      "finish_dict": dict(self.finish_dict_all[worker]),
      "offset_dict": dict(self.offset_dict_all[worker])
    }
    return state_dict
  
  def load_state_dict(self, state_dict):
    worker, num_workers = get_worker_info()
    finish_dict = state_dict["finish_dict"]
    offset_dict = state_dict["offset_dict"]

    # support old ckpt format
    tmp_finish_dict = dict()
    tmp_offset_dict = dict()

    for k, v in finish_dict.items():
      if isinstance(k, str):
        tmp_finish_dict[(k, 0)] = v
      elif isinstance(k, tuple) and len(k) == 2:
        tmp_finish_dict[k] = v
      else:
        raise NotImplementedError(
          f"Unsupported dataloader checkpoint format.") 
    
    for k, v in offset_dict.items():
      if isinstance(k, str):
        fn, group_idx = k.split("|")
        group_idx = int(group_idx)
        tmp_offset_dict[(fn, 0, group_idx)] = v
      elif isinstance(k, tuple) and len(k) == 3:
        tmp_offset_dict[k] = v
      else:
        raise NotImplementedError(
          f"Unsupported dataloader checkpoint format.") 

    # clear cur state
    self.finish_dict_all[worker].clear()
    self.offset_dict_all[worker].clear()

    # update
    self.finish_dict_all[worker].update(tmp_finish_dict)
    self.offset_dict_all[worker].update(tmp_offset_dict)
    logger.warning(f"[rank{rank}-woker{worker}] load checkpoint success.")

  def _parser(self, row, file_url):
    try:
      messages = None
      segments = None
      chosen = None
      rejected = None

      if "messages" in row:
        messages = row["messages"]
        if isinstance(messages, str):
          messages = json.loads(messages)
          
      if "segments" in row:
        segments = row["segments"]
        if isinstance(segments, str):
          segments = json.loads(segments)
          
      images = row["images"]
      data_source = row["source"]
      key = row["uuid"]

      samples = {
        "__key__": key,
        "__url__": file_url,
      }

      # process message or segments -> webdataset_key = json
      sample_data = {"source": data_source}

      if "chosen" in row:
        chosen = row["chosen"]
        if isinstance(chosen, str):
          chosen = json.loads(chosen)
        sample_data["chosen"] = chosen

      if "rejected" in row:
        rejected = row["rejected"]
        if isinstance(rejected, str):
          rejected = json.loads(rejected)
        sample_data["rejected"] = rejected

      if messages is not None and isinstance(messages, list):
        sample_data["messages"] = messages
      elif segments is not None and isinstance(segments, list):
        sample_data["segments"] = segments
      elif messages is not None and isinstance(messages, np.ndarray):
        sample_data["messages"] = messages.tolist()
      else:
        raise NotImplementedError(
          f"Unsupported sample, message type is {type(messages)}, "
          f"message={messages}, segments type is {type(segments)}, "
          f"segments={segments}")
      samples["json"] = sample_data

      # process images
      if isinstance(images, str):
        images = json.loads(images)
      elif isinstance(images, dict):
        pass
      else:
        raise NotImplementedError(
          f"Unsupported image field type, {type(row['images'])=}")

      for image_name in images:
        image_b64 = images[image_name]
        image_bytes = base64.b64decode(image_b64)
        image_bytes_stream = BytesIO(image_bytes)
        image = Image.open(image_bytes_stream)
        samples[image_name] = image
      return samples
    except:
      logger.error(
        f"ParquetDataset parse sample error!!! "
        f"err_msg={traceback.format_exc()}")
      return None

  def __iter__(self,):
    worker, num_workers = get_worker_info()
    assert num_workers == self.num_workers, "Number of workers mismatch"

    finish_dict = self.finish_dict_all[worker]
    offset_dict = self.offset_dict_all[worker]

    worker_files = [
      fn for idx, fn in enumerate(self.files) \
        if idx % num_workers == worker]
    logger.warning(
      f"ParquetDataset Info: {worker=}, {num_workers=}, "
      f"num_files={len(self.files)}, worker_files={len(worker_files)}"
    )

    # Add a progress bar, dd a progress bar
    for epoch_fn in tqdm(worker_files, desc=f"[Worker-{worker}] processing: "):
      fn, epoch_idx = epoch_fn
      if (fn, epoch_idx) in finish_dict:
        logger.warning(f"[Worker-{worker}] {fn} has been processed, skip.")
        continue
      logger.info(f"[Worker-{worker}] processing {fn}-epoch{epoch_idx}")
      # open parquet file
      try:
        parquet_file = load_parquet_file(fn)
      except Exception as e:
        logger.error(
          f"ParquetDataset error, open parquet fail!!! "
          f"{fn=}, error_msg={traceback.format_exc()}")
        continue

      for group_idx in range(parquet_file.num_row_groups):
        offset = 0
        fn_group_key = (fn, epoch_idx, group_idx)
        if fn_group_key in offset_dict:
          if offset_dict[fn_group_key] == -1:
            continue
          else:
            offset = offset_dict[fn_group_key] + 1
        
        row_group = parquet_file.read_row_group(group_idx)
        if offset >= row_group.num_rows:
          continue
        logger.warning(
          f"[Worker-{worker}] start "
          f"{fn}-epoch{epoch_idx}-group{group_idx}-offset{offset}")
        row_pandas = row_group.to_pandas().reset_index().iloc[offset:]

        for row_idx, row in row_pandas.iterrows():
          if row_idx < offset:
            continue
          try:
            sample = self._parser(row, fn)
            if sample is not None:
              yield sample
            offset_dict[fn_group_key] = row_idx
          except Exception as e:
            logger.error(
              f"Error processing row {row_idx}: "
              f"{str(e)}")
            continue
          except GeneratorExit:
            raise


class DistributedDataset(IterableDataset):
  def __init__(self, 
               sources: str,
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               num_epochs: int=1):
    self.rng = random.Random(seed)
    self.rank = rank
    self.world_size = world_size
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    self.sources = sources
    self.dataset = self._build()
    # for data_source monitor
    self.source_sample_cnt = {}
    self.source_error_cnt = {}

  def _build(self):
    file_list = []
    files = []
    # TODO: support more file types
    if self.sources.endswith(".json"):
      with open(self.sources, "r") as fp:
        files = json.loads(fp.read())
        files = sorted([
          fn for fn in files if fn.endswith(".parquet")])
    else:
      folder = Path(self.sources)
      files = list(map(str, folder.rglob("*.parquet")))

    self.rng.shuffle(files)
    total_files = len(files)
    num_files_per_rank = round(len(files) / self.world_size)
    files = files[
      self.rank * num_files_per_rank: (self.rank + 1) * num_files_per_rank]

    assert len(files) > 0, f"No file found for rank{self.rank}"

    # repeat
    for i in range(self.num_epochs):
      files.sort()
      self.rng.shuffle(files)
      file_list += [(fn, i) for fn in files]

    logger.info(
      f"DistributedDataset "
      f"rank={self.rank}, world_size={self.world_size} orig_file_num={total_files} "
      f"file_num={len(file_list)}")

    # TODO: support more file format
    dataset = ParquetDataset(file_list, self.num_workers)
    return dataset

  def state_dict(self):
    return self.dataset.state_dict()

  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)

  def __iter__(self):
    return self.dataset.__iter__()




class VisionPromptDataset(DistributedDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               num_epochs: int=1,
               max_length: int = 1024,
               datasource_config: Dict[str, Dict[str, Any]] = {},
               system_prompt: str = COT_SYSTEM_PROMPT,
               pretrained_model_name_or_path: Optional[str] = None,
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed,
      num_epochs=num_epochs)
    self.shrink_ratio = kwargs.get("shrink_ratio", 0.9)
    self.max_retry = kwargs.get("max_retry", 5)
    self.min_visual_tokens_per_image = \
      kwargs.get("min_visual_tokens_per_image", 4)
    self.max_visual_tokens_per_image = \
      kwargs.get("max_visual_tokens_per_image", 512)
    self.max_images = kwargs.get("max_images", 10)
    self.system_prompt = system_prompt
    self.max_length = max_length

    self.input_builder = Qwen2VLInputBuilder(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      **kwargs
    )

    self.datasource_config = datasource_config
    logger.info(
      f"VisionPromptDataset datasource_config: "
      f"{json.dumps(self.datasource_config, indent=2)}"
    )

  def _process_chat(self,
                    sample: Dict[str, Any],
                    **kwargs) -> Dict[str, torch.Tensor]:
    assert "messages" in sample["json"], \
        f"sample must contain 'messages' key, but got {sample['json'].keys()}"

    messages = sample["json"]["messages"]
    prompt_messages = []
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue
      for block in content:
        if block["type"] == "image":
          self.input_builder.fill_image_block(block, sample, **kwargs)
        elif block["type"] == "video":
          self.input_builder.fill_video_block(block, sample, **kwargs)
        elif block["type"] == "text":
          continue
        else:
          raise ValueError(
            f"sample process error, unsupport value type: {block['type']}")

    if messages[0]["role"] != "system":
      system = {
        "role": "system", "content": self.system_prompt}
      messages.insert(0, system)

    for turn in messages:
      if turn["role"] == "assistant":
        break
      prompt_messages.append(turn)
    
    annotation = messages[-1]["content"]
    if isinstance(annotation, list):
      annotation = annotation[-1]["text"]

    inputs, text, image_inputs, video_inputs = \
      self.input_builder.tokenize_messages(prompt_messages)

    inputs.pop("attention_mask")
    mm_data = {}
    if image_inputs:
      mm_data["image"] = image_inputs
    if video_inputs:
      mm_data["video"] = video_inputs
    return {
      "original_inputs": inputs,
      "vllm_inputs": {
        "prompt": text,
        "multi_modal_data": mm_data,
      },
      "annotation": annotation,
      "source": sample["json"]["source"],
      "__key__": sample["__key__"],
      "__url__": sample["__url__"],
    }

  def _process(self, sample, source_name=None):

    # get data format
    if "messages" in sample["json"]:
      data_format = "chatml"
    elif "segments" in sample["json"]:
      data_format = "completion"
    else:
      raise NotImplementedError(f"Unsupported dataset format.")

    kwargs = {
      "min_visual_tokens_per_image": self.min_visual_tokens_per_image,
      "max_visual_tokens_per_image": self.max_visual_tokens_per_image,
    }
    if source_name != None and source_name in self.datasource_config:
      for key in self.datasource_config[source_name]:
        kwargs[key] = self.datasource_config[source_name][key]

    for retry in range(self.max_retry):
      if data_format == "chatml":
        inputs = self._process_chat(sample, **kwargs)
      else:
        raise NotImplementedError(
          f"Unsupported dataset format `{data_format}`")

      if not inputs:
        raise ValueError("Empty inputs, skip")
      if inputs["original_inputs"]["input_ids"].shape[-1] > self.max_length:
        kwargs["max_visual_tokens_per_image"] = (
          kwargs["max_visual_tokens_per_image"] * self.shrink_ratio)
        continue
      else:
        assert inputs["original_inputs"]["input_ids"].shape[-1] \
            <= self.max_length, "inputs too long"
        return inputs
    else:
      raise ValueError(
        f"Unable to generate sample within "
        f"max_length={self.max_length} after {retry} retrys"
      )

  def __iter__(self):

    for sample in self.dataset:
      sample_key = sample["__key__"] if "__key__" in sample else ""
      sample_url = sample["__url__"] if "__url__" in sample else ""

      try:
        source_name = sample["json"]["source"]
      except:
        source_name = "None"

      self.source_sample_cnt.setdefault(source_name, 0)
      self.source_sample_cnt[source_name] += 1

      try:
        inputs = self._process(sample, source_name)
        yield inputs
      except:
        self.source_error_cnt.setdefault(source_name, 0)
        self.source_error_cnt[source_name] += 1
        error_ratio = self.source_error_cnt[source_name] * 1.0 / \
          self.source_sample_cnt[source_name]
        logger.error(
          f"PromptVisionDataset process sample error. "
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, "
          f"errmsg={traceback.format_exc()}")
        continue

class ExperienceDataset(IterableDataset):
    """Dataset for flattened RL experiences"""
    def __init__(self,
                 generated: List[Dict],
                 max_length: int,
                 multiple_of: Optional[int] = 8,
                 use_packing: Optional[bool] = True,
                 num_packing_samples: int = -1,
                 pretrained_model_name_or_path: Optional[str] = None,
                 **kwargs):
      self.experiences = generated
      self.multiple_of = multiple_of
      self.use_packing = use_packing
      self.num_packing_samples = num_packing_samples
      self.max_length = max_length
      self.input_builder = Qwen2VLInputBuilder(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        **kwargs
      )

    def _append_sample_packing(self,
                               inputs: Dict[str, torch.Tensor],
                               packed_input_ids: List[torch.Tensor],
                               packed_position_ids: List[torch.Tensor],
                               packed_output_mask: List[torch.Tensor],
                               packed_pixel_values: List[torch.Tensor],
                               packed_pixel_values_videos: List[torch.Tensor],
                               packed_image_gird_thw: List[torch.Tensor],
                               packed_video_grid_thw: List[torch.Tensor],
                               packed_advantages: List[torch.Tensor],
                               packed_log_probs_ref: List[torch.Tensor],
                               packed_old_log_probs: List[torch.Tensor],
                               packed_sample_idx: List[int],
                               cu_seqlens: List[int]):

      packed_input_ids.append(inputs["input_ids"])
      packed_output_mask.append(inputs["output_mask"])
      packed_position_ids.append(inputs["position_ids"])

      if "pixel_values" in inputs:
        packed_pixel_values.append(inputs["pixel_values"])
        packed_image_gird_thw.append(inputs["image_grid_thw"])
      if "pixel_values_videos" in inputs:
        packed_pixel_values_videos.append(inputs["pixel_values_videos"])
        packed_video_grid_thw.append(inputs["video_grid_thw"])
      if "advantages" in inputs:
        packed_advantages.append(inputs["advantages"])
      if "log_probs_ref" in inputs:
        packed_log_probs_ref.append(inputs["log_probs_ref"])
      if "old_log_probs" in inputs:
        packed_old_log_probs.append(inputs["old_log_probs"])
      cu_seqlens.append(cu_seqlens[-1] + len(inputs["input_ids"][0]))
      packed_sample_idx.append(inputs["sample_idx"])

      return len(inputs["input_ids"][0])

    def _packing(self, buffer: List[Dict[str, torch.Tensor]]):
      packed_input_ids: List[torch.Tensor] = []
      packed_position_ids: List[torch.Tensor] = []
      packed_output_mask: List[torch.Tensor] = []
      packed_pixel_values: List[torch.Tensor] = []
      packed_pixel_values_videos: List[torch.Tensor] = []
      packed_image_gird_thw: List[torch.Tensor] = []
      packed_video_grid_thw: List[torch.Tensor] = []
      packed_advantages: List[torch.Tensor] = []
      packed_log_probs_ref: List[torch.Tensor] = []
      packed_old_log_probs: List[torch.Tensor] = []
      packed_sample_idx: List[int] = []
      cu_seqlens: List[int] = [0]

      valid_seq_len = 0
      for _, inputs in enumerate(buffer):
        valid_seq_len += self._append_sample_packing(inputs,
                                                     packed_input_ids,
                                                     packed_position_ids,
                                                     packed_output_mask,
                                                     packed_pixel_values,
                                                     packed_pixel_values_videos,
                                                     packed_image_gird_thw,
                                                     packed_video_grid_thw,
                                                     packed_advantages,
                                                     packed_log_probs_ref,
                                                     packed_old_log_probs,
                                                     packed_sample_idx,
                                                     cu_seqlens)
          # append a pad image sequence to trigger ViT
      image_pad = self.input_builder.gen_img_pad()
      image_pad["sample_idx"] = -1
      # for key in ["log_probs_ref", "old_log_probs", "advantages"]:
      #   if key in inputs:
      #     image_pad[key] = torch.zeros_like(
      #       image_pad["input_ids"], dtype=inputs[key])
      self._append_sample_packing(image_pad,
                                  packed_input_ids,
                                  packed_position_ids,
                                  packed_output_mask,
                                  packed_pixel_values,
                                  packed_pixel_values_videos,
                                  packed_image_gird_thw,
                                  packed_video_grid_thw,
                                  packed_advantages,
                                  packed_log_probs_ref,
                                  packed_old_log_probs,
                                  packed_sample_idx,
                                  cu_seqlens)

      packed_input_ids = torch.cat(packed_input_ids, dim=-1)
      packed_output_mask = torch.cat(packed_output_mask, dim=-1)
      packed_position_ids = torch.cat(packed_position_ids, dim=-1)
      packed_pixel_values = None if len(packed_pixel_values) == 0 else \
        torch.cat(packed_pixel_values, dim=0)
      packed_image_gird_thw = None if len(packed_image_gird_thw) == 0 else \
        torch.cat(packed_image_gird_thw, dim=0)
      packed_pixel_values_videos = \
        None if len(packed_pixel_values_videos) == 0 else \
          torch.cat(packed_pixel_values_videos, dim=0)
      packed_video_grid_thw = None if len(packed_video_grid_thw) == 0 else \
        torch.cat(packed_video_grid_thw, dim=0)
      packed_advantages = None if len(packed_advantages) == 0 else \
        torch.cat(packed_advantages, dim=-1)
      packed_log_probs_ref = None if len(packed_log_probs_ref) == 0 else \
        torch.cat(packed_log_probs_ref, dim=-1)
      packed_old_log_probs = None if len(packed_old_log_probs) == 0 else \
        torch.cat(packed_old_log_probs, dim=-1)
    
      # pad seq len to multiple_of
      if (
        self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
      ):
        padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
        packed_input_ids = F.pad(
          packed_input_ids, (0, padding_len),
          value=self.input_builder.pad_token_id)
        packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
        packed_output_mask = F.pad(packed_output_mask, (0, padding_len), value=0)
        if packed_advantages is not None:
          packed_advantages = F.pad(packed_advantages, (0, padding_len), value=0)
        if packed_log_probs_ref is not None:
          packed_log_probs_ref = F.pad(packed_log_probs_ref, (0, padding_len), value=0)
        if packed_old_log_probs is not None:
          packed_old_log_probs = F.pad(packed_old_log_probs, (0, padding_len), value=0)
        cu_seqlens.append(cu_seqlens[-1] + padding_len)
        packed_sample_idx.append(-1)

      inputs = {
        "input_ids": packed_input_ids,
        "position_ids": packed_position_ids,
        "output_mask": packed_output_mask,
        "pixel_values": packed_pixel_values,
        "image_grid_thw": packed_image_gird_thw,
        "pixel_values_videos": packed_pixel_values_videos,
        "video_grid_thw": packed_video_grid_thw,
        "advantages": packed_advantages,
        "log_probs_ref": packed_log_probs_ref,
        "old_log_probs": packed_old_log_probs,
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "sample_idx": torch.tensor(packed_sample_idx, dtype=torch.int32),
      }
      for key in list(inputs.keys()):
        if inputs[key] is None:
          inputs.pop(key)
      return inputs

    def _make_inputs(self, sample: Dict):
      input_ids = sample["original_inputs"]["input_ids"]
      output_ids = torch.tensor(
        sample["response_ids"], dtype=torch.long).unsqueeze(0)
      output_mask = torch.cat(
        [torch.zeros_like(input_ids), torch.ones_like(output_ids)], dim=-1)
      input_len = input_ids.shape[-1]
      output_len = output_ids.shape[-1]
      position_ids = torch.arange(
        input_len, input_len + output_len).unsqueeze(0).repeat(3, 1, 1)
      position_ids = torch.cat(
        [sample["original_inputs"]["position_ids"], position_ids], dim=-1)
      sequence_ids = torch.cat([input_ids, output_ids], dim=-1)

      inputs = {
        "input_ids": sequence_ids,
        "output_mask": output_mask,
        "position_ids": position_ids,
        "sample_idx": sample["idx"],
      }
      vision_keys = [
        "pixel_values", "image_grid_thw",
        "pixel_values_videos", "video_grid_thw"]
      for key in vision_keys:
        if sample["original_inputs"].get(key) is not None:
          inputs[key] = sample["original_inputs"][key]
      
      if "advantages" in sample:
        advantages = torch.tensor(sample["advantages"]).unsqueeze(0)
        inputs["advantages"] = advantages
      if "log_probs_ref" in sample:
        log_probs_ref = torch.tensor(sample["log_probs_ref"]).unsqueeze(0)
        inputs["log_probs_ref"] = log_probs_ref
      if "old_log_probs" in sample:
        old_log_probs = torch.tensor(sample["old_log_probs"]).unsqueeze(0)
        inputs["old_log_probs"] = old_log_probs
      return inputs

    def build_collate_fn(self):
      def collate_fn(samples):
        max_length = max(sample["input_ids"].shape[-1] for sample in samples)
        # pad all input_ids to the max length
        batch = {}
        for key in ["input_ids", "output_mask", "position_ids",
                    "advantages", "log_probs_ref", "old_log_probs"]:
          for sample in samples:
            if not key in sample:
              continue
            sample[key] = F.pad(
              sample[key], (0, max_length - sample[key].shape[-1]),
              value=0
            )
        
        batch["input_ids"] = torch.cat(
          [sample["input_ids"] for sample in samples], dim=0)
        batch["output_mask"] = torch.cat(
          [sample["output_mask"] for sample in samples], dim=0)
        batch["position_ids"] = torch.cat(
          [sample["position_ids"] for sample in samples], dim=1)
        batch["advantages"] = torch.cat(
          [sample["advantages"] for sample in samples], dim=0)

        if "log_probs_ref" in sample:
          batch["log_probs_ref"] = torch.cat(
            [sample["log_probs_ref"] for sample in samples], dim=0)
        if "old_log_probs" in sample:
          batch["old_log_probs"] = torch.cat(
            [sample["old_log_probs"] for sample in samples], dim=0)

        for key in ["pixel_values", "image_grid_thw",
                    "pixel_values_videos", "video_grid_thw"]:
          for sample in samples:
            if key in sample and sample[key] is not None:
              batch[key] = torch.cat([sample[key] for sample in samples], dim=0)
        return batch

      return collate_fn

    def __iter__(self):
      buffer = []
      cur_length = 0
      def _should_pack():
        if self.num_packing_samples < 0:
          return cur_length + sample_length > self.max_length
        else:
          return len(buffer) >= self.num_packing_samples
      for sample in self.experiences:
        inputs = self._make_inputs(sample)
        if not self.use_packing:
          yield inputs
          continue
        sample_length = inputs["input_ids"].shape[-1]
        if _should_pack():
          packed_inputs = self._packing(buffer)
          yield packed_inputs
          buffer = [inputs]
          cur_length = sample_length
        else:
          buffer.append(inputs)
          cur_length += sample_length
      if len(buffer) > 0:
        yield self._packing(buffer)

class VllmInferenceDataset(DistributedDataset):
  "For vllm batch inference"
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               max_images: int = 10,
               system_prompt: str = DEFAULT_SYSTEM_PROMPT,
               pretrained_model_name_or_path: Optional[str] = None,
               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed)
    self.min_visual_tokens_per_image = \
      kwargs.get("min_visual_tokens_per_image", 4)
    self.max_visual_tokens_per_image = \
      kwargs.get("max_visual_tokens_per_image", 512)
    self.max_images = kwargs.get("max_images", 10)
    self.system_prompt = system_prompt

    self.input_builder = Qwen2VLInputBuilder(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      **kwargs
    )

  def _process(self,
               sample: Dict[str, Any],
               **kwargs) -> Dict[str, torch.Tensor]:
    assert "messages" in sample["json"], \
        f"sample must contain 'messages' key, but got {sample['json'].keys()}"

    messages = sample["json"]["messages"]
    prompt_messages = []
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue
      for block in content:
        if block["type"] == "image":
          self.input_builder.fill_image_block(block, sample, **kwargs)
        elif block["type"] == "video":
          self.input_builder.fill_video_block(block, sample, **kwargs)
        elif block["type"] == "text":
          continue
        else:
          raise ValueError(
            f"sample process error, unsupport value type: {block['type']}")

    if messages[0]["role"] != "system":
      system = {
        "role": "system", "content": self.system_prompt}
      messages.insert(0, system)

    for turn in messages:
      if turn["role"] == "assistant":
        break
      prompt_messages.append(turn)

    annotation = None
    if messages[-1]["role"] == "assistant":
      annotation = messages[-1]["content"]
      if isinstance(annotation, list):
        annotation = annotation[-1]["text"]

    text = self.input_builder.processor.apply_chat_template(
      prompt_messages, tokenize=False,
      add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    if image_inputs:
      image_inputs = image_inputs[:self.max_images]
    if video_inputs:
      video_inputs = video_inputs[:self.max_images]

    mm_data = {}
    if image_inputs:
      mm_data["image"] = image_inputs
    if video_inputs:
      mm_data["video"] = video_inputs
    return {
      "vllm_inputs": {
        "prompt": text,
        "multi_modal_data": mm_data,
      },
      "annotation": annotation,
      "source": sample["json"]["source"],
      "__key__": sample["__key__"],
      "__url__": sample["__url__"],
    }

  def __iter__(self):

    for sample in self.dataset:
      sample_key = sample["__key__"] if "__key__" in sample else ""
      sample_url = sample["__url__"] if "__url__" in sample else ""

      try:
        source_name = sample["json"]["source"]
      except:
        source_name = "None"

      self.source_sample_cnt.setdefault(source_name, 0)
      self.source_sample_cnt[source_name] += 1

      try:
        inputs = self._process(sample)
        yield inputs
      except:
        self.source_error_cnt.setdefault(source_name, 0)
        self.source_error_cnt[source_name] += 1
        error_ratio = self.source_error_cnt[source_name] * 1.0 / \
          self.source_sample_cnt[source_name]
        logger.error(
          f"PromptVisionDataset process sample error. "
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, "
          f"errmsg={traceback.format_exc()}")
        continue




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
        if assistant_start[-3:] == start_pattern:
          to_mask = True
          assistant_start = []
      else:
        if _id in end_pattern:
          assistant_end.append(_id.item())
        else:
          assistant_end = []
        if assistant_end[-2:] == end_pattern:
          to_mask = False
          assistant_end = []
    masks.append(mask)
  return torch.tensor(masks)


class ChatCompletionVisionDatasetV2(DistributedDataset):
  def __init__(self,
               sources: Union[str, List[str]],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               num_epochs: int=1,
               max_length: int = 1024,
               datasource_config: Dict[str, Dict[str, Any]] = {},
               system_prompt: str = DEFAULT_SYSTEM_PROMPT,
               pretrained_model_name_or_path: Optional[str] = None,
               multiple_of: int = 8,

               **kwargs):
    super().__init__(
      sources=sources, rank=rank, world_size=world_size,
      num_workers=num_workers, seed=seed,
      num_epochs=num_epochs)
    self.shrink_ratio = kwargs.get("shrink_ratio", 0.9)
    self.max_retry = kwargs.get("max_retry", 5)
    self.min_visual_tokens_per_image = \
      kwargs.get("min_visual_tokens_per_image", 4)
    self.max_visual_tokens_per_image = \
      kwargs.get("max_visual_tokens_per_image", 512)
    self.max_images = kwargs.get("max_images", 10)
    self.system_prompt = system_prompt
    self.max_length = max_length
    self.multiple_of = multiple_of

    kwargs["min_visual_tokens_per_image"] = self.min_visual_tokens_per_image
    kwargs["max_visual_tokens_per_image"] = self.max_visual_tokens_per_image
    self.input_builder = Qwen2VLInputBuilder(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      **kwargs
    )

    # append image_pad for each packing
    image_pad_len = self.input_builder.gen_img_pad()["input_ids"].shape[-1]
    self.max_length = max_length - image_pad_len
    assert self.max_length > 0

    self.datasource_config = datasource_config
    logger.info(
      f"ChatCompletionVisionDatasetV2 datasource_config: "
      f"{json.dumps(self.datasource_config, indent=2)}"
    )

  def _process_chat(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {},
                    **kwargs) -> Dict[str, torch.Tensor]:
    assert "message" in sample["json"] or "messages" in sample["json"]
    data_conf["max_visual_tokens_per_image"] = max(
        data_conf["max_visual_tokens_per_image"], data_conf["min_visual_tokens_per_image"])
    
    # print_rank_0(print_input_info(data_conf, "data_conf:", return_str=True))
    msg_key = "message" if "message" in sample["json"] else "messages"
    messages = sample["json"][msg_key]
    for turn in messages:
      content = turn["content"]
      if isinstance(content, str):
        continue
      for block in content:
        if block["type"] == "image":
          self.input_builder.fill_image_block(block, sample, 
                                  conf=data_conf)
        elif block["type"] == "video":
          self.input_builder.fill_video_block(block, sample,
                                  conf=data_conf)
        elif block["type"] == "text":
          continue
        else:
          raise ValueError(f"sample process error, unsupport value type: {block['type']}")

    inputs, text, image_inputs, video_inputs = \
      self.input_builder.tokenize_messages(messages)

    if inputs["input_ids"].shape[-1] > 32768:
      raise ValueError(f"Sample is too long. text_len={len(text)=}, token_len={inputs['input_ids'].shape[-1]}")
    
    inputs["loss_mask"] = get_assistant_mask(
      inputs["input_ids"],
      start_pattern=[151644, 77091, 198],
      end_pattern=[151645, 198]
    )

    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
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
      
    inputs.pop("attention_mask")
    return inputs
  # {
  #     "original_inputs": inputs,
  #     "vllm_inputs": {
  #       "prompt": text,
  #       "multi_modal_data": mm_data,
  #     },
  #     "annotation": annotation,
  #     "source": sample["json"]["source"],
  #     "__key__": sample["__key__"],
  #     "__url__": sample["__url__"],
  #   }

  def _process(self, sample, source_name=None):
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
      "video_nframe": self.input_builder.video_nframe,
      "video_fps": self.input_builder.video_fps,
      "video_min_frames": self.input_builder.video_min_frames,
      "video_max_frames": self.input_builder.video_max_frames,
    }
    if source_name != None and source_name in self.datasource_config:
      for key in self.datasource_config[source_name]:
        source_conf[key] = self.datasource_config[source_name][key]

    for retry in range(self.max_retry):
      if data_format == "chatml":
        inputs = self._process_chat(sample, source_conf)
      elif data_format == "completion":
        inputs = self._process_completion(sample, source_conf)
      else:
        raise NotImplementedError(
            f"Unsupported dataset format `{data_format}`")

      if not inputs:
        raise ValueError("Empty inputs, skip")
      
      if inputs["input_ids"].shape[-1] > self.max_length:
        source_conf["max_visual_tokens_per_image"] = (
          source_conf["max_visual_tokens_per_image"] * self.shrink_ratio)
        continue
      else:
        assert inputs["input_ids"].shape[-1] \
            <= self.max_length, "inputs too long"
        return inputs
    else:
      raise ValueError(
        f"Unable to generate sample within "
        f"max_length={self.max_length} after {retry} retrys"
      )

  def _process_completion(self,
                    sample: Dict[str, Any],
                    data_conf: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    assert "segments" in sample["json"]
    data_conf["max_visual_tokens_per_image"] = max(
        data_conf["max_visual_tokens_per_image"], data_conf["min_visual_tokens_per_image"])

    text = ""
    vision_infos = []
    segments = sample["json"]["segments"]
    for segment in segments:
      if segment["type"] == "text":
        text += segment["text"]
      elif segment["type"] == "image":
        text += "<|vision_start|><|image_pad|><|vision_end|>"
        self.input_builder.fill_image_block(segment, sample,
                                conf=data_conf)
        vision_infos.append(segment)
      elif segment["type"] == "video":
        text += "<|vision_start|><|video_pad|><|vision_end|>"
        self.input_builder.fill_video_block(segment, sample,
                                conf=data_conf)
        vision_infos.append(segment)
      else:
        logger.warning(f"!!! Unsupport {segment['type']=}, skip this segment.")
    
    # append EOS token
    text += "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(vision_infos = vision_infos)
    inputs = self.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )
    # inputs, text, image_inputs, video_inputs = \
    #   self.input_builder.tokenize_messages(vision_infos)

    # For the Warning: (add by zzx)
    #   Token indices sequence length is longer than the specified maximum 
    #   sequence length for this model (**** > 32768). Running this sequence 
    #.  through the model will result in indexing errors
    if inputs["input_ids"].shape[-1] > 32768:
      print(f"Sample is too long. token_len={inputs['input_ids'].shape[-1]}")
    
    # mask all vision token
    # <|vision_start|>: 151652 , <|vision_end|>: 151653, <|image_pad|>: 151655, <|video_pad|>: 151656
    input_ids = inputs["input_ids"]
    inputs["loss_mask"] = torch.ones_like(input_ids)
    inputs["loss_mask"][
        (input_ids == self.vision_start_token_id) | 
        (input_ids == self.vision_end_token_id) |
        (input_ids == self.image_token_id) |
        (input_ids == self.video_token_id)
      ] = 0
    # mask EOS token
    inputs["loss_mask"][-1][-1] = 0
    if inputs["loss_mask"].sum() == 0:
      raise ValueError(
        f"Unable to generate sample with 0 loss_mask."
      )

    inputs["position_ids"] = self.input_builder.get_rope_index(
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
                          sample_idx: Optional[int] = None):


    packed_input_ids.append(inputs["input_ids"].flatten())
    packed_loss_mask.append(inputs["loss_mask"].flatten())
    packed_position_ids.append(inputs["position_ids"])
    if sample_idx is None:
      sample_idx = len(cu_seqlens) - 1
    packed_sample_idx.append(
      torch.full_like(packed_input_ids[-1], sample_idx))


    if "pixel_values" in inputs:
      packed_pixel_values.append(inputs["pixel_values"])
      packed_image_gird_thw.append(inputs["image_grid_thw"])
    if "pixel_values_videos" in inputs:
      packed_pixel_values_videos.append(inputs["pixel_values_videos"])
      packed_video_grid_thw.append(inputs["video_grid_thw"])
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
    cu_seqlens: List[int] = [0]

    valid_seq_len = 0
    for _, inputs in enumerate(buffer):
      valid_seq_len += self._append_sample_packing(inputs,
                                    packed_input_ids,
                                    packed_position_ids,
                                    packed_loss_mask,
                                    packed_pixel_values,
                                    packed_pixel_values_videos,
                                    packed_image_gird_thw,
                                    packed_video_grid_thw,
                                    packed_sample_idx,
                                    cu_seqlens)

    # append a pad image sequence to trigger ViT
    image_pad = self.input_builder.gen_img_pad()
    image_pad["loss_mask"] = torch.zeros_like(image_pad["input_ids"])
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
                              sample_idx=-1)

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

    # pad seq len to multiple_of
    if (
      self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0
    ):
      padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
      packed_input_ids = F.pad(
        packed_input_ids, (0, padding_len),
        value=self.input_builder.pad_token_id)
      packed_sample_idx = F.pad(
        packed_sample_idx, (0, padding_len), value=-1)
      packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
      packed_loss_mask = F.pad(packed_loss_mask, (0, padding_len), value=0)
      cu_seqlens.append(cu_seqlens[-1] + padding_len)

    inputs = {
      "input_ids": packed_input_ids,
      "position_ids": packed_position_ids,
      "loss_mask": packed_loss_mask,
      "pixel_values": packed_pixel_values,
      "image_grid_thw": packed_image_gird_thw,
      "pixel_values_videos": packed_pixel_values_videos,
      "video_grid_thw": packed_video_grid_thw,
      "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
      "sample_idx": packed_sample_idx.to(torch.int32)
    }
    return inputs

  def __iter__(self):
    """
    实现 __iter__ 方法，用于迭代 ChatCompletionVisionDataset 对象。

    Args:
        无

    Returns:
        迭代器，用于逐个返回处理后的样本数据。

    """

    buffer = []
    source_list = []
    cur_length = 0

    for sample in self.dataset:
      sample_key = sample["__key__"] if "__key__" in sample else ""
      sample_url = sample["__url__"] if "__url__" in sample else ""

      try:
        source_name = sample["json"]["source"]
        # WARN: ugly code, for dirty dataset.
        if source_name.startswith("PDFA"):
          source_name = "PDFA"
        elif source_name.startswith("/llm_reco_ssd/luoxinchen/dataset/"):
          source_name = source_name.split("/")[4]
      except:
        source_name = "None"

      self.source_sample_cnt.setdefault(source_name, 0)
      self.source_sample_cnt[source_name] += 1

      try:
        inputs = self._process(sample, source_name)
      except:
        self.source_error_cnt.setdefault(source_name, 0)
        self.source_error_cnt[source_name] += 1
        error_ratio = self.source_error_cnt[source_name] * 1.0 / \
          self.source_sample_cnt[source_name]
        logger.error(
          f"ChatCompletionVisionDataset process sample error. "
          f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, "
          f"errmsg={traceback.format_exc()}")
        continue

      sample_length = inputs["input_ids"].shape[-1]
      if cur_length + sample_length > self.max_length:
        packed_inputs = self._packing(buffer)
        packed_inputs["data_source"] = source_list
        buffer = [inputs]
        source_list = [source_name]
        cur_length = sample_length

        # skip pure text sample
        # 有pad image，原则上不会出现纯文本输入
        if packed_inputs["pixel_values"] is None and \
            packed_inputs["pixel_values_videos"] is None:
          logger.warning("Skip pure text sample.")
          continue

        # skip 0 label pack
        if packed_inputs["loss_mask"].sum() == 0:
          logger.warning("Skip 0 lable sample.")
          continue
        
        # pixel_values_videos
        yield packed_inputs

      else:
        buffer.append(inputs)
        source_list.append(source_name)
        cur_length += sample_length



class ChatCompletionVisionV2ParquetDataset(ChatCompletionVisionDatasetV2):
  def __init__(self, sources, num_workers, max_length, world_size=1, rank=0, shuffle_seed=1024, num_epochs=1, **kargs):
    self.rng = random.Random(shuffle_seed)
    self.num_workers = num_workers
    self.num_epochs = num_epochs
    super().__init__(sources, num_workers=num_workers, max_length=max_length, world_size=world_size, rank=rank, num_epochs=num_epochs, **kargs)

  def _build_source_dataset(self, sources):
    data_file_list = []
    if dist.get_rank() == 0:
      data_files = []
      if isinstance(sources, str) and sources.endswith(".json"):
        with open(sources, "r") as fp:
          data_files = json.loads(fp.read())
          data_files = [fn for fn in data_files if fn.endswith(".parquet")]
      elif isinstance(sources, list):
        for source in sources:
          hdfs_files = shell_hdfs_ls(source)
          data_files += [fn for fn in hdfs_files if fn.endswith(".parquet")]
      # repeat
      for i in range(self.num_epochs):
        data_files.sort()
        self.rng.shuffle(data_files)
        data_file_list += [(fn, i) for fn in data_files]
      logger.error(f"ChatCompletionVisionV2ParquetDataset rank{dist.get_rank()}: ori_file_num={len(data_files)} file_num={len(data_file_list)}")

    t = [data_file_list]
    dist.broadcast_object_list(t, src=0)
    data_file_list = t[0]

    logger.error(f"ChatCompletionVisionV2ParquetDataset rank{dist.get_rank()}: file_num={len(data_file_list)}")
    if len(data_file_list) == 0:
      raise ValueError(f"no datafile found!")

    dataset = ParquetDataset(data_file_list, self.num_workers)
    return dataset, -1

  def state_dict(self, ):
    return self.dataset.state_dict()
  
  def load_state_dict(self, state_dict):
    self.dataset.load_state_dict(state_dict)