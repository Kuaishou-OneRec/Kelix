from typing import Optional, Callable, Dict, Union, Any

import os
import json
import torch
import wids

import webdataset as wds

from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from tqdm import tqdm

from recovlm.training.parallel import worker_init_fn

from recovlm.data.datasets import ImageTextPairDatasetWithPacking, \
    ChatCompletionVisionDataset

RESPONSE_TEMPLATE = "{% for message in messages %}{{message['content'] + '<|im_end|>'}}{% endfor %}"


def get_collate_fn(processor, max_length):
  def collate_fn(batch):
    prompt_messages = []
    response_messages = []
    for sample in batch:
      image = sample[".jpg"]
      text = sample[".txt"]
      if image.mode != "RGB":
        image = image.convert("RGB")
      prompt_messages.append([
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": image,
                  },
                  {"type": "text", "text": "Describe this image."},
              ],
          }
      ])
      response_messages.append([
          {"content": text}
      ])

    text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        text_kwargs={
            "padding": True,
            "padding_side": "left"
        }
    )

    response_inputs = processor.tokenizer.apply_chat_template(
        response_messages,
        chat_template=RESPONSE_TEMPLATE,
        padding=True,
        add_generation_prompt=False,
        return_tensors="pt",
        tokenizer_kwargs={
            "padding_side": "right",
        }
    )

    response_mask = (
        response_inputs != processor.tokenizer.pad_token_id).type(torch.int64)
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
      inputs[key] = inputs[key][:, :max_length]
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

def get_indexed_dataloader(sources: str,
                           processor,
                           batch_size: int,
                           num_workers: int = 8,
                           chunksize: int = 1000,
                           shuffle: bool = True,
                           max_length: Optional[int] = None,
                           rank: Optional[int] = None,
                           collator: Optional[Callable] = None):
  # TODO: concat之后会有单个dataset index out of range的情况，而且数据没有被均匀shuffle，再排查下；考虑提前合并index
  dataset = torch.utils.data.ConcatDataset(
    [wids.ShardListDataset(source) for source in sources])
  sampler = wids.DistributedChunkedSampler(
      dataset, chunksize=chunksize, shuffle=shuffle,
      rank=rank
  )
  dataloader = DataLoader(
      dataset, batch_size=batch_size, num_workers=num_workers,
      sampler=sampler,
      collate_fn=collator
  )
  return dataloader

def get_image_text_pair_with_packing_dataloader(sources: str,
                                                max_length,
                                                min_visual_tokens,
                                                max_visual_tokens,
                                                base_model_dir,
                                                shrink_ratio,
                                                max_retry,
                                                multiple_of):

    dataset = ImageTextPairDatasetWithPacking(
        sources = sources,
        max_length = max_length,
        min_visual_tokens = min_visual_tokens,
        max_visual_tokens = max_visual_tokens,
        base_model_dir = base_model_dir,
        shrink_ratio = shrink_ratio,
        max_retry = max_retry,
        multiple_of = multiple_of)

    ### packing, batching size=1; shuffle in dataset
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda x: x[0]
    )
    return dataloader

def get_chat_completion_vision_dataloader(sources: str,
                                          max_length,
                                          min_visual_tokens_per_image,
                                          max_visual_tokens_per_image,
                                          base_model_dir,
                                          shrink_ratio,
                                          max_retry,
                                          multiple_of,
                                          need_padding=False,
                                          num_workers=8,
                                          video_nframe=-1,
                                          video_fps=2.0,
                                          video_min_frames=2,
                                          video_max_frames=120):

    dataset = ChatCompletionVisionDataset(
        sources = sources,
        max_length = max_length,
        min_visual_tokens_per_image = min_visual_tokens_per_image,
        max_visual_tokens_per_image = max_visual_tokens_per_image,
        video_nframe=video_nframe,
        video_fps=video_fps,
        video_min_frames=video_min_frames,
        video_max_frames=video_max_frames,
        base_model_dir=base_model_dir,
        shrink_ratio=shrink_ratio,
        max_retry=max_retry,
        multiple_of=multiple_of,
        need_padding=need_padding)

    ### packing, batching size=1; shuffle in dataset
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda x: x[0]
    )
    return dataloader


def get_dataloader(name: str, **kwargs):
    if name == "image_text_pair":
        return get_image_text_pair_with_packing_dataloader(
            **kwargs
        )
    elif name == "chat_vision":
        return get_chat_completion_vision_dataloader(
            **kwargs
        )
    else:
        raise NotImplementedError("Unsupported dataloader.")

if __name__ == '__main__':
  # source=[
  #     "/llm_reco_ssd/luoxinchen/dataset/cc12m/cc12m-index.json",
  #     "/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json"
  # ],
  # source = "/llm_reco_ssd/luoxinchen/dataset/cc12m/cc12m-index.json"
  sources = [
    "/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json",
    "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
  ]
  processor = AutoProcessor.from_pretrained(
      "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")
  dataloader = get_indexed_dataloader(
      sources=sources,
      processor=processor,
      batch_size=32,
      num_workers=4,
      shuffle=True,
      max_length=1024,
      rank=1)
  for s in tqdm(dataloader):
    print(s["pixel_values"].shape)
    print(s["image_grid_thw"].shape)
    print(s["image_grid_thw"][0])
    print(s["image_grid_thw"].prod())
    t = 0
    for a, b, c in s["image_grid_thw"]:
        t += (a * b * c)
    print("sssss", t)
    assert t == s["pixel_values"].shape[0]
    assert s["pixel_values"].shape[0] == s["image_grid_thw"].prod(dim)
    # for input_ids in s["input_ids"]:
    #   print(processor.tokenizer.decode(input_ids))
    #   print("=" * 10)
    break
