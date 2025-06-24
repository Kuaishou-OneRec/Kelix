from typing import Optional, Callable, Dict, Union, Any

import os
import json
import torch
import wids
import psutil

import webdataset as wds
import torch.distributed as dist


from recovlm.utils.numa_bind import get_numa_bind_info
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from tqdm import tqdm

from recovlm.data.datasets import ImageTextPairDatasetWithPacking, \
    ChatCompletionVisionDataset, ChatCompletionVisionParquetDataset, \
    ChatCompletionVisionDpoDataset, ChatCompletionVisionDpoParquetDataset,InternVLChatCompletionVisionParquetDataset, \
    BalanceParquetDataset, \
    ChatCompletionVisionDataset_moonvit,ChatCompletionVisionParquetDataset_moonvit, \
    ChatCompletionVisionDataset_siglip,ChatCompletionVisionParquetDataset_siglip, ChatCompletionVisionParquetDataset_navit, ChatCompletionVisionParquetDataset_keye, ChatCompletionVisionParquetDataset_keye_vitrope, ChatCompletionVisionParquetDataset_keye_vitrope_slowfast

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
                                          num_workers=8,
                                          video_nframe=-1,
                                          video_fps=2.0,
                                          video_min_frames=2,
                                          video_max_frames=120,
                                          datasource_config={}
                                          ):
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
        datasource_config=datasource_config)

    ### packing, batching size=1; shuffle in dataset
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda x: x[0]
    )
    return dataloader

def get_chat_completion_vision_dpo_dataloader(sources: str,
                                          max_length,
                                          min_visual_tokens_per_image,
                                          max_visual_tokens_per_image,
                                          base_model_dir,
                                          shrink_ratio,
                                          max_retry,
                                          multiple_of,
                                          num_workers=8,
                                          video_nframe=-1,
                                          video_fps=2.0,
                                          video_min_frames=2,
                                          video_max_frames=120,
                                          datasource_config={}):

    dataset = ChatCompletionVisionDpoDataset(
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
        datasource_config=datasource_config)

    ### packing, batching size=1; shuffle in dataset
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda x: x[0]
    )
    return dataloader

def get_chat_completion_vision_parquet_dataloader(sources: str,
                                          max_length,
                                          min_visual_tokens_per_image,
                                          max_visual_tokens_per_image,
                                          base_model_dir,
                                          shrink_ratio,
                                          max_retry,
                                          multiple_of,
                                          num_epochs=1,
                                          shuffle_seed=1024,
                                          num_workers=8,
                                          video_nframe=-1,
                                          video_fps=2.0,
                                          video_min_frames=2,
                                          video_max_frames=120,
                                          datasource_config={},
                                          min_visual_tokens_per_frame=4,
                                          max_visual_tokens_per_frame=512,
                                          **kwargs):
    model_type = kwargs.get('model_class','Qwen2VLForConditionalGeneration')
    print('test_cut_to_pad:',kwargs.get('cut_to_pad',False))
    use_balance = kwargs.get("use_flops_balance", False)
    kwargs["use_flops_balance"] = use_balance
    ModelDataset = {'Qwen2VLForConditionalGeneration':ChatCompletionVisionParquetDataset,
                    'Qwen2_5_VLForConditionalGeneration':ChatCompletionVisionParquetDataset,
                    'Qwen2_5_VLForConditionalGeneration_moonvit':ChatCompletionVisionParquetDataset_moonvit,
                    'Qwen2_5_VLForConditionalGeneration_siglip':ChatCompletionVisionParquetDataset_siglip,
                    'Qwen3SiglipForConditionalGeneration_navit':ChatCompletionVisionParquetDataset_navit,
                    'KeyeForConditionalGeneration': ChatCompletionVisionParquetDataset_keye,
                    'KeyeForConditionalGeneration_vitrope': ChatCompletionVisionParquetDataset_keye_vitrope,
                    'KeyeForConditionalGeneration_vitrope_slowfast': ChatCompletionVisionParquetDataset_keye_vitrope_slowfast,
                    'InternVLChatModel':InternVLChatCompletionVisionParquetDataset}
    num_readers = kwargs.get("num_readers", 1)
    shuffle_window = kwargs.get("shuffle_window", 0)

    def input_creator():
        print("cjx debug model_type {}".format(model_type))
        return ModelDataset[model_type](
            sources = sources,
            num_workers = num_workers,
            num_epochs = num_epochs,
            shuffle_seed = shuffle_seed,
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
            datasource_config=datasource_config,
            num_readers=num_readers,
            shuffle_window=shuffle_window,
            min_visual_tokens_per_frame = min_visual_tokens_per_frame,
            max_visual_tokens_per_frame = max_visual_tokens_per_frame, 
            **kwargs
            )

    ### packing, batching size=1; shuffle in dataset
    if use_balance:
        assert num_workers == 1, f"use_flops_balance requires one dataset process per worker"
        dataset = BalanceParquetDataset(input_creator(), model_type, base_model_dir=base_model_dir, **kwargs)

        def worker_init_fn(worker_id: int):
            assert worker_id == 0, f"worker_id expect 0, but got {worker_id}"
            local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
            local_world_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 0))
            cpu_bind = get_numa_bind_info(local_rank, local_world_size)
            if cpu_bind is not None:
                p = psutil.Process(os.getpid())
                p.cpu_affinity(cpu_bind[:8])
            master_port = int(os.environ["MASTER_PORT"]) + 1
            os.environ["MASTER_PORT"] = str(master_port)
            rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
            world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
            if not dist.is_initialized():
                dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
            print(f"dataset_process: rank={dist.get_rank()}, pid={os.getpid()}, bind={cpu_bind[:8]}")
            
        snapshot_every_n_steps = max(kwargs.get("snapshot_step", 1), 1)
        dataloader = StatefulDataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=1,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x : x[0],
            prefetch_factor=16,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )
    else:
        dataset = input_creator()
        dataloader = StatefulDataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=lambda x: x[0],
        )
    return dataloader

def get_chat_completion_vision_dpo_parquet_dataloader(sources: str,
                                          max_length,
                                          min_visual_tokens_per_image,
                                          max_visual_tokens_per_image,
                                          base_model_dir,
                                          shrink_ratio,
                                          max_retry,
                                          multiple_of,
                                          num_epochs=1,
                                          shuffle_seed=1024,
                                          num_workers=8,
                                          video_nframe=-1,
                                          video_fps=2.0,
                                          video_min_frames=2,
                                          video_max_frames=120,
                                          datasource_config={},
                                          **kwargs):

    dataset = ChatCompletionVisionDpoParquetDataset(
        sources = sources,
        num_workers = num_workers,
        num_epochs = num_epochs,
        shuffle_seed = shuffle_seed,
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
        datasource_config=datasource_config
        )

    ### packing, batching size=1; shuffle in dataset
    dataloader = StatefulDataLoader(
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
    elif name == "chat_vision_parquet":
        return get_chat_completion_vision_parquet_dataloader(
            **kwargs
        )
    elif name == "chat_vision_dpo":
        return get_chat_completion_vision_dpo_dataloader(
            **kwargs
        )
    elif name == "chat_vision_dpo_parquet":
        return get_chat_completion_vision_dpo_parquet_dataloader(
            **kwargs
        )
    else:
        raise NotImplementedError("Unsupported dataloader.")


