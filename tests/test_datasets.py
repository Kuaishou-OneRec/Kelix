import os
import torch

import wids

from torch.utils.data import DataLoader
from recovlm.data.datasets import ChatCompletionDataset, ImageTextPairDatasetWithPacking, ChatCompletionVisionDataset
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from tests.utils import init_processes
"""
    # dataset = LLaVA_CC3M_Dataset(
    #     source="/llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K/",
    #     processor_path="/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14"
    # )
    # for idx, batch in enumerate(DataLoader(dataset, batch_size=3, collate_fn=dataset.build_collate_fn())):
    #     print(idx, batch)
    #     #print(dataset.processor.tokenizer.decode(batch["input_ids"]))
    #     if idx >= 1:
    #         break
    # for key, tensor in batch.items():
    #     print(key, tensor.shape)
    # for input_ids in batch["input_ids"]:
    #     print("=" * 10)
    #     print(dataset.processor.tokenizer.decode(input_ids))
"""

TOKENIZER = "/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct"


def test_chat_completion():
  records = [
      {
          "conversations": [
              {"role": "user", "content": "Hello!"},
              {"role": "assistant", "content": "你好👋!"},
          ]
      },
      {
          "conversations": [
              {"role": "user", "content": "こんにちは!"},
              {"role": "assistant", "content": "你好!"},
          ]
      }
  ]
  dataset = ChatCompletionDataset(
      source=records,
      tokenizer=TOKENIZER,
      input_key="conversations",
      system_prompt="You are RecoVLM",
      chat_template="chat_template_with_generation_tag",
      max_length=128
  )

  ans = {
      'input_ids': torch.tensor(
          [
              [151644, 8948, 198, 2610, 525, 31462, 53, 10994, 151645,
               198, 151644, 872, 198, 9707, 0, 151645, 198, 151644,
               77091, 198, 108386, 145707, 0, 151645, 198],
              [151644, 8948, 198, 2610, 525, 31462, 53, 10994, 151645,
               198, 151644, 872, 198, 89015, 0, 151645, 198, 151644,
               77091, 198, 108386, 0, 151645, 198, 151643]]),
      'attention_mask': torch.tensor(
          [
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
          ]),
      'loss_mask': torch.tensor(
          [
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
          ])
  }

  for batch in DataLoader(dataset,
                          batch_size=2,
                          shuffle=False,
                          collate_fn=dataset.collate_fn):
    for key, t in batch.items():
      assert torch.allclose(t, ans[key])

  records = [
      {
          "conversations": [
              {"from": "human", "value": "Hello!"},
              {"from": "gpt", "value": "你好👋!"},
          ]
      },
      {
          "conversations": [
              {"from": "human", "value": "こんにちは!"},
              {"from": "gpt", "value": "你好!"},
          ]
      }
  ]
  dataset = ChatCompletionDataset(
      source=records,
      tokenizer=TOKENIZER,
      input_key="conversations",
      role_key="from",
      content_key="value",
      user_name="human",
      assistant_name="gpt",
      system_prompt="You are RecoVLM",
      chat_template="chat_template_with_generation_tag",
      max_length=128
  )

  for batch in DataLoader(dataset,
                          batch_size=2,
                          shuffle=False,
                          collate_fn=dataset.collate_fn):
    for key, t in batch.items():
      assert torch.allclose(t, ans[key])

def test_image_text_pair_dataset_with_packing():

    # dataset = wids.ShardListDataset(
    #     "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
    # )
    processor = Qwen2VLProcessor.from_pretrained(
        "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")

    ds = ImageTextPairDatasetWithPacking(
        sources = "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json",
        processor = processor,
        max_length = 3072,
        min_visual_tokens = 64,
        max_visual_tokens = 512,
        spatial_merge_size = 2,
        image_token_id = 151655,
        video_token_id = 151656,
        vision_start_token_id = 151652,
        patch_size = 14,
        shrink_ratio = 0.9,
        max_retry = 5,
        multiple_of = 8
    )
    def collate_fn(samples):
        return samples[0]

    dataloader = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    for item in dataloader:
        print(item)
        break

def test_chat_vision_dataset_with_packing():
    # processor = Qwen2VLProcessor.from_pretrained(
    #     "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")
    init_processes(0, 1)
    ds = ChatCompletionVisionDataset(
        sources = "/llm_reco_ssd/luoxinchen/dataset/Stage2/the_cauldron/index.json",
        max_length = 3072,
        min_visual_tokens_per_image = 4,
        max_visual_tokens_per_image = 512,
        base_model_dir = "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",
        shrink_ratio = 0.9,
        max_retry = 5,
        multiple_of = 8
    )
    def collate_fn(samples):
        return samples[0]

    dataloader = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    for idx, item in enumerate(dataloader):
        print(item)
        break
