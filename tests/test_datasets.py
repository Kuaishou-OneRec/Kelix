import os
import torch

import wids
import logging
import json
import time

from recovlm.utils.common import shell_hdfs_ls

from torch.utils.data import DataLoader
from recovlm.data.datasets import ChatCompletionDataset, ImageTextPairDatasetWithPacking, ChatCompletionVisionDataset, ParquetDataset, ChatCompletionVisionParquetDataset
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


# def test_chat_completion():
#   records = [
#       {
#           "conversations": [
#               {"role": "user", "content": "Hello!"},
#               {"role": "assistant", "content": "你好👋!"},
#           ]
#       },
#       {
#           "conversations": [
#               {"role": "user", "content": "こんにちは!"},
#               {"role": "assistant", "content": "你好!"},
#           ]
#       }
#   ]
#   dataset = ChatCompletionDataset(
#       source=records,
#       tokenizer=TOKENIZER,
#       input_key="conversations",
#       system_prompt="You are RecoVLM",
#       chat_template="chat_template_with_generation_tag",
#       max_length=128
#   )

#   ans = {
#       'input_ids': torch.tensor(
#           [
#               [151644, 8948, 198, 2610, 525, 31462, 53, 10994, 151645,
#                198, 151644, 872, 198, 9707, 0, 151645, 198, 151644,
#                77091, 198, 108386, 145707, 0, 151645, 198],
#               [151644, 8948, 198, 2610, 525, 31462, 53, 10994, 151645,
#                198, 151644, 872, 198, 89015, 0, 151645, 198, 151644,
#                77091, 198, 108386, 0, 151645, 198, 151643]]),
#       'attention_mask': torch.tensor(
#           [
#               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
#           ]),
#       'loss_mask': torch.tensor(
#           [
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
#           ])
#   }

#   for batch in DataLoader(dataset,
#                           batch_size=2,
#                           shuffle=False,
#                           collate_fn=dataset.collate_fn):
#     for key, t in batch.items():
#       assert torch.allclose(t, ans[key])

#   records = [
#       {
#           "conversations": [
#               {"from": "human", "value": "Hello!"},
#               {"from": "gpt", "value": "你好👋!"},
#           ]
#       },
#       {
#           "conversations": [
#               {"from": "human", "value": "こんにちは!"},
#               {"from": "gpt", "value": "你好!"},
#           ]
#       }
#   ]
#   dataset = ChatCompletionDataset(
#       source=records,
#       tokenizer=TOKENIZER,
#       input_key="conversations",
#       role_key="from",
#       content_key="value",
#       user_name="human",
#       assistant_name="gpt",
#       system_prompt="You are RecoVLM",
#       chat_template="chat_template_with_generation_tag",
#       max_length=128
#   )

#   for batch in DataLoader(dataset,
#                           batch_size=2,
#                           shuffle=False,
#                           collate_fn=dataset.collate_fn):
#     for key, t in batch.items():
#       assert torch.allclose(t, ans[key])

# def test_image_text_pair_dataset_with_packing():

#     # dataset = wids.ShardListDataset(
#     #     "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
#     # )
#     processor = Qwen2VLProcessor.from_pretrained(
#         "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")

#     ds = ImageTextPairDatasetWithPacking(
#         sources = "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json",
#         processor = processor,
#         max_length = 3072,
#         min_visual_tokens = 64,
#         max_visual_tokens = 512,
#         spatial_merge_size = 2,
#         image_token_id = 151655,
#         video_token_id = 151656,
#         vision_start_token_id = 151652,
#         patch_size = 14,
#         shrink_ratio = 0.9,
#         max_retry = 5,
#         multiple_of = 8
#     )
#     def collate_fn(samples):
#         return samples[0]

#     dataloader = DataLoader(
#         dataset=ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=8,
#         collate_fn=collate_fn
#     )
#     for item in dataloader:
#         print(item)
#         break

def test_chat_vision_dataset_with_packing():
    # processor = Qwen2VLProcessor.from_pretrained(
    #     "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")
    init_processes(0, 1)
    ds = ChatCompletionVisionDataset(
        sources = "/llm_reco/luoxinchen/dataset/Stage2/the_cauldron_recaption_v1/index.json",
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

# def test_interleaving():
#     init_processes(0, 1)
#     ds = ChatCompletionVisionDataset(
#         sources = "/llm_reco_ssd/luoxinchen/dataset/Stage2/MMC4FF.json",
#         max_length = 512,
#         min_visual_tokens_per_image = 4,
#         max_visual_tokens_per_image = 512,
#         base_model_dir = "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct",
#         shrink_ratio = 0.9,
#         max_retry = 5,
#         multiple_of = 8
#     )
#     def collate_fn(samples):
#         return samples[0]

#     dataloader = DataLoader(
#         dataset=ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=8,
#         collate_fn=collate_fn
#     )
#     for idx, item in enumerate(dataloader):
#         input_ids = item["input_ids"]
#         loss_mask = item["loss_mask"]
#         segments = []
#         cur_mask = -1
#         for _id, _mask in zip(input_ids[0], loss_mask[0]):
#             if _mask == cur_mask:
#                 segments[-1].append(_id)
#             else:
#                 segments.append([])
#                 cur_mask = _mask
#                 segments[-1].append(_id)
#         print(input_ids)
#         print(loss_mask)
#         for segment in segments:
#             print(ds.processor.tokenizer.decode(segment))
#             print("=" * 10)
#         gg
#         break

def test_parquet_dataset():
    dataset_folder = "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_test/p_date=20250113"
    data_file = shell_hdfs_ls(dataset_folder)

    dataset = ParquetDataset(data_file, num_workers=1)
    for s in dataset:
        print(s)
        break

def test_ChatCompletionVisionParquetDataset():
    init_processes(0, 1)
    processor = Qwen2VLProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14")
    path = "./examples/vlm/configs/stage2_parquet_l2.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config.pop("name")
    dataset_config["num_workers"] = 1
    dataset_config["shuffle_seed"] = int(time.time())
    dataset_config["max_length"] = 2048

    dataset = ChatCompletionVisionParquetDataset(**dataset_config)
    ans = 0
    for s in dataset:
        
        input_ids = s["input_ids"].squeeze()
        loss_mask = s["loss_mask"].squeeze()
        decode_char = processor.tokenizer.convert_ids_to_tokens(input_ids)

        decode_char = [f"\"{word}\"" for word in decode_char]

        assert len(decode_char) == len(loss_mask)
        output = ""
        for i in range(len(decode_char)):
            output+= f"{decode_char[i]}:{loss_mask[i].item()}"
            if i % 8 == 0:
                output += "\n"
            else:
                output += "\t"
        
        print(output)
        print(s["data_source"])
        print("==========================")
        break

if __name__ == "__main__":
    test_ChatCompletionVisionParquetDataset()

