from recovlm.data.dataloaders import get_indexed_dataloader, get_dataloader
from recovlm.data.collators import ImageTextPackingCollator

import json
import os

from transformers import AutoProcessor

from torch.utils.data import DataLoader
from tests.utils import init_processes

# def test_web_dataloader():
#     dataset = get_webdataset(
#         "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json,/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json",
#     )
#     processor = AutoProcessor.from_pretrained(
#         "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")
#     collator = ImageTextPackingCollator(
#         processor = processor,
#         max_length = 1024,
#         min_visual_tokens = 64,
#         max_visual_tokens = 512,
#         max_text_length = 512,
#         spatial_merge_size = 2,
#         image_token_id = 151655,
#         video_token_id = 151656,
#         vision_start_token_id = 151652,
#         patch_size = 14,
#         shrink_ratio = 0.9,
#         max_retry = 5,
#         multiple_of = 8
#     )
#     for s in DataLoader(dataset, batch_size=8, num_workers=8, collate_fn=collator):
#         print(s)
#         gg


# def test_chat_vision():
#     init_processes(0, 1)
#     path = "./examples/vlm/configs/video.json"
#     with open(path, encoding="utf-8") as f:
#         dataset_config = json.loads(f.read())
#     dataset = dataset_config.pop("name")
#     dataloader = get_dataloader(
#         name=dataset,
#         **dataset_config)
#     # dataloader = get_dataloader(
#     #     name=dataset, num_workers=1, need_padding=True,
#     #     **dataset_config)
#     for idx, item in enumerate(dataloader):
#         print(idx)
#         # print(item)
#         # break
#         item_id = id(item)
#         if idx > 1:
#             break

if __name__ == "__main__":
    test_chat_vision()