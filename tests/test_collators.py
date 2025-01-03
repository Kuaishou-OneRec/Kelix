import os 
import torch
from torch.utils.data import DataLoader
from recovlm.data.collators import ImageTextPackingCollator
from recovlm.data.dataloaders import get_indexed_dataloader

from transformers import AutoProcessor
from tqdm import tqdm


def test_image_text_packing_collator():
    # TODO: complete tests
    sources = [
        "/llm_reco_ssd/luoxinchen/dataset/datacomp/large/index.json",
        "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json"
    ]
    processor = AutoProcessor.from_pretrained(
        "/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")
    collator = ImageTextPackingCollator(
        processor = processor,
        max_length = 1024,
        min_visual_tokens = 64,
        max_visual_tokens = 512,
        max_text_length = 512,
        spatial_merge_size = 2,
        image_token_id = 151655,
        video_token_id = 151656,
        vision_start_token_id = 151652,
        patch_size = 14,
        shrink_ratio = 0.9,
        max_retry = 5,
        multiple_of = 8
    )
    dataloader = get_indexed_dataloader(
        sources=sources,
        processor=processor,
        batch_size=128,
        num_workers=4,
        shuffle=True,
        max_length=1024,
        rank=1,
        collator=collator)
    for s in tqdm(dataloader):
        break
        # for key in s:
        #     print(key, type(s[key]))
