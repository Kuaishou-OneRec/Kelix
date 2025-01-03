from recovlm.data.dataloaders import get_indexed_dataloader
from recovlm.data.datasets import get_webdataset
from recovlm.data.collators import ImageTextPackingCollator

from transformers import AutoProcessor

from torch.utils.data import DataLoader

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
