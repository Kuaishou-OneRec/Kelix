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

def test_chat_vision():
    init_processes(0, 1)
    path = "./examples/vlm/configs/stage2_mix_v1.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset = dataset_config.pop("name")
    dataloader = get_dataloader(
        name=dataset,
        **dataset_config)
    for idx, item in enumerate(dataloader):
        # print(idx)
        # print(item)
        # break
        item_id = id(item)
        if idx > 10:
            break


def test_chat_vision_parquet():
    init_processes(0, 1)
    path = "./examples/vlm/configs/stage2_parquet.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset = dataset_config.pop("name")
    dataloader1 = get_dataloader(
        name=dataset,
        **dataset_config)
    dataloader2 = get_dataloader(
        name=dataset,
        **dataset_config)
    iter1 = iter(dataloader1)
    for idx in range(10):
        batch_data = next(iter1)
        if idx % 5 == 0:
            state_dict = dataloader1.state_dict()
            print(f"step={idx}, {json.dumps(state_dict)}")

    print("load_state_dict")
    state_dict = dataloader1.state_dict()
    dataloader2.load_state_dict(state_dict)
    iter2 = iter(dataloader2)
    print("finish state load")
    for idx in range(10):
        batch_data1 = next(iter1)
        batch_data2 = next(iter2)
        print(f"{batch_data1=}, {batch_data2=}")

        if idx % 10 == 0:
            state_dict1 = dataloader1.state_dict()
            state_dict2 = dataloader2.state_dict()
            print(f"{json.dumps(state_dict1)}")
            print(f"{json.dumps(state_dict2)}")
    
def verify_dataset():
    init_processes(0, 1)
    path = "examples/vlm/configs/stage2_parquet_nointerleave.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset = dataset_config.pop("name")
    dataset_config["num_workers"] = 1
    dataloader = get_dataloader(
        name=dataset,
        **dataset_config)
    
    result = []
    iter_data = iter(dataloader)
    for idx in range(200):
        batch_data = next(iter_data)
        result.append(batch_data)
    
    import pickle
    with open("/code/after.pkl", "wb") as fp:
        pickle.dump(result, fp)

if __name__ == "__main__":
    # verify_dataset()
    import pickle
    import torch
    with open("/code/before.pkl", "rb") as fp:
        before = pickle.load(fp)
    
    with open("/code/after.pkl", "rb") as fp:
        after = pickle.load(fp)
    
    for i in range(len(after)):
        a = after[i]
        b = before[i]
        for k in a:
            assert k in b
            if isinstance(a[k], torch.Tensor):
                assert a[k].equal(b[k])
            else:
                assert a[k] == b[k]