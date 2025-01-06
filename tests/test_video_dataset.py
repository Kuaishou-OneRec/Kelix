from recovlm.data.datasets import VisionTextDatasetWithPacking
import webdataset as wds
import torchvision
import json
import os
import time
import math
from tqdm import tqdm
# from recovlm.utils.qwen_vl_utils import process_vision_info
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from torch.utils.data import DataLoader

model_dir = "/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14"

# pic_data_path = "/llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/00000.tar"
# # video_data_path = "/llm_reco/zhangzixing/video_webdataset_demo/00000.tar"
# video_data_path = "/llm_reco/zhangzixing/video_webdataset_demo/test/abcdv.tar"


# source = "/llm_reco_ssd/zhangzixing/dataset/ShareGPT4Video_other/sharegpt4video_40k.webdataset/index.json"
# # urls = []
# # with open(source) as fp:
# #     index = json.loads(fp.read())["shardlist"]
# #     urls.extend([
# #           os.path.join(os.path.dirname(source), item["url"]) \
# #             for item in index])

# # def warn_and_continue(e):
# #     print("Warning: skipping a corrupt sample.", e)
# # dataset = wds.WebDataset(
# #         urls,
# #         handler=warn_and_continue,
# #         resampled=True,
# #         shardshuffle=True,
# #         cache_dir="/tmp/_wids_cache",
# #         nodesplitter=wds.split_by_node,
# #         workersplitter=wds.split_by_worker
# #     )

# # def test_processor_demo(data):
# #     processor = Qwen2VLProcessor.from_pretrained(model_dir)
# #  prompt = [
# #         {
# #             "role": "user",
# #             "content": [
# #                 {
# #                     "type": "video",
# #                     "video": data["mp4"],
# #                     "fps": 1.0,
# #                 },
# #                 {"type": "text", "text": "Describe this video."},
# #             ],
# #         }
# #     ]
# #     text = processor.apply_chat_template(
# #         [prompt], tokenize=False, add_generation_prompt=True
# #     )
# #     image_inputs, video_inputs = process_vision_info(prompt)
# #     inputs = processor(
# #           text=text,
# #           images=image_inputs,
# #           videos=video_inputs,
# #           padding=True,
# #           padding_side="left",
# #           return_tensors="pt",
# #     )
# #     response = None
# #     return inputs, response


# # for d in dataset:
# #     print(json.loads(d["json"])["captions"][-1]["content"])
# #     inputs, response = test_processor_demo(d)
# #     print(inputs)
# #     break

########### debug dataset ###########
processor = Qwen2VLProcessor.from_pretrained(model_dir)

### 使用WebDataset + greedy padding
dataset = VisionTextDatasetWithPacking(
    sources = "/llm_reco_ssd/luoxinchen/dataset/kwai_video_caption/20250105/index.json",
    processor = processor,
    max_length = math.ceil(480 * 480 / (28 ** 2) * (120 / 2)), # 480p, patch=14, 2 × 2 tokens into a single token, total_len=120s, merge 2 frame
    min_visual_tokens = 64,
    max_visual_tokens = 512,
    fps=2.0,
    min_frame_visual_tokens = math.ceil(480 * 10 / (28 ** 2)),
    max_frame_visual_tokens = math.ceil(480 * 480 / (28 ** 2)),
    spatial_merge_size = 2,
    image_token_id = 151655,
    video_token_id = 151656,
    vision_start_token_id = 151652,
    patch_size = 14,
    shrink_ratio = 0.9,
    max_retry = 5,
    multiple_of = 8
)

dataloader = DataLoader(
    dataset=dataset,
    shuffle=False,
    batch_size=1,
    num_workers=8,
    collate_fn=lambda x: x[0]
)

start_ts = time.time()
ans = 0
for batch in tqdm(dataloader):
    batchid = id(batch)
    ans += 1
    if ans % 100 == 0:
        print(f"{ans=}, perf={100 / (time.time() - start_ts)}")
        start_ts = time.time()


##########debug webdataset########

# source = "/llm_reco_ssd/luoxinchen/dataset/Stage2/the_cauldron/index.json"
# source = "/llm_reco_ssd/zhangzixing/dataset/ShareGPT4Video_other/sharegpt4video_40k.webdataset/index.json"
# urls = []
# with open(source) as fp:
#     index = json.loads(fp.read())["shardlist"]
#     urls.extend([
#           os.path.join(os.path.dirname(source), item["url"]) \
#             for item in index])
# dataset = wds.WebDataset(
#         urls,
#         handler=wds.warn_and_continue,
#         resampled=True,
#         shardshuffle=True,
#         cache_dir="/tmp/_wids_cache",
#         nodesplitter=wds.split_by_node,
#         workersplitter=wds.split_by_worker
#     ).decode("pil", handler=wds.warn_and_continue)

# for d in dataset:
#     print(d)
#     break