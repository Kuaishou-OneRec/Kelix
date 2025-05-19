import random
import numpy as np
import torch
import pandas as pd
import pyarrow.parquet as pq

def set_random_seeds(seed):
    """
    设置parquet、pandas、numpy、random、torch的随机种子
    
    参数：
    seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pd.options.mode.chained_assignment = None  # 避免pandas链式赋值警告
    pd.util.hash_pandas_object = lambda x, **kwargs: x  # 固定pandas哈希，确保结果一致
    # parquet本身没有随机操作，这里无需额外设置与随机相关内容

set_random_seeds(0)



import torch

import wids
import logging
import json
import time

from recovlm.utils.common import shell_hdfs_ls
from transformers import AutoTokenizer, AutoProcessor
from recovlm.data.datasets import *
from torch.utils.data import DataLoader
from recovlm.data.datasets import ChatCompletionDataset, ImageTextPairDatasetWithPacking, ChatCompletionVisionDataset, ParquetDataset, ChatCompletionVisionDpoParquetDataset
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0, \
    get_optimizer_grouped_parameters, dist_reduce_dict, Timer, heart_beat
from tests.utils import init_processes
import torch.distributed as dist
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
    processor = Qwen2VLProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct")
    path = "./examples/vlm/configs/dpo_wenjuan_0210_10w_test.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config.pop("name")
    dataset_config["num_workers"] = 8
    dataset_config["shuffle_seed"] = int(time.time())
    dataset_config["n_local_shuffle_files_window"] = 1
    dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/huqigen/recovlm_dataset/wenjuan_sft/0210_11w_cot_v2/photo_0210_11w_sft_data-train-00000-of-02048.parquet"]

    dataset = ChatCompletionVisionDpoParquetDataset(**dataset_config)
    ans = 0
    def collate_fn(samples):
        return samples[0]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    for iteration, batch in enumerate(dataloader):
        # pass
        chosen_inputs, rejected_inputs = batch
        input_ids = chosen_inputs["input_ids"].squeeze()
        loss_mask = chosen_inputs["loss_mask"].squeeze()
        decode_char = processor.tokenizer.convert_ids_to_tokens(input_ids)

        decode_char = [f"\"{word}\"" for word in decode_char]

        assert len(decode_char) == len(loss_mask)
        output = "=======start======="
        for i in range(len(decode_char)):
            output+= f"{decode_char[i]}:{loss_mask[i].item()}"
            if i % 8 == 0:
                output += "\n"
            else:
                output += "\t"
        
        print(output)
        print(chosen_inputs["data_source"])
        print("==========================")
        break

def gather_by_group(dataloader, group, buffer_size=1):
    buffer = []
    for batch in dataloader:
        buffer.append(batch)
        if len(buffer) >= buffer_size:
            yield from gather_batches(buffer, group)
            buffer = []
    if len(buffer) > 0:
        yield from gather_batches(buffer, group)

def gather_batches(buffer, group):
    world_size = dist.get_world_size(group)
    if world_size > 1:
      with Timer("Gather batches"):
        gathered_batches = [None for _ in range(world_size)]
        dist.all_gather_object(
            object_list=gathered_batches, obj=buffer,
            group=group
        )

      gathered_batches = sum(gathered_batches, [])
    else:
      gathered_batches = buffer
    print_rank_0(f"Num batches: {len(gathered_batches)}")
    return gathered_batches


def test_InternVLParquetDataset():
    init_processes(0, 1)
    torch.manual_seed(0)
    np.random.seed(0)

    processor = AutoProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/InternVL3-2B", trust_remote_code=True)
    path = "/llm_reco/chuchenglong/work_space/recovlm/examples/vlm/configs/internvl/2b_internvl_stage2.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config.pop("name")
    dataset_config["num_workers"] = 1
    dataset_config["shuffle_seed"] = int(time.time())
    dataset_config["max_length"] = 16000
    dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/pt/0421/stage2_ccl_v3_0425/_prepared/0/prep-0-5f8467a5aa2c472d9c31bbb81356540f.parquet"]
    # viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v3/rank-11-4ef695ac-2336-11f0-b166-946daee9184a.parquet
    dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v3/rank-11-4ef695ac-2336-11f0-b166-946daee9184a.parquet","viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-code/train_v1/rank-12-2a107142-2438-11f0-ba28-946daee91688.parquet"][1:]
    dataset_config["cut_to_pad"] = True
    # viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct/rank-0-0098b494-d499-11ef-9d06-946daee91052.parquet
    # dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct/rank-0-0098b494-d499-11ef-9d06-946daee91052.parquet"]

    dataset = InternVLChatCompletionVisionParquetDataset(**dataset_config)
    ans = 0
    def collate_fn(samples):
        return samples[0]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    for iteration, batch in enumerate(dataloader):
        for k, v in batch.items():
            try:
                print(k, v.shape, v.dtype, str(v)[:100])
            except:
                print(k, v)
            print("=" * 10)
        if iteration == 200: break



def test_keye_datasets():
    init_processes(0, 1)
    torch.manual_seed(0)
    np.random.seed(0)
    # processor = AutoProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/InternVL3-2B", trust_remote_code=True)
    path = "/llm_reco/lingzhixin/recovlm_qw0510/recovlm/examples/vlm/qwen3navit/debug_qwen3navit_8B256_nols2.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config.pop("name")

    # n_local_shuffle_files_window=3
    dataset_config["n_local_shuffle_files_window"] = 1
    dataset_config["num_workers"] = 1
    dataset_config["shuffle_seed"] = int(time.time())
    dataset_config["max_length"] = 1024*12
    dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/pt/0421/stage2_ccl_v3_0425/_prepared/0/prep-0-5f8467a5aa2c472d9c31bbb81356540f.parquet"]
    dataset_config["cut_to_pad"] = True
    # viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v3/rank-11-4ef695ac-2336-11f0-b166-946daee9184a.parquet
    # dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v3/rank-11-4ef695ac-2336-11f0-b166-946daee9184a.parquet","viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-code/train_v1/rank-12-2a107142-2438-11f0-ba28-946daee91688.parquet"][1:]
    
    # viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct/rank-0-0098b494-d499-11ef-9d06-946daee91052.parquet
    # dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct/rank-0-0098b494-d499-11ef-9d06-946daee91052.parquet"]

    dataset_keyi = ChatCompletionVisionParquetDataset_keye(**dataset_config)

    def collate_fn(samples):
        return samples[0]

    dataloader_keye = DataLoader(
        dataset=dataset_keyi,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    for iteration, batch in enumerate(dataloader_keye):
        print_input_info(batch, "keye")
        # break
    
    with open("./test_keye_datasets_keye.txt", "w") as f:
        f.write(print_input_info(batch, return_str=True))
        f.write(str(batch["input_ids"])[:10000] + '\n' + str(batch["input_ids"])[-10000:])
        f.write(str(batch["position_ids"])[:10000] + '\n' + str(batch["input_ids"])[-10000:])
        f.write(str(batch["pixel_values"])[:10000] + '\n' +str(batch["input_ids"])[-10000:])
        f.write(str(batch["cu_seqlens"])[:10000] + '\n' + str(batch["input_ids"])[-10000:])

    print("=" * 20)
    path = "/llm_reco/lingzhixin/recovlm_qw0510/recovlm/examples/vlm/qwen3navit/debug_qwen3navit_8B256_nols.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config.pop("name")

    # n_local_shuffle_files_window=3
    dataset_config["n_local_shuffle_files_window"] = 1
    dataset_config["num_workers"] = 1
    dataset_config["shuffle_seed"] = int(time.time())
    dataset_config["max_length"] = 1024*12
    dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/pt/0421/stage2_ccl_v3_0425/_prepared/0/prep-0-5f8467a5aa2c472d9c31bbb81356540f.parquet"]
    dataset_config["cut_to_pad"] = True
    # viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v3/rank-11-4ef695ac-2336-11f0-b166-946daee9184a.parquet
    # dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-text-code-block/train_v3/rank-11-4ef695ac-2336-11f0-b166-946daee9184a.parquet","viewfs://hadoop-lt-cluster/home/reco_wl/mpi/lingzhixin/recovlm/tools/data_helpers/scripts/convert_megamath/megamath-code/train_v1/rank-12-2a107142-2438-11f0-ba28-946daee91688.parquet"][1:]
    
    # viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct/rank-0-0098b494-d499-11ef-9d06-946daee91052.parquet
    # dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage2/Wanjuan_reconstruct/rank-0-0098b494-d499-11ef-9d06-946daee91052.parquet"]

    dataset_keyi = ChatCompletionVisionParquetDataset_keye(**dataset_config)
    dataset_navit = ChatCompletionVisionParquetDataset_navit(**dataset_config)

    dataset_navit = DataLoader(
        dataset=dataset_navit,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    for iteration, batch in enumerate(dataset_navit):
        print_input_info(batch, "navi")
        break

    with open("./test_keye_datasets_navi.txt", "w") as f:
        f.write(print_input_info(batch, return_str=True))
        f.write(str(batch["input_ids"])[:10000] + '\n' + str(batch["input_ids"])[-10000:])
        f.write(str(batch["position_ids"])[:10000] + '\n' + str(batch["input_ids"])[-10000:])
        f.write(str(batch["pixel_values"])[:10000] + '\n' +str(batch["input_ids"])[-10000:])
        f.write(str(batch["cu_seqlens"])[:10000] + '\n' + str(batch["input_ids"])[-10000:])



if __name__ == "__main__":
    test_keye_datasets()

