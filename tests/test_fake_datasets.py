import os
import torch

import wids
import logging
import json
import time

from recovlm.utils.common import shell_hdfs_ls

from torch.utils.data import DataLoader
from recovlm.data.datasets import ChatCompletionDataset, ImageTextPairDatasetWithPacking, ChatCompletionVisionDataset, ParquetDataset, ChatCompletionVisionParquetDataset, ChatCompletionVisionDpoParquetDataset
from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from recovlm.utils.common import set_random_seed, to_cuda, print_rank_0, \
    get_optimizer_grouped_parameters, dist_reduce_dict, Timer, heart_beat
from tests.utils import init_processes
import torch.distributed as dist


TOKENIZER = "/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct"


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
    from transformers import AutoTokenizer, AutoProcessor
    from recovlm.data.datasets import InternVLChatCompletionVisionParquetDataset
    processor = AutoProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/InternVL3-2B", trust_remote_code=True)
    path = "/llm_reco/chuchenglong/work_space/recovlm/examples/vlm/configs/internvl/2b_internvl_stage2.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config.pop("name")
    dataset_config["num_workers"] = 1
    dataset_config["shuffle_seed"] = int(time.time())
    dataset_config["max_length"] = 16000
    dataset_config["sources"] = ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/pt/0421/stage2_ccl_v3_0425/_prepared/0/prep-0-5f8467a5aa2c472d9c31bbb81356540f.parquet"]

    dataset = InternVLChatCompletionVisionParquetDataset(cut_to_pad=True, **dataset_config)
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
        if iteration == 20: break
        
        
'''
    {
      "id": 151665,
      "content": "<img>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
        {
      "id": 151667,
      "content": "<IMG_CONTEXT>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
'''

if __name__ == "__main__":
    test_InternVLParquetDataset()

