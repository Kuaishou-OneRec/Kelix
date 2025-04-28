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



import pandas as pd
import uuid
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import tqdm

def generate_test_dataframe(num_samples_perfile=100, num_files=10, n_texts=1000, fake_dir="/llm_reco/lingzhixin/recovlm_0427compile/recovlm/tests/test_fake_datasets/generate_test_dataframe/buffer/"):
    """
    生成测试用DataFrame，结构如下：
    - images: {"0.jpg": base64编码的1x1像素图片}
    - videos: null
    - source: "caps_fusion_v3"
    - messages: null
    - segments: [{"type": "image", "image": "0.jpg"}, ...]
    - metadata: null
    - uuid: 唯一ID字符串
    """
    os.makedirs(fake_dir, exist_ok=True)
    all_filenames = []
    # 生成1x1像素的base64图片
    def generate_base64_image():
        img = Image.fromarray(np.zeros((50,50, 3), dtype=np.uint8))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    # 生成测试数据
    data = []
    for _ in tqdm.tqdm(range(num_samples_perfile)):
        img_base64 = generate_base64_image()
        data.append({
            "images": {"0.jpg": img_base64},
            "videos": None,
            "source": "caps_fusion_v3",
            "messages": None,
            "segments": [
                {"type": "image", "image": "0.jpg"},
                {"type": "text", "text": "d" * n_texts}
            ],
            "metadata": None,
            "uuid": str(uuid.uuid4())
        })
    
    # 创建DataFrame并设置索引
    df = pd.DataFrame(data)
    df.index.name = "index"
    
    for i in tqdm.tqdm(range(num_files)):
        filename = f"{i}.parquet"
        df.to_parquet(os.path.join(fake_dir, filename), engine="fastparquet")
        all_filenames.append(os.path.join(fake_dir, filename))
    print("save parquet file successfully")

    json_path = os.path.join(fake_dir, "all_filenames.json")
    with open(json_path, "w") as f:
        import json
        json.dump(all_filenames, f)
    print(json_path)
    return all_filenames

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

    json_file = generate_test_dataframe()
    with open(json_file, "r") as f:
        all_filenames = json.load(f)

    processor = AutoProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/InternVL3-2B", trust_remote_code=True)
    path = "/llm_reco/chuchenglong/work_space/recovlm/examples/vlm/configs/internvl/2b_internvl_stage2.json"
    with open(path, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config.pop("name")
    dataset_config["num_workers"] = 1
    dataset_config["shuffle_seed"] = int(time.time())
    dataset_config["max_length"] = 16000
    dataset_config["sources"] = all_filenames # ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/pt/0421/stage2_ccl_v3_0425/_prepared/0/prep-0-5f8467a5aa2c472d9c31bbb81356540f.parquet"]

    dataset = InternVLChatCompletionVisionParquetDataset(cut_to_pad=True, **dataset_config)

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
        if iteration == 5: break
        

if __name__ == "__main__":
    test_InternVLParquetDataset()

