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


"""
有cut
图片token总数：372,864
图片token均值：≈12,857.38
图片token方差：≈1,500,000
样本token总数：464,000

没有cut
"""

def generate_test_dataframe(num_samples_perfile=400, num_files=3000, n_texts=250, fake_dir="/llm_reco/lingzhixin/recovlm_0427compile/recovlm/tests/test_fake_datasets/generate_test_dataframe/buffer/"):
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
        json.dump(all_filenames, f, indent=4)
    print(json_path)
    return json_path





def test_InternVLParquetDataset():
    init_processes(0, 1)
    from transformers import AutoTokenizer, AutoProcessor
    from recovlm.data.datasets import InternVLChatCompletionVisionParquetDataset

    json_file = generate_test_dataframe()
    with open(json_file, "r") as f:
        all_fn_json = json.load(f)

    _exp_config_fn = "/llm_reco/lingzhixin/recovlm_0427compile/recovlm/tests/test_fake_datasets/generate_test_dataframe/_exp_config.json"
    with open(_exp_config_fn, 'r') as f:
        _exp_config = json.loads(f.read())
        _exp_config["sources"] = json_file
    
    exp_config_fn = "/llm_reco/lingzhixin/recovlm_0427compile/recovlm/tests/test_fake_datasets/generate_test_dataframe/exp_config.json"
    with open(exp_config_fn, 'w') as f:
        json.dump(_exp_config, f, indent=4)
    
    with open(exp_config_fn, 'r') as f:
        exp_config = json.loads(f.read())


    processor = AutoProcessor.from_pretrained("/llm_reco_ssd/zhouyang12/models/InternVL3-2B", trust_remote_code=True)
    path = "/llm_reco/chuchenglong/work_space/recovlm/examples/vlm/configs/internvl/2b_internvl_stage2.json"
    # with open(path, encoding="utf-8") as f:
    #     dataset_config = json.loads(f.read())
    # dataset_config.pop("name")
    # dataset_config["num_workers"] = 1
    # dataset_config["shuffle_seed"] = int(time.time())
    # dataset_config["max_length"] = 16000
    # dataset_config["sources"] = json_file # all_filenames # ["viewfs://hadoop-lt-cluster/home/reco_wl/mpi/chuchenglong/pt/0421/stage2_ccl_v3_0425/_prepared/0/prep-0-5f8467a5aa2c472d9c31bbb81356540f.parquet"]

    dataset = InternVLChatCompletionVisionParquetDataset(**exp_config)

    def collate_fn(samples):
        return samples[0]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=exp_config["num_workers"],
        collate_fn=collate_fn
    )
    for iteration, batch in enumerate(dataloader):
        # for k, v in batch.items():
        k = ''
        print(batch.keys(), 31211)
        v = batch
        try:
            # print(11112312, v["input_ids"].flatten())
            # print(k, v.shape, v.dtype, str(v)[:100])
            num_im_tokens = (v["input_ids"] == 151667).int().sum().item()
            print(f'im tokens = {num_im_tokens}/{len(v["input_ids"].flatten())}')
        except Exception as e:
            print(e)
            # print(k, v)
        print("=" * 10)
        if iteration == 5: break
        

if __name__ == "__main__":
    test_InternVLParquetDataset()

