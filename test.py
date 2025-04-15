import torch
from transformers import AutoTokenizer, AutoModel
from recovlm.models.intern_vl_3 import InternVLChatModel,InternVLChatConfig
from recovlm.training.common import set_default_dtype
from recovlm.data.dataloaders_v2 import get_dataloader
#from recovlm.data.dataloaders import get_dataloader
from torch.utils.data import DataLoader
import json
import itertools
from typing import Tuple
from rich import print
import time
import torch
import random
import numpy as np
from pathlib import Path
from transformers import set_seed as set_transformers_seed
import torch.distributed as dist
import pickle
import traceback
import subprocess
import os
try:
    from infra.perflog import create_perf_context
    INFRA_AVAILABLE = True
except ImportError:
    INFRA_AVAILABLE = False
    print("Warning: infra module not available, heart_beat functionality will be disabled")
import pyarrow.parquet as pq

# def print_rank_n(*msg, rank=0):
#   if dist.get_rank() == rank:
#     print(*msg)

# def print_rank_0(*msg):
#   print_rank_n(*msg, rank=0)

# def get_worker_info():
#   worker = 0
#   num_workers = 1
#   try:
#     import torch.utils.data

#     worker_info = torch.utils.data.get_worker_info()
#     if worker_info is not None:
#       worker = worker_info.id
#       num_workers = worker_info.num_workers
#   except ModuleNotFoundError:
#     pass
#   return worker, num_workers

# def get_world_size_and_rank() -> Tuple[int, int]:
#     """Function that gets the current world size (aka total number
#     of ranks) and rank number of the current process in the default process group.

#     Returns:
#         Tuple[int, int]: world size, rank
#     """
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         return torch.distributed.get_world_size(), torch.distributed.get_rank()
#     elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         return int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
#     else:
#         return 1, 0


# def load_parquet_file(fn: str, retry=5, max_cache_files=10) -> pq.ParquetFile:
#     """Load a parquet file, with fallback to local cache if HDFS read fails.
    
#     Args:
#         fn (str): Path to parquet file, can be HDFS path
#         retry (int): Number of retries
#         max_cache_files (int): Maximum number of files to keep in cache
        
#     Returns:
#         pq.ParquetFile: Loaded parquet file object
        
#     Raises:
#         Exception: If both HDFS and local cache loading fail
#     """
#     import hashlib

#     def calculate_text_hash(text):
#         # 创建一个 SHA-256 哈希对象
#         hash_object = hashlib.sha256()
#         # 将文本编码为字节串并更新哈希对象
#         hash_object.update(text.encode('utf-8'))
#         # 获取十六进制表示的哈希值
#         hash_hex = hash_object.hexdigest()
#         return hash_hex

#     worker_id = get_worker_info()[0]
#     rank_id = get_world_size_and_rank()[1]

#     cache_dir = f'/code/dataset_cache/{worker_id}_{rank_id}'
#     os.makedirs(cache_dir, exist_ok=True)
#     filename = os.path.basename(fn)

#     cache_fn = os.path.join(cache_dir, str(calculate_text_hash(fn)) + '_' + filename)
#     import time

#     def clean_cache_if_needed():
#         files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))]
#         if len(files) > max_cache_files:
#             files.sort(key=os.path.getctime)
#             for fn in files[:max_cache_files//2]:
#                 print(f"Removing old cached file: {fn}")

#     for r in range(retry):
#         print(f"retrying for fn={fn}/{cache_fn}")
#         try:
#             if os.path.exists(cache_fn):
#                 res = pq.ParquetFile(cache_fn)
#             else:
#                 res = pq.ParquetFile(fn)
#             return res
        
#         except Exception as e:          
#             # Try to download from HDFS
#             try:
#                 clean_cache_if_needed()  # Clean cache before downloading new file
#                 cmd = f'hadoop fs -get {fn} {cache_fn}'
#                 os.system(cmd)
#                 res = pq.ParquetFile(cache_fn)
#                 return res
#             except Exception as e2:
#                 time.sleep(2 + np.random.randint(0, 5))
                
#                 if r == retry - 1:
#                     raise Exception(f"Failed to load parquet file from both original path and cache.\nOriginal error: {e}\nCache error: {e2}")

# if __name__=='__main__':
#     path = "/llm_reco/penghao03/intern-vl/InternVL3-2B"

#     # debug dataloader
#     # dataset_config='examples/vlm/configs/debug7b_fsdp_3p_intern_vl.json'
#     # # #dataset_config='./examples/vlm/configs/stage1_parquet_ocr_0207.json'
#     # with open(dataset_config, encoding="utf-8") as f:
#     #     dataset_config = json.loads(f.read())
#     # dataset = dataset_config.pop("name")

#     # dataloader = get_dataloader(name=dataset,**dataset_config)
#     # for batch in dataloader:
#     #     print(batch)
#     # json_path = '/llm_reco_ssd/zhangzixing/dataset/hdfs_data/recovlm_dataset_shuffle.ocr.0208.json'
#     # with open(json_path, encoding="utf-8") as f:
#     #      data = json.loads(f.read())
#     # print(data)
#     #parquet_path = 'viewfs://hadoop-lt-cluster/home/reco_6/mpi/lingzhixin/recovlm/parse_dataparse_data_to_parquet_vvabs_cot_v2/train_data_1w/cot_trial1/good_ids1_use_cot1-train-00060-of-00256.parquet'
#     # parquet_path='good_ids1_use_cot1-train-00045-of-00256.parquet'
#     # # #print(os.path.exists(parquet_path))
#     # # #data = load_parquet_file(parquet_path)
#     # #data = pq.read_table(parquet_path).to_pandas()
#     # data = pq.ParquetFile(parquet_path)
#     # print(data)
#     # print(data)
#     # class Parent:
#     #     def __init__(self):
#     #         self.method()  # 父类初始化时调用 self.method()
        
#     #     def method(self):
#     #         print("Parent method")

#     # class Child(Parent):
#     #     def __init__(self):
#     #         super().__init__()  # 调用父类的 __init__
        
#     #     def method(self):  # 子类重写父类方法
#     #         print("Child method")

#     # 创建子类实例
#     #child = Child()

#     # 尝试分批次读取
#     # batch_size = 10
#     # table = pq.read_table(parquet_path, use_threads=False)
#     # for i in range(0, table.num_rows, batch_size):
#     #     batch = table.slice(i, batch_size)
#     #     print(f"成功读取批次: {i}-{i+batch_size}")




# # with set_default_dtype(torch.bfloat16), torch.device("meta"):
# #     model = InternVLChatModel.from_pretrained(
# #             path,
# #             use_flash_attn=True)
base_model_dir = '/llm_reco/penghao03/intern-vl/InternVL3-2B'

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
model_config = InternVLChatConfig.from_pretrained(base_model_dir)
path_size = model_config.vision_config.patch_size
image_size = model_config.force_image_size

print(tokenizer.special_tokens_map)



# # for tensor in itertools.chain(model.parameters(), model.buffers()):
# #     assert tensor.device == torch.device("meta")
# # print(model.language_model.model.rotary_emb.inv_freq)
# # layer_num = 24
# # drop = 0.1
# # dpr = [x.item() for x in torch.linspace(0, drop, layer_num)]
# # print(dpr)
# # x_dpr = [drop * i / max(1, layer_num - 1) for i in range(layer_num)]
# # print(x_dpr)
# # print(model)
# # for n,p in model.named_parameters():
# #     print(n,p.shape)

# # model_path = '/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct'
# # qwen_model = AutoModel.from_pretrained(
# #         model_path, _attn_implementation="flash_attention_2",
# #         use_cache=False
# # )
# # print(qwen_model)
# # for n,p in qwen_model.named_parameters():
# #     print(n,p.shape)


# # import torch.nn as nn
# # import torch.distributed as dist
# # import torch.nn.functional as F
# # import numpy as np

# # from pathlib import Path
# # from torch.utils.data import DataLoader
# # from torch.utils.tensorboard import SummaryWriter

# # # from transformers import AutoTokenizer, AutoProcessor
# # from recovlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
# # from recovlm.models.qwen2_vl import Qwen2VLForConditionalGeneration

# # #inten-vl
# # from recovlm.models.intern_vl_3 import InternVLChatModel


# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info


# path = '/llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-7B-Instruct'
# # default: Load the model on the available device(s)
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     path, torch_dtype="auto", device_map="auto"
# # )

# # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     "Qwen/Qwen2.5-VL-7B-Instruct",
# #     torch_dtype=torch.bfloat16,
# #     attn_implementation="flash_attention_2",
# #     device_map="auto",
# # )

# # default processer
# processor = AutoProcessor.from_pretrained(path)

# # The default range for the number of visual tokens per image in the model is 4-16384.
# # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# # min_pixels = 256*28*28
# # max_pixels = 1280*28*28
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "/llm_reco/penghao03/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# print(text)
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")
# print(inputs.keys())
# add_bos_token = getattr(processor.tokenizer, 'add_bos_token', False)
# print(add_bos_token)

# # # Inference: Generation of the output
# # generated_ids = model.generate(**inputs, max_new_tokens=128)
# # generated_ids_trimmed = [
# #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# # ]
# # output_text = processor.batch_decode(
# #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# # )
# # print(output_text)