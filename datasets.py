import os
import re
import json
import math
import torch
import pandas as pd
import tarfile
import torch.distributed as dist

from torch.utils.data import IterableDataset, Dataset, DataLoader


import random
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from processing_qwen2_vl import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import glob

from utils import get_world_size, is_rank_0

RESPONSE_TEMPLATE = "{% for message in messages %}{{message['content'] + '<|im_end|>'}}{% endfor %}"

def extract_tar_file(tar_file_path, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=destination_path)
        print(f"Files extracted to {destination_path}")

def shard(data_dir, rank=0, world_size=1, shuffle=True, reshard=False):
    shard_dir = os.path.join(data_dir, f"sharded_{world_size}")
    if is_rank_0():
        if reshard or (not os.path.exists(shard_dir)):
            files = glob.glob(os.path.join(data_dir, "*.parquet"))
            num_files = len(files)
            num_files_per_rank = num_files // world_size
            print(f"Shard {num_files} files to {world_size} parts.")
            random.shuffle(files)
            os.makedirs(shard_dir)
            for rank in range(world_size):
                start = rank * num_files_per_rank
                end = (rank + 1) * num_files_per_rank if rank < (world_size - 1) else num_files
                with open(os.path.join(shard_dir, f"rank_{rank}.txt"), "w", encoding="utf-8") as f:
                    for file in files[start:end]:
                        f.write(file + "\n")
    else:
        dist.barrier()
    return os.path.join(shard_dir, f"rank_{rank}")

class MscocoDataset(IterableDataset):
    def __init__(self, source, world_size, rank):
        super(MscocoDataset).__init__()
        self.meta = shard(source, world_size=world_size, rank=rank)
        self.file_list = []
        with open(self.meta) as f:
            for line in f:
                self.file_list.append(line.strip())

    def tokenize(self, text, img):
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        response_messages = {"role": "assistant", "content": text}
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        response = processor.tokenizer.apply_chat_template(
            response_messages, chat_template="", add_generation_prompt=False)
        label_mask = len(inputs["input_ids"]) * 0 + len(response) * 1
        inputs["label_mask"] = label_mask
        return inputs

    def build_iter(self, files):
        def iterable():
            for file in files:
                tar_file = re.sub(r".parquet$", ".tar", file)
                img_dir = re.sub(r".parquet$", "", file)
                extract_tar_file(tar_file, img_dir)
                df = pd.read_parquet(file)
                for text, img, status in zip(df["caption"], df["key"], df["status"]):
                    if status != "success":
                        continue
                    tokenized = self.tokenize(text, image)
                    yield tokenized
        return iterable()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.file_list)
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.file_list)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return self.build_iter(self.file_list[iter_start:iter_end])

messages = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe"},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K/GCC_train_001885390.jpg",
                },
                {
                    "type": "image",
                    "image": "/llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K/GCC_train_001970366.jpg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
]

response_messages = [
    [
        {
            "content": "你好大傻逼，hello"
        }
    ],
    [
        {
            "content": "hello"
        }
    ]
]

class LLaVA_CC3M_Dataset(Dataset):
    def __init__(self, source, processor_path, max_length=None):
        super(LLaVA_CC3M_Dataset).__init__()
        self.source = source
        with open(os.path.join(self.source, "chat.json"), encoding="utf-8") as f:
            self.sessions = json.loads(f.read())
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.tokenizer =  AutoTokenizer.from_pretrained(processor_path, use_fast=False)
        self.tokenizer.padding_side = "right"
        self.max_length = max_length
        if not self.max_length:
            self.max_length = self.tokenizer.model_max_length

    def __getitem__(self, index):
        session = self.sessions[index]
        return session

    def build_collate_fn(self):
        # TODO: SUPPORT TRUNCATE
        def collate_fn(sessions):
            prompt_messages = []
            for session in sessions:
                prompt = session["conversations"][0]["value"]
                img = os.path.join(self.source, session["image"])
                if prompt.endswith("<image>"):
                    content = [
                        {"type": "text", "text": re.sub(r"\n<image>", "", prompt)},
                        {"type": "image", "image": img}
                    ]
                else:
                    content = [
                        {"type": "image", "image": img},
                        {"type": "text", "text": re.sub(r"<image>\n", "", prompt)}
                    ]
                prompt_messages.append([{
                    "role": "user",
                    "content": content,
                }])
            text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(prompt_messages)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )

            response_messages = []
            for session in sessions:
                response = session["conversations"][1]["value"]
                response_messages.append([{"content": response}])
            # right pad response to concat with prompt
            response_inputs = self.tokenizer.apply_chat_template(
                response_messages,
                chat_template=RESPONSE_TEMPLATE,
                padding=True,
                padding_side="right",
                add_generation_prompt=False,
                return_tensors="pt",
            )
            response_mask = (
                response_inputs != self.tokenizer.pad_token_id).type(torch.int64)
            loss_mask = torch.cat(
                [torch.zeros_like(inputs["input_ids"]), response_mask], dim=-1
            )
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], response_mask], dim=-1)
            inputs["input_ids"]  = torch.cat(
                [inputs["input_ids"], response_inputs], dim=-1)
            inputs["loss_mask"] = loss_mask
            _type = {
                "input_ids": torch.int64,
                "attention_mask": torch.int64,
                "pixel_values": torch.float32,
                "image_grid_thw": torch.int64,
                "video_grid_thw": torch.int64,
                "loss_mask": torch.int64
            }
            assert inputs["input_ids"].shape == inputs["loss_mask"].shape
            assert inputs["input_ids"].shape == inputs["attention_mask"].shape
            return inputs

        return collate_fn
    def __len__(self):
        return len(self.sessions)

if __name__ == "__main__":
    dataset = LLaVA_CC3M_Dataset(
        source="/llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K/",
        processor_path="/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14"
    )
    for idx, batch in enumerate(DataLoader(dataset, batch_size=3, collate_fn=dataset.build_collate_fn())):
        print(idx, batch)
        #print(dataset.processor.tokenizer.decode(batch["input_ids"]))
        if idx >= 1:
            break
    for key, tensor in batch.items():
        print(key, tensor.shape)
    for input_ids in batch["input_ids"]:
        print("=" * 10)
        print(dataset.processor.tokenizer.decode(input_ids))