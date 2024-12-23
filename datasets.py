import os
import re
import json
import math
import torch
import pandas as pd
import tarfile
import torch.distributed as dist

from torch.utils.data import IterableDataset, Dataset

import random
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
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


class LLaVA_CC3M_Dataset(Dataset):
    def __init__(self, source, processor_path, max_length=None):
        super(LLaVA_CC3M_Dataset).__init__()
        self.source = source
        with open(os.path.join(self.source, "chat.json"), encoding="utf-8") as f:
            self.sessions = json.loads(f.read())
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.max_length = max_length
        if not self.max_length:
            self.max_length = self.processor.tokenizer.model_max_length

    def __getitem__(self, index):
        session = self.sessions[index]
        img = os.path.join(self.source, session["image"])
        prompt = session["conversations"][0]["value"]
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
        prompt_messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            # return_tensors="pt",
        )

        response = session["conversations"][1]["value"]
        response_messages = [{"content": response}]
        response_inputs = self.processor.tokenizer.apply_chat_template(
            response_messages,
            chat_template=RESPONSE_TEMPLATE,
            padding=False,
            add_generation_prompt=False
        )
        label_mask = [[0] * len(inputs["input_ids"][0]) + \
            [1] * len(response_inputs)]
        inputs["attention_mask"][0] += [1] * len(response_inputs)
        inputs["input_ids"][0] += response_inputs
        inputs["label_mask"] = label_mask
        _type = {
            "input_ids": torch.int64,
            "attention_mask": torch.int64,
            "pixel_values": torch.float32,
            "image_grid_thw": torch.int64,
            "video_grid_thw": torch.int64,
            "label_mask": torch.int64
        }
        inputs = {key: torch.tensor(value, dtype=_type[key]) \
            for key, value in inputs.items()}
        assert inputs["input_ids"].shape == inputs["label_mask"].shape
        assert inputs["input_ids"].shape == inputs["attention_mask"].shape
        # PADDING & TRUNCATE

        return inputs

    def __len__(self):
        return len(self.sessions)

if __name__ == "__main__":
    dataset = LLaVA_CC3M_Dataset(
        source="/llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K/",
        processor_path="/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14"
    )
    for idx, batch in enumerate(dataset):
        print(idx, batch)
        print(dataset.processor.tokenizer.decode(batch["input_ids"][0]))
        if idx > 10:
            break